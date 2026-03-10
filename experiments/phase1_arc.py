"""
Phase 1: ARC-AGI-1 Training, Curriculum Style.

Implements the Phase 1 experiment from the manifesto:
1. Sort tasks by difficulty (grid size as proxy)
2. Run wake-sleep in curriculum order: easy first, sleep, extract library
3. Track the compounding curve: solve rate should increase across rounds
   as the library grows, without new hand-coded primitives

All runs automatically produce (in runs/):
  - <timestamp>_phase1.log       — full console output (tee'd)
  - <timestamp>_phase1.jsonl     — live per-task results (streamable)
  - <timestamp>_phase1.json      — final results with metadata + summary
  - <timestamp>_phase1_library.json — learned library snapshot

Usage:
    python -m experiments.phase1_arc                  # just run it
    python -m experiments.phase1_arc --mode quick     # fast dev loop (~2 min)
    python -m experiments.phase1_arc --mode contest   # max accuracy (~1 hr)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from io import TextIOBase

from core import (
    Learner,
    InMemoryStore,
    SearchConfig,
    SleepConfig,
    CurriculumConfig,
    WakeResult,
    RoundResult,
    extract_metrics,
    print_compounding_table,
    save_metrics_json,
    save_metrics_csv,
)
from grammars.arc import (
    ARCEnv,
    ARCGrammar,
    ARCDrive,
    make_sample_tasks,
    load_arc_dataset,
)


# =============================================================================
# Presets — the ONLY knob most users need
# =============================================================================

PRESETS = {
    # Quick: fast feedback for development. ~2 min on 400 tasks / 4 cores.
    "quick": {
        "rounds": 2,
        "beam_width": 80,
        "max_generations": 40,
        "max_tasks": 0,
    },
    # Default: balanced speed/accuracy. ~10 min on 400 tasks / 4 cores.
    "default": {
        "rounds": 3,
        "beam_width": 150,
        "max_generations": 80,
        "max_tasks": 0,
    },
    # Contest: maximize accuracy. ~1 hr on 400 tasks / 4 cores.
    "contest": {
        "rounds": 10,
        "beam_width": 500,
        "max_generations": 200,
        "max_tasks": 0,
    },
}

DEFAULT_SEED = 42
RUNS_DIR = "runs"


# =============================================================================
# Tee writer — duplicates stdout to both console and a log file
# (Adapted from agi-mvp-general/benchmark.py)
# =============================================================================

class _TeeWriter(TextIOBase):
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str, original_stdout):
        super().__init__()
        self._original = original_stdout
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log = open(log_path, "w", buffering=1)  # line-buffered

    def write(self, s):
        self._original.write(s)
        self._log.write(s)
        return len(s)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    @property
    def encoding(self):
        return self._original.encoding


# =============================================================================
# Formatting helpers (from agi-mvp-general)
# =============================================================================

def _hline(char="─", width=72):
    print(char * width)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '12.3s'."""
    s = seconds
    if s >= 3600:
        return f"{int(s) // 3600}h{(int(s) % 3600) // 60:02d}m"
    if s >= 60:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"
    return f"{s:.1f}s"


# =============================================================================
# Auto-detect ARC dataset
# =============================================================================

ARC_DATA_SEARCH_PATHS = [
    "data/ARC-AGI/data/training",
    "../ARC-AGI/data/training",
    os.path.expanduser("~/ARC-AGI/data/training"),
    "data/arc-agi/data/training",
]


def find_arc_data() -> str | None:
    """Return the first existing ARC training data directory, or None."""
    for path in ARC_DATA_SEARCH_PATHS:
        if os.path.isdir(path):
            return path
    return None


# =============================================================================
# Machine auto-detection
# =============================================================================

def detect_machine() -> dict:
    """Detect machine characteristics for adaptive configuration."""
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "performance_cores": Learner.performance_core_count(),
        "python": platform.python_version(),
    }
    if info["platform"] == "Darwin" and info["arch"] == "arm64":
        info["chip"] = "Apple Silicon"
    return info


# =============================================================================
# Signal handling — Ctrl-C kills the whole process tree immediately
# =============================================================================
# Workers ignore SIGINT (handled in core/learner.py _worker_init), so
# KeyboardInterrupt only fires in the main process. We catch it in main()
# and print partial results before exiting. Second Ctrl-C force-kills.

_interrupted = False


def _handle_sigint(signum, frame):
    """Second Ctrl-C: hard-kill immediately (no waiting for cleanup)."""
    global _interrupted
    if _interrupted:
        # Already interrupted once — force kill everything
        os.kill(os.getpid(), signal.SIGKILL)
    _interrupted = True
    raise KeyboardInterrupt


# =============================================================================
# Argument parsing with defaults shown
# =============================================================================

def parse_args():
    perf_cores = Learner.performance_core_count()

    parser = argparse.ArgumentParser(
        description="Phase 1: ARC-AGI-1 Curriculum Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Presets (the only knob most users need):
  quick     Fast dev loop     (~2 min on 400 tasks, 4 cores)
  default   Balanced          (~10 min on 400 tasks, 4 cores)
  contest   Max accuracy      (~1 hr on 400 tasks, 4 cores)

Times scale linearly with task count and inversely with core count.

Examples:
  python -m experiments.phase1_arc                    # sensible defaults
  python -m experiments.phase1_arc --mode quick       # fast iteration
  python -m experiments.phase1_arc --mode contest     # full benchmark
  python -m experiments.phase1_arc --max-tasks 50     # subset of tasks
""",
    )
    parser.add_argument("--mode", type=str, default="default",
                        choices=list(PRESETS.keys()),
                        help="Preset configuration")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI training dir (auto-detected if not set)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (0 = all, default: from preset)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Wake-sleep rounds (default: from preset)")
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Beam width (default: from preset)")
    parser.add_argument("--max-generations", type=int, default=None,
                        help="Max generations per task (default: from preset)")
    parser.add_argument("--workers", type=int, default=0,
                        help=f"Parallel workers (0 = performance cores = {perf_cores})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR,
                        help="Directory for all run artifacts")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable log file (console only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    return parser.parse_args()


# =============================================================================
# Live progress tracker with scoreboard
# (Adapted from agi-mvp-general _BenchmarkTracker)
# =============================================================================

class ProgressTracker:
    """Thread-safe live progress tracker with rolling scoreboard.

    Streams per-task results to console + JSONL file. Prints a rolling
    scoreboard summary every N tasks (like agi-mvp-general).
    """

    SCOREBOARD_INTERVAL = 10  # print scoreboard every N tasks

    def __init__(self, jsonl_path: str, t0: float):
        self._file = open(jsonl_path, "w")
        self._t0 = t0

        # Running stats
        self.done = 0
        self.solved = 0
        self.total_evals = 0
        self.total_gens = 0
        self.scores: list[float] = []   # best energy per task (lower = better)
        self.times: list[float] = []
        self.all_records: list[dict] = []

    def on_task_done(
        self,
        round_num: int,
        task_index: int,
        total_tasks: int,
        wr: WakeResult,
    ) -> None:
        """Called after each task completes. Streams live output."""
        self.done += 1
        if wr.solved:
            self.solved += 1
        self.total_evals += wr.evaluations
        self.total_gens += wr.generations_used
        if wr.best:
            self.scores.append(wr.best.energy)
        self.times.append(wr.wall_time)

        elapsed = time.time() - self._t0
        icon = "✓" if wr.solved else "✗"
        energy_str = f"{wr.best.energy:.4f}" if wr.best else "    N/A"
        program_str = repr(wr.best.program) if wr.best else ""

        # Per-task line
        slow_tag = ""
        if len(self.times) >= 5:
            med = statistics.median(self.times)
            if wr.wall_time > med * 3:
                slow_tag = "  *** SLOW ***"

        print(
            f"  {icon} R{round_num} [{task_index:>3}/{total_tasks}] "
            f"{wr.task_id:<20s} "
            f"E={energy_str}  gens={wr.generations_used:<4d} "
            f"evals={wr.evaluations:<6d} {wr.wall_time:.1f}s"
            f"{slow_tag}",
            flush=True,
        )
        if wr.solved and program_str:
            print(f"       program: {program_str}")
        print(
            f"       done={self.done}  "
            f"solved={self.solved}/{self.done}  "
            f"evals={self.total_evals:,}  "
            f"[{_fmt_duration(elapsed)} elapsed]",
        )

        # Build record for JSONL + final JSON
        record = {
            "round": round_num,
            "task_index": task_index,
            "task_id": wr.task_id,
            "solved": wr.solved,
            "energy": wr.best.energy if wr.best else None,
            "prediction_error": wr.best.prediction_error if wr.best else None,
            "generations": wr.generations_used,
            "evaluations": wr.evaluations,
            "wall_time": round(wr.wall_time, 3),
            "program": repr(wr.best.program) if wr.best else None,
            "elapsed": round(elapsed, 1),
        }
        self.all_records.append(record)

        # Stream to JSONL (live, flushable, tail -f friendly)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        # Rolling scoreboard
        if self.done % self.SCOREBOARD_INTERVAL == 0:
            self._print_scoreboard(round_num, total_tasks, elapsed)

    def _print_scoreboard(
        self, round_num: int, total_tasks: int, elapsed: float,
    ) -> None:
        """Print a rolling scoreboard summary (like agi-mvp-general)."""
        rate = self.done / max(elapsed, 0.001)
        pending = total_tasks - (self.done % total_tasks or total_tasks)
        eta = pending / rate if rate > 0 else 0
        mean_energy = statistics.mean(self.scores) if self.scores else float("inf")
        med_time = statistics.median(self.times) if self.times else 0

        print()
        print(
            f"  ┌── Progress: {self.done} tasks  "
            f"{_fmt_duration(elapsed)} elapsed  "
            f"ETA {_fmt_duration(eta)}  "
            f"{rate:.1f} tasks/s ──"
        )
        print(
            f"  │  ✓ solved={self.solved}/{self.done}  "
            f"✗ unsolved={self.done - self.solved}/{self.done}"
        )
        print(
            f"  │  Energy: mean={mean_energy:.4f}  "
            f"Time: median={med_time:.1f}s  "
            f"Evals: {self.total_evals:,}"
        )

        # Slowest tasks so far
        if len(self.all_records) >= 5:
            by_time = sorted(self.all_records, key=lambda r: -r["wall_time"])
            top3 = by_time[:3]
            slowest_str = "  ".join(
                f"{r['task_id'][:8]}({r['wall_time']:.1f}s)"
                for r in top3
            )
            print(f"  │  Slowest: {slowest_str}")

        print(f"  └{'─' * 60}")
        print()

    def close(self):
        self._file.close()


# =============================================================================
# Main
# =============================================================================

def main():
    # Install signal handler FIRST for immediate Ctrl-C response
    signal.signal(signal.SIGINT, _handle_sigint)

    args = parse_args()

    # -------------------------------------------------------------------------
    # Timestamp — shared across ALL output files for this run
    # -------------------------------------------------------------------------
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_timestamp}_phase1"

    # All files go flat into runs/ with the timestamp prefix
    runs_dir = args.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    log_path = os.path.join(runs_dir, f"{prefix}.log")
    jsonl_path = os.path.join(runs_dir, f"{prefix}.jsonl")
    results_path = os.path.join(runs_dir, f"{prefix}.json")
    library_path = os.path.join(runs_dir, f"{prefix}_library.json")
    metrics_json_path = os.path.join(runs_dir, f"{prefix}_metrics.json")
    metrics_csv_path = os.path.join(runs_dir, f"{prefix}_metrics.csv")

    # -------------------------------------------------------------------------
    # Tee stdout → console + log file (like agi-mvp-general)
    # -------------------------------------------------------------------------
    tee = None
    if not args.no_log:
        tee = _TeeWriter(log_path, sys.stdout)
        sys.stdout = tee

    try:
        _run(args, run_timestamp, runs_dir, prefix,
             jsonl_path, results_path, library_path,
             metrics_json_path, metrics_csv_path, log_path)
    except KeyboardInterrupt:
        print("\n\nAborted by user — partial results above.\n")
    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()


def _run(args, run_timestamp, runs_dir, prefix,
         jsonl_path, results_path, library_path,
         metrics_json_path, metrics_csv_path, log_path):
    """Core run logic, separated so tee cleanup always happens."""

    # -------------------------------------------------------------------------
    # Machine + preset resolution
    # -------------------------------------------------------------------------
    machine = detect_machine()
    preset = PRESETS[args.mode]

    rounds = args.rounds if args.rounds is not None else preset["rounds"]
    beam_width = args.beam_width if args.beam_width is not None else preset["beam_width"]
    max_gens = args.max_generations if args.max_generations is not None else preset["max_generations"]
    max_tasks = args.max_tasks if args.max_tasks is not None else preset["max_tasks"]
    workers = args.workers if args.workers > 0 else Learner.performance_core_count()

    # -------------------------------------------------------------------------
    # Print header + output paths (so user can tail -f immediately)
    # -------------------------------------------------------------------------
    _hline("═")
    print("  PHASE 1: ARC-AGI-1 CURRICULUM TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _hline("═")

    print(f"\n  Mode:       {args.mode}")
    print(f"  Rounds:     {rounds}")
    print(f"  Beam:       {beam_width}")
    print(f"  Gens:       {max_gens}")
    print(f"  Workers:    {workers} / {machine['cpu_count']} cores "
          f"({machine.get('chip', machine['arch'])})")
    print(f"  Seed:       {args.seed}")
    print(f"  Verbose:    {args.verbose}")

    print()
    _hline("─")
    print("  Output files (available now for tail -f):")
    _hline("─")
    print(f"  Results (live):   {jsonl_path}")
    print(f"  Results (final):  {results_path}")
    print(f"  Metrics:          {metrics_json_path}")
    print(f"  Library:          {library_path}")
    if not args.no_log:
        print(f"  Console log:      {log_path}")
    print()

    # -------------------------------------------------------------------------
    # Load tasks
    # -------------------------------------------------------------------------
    data_dir = args.data_dir or find_arc_data()

    if data_dir:
        print(f"  Loading ARC-AGI tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    else:
        print("  ARC dataset not found. Using built-in sample tasks.")
        print("    (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
        tasks = make_sample_tasks()
        print(f"  Created {len(tasks)} sample ARC tasks")

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Wire up 4 interfaces + create learner
    # -------------------------------------------------------------------------
    env = ARCEnv()
    grammar = ARCGrammar(seed=args.seed)
    drive = ARCDrive()
    memory = InMemoryStore()

    evals_per_task = beam_width * max_gens
    total_budget = evals_per_task * len(tasks) * rounds

    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=SearchConfig(
            beam_width=beam_width,
            max_generations=max_gens,
            mutations_per_candidate=2,
            crossover_fraction=0.3,
            energy_alpha=1.0,
            energy_beta=0.002,
            solve_threshold=0.001,
            seed=args.seed,
        ),
        sleep_config=SleepConfig(
            min_occurrences=2,
            min_size=2,
            max_library_size=500,
            usefulness_decay=0.95,
        ),
    )

    print(f"\n  Tasks:      {len(tasks)}")
    print(f"  Primitives: {len(grammar.base_primitives())}")
    print(f"  Budget:     ~{evals_per_task:,} evals/task, ~{total_budget:,} total")

    # -------------------------------------------------------------------------
    # Set up live progress tracker
    # -------------------------------------------------------------------------
    tracker = ProgressTracker(jsonl_path, time.time())

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    _hline("─")
    print(f"  Running {len(tasks)} tasks × {rounds} rounds on {workers} workers")
    _hline("─")
    print()

    t0 = time.time()
    results = learner.run_curriculum(
        tasks,
        CurriculumConfig(
            sort_by_difficulty=True,
            wake_sleep_rounds=rounds,
            workers=workers,
        ),
        on_task_done=tracker.on_task_done,
    )
    total_time = time.time() - t0
    tracker.close()

    # -------------------------------------------------------------------------
    # Final scoreboard
    # -------------------------------------------------------------------------
    metrics = extract_metrics(results)
    total_evals = sum(wr.evaluations for rr in results for wr in rr.wake_results)

    print()
    _hline("═")
    print("  COMPOUNDING CURVE — THE KEY METRIC")
    _hline("═")
    print_compounding_table(metrics)
    print()

    _hline("═")
    print("  FINAL RESULTS")
    _hline("═")

    print(f"  Tasks:             {tracker.done}")
    print(f"  ✓ Solved:          {tracker.solved}/{tracker.done}")
    if tracker.scores:
        print(f"  Mean energy:       {statistics.mean(tracker.scores):.4f}")
    print(f"  Total evaluations: {total_evals:,}")
    print(f"  Total generations: {tracker.total_gens:,}")
    if tracker.times:
        print(f"  Median task time:  {statistics.median(tracker.times):.2f}s")
        print(f"  Mean task time:    {statistics.mean(tracker.times):.2f}s")
    print(f"  Wall-clock time:   {_fmt_duration(total_time)}")
    throughput = tracker.done / max(total_time, 0.001)
    print(f"  Throughput:        {throughput:.1f} tasks/s ({workers} workers)")

    if len(metrics) >= 2:
        first_rate = metrics[0].solve_rate
        last_rate = metrics[-1].solve_rate
        if last_rate > first_rate:
            print(f"\n  >>> COMPOUNDING DETECTED: {first_rate:.1%} → {last_rate:.1%}")
        elif last_rate == first_rate:
            print(f"\n  >>> PLATEAU: solve rate stayed at {first_rate:.1%}")
        else:
            print(f"\n  >>> REGRESSION: {first_rate:.1%} → {last_rate:.1%}")

    # Library summary
    library = memory.get_library()
    print(f"\n  Library: {len(library)} learned abstractions")
    for entry in library[:20]:
        print(f"    {entry.name}: {entry.program} "
              f"(useful={entry.usefulness:.1f}, reused={entry.reuse_count}x, "
              f"from {len(entry.source_tasks)} tasks)")
    if len(library) > 20:
        print(f"    ... and {len(library) - 20} more")

    # Slowest tasks post-mortem
    if len(tracker.all_records) >= 5:
        by_time = sorted(tracker.all_records, key=lambda r: -r["wall_time"])
        print(f"\n  Slowest tasks:")
        for r in by_time[:5]:
            icon = "✓" if r["solved"] else "✗"
            print(f"    {icon} {r['task_id']}  {r['wall_time']:.1f}s  "
                  f"evals={r['evaluations']:,}  E={r['energy']}")

    # -------------------------------------------------------------------------
    # Save results — final JSON with metadata + summary + per-task
    # (Like agi-mvp-general: one comprehensive JSON file)
    # -------------------------------------------------------------------------
    results_data = {
        "meta": {
            "timestamp": run_timestamp,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "data_dir": data_dir,
            "rounds": rounds,
            "beam_width": beam_width,
            "max_generations": max_gens,
            "evals_per_task": evals_per_task,
            "workers": workers,
            "seed": args.seed,
            "n_tasks": len(tasks),
            "n_primitives": len(grammar.base_primitives()),
            "machine": machine,
            "verbose": args.verbose,
        },
        "summary": {
            "tasks_completed": tracker.done,
            "tasks_solved": tracker.solved,
            "solve_rate": tracker.solved / max(tracker.done, 1),
            "mean_energy": round(statistics.mean(tracker.scores), 4) if tracker.scores else None,
            "total_evaluations": total_evals,
            "total_generations": tracker.total_gens,
            "median_task_time": round(statistics.median(tracker.times), 3) if tracker.times else None,
            "mean_task_time": round(statistics.mean(tracker.times), 3) if tracker.times else None,
            "wall_clock_seconds": round(total_time, 1),
            "throughput_tasks_per_sec": round(throughput, 2),
            "library_size": len(library),
            "compounding": [
                {
                    "round": m.round_number,
                    "solve_rate": round(m.solve_rate, 4),
                    "tasks_solved": m.tasks_solved,
                    "tasks_total": m.tasks_total,
                    "library_size": m.library_size,
                    "new_abstractions": m.new_abstractions,
                    "avg_reuse": round(m.avg_reuse_per_entry, 2),
                    "avg_energy": round(m.avg_energy_of_solutions, 4) if m.avg_energy_of_solutions != float("inf") else None,
                    "wake_time": round(m.wall_time_wake, 1),
                    "sleep_time": round(m.wall_time_sleep, 1),
                }
                for m in metrics
            ],
        },
        "tasks": {
            r["task_id"]: {
                "round": r["round"],
                "solved": r["solved"],
                "energy": r["energy"],
                "prediction_error": r["prediction_error"],
                "generations": r["generations"],
                "evaluations": r["evaluations"],
                "wall_time": r["wall_time"],
                "program": r["program"],
            }
            for r in tracker.all_records
        },
        "library": [
            {
                "name": entry.name,
                "program": repr(entry.program),
                "usefulness": round(entry.usefulness, 2),
                "reuse_count": entry.reuse_count,
                "source_tasks": list(entry.source_tasks),
            }
            for entry in library
        ],
    }

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Also save structured metrics (for downstream tooling)
    save_metrics_json(metrics, metrics_json_path)
    save_metrics_csv(metrics, metrics_csv_path)
    memory.save(library_path)

    # -------------------------------------------------------------------------
    # Print artifacts
    # -------------------------------------------------------------------------
    print()
    _hline("─")
    print("  Artifacts:")
    _hline("─")
    print(f"  Results (live):   {jsonl_path}")
    print(f"  Results (final):  {results_path}")
    print(f"  Metrics JSON:     {metrics_json_path}")
    print(f"  Metrics CSV:      {metrics_csv_path}")
    print(f"  Library:          {library_path}")
    if not args.no_log:
        print(f"  Console log:      {log_path}")

    _hline("═")
    print("  Done.")
    _hline("═")
    print()


if __name__ == "__main__":
    main()
