"""
Generic experiment runner for any domain.

This is part of the invariant core. It provides:
- TeeWriter (stdout → console + log file)
- ProgressTracker (live per-task streaming + scoreboard)
- Signal handling (Ctrl-C kills immediately)
- Presets (quick/default/contest)
- Output formatting and results saving

Domain-specific scripts only need to provide tasks + the 4 interfaces.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from io import TextIOBase
from typing import Optional

from .types import Task
from .interfaces import Environment, Grammar, DriveSignal
from .config import SearchConfig, SleepConfig, CurriculumConfig
from .results import WakeResult
from .learner import Learner
from .memory import InMemoryStore
from .metrics import extract_metrics, print_compounding_table, save_metrics_json, save_metrics_csv


# =============================================================================
# Presets — the ONLY knob most users need
# =============================================================================

PRESETS = {
    "quick": {
        "rounds": 2,
        "beam_width": 80,
        "max_generations": 40,
        "max_tasks": 0,
    },
    "default": {
        "rounds": 3,
        "beam_width": 150,
        "max_generations": 80,
        "max_tasks": 0,
    },
    "contest": {
        "rounds": 10,
        "beam_width": 500,
        "max_generations": 200,
        "max_tasks": 0,
    },
}

DEFAULT_SEED = 42
DEFAULT_RUNS_DIR = "runs"


# =============================================================================
# Tee writer — duplicates stdout to both console and a log file
# =============================================================================

class TeeWriter(TextIOBase):
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str, original_stdout):
        super().__init__()
        self._original = original_stdout
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log = open(log_path, "w", buffering=1)

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
# Formatting helpers
# =============================================================================

def hline(char="─", width=72):
    print(char * width)


def fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '12.3s'."""
    s = seconds
    if s >= 3600:
        return f"{int(s) // 3600}h{(int(s) % 3600) // 60:02d}m"
    if s >= 60:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"
    return f"{s:.1f}s"


# =============================================================================
# Machine auto-detection
# =============================================================================

def detect_machine() -> dict:
    """Detect machine characteristics for metadata."""
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

_interrupted = False


def _handle_sigint(signum, frame):
    """First Ctrl-C raises KeyboardInterrupt. Second force-kills."""
    global _interrupted
    if _interrupted:
        os.kill(os.getpid(), signal.SIGKILL)
    _interrupted = True
    raise KeyboardInterrupt


def install_signal_handler():
    """Install the Ctrl-C signal handler. Call this early in main()."""
    signal.signal(signal.SIGINT, _handle_sigint)


# =============================================================================
# Human-readable number parsing
# =============================================================================

def parse_human_int(s: str) -> int:
    """Parse a human-readable integer with optional suffix or commas.

    Examples: "50M" → 50_000_000, "500K" → 500_000, "8,000,000" → 8_000_000
    """
    s = s.strip().replace(",", "").replace("_", "")
    if not s:
        raise argparse.ArgumentTypeError("empty value")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
    last = s[-1].lower()
    if last in suffixes:
        try:
            return int(float(s[:-1]) * suffixes[last])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid number: {s!r}. Examples: 50M, 8,000,000, 500K, 0")

    try:
        return int(float(s))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid number: {s!r}. Examples: 50M, 8,000,000, 500K, 0")


# =============================================================================
# Standard argument parser (reusable by any domain)
# =============================================================================

def make_parser(description: str, domain_name: str = "experiment") -> argparse.ArgumentParser:
    """Create a standard argument parser with presets + common flags."""
    perf_cores = Learner.performance_core_count()

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"""
Presets (the only knob most users need):
  quick     Fast dev loop
  default   Balanced speed/accuracy
  contest   Maximum accuracy

Compute cap examples:
  --compute-cap 50M          # 50 million total evaluations
  --compute-cap 500K         # 500 thousand
  --compute-cap 0            # unlimited (default)

Examples:
  python -m experiments.{domain_name}                    # sensible defaults
  python -m experiments.{domain_name} --mode quick       # fast iteration
  python -m experiments.{domain_name} --mode contest     # full benchmark
""",
    )
    parser.add_argument("--mode", type=str, default="default",
                        choices=list(PRESETS.keys()), help="Preset configuration")
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
    parser.add_argument("--compute-cap", type=parse_human_int,
                        default=str(0),
                        help="Total eval budget (0=unlimited). Accepts: 50M, 500K, 0")
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Directory for all run artifacts")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable log file (console only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    parser.add_argument("--exhaustive-depth", type=int, default=3,
                        help="Exhaustive enumeration depth (0=disabled, 2=pairs, 3=triples)")
    parser.add_argument("--exhaustive-pair-top-k", type=int, default=40,
                        help="Top-K singles for pair exhaustion (default 40)")
    parser.add_argument("--exhaustive-triple-top-k", type=int, default=15,
                        help="Top-K singles for triple exhaustion (default 15)")
    parser.add_argument("--sequential-compounding", action="store_true",
                        help="Process tasks sequentially with immediate concept promotion")
    parser.add_argument("--culture", type=str, default="",
                        help="Path to culture file to load (cross-run knowledge transfer)")
    parser.add_argument("--save-culture", type=str, default="",
                        help="Override auto culture save path (e.g. culture_train.json)")
    return parser


# =============================================================================
# Live progress tracker with scoreboard
# =============================================================================

class ProgressTracker:
    """Live progress tracker with rolling scoreboard.

    Streams per-task results to console + JSONL file. Prints a rolling
    scoreboard summary every N tasks.
    """

    SCOREBOARD_INTERVAL = 10

    def __init__(self, jsonl_path: str, t0: float):
        self._file = open(jsonl_path, "w")
        self._t0 = t0

        # Cumulative (across all rounds)
        self.done = 0
        self.solved = 0
        self.total_evals = 0
        self.total_gens = 0
        self.scores: list[float] = []
        self.times: list[float] = []
        self.all_records: list[dict] = []

        # Per-round tracking (reset each round)
        self._current_round = -1
        self._round_done = 0
        self._round_solved = 0

    def on_task_done(
        self, round_num: int, task_index: int, total_tasks: int, wr: WakeResult,
    ) -> None:
        """Called after each task completes. Streams live output."""
        # Reset per-round counters on new round
        if round_num != self._current_round:
            self._current_round = round_num
            self._round_done = 0
            self._round_solved = 0

        self.done += 1
        self._round_done += 1
        if wr.solved:
            self.solved += 1
            self._round_solved += 1
        self.total_evals += wr.evaluations
        self.total_gens += wr.generations_used
        if wr.best:
            self.scores.append(wr.best.energy)
        self.times.append(wr.wall_time)

        elapsed = time.time() - self._t0
        icon = "✓" if wr.solved else "✗"
        energy_str = f"{wr.best.energy:.4f}" if wr.best else "    N/A"
        program_str = repr(wr.best.program) if wr.best else ""

        # Test accuracy tag
        test_tag = ""
        if wr.test_solved is not None:
            test_tag = " [test:✓]" if wr.test_solved else " [test:✗]"

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
            f"{test_tag}{slow_tag}",
            flush=True,
        )
        if wr.solved and program_str:
            print(f"       program: {program_str}")
        print(
            f"       R{round_num}: solved={self._round_solved}/{self._round_done}  "
            f"evals={self.total_evals:,}  "
            f"[{fmt_duration(elapsed)} elapsed]",
        )

        record = {
            "round": round_num,
            "task_index": task_index,
            "task_id": wr.task_id,
            "solved": wr.solved,
            "test_solved": wr.test_solved,
            "test_error": round(wr.test_error, 6) if wr.test_error is not None else None,
            "energy": wr.best.energy if wr.best else None,
            "prediction_error": wr.best.prediction_error if wr.best else None,
            "generations": wr.generations_used,
            "evaluations": wr.evaluations,
            "wall_time": round(wr.wall_time, 3),
            "program": repr(wr.best.program) if wr.best else None,
            "elapsed": round(elapsed, 1),
        }
        self.all_records.append(record)

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        if self.done % self.SCOREBOARD_INTERVAL == 0:
            self._print_scoreboard(round_num, total_tasks, elapsed)

    def _print_scoreboard(
        self, round_num: int, total_tasks: int, elapsed: float,
    ) -> None:
        rate = self.done / max(elapsed, 0.001)
        pending = total_tasks - (self.done % total_tasks or total_tasks)
        eta = pending / rate if rate > 0 else 0
        mean_energy = statistics.mean(self.scores) if self.scores else float("inf")
        med_time = statistics.median(self.times) if self.times else 0

        print()
        print(
            f"  ┌── Progress: {self.done} tasks  "
            f"{fmt_duration(elapsed)} elapsed  "
            f"ETA {fmt_duration(eta)}  "
            f"{rate:.1f} tasks/s ──"
        )
        print(
            f"  │  R{round_num}: ✓ solved={self._round_solved}/{self._round_done}  "
            f"✗ unsolved={self._round_done - self._round_solved}/{self._round_done}"
        )
        print(
            f"  │  Energy: mean={mean_energy:.4f}  "
            f"Time: median={med_time:.1f}s  "
            f"Evals: {self.total_evals:,}"
        )
        if len(self.all_records) >= 5:
            by_time = sorted(self.all_records, key=lambda r: -r["wall_time"])
            top3 = by_time[:3]
            slowest_str = "  ".join(
                f"{r['task_id'][:8]}({r['wall_time']:.1f}s)" for r in top3
            )
            print(f"  │  Slowest: {slowest_str}")
        print(f"  └{'─' * 60}")
        print()

    def close(self):
        self._file.close()


# =============================================================================
# The generic experiment runner
# =============================================================================

@dataclass
class ExperimentConfig:
    """Everything needed to run a domain-agnostic experiment."""
    title: str
    domain_tag: str  # short tag for output filenames, e.g. "arc", "symreg"
    tasks: list[Task]

    environment: Environment
    grammar: Grammar
    drive: DriveSignal

    # Search parameters (resolved from preset + overrides)
    rounds: int = 3
    beam_width: int = 150
    max_generations: int = 80
    workers: int = 0
    seed: int = DEFAULT_SEED
    compute_cap: int = 0

    # Search tuning
    mutations_per_candidate: int = 2
    crossover_fraction: float = 0.3
    energy_alpha: float = 1.0
    energy_beta: float = 0.001
    solve_threshold: float = 0.001

    # Sleep tuning
    min_occurrences: int = 2
    min_size: int = 2
    max_library_size: int = 500

    # Exhaustive enumeration
    exhaustive_depth: int = 3
    exhaustive_pair_top_k: int = 40
    exhaustive_triple_top_k: int = 15

    # Sequential compounding
    sequential_compounding: bool = False

    # Culture file (load pre-trained library)
    culture_path: str = ""
    save_culture: str = ""  # override auto-generated culture output path

    # Output
    runs_dir: str = DEFAULT_RUNS_DIR
    no_log: bool = False
    verbose: bool = False
    mode: str = "default"


def resolve_from_preset(args, preset: dict) -> dict:
    """Resolve argument values: explicit args override preset defaults."""
    return {
        "rounds": args.rounds if args.rounds is not None else preset["rounds"],
        "beam_width": args.beam_width if args.beam_width is not None else preset["beam_width"],
        "max_generations": args.max_generations if args.max_generations is not None else preset["max_generations"],
        "max_tasks": args.max_tasks if args.max_tasks is not None else preset["max_tasks"],
        "workers": args.workers if args.workers > 0 else Learner.performance_core_count(),
    }


def run_experiment(cfg: ExperimentConfig) -> str:
    """Run a complete wake-sleep experiment on any domain.

    Returns the path to the saved culture file (for pipeline chaining).

    This is the top-level entry point. It:
    1. Sets up output files (log, jsonl, json, metrics, library)
    2. Tees stdout to log file
    3. Creates the Learner with the domain's 4 interfaces
    4. Runs the curriculum with live progress tracking
    5. Prints final results and saves all artifacts
    """
    install_signal_handler()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_timestamp}_{cfg.domain_tag}"

    runs_dir = cfg.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    log_path = os.path.join(runs_dir, f"{prefix}.log")
    jsonl_path = os.path.join(runs_dir, f"{prefix}.jsonl")
    results_path = os.path.join(runs_dir, f"{prefix}.json")
    library_path = os.path.join(runs_dir, f"{prefix}_library.json")
    metrics_json_path = os.path.join(runs_dir, f"{prefix}_metrics.json")
    metrics_csv_path = os.path.join(runs_dir, f"{prefix}_metrics.csv")

    culture_path = (cfg.save_culture if cfg.save_culture
                     else library_path.replace("_library.json", "_culture.json"))

    tee = None
    if not cfg.no_log:
        tee = TeeWriter(log_path, sys.stdout)
        sys.stdout = tee

    try:
        _run_experiment(cfg, run_timestamp, log_path, jsonl_path, results_path,
                        library_path, metrics_json_path, metrics_csv_path,
                        culture_path)
    except KeyboardInterrupt:
        print("\n\nAborted by user — partial results above.\n")
    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()

    return culture_path


def _run_experiment(cfg, run_timestamp, log_path, jsonl_path, results_path,
                    library_path, metrics_json_path, metrics_csv_path,
                    culture_path):
    """Core run logic, separated so tee cleanup always happens."""
    machine = detect_machine()
    workers = cfg.workers if cfg.workers > 0 else Learner.performance_core_count()
    rounds = cfg.rounds
    beam_width = cfg.beam_width
    max_gens = cfg.max_generations
    tasks = cfg.tasks
    compute_cap = cfg.compute_cap

    # --- Header ---
    hline("═")
    print(f"  {cfg.title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hline("═")

    print(f"\n  Mode:       {cfg.mode}")
    print(f"  Rounds:     {rounds}")
    print(f"  Beam:       {beam_width}")
    print(f"  Gens:       {max_gens}")
    print(f"  Workers:    {workers} / {machine['cpu_count']} cores "
          f"({machine.get('chip', machine['arch'])})")
    print(f"  Seed:       {cfg.seed}")
    if compute_cap > 0:
        print(f"  Compute cap: {compute_cap:,} total evals")
    else:
        print(f"  Compute cap: unlimited")

    print()
    hline("─")
    print("  Output files (available now for tail -f):")
    hline("─")
    print(f"  Results (live):   {jsonl_path}")
    print(f"  Results (final):  {results_path}")
    print(f"  Metrics:          {metrics_json_path}")
    print(f"  Library:          {library_path}")
    if not cfg.no_log:
        print(f"  Console log:      {log_path}")
    print()

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    # --- Create Learner ---
    memory = InMemoryStore()

    evals_per_task = beam_width * max_gens
    total_budget = evals_per_task * len(tasks) * rounds

    if compute_cap > 0 and total_budget > compute_cap:
        capped_gens = max(1, compute_cap // (beam_width * len(tasks) * rounds))
        print(f"\n  Compute cap active: reducing gens {max_gens} → {capped_gens} "
              f"(budget {total_budget:,} → {beam_width * capped_gens * len(tasks) * rounds:,})")
        max_gens = capped_gens
        evals_per_task = beam_width * max_gens
        total_budget = evals_per_task * len(tasks) * rounds

    # Load culture file if specified (cross-run knowledge transfer)
    if cfg.culture_path and os.path.isfile(cfg.culture_path):
        memory.load_culture(cfg.culture_path)
        print(f"\n  Culture loaded: {len(memory.get_library())} library entries, "
              f"{len(memory.get_solutions())} solutions")

    learner = Learner(
        environment=cfg.environment,
        grammar=cfg.grammar,
        drive=cfg.drive,
        memory=memory,
        search_config=SearchConfig(
            beam_width=beam_width,
            max_generations=max_gens,
            mutations_per_candidate=cfg.mutations_per_candidate,
            crossover_fraction=cfg.crossover_fraction,
            energy_alpha=cfg.energy_alpha,
            energy_beta=cfg.energy_beta,
            solve_threshold=cfg.solve_threshold,
            seed=cfg.seed,
            exhaustive_depth=cfg.exhaustive_depth,
            exhaustive_pair_top_k=cfg.exhaustive_pair_top_k,
            exhaustive_triple_top_k=cfg.exhaustive_triple_top_k,
        ),
        sleep_config=SleepConfig(
            min_occurrences=cfg.min_occurrences,
            min_size=cfg.min_size,
            max_library_size=cfg.max_library_size,
        ),
    )

    print(f"\n  Tasks:      {len(tasks)}")
    print(f"  Primitives: {len(cfg.grammar.base_primitives())}")
    print(f"  Budget:     ~{evals_per_task:,} evals/task, ~{total_budget:,} total")

    # --- Run ---
    tracker = ProgressTracker(jsonl_path, time.time())

    hline("─")
    print(f"  Running {len(tasks)} tasks × {rounds} rounds on {workers} workers")
    hline("─")
    print()

    t0 = time.time()
    results = learner.run_curriculum(
        tasks,
        CurriculumConfig(
            sort_by_difficulty=True,
            wake_sleep_rounds=rounds,
            workers=workers,
            sequential_compounding=cfg.sequential_compounding,
        ),
        on_task_done=tracker.on_task_done,
    )
    total_time = time.time() - t0
    tracker.close()

    # --- Final results ---
    metrics = extract_metrics(results)
    total_evals = sum(wr.evaluations for rr in results for wr in rr.wake_results)

    print()
    hline("═")
    print("  COMPOUNDING CURVE — THE KEY METRIC")
    hline("═")
    print_compounding_table(metrics)
    print()

    hline("═")
    print("  FINAL RESULTS")
    hline("═")

    # Use last round's metrics for the headline numbers (not cumulative)
    last = metrics[-1] if metrics else None
    n_tasks = len(tasks)

    if last:
        print(f"  Tasks:             {n_tasks}")
        print(f"  ✓ Solved (train):  {last.tasks_solved}/{n_tasks}  "
              f"({last.solve_rate:.1%})")
    else:
        print(f"  Tasks:             {n_tasks}")

    # Test accuracy (generalization) — deduplicate by task_id, last round wins
    all_wake = [wr for rr in results for wr in rr.wake_results]
    test_by_task: dict[str, bool] = {}
    for wr in all_wake:
        if wr.test_solved is not None:
            test_by_task[wr.task_id] = wr.test_solved
    if test_by_task:
        unique_test_solved = sum(1 for v in test_by_task.values() if v)
        unique_test_total = len(test_by_task)
        print(f"  ✓ Solved (test):   {unique_test_solved}/{unique_test_total}  "
              f"({unique_test_solved / max(unique_test_total, 1):.1%})")

    print(f"  Rounds:            {rounds}")
    print(f"  Total evaluations: {total_evals:,}")
    if tracker.times:
        print(f"  Median task time:  {statistics.median(tracker.times):.2f}s")
    print(f"  Wall-clock time:   {fmt_duration(total_time)}")
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

    library = memory.get_library()
    print(f"\n  Library: {len(library)} learned abstractions")
    for entry in library[:20]:
        print(f"    {entry.name}: {entry.program} "
              f"(useful={entry.usefulness:.1f}, reused={entry.reuse_count}x, "
              f"from {len(entry.source_tasks)} tasks)")
    if len(library) > 20:
        print(f"    ... and {len(library) - 20} more")

    if len(tracker.all_records) >= 5:
        by_time = sorted(tracker.all_records, key=lambda r: -r["wall_time"])
        print(f"\n  Slowest tasks:")
        for r in by_time[:5]:
            icon = "✓" if r["solved"] else "✗"
            print(f"    {icon} {r['task_id']}  {r['wall_time']:.1f}s  "
                  f"evals={r['evaluations']:,}  E={r['energy']}")

    # --- Save artifacts ---
    results_data = {
        "meta": {
            "timestamp": run_timestamp,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "title": cfg.title,
            "domain": cfg.domain_tag,
            "mode": cfg.mode,
            "rounds": rounds,
            "beam_width": beam_width,
            "max_generations": max_gens,
            "evals_per_task": evals_per_task,
            "workers": workers,
            "seed": cfg.seed,
            "n_tasks": len(tasks),
            "n_primitives": len(cfg.grammar.base_primitives()),
            "compute_cap": compute_cap,
            "machine": machine,
        },
        "summary": {
            "n_tasks": len(tasks),
            "rounds": rounds,
            "last_round_solved": last.tasks_solved if last else 0,
            "last_round_solve_rate": round(last.solve_rate, 4) if last else 0,
            "total_task_instances": tracker.done,
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

    save_metrics_json(metrics, metrics_json_path)
    save_metrics_csv(metrics, metrics_csv_path)

    # Save culture file (proper serialization with program reconstruction)
    memory.save_culture(culture_path)
    # Also save legacy format
    memory.save(library_path)

    print()
    hline("─")
    print("  Artifacts:")
    hline("─")
    print(f"  Results (live):   {jsonl_path}")
    print(f"  Results (final):  {results_path}")
    print(f"  Metrics JSON:     {metrics_json_path}")
    print(f"  Metrics CSV:      {metrics_csv_path}")
    print(f"  Library:          {library_path}")
    print(f"  Culture:          {culture_path}")
    if not cfg.no_log:
        print(f"  Console log:      {log_path}")

    hline("═")
    print("  Done.")
    hline("═")
    print()
