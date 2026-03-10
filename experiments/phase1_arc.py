"""
Phase 1: ARC-AGI-1 Training, Curriculum Style.

Implements the Phase 1 experiment from the manifesto:
1. Sort tasks by difficulty (grid size as proxy)
2. Run wake-sleep in curriculum order: easy first, sleep, extract library
3. Track the compounding curve: solve rate should increase across rounds
   as the library grows, without new hand-coded primitives

Usage:
    python -m experiments.phase1_arc                  # just run it
    python -m experiments.phase1_arc --mode quick     # fast dev loop (~2 min)
    python -m experiments.phase1_arc --mode contest   # max accuracy (~1 hr)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone

from core import (
    Learner,
    InMemoryStore,
    SearchConfig,
    SleepConfig,
    CurriculumConfig,
    WakeResult,
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
# Each preset is tuned for a specific use case. The compute budget is
# implicitly defined by beam_width × max_generations × rounds.
# Early stopping on solve means easy tasks don't waste compute.

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
        "python": platform.python_version(),
    }
    if info["platform"] == "Darwin" and info["arch"] == "arm64":
        info["chip"] = "Apple Silicon"
    return info


# =============================================================================
# Signal handling — Ctrl-C kills immediately
# =============================================================================

_interrupted = False


def _handle_sigint(signum, frame):
    """Handle Ctrl-C: first press prints message and exits, immediate."""
    global _interrupted
    if _interrupted:
        # Second Ctrl-C: hard exit
        os._exit(1)
    _interrupted = True
    print("\n\nInterrupted by Ctrl-C. Shutting down...", flush=True)
    sys.exit(130)


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
On M1 Max (10 cores): quick ~1 min, default ~4 min, contest ~25 min.

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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: runs/<timestamp>)")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    return parser.parse_args()


# =============================================================================
# Logging setup — dual output (console + file)
# =============================================================================

def setup_logging(output_dir: str, verbose: bool) -> logging.Logger:
    """Configure logging to stream to both console and a log file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "console.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler — INFO level (or DEBUG if verbose)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console)

    # File handler — always DEBUG level for full record
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


# =============================================================================
# Live progress callback
# =============================================================================

class ProgressTracker:
    """Streams per-task progress to console and JSONL file."""

    def __init__(self, progress_path: str, t0: float):
        self._file = open(progress_path, "w")
        self._t0 = t0
        self._solved_total = 0
        self._tasks_total = 0

    def on_task_done(
        self,
        round_num: int,
        task_index: int,
        total_tasks: int,
        wr: WakeResult,
    ) -> None:
        """Called after each task completes. Streams live output."""
        self._tasks_total += 1
        if wr.solved:
            self._solved_total += 1

        elapsed = time.time() - self._t0
        status = "SOLVED" if wr.solved else "      "
        energy_str = f"{wr.best.energy:.4f}" if wr.best else "    N/A"

        # Live console line
        print(
            f"  R{round_num} [{task_index:>3}/{total_tasks}] "
            f"{status} {wr.task_id:<20s} "
            f"E={energy_str}  gens={wr.generations_used:<4d} "
            f"evals={wr.evaluations:<6d} {wr.wall_time:.1f}s  "
            f"[{elapsed:.0f}s elapsed]",
            flush=True,
        )

        # Live JSONL record
        record = {
            "round": round_num,
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
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

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

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("runs", run_timestamp)

    logger = setup_logging(output_dir, args.verbose)

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
    # Load tasks
    # -------------------------------------------------------------------------
    data_dir = args.data_dir or find_arc_data()

    if data_dir:
        logger.info(f"Loading ARC-AGI tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        logger.info(f"Loaded {len(tasks)} tasks")
    else:
        logger.info("ARC dataset not found. Using built-in sample tasks.")
        logger.info("  (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
        tasks = make_sample_tasks()
        logger.info(f"Created {len(tasks)} sample ARC tasks")

    if not tasks:
        logger.error("No tasks loaded.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Wire up 4 interfaces + create learner
    # -------------------------------------------------------------------------
    env = ARCEnv()
    grammar = ARCGrammar(seed=args.seed)
    drive = ARCDrive()
    memory = InMemoryStore()

    # Compute budget per task = beam_width × max_generations
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

    # -------------------------------------------------------------------------
    # Print run config
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: ARC-AGI-1 CURRICULUM TRAINING")
    logger.info(f"  Mode:       {args.mode}")
    logger.info(f"  Tasks:      {len(tasks)}")
    logger.info(f"  Rounds:     {rounds}")
    logger.info(f"  Beam:       {beam_width}")
    logger.info(f"  Gens:       {max_gens}")
    logger.info(f"  Budget:     ~{evals_per_task:,} evals/task, ~{total_budget:,} total")
    logger.info(f"  Workers:    {workers} / {machine['cpu_count']} cores ({machine.get('chip', machine['arch'])})")
    logger.info(f"  Primitives: {len(grammar.base_primitives())}")
    logger.info(f"  Seed:       {args.seed}")
    logger.info(f"  Output:     {output_dir}/")
    logger.info(f"  Timestamp:  {run_timestamp}")
    logger.info(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Set up live progress tracker
    # -------------------------------------------------------------------------
    progress_path = os.path.join(output_dir, f"{run_timestamp}_progress.jsonl")
    tracker = ProgressTracker(progress_path, time.time())

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
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
    # Report
    # -------------------------------------------------------------------------
    metrics = extract_metrics(results)
    total_evals = sum(wr.evaluations for rr in results for wr in rr.wake_results)

    print(f"\n{'='*70}")
    print("COMPOUNDING CURVE — THE KEY METRIC")
    print(f"{'='*70}")
    print_compounding_table(metrics)
    print(f"\nTotal wall time: {total_time:.1f}s")
    print(f"Total evaluations: {total_evals:,}")

    if len(metrics) >= 2:
        first_rate = metrics[0].solve_rate
        last_rate = metrics[-1].solve_rate
        if last_rate > first_rate:
            print(f"\n>>> COMPOUNDING DETECTED: {first_rate:.1%} -> {last_rate:.1%}")
        elif last_rate == first_rate:
            print(f"\n>>> PLATEAU: solve rate stayed at {first_rate:.1%}")
        else:
            print(f"\n>>> REGRESSION: {first_rate:.1%} -> {last_rate:.1%}")

    # -------------------------------------------------------------------------
    # Save results — all files share the same timestamp prefix
    # -------------------------------------------------------------------------
    save_metrics_json(metrics, os.path.join(output_dir, f"{run_timestamp}_metrics.json"))
    save_metrics_csv(metrics, os.path.join(output_dir, f"{run_timestamp}_metrics.csv"))
    memory.save(os.path.join(output_dir, f"{run_timestamp}_library.json"))

    # Save run config for reproducibility
    run_config = {
        "timestamp": run_timestamp,
        "mode": args.mode,
        "rounds": rounds,
        "beam_width": beam_width,
        "max_generations": max_gens,
        "evals_per_task": evals_per_task,
        "workers": workers,
        "seed": args.seed,
        "n_tasks": len(tasks),
        "n_primitives": len(grammar.base_primitives()),
        "total_time_s": round(total_time, 1),
        "total_evaluations": total_evals,
        "machine": machine,
    }
    with open(os.path.join(output_dir, f"{run_timestamp}_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Library summary
    library = memory.get_library()
    print(f"\nLibrary: {len(library)} learned abstractions")
    for entry in library[:20]:
        print(f"  {entry.name}: {entry.program} "
              f"(useful={entry.usefulness:.1f}, reused={entry.reuse_count}x, "
              f"from {len(entry.source_tasks)} tasks)")
    if len(library) > 20:
        print(f"  ... and {len(library) - 20} more")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
