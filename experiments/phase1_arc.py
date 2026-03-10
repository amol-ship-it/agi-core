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
import sys
import time

from core import (
    Learner,
    InMemoryStore,
    SearchConfig,
    SleepConfig,
    CurriculumConfig,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: ARC-AGI-1 Curriculum Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
                        help="Preset: quick, default, or contest")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI training dir (auto-detected)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (0 = all)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Wake-sleep rounds (overrides preset)")
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Beam width (overrides preset)")
    parser.add_argument("--max-generations", type=int, default=None,
                        help="Max generations per task (overrides preset)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = all cores)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Machine + preset resolution
    # -------------------------------------------------------------------------
    machine = detect_machine()
    preset = PRESETS[args.mode]

    rounds = args.rounds if args.rounds is not None else preset["rounds"]
    beam_width = args.beam_width if args.beam_width is not None else preset["beam_width"]
    max_gens = args.max_generations if args.max_generations is not None else preset["max_generations"]
    max_tasks = args.max_tasks if args.max_tasks is not None else preset["max_tasks"]
    workers = args.workers if args.workers > 0 else machine["cpu_count"]

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
    # This is implicit — no separate knob needed.
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
    logger.info(f"  Workers:    {workers} ({machine.get('chip', machine['arch'])})")
    logger.info(f"  Primitives: {len(grammar.base_primitives())}")
    logger.info(f"  Seed:       {args.seed}")
    logger.info(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Set up live JSONL progress log
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "phase1_progress.jsonl")
    progress_file = open(progress_path, "w")

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
    )
    total_time = time.time() - t0

    # Write per-task results to JSONL
    for rr in results:
        for wr in rr.wake_results:
            record = {
                "round": rr.round_number,
                "task_id": wr.task_id,
                "solved": wr.solved,
                "energy": wr.best.energy if wr.best else None,
                "prediction_error": wr.best.prediction_error if wr.best else None,
                "generations": wr.generations_used,
                "evaluations": wr.evaluations,
                "wall_time": round(wr.wall_time, 3),
                "program": repr(wr.best.program) if wr.best else None,
            }
            progress_file.write(json.dumps(record) + "\n")
    progress_file.close()

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
    # Save results
    # -------------------------------------------------------------------------
    save_metrics_json(metrics, os.path.join(args.output_dir, "phase1_metrics.json"))
    save_metrics_csv(metrics, os.path.join(args.output_dir, "phase1_metrics.csv"))
    memory.save(os.path.join(args.output_dir, "phase1_library.json"))

    # Save run config for reproducibility
    run_config = {
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
    with open(os.path.join(args.output_dir, "phase1_config.json"), "w") as f:
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

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
