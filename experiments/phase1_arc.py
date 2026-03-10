"""
Phase 1: ARC-AGI-1 Training, Curriculum Style.

This script implements the Phase 1 experiment from the manifesto:

1. Sort tasks by difficulty (grid size as proxy)
2. Run wake-sleep in curriculum order: easy first, sleep, extract library
3. Track the compounding curve: tasks solved per round should increase
   as the library grows, without any new hand-coded primitives

Usage:
    # Run on sample tasks (no ARC data needed)
    python -m experiments.phase1_arc

    # Run on ARC-AGI-1 training data
    python -m experiments.phase1_arc --data-dir /path/to/arc-agi/training

    # Run on a subset
    python -m experiments.phase1_arc --data-dir /path/to/arc-agi/training --max-tasks 50
"""

from __future__ import annotations

import argparse
import logging
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: ARC-AGI-1 Curriculum Training")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI training directory (JSON files)")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max tasks to load (0 = all)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of wake-sleep rounds")
    parser.add_argument("--beam-width", type=int, default=150,
                        help="Beam width for search")
    parser.add_argument("--max-generations", type=int, default=80,
                        help="Max generations per task")
    parser.add_argument("--solve-threshold", type=float, default=0.001,
                        help="Pixel error threshold for 'solved'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Directory for output files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Load tasks
    # -------------------------------------------------------------------------
    if args.data_dir:
        logger.info(f"Loading ARC-AGI tasks from {args.data_dir}...")
        tasks = load_arc_dataset(args.data_dir, max_tasks=args.max_tasks)
        logger.info(f"Loaded {len(tasks)} tasks")
    else:
        logger.info("No --data-dir specified. Using built-in sample tasks.")
        tasks = make_sample_tasks()
        logger.info(f"Created {len(tasks)} sample ARC tasks")

    if not tasks:
        logger.error("No tasks loaded. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Wire up the 4 interfaces (all domain-specific)
    # -------------------------------------------------------------------------
    env = ARCEnv()
    grammar = ARCGrammar(seed=args.seed)
    drive = ARCDrive()
    memory = InMemoryStore()

    # -------------------------------------------------------------------------
    # Create the learner (the INVARIANT core — no ARC knowledge here)
    # -------------------------------------------------------------------------
    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=SearchConfig(
            beam_width=args.beam_width,
            max_generations=args.max_generations,
            mutations_per_candidate=2,
            crossover_fraction=0.3,
            energy_alpha=1.0,
            energy_beta=0.002,
            solve_threshold=args.solve_threshold,
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
    # Run the curriculum
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: ARC-AGI-1 CURRICULUM TRAINING")
    logger.info(f"  Tasks: {len(tasks)}")
    logger.info(f"  Rounds: {args.rounds}")
    logger.info(f"  Beam width: {args.beam_width}")
    logger.info(f"  Max generations: {args.max_generations}")
    logger.info(f"  Base primitives: {len(grammar.base_primitives())}")
    logger.info(f"{'='*70}\n")

    t0 = time.time()
    results = learner.run_curriculum(
        tasks,
        CurriculumConfig(
            sort_by_difficulty=True,
            wake_sleep_rounds=args.rounds,
        ),
    )
    total_time = time.time() - t0

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    metrics = extract_metrics(results)

    print(f"\n{'='*70}")
    print("COMPOUNDING CURVE — THE KEY METRIC")
    print(f"{'='*70}")
    print_compounding_table(metrics)
    print(f"\nTotal wall time: {total_time:.1f}s")

    # Check for compounding
    if len(metrics) >= 2:
        first_rate = metrics[0].solve_rate
        last_rate = metrics[-1].solve_rate
        if last_rate > first_rate:
            print(f"\n>>> COMPOUNDING DETECTED: solve rate {first_rate:.1%} -> {last_rate:.1%}")
        elif last_rate == first_rate:
            print(f"\n>>> PLATEAU: solve rate stayed at {first_rate:.1%}")
        else:
            print(f"\n>>> REGRESSION: solve rate {first_rate:.1%} -> {last_rate:.1%}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    save_metrics_json(metrics, os.path.join(args.output_dir, "phase1_metrics.json"))
    save_metrics_csv(metrics, os.path.join(args.output_dir, "phase1_metrics.csv"))

    # Save library
    lib_path = os.path.join(args.output_dir, "phase1_library.json")
    memory.save(lib_path)

    # Summary
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
