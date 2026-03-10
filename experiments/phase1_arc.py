"""
Phase 1: ARC-AGI-1 Training — Curriculum Style.

Thin wrapper around the generic core runner. Provides:
- ARC dataset loading (auto-detects or uses built-in samples)
- ARC-specific search tuning (energy_beta, solve_threshold)

The actual experiment loop, progress tracking, output formatting,
and results saving are all in core/runner.py — domain-agnostic.

Usage:
    python -m experiments.phase1_arc                  # just run it
    python -m experiments.phase1_arc --mode quick     # fast dev loop
    python -m experiments.phase1_arc --mode contest   # max accuracy
"""

from __future__ import annotations

import os
import sys

from core import (
    ExperimentConfig,
    run_experiment,
    make_parser,
    resolve_from_preset,
    PRESETS,
)
from grammars.arc import (
    ARCEnv,
    ARCGrammar,
    ARCDrive,
    make_sample_tasks,
    load_arc_dataset,
)


# =============================================================================
# ARC dataset auto-detection
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
# Main
# =============================================================================

def main():
    parser = make_parser(
        description="Phase 1: ARC-AGI-1 Curriculum Training",
        domain_name="phase1_arc",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI training dir (auto-detected if not set)")
    args = parser.parse_args()

    # Resolve preset + overrides
    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)

    # Load tasks
    data_dir = args.data_dir or find_arc_data()
    max_tasks = resolved["max_tasks"]

    if data_dir:
        print(f"  Loading ARC-AGI tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    else:
        print("  ARC dataset not found. Using built-in sample tasks.")
        print("    (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
        tasks = make_sample_tasks()
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        print(f"  Created {len(tasks)} sample ARC tasks")

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    # Run the experiment using the generic core runner
    run_experiment(ExperimentConfig(
        title="PHASE 1: ARC-AGI-1 CURRICULUM TRAINING",
        domain_tag="phase1",
        tasks=tasks,
        environment=ARCEnv(),
        grammar=ARCGrammar(seed=args.seed),
        drive=ARCDrive(),
        rounds=resolved["rounds"],
        beam_width=resolved["beam_width"],
        max_generations=resolved["max_generations"],
        workers=resolved["workers"],
        seed=args.seed,
        compute_cap=args.compute_cap,
        mutations_per_candidate=2,
        crossover_fraction=0.3,
        energy_alpha=1.0,
        energy_beta=0.002,
        solve_threshold=0.001,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
