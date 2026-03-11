"""
Phase 1: ARC-AGI-1 Training & Evaluation.

Thin wrapper around the generic core runner. Provides:
- ARC dataset loading (auto-detects or uses built-in samples)
- ARC-specific search tuning (energy_beta, solve_threshold)
- Train/eval pipeline: train produces a culture file, eval loads it

The actual experiment loop, progress tracking, output formatting,
and results saving are all in core/runner.py — domain-agnostic.

Usage:
    python -m experiments.phase1_arc                      # train on training set
    python -m experiments.phase1_arc --mode quick          # fast dev loop
    python -m experiments.phase1_arc --mode contest         # max accuracy
    python -m experiments.phase1_arc --pipeline             # train then eval
    python -m experiments.phase1_arc --eval --culture runs/XXX_culture.json
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
from domains.arc import (
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
    "data/ARC-AGI/data/{split}",
    "../ARC-AGI/data/{split}",
    os.path.expanduser("~/ARC-AGI/data/{split}"),
    "data/arc-agi/data/{split}",
]


def find_arc_data(split: str = "training") -> str | None:
    """Return the first existing ARC data directory for the given split."""
    for pattern in ARC_DATA_SEARCH_PATHS:
        path = pattern.format(split=split)
        if os.path.isdir(path):
            return path
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser(
        description="Phase 1: ARC-AGI-1 Curriculum Training & Evaluation",
        domain_name="phase1_arc",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI data dir (auto-detected if not set)")
    parser.add_argument("--eval", action="store_true",
                        help="Run on evaluation set (requires --culture for cross-run transfer)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Full pipeline: train on training set, then eval on evaluation set")
    args = parser.parse_args()

    # Resolve preset + overrides
    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)
    max_tasks = resolved["max_tasks"]

    if args.pipeline:
        # Full pipeline: train then eval
        _run_train_eval_pipeline(args, resolved, max_tasks)
    elif args.eval:
        # Eval only
        _run_split(args, resolved, max_tasks, split="evaluation")
    else:
        # Train only (default)
        _run_split(args, resolved, max_tasks, split="training")


def _run_split(args, resolved, max_tasks, split="training"):
    """Run experiment on a single data split."""
    data_dir = args.data_dir or find_arc_data(split)

    if data_dir:
        print(f"  Loading ARC-AGI {split} tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    else:
        if split == "evaluation":
            print(f"  ERROR: Evaluation data not found. Searched:")
            for p in ARC_DATA_SEARCH_PATHS:
                print(f"    {p.format(split=split)}")
            sys.exit(1)
        print("  ARC dataset not found. Using built-in sample tasks.")
        print("    (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
        tasks = make_sample_tasks()
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        print(f"  Created {len(tasks)} sample ARC tasks")

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    title = f"PHASE 1: ARC-AGI-1 {split.upper()}"
    domain_tag = f"phase1_{split[:5]}"

    run_experiment(ExperimentConfig(
        title=title,
        domain_tag=domain_tag,
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
        exhaustive_depth=args.exhaustive_depth,
        exhaustive_top_k=args.exhaustive_top_k,
        sequential_compounding=args.sequential_compounding,
        culture_path=args.culture,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        mode=args.mode,
    ))


def _run_train_eval_pipeline(args, resolved, max_tasks):
    """Full pipeline: train → save culture → eval with culture."""
    print("=" * 72)
    print("  PIPELINE MODE: Train → Save Culture → Evaluate")
    print("=" * 72)
    print()

    # Step 1: Train
    print("  STEP 1: Training...")
    print()
    train_dir = args.data_dir or find_arc_data("training")
    if not train_dir:
        print("  ERROR: Training data not found.")
        sys.exit(1)

    tasks = load_arc_dataset(train_dir, max_tasks=max_tasks)
    print(f"  Loaded {len(tasks)} training tasks")

    # Run training — culture file will be auto-saved by runner
    run_experiment(ExperimentConfig(
        title="PHASE 1: ARC-AGI-1 TRAINING (pipeline)",
        domain_tag="phase1_train",
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
        exhaustive_depth=args.exhaustive_depth,
        exhaustive_top_k=args.exhaustive_top_k,
        sequential_compounding=args.sequential_compounding,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        mode=args.mode,
    ))

    # Find the most recent culture file
    import glob
    culture_files = sorted(glob.glob(os.path.join(args.runs_dir, "*_culture.json")))
    if not culture_files:
        print("  WARNING: No culture file produced. Running eval without culture.")
        culture_path = ""
    else:
        culture_path = culture_files[-1]
        print(f"\n  Culture file: {culture_path}")

    # Step 2: Evaluate
    print()
    print("  STEP 2: Evaluating...")
    print()
    eval_dir = find_arc_data("evaluation")
    if not eval_dir:
        print("  ERROR: Evaluation data not found.")
        sys.exit(1)

    eval_tasks = load_arc_dataset(eval_dir, max_tasks=max_tasks)
    print(f"  Loaded {len(eval_tasks)} evaluation tasks")

    run_experiment(ExperimentConfig(
        title="PHASE 1: ARC-AGI-1 EVALUATION (pipeline)",
        domain_tag="phase1_eval",
        tasks=eval_tasks,
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
        exhaustive_depth=args.exhaustive_depth,
        exhaustive_top_k=args.exhaustive_top_k,
        sequential_compounding=args.sequential_compounding,
        culture_path=culture_path,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        mode=args.mode,
    ))

    print()
    print("=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
