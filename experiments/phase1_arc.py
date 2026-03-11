"""
Phase 1: ARC-AGI-1 Training & Evaluation.

Thin wrapper around the generic core runner. Provides:
- ARC dataset loading (auto-detects or uses built-in samples)
- ARC-specific search tuning (energy_beta, solve_threshold)
- Train/eval pipeline: train produces a culture file, eval loads it

The actual experiment loop, progress tracking, output formatting,
and results saving are all in core/runner.py — domain-agnostic.

Usage:
    python -m experiments.phase1_arc                      # full pipeline (train → eval)
    python -m experiments.phase1_arc --mode quick          # fast dev loop (pipeline)
    python -m experiments.phase1_arc --train-only          # train on training set only
    python -m experiments.phase1_arc --train-only --save-culture my_culture.json
    python -m experiments.phase1_arc --eval-only --culture runs/XXX_culture.json
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

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


def _load_tasks(split: str, data_dir: str | None, max_tasks: int):
    """Load ARC tasks for a given split, with fallback to samples for training."""
    data_dir = data_dir or find_arc_data(split)

    if data_dir:
        print(f"  Loading ARC-AGI {split} tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    elif split == "training":
        print("  ARC dataset not found. Using built-in sample tasks.")
        print("    (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
        tasks = make_sample_tasks()
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        print(f"  Created {len(tasks)} sample ARC tasks")
    else:
        print(f"  ERROR: {split.capitalize()} data not found. Searched:")
        for p in ARC_DATA_SEARCH_PATHS:
            print(f"    {p.format(split=split)}")
        sys.exit(1)

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    return tasks


def _make_config(args, resolved, max_tasks, *, title: str, domain_tag: str,
                 tasks, culture_path: str = "",
                 save_culture: str = "",
                 timestamp: str = "") -> ExperimentConfig:
    """Build an ExperimentConfig with ARC-specific defaults."""
    return ExperimentConfig(
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
        exhaustive_pair_top_k=args.exhaustive_pair_top_k,
        exhaustive_triple_top_k=args.exhaustive_triple_top_k,
        sequential_compounding=args.sequential_compounding,
        culture_path=culture_path,
        save_culture=save_culture,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        mode=args.mode,
        timestamp=timestamp,
    )


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
    parser.add_argument("--train-only", action="store_true",
                        help="Run on training set only (no eval)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run on evaluation set only (requires --culture)")
    # Keep --eval and --pipeline as hidden aliases for backward compat
    parser.add_argument("--eval", action="store_true", dest="eval_only",
                        help="Alias for --eval-only")
    parser.add_argument("--pipeline", action="store_true",
                        help="(default behavior, kept for backward compat)")
    args = parser.parse_args()

    # Resolve preset + overrides
    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)
    max_tasks = resolved["max_tasks"]

    try:
        if args.eval_only:
            if not args.culture:
                print("  ERROR: --eval-only requires --culture <path>")
                sys.exit(1)
            _run_eval(args, resolved, max_tasks)
        elif args.train_only:
            _run_train(args, resolved, max_tasks)
        else:
            # Default: full pipeline (train → eval)
            _run_pipeline(args, resolved, max_tasks)
    except KeyboardInterrupt:
        print("\n\nAborted by user.\n")
        sys.exit(1)


def _run_train(args, resolved, max_tasks):
    """Run training only."""
    tasks = _load_tasks("training", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 1: ARC-AGI-1 TRAINING",
                       domain_tag="phase1_train",
                       tasks=tasks,
                       save_culture=args.save_culture)
    culture_path = run_experiment(cfg)
    print(f"  Culture saved to: {culture_path}")


def _run_eval(args, resolved, max_tasks):
    """Run evaluation only with a pre-trained culture file."""
    tasks = _load_tasks("evaluation", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 1: ARC-AGI-1 EVALUATION",
                       domain_tag="phase1_eval",
                       tasks=tasks,
                       culture_path=args.culture)
    run_experiment(cfg)


def _run_pipeline(args, resolved, max_tasks):
    """Full pipeline: train → save culture → eval with culture."""
    # Shared timestamp so train+eval artifacts are grouped together
    shared_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  PIPELINE MODE: Train → Save Culture → Evaluate")
    print("=" * 72)
    print()

    # Step 1: Train
    print("  STEP 1/2: Training...")
    print()
    train_tasks = _load_tasks("training", args.data_dir, max_tasks)
    train_cfg = _make_config(args, resolved, max_tasks,
                             title="PHASE 1: ARC-AGI-1 TRAINING",
                             domain_tag="phase1_train",
                             tasks=train_tasks,
                             timestamp=shared_ts)
    culture_path = run_experiment(train_cfg)
    print(f"\n  Culture file: {culture_path}")

    # Step 2: Evaluate using the culture file produced by training
    print()
    print("  STEP 2/2: Evaluating with learned culture...")
    print()
    eval_tasks = _load_tasks("evaluation", args.data_dir, max_tasks)
    eval_cfg = _make_config(args, resolved, max_tasks,
                            title="PHASE 1: ARC-AGI-1 EVALUATION",
                            domain_tag="phase1_eval",
                            tasks=eval_tasks,
                            culture_path=culture_path,
                            timestamp=shared_ts)
    run_experiment(eval_cfg)

    print()
    print("=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
