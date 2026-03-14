"""
ARC-AGI-1 Training & Evaluation.

Thin wrapper around the generic core runner. Provides:
- ARC dataset loading (auto-detects or uses built-in samples)
- ARC-specific search tuning (energy_beta, solve_threshold)
- Train/eval pipeline: train produces a culture file, eval loads it
- Auto-generated HTML visualization of results

Usage:
    python -m experiments.phase1_arc                      # full pipeline (train → eval)
    python -m experiments.phase1_arc --mode quick          # fast dev loop (pipeline)
    python -m experiments.phase1_arc --train-only          # train on training set only
    python -m experiments.phase1_arc --eval-only --culture runs/XXX_culture.json
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from core import (
    ExperimentConfig, run_experiment, make_parser,
    resolve_from_preset, PRESETS,
)
from domains.arc import (
    ARCEnv, ARCGrammar, ARCDrive, make_sample_tasks, load_arc_dataset,
)
from .pipeline_common import save_pipeline_results, print_pipeline_summary, pipeline_tee


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
    for pattern in ARC_DATA_SEARCH_PATHS:
        path = pattern.format(split=split)
        if os.path.isdir(path):
            return path
    return None


def _load_tasks(split, data_dir, max_tasks):
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


def _make_config(args, resolved, max_tasks, *, title, domain_tag, tasks,
                 culture_path="", save_culture="", timestamp="",
                 suppress_files=False):
    """Build an ExperimentConfig with ARC-specific defaults."""
    compounding = getattr(args, "compounding", False)
    vocabulary = getattr(args, "vocabulary", "full")
    exhaustive_depth = args.exhaustive_depth
    rounds = resolved["rounds"]
    sequential = args.sequential_compounding
    min_occurrences = 2
    energy_beta = 0.002

    if compounding:
        exhaustive_depth = min(exhaustive_depth, 2)
        rounds = max(rounds, 3)
        sequential = True
        min_occurrences = 1
        energy_beta = 0.01

    return ExperimentConfig(
        title=title, domain_tag=domain_tag, tasks=tasks,
        environment=ARCEnv(), grammar=ARCGrammar(seed=args.seed, vocabulary=vocabulary),
        drive=ARCDrive(),
        rounds=rounds, beam_width=resolved["beam_width"],
        max_generations=resolved["max_generations"],
        workers=resolved["workers"], seed=args.seed,
        compute_cap=resolved["compute_cap"],
        mutations_per_candidate=2, crossover_fraction=0.3,
        energy_alpha=1.0, energy_beta=energy_beta, solve_threshold=0.001,
        exhaustive_depth=exhaustive_depth,
        exhaustive_pair_top_k=args.exhaustive_pair_top_k,
        exhaustive_triple_top_k=args.exhaustive_triple_top_k,
        sequential_compounding=sequential,
        adaptive_realloc=getattr(args, "adaptive_realloc", False),
        min_occurrences=min_occurrences,
        culture_path=culture_path, save_culture=save_culture,
        runs_dir=args.runs_dir, no_log=args.no_log,
        task_ids=getattr(args, "task_ids", ""), mode=args.mode,
        timestamp=timestamp, suppress_files=suppress_files,
    )


def _try_generate_viz(results_json_path: str) -> list[str]:
    """Generate HTML visualization from results JSON. Returns list of index paths."""
    if not os.path.exists(results_json_path):
        return []
    try:
        from .visualize_results import generate_html
        output_base = os.path.splitext(results_json_path)[0]
        return generate_html(results_json_path, output_base)
    except Exception as e:
        print(f"  (visualization skipped: {e})")
        return []


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser(
        description="ARC-AGI-1 Curriculum Training & Evaluation",
        domain_name="phase1_arc",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI data dir (auto-detected if not set)")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval", action="store_true", dest="eval_only",
                        help="Alias for --eval-only")
    parser.add_argument("--pipeline", action="store_true",
                        help="(default behavior, kept for backward compat)")
    args = parser.parse_args()

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
            _run_pipeline(args, resolved, max_tasks)
    except KeyboardInterrupt:
        print("\n\nAborted by user.\n")
        sys.exit(1)


def _run_train(args, resolved, max_tasks):
    tasks = _load_tasks("training", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="ARC-AGI-1 TRAINING",
                       domain_tag="phase1_arc_train", tasks=tasks,
                       save_culture=args.save_culture)
    result = run_experiment(cfg)
    viz_paths = _try_generate_viz(result.results_path)
    for vp in viz_paths:
        print(f"  Visualization: {vp}")
    print(f"  Culture saved to: {result.culture_path}")


def _run_eval(args, resolved, max_tasks):
    tasks = _load_tasks("evaluation", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="ARC-AGI-1 EVALUATION",
                       domain_tag="phase1_arc_eval", tasks=tasks,
                       culture_path=args.culture)
    result = run_experiment(cfg)
    viz_paths = _try_generate_viz(result.results_path)
    for vp in viz_paths:
        print(f"  Visualization: {vp}")


def _run_pipeline(args, resolved, max_tasks):
    shared_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"phase1_arc_pipeline_{shared_ts}"
    log_path = os.path.join(args.runs_dir, f"{prefix}.log")
    os.makedirs(args.runs_dir, exist_ok=True)

    with pipeline_tee(log_path):
        print("=" * 72)
        print("  PIPELINE MODE: Train → Save Culture → Evaluate")
        print("=" * 72)
        print()

        # Step 1: Train
        print("  STEP 1/2: Training...")
        print()
        train_tasks = _load_tasks("training", args.data_dir, max_tasks)
        train_cfg = _make_config(args, resolved, max_tasks,
                                 title="ARC-AGI-1 TRAINING",
                                 domain_tag="phase1_arc_train", tasks=train_tasks,
                                 timestamp=shared_ts, suppress_files=True)
        train_result = run_experiment(train_cfg)
        print(f"\n  Culture file: {train_result.culture_path}")

        # Step 2: Evaluate
        print()
        print("  STEP 2/2: Evaluating with learned culture...")
        print()
        eval_tasks = _load_tasks("evaluation", args.data_dir, max_tasks)
        eval_cfg = _make_config(args, resolved, max_tasks,
                                title="ARC-AGI-1 EVALUATION",
                                domain_tag="phase1_arc_eval", tasks=eval_tasks,
                                culture_path=train_result.culture_path,
                                timestamp=shared_ts, suppress_files=True)
        eval_result = run_experiment(eval_cfg)

        # Save combined pipeline artifacts
        json_path, jsonl_path = save_pipeline_results(
            train_result, eval_result,
            prefix=prefix, runs_dir=args.runs_dir,
            title="ARC-AGI-1 FULL PIPELINE (Train → Eval)",
            domain="phase1_arc", args=args, resolved=resolved,
        )

        # Generate HTML visualization from pipeline results
        viz_paths = _try_generate_viz(json_path)

        print_pipeline_summary(
            train_result, eval_result,
            title="PIPELINE SUMMARY", args=args,
            json_path=json_path, jsonl_path=jsonl_path,
            log_path=log_path,
            viz_paths=viz_paths,
        )


if __name__ == "__main__":
    main()
