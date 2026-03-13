"""
Phase 2: ARC-AGI-2 Experiment.

Thin wrapper around the generic core runner. Provides:
- ARC-AGI-2 dataset loading (auto-detects data/ARC-AGI-2/)
- ARC-specific search tuning (energy_beta, solve_threshold)
- Pipeline mode: train on ARC-AGI-2 training data, eval on ARC-AGI-2

Usage:
    python -m experiments.phase2_arc                      # full pipeline (train → eval on AGI-2)
    python -m experiments.phase2_arc --mode quick          # fast dev loop (pipeline)
    python -m experiments.phase2_arc --eval-only --culture runs/XXX_culture.json
    python -m experiments.phase2_arc --train-only          # train on ARC-AGI-2 training set only
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
# ARC-AGI-2 dataset auto-detection
# =============================================================================

ARC2_DATA_SEARCH_PATHS = [
    "data/ARC-AGI-2/data/{split}",
    "../ARC-AGI-2/data/{split}",
    os.path.expanduser("~/ARC-AGI-2/data/{split}"),
    "data/arc-agi-2/data/{split}",
]

ARC1_DATA_SEARCH_PATHS = [
    "data/ARC-AGI/data/{split}",
    "../ARC-AGI/data/{split}",
    os.path.expanduser("~/ARC-AGI/data/{split}"),
    "data/arc-agi/data/{split}",
]


def _find_data(search_paths, split):
    for pattern in search_paths:
        path = pattern.format(split=split)
        if os.path.isdir(path):
            return path
    return None


def find_arc2_data(split="evaluation"):
    return _find_data(ARC2_DATA_SEARCH_PATHS, split)


def find_arc1_data(split="training"):
    return _find_data(ARC1_DATA_SEARCH_PATHS, split)


def _load_arc2_tasks(split, data_dir, max_tasks):
    data_dir = data_dir or find_arc2_data(split)
    if data_dir:
        print(f"  Loading ARC-AGI-2 {split} tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    else:
        print(f"  ERROR: ARC-AGI-2 {split} data not found. Searched:")
        for p in ARC2_DATA_SEARCH_PATHS:
            print(f"    {p.format(split=split)}")
        print("\n  To get the data:")
        print("    git clone https://github.com/arcprize/arc-agi.git data/ARC-AGI-2")
        sys.exit(1)
    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)
    return tasks


def _load_train_tasks(data_dir, max_tasks):
    """Load training tasks — prefer ARC-AGI-2 training, fall back to ARC-AGI-1."""
    arc2_dir = data_dir or find_arc2_data("training")
    if arc2_dir:
        print(f"  Loading ARC-AGI-2 training tasks from {arc2_dir}...")
        tasks = load_arc_dataset(arc2_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
        return tasks

    arc1_dir = find_arc1_data("training")
    if arc1_dir:
        print(f"  ARC-AGI-2 training not found, falling back to ARC-AGI-1...")
        print(f"  Loading ARC-AGI-1 training tasks from {arc1_dir}...")
        tasks = load_arc_dataset(arc1_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
        return tasks

    print("  ARC training data not found. Using built-in sample tasks.")
    tasks = make_sample_tasks()
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    print(f"  Created {len(tasks)} sample ARC tasks")
    return tasks


def _make_config(args, resolved, max_tasks, *, title, domain_tag, tasks,
                 culture_path="", save_culture="", timestamp="",
                 suppress_files=False):
    return ExperimentConfig(
        title=title, domain_tag=domain_tag, tasks=tasks,
        environment=ARCEnv(), grammar=ARCGrammar(seed=args.seed), drive=ARCDrive(),
        rounds=resolved["rounds"], beam_width=resolved["beam_width"],
        max_generations=resolved["max_generations"],
        workers=resolved["workers"], seed=args.seed,
        compute_cap=resolved["compute_cap"],
        mutations_per_candidate=2, crossover_fraction=0.3,
        energy_alpha=1.0, energy_beta=0.002, solve_threshold=0.001,
        exhaustive_depth=args.exhaustive_depth,
        exhaustive_pair_top_k=args.exhaustive_pair_top_k,
        exhaustive_triple_top_k=args.exhaustive_triple_top_k,
        sequential_compounding=args.sequential_compounding,
        adaptive_realloc=getattr(args, "adaptive_realloc", False),
        culture_path=culture_path, save_culture=save_culture,
        runs_dir=args.runs_dir, no_log=args.no_log, verbose=args.verbose,
        task_ids=getattr(args, "task_ids", ""), mode=args.mode,
        timestamp=timestamp, suppress_files=suppress_files,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser(
        description="ARC-AGI-2 Training & Evaluation",
        domain_name="phase2_arc",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI-2 data dir (auto-detected if not set)")
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Override training data dir (default: ARC-AGI-2 training or ARC-AGI-1 fallback)")
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
    tasks = _load_train_tasks(args.train_data_dir or args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 2: ARC-AGI-2 TRAINING",
                       domain_tag="phase2_arc_train", tasks=tasks,
                       save_culture=args.save_culture)
    result = run_experiment(cfg)
    viz_paths = _try_generate_viz(result.results_path)
    for vp in viz_paths:
        print(f"  Visualization: {vp}")
    print(f"  Culture saved to: {result.culture_path}")


def _try_generate_viz(results_json_path: str) -> list[str]:
    """Generate HTML visualization from results JSON. Returns list of index paths."""
    if not results_json_path or not os.path.exists(results_json_path):
        return []
    try:
        from .visualize_results import generate_html
        output_base = os.path.splitext(results_json_path)[0]
        return generate_html(results_json_path, output_base)
    except Exception as e:
        print(f"  (visualization skipped: {e})")
        return []


def _run_eval(args, resolved, max_tasks):
    tasks = _load_arc2_tasks("evaluation", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 2: ARC-AGI-2 EVALUATION",
                       domain_tag="phase2_arc_eval", tasks=tasks,
                       culture_path=args.culture)
    result = run_experiment(cfg)
    viz_paths = _try_generate_viz(result.results_path)
    for vp in viz_paths:
        print(f"  Visualization: {vp}")


def _run_pipeline(args, resolved, max_tasks):
    shared_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"phase2_arc_pipeline_{shared_ts}"
    log_path = os.path.join(args.runs_dir, f"{prefix}.log")
    os.makedirs(args.runs_dir, exist_ok=True)

    with pipeline_tee(log_path):
        print("=" * 72)
        print("  PIPELINE MODE: Train → Evaluate on ARC-AGI-2")
        print("=" * 72)
        print()

        # Step 1: Train
        print("  STEP 1/2: Training...")
        print()
        train_tasks = _load_train_tasks(args.train_data_dir or args.data_dir, max_tasks)
        train_cfg = _make_config(args, resolved, max_tasks,
                                 title="PHASE 2: TRAINING (for ARC-AGI-2)",
                                 domain_tag="phase2_arc_train", tasks=train_tasks,
                                 timestamp=shared_ts, suppress_files=True)
        train_result = run_experiment(train_cfg)
        print(f"\n  Culture file: {train_result.culture_path}")

        # Step 2: Evaluate on ARC-AGI-2
        print()
        print("  STEP 2/2: Evaluating on ARC-AGI-2 with learned culture...")
        print()
        eval_tasks = _load_arc2_tasks("evaluation", args.data_dir, max_tasks)
        eval_cfg = _make_config(args, resolved, max_tasks,
                                title="PHASE 2: ARC-AGI-2 EVALUATION",
                                domain_tag="phase2_arc_eval", tasks=eval_tasks,
                                culture_path=train_result.culture_path,
                                timestamp=shared_ts, suppress_files=True)
        eval_result = run_experiment(eval_cfg)

        # Save combined pipeline artifacts
        train_src = "ARC-AGI-2 training" if find_arc2_data("training") else "ARC-AGI-1 training"
        json_path, jsonl_path = save_pipeline_results(
            train_result, eval_result,
            prefix=prefix, runs_dir=args.runs_dir,
            title="PHASE 2: ARC-AGI-2 FULL PIPELINE (Train → Eval)",
            domain="phase2_arc", args=args, resolved=resolved,
            extra_meta={"train_source": train_src, "eval_source": "ARC-AGI-2 evaluation"},
        )

        viz_paths = _try_generate_viz(json_path)

        print_pipeline_summary(
            train_result, eval_result,
            title="PHASE 2 PIPELINE SUMMARY", args=args,
            json_path=json_path, jsonl_path=jsonl_path,
            log_path=log_path,
            eval_label="ARC-AGI-2 Evaluation Results (with culture transfer)",
            viz_paths=viz_paths,
        )


if __name__ == "__main__":
    main()
