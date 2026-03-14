"""
ARC-AGI-1 Training & Evaluation.

Thin wrapper around the generic benchmark runner + ARC domain adapter.

Usage:
    python -m experiments.phase1_arc                      # full pipeline (train -> eval)
    python -m experiments.phase1_arc --mode quick          # fast dev loop (pipeline)
    python -m experiments.phase1_arc --train-only          # train on training set only
    python -m experiments.phase1_arc --eval-only --culture runs/XXX_culture.json
"""

from __future__ import annotations

import os
import sys

from common.benchmark import (
    ExperimentConfig, run_experiment, make_parser,
    resolve_from_preset, PRESETS, run_pipeline,
)
from domains.arc.adapter import ARCAdapter

# Backward-compatible re-export used by downstream scripts
from domains.arc.dataset import find_arc_data  # noqa: F401


_adapter = ARCAdapter(benchmark="arc-agi-1")


def _make_config(args, resolved, max_tasks, tasks, timestamp,
                 *, culture_path="", save_culture="", suppress_files=False,
                 split_label=""):
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

    domain_tag = "phase1_arc_train" if "TRAIN" in split_label else (
        "phase1_arc_eval" if "EVAL" in split_label else "phase1_arc")

    env, grammar, drive = _adapter.create_interfaces(seed=args.seed, vocabulary=vocabulary)
    return ExperimentConfig(
        title=f"ARC-AGI-1 {split_label}" if split_label else "ARC-AGI-1",
        domain_tag=domain_tag, tasks=tasks,
        environment=env, grammar=grammar, drive=drive,
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
        split_label=split_label,
        default_cell_size=_adapter.default_cell_size(),
    )


def _try_generate_viz(results_json_path: str) -> list[str]:
    """Generate HTML visualization from results JSON."""
    if not os.path.exists(results_json_path):
        return []
    try:
        from .visualize_results import generate_html
        output_base = os.path.splitext(results_json_path)[0]
        return generate_html(results_json_path, output_base)
    except Exception as e:
        print(f"  (visualization skipped: {e})")
        return []


# Keep _load_tasks as backward-compatible function for downstream consumers
def _load_tasks(split, data_dir, max_tasks):
    return _adapter.load_tasks(split, data_dir, max_tasks)


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
            tasks = _adapter.load_tasks("evaluation", args.data_dir, max_tasks)
            cfg = _make_config(args, resolved, max_tasks, tasks, "",
                               culture_path=args.culture, split_label="EVALUATION")
            result = run_experiment(cfg)
            for vp in _try_generate_viz(result.results_path):
                print(f"  Visualization: {vp}")
        elif args.train_only:
            tasks = _adapter.load_tasks("training", args.data_dir, max_tasks)
            cfg = _make_config(args, resolved, max_tasks, tasks, "",
                               save_culture=args.save_culture, split_label="TRAINING")
            result = run_experiment(cfg)
            for vp in _try_generate_viz(result.results_path):
                print(f"  Visualization: {vp}")
            print(f"  Culture saved to: {result.culture_path}")
        else:
            run_pipeline(
                make_train_config=lambda a, r, m, tasks, ts: _make_config(
                    a, r, m, tasks, ts, suppress_files=True, split_label="TRAINING"),
                make_eval_config=lambda a, r, m, tasks, cp, ts: _make_config(
                    a, r, m, tasks, ts, culture_path=cp, suppress_files=True,
                    split_label="EVALUATION"),
                load_train_tasks=lambda m: _adapter.load_tasks("training", args.data_dir, m),
                load_eval_tasks=lambda m: _adapter.load_tasks("evaluation", args.data_dir, m),
                args=args, resolved=resolved, max_tasks=max_tasks,
                pipeline_prefix="phase1_arc_pipeline",
                pipeline_title="ARC-AGI-1 FULL PIPELINE (Train -> Eval)",
                domain_name="phase1_arc",
                try_generate_viz=_try_generate_viz,
            )
    except KeyboardInterrupt:
        print("\n\nAborted by user.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
