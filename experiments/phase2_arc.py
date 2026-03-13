"""
Phase 2: ARC-AGI-2 Baseline Experiment.

Thin wrapper around the generic core runner. Provides:
- ARC-AGI-2 dataset loading (auto-detects data/ARC-AGI-2/)
- ARC-specific search tuning (energy_beta, solve_threshold)
- Pipeline mode: train on ARC-AGI-1 training data, eval on ARC-AGI-2

ARC-AGI-2 contains 120 evaluation tasks that are harder than ARC-AGI-1.
The pipeline mode trains on ARC-AGI-1 training data (1000 tasks in ARC-AGI-2
or 400 tasks from ARC-AGI-1) and evaluates on the ARC-AGI-2 evaluation set.

Usage:
    python -m experiments.phase2_arc                      # full pipeline (train → eval on AGI-2)
    python -m experiments.phase2_arc --mode quick          # fast dev loop (pipeline)
    python -m experiments.phase2_arc --eval-only --culture runs/XXX_culture.json
    python -m experiments.phase2_arc --train-only          # train on ARC-AGI-2 training set only
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from core import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    make_parser,
    resolve_from_preset,
    PRESETS,
    fmt_duration,
)
from domains.arc import (
    ARCEnv,
    ARCGrammar,
    ARCDrive,
    make_sample_tasks,
    load_arc_dataset,
)


# =============================================================================
# ARC-AGI-2 dataset auto-detection
# =============================================================================

# ARC-AGI-2 data paths (evaluation + training)
ARC2_DATA_SEARCH_PATHS = [
    "data/ARC-AGI-2/data/{split}",
    "../ARC-AGI-2/data/{split}",
    os.path.expanduser("~/ARC-AGI-2/data/{split}"),
    "data/arc-agi-2/data/{split}",
]

# ARC-AGI-1 data paths (for pipeline training fallback)
ARC1_DATA_SEARCH_PATHS = [
    "data/ARC-AGI/data/{split}",
    "../ARC-AGI/data/{split}",
    os.path.expanduser("~/ARC-AGI/data/{split}"),
    "data/arc-agi/data/{split}",
]


def _find_data(search_paths: list[str], split: str) -> str | None:
    """Return the first existing data directory for the given split."""
    for pattern in search_paths:
        path = pattern.format(split=split)
        if os.path.isdir(path):
            return path
    return None


def find_arc2_data(split: str = "evaluation") -> str | None:
    """Find ARC-AGI-2 data directory for the given split."""
    return _find_data(ARC2_DATA_SEARCH_PATHS, split)


def find_arc1_data(split: str = "training") -> str | None:
    """Find ARC-AGI-1 data directory for the given split."""
    return _find_data(ARC1_DATA_SEARCH_PATHS, split)


def _load_arc2_tasks(split: str, data_dir: str | None, max_tasks: int):
    """Load ARC-AGI-2 tasks for a given split."""
    data_dir = data_dir or find_arc2_data(split)

    if data_dir:
        print(f"  Loading ARC-AGI-2 {split} tasks from {data_dir}...")
        tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
    else:
        print(f"  ERROR: ARC-AGI-2 {split} data not found. Searched:")
        for p in ARC2_DATA_SEARCH_PATHS:
            print(f"    {p.format(split=split)}")
        print()
        print("  To get the data:")
        print("    git clone https://github.com/arcprize/arc-agi.git data/ARC-AGI-2")
        sys.exit(1)

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    return tasks


def _load_train_tasks(data_dir: str | None, max_tasks: int):
    """Load training tasks — prefer ARC-AGI-2 training, fall back to ARC-AGI-1."""
    # First try ARC-AGI-2 training data
    arc2_train_dir = data_dir or find_arc2_data("training")
    if arc2_train_dir:
        print(f"  Loading ARC-AGI-2 training tasks from {arc2_train_dir}...")
        tasks = load_arc_dataset(arc2_train_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
        return tasks

    # Fall back to ARC-AGI-1 training data
    arc1_train_dir = find_arc1_data("training")
    if arc1_train_dir:
        print(f"  ARC-AGI-2 training not found, falling back to ARC-AGI-1...")
        print(f"  Loading ARC-AGI-1 training tasks from {arc1_train_dir}...")
        tasks = load_arc_dataset(arc1_train_dir, max_tasks=max_tasks)
        print(f"  Loaded {len(tasks)} tasks")
        return tasks

    # Last resort: built-in samples
    print("  ARC training data not found. Using built-in sample tasks.")
    tasks = make_sample_tasks()
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    print(f"  Created {len(tasks)} sample ARC tasks")
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
        compute_cap=resolved["compute_cap"],
        mutations_per_candidate=2,
        crossover_fraction=0.3,
        energy_alpha=1.0,
        energy_beta=0.01,
        solve_threshold=0.001,
        exhaustive_depth=args.exhaustive_depth,
        exhaustive_pair_top_k=args.exhaustive_pair_top_k,
        exhaustive_triple_top_k=args.exhaustive_triple_top_k,
        sequential_compounding=args.sequential_compounding,
        adaptive_realloc=getattr(args, "adaptive_realloc", False),
        culture_path=culture_path,
        save_culture=save_culture,
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        verbose=args.verbose,
        task_ids=getattr(args, "task_ids", ""),
        mode=args.mode,
        timestamp=timestamp,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser(
        description="Phase 2: ARC-AGI-2 Baseline Experiment",
        domain_name="phase2_arc",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC-AGI-2 data dir (auto-detected if not set)")
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Path to training data dir (defaults to ARC-AGI-2 training, "
                             "then ARC-AGI-1 training)")
    parser.add_argument("--train-only", action="store_true",
                        help="Run on training set only (no eval)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run on ARC-AGI-2 evaluation set only (requires --culture)")
    parser.add_argument("--eval", action="store_true", dest="eval_only",
                        help="Alias for --eval-only")
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
            # Default: full pipeline (train → eval on ARC-AGI-2)
            _run_pipeline(args, resolved, max_tasks)
    except KeyboardInterrupt:
        print("\n\nAborted by user.\n")
        sys.exit(1)


def _run_train(args, resolved, max_tasks):
    """Run training only (on ARC-AGI-2 training set or ARC-AGI-1 fallback)."""
    tasks = _load_train_tasks(args.train_data_dir or args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 2: ARC-AGI-2 TRAINING",
                       domain_tag="phase2_train",
                       tasks=tasks,
                       save_culture=args.save_culture)
    result = run_experiment(cfg)
    print(f"  Culture saved to: {result.culture_path}")


def _run_eval(args, resolved, max_tasks):
    """Run evaluation only on ARC-AGI-2 with a pre-trained culture file."""
    tasks = _load_arc2_tasks("evaluation", args.data_dir, max_tasks)
    cfg = _make_config(args, resolved, max_tasks,
                       title="PHASE 2: ARC-AGI-2 EVALUATION",
                       domain_tag="phase2_eval",
                       tasks=tasks,
                       culture_path=args.culture)
    run_experiment(cfg)


def _run_pipeline(args, resolved, max_tasks):
    """Full pipeline: train on training data -> eval on ARC-AGI-2."""
    shared_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  PIPELINE MODE: Train -> Evaluate on ARC-AGI-2")
    print("=" * 72)
    print()

    # Step 1: Train
    print("  STEP 1/2: Training...")
    print()
    train_tasks = _load_train_tasks(args.train_data_dir or args.data_dir, max_tasks)
    train_cfg = _make_config(args, resolved, max_tasks,
                             title="PHASE 2: TRAINING (for ARC-AGI-2)",
                             domain_tag="phase2_train",
                             tasks=train_tasks,
                             timestamp=shared_ts)
    train_result = run_experiment(train_cfg)
    print(f"\n  Culture file: {train_result.culture_path}")

    # Step 2: Evaluate on ARC-AGI-2
    print()
    print("  STEP 2/2: Evaluating on ARC-AGI-2 with learned culture...")
    print()
    eval_tasks = _load_arc2_tasks("evaluation", args.data_dir, max_tasks)
    eval_cfg = _make_config(args, resolved, max_tasks,
                            title="PHASE 2: ARC-AGI-2 EVALUATION",
                            domain_tag="phase2_eval",
                            tasks=eval_tasks,
                            culture_path=train_result.culture_path,
                            timestamp=shared_ts)
    eval_result = run_experiment(eval_cfg)

    # --- Save combined pipeline output files ---
    runs_dir = args.runs_dir
    pipeline_prefix = f"{shared_ts}_phase2_pipeline"
    pipeline_json_path = os.path.join(runs_dir, f"{pipeline_prefix}.json")
    pipeline_jsonl_path = os.path.join(runs_dir, f"{pipeline_prefix}.jsonl")

    # Combined JSONL
    with open(pipeline_jsonl_path, "w") as f:
        for record in train_result.results_data.get("tasks", {}).values():
            record_with_phase = {"phase": "train", **record}
            f.write(json.dumps(record_with_phase) + "\n")
        for record in eval_result.results_data.get("tasks", {}).values():
            record_with_phase = {"phase": "eval", **record}
            f.write(json.dumps(record_with_phase) + "\n")

    # Combined JSON
    train_data = train_result.results_data
    eval_data = eval_result.results_data
    train_summary = train_data.get("summary", {})
    eval_summary = eval_data.get("summary", {})
    train_meta = train_data.get("meta", {})

    pipeline_data = {
        "meta": {
            "timestamp": shared_ts,
            "datetime": datetime.now().isoformat(),
            "title": "PHASE 2: ARC-AGI-2 FULL PIPELINE (Train -> Eval)",
            "domain": "phase2_pipeline",
            "mode": train_meta.get("mode", args.mode),
            "rounds": train_meta.get("rounds", resolved["rounds"]),
            "beam_width": train_meta.get("beam_width", resolved["beam_width"]),
            "max_generations": train_meta.get("max_generations", resolved["max_generations"]),
            "workers": train_meta.get("workers", resolved["workers"]),
            "seed": train_meta.get("seed", args.seed),
            "compute_cap": train_meta.get("compute_cap", resolved["compute_cap"]),
            "n_primitives": train_meta.get("n_primitives"),
            "exhaustive_depth": args.exhaustive_depth,
            "exhaustive_pair_top_k": args.exhaustive_pair_top_k,
            "exhaustive_triple_top_k": args.exhaustive_triple_top_k,
            "machine": train_meta.get("machine", {}),
            "train_source": "ARC-AGI-2 training" if find_arc2_data("training") else "ARC-AGI-1 training",
            "eval_source": "ARC-AGI-2 evaluation",
        },
        "train": {
            "n_tasks": train_summary.get("n_tasks", 0),
            "solved": train_summary.get("last_round_solved", 0),
            "solve_rate": train_summary.get("last_round_solve_rate", 0),
            "train_solved": train_summary.get("last_round_train_solved", 0),
            "train_solve_rate": train_summary.get("last_round_train_solve_rate", 0),
            "total_evaluations": train_summary.get("total_evaluations", 0),
            "wall_clock_seconds": train_summary.get("wall_clock_seconds", 0),
            "median_task_time": train_summary.get("median_task_time"),
            "throughput_tasks_per_sec": train_summary.get("throughput_tasks_per_sec"),
            "library_size": train_summary.get("library_size", 0),
        },
        "eval": {
            "n_tasks": eval_summary.get("n_tasks", 0),
            "solved": eval_summary.get("last_round_solved", 0),
            "solve_rate": eval_summary.get("last_round_solve_rate", 0),
            "train_solved": eval_summary.get("last_round_train_solved", 0),
            "train_solve_rate": eval_summary.get("last_round_train_solve_rate", 0),
            "total_evaluations": eval_summary.get("total_evaluations", 0),
            "wall_clock_seconds": eval_summary.get("wall_clock_seconds", 0),
            "median_task_time": eval_summary.get("median_task_time"),
            "throughput_tasks_per_sec": eval_summary.get("throughput_tasks_per_sec"),
        },
        "train_tasks": train_data.get("tasks", {}),
        "eval_tasks": eval_data.get("tasks", {}),
        "library": train_data.get("library", []),
    }

    with open(pipeline_json_path, "w") as f:
        json.dump(pipeline_data, f, indent=2)

    # --- Print pipeline summary ---
    total_wall = (train_summary.get("wall_clock_seconds", 0)
                  + eval_summary.get("wall_clock_seconds", 0))

    print()
    print("=" * 72)
    print("  PHASE 2 PIPELINE SUMMARY")
    print("=" * 72)

    # Parameters
    print()
    print("  Parameters:")
    print(f"    Mode:              {args.mode}")
    print(f"    Rounds:            {train_meta.get('rounds', '?')}")
    print(f"    Beam:              {train_meta.get('beam_width', '?')}")
    print(f"    Generations:       {train_meta.get('max_generations', '?')}")
    print(f"    Workers:           {train_meta.get('workers', '?')}")
    print(f"    Seed:              {train_meta.get('seed', '?')}")
    cap = train_meta.get("compute_cap", 0)
    print(f"    Compute cap:       {cap:,} ops" if cap else "    Compute cap:       unlimited")
    print(f"    Exhaustive depth:  {args.exhaustive_depth}")
    print(f"    Pair top-K:        {args.exhaustive_pair_top_k}")
    print(f"    Triple top-K:      {args.exhaustive_triple_top_k}")
    print(f"    Primitives:        {train_meta.get('n_primitives', '?')}")

    # Training source info
    print()
    train_src = "ARC-AGI-2 training" if find_arc2_data("training") else "ARC-AGI-1 training"
    print(f"  Training source:     {train_src}")
    print(f"  Evaluation source:   ARC-AGI-2 evaluation (120 tasks)")

    # Train results
    t_solved = train_summary.get("last_round_solved", 0)
    t_total = train_summary.get("n_tasks", 0)
    t_rate = train_summary.get("last_round_solve_rate", 0)
    t_train_solved = train_summary.get("last_round_train_solved", 0)
    t_overfit = t_train_solved - t_solved if t_train_solved > t_solved else 0
    t_wall = train_summary.get("wall_clock_seconds", 0)

    print()
    print("  Training Results:")
    print(f"    Tasks:             {t_total}")
    print(f"    Solved:            {t_solved}/{t_total} ({t_rate:.1%})")
    if t_overfit > 0:
        print(f"    Overfit:           {t_overfit}")
    print(f"    Evaluations:       {train_summary.get('total_evaluations', 0):,}")
    print(f"    Wall time:         {fmt_duration(t_wall)}")
    print(f"    Library learned:   {train_summary.get('library_size', 0)} abstractions")

    # Eval results
    e_solved = eval_summary.get("last_round_solved", 0)
    e_total = eval_summary.get("n_tasks", 0)
    e_rate = eval_summary.get("last_round_solve_rate", 0)
    e_train_solved = eval_summary.get("last_round_train_solved", 0)
    e_overfit = e_train_solved - e_solved if e_train_solved > e_solved else 0
    e_wall = eval_summary.get("wall_clock_seconds", 0)

    print()
    print("  ARC-AGI-2 Evaluation Results (with culture transfer):")
    print(f"    Tasks:             {e_total}")
    print(f"    Solved:            {e_solved}/{e_total} ({e_rate:.1%})")
    if e_overfit > 0:
        print(f"    Overfit:           {e_overfit}")
    print(f"    Evaluations:       {eval_summary.get('total_evaluations', 0):,}")
    print(f"    Wall time:         {fmt_duration(e_wall)}")

    # Total
    print()
    print(f"  Total wall time:     {fmt_duration(total_wall)}")

    # Pipeline artifacts
    print()
    print("  Pipeline artifacts:")
    print(f"    Pipeline JSON:     {pipeline_json_path}")
    print(f"    Pipeline JSONL:    {pipeline_jsonl_path}")
    print(f"    Train results:     {train_result.results_path}")
    print(f"    Eval results:      {eval_result.results_path}")
    print(f"    Culture file:      {train_result.culture_path}")

    print()
    print("=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
