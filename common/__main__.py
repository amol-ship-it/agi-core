"""
Unified CLI entry point for the Universal Learning Loop benchmark.

Usage:
    python -m common --domain arc-agi-1 --mode quick --max-tasks 10
    python -m common --domain arc-agi-1 --mode pipeline --train-data-dir data/ARC-AGI/data/training --eval-data-dir data/ARC-AGI/data/evaluation
    python -m common --domain zork --mode quick
    python -m common --domain list-ops --mode quick
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .benchmark import (
    ExperimentConfig, run_experiment, make_parser,
    resolve_from_preset, PRESETS, run_pipeline,
)

# Domain name -> (module_path, class_name, kwargs)
# Lazy-imported: common/__main__.py never hardcodes domain imports at module level.
DOMAIN_ADAPTERS = {
    "arc-agi-1": ("domains.arc.adapter", "ARCAdapter", {"benchmark": "arc-agi-1"}),
    "arc-agi-2": ("domains.arc.adapter", "ARCAdapter", {"benchmark": "arc-agi-2"}),
    "zork": ("domains.zork.adapter", "ZorkAdapter", {}),
    "list-ops": ("domains.list_ops.adapter", "ListOpsAdapter", {}),
}


def _load_adapter(domain_name: str):
    """Lazy-load a DomainAdapter by name."""
    if domain_name not in DOMAIN_ADAPTERS:
        print(f"  ERROR: Unknown domain '{domain_name}'")
        print(f"  Available domains: {', '.join(sorted(DOMAIN_ADAPTERS))}")
        sys.exit(1)
    mod_path, cls_name, kwargs = DOMAIN_ADAPTERS[domain_name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


def main():
    # Two-phase parsing: first grab --domain, then build the full parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--domain", type=str, required=True,
                            choices=sorted(DOMAIN_ADAPTERS),
                            help="Domain to run")
    pre_parser.add_argument("--run-mode", type=str, default="single",
                            choices=["single", "pipeline"],
                            help="Run mode: 'single' (train or eval) or 'pipeline' (train -> eval)")
    pre_parser.add_argument("--split", type=str, default="training",
                            choices=["training", "evaluation"],
                            help="Data split for single mode")
    pre_parser.add_argument("--data-dir", type=str, default=None,
                            help="Override data directory for task loading")
    pre_parser.add_argument("--train-data-dir", type=str, default=None,
                            help="Override training data directory (pipeline mode)")
    pre_parser.add_argument("--eval-data-dir", type=str, default=None,
                            help="Override evaluation data directory (pipeline mode)")

    # Parse known args first to get --domain
    pre_args, remaining = pre_parser.parse_known_args()

    adapter = _load_adapter(pre_args.domain)
    defaults = adapter.config_defaults()

    # Build full parser with domain info
    parser = make_parser(
        description=f"Universal Learning Loop: {adapter.name()}",
        domain_name="common",
    )
    # Re-add the pre-parser arguments so they show in --help
    parser.add_argument("--domain", type=str, required=True,
                        choices=sorted(DOMAIN_ADAPTERS),
                        help="Domain to run")
    parser.add_argument("--run-mode", type=str, default="pipeline",
                        choices=["single", "pipeline"],
                        help="Run mode: 'pipeline' (train→eval per round) or 'single'")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "evaluation"],
                        help="Data split for single mode")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory for task loading")
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Override training data directory (pipeline mode)")
    parser.add_argument("--eval-data-dir", type=str, default=None,
                        help="Override evaluation data directory (pipeline mode)")
    args = parser.parse_args()

    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)
    max_tasks = resolved["max_tasks"]

    def _make_config(args, resolved, max_tasks, tasks, timestamp,
                     *, culture_path="", suppress_files=False, split_label=""):
        env, grammar, drive = adapter.create_interfaces(seed=args.seed)
        # Merge domain defaults with resolved values
        return ExperimentConfig(
            title=f"{adapter.name().upper()} {split_label}".strip(),
            domain_tag=f"{adapter.name().replace('-', '_')}_{split_label.lower() or 'run'}",
            tasks=tasks,
            environment=env, grammar=grammar, drive=drive,
            rounds=resolved["rounds"],
            workers=defaults.get("workers", resolved["workers"]),
            seed=args.seed,
            compute_cap=resolved["compute_cap"] or defaults.get("compute_cap", 0),
            energy_beta=defaults.get("energy_beta", 0.001),
            solve_threshold=defaults.get("solve_threshold", 0.001),
            exhaustive_depth=defaults.get("exhaustive_depth", args.exhaustive_depth),
            exhaustive_pair_top_k=defaults.get("exhaustive_pair_top_k", args.exhaustive_pair_top_k),
            exhaustive_triple_top_k=defaults.get("exhaustive_triple_top_k", args.exhaustive_triple_top_k),
            sequential_compounding=defaults.get("sequential_compounding", args.sequential_compounding),
            culture_path=culture_path,
            runs_dir=args.runs_dir,
            no_log=args.no_log,
            batch=getattr(args, "batch", False),
            task_ids=getattr(args, "task_ids", ""),
            mode=args.mode,
            timestamp=timestamp,
            suppress_files=suppress_files,
            split_label=split_label,
            default_cell_size=adapter.default_cell_size(),
        )

    try:
        if args.run_mode == "pipeline":
            run_pipeline(
                make_train_config=lambda a, r, m, tasks, ts: _make_config(
                    a, r, m, tasks, ts, suppress_files=True, split_label="TRAINING"),
                make_eval_config=lambda a, r, m, tasks, cp, ts: _make_config(
                    a, r, m, tasks, ts, culture_path=cp, suppress_files=True,
                    split_label="EVALUATION"),
                load_train_tasks=lambda m: adapter.load_tasks(
                    "training", args.train_data_dir or args.data_dir, m),
                load_eval_tasks=lambda m: adapter.load_tasks(
                    "evaluation", args.eval_data_dir or args.data_dir, m),
                args=args, resolved=resolved, max_tasks=max_tasks,
                pipeline_prefix=f"{adapter.name().replace('-', '_')}_pipeline",
                pipeline_title=f"{adapter.name().upper()} FULL PIPELINE (Train -> Eval)",
                domain_name=adapter.name(),
                try_generate_viz=(None if getattr(args, "batch", False)
                                  else lambda p: adapter.post_run_hooks(
                                      type("R", (), {"results_path": p})())),
            )
        else:
            # Single mode
            split = args.split
            data_dir = args.data_dir
            tasks = adapter.load_tasks(split, data_dir, max_tasks)
            split_label = "TRAINING" if split == "training" else "EVALUATION"
            cfg = _make_config(args, resolved, max_tasks, tasks, "",
                               culture_path=getattr(args, "culture", ""),
                               split_label=split_label)
            result = run_experiment(cfg)
            if not getattr(args, "batch", False):
                for vp in adapter.post_run_hooks(result):
                    print(f"  Visualization: {vp}")
    except KeyboardInterrupt:
        print("\n\nAborted by user.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
