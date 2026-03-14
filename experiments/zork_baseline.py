"""
Zork Domain: Baseline Experiment.

Tests the core loop on a stateful, sequential domain (text adventure).
Programs compose as action sequences: go_north(take_lamp(state)).

Usage:
    python -m experiments.zork_baseline
    python -m experiments.zork_baseline --mode quick
"""

from __future__ import annotations

from common.benchmark import (
    ExperimentConfig, run_experiment, make_parser,
    resolve_from_preset, PRESETS,
)
from domains.zork.adapter import ZorkAdapter


_adapter = ZorkAdapter()


def main():
    parser = make_parser(
        description="Zork Domain: Baseline Performance",
        domain_name="zork_baseline",
    )
    args = parser.parse_args()

    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)

    tasks = _adapter.load_tasks("training")
    print(f"\n  Zork Baseline Experiment")
    print(f"  Tasks: {len(tasks)}")

    defaults = _adapter.config_defaults()
    env, grammar, drive = _adapter.create_interfaces(seed=args.seed)

    config = ExperimentConfig(
        title="Zork Baseline",
        domain_tag="zork",
        tasks=tasks,
        environment=env, grammar=grammar, drive=drive,
        rounds=resolved["rounds"],
        beam_width=resolved["beam_width"],
        max_generations=resolved["max_generations"],
        workers=defaults.get("workers", 1),
        seed=args.seed,
        compute_cap=defaults.get("compute_cap", 0),
        mutations_per_candidate=2,
        crossover_fraction=0.3,
        energy_alpha=1.0,
        energy_beta=0.001,
        solve_threshold=0.001,
        exhaustive_depth=args.exhaustive_depth if args.exhaustive_depth is not None else defaults.get("exhaustive_depth", 2),
        exhaustive_pair_top_k=defaults.get("exhaustive_pair_top_k", 30),
        exhaustive_triple_top_k=15,
        sequential_compounding=getattr(args, 'sequential_compounding', False),
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        mode=args.mode,
        default_cell_size=_adapter.default_cell_size(),
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
