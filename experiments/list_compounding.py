"""
List Operations: Compounding Validation Experiment.

Tests the core hypothesis: does library learning (sleep) improve
performance on harder tasks (wake)?

Protocol:
  Round 1: Wake on all 28 tasks -> sleep -> extract library.
           Level 1 (depth 1) should mostly solve. Level 2 (depth 2)
           should partially solve. Level 3 (depth 3) is unlikely
           without library help.

  Round 2+: Wake again with library -> sleep -> iterate.
           Library entries from solved tasks become 0-arity primitives.
           A depth-2 program using a library entry = depth-3+ effective
           depth. Level 3 solve rate should increase each round.

Success = Level 3 solve rate increases across rounds (compounding).

Usage:
    python -m experiments.list_compounding
    python -m experiments.list_compounding --mode quick
    python -m experiments.list_compounding --rounds 5
"""

from __future__ import annotations

from common.benchmark import (
    ExperimentConfig, run_experiment, make_parser,
    resolve_from_preset, PRESETS,
)
from domains.list_ops.adapter import ListOpsAdapter


_adapter = ListOpsAdapter()


def main():
    parser = make_parser(
        description="List Operations: Compounding Validation",
        domain_name="list_compounding",
    )
    parser.add_argument("--seed-tasks", type=int, default=42,
                        help="Seed for task generation")
    args = parser.parse_args()

    # Resolve preset + overrides
    preset = PRESETS[args.mode]
    resolved = resolve_from_preset(args, preset)

    # Override defaults for list domain (smaller search space)
    rounds = resolved["rounds"]
    if rounds < 3 and args.mode != "quick":
        rounds = 3  # need multiple rounds to test compounding

    from domains.list_ops import get_sample_tasks
    tasks = get_sample_tasks(seed=args.seed_tasks)
    print(f"\n  List Operations Compounding Experiment")
    print(f"  Tasks: {len(tasks)} ({sum(1 for t in tasks if t.difficulty == 1.0)} L1, "
          f"{sum(1 for t in tasks if t.difficulty == 2.0)} L2, "
          f"{sum(1 for t in tasks if t.difficulty == 3.0)} L3)")
    print(f"  Rounds: {rounds}")
    print(f"  Sequential compounding: {args.sequential_compounding or True}")

    defaults = _adapter.config_defaults()
    env, grammar, drive = _adapter.create_interfaces(seed=args.seed)

    config = ExperimentConfig(
        title="List Ops Compounding",
        domain_tag="list_ops",
        tasks=tasks,
        environment=env, grammar=grammar, drive=drive,
        rounds=rounds,
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
        exhaustive_depth=defaults.get("exhaustive_depth", 2),
        exhaustive_pair_top_k=defaults.get("exhaustive_pair_top_k", 22),
        exhaustive_triple_top_k=15,
        sequential_compounding=True,  # always compound for this experiment
        runs_dir=args.runs_dir,
        no_log=args.no_log,
        mode=args.mode,
        default_cell_size=_adapter.default_cell_size(),
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
