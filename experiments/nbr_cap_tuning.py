"""
Quick experiment: neighborhood fix rule cap tuning.

Tests different max_rules caps for _infer_neighborhood_correction
to find the optimal tradeoff between train solves and eval generalization.

Usage:
    python -m experiments.nbr_cap_tuning
"""

import json
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
from domains.arc import ARCEnv, ARCGrammar, ARCDrive, load_arc_dataset
from experiments.phase1_arc import find_arc_data


def main():
    caps_to_test = [20, 30, 40, 50, 75, 100]
    results = {}

    data_dir = find_arc_data("training")
    eval_dir = find_arc_data("evaluation")
    if not data_dir or not eval_dir:
        print("ERROR: ARC dataset not found")
        sys.exit(1)

    for cap in caps_to_test:
        print(f"\n{'='*60}")
        print(f"  Testing neighborhood fix rule cap = {cap}")
        print(f"{'='*60}\n")

        # Monkey-patch the default max_rules for this run
        original_infer = ARCEnv.infer_output_correction

        def patched_infer(self, program_outputs, expected_outputs,
                          max_rules=cap, **kwargs):
            return original_infer(self, program_outputs, expected_outputs,
                                  max_rules=max_rules)

        ARCEnv.infer_output_correction = patched_infer

        # Run quick mode (50 tasks each)
        train_tasks = load_arc_dataset(data_dir, max_tasks=50)
        eval_tasks = load_arc_dataset(eval_dir, max_tasks=50)

        # Train
        train_cfg = ExperimentConfig(
            title=f"Cap Tuning (cap={cap}) TRAIN",
            domain_tag=f"cap{cap}_train",
            tasks=train_tasks,
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            rounds=1,
            beam_width=1,
            max_generations=1,
            workers=0,
            seed=42,
            compute_cap=500_000,
            energy_beta=0.002,
            solve_threshold=0.001,
            exhaustive_depth=3,
            exhaustive_pair_top_k=40,
            exhaustive_triple_top_k=15,
            runs_dir="runs",
            no_log=True,
            verbose=False,
            mode="quick",
            suppress_files=True,
        )
        train_result = run_experiment(train_cfg)

        # Eval with culture
        eval_cfg = ExperimentConfig(
            title=f"Cap Tuning (cap={cap}) EVAL",
            domain_tag=f"cap{cap}_eval",
            tasks=eval_tasks,
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            rounds=1,
            beam_width=1,
            max_generations=1,
            workers=0,
            seed=42,
            compute_cap=500_000,
            energy_beta=0.002,
            solve_threshold=0.001,
            exhaustive_depth=3,
            exhaustive_pair_top_k=40,
            exhaustive_triple_top_k=15,
            culture_path=train_result.culture_path,
            runs_dir="runs",
            no_log=True,
            verbose=False,
            mode="quick",
            suppress_files=True,
        )
        eval_result = run_experiment(eval_cfg)

        # Restore original
        ARCEnv.infer_output_correction = original_infer

        train_data = train_result.results_data.get("summary", {})
        eval_data = eval_result.results_data.get("summary", {})

        t_solved = train_data.get("last_round_solved", 0)
        t_train_solved = train_data.get("last_round_train_solved", 0)
        e_solved = eval_data.get("last_round_solved", 0)
        e_train_solved = eval_data.get("last_round_train_solved", 0)

        results[cap] = {
            "train_solved": t_solved,
            "train_train_solved": t_train_solved,
            "eval_solved": e_solved,
            "eval_train_solved": e_train_solved,
            "train_overfit": t_train_solved - t_solved,
            "eval_overfit": e_train_solved - e_solved,
        }

        print(f"\n  Cap {cap}: Train {t_solved}/50 (overfit {t_train_solved - t_solved}), "
              f"Eval {e_solved}/50 (overfit {e_train_solved - e_solved})")

    # Summary
    print("\n" + "=" * 72)
    print("  NEIGHBORHOOD FIX RULE CAP TUNING RESULTS")
    print("=" * 72)
    print(f"\n  {'Cap':>4}  {'Train':>6}  {'T-Overfit':>9}  {'Eval':>5}  {'E-Overfit':>9}  {'Total':>5}  {'Gen Rate':>8}")
    print(f"  {'---':>4}  {'-----':>6}  {'---------':>9}  {'----':>5}  {'---------':>9}  {'-----':>5}  {'--------':>8}")
    for cap in caps_to_test:
        r = results[cap]
        total = r["train_solved"] + r["eval_solved"]
        gen_rate = r["eval_solved"] / max(r["train_solved"], 1)
        print(f"  {cap:4d}  {r['train_solved']:5d}/50  {r['train_overfit']:9d}  "
              f"{r['eval_solved']:4d}/50  {r['eval_overfit']:9d}  {total:4d}/100  {gen_rate:7.1%}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"runs/nbr_cap_tuning_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
