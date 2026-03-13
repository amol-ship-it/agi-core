"""
Research script: Analyze near-miss residuals to understand what patterns exist.

Hypothesis: Near-miss programs (error < 10%) have learnable residual patterns
(consistent color changes, spatial rules) that can be synthesized into
corrective primitives.

This script:
1. Runs quick benchmark to find near-misses
2. For each near-miss, computes predicted vs expected diff
3. Categorizes the residual patterns
4. Reports which patterns are learnable

Usage:
    python -m experiments.analyze_residuals [--max-tasks 50]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    Learner, InMemoryStore, SearchConfig, SleepConfig, CurriculumConfig,
    Program, Task,
)
from domains.arc.environment import ARCEnv
from domains.arc.grammar import ARCGrammar
from domains.arc.drive import ARCDrive
from domains.arc.dataset import load_arc_dataset


def find_arc_data():
    """Find ARC-AGI-1 training data."""
    candidates = [
        "data/ARC-AGI/data/training",
        "data/ARC-AGI-1/data/training",
        os.path.expanduser("~/github/ARC-AGI/data/training"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def analyze_residual(predicted, expected):
    """Analyze the pixel-level diff between predicted and expected grids.

    Returns a dict describing the residual pattern.
    """
    pred = np.array(predicted)
    exp = np.array(expected)

    # Shape mismatch — can't analyze pixel-level
    if pred.shape != exp.shape:
        return {
            "type": "shape_mismatch",
            "pred_shape": pred.shape,
            "exp_shape": exp.shape,
        }

    # Find differing pixels
    diff_mask = pred != exp
    n_diff = diff_mask.sum()
    n_total = pred.size

    if n_diff == 0:
        return {"type": "perfect", "n_diff": 0}

    # Collect color transitions at differing pixels
    transitions = Counter()
    for r in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            if diff_mask[r, c]:
                transitions[(int(pred[r, c]), int(exp[r, c]))] += 1

    # Check if residual is a consistent color remap
    # For each source color, check if it always maps to the same target
    src_to_targets = defaultdict(Counter)
    for (src, tgt), count in transitions.items():
        src_to_targets[src][tgt] += count

    consistent_remaps = {}
    inconsistent_remaps = {}
    for src, targets in src_to_targets.items():
        total = sum(targets.values())
        best_tgt, best_count = targets.most_common(1)[0]
        consistency = best_count / total
        if consistency >= 0.8:
            consistent_remaps[src] = (best_tgt, consistency, total)
        else:
            inconsistent_remaps[src] = dict(targets)

    # Check spatial patterns in the diff
    diff_positions = list(zip(*np.where(diff_mask)))

    # Are diffs clustered or scattered?
    if len(diff_positions) >= 2:
        rows = [p[0] for p in diff_positions]
        cols = [p[1] for p in diff_positions]
        row_span = max(rows) - min(rows) + 1
        col_span = max(cols) - min(cols) + 1
        bbox_area = row_span * col_span
        density = n_diff / bbox_area if bbox_area > 0 else 0
    else:
        density = 1.0
        bbox_area = n_diff

    # Classify
    if len(consistent_remaps) > 0 and len(inconsistent_remaps) == 0:
        remap_type = "consistent_color_remap"
    elif len(consistent_remaps) > len(inconsistent_remaps):
        remap_type = "mostly_consistent_remap"
    else:
        remap_type = "complex_spatial"

    return {
        "type": remap_type,
        "n_diff": int(n_diff),
        "n_total": int(n_total),
        "error_rate": round(n_diff / n_total, 4),
        "consistent_remaps": {str(k): v for k, v in consistent_remaps.items()},
        "inconsistent_remaps": {str(k): v for k, v in inconsistent_remaps.items()},
        "n_transitions": len(transitions),
        "density": round(density, 3),
        "bbox_area": int(bbox_area),
    }


def analyze_cross_example_consistency(residuals):
    """Check if residual pattern is consistent across training examples.

    This is key: a learnable correction must be consistent across ALL
    training examples, not just one.
    """
    if not residuals:
        return {"consistent": False, "reason": "no residuals"}

    # Filter to same-shape residuals with consistent remaps
    remap_residuals = [r for r in residuals
                       if r["type"] in ("consistent_color_remap", "mostly_consistent_remap")]

    if not remap_residuals:
        return {"consistent": False, "reason": "no consistent remaps found"}

    # Check if the same remap appears in all examples
    all_remaps = []
    for r in remap_residuals:
        remap = {}
        for src_str, (tgt, cons, count) in r["consistent_remaps"].items():
            remap[int(src_str)] = tgt
        all_remaps.append(remap)

    if len(all_remaps) < 2:
        return {
            "consistent": True,  # Only 1 example — can't verify cross-example
            "reason": "single_example",
            "remap": all_remaps[0] if all_remaps else {},
        }

    # Find intersection of remaps across all examples
    common_remap = dict(all_remaps[0])
    for remap in all_remaps[1:]:
        keys_to_remove = []
        for src, tgt in common_remap.items():
            if src not in remap or remap[src] != tgt:
                keys_to_remove.append(src)
        for k in keys_to_remove:
            del common_remap[k]

    if common_remap:
        return {
            "consistent": True,
            "reason": "cross_example_consistent",
            "remap": common_remap,
            "n_examples": len(all_remaps),
        }
    else:
        return {
            "consistent": False,
            "reason": "remaps_differ_across_examples",
            "per_example": all_remaps,
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze near-miss residuals")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--error-threshold", type=float, default=0.15,
                        help="Max error to consider as near-miss")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Find data
    data_dir = find_arc_data()
    if not data_dir:
        print("ERROR: ARC data not found. Clone it first.")
        sys.exit(1)

    print(f"Loading tasks from {data_dir}...")
    tasks = load_arc_dataset(data_dir)

    # Shuffle deterministically and limit
    import random
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    print(f"Running {len(tasks)} tasks to find near-misses...")

    # Set up learner
    env = ARCEnv()
    grammar = ARCGrammar()
    drive = ARCDrive()
    memory = InMemoryStore()

    learner = Learner(
        environment=env, grammar=grammar, drive=drive, memory=memory,
        search_config=SearchConfig(
            beam_width=1, max_generations=1,
            exhaustive_depth=3,
            exhaustive_pair_top_k=40,
            exhaustive_triple_top_k=15,
            eval_budget=3750,  # 3M / 800 cells
            seed=args.seed,
        ),
        sleep_config=SleepConfig(),
    )

    # Run wake on each task, collect near-misses
    near_misses = []
    t0 = time.time()

    for i, task in enumerate(tasks):
        wr = learner.wake_on_task(task)

        if wr.best and wr.best.prediction_error > 0 and wr.best.prediction_error < args.error_threshold:
            # Found a near-miss — analyze the residual
            program = wr.best.program

            # Execute on each training example
            residuals = []
            for inp, expected_out in task.train_examples:
                try:
                    predicted = env.execute(program, inp)
                    residual = analyze_residual(predicted, expected_out)
                    residuals.append(residual)
                except Exception:
                    residuals.append({"type": "execution_error"})

            # Check cross-example consistency
            consistency = analyze_cross_example_consistency(residuals)

            near_misses.append({
                "task_id": task.task_id,
                "program": repr(program),
                "error": round(wr.best.prediction_error, 4),
                "solved": wr.solved,
                "residuals": residuals,
                "cross_example": consistency,
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tasks)}] near_misses={len(near_misses)}  "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"NEAR-MISS RESIDUAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Tasks analyzed: {len(tasks)}")
    print(f"Near-misses (error < {args.error_threshold}): {len(near_misses)}")

    # Categorize
    type_counts = Counter()
    learnable = []

    for nm in near_misses:
        # Overall type from first residual
        types = [r["type"] for r in nm["residuals"]]
        dominant = Counter(types).most_common(1)[0][0] if types else "unknown"
        type_counts[dominant] += 1

        if nm["cross_example"]["consistent"]:
            learnable.append(nm)

    print(f"\nResidual type distribution:")
    for t, count in type_counts.most_common():
        print(f"  {t}: {count}")

    print(f"\nLearnable (cross-example consistent): {len(learnable)}/{len(near_misses)}")

    print(f"\n{'─'*60}")
    print(f"LEARNABLE NEAR-MISSES (sorted by error)")
    print(f"{'─'*60}")

    for nm in sorted(learnable, key=lambda x: x["error"]):
        remap = nm["cross_example"].get("remap", {})
        remap_str = ", ".join(f"{k}→{v}" for k, v in remap.items()) if remap else "none"
        print(f"  {nm['task_id']:<20s} err={nm['error']:.4f}  "
              f"program={nm['program']}")
        print(f"    remap: {remap_str}")
        print(f"    consistency: {nm['cross_example']['reason']}")

    print(f"\n{'─'*60}")
    print(f"NON-LEARNABLE NEAR-MISSES")
    print(f"{'─'*60}")

    non_learnable = [nm for nm in near_misses if not nm["cross_example"]["consistent"]]
    for nm in sorted(non_learnable, key=lambda x: x["error"])[:20]:
        types = [r["type"] for r in nm["residuals"]]
        print(f"  {nm['task_id']:<20s} err={nm['error']:.4f}  "
              f"types={Counter(types).most_common()}")
        print(f"    program={nm['program']}")
        print(f"    reason: {nm['cross_example']['reason']}")

    # Save full analysis
    out_path = "runs/residual_analysis.json"
    os.makedirs("runs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"max_tasks": args.max_tasks, "error_threshold": args.error_threshold},
            "summary": {
                "tasks_analyzed": len(tasks),
                "near_misses": len(near_misses),
                "learnable": len(learnable),
                "type_distribution": dict(type_counts),
            },
            "near_misses": near_misses,
        }, f, indent=2)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
