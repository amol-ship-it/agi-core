"""
Diagnostic: Near-miss landscape & size-mismatch correction opportunity analysis.

Runs the current system on ARC-AGI-1 tasks and reports:
  1a. Task categorization by output dimensions (same-shape, smaller, larger, mixed)
  1b. Near-miss landscape: distribution of prediction errors for unsolved tasks
  1c. Size-mismatch correction opportunity: which extraction primitives could help

Usage:
    python -m experiments.diagnostic_near_miss                    # quick (50 tasks)
    python -m experiments.diagnostic_near_miss --mode default     # all 400 training tasks
    python -m experiments.diagnostic_near_miss --max-tasks 100    # custom count
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from core import (
    ExperimentConfig, Task, Program, ScoredProgram,
    run_experiment, resolve_from_preset, PRESETS,
)
from core.learner import Learner
from core.config import SearchConfig
from domains.arc import (
    ARCEnv, ARCGrammar, ARCDrive, load_arc_dataset,
)
from experiments.phase1_arc import find_arc_data, _load_tasks


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TaskDiagnostic:
    task_id: str
    shape_category: str = ""  # same-shape, output-smaller, output-larger, mixed
    solved: bool = False
    best_prediction_error: float = 1.0
    best_program: str = ""
    output_shape_matches: bool = False  # does best program produce right shape?
    identity_correction_tried: bool = False
    identity_correction_result: str = ""  # "solved", "failed", "skipped-shape"
    extraction_opportunities: list[str] = field(default_factory=list)
    train_dims: list[dict] = field(default_factory=list)  # [{inp_shape, out_shape}, ...]


# =============================================================================
# Step 1a: Task categorization by output dimensions
# =============================================================================

def categorize_task_dims(task: Task) -> tuple[str, list[dict]]:
    """Categorize a task by its input/output dimension relationship."""
    categories = []
    dims = []
    for inp, out in task.train_examples:
        inp_arr = np.array(inp, dtype=np.int32)
        out_arr = np.array(out, dtype=np.int32)
        dims.append({
            "inp_shape": list(inp_arr.shape),
            "out_shape": list(out_arr.shape),
        })
        if inp_arr.shape == out_arr.shape:
            categories.append("same-shape")
        elif (out_arr.shape[0] <= inp_arr.shape[0] and
              out_arr.shape[1] <= inp_arr.shape[1]):
            categories.append("output-smaller")
        elif (out_arr.shape[0] >= inp_arr.shape[0] and
              out_arr.shape[1] >= inp_arr.shape[1]):
            categories.append("output-larger")
        else:
            categories.append("mixed-dims")

    unique = set(categories)
    if len(unique) == 1:
        return categories[0], dims
    elif "same-shape" in unique and len(unique) == 2:
        return "mixed", dims
    else:
        return "mixed", dims


# =============================================================================
# Step 1b: Near-miss landscape — run system and capture best results
# =============================================================================

def run_and_collect(tasks: list[Task], mode: str) -> dict[str, ScoredProgram]:
    """Run the full system on tasks and return best ScoredProgram per task."""
    preset = PRESETS[mode]
    # Use default mode's search parameters but override task list
    args_ns = argparse.Namespace(
        mode=mode,
        seed=42,
        rounds=None,
        beam_width=None,
        max_generations=None,
        max_tasks=None,
        workers=0,
        compute_cap=0,
        exhaustive_depth=preset.get("exhaustive_depth", 3),
        exhaustive_pair_top_k=preset.get("exhaustive_pair_top_k", 15),
        exhaustive_triple_top_k=preset.get("exhaustive_triple_top_k", 5),
        sequential_compounding=False,
        runs_dir="runs",
        no_log=True,
        verbose=False,
        task_ids="",
        culture="",
        save_culture="",
        compounding=False,
        adaptive_realloc=False,
    )
    resolved = resolve_from_preset(args_ns, preset)

    env = ARCEnv()
    grammar = ARCGrammar(seed=42)
    drive = ARCDrive()

    cfg = ExperimentConfig(
        title="DIAGNOSTIC RUN",
        domain_tag="diagnostic",
        tasks=tasks,
        environment=env,
        grammar=grammar,
        drive=drive,
        rounds=1,
        beam_width=resolved["beam_width"],
        max_generations=resolved["max_generations"],
        workers=resolved["workers"],
        seed=42,
        compute_cap=resolved["compute_cap"],
        mutations_per_candidate=2,
        crossover_fraction=0.3,
        energy_alpha=1.0,
        energy_beta=0.002,
        solve_threshold=0.001,
        exhaustive_depth=args_ns.exhaustive_depth,
        exhaustive_pair_top_k=args_ns.exhaustive_pair_top_k,
        exhaustive_triple_top_k=args_ns.exhaustive_triple_top_k,
        sequential_compounding=False,
        adaptive_realloc=False,
        min_occurrences=2,
        runs_dir="runs",
        no_log=True,
        verbose=False,
        task_ids="",
        mode=mode,
        suppress_files=True,
    )

    result = run_experiment(cfg)

    # Extract best results per task from the experiment
    best_per_task = {}
    if result.results_data and "tasks" in result.results_data:
        for tid, tdata in result.results_data["tasks"].items():
            prog_str = tdata.get("program", "identity")
            best_per_task[tid] = {
                "solved": tdata.get("solved", False),
                "prediction_error": tdata.get("prediction_error", 1.0),
                "program": prog_str,
                "energy": tdata.get("energy", 1.0),
            }
    return best_per_task


# =============================================================================
# Step 1c: Extraction opportunity analysis
# =============================================================================

# These are the extraction primitives that change grid dimensions
EXTRACTION_PRIMITIVES = [
    "crop_to_nonzero",
    "crop_to_content_border",
    "top_half",
    "bottom_half",
    "left_half",
    "right_half",
    "extract_largest_object",
    "extract_smallest_object",
    "extract_top_left_cell",
    "extract_bottom_right_cell",
    "extract_repeating_tile",
    "extract_top_left_block",
    "extract_bottom_right_block",
    "extract_unique_block",
    "extract_minority_color",
    "extract_majority_color",
]


def check_extraction_opportunities(task: Task, env: ARCEnv) -> list[str]:
    """Check which extraction primitives produce the right output shape.

    For each extraction primitive, apply it to each training input and see
    if the result matches the expected output shape for that example.
    """
    from domains.arc.primitives import _PRIM_MAP

    opportunities = []
    for prim_name in EXTRACTION_PRIMITIVES:
        prim = _PRIM_MAP.get(prim_name)
        if prim is None:
            continue

        all_match = True
        for inp, exp in task.train_examples:
            try:
                result = prim.fn(inp)
                result_arr = np.array(result, dtype=np.int32)
                exp_arr = np.array(exp, dtype=np.int32)
                if result_arr.shape != exp_arr.shape:
                    all_match = False
                    break
            except Exception:
                all_match = False
                break

        if all_match:
            opportunities.append(prim_name)

    return opportunities


def check_extraction_then_correct(task: Task, env: ARCEnv, prim_name: str) -> tuple[bool, float]:
    """Check if extraction + correction can solve this task.

    Apply extraction to get right shape, then try correction pipeline on
    (extracted, expected) pairs.
    Returns (solvable, best_error).
    """
    from domains.arc.primitives import _PRIM_MAP

    prim = _PRIM_MAP.get(prim_name)
    if prim is None:
        return False, 1.0

    extracted_outputs = []
    expected_outputs = []
    for inp, exp in task.train_examples:
        try:
            result = prim.fn(inp)
            extracted_outputs.append(result)
            expected_outputs.append(exp)
        except Exception:
            return False, 1.0

    # Try the correction pipeline
    correction = env.infer_output_correction(
        extracted_outputs, expected_outputs,
        max_rules=100, try_5x5=True)

    if correction is not None:
        # LOOCV: hold out each example and verify
        drive = ARCDrive()
        total_error = 0.0
        for i in range(len(task.train_examples)):
            # Leave out example i
            train_ext = extracted_outputs[:i] + extracted_outputs[i+1:]
            train_exp = expected_outputs[:i] + expected_outputs[i+1:]

            if len(train_ext) < 1:
                continue

            loo_correction = env.infer_output_correction(
                train_ext, train_exp, max_rules=100, try_5x5=True)

            if loo_correction is None:
                return False, 1.0

            # Test on held-out example
            try:
                pred = env.execute(loo_correction, extracted_outputs[i])
                err = drive.prediction_error(pred, expected_outputs[i])
                total_error += err
            except Exception:
                return False, 1.0

        avg_error = total_error / len(task.train_examples)
        return avg_error < 0.001, avg_error

    # Check if extraction alone solves it
    drive = ARCDrive()
    total_error = 0.0
    for ext, exp in zip(extracted_outputs, expected_outputs):
        total_error += drive.prediction_error(ext, exp)
    avg_error = total_error / len(task.train_examples)
    return avg_error < 0.001, avg_error


# =============================================================================
# Main diagnostic
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Near-miss landscape")
    parser.add_argument("--mode", default="quick", choices=["quick", "default", "contest"])
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Override max tasks (0 = use mode default)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip running the system; only do dim analysis")
    parser.add_argument("--results-json", type=str, default=None,
                        help="Use existing results JSON instead of running")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Output directory for CSV and summary")
    args = parser.parse_args()

    preset = PRESETS[args.mode]
    max_tasks = args.max_tasks if args.max_tasks > 0 else preset.get("max_tasks", 0)

    print("=" * 72)
    print("  DIAGNOSTIC: Near-Miss Landscape & Size-Mismatch Analysis")
    print(f"  Mode: {args.mode}  Max tasks: {max_tasks or 'all'}")
    print("=" * 72)
    print()

    # Load tasks
    t0 = time.time()
    tasks = _load_tasks("training", args.data_dir, max_tasks)
    print()

    # =========================================================================
    # Step 1a: Task categorization by dimensions
    # =========================================================================
    print("─" * 72)
    print("  STEP 1a: Task categorization by output dimensions")
    print("─" * 72)

    diagnostics: dict[str, TaskDiagnostic] = {}
    shape_counts: Counter = Counter()

    for task in tasks:
        cat, dims = categorize_task_dims(task)
        diag = TaskDiagnostic(task_id=task.task_id, shape_category=cat, train_dims=dims)
        diagnostics[task.task_id] = diag
        shape_counts[cat] += 1

    total = len(tasks)
    print(f"\n  Total tasks: {total}")
    for cat in ["same-shape", "output-smaller", "output-larger", "mixed"]:
        n = shape_counts.get(cat, 0)
        pct = 100 * n / total if total else 0
        print(f"    {cat:20s}: {n:4d} ({pct:5.1f}%)")

    # =========================================================================
    # Step 1b: Run system and capture near-miss landscape
    # =========================================================================
    results_per_task = {}
    solved_count = 0
    unsolved_errors = []

    if args.results_json:
        print(f"\n  Loading existing results from {args.results_json}...")
        with open(args.results_json) as f:
            rdata = json.load(f)
        for tid, tdata in rdata.get("tasks", {}).items():
            results_per_task[tid] = {
                "solved": tdata.get("solved", False),
                "prediction_error": tdata.get("prediction_error", 1.0),
                "program": tdata.get("program", "identity"),
                "energy": tdata.get("energy", 1.0),
            }
    elif not args.skip_run:
        print(f"\n  Running system on {len(tasks)} tasks...")
        results_per_task = run_and_collect(tasks, args.mode)

    if results_per_task:
        print("\n" + "─" * 72)
        print("  STEP 1b: Near-miss landscape")
        print("─" * 72)

        solved_count = 0
        unsolved_errors = []
        unsolved_by_cat: dict[str, list[float]] = defaultdict(list)

        for tid, diag in diagnostics.items():
            if tid in results_per_task:
                rdata = results_per_task[tid]
                diag.solved = rdata["solved"]
                diag.best_prediction_error = rdata["prediction_error"]
                diag.best_program = rdata["program"]
                if diag.solved:
                    solved_count += 1
                else:
                    unsolved_errors.append(rdata["prediction_error"])
                    unsolved_by_cat[diag.shape_category].append(rdata["prediction_error"])

        n_unsolved = len(unsolved_errors)
        print(f"\n  Solved: {solved_count}/{total} ({100*solved_count/total:.1f}%)")
        print(f"  Unsolved: {n_unsolved}")

        # Solved breakdown by dimension category
        print(f"\n  Solve rate by dimension category:")
        for cat in ["same-shape", "output-smaller", "output-larger", "mixed"]:
            cat_total = shape_counts.get(cat, 0)
            cat_solved = sum(1 for d in diagnostics.values()
                           if d.shape_category == cat and d.solved)
            rate = 100 * cat_solved / cat_total if cat_total else 0
            print(f"    {cat:20s}: {cat_solved:3d}/{cat_total:3d} ({rate:5.1f}%)")

        if unsolved_errors:
            # Near-miss distribution
            unsolved_errors.sort()
            thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
            print(f"\n  Near-miss distribution (unsolved tasks):")
            for thresh in thresholds:
                n = sum(1 for e in unsolved_errors if e <= thresh)
                print(f"    error ≤ {thresh:.2f}: {n:4d} ({100*n/n_unsolved:.1f}%)")

            print(f"\n  Near-miss distribution by dimension category:")
            for cat in ["same-shape", "output-smaller", "output-larger", "mixed"]:
                errors = unsolved_by_cat.get(cat, [])
                if not errors:
                    continue
                n10 = sum(1 for e in errors if e <= 0.10)
                n20 = sum(1 for e in errors if e <= 0.20)
                n30 = sum(1 for e in errors if e <= 0.30)
                print(f"    {cat:20s}: total={len(errors):3d}  "
                      f"≤10%={n10:3d}  ≤20%={n20:3d}  ≤30%={n30:3d}")

            # Check shape match for best programs on unsolved tasks
            # Only possible for simple (depth-1) programs we can reconstruct
            env = ARCEnv()
            shape_match_count = 0
            shape_mismatch_count = 0
            shape_unknown_count = 0
            for tid, diag in diagnostics.items():
                if diag.solved or tid not in results_per_task:
                    continue
                task = next((t for t in tasks if t.task_id == tid), None)
                if task is None:
                    continue

                # Try to reconstruct and execute the best program
                # Only works for depth-1 programs (simple primitive name)
                prog_str = diag.best_program
                if not prog_str or "(" in prog_str:
                    # Composed program — can't easily reconstruct
                    shape_unknown_count += 1
                    continue

                all_shapes_match = True
                try:
                    prog = Program(root=prog_str)
                    for inp, exp in task.train_examples:
                        out = env.execute(prog, inp)
                        out_arr = np.array(out, dtype=np.int32)
                        exp_arr = np.array(exp, dtype=np.int32)
                        if out_arr.shape != exp_arr.shape:
                            all_shapes_match = False
                            break
                except Exception:
                    shape_unknown_count += 1
                    continue

                diag.output_shape_matches = all_shapes_match
                if all_shapes_match:
                    shape_match_count += 1
                else:
                    shape_mismatch_count += 1

            print(f"\n  Best program shape analysis (unsolved tasks):")
            print(f"    Output shape matches expected: {shape_match_count}")
            print(f"    Output shape mismatches:       {shape_mismatch_count}")
            print(f"    Unknown (composed programs):   {shape_unknown_count}")

    # =========================================================================
    # Step 1c: Size-mismatch correction opportunity
    # =========================================================================
    print("\n" + "─" * 72)
    print("  STEP 1c: Extraction primitive opportunity analysis")
    print("─" * 72)

    env = ARCEnv()
    extraction_hits: Counter = Counter()
    tasks_with_opportunity: list[str] = []
    solvable_with_extraction: list[tuple[str, str, float]] = []  # (task_id, prim, error)

    # Focus on unsolved non-same-shape tasks (where extraction could help)
    target_tasks = [
        t for t in tasks
        if diagnostics[t.task_id].shape_category != "same-shape"
        and (not results_per_task or not diagnostics[t.task_id].solved)
    ]

    print(f"\n  Checking {len(target_tasks)} unsolved non-same-shape tasks...")

    for i, task in enumerate(target_tasks):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(target_tasks)}")

        opportunities = check_extraction_opportunities(task, env)
        diagnostics[task.task_id].extraction_opportunities = opportunities

        if opportunities:
            tasks_with_opportunity.append(task.task_id)
            for prim_name in opportunities:
                extraction_hits[prim_name] += 1

            # Check if extraction + correction can solve
            for prim_name in opportunities[:5]:  # limit to top 5
                solvable, error = check_extraction_then_correct(task, env, prim_name)
                if solvable:
                    solvable_with_extraction.append((task.task_id, prim_name, error))
                    break  # found a solution, move on

    print(f"\n  Tasks with shape-matching extraction primitive: {len(tasks_with_opportunity)}")
    print(f"  Tasks solvable by extraction + correction:     {len(solvable_with_extraction)}")

    if extraction_hits:
        print(f"\n  Extraction primitive hit counts (shape match):")
        for prim, count in extraction_hits.most_common():
            print(f"    {prim:35s}: {count:3d}")

    if solvable_with_extraction:
        print(f"\n  Solvable tasks (extraction + correction):")
        for tid, prim, err in solvable_with_extraction[:20]:
            print(f"    {tid:40s} via {prim} (err={err:.4f})")

    # Also check unsolved same-shape tasks where identity correction didn't help
    same_shape_unsolved = [
        t for t in tasks
        if diagnostics[t.task_id].shape_category == "same-shape"
        and (not results_per_task or not diagnostics[t.task_id].solved)
    ]
    if same_shape_unsolved:
        print(f"\n  Unsolved same-shape tasks: {len(same_shape_unsolved)}")
        print(f"  (Identity correction tried but failed on these)")

    # =========================================================================
    # Summary & Output
    # =========================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Tasks analyzed: {total}")

    if results_per_task:
        print(f"  Solved: {solved_count}/{total}")

    print(f"\n  Dimension categories:")
    for cat in ["same-shape", "output-smaller", "output-larger", "mixed"]:
        n = shape_counts.get(cat, 0)
        print(f"    {cat:20s}: {n:4d}")

    print(f"\n  Extraction correction opportunity:")
    print(f"    Tasks with shape-matching extraction: {len(tasks_with_opportunity)}")
    print(f"    Tasks solvable by extraction+correct: {len(solvable_with_extraction)}")

    # Key recommendation
    if len(solvable_with_extraction) >= 20:
        print(f"\n  ✓ RECOMMENDATION: Proceed with Phase 2 (size-adaptive correction)")
        print(f"    {len(solvable_with_extraction)} tasks can benefit → worth implementing")
    elif len(solvable_with_extraction) >= 5:
        print(f"\n  ~ RECOMMENDATION: Phase 2 marginal ({len(solvable_with_extraction)} tasks)")
        print(f"    Consider implementing but don't expect large gains")
    else:
        print(f"\n  ✗ RECOMMENDATION: Skip Phase 2 ({len(solvable_with_extraction)} tasks)")
        print(f"    Too few tasks benefit — focus elsewhere")

    # =========================================================================
    # Save CSV for audit
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "diagnostic_near_miss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "shape_category", "solved", "best_error",
            "best_program", "output_shape_matches",
            "extraction_opportunities", "solvable_by_extraction",
        ])
        for tid, diag in sorted(diagnostics.items()):
            solvable = any(tid == s[0] for s in solvable_with_extraction)
            writer.writerow([
                tid,
                diag.shape_category,
                diag.solved,
                f"{diag.best_prediction_error:.4f}",
                diag.best_program[:80] if diag.best_program else "",
                diag.output_shape_matches,
                "|".join(diag.extraction_opportunities),
                solvable,
            ])

    print(f"\n  Per-task CSV: {csv_path}")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "diagnostic_near_miss_summary.json")
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "total_tasks": total,
        "solved": solved_count if results_per_task else None,
        "shape_categories": dict(shape_counts),
        "near_miss_thresholds": {},
        "extraction_opportunity": {
            "tasks_with_shape_match": len(tasks_with_opportunity),
            "tasks_solvable": len(solvable_with_extraction),
            "solvable_tasks": [
                {"task_id": tid, "primitive": prim, "error": err}
                for tid, prim, err in solvable_with_extraction
            ],
            "primitive_hit_counts": dict(extraction_hits.most_common()),
        },
    }
    if unsolved_errors:
        for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            summary["near_miss_thresholds"][str(thresh)] = sum(
                1 for e in unsolved_errors if e <= thresh)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {summary_path}")
    print()


if __name__ == "__main__":
    main()
