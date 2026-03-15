#!/usr/bin/env python3
"""Analyze failure patterns in ARC-AGI benchmark results.

Reads a pipeline JSONL file and categorizes unsolved tasks by:
- Dimension change type (same, shrink, grow, variable)
- Best error achieved (near-miss, medium, distant)
- Object count in inputs
- Near-miss programs (what almost worked)

Usage:
    python scripts/analyze_failures.py [JSONL_FILE]

If no file given, uses the most recent pipeline JSONL in runs/.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_results(jsonl_path: str) -> list[dict]:
    """Load pipeline results from JSONL file."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_task_data(task_id: str) -> dict | None:
    """Load raw ARC task data to analyze input/output properties."""
    for subdir in ["training", "evaluation"]:
        path = Path(f"data/ARC-AGI/data/{subdir}/{task_id}.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def classify_dims(task_data: dict) -> str:
    """Classify dimension change pattern."""
    patterns = set()
    for ex in task_data.get("train", []):
        ih, iw = len(ex["input"]), len(ex["input"][0]) if ex["input"] else 0
        oh, ow = len(ex["output"]), len(ex["output"][0]) if ex["output"] else 0
        if ih == oh and iw == ow:
            patterns.add("same")
        elif oh < ih or ow < iw:
            if oh > ih or ow > iw:
                patterns.add("mixed")
            else:
                patterns.add("shrink")
        else:
            patterns.add("grow")
    if len(patterns) > 1:
        return "variable"
    return patterns.pop() if patterns else "unknown"


def count_objects(grid: list[list[int]]) -> int:
    """Count connected foreground components (4-connected)."""
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    visited = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and (r, c) not in visited:
                count += 1
                stack = [(r, c)]
                color = grid[r][c]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] != color:
                        continue
                    visited.add((cr, cc))
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
    return count


def classify_object_count(task_data: dict) -> str:
    """Classify task by typical object count."""
    counts = []
    for ex in task_data.get("train", []):
        counts.append(count_objects(ex["input"]))
    avg = sum(counts) / len(counts) if counts else 0
    if avg == 0:
        return "no-objects"
    elif avg <= 1.5:
        return "single"
    elif avg <= 5:
        return "few (2-5)"
    else:
        return "many (6+)"


def has_separators(grid: list[list[int]]) -> bool:
    """Check if grid has separator lines."""
    h, w = len(grid), len(grid[0]) if grid else 0
    # Horizontal separators
    for r in range(h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return True
    # Vertical separators
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] != 0:
            return True
    return False


def main():
    # Find JSONL file
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        runs_dir = Path("runs")
        jsonl_files = sorted(runs_dir.glob("arc_agi_1_pipeline_*.jsonl"),
                             key=lambda p: p.stat().st_size, reverse=True)
        if not jsonl_files:
            print("No pipeline JSONL files found in runs/")
            sys.exit(1)
        jsonl_path = str(jsonl_files[0])
        print(f"Using: {jsonl_path}")

    results = load_results(jsonl_path)

    # Filter to training phase, round 1 (most complete)
    train_r1 = [r for r in results if r.get("phase") == "train" and r.get("round", 1) == 1]
    if not train_r1:
        train_r1 = [r for r in results if r.get("phase") == "train"]

    solved = [r for r in train_r1 if r.get("train_solved")]
    unsolved = [r for r in train_r1 if not r.get("train_solved")]

    print(f"\n{'='*70}")
    print(f"  FAILURE ANALYSIS — {len(solved)}/{len(train_r1)} solved")
    print(f"{'='*70}\n")

    # --- Categorize by dimension change ---
    dim_counts = Counter()
    dim_best_errors = defaultdict(list)

    # --- Categorize by error level ---
    near_miss = []   # < 0.3
    medium = []      # 0.3 - 0.7
    distant = []     # > 0.7

    # --- Categorize by object count ---
    obj_counts = Counter()

    # --- Track near-miss programs ---
    near_miss_programs = Counter()

    # --- Track separator tasks ---
    separator_unsolved = []

    for r in unsolved:
        task_id = r["task_id"]
        error = r.get("prediction_error", 999)
        program = r.get("program", "none")

        task_data = load_task_data(task_id)
        if task_data is None:
            continue

        # Dimension classification
        dim_type = classify_dims(task_data)
        dim_counts[dim_type] += 1
        dim_best_errors[dim_type].append(error)

        # Error level
        if error < 0.3:
            near_miss.append((task_id, error, program))
        elif error < 0.7:
            medium.append((task_id, error, program))
        else:
            distant.append((task_id, error, program))

        # Object count
        obj_type = classify_object_count(task_data)
        obj_counts[obj_type] += 1

        # Near-miss programs
        if error < 0.5:
            near_miss_programs[program] += 1

        # Separator detection
        if any(has_separators(ex["input"]) for ex in task_data.get("train", [])):
            separator_unsolved.append((task_id, error, program))

    # --- Print results ---
    print("  BY DIMENSION CHANGE:")
    for dim_type, count in sorted(dim_counts.items(), key=lambda x: -x[1]):
        errors = dim_best_errors[dim_type]
        avg_err = sum(errors) / len(errors) if errors else 0
        nm_count = sum(1 for e in errors if e < 0.3)
        print(f"    {dim_type:12s}: {count:3d} unsolved  (avg err={avg_err:.3f}, {nm_count} near-miss)")

    print(f"\n  BY ERROR LEVEL:")
    print(f"    Near-miss (<0.3):  {len(near_miss):3d}  — these are close to solved!")
    print(f"    Medium (0.3-0.7):  {len(medium):3d}")
    print(f"    Distant (>0.7):    {len(distant):3d}")

    print(f"\n  BY OBJECT COUNT:")
    for obj_type, count in sorted(obj_counts.items(), key=lambda x: -x[1]):
        print(f"    {obj_type:15s}: {count:3d}")

    print(f"\n  SEPARATOR TASKS (unsolved): {len(separator_unsolved)}")
    for tid, err, prog in sorted(separator_unsolved, key=lambda x: x[1])[:10]:
        print(f"    {tid}: err={err:.4f}  prog={prog}")

    print(f"\n  TOP NEAR-MISS PROGRAMS (error < 0.5):")
    for prog, count in near_miss_programs.most_common(15):
        print(f"    {count:3d}x  {prog}")

    print(f"\n  CLOSEST UNSOLVED TASKS (near-miss, error < 0.3):")
    for tid, err, prog in sorted(near_miss, key=lambda x: x[1])[:20]:
        print(f"    {tid}: err={err:.4f}  prog={prog}")

    print(f"\n{'='*70}")
    print(f"  KEY INSIGHTS")
    print(f"{'='*70}")

    total_unsolved = len(unsolved)
    if total_unsolved > 0:
        same_pct = dim_counts.get("same", 0) / total_unsolved * 100
        shrink_pct = dim_counts.get("shrink", 0) / total_unsolved * 100
        grow_pct = dim_counts.get("grow", 0) / total_unsolved * 100
        nm_pct = len(near_miss) / total_unsolved * 100
        print(f"    Same-dims unsolved: {dim_counts.get('same', 0)} ({same_pct:.0f}%) — target for per-object/row/col strategies")
        print(f"    Shrink unsolved: {dim_counts.get('shrink', 0)} ({shrink_pct:.0f}%) — target for crop/extract strategies")
        print(f"    Grow unsolved: {dim_counts.get('grow', 0)} ({grow_pct:.0f}%) — target for scale/tile/stamp strategies")
        print(f"    Near-miss: {len(near_miss)} ({nm_pct:.0f}%) — color fix or small adjustments could solve these")
        print(f"    Separator tasks: {len(separator_unsolved)} — cell algebra could help")


if __name__ == "__main__":
    main()
