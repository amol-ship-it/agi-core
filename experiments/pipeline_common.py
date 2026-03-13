"""
Shared pipeline utilities for train→eval experiment scripts.

Extracts common logic: JSONL/JSON writing, summary printing, artifact naming,
and pipeline-level log teeing.
Used by phase1_arc.py and phase2_arc.py.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

from core import ExperimentResult, fmt_duration
from core.runner import TeeWriter


@contextmanager
def pipeline_tee(log_path: str):
    """Context manager that tees stdout to a pipeline log file."""
    tee = TeeWriter(log_path, sys.stdout)
    sys.stdout = tee
    try:
        yield log_path
    finally:
        sys.stdout = tee._original
        tee.close()


def save_pipeline_results(
    train_result: ExperimentResult,
    eval_result: ExperimentResult,
    *,
    prefix: str,
    runs_dir: str,
    title: str,
    domain: str,
    args,
    resolved: dict,
    extra_meta: dict | None = None,
) -> tuple[str, str]:
    """Save combined pipeline JSON + JSONL files.

    Returns (json_path, jsonl_path).
    """
    json_path = os.path.join(runs_dir, f"{prefix}.json")
    jsonl_path = os.path.join(runs_dir, f"{prefix}.jsonl")

    # Combined JSONL: train + eval records with phase tags
    with open(jsonl_path, "w") as f:
        for record in train_result.results_data.get("tasks", {}).values():
            f.write(json.dumps({"phase": "train", **record}) + "\n")
        for record in eval_result.results_data.get("tasks", {}).values():
            f.write(json.dumps({"phase": "eval", **record}) + "\n")

    # Combined JSON
    train_data = train_result.results_data
    eval_data = eval_result.results_data
    train_summary = train_data.get("summary", {})
    eval_summary = eval_data.get("summary", {})
    train_meta = train_data.get("meta", {})

    meta = {
        "timestamp": prefix.split("_")[-2] + "_" + prefix.split("_")[-1]
            if "_" in prefix else datetime.now().strftime("%Y%m%d_%H%M%S"),
        "datetime": datetime.now().isoformat(),
        "title": title,
        "domain": domain,
        "mode": train_meta.get("mode", getattr(args, "mode", "?")),
        "rounds": train_meta.get("rounds", resolved.get("rounds")),
        "beam_width": train_meta.get("beam_width", resolved.get("beam_width")),
        "max_generations": train_meta.get("max_generations", resolved.get("max_generations")),
        "workers": train_meta.get("workers", resolved.get("workers")),
        "seed": train_meta.get("seed", getattr(args, "seed", None)),
        "compute_cap": train_meta.get("compute_cap", resolved.get("compute_cap")),
        "n_primitives": train_meta.get("n_primitives"),
        "exhaustive_depth": getattr(args, "exhaustive_depth", None),
        "exhaustive_pair_top_k": getattr(args, "exhaustive_pair_top_k", None),
        "exhaustive_triple_top_k": getattr(args, "exhaustive_triple_top_k", None),
        "machine": train_meta.get("machine", {}),
    }
    if extra_meta:
        meta.update(extra_meta)

    def _phase_summary(summary):
        return {
            "n_tasks": summary.get("n_tasks", 0),
            "solved": summary.get("last_round_solved", 0),
            "solve_rate": summary.get("last_round_solve_rate", 0),
            "train_solved": summary.get("last_round_train_solved", 0),
            "train_solve_rate": summary.get("last_round_train_solve_rate", 0),
            "total_evaluations": summary.get("total_evaluations", 0),
            "wall_clock_seconds": summary.get("wall_clock_seconds", 0),
            "median_task_time": summary.get("median_task_time"),
            "throughput_tasks_per_sec": summary.get("throughput_tasks_per_sec"),
        }

    pipeline_data = {
        "meta": meta,
        "train": {**_phase_summary(train_summary),
                  "library_size": train_summary.get("library_size", 0)},
        "eval": _phase_summary(eval_summary),
        "train_tasks": train_data.get("tasks", {}),
        "eval_tasks": eval_data.get("tasks", {}),
        "library": train_data.get("library", []),
    }

    with open(json_path, "w") as f:
        json.dump(pipeline_data, f, indent=2)

    return json_path, jsonl_path


def print_pipeline_summary(
    train_result: ExperimentResult,
    eval_result: ExperimentResult,
    *,
    title: str,
    args,
    json_path: str,
    jsonl_path: str,
    log_path: str = "",
    eval_label: str = "Evaluation Results (with culture transfer)",
):
    """Print a formatted pipeline summary to stdout."""
    train_data = train_result.results_data
    eval_data = eval_result.results_data
    train_summary = train_data.get("summary", {})
    eval_summary = eval_data.get("summary", {})
    train_meta = train_data.get("meta", {})

    total_wall = (train_summary.get("wall_clock_seconds", 0)
                  + eval_summary.get("wall_clock_seconds", 0))

    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

    # Parameters
    print()
    print("  Parameters:")
    print(f"    Mode:              {getattr(args, 'mode', '?')}")
    print(f"    Rounds:            {train_meta.get('rounds', '?')}")
    print(f"    Beam:              {train_meta.get('beam_width', '?')}")
    print(f"    Generations:       {train_meta.get('max_generations', '?')}")
    print(f"    Workers:           {train_meta.get('workers', '?')}")
    print(f"    Seed:              {train_meta.get('seed', '?')}")
    cap = train_meta.get("compute_cap", 0)
    print(f"    Compute cap:       {cap:,} ops" if cap else "    Compute cap:       unlimited")
    print(f"    Exhaustive depth:  {getattr(args, 'exhaustive_depth', '?')}")
    print(f"    Pair top-K:        {getattr(args, 'exhaustive_pair_top_k', '?')}")
    print(f"    Triple top-K:      {getattr(args, 'exhaustive_triple_top_k', '?')}")
    print(f"    Primitives:        {train_meta.get('n_primitives', '?')}")

    # Train + eval results
    for label, summary in [("Training Results", train_summary),
                           (eval_label, eval_summary)]:
        solved = summary.get("last_round_solved", 0)
        total = summary.get("n_tasks", 0)
        rate = summary.get("last_round_solve_rate", 0)
        train_solved = summary.get("last_round_train_solved", 0)
        overfit = max(0, train_solved - solved)
        wall = summary.get("wall_clock_seconds", 0)

        print()
        print(f"  {label}:")
        print(f"    Tasks:             {total}")
        print(f"    Solved:            {solved}/{total} ({rate:.1%})")
        if overfit > 0:
            print(f"    Overfit:           {overfit}")
        print(f"    Evaluations:       {summary.get('total_evaluations', 0):,}")
        print(f"    Wall time:         {fmt_duration(wall)}")
        if label == "Training Results":
            print(f"    Library learned:   {summary.get('library_size', 0)} abstractions")

    print()
    print(f"  Total wall time:     {fmt_duration(total_wall)}")

    print()
    print("  Pipeline artifacts:")
    print(f"    Pipeline JSON:     {json_path}")
    print(f"    Pipeline JSONL:    {jsonl_path}")
    print(f"    Culture file:      {train_result.culture_path}")
    if log_path:
        print(f"    Console log:       {log_path}")

    print()
    print("=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
