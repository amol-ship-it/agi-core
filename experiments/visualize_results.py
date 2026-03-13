"""
Visualize ARC-AGI results as HTML: index page + per-task detail pages.

Generates:
  runs/<run_name>_viz/index.html     — summary with task table linking to details
  runs/<run_name>_viz/tasks/<id>.html — per-task detail with grids, predictions, steps

Predictions are read from stored results JSON (computed during the run while
dynamic primitives are in memory). Falls back to re-execution for older results.

Usage:
    python -m experiments.visualize_results runs/phase1_arc_pipeline_XXXX.json
    python -m experiments.visualize_results runs/phase1_arc_train_XXXX.json --filter unsolved
    python -m experiments.visualize_results runs/phase1_arc_train_XXXX.json --max-tasks 20
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import Program
from domains.arc import ARCEnv, load_arc_dataset
from experiments.phase1_arc import find_arc_data


# ARC official colors (from ARC-AGI/apps/css/common.css)
ARC_COLORS = {
    0: "#000000",  # Black (background)
    1: "#0074D9",  # Blue
    2: "#FF4136",  # Red
    3: "#2ECC40",  # Green
    4: "#FFDC00",  # Yellow
    5: "#AAAAAA",  # Grey
    6: "#F012BE",  # Magenta
    7: "#FF851B",  # Orange
    8: "#7FDBFF",  # Cyan
    9: "#870C25",  # Maroon
}

# --------------------------------------------------------------------------
# Shared CSS used by both index and detail pages
# --------------------------------------------------------------------------
SHARED_CSS = """
body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
h1 { color: #7FDBFF; border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #FFDC00; margin-top: 30px; }
a { color: #7FDBFF; text-decoration: none; }
a:hover { text-decoration: underline; }
.summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }
.status { padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; display: inline-block; }
.status.solved { background: #2ECC40; color: #000; }
.status.overfit { background: #FFDC00; color: #000; }
.status.near-miss { background: #FF851B; color: #000; }
.status.unsolved { background: #FF4136; color: #fff; }
.program { background: #0f3460; padding: 8px 12px; border-radius: 4px; margin: 8px 0; font-size: 0.9em; word-break: break-all; }
.examples { display: flex; flex-wrap: wrap; gap: 20px; margin: 10px 0; }
.example { display: flex; align-items: flex-start; gap: 8px; }
.example-label { font-size: 0.8em; color: #999; margin-bottom: 4px; text-align: center; }
.arrow { font-size: 1.5em; color: #666; padding: 0 5px; align-self: center; margin-top: 15px; }
.grid-wrapper { text-align: center; }
.grid { display: inline-grid; gap: 1px; background: #333; border: 2px solid #555; }
.grid.diff-border { border: 2px solid #FF4136; }
.grid.match-border { border: 2px solid #2ECC40; }
.cell { width: 20px; height: 20px; min-width: 12px; min-height: 12px; }
.cell.diff { outline: 2px solid #FF4136; outline-offset: -2px; }
.steps { margin: 10px 0; }
.step { display: flex; align-items: flex-start; gap: 10px; margin: 5px 0; padding: 8px; background: #0f3460; border-radius: 4px; }
.step-label { color: #7FDBFF; font-size: 0.85em; min-width: 200px; word-break: break-all; }
.section-label { font-size: 0.9em; color: #7FDBFF; margin: 15px 0 5px 0; font-weight: bold; }
.error-info { color: #FF851B; font-size: 0.85em; }
"""

INDEX_CSS = """
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #333; }
th { color: #7FDBFF; font-size: 0.9em; }
tr:hover { background: #1f2f4f; }
.filter-bar { margin: 10px 0; }
.filter-bar a { margin-right: 15px; }
.filter-bar a.active { color: #FFDC00; font-weight: bold; }
"""


def _html_page(title: str, body: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
<style>
{SHARED_CSS}
{extra_css}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# --------------------------------------------------------------------------
# Grid rendering helpers
# --------------------------------------------------------------------------

def render_grid(grid: list[list[int]], diff_grid: Optional[list[list[int]]] = None,
                border_class: str = "") -> str:
    """Render an ARC grid as colored CSS-grid cells."""
    if not grid or not grid[0]:
        return '<div class="grid-wrapper"><em>empty</em></div>'

    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape

    cell_size = 20
    if max(h, w) > 15:
        cell_size = max(12, min(20, 300 // max(h, w)))

    border_cls = f" {border_class}" if border_class else ""
    parts = [f'<div class="grid-wrapper"><div class="grid{border_cls}" '
             f'style="grid-template-columns: repeat({w}, {cell_size}px);">']

    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(arr[r, c]), "#333")
            is_diff = (diff_grid is not None
                       and r < len(diff_grid) and c < len(diff_grid[0])
                       and int(arr[r, c]) != int(np.array(diff_grid)[r, c]))
            diff_cls = ' diff' if is_diff else ''
            parts.append(
                f'<div class="cell{diff_cls}" '
                f'style="width:{cell_size}px;height:{cell_size}px;background:{color};"></div>'
            )

    parts.append('</div></div>')
    return ''.join(parts)


def _render_prediction_row(label_prefix: str, idx: int, inp: list, expected: list,
                           prediction: Optional[list]) -> str:
    """Render input -> prediction -> expected for one example."""
    parts = ['<div class="examples">', '<div class="example">']
    parts.append(f'<div>{render_grid(inp)}'
                 f'<div class="example-label">{label_prefix} {idx+1} Input</div></div>')
    parts.append('<div class="arrow">&rarr;</div>')

    if prediction is not None and expected is not None:
        pred_arr = np.array(prediction, dtype=np.int32)
        exp_arr = np.array(expected, dtype=np.int32)
        is_match = pred_arr.shape == exp_arr.shape and np.array_equal(pred_arr, exp_arr)
        border = "match-border" if is_match else "diff-border"
        diff_ref = expected if pred_arr.shape == exp_arr.shape else None
        parts.append(f'<div>{render_grid(prediction, diff_ref, border)}'
                     f'<div class="example-label">Our Prediction</div></div>')
        parts.append('<div class="arrow">vs</div>')
        parts.append(f'<div>{render_grid(expected)}'
                     f'<div class="example-label">Expected</div></div>')
    elif prediction is not None:
        parts.append(f'<div>{render_grid(prediction)}'
                     f'<div class="example-label">Our Prediction</div></div>')
        if expected is not None:
            parts.append('<div class="arrow">vs</div>')
            parts.append(f'<div>{render_grid(expected)}'
                         f'<div class="example-label">Expected</div></div>')
    else:
        if expected is not None:
            parts.append(f'<div>{render_grid(expected)}'
                         f'<div class="example-label">Expected</div></div>')

    parts.append('</div></div>')
    return '\n'.join(parts)


# --------------------------------------------------------------------------
# Program parsing and intermediate-step collection
# --------------------------------------------------------------------------

def parse_program_tree(prog_str: str) -> Optional[Program]:
    """Parse a program repr string back into a Program tree."""
    prog_str = prog_str.strip()
    if not prog_str:
        return None

    paren_idx = prog_str.find('(')
    if paren_idx == -1:
        return Program(root=prog_str)

    name = prog_str[:paren_idx]
    inner = prog_str[paren_idx + 1:-1]

    children_strs: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            children_strs.append(inner[start:i].strip())
            start = i + 1
    children_strs.append(inner[start:].strip())

    children = []
    for cs in children_strs:
        if cs:
            child = parse_program_tree(cs)
            if child:
                children.append(child)

    return Program(root=name, children=children)


def collect_intermediate_steps(prog: Program, grid: list[list[int]],
                               env: ARCEnv) -> list[tuple[str, list[list[int]]]]:
    """Walk the program tree bottom-up, executing each subtree."""
    steps: list[tuple[str, list[list[int]]]] = [("input", grid)]

    def _subtrees(node: Program) -> list[Program]:
        subs: list[Program] = []
        if node.children:
            for child in node.children:
                subs.extend(_subtrees(child))
        subs.append(node)
        return subs

    try:
        prev = grid
        for sub in _subtrees(prog):
            result = env.execute(sub, grid)
            if isinstance(result, list) and result and result != prev:
                steps.append((repr(sub), result))
                prev = result
    except Exception:
        pass

    return steps


# --------------------------------------------------------------------------
# Classification helpers
# --------------------------------------------------------------------------

def classify_task(tdata: dict) -> str:
    if tdata.get("solved"):
        return "solved"
    if tdata.get("train_solved"):
        return "overfit"
    if tdata.get("prediction_error", 1.0) <= 0.10:
        return "near-miss"
    return "unsolved"


STATUS_COLORS = {
    "solved": "#2ECC40",
    "overfit": "#FFDC00",
    "near-miss": "#FF851B",
    "unsolved": "#FF4136",
}


def _get_prediction(stored_preds: Optional[list], idx: int,
                    prog: Program, inp: list, env: ARCEnv) -> list:
    if stored_preds and idx < len(stored_preds) and stored_preds[idx] is not None:
        return stored_preds[idx]
    try:
        return env.execute(prog, inp)
    except Exception:
        return inp


# --------------------------------------------------------------------------
# Per-task detail page
# --------------------------------------------------------------------------

def _generate_task_page(tid: str, tdata: dict, task, env: ARCEnv) -> str:
    """Generate the full HTML for a single task's detail page."""
    status = classify_task(tdata)
    prog_str = tdata.get("program", "identity") or "identity"
    err = tdata.get("prediction_error", 1.0)
    train_preds = tdata.get("train_predictions")
    test_preds = tdata.get("test_predictions")

    prog = parse_program_tree(prog_str) or Program(root="identity")

    body_parts: list[str] = []
    body_parts.append(f'<p><a href="../index.html">&larr; Back to index</a></p>')
    body_parts.append(f'<h1>{html.escape(tid)} <span class="status {status}">'
                      f'{status.upper()}</span></h1>')

    body_parts.append(f'<div class="program">Program: {html.escape(prog_str)}</div>')
    if err < 1.0:
        body_parts.append(f'<div class="error-info">Prediction error: {err:.4f}</div>')

    te = tdata.get("test_error")
    if te is not None:
        body_parts.append(f'<div class="error-info">Test error: {te:.4f}</div>')

    # Training examples with predictions
    body_parts.append('<div class="section-label">Training Examples '
                      '(Our Prediction vs Expected)</div>')
    for i, (inp, exp) in enumerate(task.train_examples):
        prediction = _get_prediction(train_preds, i, prog, inp, env)
        body_parts.append(_render_prediction_row("Train", i, inp, exp, prediction))

    # Test examples with predictions
    if task.test_inputs:
        body_parts.append('<div class="section-label">Test Examples '
                          '(Our Prediction vs Expected)</div>')
        for i, test_inp in enumerate(task.test_inputs):
            test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
            prediction = _get_prediction(test_preds, i, prog, test_inp, env)
            body_parts.append(_render_prediction_row("Test", i, test_inp, test_exp, prediction))

            # Intermediate steps for composed programs
            if prog.children:
                steps = collect_intermediate_steps(prog, test_inp, env)
                if len(steps) > 2:
                    body_parts.append('<div class="section-label">'
                                     'Intermediate Steps (Test)</div>')
                    body_parts.append('<div class="steps">')
                    for step_name, step_grid in steps:
                        body_parts.append(
                            f'<div class="step">'
                            f'<div class="step-label">{html.escape(step_name)}</div>'
                            f'{render_grid(step_grid)}'
                            f'</div>')
                    body_parts.append('</div>')

    return _html_page(f"Task {tid}", '\n'.join(body_parts))


# --------------------------------------------------------------------------
# Index page
# --------------------------------------------------------------------------

def _generate_index_page(source_name: str, task_items: list, tasks_dir_name: str) -> str:
    """Generate the index HTML with summary table linking to per-task pages."""
    total = len(task_items)
    solved = sum(1 for *_, s, _ in task_items if s == "solved")
    overfit = sum(1 for *_, s, _ in task_items if s == "overfit")
    near_miss = sum(1 for *_, s, _ in task_items if s == "near-miss")
    unsolved = total - solved - overfit - near_miss

    body: list[str] = []
    body.append('<h1>ARC-AGI Results Visualization</h1>')
    body.append('<div class="summary">')
    body.append(f'<strong>Source:</strong> {html.escape(source_name)}<br>')
    body.append(f'<strong>Tasks:</strong> {total} &nbsp; '
                f'<span style="color:#2ECC40">Solved: {solved}</span> &nbsp; '
                f'<span style="color:#FFDC00">Overfit: {overfit}</span> &nbsp; '
                f'<span style="color:#FF851B">Near-miss: {near_miss}</span> &nbsp; '
                f'<span style="color:#FF4136">Unsolved: {unsolved}</span>')
    body.append('</div>')

    # Task table
    body.append('<table>')
    body.append('<tr><th>#</th><th>Task ID</th><th>Status</th>'
                '<th>Program</th><th>Pred. Error</th><th>Test Error</th></tr>')

    for i, (key, tid, tdata, status, err) in enumerate(task_items):
        prog = html.escape(tdata.get("program", "identity") or "identity")
        # Truncate long programs in the table
        prog_display = prog if len(prog) <= 60 else prog[:57] + "..."
        te = tdata.get("test_error")
        te_str = f"{te:.4f}" if te is not None else "&mdash;"
        err_str = f"{err:.4f}" if err < 1.0 else "&mdash;"
        color = STATUS_COLORS.get(status, "#e0e0e0")

        body.append(
            f'<tr>'
            f'<td>{i+1}</td>'
            f'<td><a href="{tasks_dir_name}/{html.escape(tid)}.html">{html.escape(tid)}</a></td>'
            f'<td><span class="status {status}">{status.upper()}</span></td>'
            f'<td style="font-size:0.8em">{prog_display}</td>'
            f'<td>{err_str}</td>'
            f'<td>{te_str}</td>'
            f'</tr>')

    body.append('</table>')

    return _html_page("ARC-AGI Results", '\n'.join(body), extra_css=INDEX_CSS)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def generate_html(results_path: str, output_dir: str,
                  filter_status: str = "", max_tasks: int = 0) -> str:
    """Generate HTML visualization: index + per-task detail pages.

    Args:
        results_path: path to results JSON
        output_dir: directory to write index.html + tasks/*.html
        filter_status: only show tasks with this status
        max_tasks: limit number of tasks (0 = all)

    Returns:
        path to index.html
    """
    with open(results_path) as f:
        results = json.load(f)

    # Handle both pipeline and single-run formats
    if "train_tasks" in results and "eval_tasks" in results:
        tasks_data: dict[str, dict] = {}
        for tid, t in results.get("train_tasks", {}).items():
            t["_phase"] = "train"
            tasks_data[tid] = t
        for tid, t in results.get("eval_tasks", {}).items():
            t["_phase"] = "eval"
            tasks_data[f"eval_{tid}"] = t
    else:
        tasks_data = results.get("tasks", {})

    # Load actual ARC tasks for grids
    task_map: dict = {}
    for split in ("training", "evaluation"):
        data_dir = find_arc_data(split)
        if data_dir:
            for t in load_arc_dataset(data_dir):
                task_map[t.task_id] = t

    env = ARCEnv()

    # Build sorted task list
    task_items = []
    for key, tdata in tasks_data.items():
        tid = key.replace("eval_", "") if key.startswith("eval_") else key
        status = classify_task(tdata)
        err = tdata.get("prediction_error", 1.0)
        task_items.append((key, tid, tdata, status, err))

    if filter_status:
        task_items = [t for t in task_items if t[3] == filter_status]

    status_order = {"solved": 0, "overfit": 1, "near-miss": 2, "unsolved": 3}
    task_items.sort(key=lambda x: (status_order.get(x[3], 9), x[4]))

    if max_tasks > 0:
        task_items = task_items[:max_tasks]

    # Create output directories
    tasks_dir_name = "tasks"
    tasks_dir = os.path.join(output_dir, tasks_dir_name)
    os.makedirs(tasks_dir, exist_ok=True)

    # Generate per-task detail pages
    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue
        task_html = _generate_task_page(tid, tdata, task, env)
        with open(os.path.join(tasks_dir, f"{tid}.html"), "w") as f:
            f.write(task_html)

    # Generate index page
    source_name = os.path.basename(results_path)
    index_html = _generate_index_page(source_name, task_items, tasks_dir_name)
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)

    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ARC-AGI results as HTML (index + per-task pages)")
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <results>_viz/)")
    parser.add_argument("--filter",
                        choices=["solved", "overfit", "near-miss", "unsolved"],
                        default="", help="Show only tasks with this status")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Limit number of tasks shown")
    args = parser.parse_args()

    if args.output:
        output_dir = args.output
    else:
        base = os.path.splitext(args.results_json)[0]
        suffix = f"_{args.filter}" if args.filter else ""
        output_dir = f"{base}{suffix}_viz"

    index_path = generate_html(args.results_json, output_dir,
                               filter_status=args.filter, max_tasks=args.max_tasks)
    print(f"  Visualization: {output_dir}/")
    print(f"  Index page:    {index_path}")
    print(f"  Open in browser: open {index_path}")


if __name__ == "__main__":
    main()
