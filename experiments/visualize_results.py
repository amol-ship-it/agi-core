"""
Visualize ARC-AGI results as HTML: index page + per-task detail pages.

Generates:
  runs/<run_name>_viz/index.html     — summary with visual task previews
  runs/<run_name>_viz/tasks/<id>.html — per-task detail with step-by-step execution

Index page shows: summary stats, and for each task a visual preview
(first train example input→expected, test input→prediction vs expected).

Detail page shows: for every train and test example, the step-by-step
execution of each primitive in the composed program, showing intermediate
grids at each stage, then the final prediction compared to expected.

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
    0: "#000000",  # Black
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
# Shared CSS
# --------------------------------------------------------------------------
SHARED_CSS = """
body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
h1 { color: #7FDBFF; border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #FFDC00; margin-top: 25px; margin-bottom: 8px; }
h3 { color: #7FDBFF; margin-top: 18px; margin-bottom: 5px; font-size: 1em; }
a { color: #7FDBFF; text-decoration: none; }
a:hover { text-decoration: underline; }
.summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }
.status { padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; display: inline-block; }
.status.solved { background: #2ECC40; color: #000; }
.status.overfit { background: #FFDC00; color: #000; }
.status.near-miss { background: #FF851B; color: #000; }
.status.unsolved { background: #FF4136; color: #fff; }
.program { background: #0f3460; padding: 8px 12px; border-radius: 4px; margin: 8px 0; font-size: 0.9em; word-break: break-all; }
.grid-row { display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; margin: 6px 0; }
.grid-item { text-align: center; }
.grid-item-label { font-size: 0.75em; color: #999; margin-bottom: 3px; }
.arrow { font-size: 1.3em; color: #666; padding: 0 4px; align-self: center; margin-top: 12px; }
.grid-wrapper { text-align: center; }
.grid { display: inline-grid; gap: 1px; background: #333; border: 2px solid #555; }
.grid.diff-border { border: 2px solid #FF4136; }
.grid.match-border { border: 2px solid #2ECC40; }
.cell { min-width: 8px; min-height: 8px; }
.cell.diff { outline: 2px solid #FF4136; outline-offset: -2px; }
.section-label { font-size: 0.9em; color: #7FDBFF; margin: 15px 0 5px 0; font-weight: bold; }
.error-info { color: #FF851B; font-size: 0.85em; }
.step-arrow { color: #666; font-size: 1.1em; text-align: center; margin: 2px 0; }
.step-prim-name { color: #FF851B; font-size: 0.8em; text-align: center; margin: 1px 0; }
.step-flow { display: flex; align-items: flex-start; gap: 6px; flex-wrap: wrap; margin: 8px 0;
             padding: 10px; background: #0f3460; border-radius: 6px; }
.step-stage { text-align: center; }
"""

INDEX_CSS = """
.task-card { background: #16213e; border: 1px solid #333; border-radius: 8px;
             margin: 15px 0; padding: 12px; }
.task-card.solved { border-left: 4px solid #2ECC40; }
.task-card.overfit { border-left: 4px solid #FFDC00; }
.task-card.near-miss { border-left: 4px solid #FF851B; }
.task-card.unsolved { border-left: 4px solid #FF4136; }
.task-card-header { display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 6px; }
.task-card-header a { font-size: 1.05em; font-weight: bold; }
.task-card-meta { font-size: 0.8em; color: #999; margin-bottom: 8px; }
.task-card-grids { display: flex; gap: 30px; flex-wrap: wrap; }
.task-card-example { display: flex; align-items: flex-start; gap: 6px; }
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
# Grid rendering
# --------------------------------------------------------------------------

def render_grid(grid: list[list[int]], diff_grid: Optional[list[list[int]]] = None,
                border_class: str = "", cell_size: int = 0) -> str:
    """Render an ARC grid as colored CSS-grid cells."""
    if not grid or not grid[0]:
        return '<div class="grid-wrapper"><em>empty</em></div>'

    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape

    if cell_size <= 0:
        cell_size = 20
        if max(h, w) > 15:
            cell_size = max(10, min(20, 300 // max(h, w)))

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


def _grid_with_label(grid, label: str, diff_grid=None, border_class: str = "",
                     cell_size: int = 0) -> str:
    """Render a grid with a label underneath."""
    return (f'<div class="grid-item">'
            f'<div class="grid-item-label">{html.escape(label)}</div>'
            f'{render_grid(grid, diff_grid, border_class, cell_size)}'
            f'</div>')


def _prediction_match_info(prediction, expected) -> tuple[bool, Optional[list]]:
    """Compare prediction to expected. Returns (is_match, diff_grid_or_None)."""
    if prediction is None or expected is None:
        return False, None
    pred_arr = np.array(prediction, dtype=np.int32)
    exp_arr = np.array(expected, dtype=np.int32)
    if pred_arr.shape != exp_arr.shape:
        return False, None
    is_match = np.array_equal(pred_arr, exp_arr)
    return is_match, expected


# --------------------------------------------------------------------------
# Program parsing
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


def _program_steps(prog: Program) -> list[str]:
    """Extract the linear chain of primitive names from a program tree.

    For compose(A, B(C)): returns [C, B, A] (execution order, innermost first).
    """
    names: list[str] = []

    def _walk(node: Program):
        if node.children:
            for child in node.children:
                _walk(child)
        names.append(node.root)

    _walk(prog)
    return names


# --------------------------------------------------------------------------
# Step-by-step execution
# --------------------------------------------------------------------------

def _execute_steps(prog: Program, grid: list[list[int]],
                   env: ARCEnv) -> list[tuple[str, list[list[int]]]]:
    """Execute a program tree step-by-step, returning (prim_name, output) pairs.

    For a tree like A(B(C)), execution order is:
      1. C(input) -> grid1
      2. B(grid1) -> grid2
      3. A(grid2) -> grid3   (final)

    Returns [(prim_name, output_grid), ...] in execution order.
    Does NOT include the initial input — caller adds that.
    """
    steps: list[tuple[str, list[list[int]]]] = []

    def _eval_recursive(node: Program, inp: list[list[int]]) -> list[list[int]]:
        # First evaluate children (innermost-first)
        child_result = inp
        if node.children:
            for child in node.children:
                child_result = _eval_recursive(child, child_result)

        # Now apply this node's primitive to child result
        # Build a leaf program for just this primitive
        leaf = Program(root=node.root)
        try:
            result = env.execute(leaf, child_result)
            if not isinstance(result, list) or not result:
                result = child_result
        except Exception:
            result = child_result

        steps.append((node.root, result))
        return result

    try:
        _eval_recursive(prog, grid)
    except Exception:
        pass

    return steps


def _get_prediction(stored_preds: Optional[list], idx: int,
                    prog: Program, inp: list, env: ARCEnv) -> list:
    if stored_preds and idx < len(stored_preds) and stored_preds[idx] is not None:
        return stored_preds[idx]
    try:
        return env.execute(prog, inp)
    except Exception:
        return inp


# --------------------------------------------------------------------------
# Classification
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


# --------------------------------------------------------------------------
# Render one example with step-by-step execution
# --------------------------------------------------------------------------

def _render_example_with_steps(label: str, inp: list, expected: Optional[list],
                               prediction: Optional[list], prog: Program,
                               env: ARCEnv) -> str:
    """Render a single example showing step-by-step primitive execution.

    Shows: input --prim1--> intermediate1 --prim2--> ... --primN--> prediction  vs  expected
    """
    parts: list[str] = []
    parts.append(f'<h3>{html.escape(label)}</h3>')

    has_steps = prog.children or prog.root != "identity"

    if has_steps:
        # Execute step-by-step
        steps = _execute_steps(prog, inp, env)

        parts.append('<div class="step-flow">')
        # Input
        parts.append(_grid_with_label(inp, "Input"))

        for prim_name, step_grid in steps:
            # Arrow with primitive name
            parts.append(f'<div class="step-stage">'
                         f'<div class="step-prim-name">{html.escape(prim_name)}</div>'
                         f'<div class="arrow">&rarr;</div>'
                         f'</div>')
            parts.append(_grid_with_label(step_grid, f"after {prim_name}"))

        # Final comparison: vs expected
        if expected is not None:
            is_match, diff_ref = _prediction_match_info(prediction, expected)
            parts.append(f'<div class="step-stage">'
                         f'<div class="step-prim-name">compare</div>'
                         f'<div class="arrow">vs</div>'
                         f'</div>')
            border = "match-border" if is_match else "diff-border"
            parts.append(_grid_with_label(expected, "Expected", border_class=border))

        parts.append('</div>')  # close step-flow
    else:
        # Simple program (single primitive or identity): show input -> prediction vs expected
        parts.append('<div class="step-flow">')
        parts.append(_grid_with_label(inp, "Input"))

        if prediction is not None:
            parts.append(f'<div class="step-stage">'
                         f'<div class="step-prim-name">{html.escape(prog.root)}</div>'
                         f'<div class="arrow">&rarr;</div>'
                         f'</div>')

            if expected is not None:
                is_match, diff_ref = _prediction_match_info(prediction, expected)
                border = "match-border" if is_match else "diff-border"
                parts.append(_grid_with_label(prediction, "Prediction",
                                              diff_ref, border))
                parts.append(f'<div class="step-stage">'
                             f'<div class="step-prim-name">compare</div>'
                             f'<div class="arrow">vs</div>'
                             f'</div>')
                parts.append(_grid_with_label(expected, "Expected"))
            else:
                parts.append(_grid_with_label(prediction, "Prediction"))

        elif expected is not None:
            parts.append(f'<div class="arrow">&rarr;</div>')
            parts.append(_grid_with_label(expected, "Expected"))

        parts.append('</div>')

    return '\n'.join(parts)


# --------------------------------------------------------------------------
# Per-task detail page
# --------------------------------------------------------------------------

def _generate_task_page(tid: str, tdata: dict, task, env: ARCEnv) -> str:
    """Generate full HTML for a single task's detail page."""
    status = classify_task(tdata)
    prog_str = tdata.get("program", "identity") or "identity"
    err = tdata.get("prediction_error", 1.0)
    train_preds = tdata.get("train_predictions")
    test_preds = tdata.get("test_predictions")

    prog = parse_program_tree(prog_str) or Program(root="identity")

    b: list[str] = []
    b.append('<p><a href="../index.html">&larr; Back to index</a></p>')
    b.append(f'<h1>{html.escape(tid)} <span class="status {status}">'
             f'{status.upper()}</span></h1>')
    b.append(f'<div class="program">Program: {html.escape(prog_str)}</div>')

    if err < 1.0:
        b.append(f'<div class="error-info">Prediction error: {err:.4f}</div>')
    te = tdata.get("test_error")
    if te is not None:
        b.append(f'<div class="error-info">Test error: {te:.4f}</div>')

    # Training examples — step-by-step
    b.append('<h2>Training Examples</h2>')
    for i, (inp, exp) in enumerate(task.train_examples):
        prediction = _get_prediction(train_preds, i, prog, inp, env)
        b.append(_render_example_with_steps(
            f"Train {i+1}", inp, exp, prediction, prog, env))

    # Test examples — step-by-step
    if task.test_inputs:
        b.append('<h2>Test Examples</h2>')
        for i, test_inp in enumerate(task.test_inputs):
            test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
            prediction = _get_prediction(test_preds, i, prog, test_inp, env)
            b.append(_render_example_with_steps(
                f"Test {i+1}", test_inp, test_exp, prediction, prog, env))

    return _html_page(f"Task {tid}", '\n'.join(b))


# --------------------------------------------------------------------------
# Index page — visual previews
# --------------------------------------------------------------------------

def _generate_index_page(source_name: str, task_items: list,
                         task_map: dict, tasks_dir_name: str,
                         stored_data: dict) -> str:
    """Generate index HTML with visual task previews."""
    total = len(task_items)
    solved = sum(1 for *_, s, _ in task_items if s == "solved")
    overfit = sum(1 for *_, s, _ in task_items if s == "overfit")
    near_miss = sum(1 for *_, s, _ in task_items if s == "near-miss")
    unsolved = total - solved - overfit - near_miss

    b: list[str] = []
    b.append('<h1>ARC-AGI Results Visualization</h1>')
    b.append('<div class="summary">')
    b.append(f'<strong>Source:</strong> {html.escape(source_name)}<br>')
    b.append(f'<strong>Tasks:</strong> {total} &nbsp; '
             f'<span style="color:#2ECC40">Solved: {solved}</span> &nbsp; '
             f'<span style="color:#FFDC00">Overfit: {overfit}</span> &nbsp; '
             f'<span style="color:#FF851B">Near-miss: {near_miss}</span> &nbsp; '
             f'<span style="color:#FF4136">Unsolved: {unsolved}</span>')
    b.append('</div>')

    # Small thumbnails for index (8px cells)
    thumb = 8

    for i, (key, tid, tdata, status, err) in enumerate(task_items):
        task = task_map.get(tid)
        if task is None:
            continue

        prog_str = tdata.get("program", "identity") or "identity"
        te = tdata.get("test_error")

        b.append(f'<div class="task-card {status}">')

        # Header: task id + status
        b.append('<div class="task-card-header">')
        b.append(f'<a href="{tasks_dir_name}/{html.escape(tid)}.html">'
                 f'{html.escape(tid)}</a>')
        b.append(f'<span class="status {status}">{status.upper()}</span>')
        b.append('</div>')

        # Meta line
        prog_display = prog_str if len(prog_str) <= 70 else prog_str[:67] + "..."
        meta_parts = [f'Program: {html.escape(prog_display)}']
        if err < 1.0:
            meta_parts.append(f'error: {err:.4f}')
        if te is not None:
            meta_parts.append(f'test_error: {te:.4f}')
        b.append(f'<div class="task-card-meta">{" &nbsp;|&nbsp; ".join(meta_parts)}</div>')

        # Visual preview: first train example input → expected (compact)
        b.append('<div class="task-card-grids">')
        if task.train_examples:
            inp0, exp0 = task.train_examples[0]
            b.append('<div class="task-card-example">')
            b.append(_grid_with_label(inp0, "Input", cell_size=thumb))
            b.append('<div class="arrow">&rarr;</div>')
            b.append(_grid_with_label(exp0, "Expected", cell_size=thumb))
            b.append('</div>')
        b.append('</div>')  # task-card-grids
        b.append('</div>')  # task-card

    return _html_page("ARC-AGI Results", '\n'.join(b), extra_css=INDEX_CSS)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def generate_html(results_path: str, output_dir: str,
                  filter_status: str = "", max_tasks: int = 0) -> str:
    """Generate HTML visualization: index + per-task detail pages."""
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
    index_html = _generate_index_page(source_name, task_items, task_map,
                                      tasks_dir_name, tasks_data)
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
