"""
Visualize ARC-AGI results as an HTML report.

For each task, shows:
- Training examples: input → our prediction → expected output (with diff)
- Test examples: input → our prediction → expected output (with diff)
- Program used and intermediate steps (for composed programs)
- Solve status (solved / overfit / near-miss / unsolved)

Predictions are read from stored results JSON (computed during the run while
dynamic primitives are in memory). Falls back to re-execution for older
results files that don't include stored predictions.

Usage:
    python -m experiments.visualize_results runs/phase1_arc_train_XXXX.json
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
from domains.arc import ARCEnv, ARCDrive, load_arc_dataset
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

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARC-AGI Results Visualization</title>
<style>
body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
h1 { color: #7FDBFF; border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #FFDC00; margin-top: 30px; }
.summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }
.task { background: #16213e; border: 1px solid #333; border-radius: 8px; margin: 20px 0; padding: 15px; }
.task.solved { border-left: 4px solid #2ECC40; }
.task.overfit { border-left: 4px solid #FFDC00; }
.task.near-miss { border-left: 4px solid #FF851B; }
.task.unsolved { border-left: 4px solid #FF4136; }
.task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; cursor: pointer; }
.task-id { font-size: 1.1em; font-weight: bold; }
.status { padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
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
.task-body { display: block; }
.task-body.collapsed { display: none; }
.toggle-hint { font-size: 0.8em; color: #666; }
a.toc-link { color: #7FDBFF; text-decoration: none; }
a.toc-link:hover { text-decoration: underline; }
.toc { column-count: 3; column-gap: 20px; margin: 10px 0; }
.toc-item { margin: 2px 0; font-size: 0.85em; }
</style>
<script>
function toggleTask(id) {
    var body = document.getElementById('body-' + id);
    body.classList.toggle('collapsed');
}
</script>
</head>
<body>
"""

HTML_FOOTER = """
</body>
</html>
"""


def render_grid(grid: list[list[int]], diff_grid: Optional[list[list[int]]] = None,
                border_class: str = "") -> str:
    """Render an ARC grid as an HTML table with colored cells."""
    if not grid or not grid[0]:
        return '<div class="grid-wrapper"><em>empty</em></div>'

    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape

    # Adaptive cell size for large grids
    cell_size = 20
    if max(h, w) > 15:
        cell_size = max(12, min(20, 300 // max(h, w)))

    border_cls = f" {border_class}" if border_class else ""
    html_parts = [f'<div class="grid-wrapper"><div class="grid{border_cls}" '
                  f'style="grid-template-columns: repeat({w}, {cell_size}px);">']

    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(arr[r, c]), "#333")
            is_diff = (diff_grid is not None
                       and r < len(diff_grid) and c < len(diff_grid[0])
                       and int(arr[r, c]) != int(np.array(diff_grid)[r, c]))
            diff_cls = ' diff' if is_diff else ''
            html_parts.append(
                f'<div class="cell{diff_cls}" '
                f'style="width:{cell_size}px;height:{cell_size}px;background:{color};"></div>'
            )

    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _render_prediction_row(label_prefix: str, idx: int, inp: list, expected: list,
                           prediction: Optional[list]) -> list[str]:
    """Render input → prediction → expected for one example."""
    parts = []
    parts.append('<div class="examples">')
    parts.append('<div class="example">')
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
        # No prediction available — just show expected
        parts.append(f'<div>{render_grid(expected)}'
                     f'<div class="example-label">Expected</div></div>')

    parts.append('</div>')
    parts.append('</div>')
    return parts


def parse_program_tree(prog_str: str) -> Optional[Program]:
    """Parse a program repr string back into a Program tree."""
    prog_str = prog_str.strip()
    if not prog_str:
        return None

    # Find outermost function call: name(child1, child2)
    paren_idx = prog_str.find('(')
    if paren_idx == -1:
        # Simple leaf: just a primitive name
        return Program(root=prog_str)

    name = prog_str[:paren_idx]
    # Find matching closing paren
    inner = prog_str[paren_idx + 1:-1]  # strip outer parens

    # Split on commas at depth 0
    children_strs = []
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
    """Execute a program tree step-by-step, collecting intermediates.

    Walks the tree from innermost child to outermost parent, executing
    each subtree via env.execute() to handle dynamically registered
    primitives (neighborhood fixes, color remaps, etc.).
    """
    steps = [("input", grid)]

    def _collect_subtrees(node: Program) -> list[Program]:
        """Collect subtrees bottom-up: innermost child first."""
        subtrees = []
        if node.children:
            for child in node.children:
                subtrees.extend(_collect_subtrees(child))
        subtrees.append(node)
        return subtrees

    try:
        subtrees = _collect_subtrees(prog)
        prev_grid = grid
        for subtree in subtrees:
            result = env.execute(subtree, grid)
            if isinstance(result, list) and result:
                # Only add a step if it changed the grid from the previous step
                if result != prev_grid:
                    steps.append((repr(subtree), result))
                    prev_grid = result
    except Exception:
        pass

    return steps


def classify_task(tdata: dict) -> str:
    """Classify task status."""
    if tdata.get("solved"):
        return "solved"
    train_solved = tdata.get("train_solved", False)
    if train_solved:
        return "overfit"
    err = tdata.get("prediction_error", 1.0)
    if err <= 0.10:
        return "near-miss"
    return "unsolved"


def _get_prediction(stored_preds: Optional[list], idx: int,
                    prog: Program, inp: list, env: ARCEnv) -> list:
    """Get prediction from stored results, falling back to re-execution."""
    if stored_preds and idx < len(stored_preds) and stored_preds[idx] is not None:
        return stored_preds[idx]
    # Fallback: re-execute (may fail for dynamic primitives)
    try:
        return env.execute(prog, inp)
    except Exception:
        return inp


def generate_html(results_path: str, output_path: str,
                  filter_status: str = "", max_tasks: int = 0) -> str:
    """Generate HTML visualization from a results JSON file."""

    with open(results_path) as f:
        results = json.load(f)

    # Handle both pipeline and single-run formats
    if "train_tasks" in results and "eval_tasks" in results:
        tasks_data = {}
        for tid, t in results.get("train_tasks", {}).items():
            t["_phase"] = "train"
            tasks_data[tid] = t
        for tid, t in results.get("eval_tasks", {}).items():
            t["_phase"] = "eval"
            tasks_data[f"eval_{tid}"] = t
    else:
        tasks_data = results.get("tasks", {})

    # Load actual ARC tasks for grids
    data_dir = find_arc_data("training")
    eval_dir = find_arc_data("evaluation")
    task_map = {}
    if data_dir:
        for t in load_arc_dataset(data_dir):
            task_map[t.task_id] = t
    if eval_dir:
        for t in load_arc_dataset(eval_dir):
            task_map[t.task_id] = t

    env = ARCEnv()

    # Build HTML
    parts = [HTML_HEADER]

    # Summary
    total = len(tasks_data)
    solved = sum(1 for t in tasks_data.values() if t.get("solved"))
    overfit = sum(1 for t in tasks_data.values()
                  if not t.get("solved") and t.get("train_solved"))
    near_miss = sum(1 for t in tasks_data.values()
                    if not t.get("solved") and not t.get("train_solved")
                    and t.get("prediction_error", 1.0) <= 0.10)

    parts.append('<h1>ARC-AGI Results Visualization</h1>')
    parts.append('<div class="summary">')
    parts.append(f'<strong>Source:</strong> {html.escape(os.path.basename(results_path))}<br>')
    parts.append(f'<strong>Tasks:</strong> {total} &nbsp; '
                 f'<span style="color:#2ECC40">Solved: {solved}</span> &nbsp; '
                 f'<span style="color:#FFDC00">Overfit: {overfit}</span> &nbsp; '
                 f'<span style="color:#FF851B">Near-miss: {near_miss}</span> &nbsp; '
                 f'<span style="color:#FF4136">Unsolved: {total - solved - overfit - near_miss}</span>')
    parts.append('</div>')

    # Sort tasks: solved first, then by error
    task_items = []
    for key, tdata in tasks_data.items():
        tid = key.replace("eval_", "") if key.startswith("eval_") else key
        status = classify_task(tdata)
        err = tdata.get("prediction_error", 1.0)
        task_items.append((key, tid, tdata, status, err))

    # Apply filter
    if filter_status:
        task_items = [t for t in task_items if t[3] == filter_status]

    # Sort: solved, overfit, near-miss, unsolved; within each by error
    status_order = {"solved": 0, "overfit": 1, "near-miss": 2, "unsolved": 3}
    task_items.sort(key=lambda x: (status_order.get(x[3], 9), x[4]))

    if max_tasks > 0:
        task_items = task_items[:max_tasks]

    parts.append(f'<p>Showing {len(task_items)} tasks</p>')

    # Table of contents
    parts.append('<div class="section-label">Task Index</div>')
    parts.append('<div class="toc">')
    for key, tid, tdata, status, err in task_items:
        status_color = {"solved": "#2ECC40", "overfit": "#FFDC00",
                        "near-miss": "#FF851B", "unsolved": "#FF4136"}
        parts.append(f'<div class="toc-item">'
                     f'<span style="color:{status_color.get(status, "#e0e0e0")}">&#9679;</span> '
                     f'<a class="toc-link" href="#task-{html.escape(tid)}">{html.escape(tid)}</a>'
                     f'</div>')
    parts.append('</div>')

    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue

        prog_str = tdata.get("program", "identity") or "identity"
        train_preds = tdata.get("train_predictions")
        test_preds = tdata.get("test_predictions")

        prog = parse_program_tree(prog_str)
        if prog is None:
            prog = Program(root="identity")

        parts.append(f'<div id="task-{html.escape(tid)}" class="task {status}">')
        parts.append(f'<div class="task-header" onclick="toggleTask(\'{html.escape(tid)}\')">')
        parts.append(f'<span class="task-id">{html.escape(tid)} '
                     f'<span class="toggle-hint">(click to expand/collapse)</span></span>')
        parts.append(f'<span class="status {status}">{status.upper()}</span>')
        parts.append('</div>')

        parts.append(f'<div id="body-{html.escape(tid)}" class="task-body">')

        parts.append(f'<div class="program">Program: {html.escape(prog_str)}</div>')
        if err < 1.0:
            parts.append(f'<div class="error-info">Prediction error: {err:.4f}</div>')

        # Training examples with predictions
        parts.append('<div class="section-label">Training Examples (Our Prediction vs Expected)</div>')
        for i, (inp, exp) in enumerate(task.train_examples):
            prediction = _get_prediction(train_preds, i, prog, inp, env)
            parts.extend(_render_prediction_row("Train", i, inp, exp, prediction))

        # Test examples with our prediction
        if task.test_inputs:
            parts.append('<div class="section-label">Test Examples (Our Prediction vs Expected)</div>')

            for i, test_inp in enumerate(task.test_inputs):
                test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
                prediction = _get_prediction(test_preds, i, prog, test_inp, env)
                parts.extend(_render_prediction_row("Test", i, test_inp, test_exp, prediction))

                # Intermediate steps for test examples (for composed programs)
                if prog.children:
                    steps = collect_intermediate_steps(prog, test_inp, env)
                    if len(steps) > 2:  # more than just input→output
                        parts.append('<div class="section-label">Intermediate Steps (Test)</div>')
                        parts.append('<div class="steps">')
                        for step_name, step_grid in steps:
                            parts.append('<div class="step">')
                            parts.append(f'<div class="step-label">{html.escape(step_name)}</div>')
                            parts.append(render_grid(step_grid))
                            parts.append('</div>')
                        parts.append('</div>')

        parts.append('</div>')  # close task-body
        parts.append('</div>')  # close task div

    parts.append(HTML_FOOTER)

    html_content = '\n'.join(parts)
    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC-AGI results as HTML")
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path (auto-generated if not set)")
    parser.add_argument("--filter", choices=["solved", "overfit", "near-miss", "unsolved"],
                        default="", help="Show only tasks with this status")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks shown")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.results_json)[0]
        suffix = f"_{args.filter}" if args.filter else ""
        output_path = f"{base}{suffix}_viz.html"

    path = generate_html(args.results_json, output_path,
                         filter_status=args.filter, max_tasks=args.max_tasks)
    print(f"  HTML visualization: {path}")
    print(f"  Open in browser: open {path}")


if __name__ == "__main__":
    main()
