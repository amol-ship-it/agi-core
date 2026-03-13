"""
Visualize ARC-AGI results as HTML: separate train/eval index files + per-task pages.

Output structure (pipeline runs):
  runs/<prefix>_train_viz.html        — train index directly in runs/
  runs/<prefix>_train_viz/<tid>.html  — per-task train detail pages
  runs/<prefix>_eval_viz.html         — eval index directly in runs/
  runs/<prefix>_eval_viz/<tid>.html   — per-task eval detail pages

Output structure (single-run):
  runs/<prefix>_viz.html              — index directly in runs/
  runs/<prefix>_viz/<tid>.html        — per-task detail pages

Index: each task card shows ALL examples (train + test), one row per example:
  Input | Expected | Predicted (with match/diff highlighting)

Detail page: each example shown as:
  Input | Expected | Predicted
  Then below: step-by-step derivation from input → predicted

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


ARC_COLORS = {
    0: "#000000", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40", 4: "#FFDC00",
    5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B", 8: "#7FDBFF", 9: "#870C25",
}

SHARED_CSS = """
body { font-family: 'SF Mono','Menlo','Monaco',monospace; background:#1a1a2e; color:#e0e0e0; margin:20px; }
h1 { color:#7FDBFF; border-bottom:2px solid #333; padding-bottom:10px; }
h2 { color:#FFDC00; margin-top:25px; margin-bottom:8px; }
h3 { color:#7FDBFF; margin-top:18px; margin-bottom:5px; font-size:1em; }
a { color:#7FDBFF; text-decoration:none; }
a:hover { text-decoration:underline; }
.summary { background:#16213e; padding:15px; border-radius:8px; margin:10px 0; }
.status { padding:3px 10px; border-radius:4px; font-size:0.85em; font-weight:bold; display:inline-block; }
.status.solved { background:#2ECC40; color:#000; }
.status.overfit { background:#FFDC00; color:#000; }
.status.near-miss { background:#FF851B; color:#000; }
.status.unsolved { background:#FF4136; color:#fff; }
.program { background:#0f3460; padding:8px 12px; border-radius:4px; margin:8px 0; font-size:0.9em; word-break:break-all; }
.error-info { color:#FF851B; font-size:0.85em; }
.grid-wrapper { text-align:center; }
.grid { display:inline-grid; gap:1px; background:#333; border:2px solid #555; }
.grid.diff-border { border:2px solid #FF4136; }
.grid.match-border { border:2px solid #2ECC40; }
.cell { min-width:8px; min-height:8px; }
.cell.diff { outline:2px solid #FF4136; outline-offset:-2px; }
.grid-item { text-align:center; }
.grid-item-label { font-size:0.75em; color:#999; margin-bottom:3px; }
.example-row { display:flex; align-items:flex-start; gap:10px; margin:6px 0; padding:6px;
               background:#16213e; border-radius:4px; }
.example-row-label { color:#7FDBFF; font-size:0.8em; min-width:60px; align-self:center; }
.arrow { font-size:1.2em; color:#666; align-self:center; }
.step-flow { display:flex; align-items:flex-start; gap:6px; flex-wrap:wrap; margin:8px 0;
             padding:10px; background:#0f3460; border-radius:6px; }
.step-stage { text-align:center; }
.step-prim-name { color:#FF851B; font-size:0.8em; text-align:center; }
"""

INDEX_CSS = """
.task-card { background:#16213e; border:1px solid #333; border-radius:8px; margin:15px 0; padding:12px; }
.task-card.solved { border-left:4px solid #2ECC40; }
.task-card.overfit { border-left:4px solid #FFDC00; }
.task-card.near-miss { border-left:4px solid #FF851B; }
.task-card.unsolved { border-left:4px solid #FF4136; }
.task-card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.task-card-header a { font-size:1.05em; font-weight:bold; }
.task-card-meta { font-size:0.8em; color:#999; margin-bottom:8px; }
"""


def _html_page(title: str, body: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{html.escape(title)}</title>
<style>{SHARED_CSS}{extra_css}</style>
</head><body>
{body}
</body></html>"""


# --------------------------------------------------------------------------
# Grid rendering
# --------------------------------------------------------------------------

def render_grid(grid, diff_grid=None, border_class="", cell_size=0) -> str:
    if not grid or not grid[0]:
        return '<div class="grid-wrapper"><em>empty</em></div>'
    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape
    if cell_size <= 0:
        cell_size = 20
        if max(h, w) > 15:
            cell_size = max(10, min(20, 300 // max(h, w)))
    bcls = f" {border_class}" if border_class else ""
    parts = [f'<div class="grid-wrapper"><div class="grid{bcls}" '
             f'style="grid-template-columns:repeat({w},{cell_size}px);">']
    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(arr[r, c]), "#333")
            is_diff = (diff_grid is not None
                       and r < len(diff_grid) and c < len(diff_grid[0])
                       and int(arr[r, c]) != int(np.array(diff_grid)[r, c]))
            dc = ' diff' if is_diff else ''
            parts.append(f'<div class="cell{dc}" '
                         f'style="width:{cell_size}px;height:{cell_size}px;background:{color}"></div>')
    parts.append('</div></div>')
    return ''.join(parts)


def _grid_with_label(grid, label, diff_grid=None, border_class="", cell_size=0) -> str:
    return (f'<div class="grid-item"><div class="grid-item-label">{html.escape(label)}</div>'
            f'{render_grid(grid, diff_grid, border_class, cell_size)}</div>')


def _pred_border(prediction, expected) -> tuple[str, Optional[list]]:
    """Returns (border_class, diff_grid) for a prediction vs expected."""
    if prediction is None or expected is None:
        return "", None
    pa = np.array(prediction, dtype=np.int32)
    ea = np.array(expected, dtype=np.int32)
    if pa.shape != ea.shape:
        return "diff-border", None
    if np.array_equal(pa, ea):
        return "match-border", None
    return "diff-border", expected


# --------------------------------------------------------------------------
# Program parsing & step-by-step execution
# --------------------------------------------------------------------------

def parse_program_tree(prog_str: str) -> Optional[Program]:
    s = prog_str.strip()
    if not s:
        return None
    pi = s.find('(')
    if pi == -1:
        return Program(root=s)
    name = s[:pi]
    inner = s[pi + 1:-1]
    children_strs, depth, start = [], 0, 0
    for i, ch in enumerate(inner):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            children_strs.append(inner[start:i].strip())
            start = i + 1
    children_strs.append(inner[start:].strip())
    children = [parse_program_tree(cs) for cs in children_strs if cs]
    return Program(root=name, children=[c for c in children if c])


def _execute_steps(prog: Program, grid, env: ARCEnv) -> list[tuple[str, list]]:
    """Execute program tree step-by-step. Returns [(prim_name, output), ...]."""
    steps: list[tuple[str, list]] = []

    def _eval(node: Program, inp):
        child_result = inp
        if node.children:
            for child in node.children:
                child_result = _eval(child, child_result)
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
        _eval(prog, grid)
    except Exception:
        pass
    return steps


def _get_prediction(stored_preds, idx, prog, inp, env):
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


# --------------------------------------------------------------------------
# Render one example row: Input | Expected | Predicted
# --------------------------------------------------------------------------

def _render_example_row(label, inp, expected, prediction, cell_size=0) -> str:
    """One row: label  Input  Expected  Predicted (with diff)."""
    border, diff_ref = _pred_border(prediction, expected)
    parts = [f'<div class="example-row">',
             f'<div class="example-row-label">{html.escape(label)}</div>',
             _grid_with_label(inp, "Input", cell_size=cell_size)]
    if expected is not None:
        parts.append(_grid_with_label(expected, "Expected", cell_size=cell_size))
    if prediction is not None:
        parts.append(_grid_with_label(prediction, "Predicted", diff_ref, border,
                                      cell_size=cell_size))
    parts.append('</div>')
    return '\n'.join(parts)


# --------------------------------------------------------------------------
# Render derivation: step-by-step from input → predicted
# --------------------------------------------------------------------------

def _render_derivation(inp, prog, env) -> str:
    """Show step-by-step execution: input --prim→ ... --prim→ predicted."""
    has_steps = prog.children or prog.root != "identity"
    if not has_steps:
        return ''

    steps = _execute_steps(prog, inp, env)
    if not steps:
        return ''

    parts = ['<div class="step-flow">', _grid_with_label(inp, "Input")]
    for prim_name, step_grid in steps:
        parts.append(f'<div class="step-stage">'
                     f'<div class="step-prim-name">{html.escape(prim_name)}</div>'
                     f'<div class="arrow">&rarr;</div></div>')
        parts.append(_grid_with_label(step_grid, f"after {prim_name}"))
    parts.append('</div>')
    return '\n'.join(parts)


# --------------------------------------------------------------------------
# Per-task detail page
# --------------------------------------------------------------------------

def _generate_task_page(tid, tdata, task, env, back_link="../index.html") -> str:
    status = classify_task(tdata)
    prog_str = tdata.get("program", "identity") or "identity"
    err = tdata.get("prediction_error", 1.0)
    train_preds = tdata.get("train_predictions")
    test_preds = tdata.get("test_predictions")
    prog = parse_program_tree(prog_str) or Program(root="identity")

    b: list[str] = []
    b.append(f'<p><a href="{back_link}">&larr; Back to index</a></p>')
    b.append(f'<h1>{html.escape(tid)} <span class="status {status}">'
             f'{status.upper()}</span></h1>')
    b.append(f'<div class="program">Program: {html.escape(prog_str)}</div>')
    if err < 1.0:
        b.append(f'<div class="error-info">Prediction error: {err:.4f}</div>')
    te = tdata.get("test_error")
    if te is not None:
        b.append(f'<div class="error-info">Test error: {te:.4f}</div>')

    # Training examples
    b.append('<h2>Training Examples</h2>')
    for i, (inp, exp) in enumerate(task.train_examples):
        prediction = _get_prediction(train_preds, i, prog, inp, env)
        b.append(_render_example_row(f"Train {i+1}", inp, exp, prediction))
        b.append(_render_derivation(inp, prog, env))

    # Test examples
    if task.test_inputs:
        b.append('<h2>Test Examples</h2>')
        for i, test_inp in enumerate(task.test_inputs):
            test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
            prediction = _get_prediction(test_preds, i, prog, test_inp, env)
            b.append(_render_example_row(f"Test {i+1}", test_inp, test_exp, prediction))
            b.append(_render_derivation(test_inp, prog, env))

    return _html_page(f"Task {tid}", '\n'.join(b))


# --------------------------------------------------------------------------
# Index page
# --------------------------------------------------------------------------

def _generate_index(title, source_name, task_items, task_map, tasks_dir_name,
                    env, thumb=10) -> str:
    """Index page with visual preview rows for each task."""
    total = len(task_items)
    solved = sum(1 for *_, s, _ in task_items if s == "solved")
    overfit = sum(1 for *_, s, _ in task_items if s == "overfit")
    near_miss = sum(1 for *_, s, _ in task_items if s == "near-miss")
    unsolved = total - solved - overfit - near_miss

    b: list[str] = []
    b.append(f'<h1>{html.escape(title)}</h1>')
    b.append('<div class="summary">')
    b.append(f'<strong>Source:</strong> {html.escape(source_name)}<br>')
    b.append(f'<strong>Tasks:</strong> {total} &nbsp; '
             f'<span style="color:#2ECC40">Solved: {solved}</span> &nbsp; '
             f'<span style="color:#FFDC00">Overfit: {overfit}</span> &nbsp; '
             f'<span style="color:#FF851B">Near-miss: {near_miss}</span> &nbsp; '
             f'<span style="color:#FF4136">Unsolved: {unsolved}</span>')
    b.append('</div>')

    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue

        prog_str = tdata.get("program", "identity") or "identity"
        te = tdata.get("test_error")
        train_preds = tdata.get("train_predictions")
        test_preds = tdata.get("test_predictions")
        prog = parse_program_tree(prog_str) or Program(root="identity")

        b.append(f'<div class="task-card {status}">')
        b.append('<div class="task-card-header">')
        b.append(f'<a href="{tasks_dir_name}/{html.escape(tid)}.html">'
                 f'{html.escape(tid)}</a>')
        b.append(f'<span class="status {status}">{status.upper()}</span>')
        b.append('</div>')

        prog_display = prog_str if len(prog_str) <= 70 else prog_str[:67] + "..."
        meta = [f'Program: {html.escape(prog_display)}']
        if err < 1.0:
            meta.append(f'error: {err:.4f}')
        if te is not None:
            meta.append(f'test_error: {te:.4f}')
        b.append(f'<div class="task-card-meta">{" &nbsp;|&nbsp; ".join(meta)}</div>')

        # All train examples: Input | Expected | Predicted
        for i, (inp, exp) in enumerate(task.train_examples):
            prediction = _get_prediction(train_preds, i, prog, inp, env)
            b.append(_render_example_row(f"Train {i+1}", inp, exp, prediction,
                                         cell_size=thumb))

        # All test examples: Input | Expected | Predicted
        if task.test_inputs:
            for i, test_inp in enumerate(task.test_inputs):
                test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
                prediction = _get_prediction(test_preds, i, prog, test_inp, env)
                b.append(_render_example_row(f"Test {i+1}", test_inp, test_exp,
                                             prediction, cell_size=thumb))

        b.append('</div>')  # task-card

    return _html_page(title, '\n'.join(b), extra_css=INDEX_CSS)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _sort_and_filter(task_items, filter_status="", max_tasks=0):
    if filter_status:
        task_items = [t for t in task_items if t[3] == filter_status]
    order = {"solved": 0, "overfit": 1, "near-miss": 2, "unsolved": 3}
    task_items.sort(key=lambda x: (order.get(x[3], 9), x[4]))
    if max_tasks > 0:
        task_items = task_items[:max_tasks]
    return task_items


def _build_task_items(tasks_data: dict) -> list:
    items = []
    for key, tdata in tasks_data.items():
        tid = key.replace("eval_", "") if key.startswith("eval_") else key
        status = classify_task(tdata)
        err = tdata.get("prediction_error", 1.0)
        items.append((key, tid, tdata, status, err))
    return items


def _generate_split(title, source_name, tasks_data, task_map, env,
                    index_path, tasks_dir, tasks_dir_name,
                    filter_status="", max_tasks=0, back_link_prefix=""):
    """Generate one split (train or eval): index file + task detail pages."""
    task_items = _sort_and_filter(_build_task_items(tasks_data),
                                 filter_status, max_tasks)

    os.makedirs(tasks_dir, exist_ok=True)

    # Per-task detail pages
    back_link = f"../{back_link_prefix}" if back_link_prefix else "../index.html"
    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue
        page = _generate_task_page(tid, tdata, task, env, back_link=back_link)
        with open(os.path.join(tasks_dir, f"{tid}.html"), "w") as f:
            f.write(page)

    # Index page
    index_html = _generate_index(title, source_name, task_items,
                                 task_map, tasks_dir_name, env)
    with open(index_path, "w") as f:
        f.write(index_html)

    return index_path, len(task_items)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def generate_html(results_path: str, output_base: str,
                  filter_status: str = "", max_tasks: int = 0) -> str:
    """Generate HTML visualization.

    For pipeline results (train + eval), generates separate train/eval files:
      <output_base>_train_viz.html + <output_base>_train_viz/
      <output_base>_eval_viz.html  + <output_base>_eval_viz/

    For single-run results, generates:
      <output_base>_viz.html + <output_base>_viz/

    Args:
        results_path: path to results JSON
        output_base: base path prefix (without _viz suffix)

    Returns:
        first index path generated
    """
    with open(results_path) as f:
        results = json.load(f)

    # Load actual ARC tasks for grids
    task_map: dict = {}
    for split in ("training", "evaluation"):
        data_dir = find_arc_data(split)
        if data_dir:
            for t in load_arc_dataset(data_dir):
                task_map[t.task_id] = t

    env = ARCEnv()
    source_name = os.path.basename(results_path)
    first_index = None

    if "train_tasks" in results and "eval_tasks" in results:
        # Pipeline: separate train and eval
        for split, split_label in [("train", "Training"), ("eval", "Evaluation")]:
            tasks_key = f"{split}_tasks"
            tasks_data = results.get(tasks_key, {})
            if not tasks_data:
                continue

            suffix = f"_{split}_viz"
            index_path = f"{output_base}{suffix}.html"
            tasks_dir = f"{output_base}{suffix}"
            # Back link from task page to index: ../prefix_split_viz.html
            back_link_name = os.path.basename(index_path)

            idx, n = _generate_split(
                title=f"ARC-AGI {split_label} Results",
                source_name=source_name,
                tasks_data=tasks_data,
                task_map=task_map,
                env=env,
                index_path=index_path,
                tasks_dir=tasks_dir,
                tasks_dir_name=os.path.basename(tasks_dir),
                filter_status=filter_status,
                max_tasks=max_tasks,
                back_link_prefix=back_link_name,
            )
            print(f"  {split_label}: {idx} ({n} tasks)")
            if first_index is None:
                first_index = idx
    else:
        # Single run
        tasks_data = results.get("tasks", {})
        index_path = f"{output_base}_viz.html"
        tasks_dir = f"{output_base}_viz"
        back_link_name = os.path.basename(index_path)

        idx, n = _generate_split(
            title="ARC-AGI Results",
            source_name=source_name,
            tasks_data=tasks_data,
            task_map=task_map,
            env=env,
            index_path=index_path,
            tasks_dir=tasks_dir,
            tasks_dir_name=os.path.basename(tasks_dir),
            filter_status=filter_status,
            max_tasks=max_tasks,
            back_link_prefix=back_link_name,
        )
        print(f"  Results: {idx} ({n} tasks)")
        first_index = idx

    return first_index or ""


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ARC-AGI results as HTML")
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output base path (default: derived from results path)")
    parser.add_argument("--filter",
                        choices=["solved", "overfit", "near-miss", "unsolved"],
                        default="", help="Show only tasks with this status")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Limit number of tasks shown")
    args = parser.parse_args()

    output_base = args.output or os.path.splitext(args.results_json)[0]

    generate_html(args.results_json, output_base,
                  filter_status=args.filter, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
