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
from domains.arc.dataset import find_arc_data


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
.step-prim-name.perception { color:#7FDBFF; }
.step-prim-name.parameterized { color:#F012BE; }
.perception-value { background:#16213e; border:1px solid #333; border-radius:6px;
                    padding:12px 18px; text-align:center; min-width:60px; }
.perception-value .label { font-size:0.75em; color:#999; margin-bottom:4px; }
.perception-value .value { font-size:1.4em; font-weight:bold; color:#7FDBFF; }
.color-swatch { display:inline-block; width:16px; height:16px; border-radius:3px;
                border:1px solid #555; vertical-align:middle; margin-left:4px; }
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


EXPLANATION_CSS = """
.program-explanation { background:#1a2744; border:1px solid #334; border-radius:6px;
    padding:12px 16px; margin:8px 0; font-size:0.85em; line-height:1.5; }
.program-explanation h3 { color:#FF851B; margin:0 0 8px 0; font-size:0.95em; }
.program-explanation .rule-table { width:100%; border-collapse:collapse; margin:4px 0; }
.program-explanation .rule-table td { padding:2px 8px; border-bottom:1px solid #222; }
.program-explanation .rule-table td:first-child { color:#7FDBFF; font-weight:bold; }
.program-explanation code { background:#0f3460; padding:2px 6px; border-radius:3px; }
"""


def _explain_program(prog_str: str, env=None) -> str:
    """Generate a human-readable HTML explanation of a program.

    Handles dynamic primitives (per_pixel_stamp, half_colormap, etc.)
    by describing what they do and showing learned rules when available.
    """
    from domains.arc.primitives import _PRIM_MAP

    parts: list[str] = []

    # Extract the root primitive name
    root = prog_str.split("(")[0] if "(" in prog_str else prog_str

    # Look up the primitive to get its description
    prim = _PRIM_MAP.get(prog_str) or _PRIM_MAP.get(root)
    if prim and prim.description:
        parts.append(f'<p>{html.escape(prim.description)}</p>')

    # Pattern-based explanations for dynamic primitives
    explanations = {
        "half_colormap": (
            "Split the input grid in half (horizontally or vertically, "
            "possibly by a separator line), then learn a pixel-level color mapping: "
            "for each position, map <code>(left_pixel, right_pixel)</code> → "
            "<code>output_pixel</code>. The mapping is learned from training examples "
            "and LOOCV-validated."
        ),
        "nway_colormap": (
            "Split the input grid into 3+ sections (by separator lines), "
            "then learn a mapping from the tuple of corresponding pixels across "
            "all sections to the output pixel color."
        ),
        "quad_colormap": (
            "Split the input grid into 2×2 quadrants, then learn a mapping "
            "from <code>(top_left, top_right, bottom_left, bottom_right)</code> → "
            "<code>output_pixel</code>."
        ),
        "transform_colormap": (
            "Apply a transform T to the input, then learn a mapping "
            "<code>(original_pixel, T(original)_pixel)</code> → "
            "<code>output_pixel</code>. The transform reveals structure "
            "that the color mapping can exploit."
        ),
        "per_pixel_stamp": (
            "Each non-zero pixel in the input acts as an anchor. A stamp pattern "
            "is learned: <code>(anchor_color, Δrow, Δcol)</code> → "
            "<code>fill_color</code>. The pattern is stamped around each anchor, "
            "filling only zero-valued positions."
        ),
        "per_object_recolor": (
            "Detect connected components (objects) in the input. Learn a "
            "property-based rule that maps each object's property to a new color. "
            "Properties tried: size, shape, compactness, has_hole, position, etc."
        ),
        "cond_bbox_fill": (
            "For each connected component, optionally fill zeros inside its "
            "bounding box with a learned color. Which objects get filled is "
            "determined by a property (compactness, has_hole, etc.)."
        ),
        "procedural": (
            "Pixel-diff engine: compute what changed between input and output, "
            "attribute changes to objects, match action templates (fill_bbox, "
            "extend_ray, fill_between, etc.), learn property→action mapping."
        ),
        "procedural_move": (
            "Match objects between input and output by shape signature, "
            "learn displacement rules: each object moves by a (Δrow, Δcol) "
            "determined by its properties (color, size, etc.)."
        ),
        "procedural_extract": (
            "Extract a subgrid from the input at the position of a selected object. "
            "The object is identified by a property (is_largest, has_hole, unique_color, etc.)."
        ),
        "pixel_to_tile": (
            "Upscale: each input pixel maps to a k×k output tile. "
            "The tile pattern is determined by the pixel's color."
        ),
        "compact_local_rule": (
            "Cellular automaton: <code>(center_pixel, n_nonzero_4neighbors, "
            "majority_4neighbor_color)</code> → <code>output_pixel</code>. "
            "Learned from training, LOOCV-validated."
        ),
        "count_local_rule": (
            "Cellular automaton: <code>(center_pixel, n_nonzero_8neighbors)</code> "
            "→ <code>output_pixel</code>."
        ),
        "raw3x3_local_rule": (
            "Cellular automaton using the full raw 3×3 neighborhood (9 pixels) "
            "as the rule key → <code>output_pixel</code>."
        ),
        "pos_mod": (
            "Position-modular rule: <code>(center_pixel, row % period, "
            "col % period)</code> → <code>output_pixel</code>. "
            "Captures periodic position-dependent patterns."
        ),
        "ncolors_local_rule": (
            "Neighborhood diversity rule: <code>(center_pixel, "
            "n_distinct_4neighbor_colors)</code> → <code>output_pixel</code>."
        ),
        "input_pred_correct": (
            "Two-stage correction: first apply a base program, then correct "
            "each pixel using <code>(original_input_pixel, predicted_pixel)</code> "
            "→ <code>corrected_pixel</code>. LOOCV-validated."
        ),
        "color_remap": (
            "Global color remapping: learn a consistent "
            "<code>predicted_color → corrected_color</code> mapping "
            "from near-miss program outputs."
        ),
        "cell_patch": (
            "Fixed position-based correction: for specific <code>(row, col)</code> "
            "positions where the prediction differs from expected, apply a "
            "learned patch. Only works when patches are consistent across examples."
        ),
        "cross_ref": (
            "Split the grid in half (horizontally or vertically) and apply a "
            "boolean operation (XOR, AND, OR, subtract) between the two halves."
        ),
        "extract_unique_quadrant": (
            "Split the grid by separator lines into sections, find the one "
            "section that differs from the majority, and extract it."
        ),
        "overlay_all_sections": (
            "Split the grid by separator lines into sections, overlay all "
            "sections (non-zero pixels from any section win)."
        ),
        "recolor_markers_by_nearest_sep": (
            "Recolor each non-separator pixel to the color of the nearest "
            "separator line (horizontal or vertical)."
        ),
        "slide_markers_to_matching_sep": (
            "Move each colored marker pixel toward the separator line "
            "that matches its color."
        ),
        "if_": (
            "Conditional program: evaluate a predicate on the input grid "
            "(e.g., is_tall, has_symmetry, is_square), then apply one of "
            "two different transforms based on the result."
        ),
    }

    # Match against known patterns
    for pattern, explanation in explanations.items():
        if pattern in prog_str:
            parts.append(f'<p>{explanation}</p>')
            break

    # For learned library entries, show expansion
    # (already handled by _format_expanded_program)

    # Show learned rules if available (from in-memory or task data)
    rule_data = None
    try:
        from domains.arc.primitives import _PRIM_RULES
        rule_data = _PRIM_RULES.get(prog_str)
    except ImportError:
        pass
    if rule_data:
        escaped = html.escape(rule_data)
        parts.append(f'<details><summary>Learned rules (click to expand)</summary>'
                     f'<pre style="color:#7FDBFF;font-size:0.85em;margin:4px 0;">'
                     f'{escaped}</pre></details>')

    if not parts:
        return ""

    return (f'<div class="program-explanation">'
            f'<h3>How this program works</h3>'
            f'{"".join(parts)}</div>')


def _reconstruct_rules(prog_str: str, task) -> str:
    """Reconstruct learned rules from task data for display purposes.

    Re-analyzes the training examples to show what mapping was learned.
    """
    if not task or not task.train_examples:
        return ""

    examples = task.train_examples

    # half_colormap: reconstruct the (both_nz, a, b) → output mapping
    if "half_colormap" in prog_str:
        fi, fo = examples[0]
        h, w = len(fi), len(fi[0])
        oh, ow = len(fo), len(fo[0])
        # Determine split type from name
        is_sep = "sep" in prog_str
        is_v = "vsplit" in prog_str
        if is_v:
            sec_h = oh
            if h == sec_h * 2 + (1 if is_sep else 0) and w == ow:
                sep = 1 if is_sep else 0
                rule = {}
                for inp, out in examples:
                    a = [inp[r][:] for r in range(sec_h)]
                    b = [inp[r][:] for r in range(sec_h + sep, 2 * sec_h + sep)]
                    for r in range(min(sec_h, len(out))):
                        for c in range(min(ow, len(out[0]))):
                            key = (int(a[r][c] != 0 and b[r][c] != 0), a[r][c], b[r][c])
                            rule[key] = out[r][c]
                lines = ["Color mapping (both_nonzero, top_half, bottom_half) → output:"]
                for k, v in sorted(rule.items()):
                    lines.append(f"  ({k[0]}, {k[1]}, {k[2]}) → {v}")
                return "\n".join(lines)
        elif "hsplit" in prog_str:
            sec_w = ow
            if w == sec_w * 2 + (1 if is_sep else 0) and h == oh:
                sep = 1 if is_sep else 0
                rule = {}
                for inp, out in examples:
                    a = [row[:sec_w] for row in inp]
                    b = [row[sec_w + sep:] for row in inp]
                    for r in range(min(oh, len(out))):
                        for c in range(min(sec_w, len(out[0]))):
                            key = (int(a[r][c] != 0 and b[r][c] != 0), a[r][c], b[r][c])
                            rule[key] = out[r][c]
                lines = ["Color mapping (both_nonzero, left_half, right_half) → output:"]
                for k, v in sorted(rule.items()):
                    lines.append(f"  ({k[0]}, {k[1]}, {k[2]}) → {v}")
                return "\n".join(lines)

    # per_pixel_stamp: reconstruct (color, dr, dc) → fill mapping
    if "per_pixel_stamp" in prog_str:
        rule = {}
        for inp, out in examples:
            h, w = len(inp), len(inp[0])
            sources = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != 0]
            for r in range(h):
                for c in range(w):
                    if inp[r][c] == 0 and out[r][c] != 0:
                        best_d, best_src = float('inf'), None
                        for sr, sc, sc_color in sources:
                            d = abs(r - sr) + abs(c - sc)
                            if d < best_d:
                                best_d = d
                                best_src = (sr, sc, sc_color)
                        if best_src:
                            key = (best_src[2], r - best_src[0], c - best_src[1])
                            rule[key] = out[r][c]
        if rule:
            lines = ["Stamp pattern (source_color, Δrow, Δcol) → fill_color:"]
            for (sc, dr, dc), fc in sorted(rule.items()):
                lines.append(f"  color {sc} + offset ({dr:+d}, {dc:+d}) → {fc}")
            return "\n".join(lines)

    # per_object_recolor: show strategy name from program
    if "per_object_recolor" in prog_str:
        strategy = prog_str.split("(")[1].rstrip(")") if "(" in prog_str else "unknown"
        return f"Recolor strategy: {strategy}\nEach object's color is mapped based on its {strategy} property."

    # Local rules: reconstruct from examples
    if "local_rule" in prog_str or "pos_mod" in prog_str:
        return f"Learned cellular automaton rule from training examples.\nRule type: {prog_str}"

    # input_pred_correct: show base program
    if "input_pred_correct" in prog_str:
        base = prog_str.replace("input_pred_correct(", "").rstrip(")")
        return (f"Base program: {base}\n"
                f"Correction: (original_pixel, predicted_pixel) → corrected_pixel")

    return ""


def _html_page(title: str, body: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{html.escape(title)}</title>
<style>{SHARED_CSS}{EXPLANATION_CSS}{extra_css}</style>
</head><body>
{body}
</body></html>"""


# --------------------------------------------------------------------------
# Grid rendering
# --------------------------------------------------------------------------

def _safe_grid_array(grid) -> Optional[np.ndarray]:
    """Convert a grid to a 2D numpy array, normalizing inhomogeneous rows.

    ARC grids should be list[list[int]] but malformed predictions may have
    inconsistent row lengths or nested structures. This normalizes them by
    padding short rows with 0 and truncating any deeper nesting.
    """
    if not grid or not isinstance(grid, list):
        return None
    # Flatten any deeper nesting: ensure each row is a flat list of ints
    rows = []
    for row in grid:
        if not isinstance(row, list):
            return None
        flat = []
        for cell in row:
            if isinstance(cell, (int, float, np.integer, np.floating)):
                flat.append(int(cell))
            elif isinstance(cell, (list, tuple)):
                # Nested structure — take first element or 0
                flat.append(int(cell[0]) if cell else 0)
            else:
                flat.append(0)
        rows.append(flat)
    if not rows:
        return None
    # Pad to uniform width
    max_w = max(len(r) for r in rows)
    if max_w == 0:
        return None
    for r in rows:
        while len(r) < max_w:
            r.append(0)
    return np.array(rows, dtype=np.int32)


def render_grid(grid, diff_grid=None, border_class="", cell_size=0) -> str:
    if not grid or not grid[0]:
        return '<div class="grid-wrapper"><em>empty</em></div>'
    arr = _safe_grid_array(grid)
    if arr is None:
        return '<div class="grid-wrapper"><em>malformed grid</em></div>'
    h, w = arr.shape
    if cell_size <= 0:
        cell_size = 20
        if max(h, w) > 15:
            cell_size = max(10, min(20, 300 // max(h, w)))
    bcls = f" {border_class}" if border_class else ""
    parts = [f'<div class="grid-wrapper"><div class="grid{bcls}" '
             f'style="grid-template-columns:repeat({w},{cell_size}px);">']
    diff_arr = _safe_grid_array(diff_grid) if diff_grid is not None else None
    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(arr[r, c]), "#333")
            is_diff = (diff_arr is not None
                       and r < diff_arr.shape[0] and c < diff_arr.shape[1]
                       and int(arr[r, c]) != int(diff_arr[r, c]))
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
    pa = _safe_grid_array(prediction)
    ea = _safe_grid_array(expected)
    if pa is None or ea is None:
        return "diff-border", None
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


def _expand_learned(prog: Program, library_map: dict) -> Program:
    """Recursively expand learned entries inline.

    If prog is `crop_half_left(learned_14)` and library_map has
    `learned_14 -> crop_half_top(crop_to_content)`, returns
    `crop_half_left(crop_half_top(crop_to_content))`.
    """
    if not library_map:
        return prog
    expanded_children = [_expand_learned(c, library_map) for c in prog.children]
    if prog.root in library_map:
        # Replace this node with the library entry's expanded program tree
        inner = _expand_learned(library_map[prog.root], library_map)
        if expanded_children:
            # Learned entry used as outer: learned_14(something)
            # Graft children onto the innermost leaf of the expansion
            return _graft_children(inner, expanded_children)
        return inner
    return Program(root=prog.root, children=expanded_children)


def _graft_children(tree: Program, children: list[Program]) -> Program:
    """Attach children to the innermost (deepest-left) leaf of a tree.

    Used when a learned entry wraps sub-expressions, e.g. learned_14(X)
    where learned_14 = crop_half_top(crop_to_content). Result is
    crop_half_top(crop_to_content(X)).
    """
    if not tree.children:
        return Program(root=tree.root, children=children)
    new_children = list(tree.children)
    new_children[0] = _graft_children(new_children[0], children)
    return Program(root=tree.root, children=new_children)


def _execute_steps(prog: Program, grid, env: ARCEnv,
                   library_map: Optional[dict] = None,
                   ) -> list[dict]:
    """Execute program tree step-by-step.

    Returns list of step dicts:
        {"name": str, "type": "grid"|"perception"|"parameterized",
         "output": grid_or_value, "perception_args": [...]}

    If library_map is provided, learned entries are expanded inline so that
    every primitive in the expansion gets its own step.
    """
    from domains.arc.primitives import _PRIM_MAP

    expanded = _expand_learned(prog, library_map or {})
    steps: list[dict] = []

    def _get_kind(name: str) -> str:
        prim = _PRIM_MAP.get(name)
        return prim.kind if prim else "transform"

    def _eval(node: Program, inp):
        kind = _get_kind(node.root)

        if kind == "perception":
            # Perception: extract value from grid, don't transform
            prim = _PRIM_MAP.get(node.root)
            value = prim.fn(inp) if prim else None
            steps.append({"name": node.root, "type": "perception", "output": value})
            return value

        if kind == "parameterized":
            # Parameterized: evaluate perception children, then apply factory
            perception_args = []
            for child in node.children:
                val = _eval(child, inp)
                perception_args.append(val)
            prim = _PRIM_MAP.get(node.root)
            try:
                transform_fn = prim.fn(*perception_args)
                result = transform_fn(inp) if callable(transform_fn) else inp
                if not isinstance(result, list) or not result:
                    result = inp
            except Exception:
                result = inp
            steps.append({
                "name": node.root, "type": "parameterized",
                "output": result, "perception_args": perception_args,
            })
            return result

        # Transform: evaluate children first, then apply
        child_result = inp
        if node.children:
            for child in node.children:
                child_result = _eval(child, child_result)
                # If child was perception, child_result is a value — use original grid
                if not isinstance(child_result, list):
                    child_result = inp
        leaf = Program(root=node.root)
        try:
            result = env.execute(leaf, child_result)
            if not isinstance(result, list) or not result:
                result = child_result
        except Exception:
            result = child_result
        steps.append({"name": node.root, "type": "grid", "output": result})
        return result

    try:
        _eval(expanded, grid)
    except Exception:
        pass
    return steps


def _build_library_map(results: dict) -> dict[str, Program]:
    """Build a mapping from learned entry names to their Program trees.

    The results JSON stores library entries as:
      {"name": "learned_14", "program": "crop_half_top(crop_to_content)", ...}
    """
    library_map: dict[str, Program] = {}
    for entry in results.get("library", []):
        name = entry.get("name", "")
        prog_str = entry.get("program", "")
        if name and prog_str:
            parsed = parse_program_tree(prog_str)
            if parsed:
                library_map[name] = parsed
    return library_map


def _format_expanded_program(prog_str: str, library_map: dict[str, Program]) -> str:
    """Format a program string with learned entries expanded inline.

    Example: crop_half_left(learned_14)
         --> crop_half_left(learned_14=crop_half_top(crop_to_content))
    """
    if not library_map:
        return prog_str
    prog = parse_program_tree(prog_str)
    if not prog:
        return prog_str

    def _fmt(node: Program) -> str:
        children_str = ""
        if node.children:
            args = ", ".join(_fmt(c) for c in node.children)
            children_str = f"({args})"
        if node.root in library_map:
            inner = library_map[node.root]
            expanded = _fmt(inner)
            return f"{node.root}={expanded}{children_str}"
        return f"{node.root}{children_str}"

    return _fmt(prog)


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

_PERCEPTION_COLOR_PRIMS = {
    "background_color", "dominant_color", "rarest_color", "accent_color",
    "largest_object_color", "smallest_object_color",
}


def _render_perception_value(name: str, value) -> str:
    """Render a perception value, with color swatch if it's a color."""
    val_str = str(value)
    swatch = ""
    if name in _PERCEPTION_COLOR_PRIMS and isinstance(value, int) and 0 <= value <= 9:
        color_hex = ARC_COLORS.get(value, "#000")
        swatch = f' <span class="color-swatch" style="background:{color_hex}"></span>'
    return (f'<div class="perception-value">'
            f'<div class="label">{html.escape(name)}</div>'
            f'<div class="value">{val_str}{swatch}</div></div>')


def _render_derivation(inp, prog, env,
                       library_map: Optional[dict] = None) -> str:
    """Show step-by-step execution: input --prim→ ... --prim→ predicted.

    Shows perception values inline (with color swatches for color prims),
    parameterized prims with their perception arguments, and grid transforms
    with intermediate outputs.
    """
    has_steps = prog.children or prog.root != "identity"
    if not has_steps:
        return ''

    steps = _execute_steps(prog, inp, env, library_map=library_map)
    if not steps:
        return ''

    parts = ['<div class="step-flow">', _grid_with_label(inp, "Input")]
    for step in steps:
        name = step["name"]
        step_type = step["type"]
        output = step["output"]

        if step_type == "perception":
            css_class = "step-prim-name perception"
            parts.append(f'<div class="step-stage">'
                         f'<div class="{css_class}">{html.escape(name)}</div>'
                         f'<div class="arrow">&rarr;</div></div>')
            parts.append(_render_perception_value(name, output))

        elif step_type == "parameterized":
            css_class = "step-prim-name parameterized"
            # Show perception args that fed into this prim
            args_str = ", ".join(str(a) for a in step.get("perception_args", []))
            label = f"{name}({args_str})"
            parts.append(f'<div class="step-stage">'
                         f'<div class="{css_class}">{html.escape(label)}</div>'
                         f'<div class="arrow">&rarr;</div></div>')
            parts.append(_grid_with_label(output, f"after {label}"))

        else:  # grid transform
            parts.append(f'<div class="step-stage">'
                         f'<div class="step-prim-name">{html.escape(name)}</div>'
                         f'<div class="arrow">&rarr;</div></div>')
            parts.append(_grid_with_label(output, f"after {name}"))

    parts.append('</div>')
    return '\n'.join(parts)


# --------------------------------------------------------------------------
# Per-task detail page
# --------------------------------------------------------------------------

def _generate_task_page(tid, tdata, task, env, back_link="../index.html",
                        library_map: Optional[dict] = None) -> str:
    status = classify_task(tdata)
    prog_str = tdata.get("program", "identity") or "identity"
    err = tdata.get("prediction_error", 1.0)
    train_preds = tdata.get("train_predictions")
    test_preds = tdata.get("test_predictions")
    prog = parse_program_tree(prog_str) or Program(root="identity")

    # Show expanded program string when learned entries are present
    display_str = _format_expanded_program(prog_str, library_map or {})

    b: list[str] = []
    b.append(f'<p><a href="{back_link}">&larr; Back to index</a></p>')
    b.append(f'<h1>{html.escape(tid)} <span class="status {status}">'
             f'{status.upper()}</span></h1>')
    b.append(f'<div class="program">Program: {html.escape(display_str)}</div>')
    explanation = _explain_program(prog_str, env)
    if explanation:
        b.append(explanation)
    # Show learned rules: from JSON data, in-memory _PRIM_RULES, or reconstructed
    learned_rules = tdata.get("learned_rules")
    if not learned_rules:
        learned_rules = _reconstruct_rules(prog_str, task)
    if learned_rules:
        escaped_rules = html.escape(learned_rules)
        b.append(f'<details open><summary style="color:#FF851B;cursor:pointer;">'
                 f'Learned rules</summary>'
                 f'<pre style="color:#7FDBFF;font-size:0.85em;margin:4px 0;'
                 f'background:#0f3460;padding:8px;border-radius:4px;">'
                 f'{escaped_rules}</pre></details>')
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
        b.append(_render_derivation(inp, prog, env, library_map=library_map))

    # Test examples
    if task.test_inputs:
        b.append('<h2>Test Examples</h2>')
        for i, test_inp in enumerate(task.test_inputs):
            test_exp = task.test_outputs[i] if i < len(task.test_outputs) else None
            prediction = _get_prediction(test_preds, i, prog, test_inp, env)
            b.append(_render_example_row(f"Test {i+1}", test_inp, test_exp, prediction))
            b.append(_render_derivation(test_inp, prog, env, library_map=library_map))

    return _html_page(f"Task {tid}", '\n'.join(b))


# --------------------------------------------------------------------------
# Index page
# --------------------------------------------------------------------------

def _generate_index(title, source_name, task_items, task_map, tasks_dir_name,
                    env, thumb=10, library_map: Optional[dict] = None) -> str:
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

    # Table of contents: colored task IDs linking to cards below
    status_colors = {"solved": "#2ECC40", "overfit": "#FFDC00",
                     "near-miss": "#FF851B", "unsolved": "#FF4136"}
    b.append('<div style="margin:12px 0">')
    b.append('<div class="section-label">Task Index</div>')
    b.append('<div style="column-count:4;column-gap:16px;margin:6px 0">')
    for key, tid, tdata, status, err in task_items:
        color = status_colors.get(status, "#e0e0e0")
        b.append(f'<div style="font-size:0.82em;margin:1px 0">'
                 f'<span style="color:{color}">&#9679;</span> '
                 f'<a href="#task-{html.escape(tid)}" style="color:{color}">'
                 f'{html.escape(tid)}</a></div>')
    b.append('</div></div>')

    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue

        prog_str = tdata.get("program", "identity") or "identity"
        te = tdata.get("test_error")
        train_preds = tdata.get("train_predictions")
        test_preds = tdata.get("test_predictions")
        prog = parse_program_tree(prog_str) or Program(root="identity")

        b.append(f'<div id="task-{html.escape(tid)}" class="task-card {status}">')
        b.append('<div class="task-card-header">')
        b.append(f'<a href="{tasks_dir_name}/{html.escape(tid)}.html">'
                 f'{html.escape(tid)}</a>')
        b.append(f'<span class="status {status}">{status.upper()}</span>')
        b.append('</div>')

        expanded_str = _format_expanded_program(prog_str, library_map or {})
        prog_display = expanded_str if len(expanded_str) <= 70 else expanded_str[:67] + "..."
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

        # Link to detail page
        detail_url = f"{tasks_dir_name}/{html.escape(tid)}.html"
        b.append(f'<div style="margin-top:8px;font-size:0.85em">'
                 f'<a href="{detail_url}">'
                 f'View full step-by-step transformation details &rarr;</a></div>')

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
                    filter_status="", max_tasks=0, back_link_prefix="",
                    library_map: Optional[dict] = None):
    """Generate one split (train or eval): index file + task detail pages."""
    task_items = _sort_and_filter(_build_task_items(tasks_data),
                                 filter_status, max_tasks)

    os.makedirs(tasks_dir, exist_ok=True)

    # Per-task detail pages — isolate failures so one bad task doesn't
    # kill the entire visualization
    back_link = f"../{back_link_prefix}" if back_link_prefix else "../index.html"
    failed_tasks = []
    for key, tid, tdata, status, err in task_items:
        task = task_map.get(tid)
        if task is None:
            continue
        try:
            page = _generate_task_page(tid, tdata, task, env, back_link=back_link,
                                       library_map=library_map)
            with open(os.path.join(tasks_dir, f"{tid}.html"), "w") as f:
                f.write(page)
        except Exception as e:
            failed_tasks.append((tid, e))
            print(f"  (viz warning: {title} task {tid}: {e})")

    # Index page
    try:
        index_html = _generate_index(title, source_name, task_items,
                                     task_map, tasks_dir_name, env,
                                     library_map=library_map)
        with open(index_path, "w") as f:
            f.write(index_html)
    except Exception as e:
        # Index failed — still return what we generated
        print(f"  (viz warning: {title} index page: {e})")

    if failed_tasks:
        print(f"  (viz: {len(failed_tasks)} task pages skipped in {title})")

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
        list of generated index paths
    """
    with open(results_path) as f:
        results = json.load(f)

    # Build library map for expanding learned abstractions
    library_map = _build_library_map(results)

    # Load actual ARC tasks for grids
    task_map: dict = {}
    for split in ("training", "evaluation"):
        data_dir = find_arc_data(split)
        if data_dir:
            for t in load_arc_dataset(data_dir):
                task_map[t.task_id] = t

    env = ARCEnv()
    source_name = os.path.basename(results_path)
    index_paths = []

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
                library_map=library_map,
            )
            index_paths.append(idx)
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
            library_map=library_map,
        )
        index_paths.append(idx)

    return index_paths


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

    paths = generate_html(args.results_json, output_base,
                          filter_status=args.filter, max_tasks=args.max_tasks)
    for p in paths:
        print(f"  Generated: {p}")


if __name__ == "__main__":
    main()
