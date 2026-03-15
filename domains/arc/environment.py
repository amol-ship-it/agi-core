"""
ARC-AGI Environment: execute grid transformation programs.

The core execution engine. Programs are trees of primitives; _eval_tree
interprets them recursively on input grids.

Execution model:
  - Perception nodes (kind="perception"): call fn(grid) → scalar value
  - Parameterized nodes (kind="parameterized"): evaluate perception children
    as values, call factory(*values) → transform, apply transform(grid)
  - Transform nodes (arity 0): call fn(grid) directly
  - Transform nodes (arity 1): evaluate child grid, apply fn(child_result)
  - Transform nodes (arity 2): evaluate both children, apply fn(left, right)

Structural search strategies:
  - Object decomposition: try applying primitives per-object
  - Cross-reference: one grid part informs another
  - Color fix: learn color remapping from near-miss programs
  These are search strategies, not vocabulary choices — they compose
  existing atomic primitives in structurally different ways.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from core import Environment, Primitive, Program, Task, Observation
from core.types import ScoredProgram
from .primitives import Grid, _PRIM_MAP


class ARCEnv(Environment):
    """ARC-AGI environment: execute programs on grids."""

    def __init__(self):
        self._current_task: Optional[Task] = None

    def load_task(self, task: Task) -> Observation:
        self._current_task = task
        return Observation(
            data=[inp for inp, _ in task.train_examples],
            metadata={"task_id": task.task_id},
        )

    def execute(self, program: Program, input_data: Any) -> Any:
        """Execute the program tree on an input grid."""
        return self._eval_tree(program, input_data)

    def reset(self):
        self._current_task = None

    def register_primitive(self, primitive) -> None:
        """Register a dynamically created primitive for ARC execution."""
        _PRIM_MAP[primitive.name] = primitive

    # --- Structural search strategies ---
    # These are search strategies, not vocabulary choices. They compose
    # existing atomic primitives in structurally different ways.

    def try_object_decomposition(
        self, task: Task, primitives: list[Primitive],
    ) -> Optional[tuple[str, Any]]:
        """Try solving a task by applying the same transform per object."""
        from .objects import try_object_decomposition as _try_obj_decomp
        result = _try_obj_decomp(task.train_examples, primitives)
        if result is None:
            return None
        name, fn = result
        # Register as a dynamic primitive so execute() can find it
        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
        self.register_primitive(prim)
        return (name, fn)

    def try_for_each_object(
        self, task: Task, candidate_programs: list[ScoredProgram],
        top_k: int = 10,
    ) -> Optional[tuple[str, Any]]:
        """Try applying top-K candidate programs per-object."""
        from .objects import (
            apply_transform_per_object, apply_transform_per_multicolor_object,
            _get_background_color, _test_on_examples,
        )
        if not task.train_examples:
            return None
        # Only same-dims tasks
        for inp, out in task.train_examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        bg_color = _get_background_color(task.train_examples[0][0])
        # Sort candidates by quality, take top_k
        sorted_cands = sorted(candidate_programs,
                              key=lambda s: s.prediction_error)[:top_k]

        for sp in sorted_cands:
            prog = sp.program
            env = self

            def _make_per_obj_fn(p=prog, e=env, bg=bg_color):
                def transform(subgrid):
                    return e._eval_tree(p, subgrid)
                def fn(grid):
                    result = apply_transform_per_object(grid, transform, bg)
                    return result if result is not None else grid
                return fn

            per_obj_fn = _make_per_obj_fn()
            if _test_on_examples(per_obj_fn, task.train_examples):
                name = f"for_each_object({repr(prog)})"
                prim = Primitive(name=name, arity=0, fn=per_obj_fn, domain="arc")
                self.register_primitive(prim)
                return (name, per_obj_fn)

            # Also try per multi-color object
            def _make_mc_fn(p=prog, e=env, bg=bg_color):
                def transform(subgrid):
                    return e._eval_tree(p, subgrid)
                def fn(grid):
                    result = apply_transform_per_multicolor_object(
                        grid, transform, bg)
                    return result if result is not None else grid
                return fn

            mc_fn = _make_mc_fn()
            if _test_on_examples(mc_fn, task.train_examples):
                name = f"for_each_mc_object({repr(prog)})"
                prim = Primitive(name=name, arity=0, fn=mc_fn, domain="arc")
                self.register_primitive(prim)
                return (name, mc_fn)

        return None

    def try_conditional_per_object(
        self, task: Task, candidate_programs: list[ScoredProgram],
        predicates: list, top_k: int = 8,
    ) -> Optional[tuple[str, Any]]:
        """Try if(pred, A, B) per object.

        For each predicate × (branch_A, branch_B) from top candidates,
        apply branch_A to objects where pred is true, branch_B otherwise.
        """
        from .objects import (
            find_foreground_shapes, place_subgrid,
            _get_background_color, _test_on_examples,
        )
        if not task.train_examples or not predicates:
            return None
        # Only same-dims tasks
        for inp, out in task.train_examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        bg_color = _get_background_color(task.train_examples[0][0])
        sorted_cands = sorted(candidate_programs,
                              key=lambda s: s.prediction_error)[:top_k]
        env = self

        for pred_name, pred_fn in predicates:
            for i, sp_a in enumerate(sorted_cands):
                for sp_b in sorted_cands[i + 1:]:
                    prog_a, prog_b = sp_a.program, sp_b.program

                    def _make_cond_fn(pa=prog_a, pb=prog_b, pf=pred_fn,
                                      e=env, bg=bg_color):
                        def fn(grid):
                            shapes = find_foreground_shapes(grid)
                            if not shapes:
                                return grid
                            h, w = len(grid), len(grid[0]) if grid else 0
                            canvas = [[bg] * w for _ in range(h)]
                            for shape in shapes:
                                sg = shape["subgrid"]
                                try:
                                    use_a = pf(sg)
                                except Exception:
                                    use_a = False
                                prog = pa if use_a else pb
                                try:
                                    transformed = e._eval_tree(prog, sg)
                                    if not isinstance(transformed, list):
                                        transformed = sg
                                except Exception:
                                    transformed = sg
                                canvas = place_subgrid(
                                    canvas, transformed, shape["position"],
                                    transparent_color=bg)
                            return canvas
                        return fn

                    fn = _make_cond_fn()
                    if _test_on_examples(fn, task.train_examples):
                        name = f"cond_per_obj({pred_name},{repr(prog_a)},{repr(prog_b)})"
                        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                        self.register_primitive(prim)
                        return (name, fn)

                    # Try swapped branches
                    fn2 = _make_cond_fn(pa=prog_b, pb=prog_a)
                    if _test_on_examples(fn2, task.train_examples):
                        name = f"cond_per_obj({pred_name},{repr(prog_b)},{repr(prog_a)})"
                        prim = Primitive(name=name, arity=0, fn=fn2, domain="arc")
                        self.register_primitive(prim)
                        return (name, fn2)

        return None

    def try_cross_reference(
        self, task: Task, primitives: list[Primitive],
    ) -> Optional[tuple[str, Any]]:
        """Try cross-reference: one grid part informs another.

        Strategies:
        1. Split grid by separator lines, use one cell to transform others
        2. Boolean ops on grid halves (AND, OR, XOR)
        3. Small grid stamps onto large grid
        """
        if not task.train_examples:
            return None
        # Strategy 1: Boolean ops on grid halves (same dims required)
        result = self._try_boolean_halves(task)
        if result is not None:
            return result
        # Strategy 2: Separator-based cross-reference
        result = self._try_separator_cross_ref(task)
        if result is not None:
            return result
        return None

    def _try_boolean_halves(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try boolean ops between grid halves."""
        from .objects import _test_on_examples
        examples = task.train_examples
        first_inp = examples[0][0]
        first_out = examples[0][1]
        h, w = len(first_inp), len(first_inp[0]) if first_inp else 0
        oh, ow = len(first_out), len(first_out[0]) if first_out else 0

        splits = []
        if h == oh * 2 and w == ow:
            # Vertical split: top/bottom halves → output-sized
            splits.append(("vsplit", lambda g: (
                [g[r][:] for r in range(len(g) // 2)],
                [g[r][:] for r in range(len(g) // 2, len(g))])))
        if w == ow * 2 and h == oh:
            # Horizontal split: left/right halves → output-sized
            splits.append(("hsplit", lambda g: (
                [row[:len(row) // 2] for row in g],
                [row[len(row) // 2:] for row in g])))

        if not splits:
            return None

        ops = [
            ("xor", lambda a, b, h, w: [
                [int(a[r][c] != 0) ^ int(b[r][c] != 0)
                 for c in range(w)] for r in range(h)]),
            ("or", lambda a, b, h, w: [
                [a[r][c] if a[r][c] != 0 else b[r][c]
                 for c in range(w)] for r in range(h)]),
            ("and", lambda a, b, h, w: [
                [a[r][c] if b[r][c] != 0 else 0
                 for c in range(w)] for r in range(h)]),
            ("a_minus_b", lambda a, b, h, w: [
                [a[r][c] if b[r][c] == 0 else 0
                 for c in range(w)] for r in range(h)]),
            ("b_minus_a", lambda a, b, h, w: [
                [b[r][c] if a[r][c] == 0 else 0
                 for c in range(w)] for r in range(h)]),
        ]

        for split_name, split_fn in splits:
            for op_name, op_fn in ops:
                def _make_fn(sf=split_fn, of=op_fn):
                    def fn(grid):
                        a, b = sf(grid)
                        rh = min(len(a), len(b))
                        rw = min(len(a[0]) if a else 0, len(b[0]) if b else 0)
                        return of(a, b, rh, rw)
                    return fn
                fn = _make_fn()
                if _test_on_examples(fn, examples):
                    name = f"cross_ref({split_name}_{op_name})"
                    prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                    self.register_primitive(prim)
                    return (name, fn)

        return None

    def _try_separator_cross_ref(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try separator-based cross-reference."""
        from .objects import _test_on_examples
        examples = task.train_examples
        first_inp = examples[0][0]
        h, w = len(first_inp), len(first_inp[0]) if first_inp else 0
        if h < 3 or w < 3:
            return None

        # Detect horizontal separator lines (entire row same non-zero color)
        def _find_h_separators(grid):
            seps = []
            for r in range(len(grid)):
                row = grid[r]
                if len(set(row)) == 1 and row[0] != 0:
                    seps.append(r)
            return seps

        # Detect vertical separator lines
        def _find_v_separators(grid):
            seps = []
            h = len(grid)
            w = len(grid[0]) if grid else 0
            for c in range(w):
                col = [grid[r][c] for r in range(h)]
                if len(set(col)) == 1 and col[0] != 0:
                    seps.append(c)
            return seps

        h_seps = _find_h_separators(first_inp)
        v_seps = _find_v_separators(first_inp)

        # Check consistency across examples
        for inp, _ in examples[1:]:
            if _find_h_separators(inp) != h_seps:
                h_seps = []
            if _find_v_separators(inp) != v_seps:
                v_seps = []

        if not h_seps and not v_seps:
            return None

        # Try: output = one of the grid cells
        # Split by separators into cells
        def _split_into_cells(grid, h_seps, v_seps):
            h = len(grid)
            w = len(grid[0]) if grid else 0
            row_bounds = [0] + [s + 1 for s in h_seps] + [h]
            col_bounds = [0] + [s + 1 for s in v_seps] + [w]
            cells = []
            for i in range(len(row_bounds) - 1):
                row_cells = []
                for j in range(len(col_bounds) - 1):
                    r0, r1 = row_bounds[i], row_bounds[i + 1]
                    c0, c1 = col_bounds[j], col_bounds[j + 1]
                    # Skip separator rows/cols
                    if i > 0:
                        r0 = max(r0, h_seps[i - 1] + 1) if i - 1 < len(h_seps) else r0
                    if j > 0:
                        c0 = max(c0, v_seps[j - 1] + 1) if j - 1 < len(v_seps) else c0
                    cell = [grid[r][c0:c1] for r in range(r0, r1)]
                    if cell and cell[0]:
                        row_cells.append(cell)
                if row_cells:
                    cells.append(row_cells)
            return cells

        cells = _split_into_cells(first_inp, h_seps, v_seps)
        if not cells or not cells[0]:
            return None

        # Check if output matches any single cell
        first_out = examples[0][1]
        for ri, row in enumerate(cells):
            for ci, cell in enumerate(row):
                if cell == first_out:
                    # Verify across all examples
                    def _make_extract_fn(r_idx=ri, c_idx=ci,
                                         hs=h_seps, vs=v_seps):
                        def fn(grid):
                            cs = _split_into_cells(grid, hs, vs)
                            if r_idx < len(cs) and c_idx < len(cs[r_idx]):
                                return cs[r_idx][c_idx]
                            return grid
                        return fn
                    fn = _make_extract_fn()
                    if _test_on_examples(fn, examples):
                        name = f"cross_ref(extract_cell_{ri}_{ci})"
                        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                        self.register_primitive(prim)
                        return (name, fn)

        return None

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        **kwargs,
    ) -> Optional[Program]:
        """Learn a color remapping from near-miss outputs to expected outputs.

        For each (predicted, expected) pair, computes a consistent pixel-level
        color mapping. If a single mapping works across all examples, returns
        a Program node that applies it.
        """
        if len(program_outputs) != len(expected_outputs):
            return None

        # Build color mapping from predicted → expected
        color_map: dict[int, int] = {}
        for pred, exp in zip(program_outputs, expected_outputs):
            if not isinstance(pred, list) or not isinstance(exp, list):
                return None
            if len(pred) != len(exp):
                return None
            for r in range(len(pred)):
                if not pred[r] or not exp[r]:
                    return None
                if len(pred[r]) != len(exp[r]):
                    return None
                for c in range(len(pred[r])):
                    p_val, e_val = pred[r][c], exp[r][c]
                    if p_val in color_map:
                        if color_map[p_val] != e_val:
                            return None  # inconsistent mapping
                    else:
                        color_map[p_val] = e_val

        # Check if mapping is non-trivial (at least one color changes)
        if all(k == v for k, v in color_map.items()):
            return None

        # Create a color remap function
        def _make_remap(cmap=color_map):
            def remap(grid):
                return [[cmap.get(cell, cell) for cell in row] for row in grid]
            return remap

        remap_fn = _make_remap()
        name = f"color_remap({dict(sorted(color_map.items()))})"
        prim = Primitive(name=name, arity=1, fn=remap_fn, domain="arc")
        self.register_primitive(prim)
        return Program(root=name)

    # Maximum intermediate grid size (pixels). Guards against runaway expansion
    # from composed grid-expanding primitives (tile_3x3=9x, scale_5x=25x).
    MAX_GRID_PIXELS = 10_000  # ~100x100 — generous for ARC (max 30x30)

    def _eval_tree(self, node: Program, grid: Grid):
        """Recursively evaluate a program tree on a grid.

        Returns a Grid for transform/parameterized nodes, or a scalar
        value (int) for perception nodes.
        """
        prim = _PRIM_MAP.get(node.root)
        if prim is None:
            return grid

        try:
            # --- Perception: Grid → Value ---
            if prim.kind == "perception":
                return prim.fn(grid)

            # --- Parameterized: evaluate perception children, build transform ---
            if prim.kind == "parameterized":
                params = []
                for child in node.children:
                    child_prim = _PRIM_MAP.get(child.root)
                    if child_prim and child_prim.kind == "perception":
                        params.append(child_prim.fn(grid))
                    else:
                        val = self._eval_tree(child, grid)
                        params.append(val)
                transform_fn = prim.fn(*params)
                if callable(transform_fn):
                    result = transform_fn(grid)
                    if isinstance(result, list) and result:
                        return result
                return grid

            # --- Transform: Grid → Grid ---
            if prim.arity == 0:
                if isinstance(prim.fn, Program):
                    return self._eval_tree(prim.fn, grid)
                elif callable(prim.fn):
                    result = prim.fn(grid)
                    if not isinstance(result, list) or not result:
                        return grid
                    return result
                return grid
            elif prim.arity == 1:
                child_grid = self._eval_tree(node.children[0], grid) if node.children else grid
                if not isinstance(child_grid, list):
                    return grid  # child was perception, not a grid
                h = len(child_grid)
                w = len(child_grid[0]) if child_grid else 0
                if h * w > self.MAX_GRID_PIXELS:
                    return grid
                result = prim.fn(child_grid)
                if not isinstance(result, list) or not result:
                    return grid
                rh, rw = len(result), len(result[0]) if result else 0
                if rh * rw > self.MAX_GRID_PIXELS:
                    return grid
                return result
            elif prim.arity == 2:
                left = self._eval_tree(node.children[0], grid) if len(node.children) > 0 else grid
                right = self._eval_tree(node.children[1], grid) if len(node.children) > 1 else grid
                result = prim.fn(left, right)
                if not isinstance(result, list) or not result:
                    return grid
                return result
        except Exception:
            return grid

        return grid
