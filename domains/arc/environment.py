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

STRIPPED TO CORE: Only execution engine remains. Structural strategies
(per-object, cross-reference, etc.) will be added back when needed.
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

    # --- Color fix / correction ---

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
                            # Inconsistent color mapping — try cell-wise patch
                            return self._try_cell_patch_correction(
                                program_outputs, expected_outputs)
                    else:
                        color_map[p_val] = e_val

        # Check if mapping is non-trivial (at least one color changes)
        if all(k == v for k, v in color_map.items()):
            # Color remap is trivial — fall through to cell-wise patch
            return self._try_cell_patch_correction(
                program_outputs, expected_outputs)

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

    def _try_cell_patch_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
    ) -> Optional[Program]:
        """Learn a cell-wise correction patch for near-miss outputs.

        For same-dims grids where <15% of pixels differ, learn a fixed
        set of (r, c) → value patches that are consistent across all
        training examples.
        """
        if len(program_outputs) != len(expected_outputs):
            return None

        # Validate all pairs are same-dims grids
        for pred, exp in zip(program_outputs, expected_outputs):
            if not isinstance(pred, list) or not isinstance(exp, list):
                return None
            if len(pred) != len(exp):
                return None
            for r in range(len(pred)):
                if not pred[r] or not exp[r] or len(pred[r]) != len(exp[r]):
                    return None

        h = len(program_outputs[0])
        w = len(program_outputs[0][0]) if h > 0 else 0
        total_pixels = h * w
        if total_pixels == 0:
            return None

        # Build per-cell patch
        patch: dict[tuple[int, int], tuple[int, int]] = {}

        for pred, exp in zip(program_outputs, expected_outputs):
            for r in range(len(pred)):
                for c in range(len(pred[r])):
                    if pred[r][c] != exp[r][c]:
                        key = (r, c)
                        fix = (pred[r][c], exp[r][c])
                        if key in patch:
                            if patch[key] != fix:
                                return None
                        else:
                            patch[key] = fix

        if not patch:
            return None
        if len(patch) > total_pixels * 0.15:
            return None

        patch_map = {pos: to_val for pos, (_, to_val) in patch.items()}

        def _make_patch_fn(pm=patch_map):
            def patch_fn(grid):
                result = [row[:] for row in grid]
                for (r, c), val in pm.items():
                    if r < len(result) and c < len(result[0]):
                        result[r][c] = val
                return result
            return patch_fn

        patch_fn = _make_patch_fn()
        name = f"cell_patch({len(patch_map)}_cells)"
        prim = Primitive(name=name, arity=1, fn=patch_fn, domain="arc")
        self.register_primitive(prim)
        return Program(root=name)

    # --- Structural strategies ---

    def try_object_decomposition(
        self, task: Task, primitives: list[Primitive],
    ) -> Optional[tuple[str, Any]]:
        """Try solving a task by applying the same transform per object."""
        from .objects import try_object_decomposition as _try_obj_decomp
        result = _try_obj_decomp(task.train_examples, primitives)
        if result is None:
            return None
        name, fn = result
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
        for inp, out in task.train_examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        bg_color = _get_background_color(task.train_examples[0][0])
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

    def try_cross_reference(
        self, task: Task, primitives: list[Primitive],
    ) -> Optional[tuple[str, Any]]:
        """Try cross-reference: one grid part informs another."""
        if not task.train_examples:
            return None
        result = self._try_boolean_halves(task)
        if result is not None:
            return result
        result = self._try_separator_cross_ref(task)
        if result is not None:
            return result
        result = self._try_scale_tile_detection(task)
        if result is not None:
            return result
        result = self._try_template_stamp(task)
        if result is not None:
            return result
        result = self._try_separator_marker_ops(task)
        if result is not None:
            return result
        result = self._try_subgrid_selection(task)
        if result is not None:
            return result
        result = self._try_half_colormap(task)
        if result is not None:
            return result
        result = self._try_nway_colormap(task)
        if result is not None:
            return result
        result = self._try_quadrant_colormap(task)
        if result is not None:
            return result
        result = self._try_transform_colormap(task)
        if result is not None:
            return result
        result = self._try_per_pixel_stamp(task)
        if result is not None:
            return result
        result = self._try_conditional_bbox_fill(task)
        if result is not None:
            return result
        result = self._try_cell_grid_colormap(task)
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
            splits.append(("vsplit", lambda g: (
                [g[r][:] for r in range(len(g) // 2)],
                [g[r][:] for r in range(len(g) // 2, len(g))])))
        if w == ow * 2 and h == oh:
            splits.append(("hsplit", lambda g: (
                [row[:len(row) // 2] for row in g],
                [row[len(row) // 2:] for row in g])))

        # Also try: split by separator row/col (single-color row/col)
        if w == ow and h == oh * 2 + 1:
            mid = h // 2
            if len(set(first_inp[mid])) == 1:
                splits.append(("vsplit_sep", lambda g: (
                    [g[r][:] for r in range(len(g) // 2)],
                    [g[r][:] for r in range(len(g) // 2 + 1, len(g))])))
        if h == oh and w == ow * 2 + 1:
            mid = w // 2
            if len(set(first_inp[r][mid] for r in range(h))) == 1:
                splits.append(("hsplit_sep", lambda g: (
                    [row[:len(row) // 2] for row in g],
                    [row[len(row) // 2 + 1:] for row in g])))

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
            # Color-preserving XOR: keep the pixel from whichever half has it
            ("xor_color", lambda a, b, h, w: [
                [a[r][c] if a[r][c] != 0 and b[r][c] == 0
                 else b[r][c] if b[r][c] != 0 and a[r][c] == 0
                 else 0
                 for c in range(w)] for r in range(h)]),
            # Diff: where halves disagree (both nonzero but different), keep a
            ("diff_a", lambda a, b, h, w: [
                [a[r][c] if a[r][c] != b[r][c] else 0
                 for c in range(w)] for r in range(h)]),
            # Diff: where halves disagree, keep b
            ("diff_b", lambda a, b, h, w: [
                [b[r][c] if a[r][c] != b[r][c] else 0
                 for c in range(w)] for r in range(h)]),
            # Same: where halves agree, keep value
            ("same", lambda a, b, h, w: [
                [a[r][c] if a[r][c] == b[r][c] else 0
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

    def _try_half_colormap(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Split grid in half, learn (both_nonzero, a_val, b_val) → output mapping."""
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        first_inp, first_out = examples[0]
        h, w = len(first_inp), len(first_inp[0]) if first_inp else 0
        oh, ow = len(first_out), len(first_out[0]) if first_out else 0

        # Helper: make a split function with optional pre-transform
        def _make_split(pre_fn, split_type, sep_idx=None):
            """Build a split function: pre_fn → split."""
            def split_fn(grid):
                t = pre_fn(grid)
                th, tw = len(t), len(t[0])
                if split_type == "v":
                    return ([t[r][:] for r in range(th // 2)],
                            [t[r][:] for r in range(th // 2, th)])
                elif split_type == "h":
                    return ([row[:tw // 2] for row in t],
                            [row[tw // 2:] for row in t])
                elif split_type == "vs":
                    mid = th // 2
                    return ([t[r][:] for r in range(mid)],
                            [t[r][:] for r in range(mid + 1, th)])
                elif split_type == "hs":
                    mid = tw // 2
                    return ([row[:mid] for row in t],
                            [row[mid + 1:] for row in t])
                return (t, t)
            return split_fn

        # Try with original input and also with pre-transforms
        from .transformation_primitives import (
            trim_rows, trim_cols, crop_to_content, fill_enclosed,
        )
        input_variants = [("", lambda g: g, first_inp)]
        for pname, pfn in [("trim_rows_", trim_rows), ("trim_cols_", trim_cols),
                           ("crop_", crop_to_content),
                           ("fill_enclosed_", fill_enclosed)]:
            try:
                ti = pfn(first_inp)
                if isinstance(ti, list) and ti and isinstance(ti[0], list):
                    if (len(ti), len(ti[0])) != (h, w):
                        input_variants.append((pname, pfn, ti))
            except Exception:
                pass

        splits = []
        for prefix, pre_fn, pre_inp in input_variants:
            ph, pw = len(pre_inp), len(pre_inp[0]) if pre_inp else 0
            if ph == oh * 2 and pw == ow:
                splits.append((f"{prefix}vsplit",
                               _make_split(pre_fn, "v")))
            if pw == ow * 2 and ph == oh:
                splits.append((f"{prefix}hsplit",
                               _make_split(pre_fn, "h")))
            if ph == oh * 2 + 1 and pw == ow:
                mid = ph // 2
                if len(set(pre_inp[mid])) == 1:
                    splits.append((f"{prefix}vsplit_sep",
                                   _make_split(pre_fn, "vs")))
            if pw == ow * 2 + 1 and ph == oh:
                mid = pw // 2
                if len(set(pre_inp[r][mid] for r in range(ph))) == 1:
                    splits.append((f"{prefix}hsplit_sep",
                                   _make_split(pre_fn, "hs")))

        for split_name, split_fn in splits:
            color_map: dict[tuple, int] = {}
            consistent = True
            for inp, out in examples:
                a, b = split_fn(inp)
                rh = min(len(a), len(b), len(out))
                rw = min(len(a[0]) if a else 0, len(b[0]) if b else 0,
                         len(out[0]) if out else 0)
                for r in range(rh):
                    for c in range(rw):
                        key = (int(a[r][c] != 0 and b[r][c] != 0), a[r][c], b[r][c])
                        val = out[r][c]
                        if key in color_map and color_map[key] != val:
                            consistent = False
                            break
                        color_map[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break

            if not consistent:
                continue

            def _make_fn(sf=split_fn, cm=dict(color_map)):
                def fn(grid):
                    a, b = sf(grid)
                    rh = min(len(a), len(b))
                    rw = min(len(a[0]) if a else 0, len(b[0]) if b else 0)
                    return [[cm.get((int(a[r][c] != 0 and b[r][c] != 0),
                                     a[r][c], b[r][c]), 0)
                             for c in range(rw)] for r in range(rh)]
                return fn

            fn = _make_fn()
            if _test_on_examples(fn, examples):
                # LOOCV
                loocv_pass = True
                for hold_idx in range(len(examples)):
                    train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                    sub_map: dict[tuple, int] = {}
                    sub_ok = True
                    for inp, out in train_sub:
                        a, b = split_fn(inp)
                        rh = min(len(a), len(b), len(out))
                        rw = min(len(a[0]) if a else 0, len(b[0]) if b else 0,
                                 len(out[0]) if out else 0)
                        for r in range(rh):
                            for c in range(rw):
                                key = (int(a[r][c] != 0 and b[r][c] != 0),
                                       a[r][c], b[r][c])
                                if key in sub_map and sub_map[key] != out[r][c]:
                                    sub_ok = False
                                    break
                                sub_map[key] = out[r][c]
                            if not sub_ok:
                                break
                        if not sub_ok:
                            break
                    if not sub_ok:
                        loocv_pass = False
                        break
                    sub_fn = _make_fn(split_fn, sub_map)
                    held_inp, held_exp = examples[hold_idx]
                    try:
                        if sub_fn(held_inp) != held_exp:
                            loocv_pass = False
                            break
                    except Exception:
                        loocv_pass = False
                        break

                if not loocv_pass:
                    continue

                name = f"half_colormap({split_name})"
                prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                self.register_primitive(prim)
                # Store learned rules for visualization
                from .primitives import _PRIM_RULES
                rules_desc = "Color mapping (both_nz, left, right) → output:\n"
                for (bnz, a, b), out in sorted(color_map.items()):
                    rules_desc += f"  ({bnz}, {a}, {b}) → {out}\n"
                _PRIM_RULES[name] = rules_desc
                return (name, fn)

        return None

    def _try_nway_colormap(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Split grid into 3+ sections, learn pixel-tuple → output mapping."""
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        h, w = len(fi), len(fi[0]) if fi else 0
        oh, ow = len(fo), len(fo[0]) if fo else 0
        if oh == 0 or ow == 0:
            return None

        splits = []
        for n in [3, 4]:
            # Vertical N-way with separators
            sec_h = oh
            total_h = sec_h * n + (n - 1)
            if h == total_h and w == ow:
                sep_positions = [sec_h * i + i for i in range(1, n)]
                all_sep = all(len(set(fi[sp])) == 1 for sp in sep_positions if sp < h)
                if all_sep:
                    def _sf(g, sh=sec_h, nn=n):
                        parts = []
                        for i in range(nn):
                            start = sh * i + i
                            parts.append([g[r][:] for r in range(start, start + sh)])
                        return parts
                    splits.append((f"vsplit{n}", _sf))

            # Vertical N-way without separators
            if h == sec_h * n and w == ow:
                def _sf(g, sh=sec_h, nn=n):
                    return [[g[r][:] for r in range(sh * i, sh * (i + 1))] for i in range(nn)]
                splits.append((f"vsplit{n}_nosep", _sf))

            # Horizontal N-way with separators
            sec_w = ow
            total_w = sec_w * n + (n - 1)
            if w == total_w and h == oh:
                sep_positions = [sec_w * i + i for i in range(1, n)]
                all_sep = all(len(set(fi[r][sp] for r in range(h))) == 1
                              for sp in sep_positions if sp < w)
                if all_sep:
                    def _sf(g, sw=sec_w, nn=n):
                        parts = []
                        for i in range(nn):
                            start = sw * i + i
                            parts.append([row[start:start + sw] for row in g])
                        return parts
                    splits.append((f"hsplit{n}", _sf))

            # Horizontal N-way without separators
            if w == sec_w * n and h == oh:
                def _sf(g, sw=sec_w, nn=n):
                    return [[row[sw * i:sw * (i + 1)] for row in g] for i in range(nn)]
                splits.append((f"hsplit{n}_nosep", _sf))

        for split_name, split_fn in splits:
            color_map: dict[tuple, int] = {}
            consistent = True
            for inp, out in examples:
                try:
                    parts = split_fn(inp)
                except Exception:
                    consistent = False
                    break
                rh = min(len(p) for p in parts)
                rh = min(rh, len(out))
                rw = min((len(p[0]) if p else 0) for p in parts)
                rw = min(rw, len(out[0]) if out else 0)
                for r in range(rh):
                    for c in range(rw):
                        key = tuple(p[r][c] for p in parts)
                        val = out[r][c]
                        if key in color_map and color_map[key] != val:
                            consistent = False
                            break
                        color_map[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break

            if not consistent:
                continue

            def _make_fn(sf=split_fn, cm=dict(color_map)):
                def fn(grid):
                    parts = sf(grid)
                    rh = min(len(p) for p in parts)
                    rw = min((len(p[0]) if p else 0) for p in parts)
                    return [[cm.get(tuple(p[r][c] for p in parts), 0)
                             for c in range(rw)] for r in range(rh)]
                return fn

            fn = _make_fn()
            if _test_on_examples(fn, examples):
                # LOOCV
                loocv_pass = True
                for hold_idx in range(len(examples)):
                    train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                    sub_map: dict[tuple, int] = {}
                    sub_ok = True
                    for inp, out in train_sub:
                        try:
                            parts = split_fn(inp)
                        except Exception:
                            sub_ok = False
                            break
                        rh = min(len(p) for p in parts)
                        rh = min(rh, len(out))
                        rw = min((len(p[0]) if p else 0) for p in parts)
                        rw = min(rw, len(out[0]) if out else 0)
                        for r in range(rh):
                            for c in range(rw):
                                key = tuple(p[r][c] for p in parts)
                                if key in sub_map and sub_map[key] != out[r][c]:
                                    sub_ok = False
                                    break
                                sub_map[key] = out[r][c]
                            if not sub_ok:
                                break
                        if not sub_ok:
                            break
                    if not sub_ok:
                        loocv_pass = False
                        break
                    sub_fn = _make_fn(split_fn, sub_map)
                    held_inp, held_exp = examples[hold_idx]
                    try:
                        if sub_fn(held_inp) != held_exp:
                            loocv_pass = False
                            break
                    except Exception:
                        loocv_pass = False
                        break

                if not loocv_pass:
                    continue

                name = f"nway_colormap({split_name})"
                prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                self.register_primitive(prim)
                return (name, fn)

        return None

    def _try_quadrant_colormap(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Split grid into 2x2 quadrants, learn 4-pixel-tuple → output mapping."""
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        h, w = len(fi), len(fi[0]) if fi else 0
        oh, ow = len(fo), len(fo[0]) if fo else 0

        configs = []
        if h == oh * 2 and w == ow * 2:
            configs.append(("quad", oh, ow, 0))
        if h == oh * 2 + 1 and w == ow * 2 + 1:
            if len(set(fi[oh])) == 1 and len(set(fi[r][ow] for r in range(h))) == 1:
                configs.append(("quad_sep", oh, ow, 1))

        for cfg_name, sh, sw, sep in configs:
            color_map: dict[tuple, int] = {}
            consistent = True
            for inp, out in examples:
                try:
                    tl = [[inp[r][c] for c in range(sw)] for r in range(sh)]
                    tr = [[inp[r][c] for c in range(sw + sep, 2 * sw + sep)] for r in range(sh)]
                    bl = [[inp[r][c] for c in range(sw)] for r in range(sh + sep, 2 * sh + sep)]
                    br = [[inp[r][c] for c in range(sw + sep, 2 * sw + sep)]
                          for r in range(sh + sep, 2 * sh + sep)]
                except IndexError:
                    consistent = False
                    break
                rh, rw = min(sh, len(out)), min(sw, len(out[0]) if out else 0)
                for r in range(rh):
                    for c in range(rw):
                        key = (tl[r][c], tr[r][c], bl[r][c], br[r][c])
                        val = out[r][c]
                        if key in color_map and color_map[key] != val:
                            consistent = False
                            break
                        color_map[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                continue

            def _make_fn(cm=dict(color_map), s_h=sh, s_w=sw, s=sep):
                def fn(grid):
                    tl = [[grid[r][c] for c in range(s_w)] for r in range(s_h)]
                    tr = [[grid[r][c] for c in range(s_w + s, 2 * s_w + s)] for r in range(s_h)]
                    bl = [[grid[r][c] for c in range(s_w)] for r in range(s_h + s, 2 * s_h + s)]
                    br = [[grid[r][c] for c in range(s_w + s, 2 * s_w + s)]
                          for r in range(s_h + s, 2 * s_h + s)]
                    return [[cm.get((tl[r][c], tr[r][c], bl[r][c], br[r][c]), 0)
                             for c in range(s_w)] for r in range(s_h)]
                return fn

            fn = _make_fn()
            if _test_on_examples(fn, examples):
                # LOOCV
                loocv_pass = True
                for hold_idx in range(len(examples)):
                    train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                    sub_map: dict[tuple, int] = {}
                    sub_ok = True
                    for inp, out in train_sub:
                        try:
                            tl = [[inp[r][c] for c in range(sw)] for r in range(sh)]
                            tr = [[inp[r][c] for c in range(sw + sep, 2 * sw + sep)] for r in range(sh)]
                            bl = [[inp[r][c] for c in range(sw)] for r in range(sh + sep, 2 * sh + sep)]
                            br = [[inp[r][c] for c in range(sw + sep, 2 * sw + sep)]
                                  for r in range(sh + sep, 2 * sh + sep)]
                        except IndexError:
                            sub_ok = False
                            break
                        rh, rw = min(sh, len(out)), min(sw, len(out[0]) if out else 0)
                        for r in range(rh):
                            for c in range(rw):
                                key = (tl[r][c], tr[r][c], bl[r][c], br[r][c])
                                if key in sub_map and sub_map[key] != out[r][c]:
                                    sub_ok = False
                                    break
                                sub_map[key] = out[r][c]
                            if not sub_ok:
                                break
                        if not sub_ok:
                            break
                    if not sub_ok:
                        loocv_pass = False
                        break
                    sub_fn = _make_fn(sub_map, sh, sw, sep)
                    held_inp, held_exp = examples[hold_idx]
                    try:
                        if sub_fn(held_inp) != held_exp:
                            loocv_pass = False
                            break
                    except Exception:
                        loocv_pass = False
                        break
                if not loocv_pass:
                    continue
                name = f"quad_colormap({cfg_name})"
                prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                self.register_primitive(prim)
                return (name, fn)
        return None

    def _try_transform_colormap(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try (input_pixel, transform(input)_pixel) → output_pixel mapping.

        For same-dimension tasks, applies a transform to the input,
        then learns a per-pixel color mapping from (original, transformed) pairs.
        """
        from .objects import _test_on_examples
        from .transformation_primitives import (
            fill_enclosed, dilate, erode, connect_same_color_h,
            connect_same_color_v, flood_fill_by_neighbor,
        )
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        if len(fi) != len(fo) or len(fi[0]) != len(fo[0]):
            return None

        transforms = [
            ("fill_enclosed", fill_enclosed),
            ("dilate", dilate),
            ("erode", erode),
            ("connect_h", connect_same_color_h),
            ("connect_v", connect_same_color_v),
            ("flood_fill", flood_fill_by_neighbor),
        ]

        for tname, tfn in transforms:
            # Check transform produces same dims
            try:
                t0 = tfn(fi)
                if len(t0) != len(fi) or len(t0[0]) != len(fi[0]):
                    continue
            except Exception:
                continue

            color_map: dict[tuple[int, int], int] = {}
            consistent = True
            for inp, out in examples:
                try:
                    t = tfn(inp)
                except Exception:
                    consistent = False
                    break
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        key = (inp[r][c], t[r][c])
                        val = out[r][c]
                        if key in color_map and color_map[key] != val:
                            consistent = False
                            break
                        color_map[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                continue
            # Must be non-trivial
            if all(color_map.get((i, p), i) == i for i, p in color_map):
                continue

            def _make_fn(cm=dict(color_map), tf=tfn):
                def fn(grid):
                    t = tf(grid)
                    return [[cm.get((grid[r][c], t[r][c]), grid[r][c])
                             for c in range(len(grid[0]))] for r in range(len(grid))]
                return fn

            fn = _make_fn()
            if not _test_on_examples(fn, examples):
                continue

            # LOOCV
            loocv_pass = True
            for hold_idx in range(len(examples)):
                sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                sub_map: dict[tuple[int, int], int] = {}
                sub_ok = True
                for inp, out in sub:
                    try:
                        t = tfn(inp)
                    except Exception:
                        sub_ok = False
                        break
                    for r in range(len(inp)):
                        for c in range(len(inp[0])):
                            key = (inp[r][c], t[r][c])
                            if key in sub_map and sub_map[key] != out[r][c]:
                                sub_ok = False
                                break
                            sub_map[key] = out[r][c]
                        if not sub_ok:
                            break
                    if not sub_ok:
                        break
                if not sub_ok:
                    loocv_pass = False
                    break
                sub_fn = _make_fn(sub_map, tfn)
                held_inp, held_exp = examples[hold_idx]
                try:
                    if sub_fn(held_inp) != held_exp:
                        loocv_pass = False
                        break
                except Exception:
                    loocv_pass = False
                    break

            if not loocv_pass:
                continue

            name = f"transform_colormap({tname})"
            prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
            self.register_primitive(prim)
            return (name, fn)

        return None

    def _try_per_pixel_stamp(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Learn a stamp pattern per input pixel color.

        For fill-only tasks where non-zero input pixels stay unchanged,
        learns (source_color, delta_row, delta_col) → fill_color mapping.
        Each non-zero pixel stamps a learned pattern around itself.
        """
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        if len(fi) != len(fo) or len(fi[0]) != len(fo[0]):
            return None

        # Verify fill-only: non-zero pixels don't change
        for inp, out in examples:
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != 0 and inp[r][c] != out[r][c]:
                        return None

        stamp_rule: dict[tuple[int, int, int], int] = {}
        consistent = True
        for inp, out in examples:
            h, w = len(inp), len(inp[0])
            sources = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != 0]
            for r in range(h):
                for c in range(w):
                    if inp[r][c] == 0 and out[r][c] != 0:
                        best_d, best_src = float('inf'), None
                        for sr, sc, scolor in sources:
                            d = abs(r - sr) + abs(c - sc)
                            if d < best_d:
                                best_d = d
                                best_src = (sr, sc, scolor)
                        if best_src is None:
                            consistent = False
                            break
                        key = (best_src[2], r - best_src[0], c - best_src[1])
                        if key in stamp_rule and stamp_rule[key] != out[r][c]:
                            consistent = False
                            break
                        stamp_rule[key] = out[r][c]
                if not consistent:
                    break
            if not consistent:
                break

        if not consistent or not stamp_rule:
            return None

        def _make_fn(rule=dict(stamp_rule)):
            def fn(grid):
                h, w = len(grid), len(grid[0])
                result = [row[:] for row in grid]
                for r in range(h):
                    for c in range(w):
                        if grid[r][c] != 0:
                            for (sc, dr, dc), fc in rule.items():
                                if sc == grid[r][c]:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == 0:
                                        result[nr][nc] = fc
                return result
            return fn

        fn = _make_fn()
        if not _test_on_examples(fn, examples):
            return None

        # LOOCV
        for hold_idx in range(len(examples)):
            train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
            sub_rule: dict[tuple[int, int, int], int] = {}
            sub_ok = True
            for inp, out in train_sub:
                h, w = len(inp), len(inp[0])
                sources = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != 0]
                for r in range(h):
                    for c in range(w):
                        if inp[r][c] == 0 and out[r][c] != 0:
                            best_d, best_src = float('inf'), None
                            for sr, sc, scolor in sources:
                                d = abs(r - sr) + abs(c - sc)
                                if d < best_d:
                                    best_d = d
                                    best_src = (sr, sc, scolor)
                            if best_src is None:
                                sub_ok = False
                                break
                            key = (best_src[2], r - best_src[0], c - best_src[1])
                            if key in sub_rule and sub_rule[key] != out[r][c]:
                                sub_ok = False
                                break
                            sub_rule[key] = out[r][c]
                    if not sub_ok:
                        break
                if not sub_ok:
                    break
            if not sub_ok:
                return None
            sub_fn = _make_fn(sub_rule)
            held_inp, held_exp = examples[hold_idx]
            try:
                if sub_fn(held_inp) != held_exp:
                    return None
            except Exception:
                return None

        name = "per_pixel_stamp"
        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
        self.register_primitive(prim)
        from .primitives import _PRIM_RULES
        rules_desc = "Stamp pattern (source_color, Δrow, Δcol) → fill_color:\n"
        for (sc, dr, dc), fc in sorted(stamp_rule.items()):
            rules_desc += f"  color {sc} + offset ({dr:+d}, {dc:+d}) → {fc}\n"
        _PRIM_RULES[name] = rules_desc
        return (name, fn)

    def _try_conditional_bbox_fill(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Fill object bounding boxes based on object properties.

        For fill-only tasks, learns which objects get their bbox filled
        based on a property (color, has_hole, compactness, etc.).
        """
        from .objects import (
            _test_on_examples, _find_connected_components,
            _has_hole, _compactness,
        )
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        if len(fi) != len(fo) or len(fi[0]) != len(fo[0]):
            return None
        # Verify fill-only
        for inp, out in examples:
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != 0 and inp[r][c] != out[r][c]:
                        return None

        prop_fns = [
            ("color", lambda o, objs, g: o["color"]),
            ("has_hole", lambda o, objs, g: _has_hole(o)),
            ("is_largest", lambda o, objs, g: o["size"] == max(x["size"] for x in objs)),
            ("compactness", lambda o, objs, g: round(_compactness(o), 1)),
        ]

        for prop_name, prop_fn in prop_fns:
            rule: dict = {}
            consistent = True
            for inp, out in examples:
                objects = _find_connected_components(inp)
                if not objects:
                    consistent = False
                    break
                for obj in objects:
                    r0, c0, r1, c1 = obj["bbox"]
                    sg = [[0] * (c1-c0+1) for _ in range(r1-r0+1)]
                    for r, c in obj["pixels"]:
                        sg[r-r0][c-c0] = obj["color"]
                    obj["subgrid"] = sg

                    pv = prop_fn(obj, objects, inp)
                    fills = set()
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            if inp[r][c] == 0 and out[r][c] != 0:
                                fills.add(out[r][c])
                    fill_val = fills.pop() if len(fills) == 1 else (None if not fills else -1)
                    if fill_val == -1:
                        consistent = False
                        break
                    if pv in rule and rule[pv] != fill_val:
                        consistent = False
                        break
                    rule[pv] = fill_val
                if not consistent:
                    break
            if not consistent or not any(v is not None for v in rule.values()):
                continue

            def _make_fn(r=dict(rule), pfn=prop_fn):
                def fn(grid):
                    objects = _find_connected_components(grid)
                    result = [row[:] for row in grid]
                    for obj in objects:
                        r0, c0, r1, c1 = obj["bbox"]
                        sg = [[0] * (c1-c0+1) for _ in range(r1-r0+1)]
                        for rr, cc in obj["pixels"]:
                            sg[rr-r0][cc-c0] = obj["color"]
                        obj["subgrid"] = sg
                        pv = pfn(obj, objects, grid)
                        fc = r.get(pv)
                        if fc is not None:
                            for rr in range(r0, r1 + 1):
                                for cc in range(c0, c1 + 1):
                                    if result[rr][cc] == 0:
                                        result[rr][cc] = fc
                    return result
                return fn

            fn = _make_fn()
            if not _test_on_examples(fn, examples):
                continue

            # LOOCV
            loocv_pass = True
            for hold_idx in range(len(examples)):
                sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                sub_rule: dict = {}
                sub_ok = True
                for inp, out in sub:
                    objects = _find_connected_components(inp)
                    if not objects:
                        sub_ok = False
                        break
                    for obj in objects:
                        r0, c0, r1, c1 = obj["bbox"]
                        sg = [[0] * (c1-c0+1) for _ in range(r1-r0+1)]
                        for r, c in obj["pixels"]:
                            sg[r-r0][c-c0] = obj["color"]
                        obj["subgrid"] = sg
                        pv = prop_fn(obj, objects, inp)
                        fills = set()
                        for r in range(r0, r1 + 1):
                            for c in range(c0, c1 + 1):
                                if inp[r][c] == 0 and out[r][c] != 0:
                                    fills.add(out[r][c])
                        fill_val = fills.pop() if len(fills) == 1 else (None if not fills else -1)
                        if fill_val == -1:
                            sub_ok = False
                            break
                        if pv in sub_rule and sub_rule[pv] != fill_val:
                            sub_ok = False
                            break
                        sub_rule[pv] = fill_val
                    if not sub_ok:
                        break
                if not sub_ok:
                    loocv_pass = False
                    break
                sub_fn = _make_fn(sub_rule, prop_fn)
                try:
                    if sub_fn(examples[hold_idx][0]) != examples[hold_idx][1]:
                        loocv_pass = False
                        break
                except Exception:
                    loocv_pass = False
                    break
            if not loocv_pass:
                continue

            name = f"cond_bbox_fill({prop_name})"
            prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
            self.register_primitive(prim)
            return (name, fn)

        return None

    def _try_cell_grid_colormap(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Map a separator-divided cell grid to a smaller output grid.

        For grids divided by separator lines into NR×NC cells:
        each cell's sorted pixel content maps to one output pixel.
        Output dimensions = NR × NC.
        """
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        fi, fo = examples[0]
        h, w = len(fi), len(fi[0]) if fi else 0
        oh, ow = len(fo), len(fo[0]) if fo else 0
        if oh >= h or ow >= w or oh == 0 or ow == 0:
            return None

        def _get_cell_grid(grid):
            gh, gw = len(grid), len(grid[0])
            sr = sorted(set(r for r in range(gh) if len(set(grid[r])) == 1 and grid[r][0] != 0))
            sc = sorted(set(c for c in range(gw) if len(set(grid[r][c] for r in range(gh))) == 1 and grid[0][c] != 0))
            rb, prev = [], 0
            for r in sr:
                if r > prev:
                    rb.append((prev, r))
                prev = r + 1
            if prev < gh:
                rb.append((prev, gh))
            cb, prev = [], 0
            for c in sc:
                if c > prev:
                    cb.append((prev, c))
                prev = c + 1
            if prev < gw:
                cb.append((prev, gw))
            return rb, cb

        rb0, cb0 = _get_cell_grid(fi)
        if len(rb0) != oh or len(cb0) != ow:
            return None

        color_map: dict[tuple, int] = {}
        consistent = True
        for inp, out in examples:
            rb, cb = _get_cell_grid(inp)
            if len(rb) != oh or len(cb) != ow:
                consistent = False
                break
            for ri, (r0, r1) in enumerate(rb):
                for ci, (c0, c1) in enumerate(cb):
                    cell = tuple(sorted(inp[r][c] for r in range(r0, r1) for c in range(c0, c1)))
                    val = out[ri][ci]
                    if cell in color_map and color_map[cell] != val:
                        consistent = False
                        break
                    color_map[cell] = val
                if not consistent:
                    break
            if not consistent:
                break
        if not consistent or not color_map:
            return None

        def _make_fn(cm=dict(color_map), n_r=oh, n_c=ow):
            def fn(grid):
                rb, cb = _get_cell_grid(grid)
                result = []
                for ri, (r0, r1) in enumerate(rb):
                    row = []
                    for ci, (c0, c1) in enumerate(cb):
                        cell = tuple(sorted(grid[r][c] for r in range(r0, r1) for c in range(c0, c1)))
                        row.append(cm.get(cell, 0))
                    result.append(row)
                return result
            return fn

        fn = _make_fn()
        if not _test_on_examples(fn, examples):
            return None

        # LOOCV
        for hold_idx in range(len(examples)):
            sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
            sub_map: dict[tuple, int] = {}
            sub_ok = True
            for inp, out in sub:
                rb, cb = _get_cell_grid(inp)
                if len(rb) != oh or len(cb) != ow:
                    sub_ok = False
                    break
                for ri, (r0, r1) in enumerate(rb):
                    for ci, (c0, c1) in enumerate(cb):
                        cell = tuple(sorted(inp[r][c] for r in range(r0, r1) for c in range(c0, c1)))
                        if cell in sub_map and sub_map[cell] != out[ri][ci]:
                            sub_ok = False
                            break
                        sub_map[cell] = out[ri][ci]
                    if not sub_ok:
                        break
                if not sub_ok:
                    break
            if not sub_ok:
                return None
            sub_fn = _make_fn(sub_map, oh, ow)
            try:
                if sub_fn(examples[hold_idx][0]) != examples[hold_idx][1]:
                    return None
            except Exception:
                return None

        name = "cell_grid_colormap"
        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
        self.register_primitive(prim)
        from .primitives import _PRIM_RULES
        rules_desc = f"Cell grid {oh}×{ow}: sorted_cell_content → output_pixel\n"
        for cell, val in sorted(color_map.items(), key=lambda x: x[1]):
            rules_desc += f"  {cell} → {val}\n"
        _PRIM_RULES[name] = rules_desc
        return (name, fn)

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

        def _find_h_separators(grid):
            seps = []
            for r in range(len(grid)):
                row = grid[r]
                if len(set(row)) == 1 and row[0] != 0:
                    seps.append(r)
            return seps

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

        for inp, _ in examples[1:]:
            if _find_h_separators(inp) != h_seps:
                h_seps = []
            if _find_v_separators(inp) != v_seps:
                v_seps = []

        if not h_seps and not v_seps:
            return None

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

        first_out = examples[0][1]
        for ri, row in enumerate(cells):
            for ci, cell in enumerate(row):
                if cell == first_out:
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

        # Boolean ops between pairs of cells
        n_rows = len(cells)
        n_cols = len(cells[0]) if cells else 0
        all_cells_flat = [(ri, ci, cells[ri][ci])
                          for ri in range(n_rows)
                          for ci in range(n_cols)]

        cell_h = len(cells[0][0]) if cells and cells[0] and cells[0][0] else 0
        cell_w = len(cells[0][0][0]) if cell_h > 0 and cells[0][0][0] else 0
        same_size = all(
            len(c) == cell_h and all(len(r) == cell_w for r in c)
            for _, _, c in all_cells_flat
        )

        if same_size and cell_h > 0 and cell_w > 0:
            out_h = len(first_out)
            out_w = len(first_out[0]) if first_out else 0

            if out_h == cell_h and out_w == cell_w:
                cell_ops = [
                    ("xor", lambda a, b, ch, cw: [
                        [int(a[r][c] != 0) ^ int(b[r][c] != 0)
                         for c in range(cw)] for r in range(ch)]),
                    ("or", lambda a, b, ch, cw: [
                        [a[r][c] if a[r][c] != 0 else b[r][c]
                         for c in range(cw)] for r in range(ch)]),
                    ("and", lambda a, b, ch, cw: [
                        [a[r][c] if b[r][c] != 0 else 0
                         for c in range(cw)] for r in range(ch)]),
                    ("a_minus_b", lambda a, b, ch, cw: [
                        [a[r][c] if b[r][c] == 0 else 0
                         for c in range(cw)] for r in range(ch)]),
                    ("mask_a_color_b", lambda a, b, ch, cw: [
                        [b[r][c] if a[r][c] != 0 else 0
                         for c in range(cw)] for r in range(ch)]),
                ]

                for i, (ri1, ci1, _) in enumerate(all_cells_flat):
                    for j, (ri2, ci2, _) in enumerate(all_cells_flat):
                        if i == j:
                            continue
                        for op_name, op_fn in cell_ops:
                            def _make_cell_op_fn(
                                r1=ri1, c1=ci1, r2=ri2, c2=ci2,
                                of=op_fn, hs=h_seps, vs=v_seps,
                                ch=cell_h, cw=cell_w,
                            ):
                                def fn(grid):
                                    cs = _split_into_cells(grid, hs, vs)
                                    if (r1 < len(cs) and c1 < len(cs[r1])
                                            and r2 < len(cs) and c2 < len(cs[r2])):
                                        return of(cs[r1][c1], cs[r2][c2], ch, cw)
                                    return grid
                                return fn
                            fn = _make_cell_op_fn()
                            if _test_on_examples(fn, examples):
                                name = f"cross_ref(cell_{ri1}_{ci1}_{op_name}_cell_{ri2}_{ci2})"
                                prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                                self.register_primitive(prim)
                                return (name, fn)

                # OR-reduction across all cells
                if len(all_cells_flat) >= 2:
                    def _make_or_reduce_fn(hs=h_seps, vs=v_seps,
                                           ch=cell_h, cw=cell_w):
                        def fn(grid):
                            cs = _split_into_cells(grid, hs, vs)
                            flat = [cs[ri][ci]
                                    for ri in range(len(cs))
                                    for ci in range(len(cs[ri]))]
                            if not flat:
                                return grid
                            result = [[0] * cw for _ in range(ch)]
                            for cell in flat:
                                for r in range(min(ch, len(cell))):
                                    for c in range(min(cw, len(cell[r]))):
                                        if cell[r][c] != 0:
                                            result[r][c] = cell[r][c]
                            return result
                        return fn
                    fn = _make_or_reduce_fn()
                    if _test_on_examples(fn, examples):
                        name = "cross_ref(or_reduce_cells)"
                        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                        self.register_primitive(prim)
                        return (name, fn)

                    # AND/majority reduction
                    def _make_and_reduce_fn(hs=h_seps, vs=v_seps,
                                            ch=cell_h, cw=cell_w):
                        def fn(grid):
                            cs = _split_into_cells(grid, hs, vs)
                            flat = [cs[ri][ci]
                                    for ri in range(len(cs))
                                    for ci in range(len(cs[ri]))]
                            if not flat:
                                return grid
                            n = len(flat)
                            result = [[0] * cw for _ in range(ch)]
                            for r in range(ch):
                                for c in range(cw):
                                    vals = [flat[k][r][c] for k in range(n)
                                            if r < len(flat[k]) and c < len(flat[k][r])
                                            and flat[k][r][c] != 0]
                                    if len(vals) > n // 2:
                                        result[r][c] = Counter(vals).most_common(1)[0][0]
                            return result
                        return fn
                    fn = _make_and_reduce_fn()
                    if _test_on_examples(fn, examples):
                        name = "cross_ref(majority_reduce_cells)"
                        prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                        self.register_primitive(prim)
                        return (name, fn)

        return None

    def _try_scale_tile_detection(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try scale/tile/downscale based on integer dimension ratios."""
        from .objects import _test_on_examples
        from .transformation_primitives import (
            _scale_factory, _tile_factory, _downscale_factory,
        )
        examples = task.train_examples
        if not examples:
            return None
        first_inp, first_out = examples[0]
        ih, iw = len(first_inp), len(first_inp[0]) if first_inp else 0
        oh, ow = len(first_out), len(first_out[0]) if first_out else 0
        if ih == 0 or iw == 0 or oh == 0 or ow == 0:
            return None

        if oh >= ih * 2 and ow >= iw * 2 and oh % ih == 0 and ow % iw == 0:
            h_ratio, w_ratio = oh // ih, ow // iw
            if h_ratio == w_ratio and h_ratio >= 2:
                n = h_ratio
                scale_fn = _scale_factory(n)
                if _test_on_examples(scale_fn, examples):
                    name = f"cross_ref(scale_{n}x)"
                    prim = Primitive(name=name, arity=0, fn=scale_fn, domain="arc")
                    self.register_primitive(prim)
                    return (name, scale_fn)
                tile_fn = _tile_factory(n)
                if _test_on_examples(tile_fn, examples):
                    name = f"cross_ref(tile_{n}x)"
                    prim = Primitive(name=name, arity=0, fn=tile_fn, domain="arc")
                    self.register_primitive(prim)
                    return (name, tile_fn)

        if ih >= oh * 2 and iw >= ow * 2 and ih % oh == 0 and iw % ow == 0:
            h_ratio, w_ratio = ih // oh, iw // ow
            if h_ratio == w_ratio and h_ratio >= 2:
                n = h_ratio
                ds_fn = _downscale_factory(n)
                if _test_on_examples(ds_fn, examples):
                    name = f"cross_ref(downscale_{n}x)"
                    prim = Primitive(name=name, arity=0, fn=ds_fn, domain="arc")
                    self.register_primitive(prim)
                    return (name, ds_fn)

        # Pixel-to-tile: each input pixel maps to a k×k output tile by color
        if oh >= ih * 2 and ow >= iw * 2 and oh % ih == 0 and ow % iw == 0:
            kh, kw = oh // ih, ow // iw
            if kh == kw and 2 <= kh <= 5:
                k = kh
                color_to_tile: dict[int, tuple] = {}
                consistent = True
                for inp, out in examples:
                    h_i, w_i = len(inp), len(inp[0])
                    if len(out) != h_i * k or len(out[0]) != w_i * k:
                        consistent = False
                        break
                    for r in range(h_i):
                        for c in range(w_i):
                            tile = tuple(tuple(out[r*k+dr][c*k+dc] for dc in range(k))
                                         for dr in range(k))
                            color = inp[r][c]
                            if color in color_to_tile and color_to_tile[color] != tile:
                                consistent = False
                                break
                            color_to_tile[color] = tile
                        if not consistent:
                            break
                    if not consistent:
                        break

                if consistent and color_to_tile:
                    def _make_tile_fn(ct=dict(color_to_tile), kk=k):
                        def fn(grid):
                            h_g, w_g = len(grid), len(grid[0])
                            result = [[0] * (w_g * kk) for _ in range(h_g * kk)]
                            for r in range(h_g):
                                for c in range(w_g):
                                    tile = ct.get(grid[r][c])
                                    if tile:
                                        for dr in range(kk):
                                            for dc in range(kk):
                                                result[r*kk+dr][c*kk+dc] = tile[dr][dc]
                            return result
                        return fn

                    tile_fn = _make_tile_fn()
                    if _test_on_examples(tile_fn, examples):
                        # LOOCV
                        loocv_pass = True
                        for hold_idx in range(len(examples)):
                            sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                            sub_ct: dict[int, tuple] = {}
                            sub_ok = True
                            for inp, out in sub:
                                for r in range(len(inp)):
                                    for c in range(len(inp[0])):
                                        tile = tuple(tuple(out[r*k+dr][c*k+dc]
                                                           for dc in range(k)) for dr in range(k))
                                        color = inp[r][c]
                                        if color in sub_ct and sub_ct[color] != tile:
                                            sub_ok = False
                                            break
                                        sub_ct[color] = tile
                                    if not sub_ok:
                                        break
                                if not sub_ok:
                                    break
                            if not sub_ok:
                                loocv_pass = False
                                break
                            sub_fn = _make_tile_fn(sub_ct, k)
                            held_inp, held_exp = examples[hold_idx]
                            try:
                                if sub_fn(held_inp) != held_exp:
                                    loocv_pass = False
                                    break
                            except Exception:
                                loocv_pass = False
                                break

                        if loocv_pass:
                            name = f"pixel_to_tile({k}x{k})"
                            prim = Primitive(name=name, arity=0, fn=tile_fn, domain="arc")
                            self.register_primitive(prim)
                            return (name, tile_fn)

        return None

    def _try_template_stamp(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try template stamping: find a small pattern, stamp at marker positions."""
        from .objects import (
            _test_on_examples, _find_connected_components,
            _get_background_color,
        )
        examples = task.train_examples
        if not examples:
            return None
        for inp, out in examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        first_inp = examples[0][0]
        bg = _get_background_color(first_inp)
        comps = _find_connected_components(first_inp)
        if len(comps) < 2:
            return None

        comps.sort(key=lambda c: c["size"])

        for template_idx in range(min(3, len(comps))):
            template_comp = comps[template_idx]
            if template_comp["size"] > 25:
                continue
            t_r0, t_c0, t_r1, t_c1 = template_comp["bbox"]
            t_h = t_r1 - t_r0 + 1
            t_w = t_c1 - t_c0 + 1
            t_color = template_comp["color"]

            template = [[0] * t_w for _ in range(t_h)]
            for r, c in template_comp["pixels"]:
                template[r - t_r0][c - t_c0] = t_color

            for marker_color in set(
                c["color"] for c in comps if c["color"] != t_color and c["color"] != bg
            ):
                marker_comps = [c for c in comps
                                if c["color"] == marker_color and c["size"] == 1]
                if not marker_comps:
                    continue

                def _make_stamp_fn(tmpl=template, m_color=marker_color,
                                   t_bg=bg, th=t_h, tw=t_w):
                    def fn(grid):
                        h, w = len(grid), len(grid[0]) if grid else 0
                        markers = [(r, c) for r in range(h) for c in range(w)
                                   if grid[r][c] == m_color]
                        result = [row[:] for row in grid]
                        for mr, mc in markers:
                            result[mr][mc] = t_bg
                        for mr, mc in markers:
                            sr = mr - th // 2
                            sc = mc - tw // 2
                            for tr in range(th):
                                for tc in range(tw):
                                    if tmpl[tr][tc] != 0:
                                        nr, nc = sr + tr, sc + tc
                                        if 0 <= nr < h and 0 <= nc < w:
                                            result[nr][nc] = tmpl[tr][tc]
                        return result
                    return fn

                fn = _make_stamp_fn()
                if _test_on_examples(fn, examples):
                    name = f"template_stamp({t_color}_at_{marker_color})"
                    prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                    self.register_primitive(prim)
                    return (name, fn)

        return None

    def _try_separator_marker_ops(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Try marker operations relative to separator lines.

        Strategy 1: Recolor each non-separator pixel to nearest separator's color.
        Strategy 2: Slide each marker to the separator matching its color.

        Justified by tasks 2204b7a8 and 1a07d186.
        """
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples:
            return None
        # Only same-dims tasks
        for inp, out in examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        first_inp = examples[0][0]
        h, w = len(first_inp), len(first_inp[0]) if first_inp else 0

        # Detect separators
        def _find_seps(grid):
            gh, gw = len(grid), len(grid[0]) if grid else 0
            v_seps, h_seps = {}, {}
            for c in range(gw):
                col = [grid[r][c] for r in range(gh)]
                if len(set(col)) == 1 and col[0] != 0:
                    v_seps.setdefault(col[0], []).append(c)
            for r in range(gh):
                if len(set(grid[r])) == 1 and grid[r][0] != 0:
                    h_seps.setdefault(grid[r][0], []).append(r)
            return v_seps, h_seps

        v_seps, h_seps = _find_seps(first_inp)
        if not v_seps and not h_seps:
            return None

        # Strategy 1: Recolor markers to nearest separator color
        def _make_recolor_fn():
            def fn(grid):
                gh, gw = len(grid), len(grid[0]) if grid else 0
                vs, hs = _find_seps(grid)
                sep_rows = {r for rows in hs.values() for r in rows}
                sep_cols = {c for cols in vs.values() for c in cols}
                all_seps = []
                for color, rows in hs.items():
                    for r in rows:
                        all_seps.append(('h', r, color))
                for color, cols in vs.items():
                    for c in cols:
                        all_seps.append(('v', c, color))
                result = [row[:] for row in grid]
                for r in range(gh):
                    if r in sep_rows:
                        continue
                    for c in range(gw):
                        if c in sep_cols:
                            continue
                        if grid[r][c] != 0:
                            best_d, best_color = float('inf'), grid[r][c]
                            for kind, pos, color in all_seps:
                                d = abs(r - pos) if kind == 'h' else abs(c - pos)
                                if d < best_d:
                                    best_d, best_color = d, color
                            result[r][c] = best_color
                return result
            return fn

        fn1 = _make_recolor_fn()
        if _test_on_examples(fn1, examples):
            name = "recolor_markers_by_nearest_sep"
            prim = Primitive(name=name, arity=0, fn=fn1, domain="arc")
            self.register_primitive(prim)
            return (name, fn1)

        # Strategy 2: Slide markers to matching-color separator
        def _make_slide_fn():
            def fn(grid):
                gh, gw = len(grid), len(grid[0]) if grid else 0
                vs, hs = _find_seps(grid)
                sep_rows = {r for rows in hs.values() for r in rows}
                sep_cols = {c for cols in vs.values() for c in cols}
                result = [row[:] for row in grid]
                for r in range(gh):
                    if r in sep_rows:
                        continue
                    for c in range(gw):
                        if c in sep_cols:
                            continue
                        if grid[r][c] == 0:
                            continue
                        mcolor = grid[r][c]
                        result[r][c] = 0
                        if mcolor in vs:
                            best_c = min(vs[mcolor], key=lambda sc: abs(c - sc))
                            nc = best_c - 1 if c < best_c else best_c + 1
                            if 0 <= nc < gw:
                                result[r][nc] = mcolor
                        elif mcolor in hs:
                            best_r = min(hs[mcolor], key=lambda sr: abs(r - sr))
                            nr = best_r - 1 if r < best_r else best_r + 1
                            if 0 <= nr < gh:
                                result[nr][c] = mcolor
                return result
            return fn

        fn2 = _make_slide_fn()
        if _test_on_examples(fn2, examples):
            name = "slide_markers_to_matching_sep"
            prim = Primitive(name=name, arity=0, fn=fn2, domain="arc")
            self.register_primitive(prim)
            return (name, fn2)

        return None

    def _try_subgrid_selection(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Extract subgrid by property: densest or most-colorful region.

        For tasks where output is smaller than input, tries extracting
        the output-sized subgrid that maximizes a specific property.

        Justified by tasks a87f7484, d9fac9be, 2013d3e2, d10ecb37.
        """
        from .objects import _test_on_examples
        examples = task.train_examples
        if not examples:
            return None
        # Output must be smaller than input (extraction task)
        oh = len(examples[0][1])
        ow = len(examples[0][1][0]) if examples[0][1] else 0
        for inp, out in examples:
            if len(out) != oh or len(out[0]) != ow:
                return None
            if oh >= len(inp) and ow >= len(inp[0]):
                return None  # not an extraction

        def _select(grid, oh, ow, score_fn):
            ih, iw = len(grid), len(grid[0])
            if oh > ih or ow > iw:
                return grid
            best_pos, best_score = (0, 0), -1
            for r in range(ih - oh + 1):
                for c in range(iw - ow + 1):
                    s = score_fn(grid, r, c, oh, ow)
                    if s > best_score:
                        best_score = s
                        best_pos = (r, c)
            r, c = best_pos
            return [grid[r + dr][c:c + ow] for dr in range(oh)]

        def _densest(grid, r, c, oh, ow):
            return sum(1 for dr in range(oh) for dc in range(ow)
                       if grid[r + dr][c + dc] != 0)

        def _most_colorful(grid, r, c, oh, ow):
            return len(set(grid[r + dr][c + dc]
                           for dr in range(oh) for dc in range(ow)))

        for strat_name, score_fn in [
            (f"densest_subgrid_{oh}x{ow}", _densest),
            (f"most_colorful_subgrid_{oh}x{ow}", _most_colorful),
        ]:
            def _make_fn(sf=score_fn, h=oh, w=ow):
                def fn(grid):
                    return _select(grid, h, w, sf)
                return fn

            fn = _make_fn()
            if _test_on_examples(fn, examples):
                prim = Primitive(name=strat_name, arity=0, fn=fn, domain="arc")
                self.register_primitive(prim)
                return (strat_name, fn)

        return None

    def try_local_rules(
        self, task: Task,
    ) -> Optional[tuple[str, Any]]:
        """Learn cellular automaton rules from training examples.

        Tries three rule types in order of compactness:
        1. (center, n_nonzero_4neighbors, majority_4neighbor) → output
        2. (center, n_nonzero_8neighbors) → output
        3. Raw 3x3 neighborhood → output

        Each rule is LOOCV-validated to avoid overfitting.
        """
        from .objects import _test_on_examples
        from collections import Counter
        examples = task.train_examples
        if not examples or len(examples) < 2:
            return None
        # Only same-dims tasks
        for inp, out in examples:
            if len(inp) != len(out):
                return None
            if inp and out and len(inp[0]) != len(out[0]):
                return None

        def _nbr_4(grid, r, c):
            h, w = len(grid), len(grid[0])
            return [grid[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1
                    for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]]

        def _learn_compact(exs):
            """Rule type 1: (center, n_nz_4, majority_4) → output."""
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        nb = _nbr_4(inp, r, c)
                        nz = [n for n in nb if n > 0]
                        key = (inp[r][c], len(nz),
                               Counter(nz).most_common(1)[0][0] if nz else 0)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_compact(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    nb = _nbr_4(grid, r, c)
                    nz = [n for n in nb if n > 0]
                    key = (grid[r][c], len(nz),
                           Counter(nz).most_common(1)[0][0] if nz else 0)
                    row.append(rule.get(key, grid[r][c]))
                result.append(row)
            return result

        def _learn_v2(exs):
            """Rule type 2: (center, n_nz_8) → output."""
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        n_nz = sum(1 for dr in range(-1,2) for dc in range(-1,2)
                                   if (dr or dc) and 0<=r+dr<h and 0<=c+dc<w
                                   and inp[r+dr][c+dc] != 0)
                        key = (inp[r][c], n_nz)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_v2(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    n_nz = sum(1 for dr in range(-1,2) for dc in range(-1,2)
                               if (dr or dc) and 0<=r+dr<h and 0<=c+dc<w
                               and grid[r+dr][c+dc] != 0)
                    key = (grid[r][c], n_nz)
                    row.append(rule.get(key, grid[r][c]))
                result.append(row)
            return result

        def _learn_raw3(exs):
            """Rule type 3: raw 3x3 neighborhood → output."""
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        key = tuple(inp[r+dr][c+dc] if 0<=r+dr<h and 0<=c+dc<w else -1
                                    for dr in range(-1,2) for dc in range(-1,2))
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_raw3(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    key = tuple(grid[r+dr][c+dc] if 0<=r+dr<h and 0<=c+dc<w else -1
                                for dr in range(-1,2) for dc in range(-1,2))
                    row.append(rule.get(key, grid[r][c]))
                result.append(row)
            return result

        # Rule type 4: position-modular (handles periodic patterns)
        def _learn_pos_mod(exs, period):
            """Rule: (center, r%period, c%period) → output."""
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        key = (inp[r][c], r % period, c % period)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_pos_mod(grid, rule, period):
            h, w = len(grid), len(grid[0])
            return [[rule.get((grid[r][c], r % period, c % period), grid[r][c])
                      for c in range(w)] for r in range(h)]

        # Try each rule type with LOOCV
        rule_types = [
            ("compact_local_rule", _learn_compact, _apply_compact),
            ("count_local_rule", _learn_v2, _apply_v2),
            ("raw3x3_local_rule", _learn_raw3, _apply_raw3),
        ]
        # Add position-modular rules for small periods
        for period in [2, 3, 4, 5]:
            rule_types.append((
                f"pos_mod{period}_rule",
                lambda exs, p=period: _learn_pos_mod(exs, p),
                lambda grid, rule, p=period: _apply_pos_mod(grid, rule, p),
            ))

        # Rule type 5: (center, n_distinct_4neighbor_colors) → output
        def _learn_ncolors(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        nc = len(set(inp[r+dr][c+dc]
                                     for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                     if 0 <= r+dr < h and 0 <= c+dc < w))
                        key = (inp[r][c], nc)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_ncolors(grid, rule):
            h, w = len(grid), len(grid[0])
            return [[rule.get((grid[r][c], len(set(
                        grid[r+dr][c+dc]
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= r+dr < h and 0 <= c+dc < w))),
                     grid[r][c])
                     for c in range(w)] for r in range(h)]

        rule_types.append(("ncolors_local_rule", _learn_ncolors, _apply_ncolors))

        # Rule type 6: (center, sorted_set_of_4neighbor_colors) → output
        def _learn_nbr_set(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        nbr = tuple(sorted(set(
                            inp[r+dr][c+dc]
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                            if 0 <= r+dr < h and 0 <= c+dc < w)))
                        key = (inp[r][c], nbr)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_nbr_set(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    nbr = tuple(sorted(set(
                        grid[r+dr][c+dc]
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= r+dr < h and 0 <= c+dc < w)))
                    row.append(rule.get((grid[r][c], nbr), grid[r][c]))
                result.append(row)
            return result

        rule_types.append(("nbr_set_local_rule", _learn_nbr_set, _apply_nbr_set))

        # Rule type 7: (center, n_nonzero_8neighbors, has_diagonal_nonzero)
        def _learn_8nbr_diag(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        n_nz = sum(1 for dr in range(-1, 2) for dc in range(-1, 2)
                                   if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w
                                   and inp[r+dr][c+dc] != 0)
                        has_diag = int(any(
                            0 <= r+dr < h and 0 <= c+dc < w
                            and inp[r+dr][c+dc] != 0
                            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]))
                        key = (inp[r][c], n_nz, has_diag)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_8nbr_diag(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    n_nz = sum(1 for dr in range(-1, 2) for dc in range(-1, 2)
                               if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w
                               and grid[r+dr][c+dc] != 0)
                    has_diag = int(any(
                        0 <= r+dr < h and 0 <= c+dc < w
                        and grid[r+dr][c+dc] != 0
                        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]))
                    row.append(rule.get((grid[r][c], n_nz, has_diag), grid[r][c]))
                result.append(row)
            return result

        rule_types.append(("8nbr_diag_local_rule", _learn_8nbr_diag, _apply_8nbr_diag))

        # Rule type 8: (center, min_nonzero_8neighbor_color) → output
        def _learn_min_nz_nbr(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        mn = min((inp[r+dr][c+dc]
                                  for dr in range(-1, 2) for dc in range(-1, 2)
                                  if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w
                                  and inp[r+dr][c+dc] != 0),
                                 default=-1)
                        key = (inp[r][c], mn)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_min_nz_nbr(grid, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    mn = min((grid[r+dr][c+dc]
                              for dr in range(-1, 2) for dc in range(-1, 2)
                              if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w
                              and grid[r+dr][c+dc] != 0),
                             default=-1)
                    row.append(rule.get((grid[r][c], mn), grid[r][c]))
                result.append(row)
            return result

        rule_types.append(("min_nz_nbr_local_rule", _learn_min_nz_nbr, _apply_min_nz_nbr))

        # Rule type 9: (center, left_color, right_color) → output
        def _learn_lr(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                for r in range(h):
                    for c in range(w):
                        key = (inp[r][c],
                               inp[r][c-1] if c > 0 else -1,
                               inp[r][c+1] if c < w-1 else -1)
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_lr(grid, rule):
            h, w = len(grid), len(grid[0])
            return [[rule.get((grid[r][c],
                               grid[r][c-1] if c > 0 else -1,
                               grid[r][c+1] if c < w-1 else -1),
                              grid[r][c])
                     for c in range(w)] for r in range(h)]

        rule_types.append(("lr_context_rule", _learn_lr, _apply_lr))

        # Rule type 10: (center, n_nonzero_in_row, n_nonzero_in_col) → output
        def _learn_rowcol_nz(exs):
            rule = {}
            for inp, out in exs:
                h, w = len(inp), len(inp[0])
                row_nz = [sum(1 for c in range(w) if inp[r][c] != 0)
                          for r in range(h)]
                col_nz = [sum(1 for r in range(h) if inp[r][c] != 0)
                          for c in range(w)]
                for r in range(h):
                    for c in range(w):
                        key = (inp[r][c], row_nz[r], col_nz[c])
                        val = out[r][c]
                        if key in rule and rule[key] != val:
                            return None
                        rule[key] = val
            return rule

        def _apply_rowcol_nz(grid, rule):
            h, w = len(grid), len(grid[0])
            row_nz = [sum(1 for c in range(w) if grid[r][c] != 0)
                      for r in range(h)]
            col_nz = [sum(1 for r in range(h) if grid[r][c] != 0)
                      for c in range(w)]
            return [[rule.get((grid[r][c], row_nz[r], col_nz[c]), grid[r][c])
                     for c in range(w)] for r in range(h)]

        rule_types.append(("rowcol_nz_rule", _learn_rowcol_nz, _apply_rowcol_nz))

        for rule_name, learn_fn, apply_fn in rule_types:
            # First: check if rule is consistent on ALL training
            rule = learn_fn(examples)
            if rule is None:
                continue
            if not _test_on_examples(lambda g, r=rule, a=apply_fn: a(g, r), examples):
                continue

            # LOOCV validation
            loocv_pass = True
            for hold_idx in range(len(examples)):
                train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
                rule_sub = learn_fn(train_sub)
                if rule_sub is None:
                    loocv_pass = False
                    break
                held_inp, held_exp = examples[hold_idx]
                result = apply_fn(held_inp, rule_sub)
                if result != held_exp:
                    loocv_pass = False
                    break

            if not loocv_pass:
                continue

            # LOOCV passed — create the transform
            def _make_fn(r=rule, a=apply_fn):
                def fn(grid):
                    return a(grid, r)
                return fn

            fn = _make_fn()
            prim = Primitive(name=rule_name, arity=0, fn=fn, domain="arc")
            self.register_primitive(prim)
            from .primitives import _PRIM_RULES
            n_rules = len(rule)
            sample = list(rule.items())[:8]
            rules_desc = f"Learned {n_rules} rules: key → output_pixel\n"
            for k, v in sample:
                rules_desc += f"  {k} → {v}\n"
            if n_rules > 8:
                rules_desc += f"  ... ({n_rules - 8} more)\n"
            _PRIM_RULES[rule_name] = rules_desc
            return (rule_name, fn)

        # Also try: local rules on TRANSFORMED inputs (depth-2 composition)
        from .transformation_primitives import fill_enclosed, dilate
        for transform_name, transform_fn in [
            ("fill_enclosed", fill_enclosed),
            ("dilate", dilate),
        ]:
            transformed_examples = []
            ok = True
            for inp, out in examples:
                try:
                    t_inp = transform_fn(inp)
                    if (not isinstance(t_inp, list) or not t_inp or
                            len(t_inp) != len(out) or len(t_inp[0]) != len(out[0])):
                        ok = False
                        break
                    transformed_examples.append((t_inp, out))
                except Exception:
                    ok = False
                    break
            if not ok or len(transformed_examples) < 2:
                continue

            for rule_name, learn_fn, apply_fn in [
                ("compact_local_rule", _learn_compact, _apply_compact),
            ]:
                rule = learn_fn(transformed_examples)
                if rule is None:
                    continue
                if not _test_on_examples(
                        lambda g, r=rule, a=apply_fn, t=transform_fn: a(t(g), r),
                        examples):
                    continue

                loocv_pass = True
                for hold_idx in range(len(transformed_examples)):
                    train_sub = [ex for i, ex in enumerate(transformed_examples)
                                 if i != hold_idx]
                    rule_sub = learn_fn(train_sub)
                    if rule_sub is None:
                        loocv_pass = False
                        break
                    held_inp = transformed_examples[hold_idx][0]
                    held_exp = transformed_examples[hold_idx][1]
                    if apply_fn(held_inp, rule_sub) != held_exp:
                        loocv_pass = False
                        break

                if not loocv_pass:
                    continue

                def _make_composed(r=rule, a=apply_fn, t=transform_fn):
                    def fn(grid):
                        return a(t(grid), r)
                    return fn

                fn = _make_composed()
                name = f"local_rule({transform_name})"
                prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
                self.register_primitive(prim)
                return (name, fn)

        return None

    def try_procedural(
        self, task,
    ) -> Optional[tuple[str, Any]]:
        """Learn per-object action rules from pixel diffs.

        Delegates to procedural.py which:
        1. Computes what changed between input and output
        2. Attributes changes to input objects
        3. Matches action templates (fill_bbox, extend_ray, etc.)
        4. Learns which objects get which action via property-based rules
        5. LOOCV-validates before returning
        """
        from .procedural import try_procedural as _try_procedural
        result = _try_procedural(task)
        if result is not None:
            name, fn = result
            prim = Primitive(name=name, arity=0, fn=fn, domain="arc")
            self.register_primitive(prim)
            return (name, fn)
        return None

    # Maximum intermediate grid size (pixels).
    MAX_GRID_PIXELS = 10_000

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
                    return grid
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
