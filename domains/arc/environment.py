"""
ARC-AGI Environment: execute grid transformation programs.
"""

from __future__ import annotations

from typing import Any, Optional

from collections import Counter

import numpy as np

from core import Environment, Primitive, Program, Task, Observation
from .primitives import Grid, _PRIM_MAP, _make_color_remap
from .objects import try_object_decomposition


def _extract_patch(arr, r, c, radius):
    """Extract a flat tuple of pixel values in a (2*radius+1)² neighborhood.
    Out-of-bounds pixels are encoded as -1."""
    h, w = arr.shape
    patch = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                patch.append(int(arr[nr, nc]))
            else:
                patch.append(-1)
    return tuple(patch)


class ARCEnv(Environment):
    """
    ARC-AGI environment.

    Programs are trees of grid transformations.
    Execute means: apply the transformation pipeline to an input grid.
    """

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

    def try_object_decomposition(self, task, primitives):
        """Try per-object transform decomposition for ARC grids."""
        result = try_object_decomposition(task.train_examples, primitives)
        if result is None:
            return None
        name, fn = result
        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
        _PRIM_MAP[name] = prim
        return (name, fn)

    def try_for_each_object(self, task, candidate_programs, top_k=10):
        """Try applying top-K programs per-object.

        Takes scored programs from enumeration (including depth-2+ compositions)
        and applies each one per-object. This enables compositions like
        for_each_object(mirror_h(crop_to_nonzero)).
        """
        if not task.train_examples:
            return None
        # Only same-dims tasks
        for inp, out in task.train_examples:
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                return None

        from .objects import apply_transform_per_object, _get_background_color, _test_on_examples
        bg = _get_background_color(task.train_examples[0][0])

        # Sort by error, try top-K
        sorted_cands = sorted(candidate_programs, key=lambda s: s.prediction_error)[:top_k]
        for sp in sorted_cands:
            prog = sp.program

            def _make_per_obj_fn(p=prog, env=self, bg_color=bg):
                def transform(subgrid):
                    return env.execute(p, subgrid)

                def fn(grid):
                    result = apply_transform_per_object(grid, transform, bg_color)
                    return result if result is not None else grid
                return fn

            per_obj_fn = _make_per_obj_fn()
            if _test_on_examples(per_obj_fn, task.train_examples):
                name = f"for_each_object({repr(prog)})"
                prim = Primitive(name=name, arity=1, fn=per_obj_fn, domain="arc")
                _PRIM_MAP[name] = prim
                return (name, per_obj_fn)

        return None

    def try_cross_reference(self, task, primitives):
        """Try solving via cross-reference: use one grid part to transform another.

        Strategy 1: Grid has separator lines → cells. Try using each cell as a
        mask/template applied to the others via overlay or boolean ops.

        Strategy 2: Two objects of different sizes. The smaller one acts as a
        pattern/template applied at the position of (or overlaid onto) the larger.
        """
        if not task.train_examples:
            return None

        from .primitives import (
            _detect_any_separator_lines, _split_grid_cells, Grid,
        )
        from .objects import (
            find_foreground_shapes, find_multicolor_objects,
            _get_background_color, _test_on_examples, place_subgrid,
        )
        import numpy as np

        # --- Strategy 1: Grid cells cross-reference ---
        # If the grid has separator lines, try: output = apply(template_cell, target_cell)
        first_inp = task.train_examples[0][0]
        try:
            h_lines, v_lines = _detect_any_separator_lines(first_inp)
        except Exception:
            h_lines, v_lines = [], []

        if h_lines or v_lines:
            cells_first = _split_grid_cells(first_inp)
            if cells_first and len(cells_first) >= 2:
                # Try using the smallest non-empty cell as template
                # applied to other cells via overlay
                n_cells = len(cells_first)

                for template_idx in range(min(n_cells, 4)):
                    def _make_xref(tidx=template_idx):
                        def xref_fn(grid):
                            try:
                                cells = _split_grid_cells(grid)
                                if not cells or len(cells) <= tidx:
                                    return grid
                                template = cells[tidx]
                                # Apply template as overlay to each other cell
                                result_cells = []
                                for i, cell in enumerate(cells):
                                    if i == tidx:
                                        result_cells.append(cell)
                                        continue
                                    # Overlay: template non-zero pixels onto cell
                                    th, tw = len(template), len(template[0]) if template else 0
                                    ch, cw = len(cell), len(cell[0]) if cell else 0
                                    if th != ch or tw != cw:
                                        result_cells.append(cell)
                                        continue
                                    merged = [row[:] for row in cell]
                                    for r in range(th):
                                        for c in range(tw):
                                            if template[r][c] != 0:
                                                merged[r][c] = template[r][c]
                                    result_cells.append(merged)
                                # Recompose — use grammar's recompose if available
                                from .grammar import ARCGrammar
                                from core.types import Decomposition
                                g = ARCGrammar()
                                h_l, v_l = _detect_any_separator_lines(grid)
                                sep_color = 0
                                h_grid, w_grid = len(grid), len(grid[0]) if grid else 0
                                if h_l:
                                    sep_color = grid[h_l[0]][0]
                                elif v_l:
                                    sep_color = grid[0][v_l[0]]
                                decomp = Decomposition(
                                    strategy="grid_partition",
                                    parts=cells,
                                    context={"h_lines": h_l, "v_lines": v_l,
                                             "sep_color": sep_color, "bg_color": 0,
                                             "grid_h": h_grid, "grid_w": w_grid},
                                )
                                return g.recompose(decomp, result_cells)
                            except Exception:
                                return grid
                        return xref_fn

                    fn = _make_xref()
                    if _test_on_examples(fn, task.train_examples):
                        name = f"cross_ref_cell_{template_idx}_overlay"
                        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
                        _PRIM_MAP[name] = prim
                        return (name, fn)

        # --- Strategy 2: Small object as template for large object ---
        bg = _get_background_color(first_inp)
        shapes = find_foreground_shapes(first_inp)
        if shapes and len(shapes) >= 2:
            sizes = sorted(set(s["size"] for s in shapes))
            if len(sizes) >= 2:
                small_size = sizes[0]
                large_size = sizes[-1]
                if large_size >= small_size * 2:
                    # Small object might be a pattern to stamp onto large
                    def _make_stamp_small_on_large(bg_c=bg):
                        def stamp_fn(grid):
                            objs = find_foreground_shapes(grid)
                            if len(objs) < 2:
                                return grid
                            by_size = sorted(objs, key=lambda o: o["size"])
                            small = by_size[0]
                            template = small["subgrid"]
                            result = [row[:] for row in grid]
                            # Zero out the small object
                            sr, sc = small["position"]
                            for r in range(len(template)):
                                for c in range(len(template[0])):
                                    if template[r][c] != 0:
                                        tr, tc = sr + r, sc + c
                                        if 0 <= tr < len(grid) and 0 <= tc < len(grid[0]):
                                            result[tr][tc] = bg_c
                            # Overlay template onto each larger object
                            for obj in by_size[1:]:
                                pos = obj["position"]
                                result = place_subgrid(result, template, pos,
                                                       transparent_color=bg_c)
                            return result
                        return stamp_fn

                    fn = _make_stamp_small_on_large()
                    if _test_on_examples(fn, task.train_examples):
                        name = "cross_ref_stamp_small_on_large"
                        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
                        _PRIM_MAP[name] = prim
                        return (name, fn)

        return None

    # -------------------------------------------------------------------------
    # Output correction: infer post-hoc fixes for near-miss programs
    # -------------------------------------------------------------------------

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        max_rules: int = 10,
        **kwargs,
    ) -> Optional[Program]:
        """Infer a correction that fixes mismatches between program outputs
        and expected outputs.

        Tries strategies in order: color remap -> adjacency -> 3x3 neighborhood
        -> row/column transforms.

        Single correction only (no chaining).
        """
        for strategy in [
            lambda: self._infer_color_correction(program_outputs, expected_outputs),
            lambda: self._infer_adjacency_correction(program_outputs, expected_outputs),
            lambda: self._infer_neighborhood_correction(program_outputs, expected_outputs, radius=1, max_rules=max_rules),
            lambda: self._infer_row_col_correction(program_outputs, expected_outputs),
        ]:
            result = strategy()
            if result is not None:
                return result
        return None

    def _infer_color_correction(self, program_outputs, expected_outputs):
        """Infer a color remapping that fixes mismatches.

        Collects pixel-level color mismatches. If a consistent remap exists
        (>80% agreement per source color), creates a correction Program.
        """
        votes: Counter = Counter()
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            if got_arr.shape != exp_arr.shape:
                return None
            diff = got_arr != exp_arr
            if not diff.any():
                continue
            for g, w in zip(got_arr[diff].flat, exp_arr[diff].flat):
                if g != w:
                    votes[(int(g), int(w))] += 1

        if not votes:
            return None

        # Build per-source-color vote tallies
        by_src: dict[int, Counter] = {}
        for (g, w), count in votes.items():
            by_src.setdefault(g, Counter())[w] += count

        # Build remap from consistent transitions (>80% agreement)
        remap: dict[int, int] = {}
        for g, tally in by_src.items():
            best_w, best_count = tally.most_common(1)[0]
            if best_count / sum(tally.values()) < 0.80:
                continue
            remap[g] = best_w

        if not remap:
            return None

        # Verify: remap must fix ALL diffs
        remap_fn = _make_color_remap(remap)
        for got, expected in zip(program_outputs, expected_outputs):
            if np.array(remap_fn(got), dtype=np.int32).tolist() != np.array(expected, dtype=np.int32).tolist():
                return None

        name = f"color_remap_{'_'.join(f'{k}to{v}' for k, v in sorted(remap.items()))}"
        if name not in _PRIM_MAP:
            _PRIM_MAP[name] = Primitive(name=name, arity=1, fn=remap_fn, domain="arc")
        return Program(root=name)

    def _infer_adjacency_correction(self, program_outputs, expected_outputs):
        """Infer adjacency-based pixel correction.

        Learns rules: "if pixel is color A with 4-neighbor of color B → color C".
        """
        rules: dict[tuple[int, frozenset], int] = {}

        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            if got_arr.shape != exp_arr.shape:
                return None
            h, w = got_arr.shape
            diff = got_arr != exp_arr
            if not diff.any():
                continue
            for r in range(h):
                for c in range(w):
                    if not diff[r, c]:
                        continue
                    center = int(got_arr[r, c])
                    nbrs = frozenset(
                        int(got_arr[r + dr, c + dc])
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        if 0 <= r + dr < h and 0 <= c + dc < w
                    )
                    key = (center, nbrs)
                    out_color = int(exp_arr[r, c])
                    if key in rules and rules[key] != out_color:
                        return None
                    rules[key] = out_color

        if not rules:
            return None

        # Verify on all training examples
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            h, w = got_arr.shape
            result = got_arr.copy()
            for r in range(h):
                for c in range(w):
                    center = int(got_arr[r, c])
                    nbrs = frozenset(
                        int(got_arr[r + dr, c + dc])
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        if 0 <= r + dr < h and 0 <= c + dc < w
                    )
                    if (center, nbrs) in rules:
                        result[r, c] = rules[(center, nbrs)]
            if not np.array_equal(result, exp_arr):
                return None

        # Convert frozenset keys to sorted tuples for the closure
        hashable = {(c, tuple(sorted(ns))): v for (c, ns), v in rules.items()}

        def _make_adj_fix(adj_rules):
            def adj_fix(grid: Grid) -> Grid:
                arr = np.array(grid, dtype=np.int32)
                out = arr.copy()
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        nbrs = tuple(sorted(set(
                            int(arr[r + dr, c + dc])
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            if 0 <= r + dr < h and 0 <= c + dc < w
                        )))
                        hk = (int(arr[r, c]), nbrs)
                        if hk in adj_rules:
                            out[r, c] = adj_rules[hk]
                return out.tolist()
            return adj_fix

        name = f"adjacency_fix_{len(rules)}r"
        fn = _make_adj_fix(hashable)
        _PRIM_MAP[name] = Primitive(name=name, arity=1, fn=fn, domain="arc")
        return Program(root=name)

    def _infer_neighborhood_correction(self, program_outputs, expected_outputs,
                                        radius=1, max_rules=50):
        """Infer neighborhood-based pixel correction (parameterized by radius).

        Encodes the (2r+1)² neighborhood around each mismatched pixel as a
        feature key and learns a mapping to the correct output color.
        Acts as a learned cellular automaton rule set, capped at max_rules.

        radius=1 → 3x3, radius=2 → 5x5.
        """
        rules: dict[tuple, int] = {}

        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            if got_arr.shape != exp_arr.shape:
                return None
            h, w = got_arr.shape
            diff = got_arr != exp_arr
            if not diff.any():
                continue
            for r in range(h):
                for c in range(w):
                    if not diff[r, c]:
                        continue
                    key = _extract_patch(got_arr, r, c, radius)
                    out_color = int(exp_arr[r, c])
                    if key in rules and rules[key] != out_color:
                        return None
                    rules[key] = out_color

        if not rules or len(rules) > max_rules:
            return None

        # Verify on all training examples
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            h, w = got_arr.shape
            result = got_arr.copy()
            for r in range(h):
                for c in range(w):
                    key = _extract_patch(got_arr, r, c, radius)
                    if key in rules:
                        result[r, c] = rules[key]
            if not np.array_equal(result, exp_arr):
                return None

        # Build correction primitive
        patch_radius = radius
        patch_name = f"neighborhood_{2*radius+1}x{2*radius+1}_fix_{len(rules)}r"

        def _make_nbr_fix(nbr_rules, rad):
            def nbr_fix(grid: Grid) -> Grid:
                arr = np.array(grid, dtype=np.int32)
                out = arr.copy()
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        key = _extract_patch(arr, r, c, rad)
                        if key in nbr_rules:
                            out[r, c] = nbr_rules[key]
                return out.tolist()
            return nbr_fix

        fn = _make_nbr_fix(dict(rules), patch_radius)
        _PRIM_MAP[patch_name] = Primitive(name=patch_name, arity=1, fn=fn, domain="arc")
        return Program(root=patch_name)

    def _infer_row_col_correction(self, program_outputs, expected_outputs):
        """Infer row/column-level corrections (reversal, cyclic shift, transpose)."""
        has_diff = False
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            if got_arr.shape != exp_arr.shape:
                return None
            if not np.array_equal(got_arr, exp_arr):
                has_diff = True
        if not has_diff:
            return None

        transforms = [
            ("row_reverse", lambda arr: arr[::-1].copy()),
            ("col_reverse", lambda arr: arr[:, ::-1].copy()),
            ("transpose", lambda arr: arr.T.copy() if arr.shape[0] == arr.shape[1] else None),
        ]
        for shift in range(1, 5):
            s = shift
            transforms.append((f"row_shift_{s}", lambda arr, s=s: np.roll(arr, s, axis=0)))
            transforms.append((f"col_shift_{s}", lambda arr, s=s: np.roll(arr, s, axis=1)))

        for name, transform_fn in transforms:
            if all(
                (r := transform_fn(np.array(got, dtype=np.int32))) is not None
                and np.array_equal(r, np.array(expected, dtype=np.int32))
                for got, expected in zip(program_outputs, expected_outputs)
            ):
                def _make_row_col_fix(tfn):
                    def row_col_fix(grid: Grid) -> Grid:
                        arr = np.array(grid, dtype=np.int32)
                        result = tfn(arr)
                        return grid if result is None else result.tolist()
                    return row_col_fix

                prim_name = f"row_col_fix_{name}"
                fn = _make_row_col_fix(transform_fn)
                _PRIM_MAP[prim_name] = Primitive(name=prim_name, arity=1, fn=fn, domain="arc")
                return Program(root=prim_name)

        return None

    # -------------------------------------------------------------------------
    # Program execution
    # -------------------------------------------------------------------------

    # Maximum intermediate grid size (pixels). Guards against runaway expansion
    # from composed grid-expanding primitives (tile_3x3=9x, scale_5x=25x).
    MAX_GRID_PIXELS = 10_000  # ~100x100 — generous for ARC (max 30x30)

    def _eval_tree(self, node: Program, grid: Grid) -> Grid:
        """Recursively evaluate a program tree on a grid."""
        prim = _PRIM_MAP.get(node.root)
        if prim is None:
            return grid

        try:
            if prim.arity == 0:
                if isinstance(prim.fn, Program):
                    return self._eval_tree(prim.fn, grid)
                return grid
            elif prim.arity == 1:
                child_grid = self._eval_tree(node.children[0], grid) if node.children else grid
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
