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
        """Try solving via cross-reference: one grid part informs another.

        Strategy 1: Boolean ops on grid halves — split by separator,
                    combine halves with AND/OR/XOR, optionally recolor.
        Strategy 2: Cell propagation — colored markers in cells propagate
                    across their row or column of cells.
        Strategy 3: Small-on-large stamping — smallest object used as
                    template stamped at positions of larger objects.
        """
        if not task.train_examples:
            return None

        from .primitives import _detect_any_separator_lines, _split_grid_cells
        from .objects import (
            find_foreground_shapes, _get_background_color,
            _test_on_examples, place_subgrid,
        )
        import numpy as np

        first_inp = task.train_examples[0][0]
        first_out = task.train_examples[0][1]

        # --- Strategy 1: Boolean ops on halves ---
        # Grid split by ONE separator into 2 equal halves → output is
        # AND/OR/XOR of the halves, optionally recolored.
        try:
            h_lines, v_lines = _detect_any_separator_lines(first_inp)
        except Exception:
            h_lines, v_lines = [], []

        out_h, out_w = len(first_out), len(first_out[0]) if first_out else 0

        # Vertical split: find the consistent separator across all examples
        # (must be same position and same color in every training input)
        if v_lines and not h_lines:
            # Find separators consistent across all examples
            consistent_v = set(v_lines)
            for inp, _ in task.train_examples[1:]:
                try:
                    _, vl = _detect_any_separator_lines(inp)
                    consistent_v &= set(vl)
                except Exception:
                    consistent_v = set()
            v_lines = sorted(consistent_v)

        if len(v_lines) == 1 and not h_lines:
            vc = v_lines[0]
            vc = v_lines[0]
            for op_name, op_fn in [
                ("and", lambda a, b: int(a != 0 and b != 0)),
                ("or",  lambda a, b: int(a != 0 or b != 0)),
                ("xor", lambda a, b: int((a != 0) != (b != 0))),
            ]:
                out_colors = {int(c) for row in first_out for c in row if c != 0}
                for recolor in sorted(out_colors | {1}):
                    def _make_bool_v(sep_col=vc, op=op_fn, rc=recolor):
                        def fn(grid):
                            arr = np.array(grid, dtype=np.int32)
                            # Use the known separator column position
                            sc = sep_col
                            if sc >= arr.shape[1]:
                                return grid
                            left = arr[:, :sc]
                            right = arr[:, sc+1:]
                            mw = min(left.shape[1], right.shape[1])
                            result = np.zeros((left.shape[0], mw), dtype=np.int32)
                            for r in range(left.shape[0]):
                                for c in range(mw):
                                    if op(int(left[r, c]), int(right[r, c])):
                                        result[r, c] = rc
                            return result.tolist()
                        return fn

                    fn = _make_bool_v()
                    if _test_on_examples(fn, task.train_examples):
                        name = f"cross_ref_{op_name}_halves_v_color_{recolor}"
                        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
                        _PRIM_MAP[name] = prim
                        return (name, fn)

        # Horizontal split — also find consistent separator
        if h_lines and not v_lines:
            consistent_h = set(h_lines)
            for inp, _ in task.train_examples[1:]:
                try:
                    hl, _ = _detect_any_separator_lines(inp)
                    consistent_h &= set(hl)
                except Exception:
                    consistent_h = set()
            h_lines = sorted(consistent_h)

        if len(h_lines) == 1 and not v_lines:
            hr = h_lines[0]
            for op_name, op_fn in [
                ("and", lambda a, b: int(a != 0 and b != 0)),
                ("or",  lambda a, b: int(a != 0 or b != 0)),
                ("xor", lambda a, b: int((a != 0) != (b != 0))),
            ]:
                out_colors = {int(c) for row in first_out for c in row if c != 0}
                for recolor in sorted(out_colors | {1}):
                    def _make_bool_h(sep_row=hr, op=op_fn, rc=recolor):
                        def fn(grid):
                            arr = np.array(grid, dtype=np.int32)
                            sr = sep_row
                            if sr >= arr.shape[0]:
                                return grid
                            top = arr[:sr, :]
                            bottom = arr[sr+1:, :]
                            mh = min(top.shape[0], bottom.shape[0])
                            mw = min(top.shape[1], bottom.shape[1])
                            result = np.zeros((mh, mw), dtype=np.int32)
                            for r in range(mh):
                                for c in range(mw):
                                    if op(int(top[r, c]), int(bottom[r, c])):
                                        result[r, c] = rc
                            return result.tolist()
                        return fn

                    fn = _make_bool_h()
                    if _test_on_examples(fn, task.train_examples):
                        name = f"cross_ref_{op_name}_halves_h_color_{recolor}"
                        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
                        _PRIM_MAP[name] = prim
                        return (name, fn)

        # --- Strategy 2: Cell propagation ---
        # Grid with separators → cells in rows/cols. Colored cells propagate
        # their color to fill empty cells between them in the same row.
        if (h_lines or v_lines):
            cells = _split_grid_cells(first_inp)
            if cells and len(cells) >= 4:
                # Compute grid layout (n_rows × n_cols of cells)
                n_col_cells = len(v_lines) + 1 if v_lines else 1
                n_row_cells = len(h_lines) + 1 if h_lines else 1
                if n_row_cells * n_col_cells == len(cells) and n_col_cells >= 2:

                    def _make_cell_propagate():
                        def fn(grid):
                            try:
                                hl, vl = _detect_any_separator_lines(grid)
                                gc = _split_grid_cells(grid)
                                if not gc:
                                    return grid
                                nc = len(vl) + 1 if vl else 1
                                nr = len(hl) + 1 if hl else 1
                                if nr * nc != len(gc):
                                    return grid

                                result_cells = [c for c in gc]  # copy list

                                # For each row of cells, find colored cells and
                                # fill empty cells between same-colored ones
                                for row in range(nr):
                                    row_cells = [(col, gc[row * nc + col])
                                                 for col in range(nc)]
                                    # Find cells with content
                                    colored = {}
                                    for col, cell in row_cells:
                                        colors = {int(cell[r][c])
                                                  for r in range(len(cell))
                                                  for c in range(len(cell[0]))
                                                  if cell[r][c] != 0}
                                        if colors:
                                            for clr in colors:
                                                colored.setdefault(clr, []).append(col)

                                    # Fill between endpoints of each color
                                    for clr, cols in colored.items():
                                        if len(cols) >= 2:
                                            lo, hi = min(cols), max(cols)
                                            template = gc[row * nc + cols[0]]
                                            for col in range(lo, hi + 1):
                                                idx = row * nc + col
                                                cell = result_cells[idx]
                                                # If cell is empty, fill with template
                                                has_content = any(
                                                    cell[r][c] != 0
                                                    for r in range(len(cell))
                                                    for c in range(len(cell[0])))
                                                if not has_content:
                                                    result_cells[idx] = [
                                                        row[:] for row in template]

                                # Recompose
                                from .grammar import ARCGrammar
                                from core.types import Decomposition
                                g = ARCGrammar()
                                sep_color = 0
                                gh, gw = len(grid), len(grid[0]) if grid else 0
                                if hl:
                                    sep_color = grid[hl[0]][0]
                                elif vl:
                                    sep_color = grid[0][vl[0]]
                                decomp = Decomposition(
                                    strategy="grid_partition", parts=gc,
                                    context={"h_lines": hl, "v_lines": vl,
                                             "sep_color": sep_color, "bg_color": 0,
                                             "grid_h": gh, "grid_w": gw},
                                )
                                return g.recompose(decomp, result_cells)
                            except Exception:
                                return grid
                        return fn

                    fn = _make_cell_propagate()
                    if _test_on_examples(fn, task.train_examples):
                        name = "cross_ref_cell_propagate_row"
                        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
                        _PRIM_MAP[name] = prim
                        return (name, fn)

        # --- Strategy 3: Small object stamped onto large objects ---
        bg = _get_background_color(first_inp)
        shapes = find_foreground_shapes(first_inp)
        if shapes and len(shapes) >= 2:
            sizes = sorted(set(s["size"] for s in shapes))
            if len(sizes) >= 2 and sizes[-1] >= sizes[0] * 2:
                def _make_stamp(bg_c=bg):
                    def fn(grid):
                        objs = find_foreground_shapes(grid)
                        if len(objs) < 2:
                            return grid
                        by_size = sorted(objs, key=lambda o: o["size"])
                        small = by_size[0]
                        template = small["subgrid"]
                        result = [row[:] for row in grid]
                        sr, sc = small["position"]
                        for r in range(len(template)):
                            for c in range(len(template[0])):
                                if template[r][c] != 0:
                                    tr, tc = sr + r, sc + c
                                    if 0 <= tr < len(grid) and 0 <= tc < len(grid[0]):
                                        result[tr][tc] = bg_c
                        for obj in by_size[1:]:
                            result = place_subgrid(result, template,
                                                   obj["position"],
                                                   transparent_color=bg_c)
                        return result
                    return fn

                fn = _make_stamp()
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
