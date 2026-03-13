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

    # -------------------------------------------------------------------------
    # Output correction: infer post-hoc fixes for near-miss programs
    # -------------------------------------------------------------------------

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        max_rules: int = 50,
        try_5x5: bool = False,
        max_chain_depth: int = 2,
    ) -> Optional[Program]:
        """Infer a correction that fixes mismatches between program outputs
        and expected outputs.

        Tries strategies in order: color remap → adjacency → 3x3 neighborhood
        → 5x5 neighborhood → row/column transforms.

        If a correction reduces but doesn't eliminate error, recursively tries
        a second correction on the residual (max depth 2).
        """
        correction = self._infer_single_correction(
            program_outputs, expected_outputs, max_rules, try_5x5)
        if correction is None:
            return None

        if max_chain_depth <= 1:
            return correction

        # Apply correction and check for residual error
        corrected_outputs = []
        has_residual = False
        for got, expected in zip(program_outputs, expected_outputs):
            try:
                result = self.execute(correction, got)
                corrected_outputs.append(result)
                if np.array(result, dtype=np.int32).tolist() != np.array(expected, dtype=np.int32).tolist():
                    has_residual = True
            except Exception:
                return correction

        if not has_residual:
            return correction

        # Try a second correction on the residual
        second = self._infer_single_correction(
            corrected_outputs, expected_outputs, max_rules, try_5x5)
        if second is None:
            return correction

        return Program(root=second.root, children=[correction], params=second.params)

    def _infer_single_correction(self, program_outputs, expected_outputs,
                                  max_rules=50, try_5x5=False):
        """Try each correction strategy in order of specificity."""
        for strategy in [
            lambda: self._infer_color_correction(program_outputs, expected_outputs),
            lambda: self._infer_adjacency_correction(program_outputs, expected_outputs),
            lambda: self._infer_neighborhood_correction(program_outputs, expected_outputs, radius=1, max_rules=max_rules),
            lambda: (self._infer_neighborhood_correction(program_outputs, expected_outputs, radius=2, max_rules=min(max_rules, 30)) if try_5x5 else None),
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

    # Keep legacy names so tests that call them directly still work
    _infer_neighborhood_correction_5x5 = lambda self, po, eo, max_rules=30: \
        self._infer_neighborhood_correction(po, eo, radius=2, max_rules=max_rules)

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
