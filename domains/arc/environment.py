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
        # Register as a primitive so it can be executed
        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
        _PRIM_MAP[name] = prim
        return (name, fn)

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        max_rules: int = 50,
        try_5x5: bool = False,
    ) -> Optional[Program]:
        """Infer a correction that fixes mismatches between program outputs
        and expected outputs.

        Tries multiple strategies in order of specificity:
        1. Color remapping — consistent color→color substitutions (cheapest)
        2. Adjacency correction — pixel changes conditioned on 4-connected neighbors
        3. 3x3 neighborhood patch — full 3x3 context-based pixel correction
           (acts as learned cellular automaton rules, ≤max_rules rules)
        4. 5x5 neighborhood patch — longer-range dependencies (≤30 rules)
        5. Row/column correction — spatial rearrangements (reverse, shift, transpose)

        Generated primitives follow the naming convention:
        - color_remap_XtoY — color substitution
        - adjacency_fix_Nr — N adjacency rules
        - neighborhood_3x3_fix_Nr — N 3x3 neighborhood rules
        - neighborhood_5x5_fix_Nr — N 5x5 neighborhood rules
        - row_col_fix_<transform> — spatial rearrangement

        Returns a single best correction, or None. The caller (_try_color_fix)
        evaluates whether the correction actually improves accuracy.
        """
        # Strategy 1: Color remapping (cheapest, most general)
        color_fix = self._infer_color_correction(program_outputs, expected_outputs)
        if color_fix is not None:
            return color_fix

        # Strategy 2: Adjacency-based correction — pixel changes conditioned
        # on 4-connected neighbor colors (e.g., "0 next to 2 → becomes 1")
        adjacency_fix = self._infer_adjacency_correction(program_outputs, expected_outputs)
        if adjacency_fix is not None:
            return adjacency_fix

        # Strategy 3: 3x3 neighborhood patch — encodes the full 3x3 pixel
        # neighborhood as a feature key and learns output color per key.
        # Acts as a learned cellular automaton rule set (≤max_rules rules).
        neighborhood_3x3_fix = self._infer_neighborhood_correction(
            program_outputs, expected_outputs, max_rules=max_rules)
        if neighborhood_3x3_fix is not None:
            return neighborhood_3x3_fix

        # Strategy 4: 5x5 neighborhood patch — same as 3x3 but uses a larger
        # patch for longer-range dependencies (pixel depends on cell 2 away).
        # Stricter cap (30) since 5x5 has higher overfitting risk.
        if try_5x5:
            neighborhood_5x5_fix = self._infer_neighborhood_correction_5x5(
                program_outputs, expected_outputs, max_rules=min(max_rules, 30))
            if neighborhood_5x5_fix is not None:
                return neighborhood_5x5_fix

        # Strategy 5: Row/column-level corrections — detects systematic spatial
        # rearrangements (row/col reversal, cyclic shifts, transpose)
        row_col_fix = self._infer_row_col_correction(program_outputs, expected_outputs)
        if row_col_fix is not None:
            return row_col_fix

        return None

    def _infer_color_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
    ) -> Optional[Program]:
        """Infer a color remapping that fixes mismatches.

        For each (got, expected) grid pair, collects pixel-level color
        mismatches.  If a consistent remap exists (>80% agreement per
        source color), creates a correction Program.
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
            if g not in by_src:
                by_src[g] = Counter()
            by_src[g][w] += count

        # Build remap from consistent transitions (>80% agreement)
        remap: dict[int, int] = {}
        for g, tally in by_src.items():
            best_w, best_count = tally.most_common(1)[0]
            total_wrong = sum(tally.values())
            if best_count / total_wrong < 0.80:
                continue  # ambiguous — skip this color
            remap[g] = best_w

        if not remap:
            return None

        # Verify: applying the remap must fix ALL diffs (not just the ones
        # it covers). If the remap overgeneralizes, fall through to spatial.
        remap_fn = _make_color_remap(remap)
        for got, expected in zip(program_outputs, expected_outputs):
            result = remap_fn(got)
            if np.array(result, dtype=np.int32).tolist() != np.array(expected, dtype=np.int32).tolist():
                return None  # remap doesn't fully fix — try spatial strategies

        # Register the remap as a primitive and return a Program node
        name = f"color_remap_{'_'.join(f'{k}to{v}' for k, v in sorted(remap.items()))}"
        if name not in _PRIM_MAP:
            prim = Primitive(name=name, arity=1, fn=remap_fn, domain="arc")
            _PRIM_MAP[name] = prim
        return Program(root=name)

    def _infer_adjacency_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
    ) -> Optional[Program]:
        """Infer adjacency-based pixel correction.

        Learns rules of the form: "if pixel is color A and has a
        4-neighbor of color B, change to color C".  This covers many
        same-shape tasks where a program gets the geometry right but
        the local context-dependent coloring wrong.
        """
        # Collect adjacency rules: (center_color, neighbor_color_set) → out_color
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
                    # 4-neighbor colors
                    nbrs = set()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            nbrs.add(int(got_arr[nr, nc]))
                    key = (center, frozenset(nbrs))
                    out_color = int(exp_arr[r, c])
                    if key in rules and rules[key] != out_color:
                        return None  # inconsistent
                    rules[key] = out_color

        if not rules:
            return None

        # Verify: applying the rules should fix ALL diffs in training
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            h, w = got_arr.shape
            result = got_arr.copy()
            for r in range(h):
                for c in range(w):
                    center = int(got_arr[r, c])
                    nbrs = set()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            nbrs.add(int(got_arr[nr, nc]))
                    key = (center, frozenset(nbrs))
                    if key in rules:
                        result[r, c] = rules[key]
            if not np.array_equal(result, exp_arr):
                return None

        # Build correction primitive
        # Convert frozenset keys to sorted tuples for hashing
        hashable_rules = {(c, tuple(sorted(ns))): v
                          for (c, ns), v in rules.items()}

        def _make_adjacency_fix(adj_rules):
            def adjacency_fix(grid: Grid) -> Grid:
                arr = np.array(grid, dtype=np.int32)
                result = arr.copy()
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        center = int(arr[r, c])
                        nbrs = set()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                nbrs.add(int(arr[nr, nc]))
                        hkey = (center, tuple(sorted(nbrs)))
                        if hkey in adj_rules:
                            result[r, c] = adj_rules[hkey]
                return result.tolist()
            return adjacency_fix

        name = f"adjacency_fix_{len(rules)}r"
        fn = _make_adjacency_fix(hashable_rules)
        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
        _PRIM_MAP[name] = prim
        return Program(root=name)

    def _infer_neighborhood_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        max_rules: int = 50,
    ) -> Optional[Program]:
        """Infer 3x3 neighborhood-based pixel correction.

        For each mismatched pixel, encodes the full 3x3 neighborhood as
        a feature and learns a mapping to the correct output color.
        More specific than adjacency rules but catches more patterns.
        Bounded to <=max_rules rules to avoid overfitting.
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
                    # 3x3 neighborhood (use -1 for out-of-bounds)
                    patch = []
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                patch.append(int(got_arr[nr, nc]))
                            else:
                                patch.append(-1)
                    key = tuple(patch)
                    out_color = int(exp_arr[r, c])
                    if key in rules and rules[key] != out_color:
                        return None  # inconsistent
                    rules[key] = out_color

        if not rules or len(rules) > max_rules:
            return None  # too many rules → likely overfitting

        # Verify on all training examples
        for got, expected in zip(program_outputs, expected_outputs):
            got_arr = np.array(got, dtype=np.int32)
            exp_arr = np.array(expected, dtype=np.int32)
            h, w = got_arr.shape
            result = got_arr.copy()
            for r in range(h):
                for c in range(w):
                    patch = []
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                patch.append(int(got_arr[nr, nc]))
                            else:
                                patch.append(-1)
                    key = tuple(patch)
                    if key in rules:
                        result[r, c] = rules[key]
            if not np.array_equal(result, exp_arr):
                return None

        # Build correction primitive
        def _make_neighborhood_3x3_fix(nbr_rules):
            def neighborhood_3x3_fix(grid: Grid) -> Grid:
                arr = np.array(grid, dtype=np.int32)
                result = arr.copy()
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        patch = []
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    patch.append(int(arr[nr, nc]))
                                else:
                                    patch.append(-1)
                        key = tuple(patch)
                        if key in nbr_rules:
                            result[r, c] = nbr_rules[key]
                return result.tolist()
            return neighborhood_3x3_fix

        name = f"neighborhood_3x3_fix_{len(rules)}r"
        fn = _make_neighborhood_3x3_fix(dict(rules))
        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
        _PRIM_MAP[name] = prim
        return Program(root=name)

    def _infer_neighborhood_correction_5x5(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        max_rules: int = 30,
    ) -> Optional[Program]:
        """Infer 5x5 neighborhood-based pixel correction.

        Like 3x3 but uses a larger 5x5 patch to capture longer-range
        dependencies (e.g., color depends on pixel 2 cells away).
        Stricter default rule cap (30) since 5x5 has higher overfitting risk.
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
                    # 5x5 neighborhood (use -1 for out-of-bounds)
                    patch = []
                    for dr in (-2, -1, 0, 1, 2):
                        for dc in (-2, -1, 0, 1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                patch.append(int(got_arr[nr, nc]))
                            else:
                                patch.append(-1)
                    key = tuple(patch)
                    out_color = int(exp_arr[r, c])
                    if key in rules and rules[key] != out_color:
                        return None  # inconsistent
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
                    patch = []
                    for dr in (-2, -1, 0, 1, 2):
                        for dc in (-2, -1, 0, 1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                patch.append(int(got_arr[nr, nc]))
                            else:
                                patch.append(-1)
                    key = tuple(patch)
                    if key in rules:
                        result[r, c] = rules[key]
            if not np.array_equal(result, exp_arr):
                return None

        # Build correction primitive
        def _make_neighborhood_5x5_fix(nbr_rules):
            def neighborhood_5x5_fix(grid: Grid) -> Grid:
                arr = np.array(grid, dtype=np.int32)
                result = arr.copy()
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        patch = []
                        for dr in (-2, -1, 0, 1, 2):
                            for dc in (-2, -1, 0, 1, 2):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    patch.append(int(arr[nr, nc]))
                                else:
                                    patch.append(-1)
                        key = tuple(patch)
                        if key in nbr_rules:
                            result[r, c] = nbr_rules[key]
                return result.tolist()
            return neighborhood_5x5_fix

        name = f"neighborhood_5x5_fix_{len(rules)}r"
        fn = _make_neighborhood_5x5_fix(dict(rules))
        prim = Primitive(name=name, arity=1, fn=fn, domain="arc")
        _PRIM_MAP[name] = prim
        return Program(root=name)

    def _infer_row_col_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
    ) -> Optional[Program]:
        """Infer row/column-level corrections.

        Detects systematic row/column transforms in the diff pattern:
        - Row reversal, column reversal
        - Row/column shifting (cyclic rotation)
        - Row/column swaps

        These catch near-misses where the content is right but the
        spatial arrangement is wrong.
        """
        # Only works on same-shape grids, and there must be actual diffs
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

        # Try each row/column transform and check if it fixes ALL examples
        transforms = [
            ("row_reverse", lambda arr: arr[::-1].copy()),
            ("col_reverse", lambda arr: arr[:, ::-1].copy()),
            ("transpose", lambda arr: arr.T.copy() if arr.shape[0] == arr.shape[1] else None),
        ]

        # Add cyclic row/column shifts
        for shift in range(1, 5):
            s = shift  # capture
            transforms.append(
                (f"row_shift_{s}", lambda arr, s=s: np.roll(arr, s, axis=0)))
            transforms.append(
                (f"col_shift_{s}", lambda arr, s=s: np.roll(arr, s, axis=1)))

        for name, transform_fn in transforms:
            all_match = True
            for got, expected in zip(program_outputs, expected_outputs):
                got_arr = np.array(got, dtype=np.int32)
                exp_arr = np.array(expected, dtype=np.int32)
                result = transform_fn(got_arr)
                if result is None or not np.array_equal(result, exp_arr):
                    all_match = False
                    break
            if all_match:
                def _make_row_col_fix(tfn):
                    def row_col_fix(grid: Grid) -> Grid:
                        arr = np.array(grid, dtype=np.int32)
                        result = tfn(arr)
                        if result is None:
                            return grid
                        return result.tolist()
                    return row_col_fix

                prim_name = f"row_col_fix_{name}"
                fn = _make_row_col_fix(transform_fn)
                prim = Primitive(name=prim_name, arity=1, fn=fn, domain="arc")
                _PRIM_MAP[prim_name] = prim
                return Program(root=prim_name)

        return None

    # Maximum intermediate grid size (pixels). Grid-expanding primitives
    # (tile_3x3=9x, scale_5x=25x) composed at depth 2-3 can create grids
    # with millions of pixels. Numba JIT functions on such grids run for
    # minutes/hours in compiled code that ignores Ctrl-C and consumes GBs
    # of RAM. This guard prevents runaway expansion.
    MAX_GRID_PIXELS = 10_000  # ~100x100 — generous for ARC (max 30x30)

    def _eval_tree(self, node: Program, grid: Grid) -> Grid:
        """Recursively evaluate a program tree on a grid."""
        prim = _PRIM_MAP.get(node.root)
        if prim is None:
            # Unknown primitive (possibly a learned library entry)
            # Return grid unchanged to avoid crashes
            return grid

        try:
            if prim.arity == 0:
                # Learned library entries have fn=Program (a stored sub-tree).
                # Execute the stored program recursively.
                if isinstance(prim.fn, Program):
                    return self._eval_tree(prim.fn, grid)
                # Other nullary: return the input grid (identity-like)
                return grid
            elif prim.arity == 1:
                # Unary: apply to the result of the single child
                if node.children:
                    child_grid = self._eval_tree(node.children[0], grid)
                else:
                    child_grid = grid
                # Guard: reject oversized intermediate grids from expansion
                h = len(child_grid)
                w = len(child_grid[0]) if child_grid else 0
                if h * w > self.MAX_GRID_PIXELS:
                    return grid
                result = prim.fn(child_grid)
                if not isinstance(result, list) or not result:
                    return grid
                # Guard: reject oversized output grids
                rh = len(result)
                rw = len(result[0]) if result else 0
                if rh * rw > self.MAX_GRID_PIXELS:
                    return grid
                return result
            elif prim.arity == 2:
                # Binary: apply to results of both children
                left = self._eval_tree(node.children[0], grid) if len(node.children) > 0 else grid
                right = self._eval_tree(node.children[1], grid) if len(node.children) > 1 else grid
                result = prim.fn(left, right)
                if not isinstance(result, list) or not result:
                    return grid
                return result
        except Exception:
            return grid

        return grid
