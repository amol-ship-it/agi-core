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
