"""
ARC-AGI Environment: execute grid transformation programs.
"""

from __future__ import annotations

from typing import Any, Optional

from core import Environment, Program, Task, Observation
from .primitives import Grid, _PRIM_MAP


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

    def _eval_tree(self, node: Program, grid: Grid) -> Grid:
        """Recursively evaluate a program tree on a grid."""
        prim = _PRIM_MAP.get(node.root)
        if prim is None:
            # Unknown primitive (possibly a learned library entry)
            # Return grid unchanged to avoid crashes
            return grid

        try:
            if prim.arity == 0:
                # Nullary: return the input grid (identity-like)
                return grid
            elif prim.arity == 1:
                # Unary: apply to the result of the single child
                if node.children:
                    child_grid = self._eval_tree(node.children[0], grid)
                else:
                    child_grid = grid
                result = prim.fn(child_grid)
                if not isinstance(result, list) or not result:
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
