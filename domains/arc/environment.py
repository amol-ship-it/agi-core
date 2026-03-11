"""
ARC-AGI Environment: execute grid transformation programs.
"""

from __future__ import annotations

from typing import Any, Optional

from collections import Counter

import numpy as np

from core import Environment, Primitive, Program, Task, Observation
from .primitives import Grid, _PRIM_MAP, _make_color_remap


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

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
    ) -> Optional[Program]:
        """Infer a color remapping that fixes mismatches between outputs.

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

        # Check consistency: each source color must map to one target >80%
        remap: dict[int, int] = {}
        for g, tally in by_src.items():
            best_w, best_count = tally.most_common(1)[0]
            total = sum(tally.values())
            if best_count / total < 0.80:
                return None  # ambiguous remap
            remap[g] = best_w

        if not remap:
            return None

        # Register the remap as a primitive and return a Program node
        name = f"color_remap_{'_'.join(f'{k}to{v}' for k, v in sorted(remap.items()))}"
        if name not in _PRIM_MAP:
            prim = Primitive(name=name, arity=1, fn=_make_color_remap(remap), domain="arc")
            _PRIM_MAP[name] = prim
        return Program(root=name)

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
