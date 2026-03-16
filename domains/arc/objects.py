"""
Connected component detection and object decomposition for ARC grids.

STRIPPED TO ZERO: Only the test helper remains. Everything else
will be added back when justified by a specific task.
"""

from __future__ import annotations

from typing import Optional, Callable

from .primitives import Grid


def _test_on_examples(fn: Callable, examples: list[tuple]) -> bool:
    """Test if a function produces pixel-perfect output on all examples."""
    for inp, expected in examples:
        try:
            result = fn(inp)
            if result != expected:
                return False
        except Exception:
            return False
    return True


def try_object_decomposition(
    task_examples: list[tuple],
    primitives: list,
) -> Optional[tuple[str, Callable]]:
    """Try to solve a task by applying the same transform per object.
    Currently disabled — returns None."""
    return None
