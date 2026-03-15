"""
ARC primitive registry and utilities.

This module provides:
- Grid type alias
- _PRIM_MAP: the global execution registry used by _eval_tree
- register_prim / register_atomic_primitives: populate the registry
- to_np / from_np: numpy conversion utilities

All primitive implementations live in:
- transformation_primitives.py (Grid→Grid transforms + parameterized factories)
- perception_primitives.py (Grid→Value perception)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core import Primitive

# Type alias for grids
Grid = list[list[int]]


# =============================================================================
# Grid utilities
# =============================================================================

def to_np(grid: Grid) -> np.ndarray:
    """Convert list-of-lists grid to numpy array."""
    return np.array(grid, dtype=np.int32)


def from_np(arr: np.ndarray) -> Grid:
    """Convert numpy array back to list-of-lists."""
    return arr.tolist()


# =============================================================================
# Primitive registry — used by _eval_tree in environment.py
# =============================================================================

_PRIM_MAP: dict[str, Primitive] = {}


def register_prim(p: Primitive) -> None:
    """Register a primitive in the lookup map."""
    _PRIM_MAP[p.name] = p


def register_atomic_primitives() -> None:
    """Register atomic, perception, and parameterized primitives in _PRIM_MAP."""
    from .transformation_primitives import build_atomic_primitives, build_parameterized_primitives
    from .perception_primitives import build_perception_primitives
    for p in build_atomic_primitives():
        if p.name not in _PRIM_MAP:
            _PRIM_MAP[p.name] = p
    for p in build_perception_primitives():
        _PRIM_MAP[p.name] = p
    for p in build_parameterized_primitives():
        _PRIM_MAP[p.name] = p


def lookup_prim(name: str) -> Optional[Primitive]:
    """Look up a primitive by name."""
    return _PRIM_MAP.get(name)
