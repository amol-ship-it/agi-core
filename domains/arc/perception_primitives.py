"""
Atomic perception primitives: Grid → Value.

Each extracts exactly ONE property from a grid. These are the "eyes"
that feed into parameterized action primitives.

STRIPPED TO ZERO: Rebuilding one primitive at a time, justified by specific tasks.
"""

from __future__ import annotations

from core import Primitive


Grid = list[list[int]]


def build_perception_primitives() -> list[Primitive]:
    """Build perception primitives. Currently empty — rebuilding."""
    return []
