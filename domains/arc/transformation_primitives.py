"""
Atomic transformation primitives for ARC-AGI.

Self-contained implementations — no imports from primitives.py.
Each primitive performs exactly ONE visual concept.

Three categories:
1. Transform (Grid → Grid): geometric, spatial, color, morphological, physics
2. Parameterized ((Value,...) → Grid → Grid): color ops, scale/tile with
   perception-derived parameters
3. Binary (Grid, Grid → Grid): overlay, mask_by

Atomicity principle: each primitive is one intuitive visual concept.

STRIPPED TO ZERO: Rebuilding one primitive at a time, justified by specific tasks.
"""

from __future__ import annotations

from core import Primitive

Grid = list[list[int]]


# =============================================================================
# Build functions — empty, will be populated one primitive at a time
# =============================================================================

def build_atomic_primitives() -> list[Primitive]:
    """Build atomic transformation primitives. Currently empty — rebuilding."""
    return []


def build_parameterized_primitives() -> list[Primitive]:
    """Build parameterized action primitives. Currently empty — rebuilding."""
    return []


# Essential pair concepts — empty until we have primitives
ATOMIC_ESSENTIAL_PAIR_CONCEPTS: frozenset = frozenset()
