"""
Deterministic Input-Output Analysis for ARC tasks.

Before searching, compute a task signature from all training examples.
This signature enables principled phase ordering and pruning without
any neural component — a purely algorithmic decision tree based on
observable properties.

Task signature fields:
  - dim_relation: same / scaled / different
  - color_relation: preserved / subset / superset / remapped
  - pixel_relation: subset / superset / overlap / disjoint
  - object_count_change: same / increased / decreased / variable
  - has_symmetry: h / v / both / none
  - scale_factor: (h_factor, w_factor) if dimensions are integer multiples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .primitives import Grid, to_np


@dataclass(frozen=True)
class TaskSignature:
    """Deterministic analysis of a task's input-output relationship."""
    dim_relation: str          # "same", "scaled", "different"
    color_relation: str        # "preserved", "subset", "superset", "remapped"
    pixel_relation: str        # "subset", "superset", "overlap", "disjoint"
    object_count_change: str   # "same", "increased", "decreased", "variable"
    has_symmetry: str          # "h", "v", "both", "none"
    scale_factor: Optional[tuple[int, int]] = None  # (h_factor, w_factor) if scaled
    output_smaller: bool = False
    output_larger: bool = False
    same_nonzero_count: bool = False
    bg_color: int = 0

    # Phase ordering hints derived from signature
    skip_gravity: bool = False
    skip_per_object: bool = False
    prioritize_cross_ref: bool = False
    prioritize_scale_tile: bool = False
    prioritize_color_ops: bool = False
    prioritize_symmetry: bool = False
    recommended_phases: list[str] = field(default_factory=list)


def analyze_task(train_examples: list[tuple[Grid, Grid]]) -> TaskSignature:
    """Compute deterministic task signature from training examples.

    Cheap to compute: O(pixels × examples). No search involved.
    """
    if not train_examples:
        return TaskSignature(
            dim_relation="same", color_relation="preserved",
            pixel_relation="overlap", object_count_change="same",
            has_symmetry="none",
        )

    dim_rels = []
    color_rels = []
    pixel_rels = []
    obj_count_changes = []
    symmetries = []
    scale_factors = []
    size_rels = []

    bg_color = _detect_bg_color(train_examples)

    for inp, out in train_examples:
        inp_np = to_np(inp)
        out_np = to_np(out)

        # Dimension relationship
        dim_rel, sf = _analyze_dimensions(inp_np, out_np)
        dim_rels.append(dim_rel)
        if sf is not None:
            scale_factors.append(sf)

        # Size relationship
        if inp_np.size > out_np.size:
            size_rels.append("smaller")
        elif inp_np.size < out_np.size:
            size_rels.append("larger")
        else:
            size_rels.append("same")

        # Color relationship
        color_rels.append(_analyze_colors(inp_np, out_np, bg_color))

        # Pixel relationship
        pixel_rels.append(_analyze_pixels(inp_np, out_np))

        # Object count change
        obj_count_changes.append(_analyze_object_counts(inp_np, out_np, bg_color))

        # Symmetry in output
        symmetries.append(_detect_symmetry(out_np))

    # Consensus across examples
    dim_relation = _consensus(dim_rels, "different")
    color_relation = _consensus(color_rels, "remapped")
    pixel_relation = _consensus(pixel_rels, "overlap")
    obj_change = _consensus(obj_count_changes, "variable")
    symmetry = _consensus(symmetries, "none")

    scale_factor = scale_factors[0] if scale_factors and len(set(scale_factors)) == 1 else None
    output_smaller = all(r == "smaller" for r in size_rels)
    output_larger = all(r == "larger" for r in size_rels)

    # Check if nonzero pixel count is preserved
    same_nz = all(
        np.count_nonzero(to_np(inp) != bg_color) == np.count_nonzero(to_np(out) != bg_color)
        for inp, out in train_examples
    )

    # Derive phase ordering hints
    skip_gravity = dim_relation != "same"
    skip_per_object = (obj_change == "decreased" and output_smaller)
    prioritize_cross_ref = output_smaller or dim_relation == "different"
    prioritize_scale_tile = scale_factor is not None
    prioritize_color_ops = (
        dim_relation == "same" and color_relation in ("remapped", "subset", "superset")
    )
    prioritize_symmetry = symmetry in ("h", "v", "both")

    # Build recommended phase ordering
    recommended = _build_phase_ordering(
        dim_relation, color_relation, pixel_relation, obj_change,
        symmetry, scale_factor, output_smaller, output_larger,
        prioritize_color_ops, prioritize_symmetry,
    )

    return TaskSignature(
        dim_relation=dim_relation,
        color_relation=color_relation,
        pixel_relation=pixel_relation,
        object_count_change=obj_change,
        has_symmetry=symmetry,
        scale_factor=scale_factor,
        output_smaller=output_smaller,
        output_larger=output_larger,
        same_nonzero_count=same_nz,
        bg_color=bg_color,
        skip_gravity=skip_gravity,
        skip_per_object=skip_per_object,
        prioritize_cross_ref=prioritize_cross_ref,
        prioritize_scale_tile=prioritize_scale_tile,
        prioritize_color_ops=prioritize_color_ops,
        prioritize_symmetry=prioritize_symmetry,
        recommended_phases=recommended,
    )


# ---------------------------------------------------------------------------
# Dimension analysis
# ---------------------------------------------------------------------------

def _analyze_dimensions(inp: np.ndarray, out: np.ndarray) -> tuple[str, Optional[tuple[int, int]]]:
    """Compare input/output dimensions."""
    ih, iw = inp.shape
    oh, ow = out.shape

    if ih == oh and iw == ow:
        return "same", None

    # Check for integer scaling
    if ih > 0 and iw > 0:
        h_factor = oh / ih
        w_factor = ow / iw
        if h_factor == int(h_factor) and w_factor == int(w_factor) and h_factor >= 1 and w_factor >= 1:
            return "scaled", (int(h_factor), int(w_factor))
        # Check for downscaling
        if ih % oh == 0 and iw % ow == 0:
            return "scaled", (oh // ih if ih > 0 else 1, ow // iw if iw > 0 else 1)

    return "different", None


# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------

def _analyze_colors(inp: np.ndarray, out: np.ndarray, bg: int) -> str:
    """Analyze color set relationship between input and output."""
    inp_colors = set(inp.flatten()) - {bg}
    out_colors = set(out.flatten()) - {bg}

    if inp_colors == out_colors:
        return "preserved"
    if out_colors < inp_colors:
        return "subset"
    if inp_colors < out_colors:
        return "superset"
    return "remapped"


# ---------------------------------------------------------------------------
# Pixel analysis
# ---------------------------------------------------------------------------

def _analyze_pixels(inp: np.ndarray, out: np.ndarray) -> str:
    """Analyze pixel relationship (for same-dimension grids)."""
    if inp.shape != out.shape:
        return "different_dims"

    inp_fg = set(zip(*np.where(inp != 0)))
    out_fg = set(zip(*np.where(out != 0)))

    if not inp_fg and not out_fg:
        return "overlap"
    if out_fg <= inp_fg:
        return "subset"
    if inp_fg <= out_fg:
        return "superset"
    if inp_fg & out_fg:
        return "overlap"
    return "disjoint"


# ---------------------------------------------------------------------------
# Object count analysis
# ---------------------------------------------------------------------------

def _analyze_object_counts(inp: np.ndarray, out: np.ndarray, bg: int) -> str:
    """Compare object counts between input and output."""
    inp_count = _count_objects(inp, bg)
    out_count = _count_objects(out, bg)

    if inp_count == out_count:
        return "same"
    if out_count > inp_count:
        return "increased"
    return "decreased"


def _count_objects(grid: np.ndarray, bg: int) -> int:
    """Count connected components (4-connectivity) excluding background."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and not visited[r, c]:
                # BFS flood fill
                count += 1
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != bg:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    return count


# ---------------------------------------------------------------------------
# Symmetry detection
# ---------------------------------------------------------------------------

def _detect_symmetry(grid: np.ndarray) -> str:
    """Detect horizontal/vertical symmetry in a grid."""
    h_sym = np.array_equal(grid, grid[:, ::-1])
    v_sym = np.array_equal(grid, grid[::-1, :])

    if h_sym and v_sym:
        return "both"
    if h_sym:
        return "h"
    if v_sym:
        return "v"
    return "none"


# ---------------------------------------------------------------------------
# Background color detection
# ---------------------------------------------------------------------------

def _detect_bg_color(train_examples: list[tuple[Grid, Grid]]) -> int:
    """Detect the most common background color across all examples."""
    color_counts: dict[int, int] = {}
    for inp, out in train_examples:
        for grid in [inp, out]:
            for row in grid:
                for val in row:
                    color_counts[val] = color_counts.get(val, 0) + 1

    if not color_counts:
        return 0

    # Background is typically the most frequent color AND appears at corners
    corner_colors: dict[int, int] = {}
    for inp, out in train_examples:
        for grid in [inp, out]:
            if grid and grid[0]:
                for val in [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]:
                    corner_colors[val] = corner_colors.get(val, 0) + 1

    # Prefer corner-dominant color, fallback to most frequent
    if corner_colors:
        most_corner = max(corner_colors, key=corner_colors.get)
        most_frequent = max(color_counts, key=color_counts.get)
        # Use corner color if it's also frequent (>30% of total)
        total = sum(color_counts.values())
        if color_counts.get(most_corner, 0) / max(total, 1) > 0.3:
            return most_corner
        return most_frequent

    return max(color_counts, key=color_counts.get)


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------

def _build_phase_ordering(
    dim_rel: str, color_rel: str, pixel_rel: str, obj_change: str,
    symmetry: str, scale_factor: Optional[tuple[int, int]],
    output_smaller: bool, output_larger: bool,
    prioritize_color: bool, prioritize_sym: bool,
) -> list[str]:
    """Build recommended phase ordering based on task signature.

    Returns a list of phase hints that the learner can use to
    reorder its wake phases for this specific task.
    """
    phases = []

    # Scale/tile detection is cheap and high-confidence when scale_factor exists
    if scale_factor is not None:
        phases.append("prioritize_scale_tile")

    # Cross-reference is good when output is smaller (extraction pattern)
    if output_smaller or dim_rel == "different":
        phases.append("prioritize_cross_reference")

    # Color operations when colors change but dimensions don't
    if prioritize_color:
        phases.append("prioritize_color_ops")

    # Symmetry completion
    if prioritize_sym:
        phases.append("prioritize_symmetry")

    # Per-object when objects are preserved
    if obj_change == "same" and dim_rel == "same":
        phases.append("prioritize_per_object")

    # Skip gravity when dimensions change (gravity preserves dimensions)
    if dim_rel != "same":
        phases.append("skip_gravity")

    return phases


# ---------------------------------------------------------------------------
# Consensus helper
# ---------------------------------------------------------------------------

def _consensus(values: list[str], default: str) -> str:
    """Return the value if all examples agree, else the default."""
    if not values:
        return default
    if all(v == values[0] for v in values):
        return values[0]
    return default
