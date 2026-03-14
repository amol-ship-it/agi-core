"""
Atomic Primitive Decomposition for ARC-AGI.

~27 truly atomic operations + compositional combinators.
Each atomic performs exactly ONE visual concept — no embedded object detection
or iteration. Combinators wrap inner programs with perception logic.

Design:
- Region-level atomics (not pixel-level)
- Perception embedded in combinators (Grid->Grid everywhere)
- New vocabulary="atomic" mode alongside existing full/minimal
"""

from __future__ import annotations

from typing import Callable, Optional

from core import Primitive, Program
from .primitives import (
    Grid, register_prim,
    # Geometric — reuse directly
    identity, rotate_90_cw, rotate_90_ccw, rotate_180,
    mirror_horizontal, mirror_vertical, transpose,
    # Spatial — reuse directly
    crop_to_nonzero, get_top_half, get_bottom_half,
    get_left_half, get_right_half,
    # Scale — reuse directly
    scale_2x, scale_3x, downscale_2x,
    # Color — reuse directly
    binarize, invert_colors,
    # Placement — reuse directly
    overlay, tile_2x2,
    # Task color primitives — reuse for atomic subset
    build_task_color_primitives,
    _make_keep_color, _make_erase_color, _make_replace_color,
    _make_swap_colors,
)


# =============================================================================
# 1A. Atomic Action Primitives (~27, all Grid -> Grid)
# =============================================================================

# --- Pixel/Region (3) — new ---

def fill_region(grid: Grid, pixels: set[tuple[int, int]], color: int) -> Grid:
    """Set a set of pixels to a color."""
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0]) if grid else 0
    for r, c in pixels:
        if 0 <= r < h and 0 <= c < w:
            result[r][c] = color
    return result


def copy_region(grid: Grid, src_pixels: set[tuple[int, int]],
                dest_offset: tuple[int, int]) -> Grid:
    """Copy pixels to new position (offset = (dr, dc))."""
    result = [row[:] for row in grid]
    h, w = len(grid), len(grid[0]) if grid else 0
    dr, dc = dest_offset
    for r, c in src_pixels:
        if 0 <= r < h and 0 <= c < w:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = grid[r][c]
    return result


def clear_region(grid: Grid, pixels: set[tuple[int, int]]) -> Grid:
    """Set pixels to background (0)."""
    return fill_region(grid, pixels, 0)


# --- Spatial (1 new) — pad_border ---

def pad_border(grid: Grid) -> Grid:
    """Add 1-pixel border of zeros around the grid."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * (w + 2) for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            result[r + 1][c + 1] = grid[r][c]
    return result


# --- Color (1 new) — recolor (parameterized old->new) ---

def recolor(grid: Grid, old_color: int, new_color: int) -> Grid:
    """Replace old_color with new_color throughout grid."""
    return [[new_color if c == old_color else c for c in row] for row in grid]


def erase_color(grid: Grid, color: int) -> Grid:
    """Replace a specific color with background (0)."""
    return [[0 if c == color else c for c in row] for row in grid]


# --- Morphological (2) — new, simplified ---

def dilate(grid: Grid) -> Grid:
    """Grow non-zero regions by 1 pixel (4-connectivity)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                # Check 4-neighbors for any non-zero
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        result[r][c] = grid[nr][nc]
                        break
    return result


def erode(grid: Grid) -> Grid:
    """Shrink non-zero regions by 1 pixel (remove pixels with any zero 4-neighbor)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                # If any 4-neighbor is zero or boundary, erode
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == 0:
                        result[r][c] = 0
                        break
    return result


# =============================================================================
# 1B. Combinator Generators (3 types)
# =============================================================================

def make_for_each_object(inner_fn: Callable, bg_color: int = 0) -> Callable:
    """Grid->Grid: find objects, apply inner_fn to each, reassemble.

    Uses 4-connectivity same-color object detection.
    """
    from .objects import apply_transform_per_object

    def transform(grid: Grid) -> Grid:
        result = apply_transform_per_object(grid, inner_fn, bg_color)
        return result if result is not None else grid
    return transform


def make_apply_to_enclosed(inner_fn: Callable) -> Callable:
    """Grid->Grid: find enclosed bg regions, apply inner_fn to each.

    Detects background regions not reachable from grid border, extracts
    each as a subgrid, transforms it, and places it back.
    """
    from .objects import _get_background_color, place_subgrid

    def transform(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        bg = _get_background_color(grid)

        # BFS from border to find non-enclosed bg pixels
        reachable: set[tuple[int, int]] = set()
        queue: list[tuple[int, int]] = []
        for r in range(h):
            for c in (0, w - 1):
                if grid[r][c] == bg and (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if grid[r][c] == bg and (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
        while queue:
            cr, cc = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < h and 0 <= nc < w
                        and (nr, nc) not in reachable
                        and grid[nr][nc] == bg):
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

        # Find enclosed bg regions via flood fill
        enclosed_visited: set[tuple[int, int]] = set()
        regions: list[set[tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if (grid[r][c] == bg and (r, c) not in reachable
                        and (r, c) not in enclosed_visited):
                    region: set[tuple[int, int]] = set()
                    rq: list[tuple[int, int]] = [(r, c)]
                    while rq:
                        pr, pc = rq.pop()
                        if (pr, pc) in enclosed_visited:
                            continue
                        if (pr < 0 or pr >= h or pc < 0 or pc >= w
                                or grid[pr][pc] != bg or (pr, pc) in reachable):
                            continue
                        enclosed_visited.add((pr, pc))
                        region.add((pr, pc))
                        rq.extend([(pr-1, pc), (pr+1, pc), (pr, pc-1), (pr, pc+1)])
                    if region:
                        regions.append(region)

        if not regions:
            return grid

        result = [row[:] for row in grid]
        for region in regions:
            rows = [p[0] for p in region]
            cols = [p[1] for p in region]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            rh, rw = max_r - min_r + 1, max_c - min_c + 1
            subgrid = [[bg] * rw for _ in range(rh)]
            for pr, pc in region:
                subgrid[pr - min_r][pc - min_c] = grid[pr][pc]
            transformed = inner_fn(subgrid)
            if isinstance(transformed, list) and transformed:
                result = place_subgrid(result, transformed, (min_r, min_c),
                                       transparent_color=bg)
        return result
    return transform


def make_conditional_objects(
    predicate: Callable[[Grid], bool],
    prog_true_fn: Callable,
    prog_false_fn: Callable,
    bg_color: int = 0,
) -> Callable:
    """Grid->Grid: per-object if(pred, A, B).

    For each object, applies prog_true_fn if predicate(subgrid) is True,
    otherwise applies prog_false_fn.
    """
    from .objects import find_foreground_shapes, place_subgrid

    def transform(grid: Grid) -> Grid:
        shapes = find_foreground_shapes(grid)
        if not shapes:
            return grid
        h, w = len(grid), len(grid[0]) if grid else 0
        canvas = [[bg_color] * w for _ in range(h)]
        for shape in shapes:
            sub = shape["subgrid"]
            try:
                fn = prog_true_fn if predicate(sub) else prog_false_fn
                transformed = fn(sub)
                if not isinstance(transformed, list) or not transformed:
                    transformed = sub
            except Exception:
                transformed = sub
            canvas = place_subgrid(canvas, transformed, shape["position"],
                                   transparent_color=bg_color)
        return canvas
    return transform


# =============================================================================
# Build functions
# =============================================================================

def build_atomic_primitives() -> list[Primitive]:
    """Build the atomic primitives vocabulary (~27 ops + overlay).

    Each performs exactly ONE visual concept. No embedded object detection.
    """
    unary_ops = [
        # --- Geometric (7): reuse from primitives.py ---
        ("identity",                    identity),
        ("rotate_90_clockwise",         rotate_90_cw),
        ("rotate_90_counterclockwise",  rotate_90_ccw),
        ("rotate_180",                  rotate_180),
        ("mirror_horizontal",           mirror_horizontal),
        ("mirror_vertical",             mirror_vertical),
        ("transpose",                   transpose),

        # --- Spatial (6): reuse/simplify from primitives.py ---
        ("crop_to_content",             crop_to_nonzero),  # alias for clarity
        ("crop_half_top",               get_top_half),
        ("crop_half_bottom",            get_bottom_half),
        ("crop_half_left",              get_left_half),
        ("crop_half_right",             get_right_half),
        ("pad_border",                  pad_border),

        # --- Scale (3): reuse ---
        ("scale_2x",                    scale_2x),
        ("scale_3x",                    scale_3x),
        ("downscale_2x",               downscale_2x),

        # --- Color (2): reuse ---
        ("binarize",                    binarize),
        ("invert_colors",              invert_colors),

        # --- Placement (1 unary): tile ---
        ("tile_2x2",                    tile_2x2),

        # --- Morphological (2): new ---
        ("dilate",                      dilate),
        ("erode",                       erode),
    ]

    prims = []
    for name, fn in unary_ops:
        prims.append(Primitive(name=name, arity=1, fn=fn, domain="arc"))

    # Arity-2: overlay (OR-combine two grids)
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))

    return prims


def build_atomic_task_color_primitives(task_colors: set[int]) -> list[Primitive]:
    """Generate only the atomic color subset for a task's palette.

    Includes: keep_only_color_N, erase_color_N, recolor_A_to_B, swap_A_and_B.
    Excludes composite color primitives like fill_rectangle_interior_color_N.
    """
    prims: list[Primitive] = []
    colors = sorted(task_colors - {0})

    # Keep only pixels of a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"keep_only_color_{c}", arity=1,
            fn=_make_keep_color(c), domain="arc",
        ))

    # Erase a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"erase_color_{c}", arity=1,
            fn=_make_erase_color(c), domain="arc",
        ))

    # Recolor A to B (one direction)
    for a in colors:
        for b in colors:
            if a != b:
                prims.append(Primitive(
                    name=f"replace_color_{a}_with_color_{b}", arity=1,
                    fn=_make_replace_color(a, b), domain="arc",
                ))

    # Swap two colors
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            prims.append(Primitive(
                name=f"swap_color_{a}_and_color_{b}", arity=1,
                fn=_make_swap_colors(a, b), domain="arc",
            ))

    return prims


# =============================================================================
# Essential pair concepts for atomic vocabulary
# =============================================================================

ATOMIC_ESSENTIAL_PAIR_CONCEPTS: frozenset = frozenset([
    "crop_to_content",
    "overlay",
    "mirror_horizontal",
    "mirror_vertical",
    "binarize",
    "dilate",
    "erode",
    "scale_2x",
])


# =============================================================================
# Combinator expansion
# =============================================================================

def expand_with_combinators(
    scored_programs: list,
    task,
    env,
    top_k: int = 5,
) -> list[Primitive]:
    """Create combinator-wrapped versions of top-K scored programs.

    Takes scored programs from enumeration and creates:
    1. for_each_object(program) variants
    2. apply_to_enclosed(program) variants

    Returns additional primitives to include in the enumeration pool.
    Called from grammar, not learner — keeps core/ clean.
    """
    from .objects import _get_background_color

    if not scored_programs or not task.train_examples:
        return []

    bg = _get_background_color(task.train_examples[0][0])
    sorted_cands = sorted(scored_programs, key=lambda s: s.prediction_error)[:top_k]
    new_prims: list[Primitive] = []

    for sp in sorted_cands:
        prog = sp.program

        # 1. for_each_object(program)
        def _make_feo(p=prog, e=env, b=bg):
            def inner_fn(subgrid):
                return e.execute(p, subgrid)
            return make_for_each_object(inner_fn, b)

        feo_fn = _make_feo()
        feo_name = f"for_each_object({repr(prog)})"
        new_prims.append(Primitive(name=feo_name, arity=1, fn=feo_fn, domain="arc"))

        # 2. apply_to_enclosed(program)
        def _make_ate(p=prog, e=env):
            def inner_fn(subgrid):
                return e.execute(p, subgrid)
            return make_apply_to_enclosed(inner_fn)

        ate_fn = _make_ate()
        ate_name = f"apply_to_enclosed({repr(prog)})"
        new_prims.append(Primitive(name=ate_name, arity=1, fn=ate_fn, domain="arc"))

    return new_prims
