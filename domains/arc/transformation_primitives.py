"""
Atomic transformation primitives for ARC-AGI.

Self-contained implementations — no imports from primitives.py.
Each primitive performs exactly ONE visual concept.

Three categories:
1. Transform (Grid → Grid): geometric, spatial, color, morphological, physics
2. Parameterized ((Value,...) → Grid → Grid): color ops, scale/tile with
   perception-derived parameters
3. Binary (Grid, Grid → Grid): overlay

DISCOVERY GOALS — compound operations we want to discover through composition:
- extract_largest_object = find_components + rank_by_size + extract
- extract_smallest_object = find_components + rank_by_size + extract
- keep_largest/smallest_component = find_components + rank + mask
- recolor_by_size_rank = find_components + rank + recolor
- extend_lines_to_contact = find_line_segments + extend_until_collision
- complete_symmetry_90/h/v = detect_partial_symmetry + fill
- upscale_pattern = detect_small_pattern + upscale
- fill_tile_pattern = detect_tiling + fill
- fill_between_diagonal = detect_diagonals + interpolate
- mark_intersections = find_lines + mark_crossings
"""

from __future__ import annotations

from collections import Counter

from core import Primitive

Grid = list[list[int]]


# =============================================================================
# Geometric transforms (6)
# =============================================================================

def rotate_90_cw(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid[::-1])]


def rotate_90_ccw(grid: Grid) -> Grid:
    """Rotate 90 degrees counter-clockwise."""
    if not grid:
        return grid
    return [list(row) for row in zip(*[r[::-1] for r in grid])]


def rotate_180(grid: Grid) -> Grid:
    """Rotate 180 degrees."""
    if not grid:
        return grid
    return [row[::-1] for row in grid[::-1]]


def mirror_horizontal(grid: Grid) -> Grid:
    """Mirror left-right (flip along vertical axis)."""
    if not grid:
        return grid
    return [row[::-1] for row in grid]


def mirror_vertical(grid: Grid) -> Grid:
    """Mirror top-bottom (flip along horizontal axis)."""
    if not grid:
        return grid
    return list(reversed([row[:] for row in grid]))


def transpose(grid: Grid) -> Grid:
    """Transpose (swap rows and columns)."""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)]


# =============================================================================
# Spatial transforms (6)
# =============================================================================

def crop_to_content(grid: Grid) -> Grid:
    """Crop to bounding box of non-zero pixels."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = h, -1, w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return grid
    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


def crop_half_top(grid: Grid) -> Grid:
    """Return the top half of the grid."""
    if not grid:
        return grid
    return [row[:] for row in grid[:len(grid) // 2]]


def crop_half_bottom(grid: Grid) -> Grid:
    """Return the bottom half of the grid."""
    if not grid:
        return grid
    return [row[:] for row in grid[len(grid) // 2:]]


def crop_half_left(grid: Grid) -> Grid:
    """Return the left half of the grid."""
    if not grid or not grid[0]:
        return grid
    mid = len(grid[0]) // 2
    return [row[:mid] for row in grid]


def crop_half_right(grid: Grid) -> Grid:
    """Return the right half of the grid."""
    if not grid or not grid[0]:
        return grid
    mid = len(grid[0]) // 2
    return [row[mid:] for row in grid]


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


# =============================================================================
# Color transforms (2)
# =============================================================================

def binarize(grid: Grid) -> Grid:
    """Convert all non-zero colors to 1."""
    return [[0 if c == 0 else 1 for c in row] for row in grid]


def invert_colors(grid: Grid) -> Grid:
    """Invert colors: c → 9 - c."""
    return [[9 - c for c in row] for row in grid]


# =============================================================================
# Morphological transforms (2)
# =============================================================================

def dilate(grid: Grid) -> Grid:
    """Grow non-zero regions by 1 pixel (4-connectivity)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        result[r][c] = grid[nr][nc]
                        break
    return result


def erode(grid: Grid) -> Grid:
    """Shrink non-zero regions by 1 pixel (4-connectivity)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == 0:
                        result[r][c] = 0
                        break
    return result


# =============================================================================
# Physics transforms (1)
# =============================================================================

def gravity_down(grid: Grid) -> Grid:
    """Move all non-zero pixels down by gravity (within each column)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        non_zero = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            result[h - len(non_zero) + i][c] = val
    return result


# =============================================================================
# Fill transforms (1)
# =============================================================================

def fill_enclosed(grid: Grid) -> Grid:
    """Fill zero-valued pixels enclosed by non-zero pixels.

    Uses BFS from border to find reachable zeros. Unreachable zeros
    are enclosed and get filled with the most common non-zero color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    non_zero = [grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0]
    if not non_zero:
        return grid
    fill_color = Counter(non_zero).most_common(1)[0][0]

    # BFS from border zeros
    reachable = set()
    queue = []
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == 0 and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == 0 and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    while queue:
        cr, cc = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < h and 0 <= nc < w
                    and (nr, nc) not in reachable
                    and grid[nr][nc] == 0):
                reachable.add((nr, nc))
                queue.append((nr, nc))

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in reachable:
                result[r][c] = fill_color
    return result


# =============================================================================
# Binary transforms (1)
# =============================================================================

def overlay(grid1: Grid, grid2: Grid) -> Grid:
    """Overlay two grids: non-zero pixels from grid1 take priority."""
    if not grid1:
        return grid2
    if not grid2:
        return grid1
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    return [[grid1[r][c] if grid1[r][c] != 0 else grid2[r][c]
             for c in range(w)] for r in range(h)]


# =============================================================================
# Parameterized action primitives: (Value, ...) → (Grid → Grid) factories
# =============================================================================

def _swap_colors_factory(c1: int, c2: int):
    """Factory: swap two colors in a grid."""
    def swap(grid: Grid) -> Grid:
        return [[c2 if cell == c1 else c1 if cell == c2 else cell
                 for cell in row] for row in grid]
    return swap


def _replace_color_factory(src: int, dst: int):
    """Factory: replace one color with another."""
    def replace(grid: Grid) -> Grid:
        return [[dst if cell == src else cell for cell in row] for row in grid]
    return replace


def _keep_color_factory(color: int):
    """Factory: keep only pixels of one color, zero the rest."""
    def keep(grid: Grid) -> Grid:
        return [[cell if cell == color else 0 for cell in row] for row in grid]
    return keep


def _erase_color_factory(color: int):
    """Factory: erase pixels of one color (set to 0)."""
    def erase(grid: Grid) -> Grid:
        return [[0 if cell == color else cell for cell in row] for row in grid]
    return erase


def _fill_bg_with_color_factory(color: int):
    """Factory: fill background pixels with given color."""
    def fill(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
        bg = Counter(flat).most_common(1)[0][0]
        return [[color if cell == bg else cell for cell in row] for row in grid]
    return fill


def _scale_factory(n: int):
    """Factory: upscale each pixel to n×n block."""
    if not isinstance(n, int) or n < 1 or n > 10:
        return lambda grid: grid
    def scale(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        return [[grid[r // n][c // n]
                 for c in range(len(grid[0]) * n)]
                for r in range(len(grid) * n)]
    return scale


def _tile_factory(n: int):
    """Factory: tile the grid n×n times."""
    if not isinstance(n, int) or n < 1 or n > 10:
        return lambda grid: grid
    def tile(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        return [[grid[r % h][c % w]
                 for c in range(w * n)]
                for r in range(h * n)]
    return tile


def _downscale_factory(n: int):
    """Factory: downscale by factor n (majority vote per block)."""
    if not isinstance(n, int) or n < 1 or n > 10:
        return lambda grid: grid
    def downscale(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        nh, nw = h // n, w // n
        if nh == 0 or nw == 0:
            return grid
        result = []
        for r in range(nh):
            row = []
            for c in range(nw):
                block = [grid[r * n + dr][c * n + dc]
                         for dr in range(n) for dc in range(n)
                         if r * n + dr < h and c * n + dc < w]
                row.append(Counter(block).most_common(1)[0][0] if block else 0)
            result.append(row)
        return result
    return downscale


# =============================================================================
# Build functions
# =============================================================================

def build_atomic_primitives() -> list[Primitive]:
    """Build the truly atomic transformation primitives.

    18 unary transforms + 1 binary (overlay) = 19 total.
    Each performs exactly ONE visual concept.
    """
    unary_ops = [
        # Geometric (6)
        ("rotate_90_clockwise",         rotate_90_cw),
        ("rotate_90_counterclockwise",  rotate_90_ccw),
        ("rotate_180",                  rotate_180),
        ("mirror_horizontal",           mirror_horizontal),
        ("mirror_vertical",             mirror_vertical),
        ("transpose",                   transpose),
        # Spatial (6)
        ("crop_to_content",             crop_to_content),
        ("crop_half_top",               crop_half_top),
        ("crop_half_bottom",            crop_half_bottom),
        ("crop_half_left",              crop_half_left),
        ("crop_half_right",             crop_half_right),
        ("pad_border",                  pad_border),
        # Color (2)
        ("binarize",                    binarize),
        ("invert_colors",               invert_colors),
        # Morphological (2)
        ("dilate",                      dilate),
        ("erode",                       erode),
        # Physics (1)
        ("gravity_down",                gravity_down),
        # Fill (1)
        ("fill_enclosed",               fill_enclosed),
    ]

    prims = [Primitive(name=name, arity=1, fn=fn, domain="arc")
             for name, fn in unary_ops]
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))
    return prims


def build_parameterized_primitives() -> list[Primitive]:
    """Build parameterized action primitives (factory functions).

    Each takes perception values as arguments and returns a Grid→Grid transform.
    Children in the program tree are perception primitives.
    """
    return [
        # Color parameterized
        Primitive(name="swap_colors", arity=2,
                  fn=_swap_colors_factory, domain="arc", kind="parameterized"),
        Primitive(name="replace_color", arity=2,
                  fn=_replace_color_factory, domain="arc", kind="parameterized"),
        Primitive(name="keep_color", arity=1,
                  fn=_keep_color_factory, domain="arc", kind="parameterized"),
        Primitive(name="erase_color", arity=1,
                  fn=_erase_color_factory, domain="arc", kind="parameterized"),
        Primitive(name="fill_bg_with", arity=1,
                  fn=_fill_bg_with_color_factory, domain="arc", kind="parameterized"),
        # Scale/tile parameterized
        Primitive(name="scale", arity=1,
                  fn=_scale_factory, domain="arc", kind="parameterized"),
        Primitive(name="tile", arity=1,
                  fn=_tile_factory, domain="arc", kind="parameterized"),
        Primitive(name="downscale", arity=1,
                  fn=_downscale_factory, domain="arc", kind="parameterized"),
    ]


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
])
