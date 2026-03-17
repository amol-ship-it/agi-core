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
# Primitives — added one at a time, justified by specific tasks
# =============================================================================

# --- Geometric transforms ---

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
    """Transpose (swap rows and columns). Justifying task: 9dfd6313."""
    if not grid:
        return grid
    return [list(row) for row in zip(*grid)]


# --- Spatial transforms ---

def trim_rows(grid: Grid) -> Grid:
    """Remove leading and trailing all-zero rows."""
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    top = 0
    while top < h and all(c == 0 for c in grid[top]):
        top += 1
    bot = h - 1
    while bot >= top and all(c == 0 for c in grid[bot]):
        bot -= 1
    if top > bot:
        return grid
    return [row[:] for row in grid[top:bot + 1]]


def trim_cols(grid: Grid) -> Grid:
    """Remove leading and trailing all-zero columns."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    left = 0
    while left < w and all(grid[r][left] == 0 for r in range(h)):
        left += 1
    right = w - 1
    while right >= left and all(grid[r][right] == 0 for r in range(h)):
        right -= 1
    if left > right:
        return grid
    return [row[left:right + 1] for row in grid]


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


# --- Color transforms ---

def binarize(grid: Grid) -> Grid:
    """Convert all non-zero colors to 1."""
    return [[0 if c == 0 else 1 for c in row] for row in grid]


def invert_colors(grid: Grid) -> Grid:
    """Invert colors: c → 9 - c."""
    return [[9 - c for c in row] for row in grid]


# --- Physics transforms ---

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


def gravity_up(grid: Grid) -> Grid:
    """Move all non-zero pixels up by gravity (within each column)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        non_zero = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            result[i][c] = val
    return result


def gravity_left(grid: Grid) -> Grid:
    """Move all non-zero pixels left by gravity (within each row)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        non_zero = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            result[r][i] = val
    return result


def gravity_right(grid: Grid) -> Grid:
    """Move all non-zero pixels right by gravity (within each row)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        non_zero = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            result[r][w - len(non_zero) + i] = val
    return result


# --- Fill transforms ---

def fill_enclosed(grid: Grid) -> Grid:
    """Fill zero-valued pixels enclosed by non-zero pixels."""
    from collections import Counter
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    non_zero = [grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0]
    if not non_zero:
        return grid
    fill_color = Counter(non_zero).most_common(1)[0][0]
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


# --- Morphological transforms ---

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


# --- Sorting transforms ---

def sort_rows_by_nonzero(grid: Grid) -> Grid:
    """Sort rows ascending by count of non-zero pixels."""
    if not grid or not grid[0]:
        return grid
    return sorted([row[:] for row in grid], key=lambda r: sum(1 for c in r if c != 0))


def sort_cols_by_nonzero(grid: Grid) -> Grid:
    """Sort columns ascending by count of non-zero pixels."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    cols = [[grid[r][c] for r in range(h)] for c in range(w)]
    cols.sort(key=lambda col: sum(1 for v in col if v != 0))
    return [[cols[c][r] for c in range(w)] for r in range(h)]


# --- Extraction transforms ---

def extract_largest_cc(grid: Grid) -> Grid:
    """Extract bounding box of the largest connected component.

    Finds the largest 4-connected non-zero region and returns its
    bounding box as a cropped subgrid.

    Justified by tasks be94b721, 1f85a75f.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    best_comp: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                if len(comp) > len(best_comp):
                    best_comp = comp

    if not best_comp:
        return grid
    rows_c = [p[0] for p in best_comp]
    cols_c = [p[1] for p in best_comp]
    r0, r1 = min(rows_c), max(rows_c)
    c0, c1 = min(cols_c), max(cols_c)
    return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


def extract_unique_color_region(grid: Grid) -> Grid:
    """Extract bounding box of the region with the rarest non-zero color.

    Finds the least-common non-zero color and returns the bounding box
    containing all pixels of that color.

    Justified by tasks c909285e, 0b148d64, 23b5c85d.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    h, w = len(grid), len(grid[0])
    colors = Counter(c for row in grid for c in row if c != 0)
    if not colors:
        return grid
    rarest = colors.most_common()[-1][0]
    rows = [r for r in range(h) for c in range(w) if grid[r][c] == rarest]
    cols = [c for r in range(h) for c in range(w) if grid[r][c] == rarest]
    if not rows:
        return grid
    return [grid[r][min(cols):max(cols) + 1] for r in range(min(rows), max(rows) + 1)]


# --- Inpainting transforms ---

def inpaint_periodic(grid: Grid) -> Grid:
    """Fill zeros by detecting and extrapolating the periodic tile pattern.

    Algorithm: try every possible tile size (ph, pw). For each, check if
    all non-zero cells are consistent with grid[r][c] == tile[r%ph][c%pw].
    Use the smallest consistent tile to fill zeros.

    Justified by tasks 73251a56, 29ec7d0e.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Check if there are any zeros to fill
    has_zero = any(grid[r][c] == 0 for r in range(h) for c in range(w))
    if not has_zero:
        return grid

    # Try all tile periods from smallest to largest
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            # Build tile from non-zero cells
            tile = [[None] * pw for _ in range(ph)]
            consistent = True
            for r in range(h):
                if not consistent:
                    break
                for c in range(w):
                    val = grid[r][c]
                    if val == 0:
                        continue
                    tr, tc = r % ph, c % pw
                    if tile[tr][tc] is None:
                        tile[tr][tc] = val
                    elif tile[tr][tc] != val:
                        consistent = False
                        break

            if not consistent:
                continue

            # Check tile is fully determined (no None cells)
            if any(tile[r][c] is None for r in range(ph) for c in range(pw)):
                continue

            # Fill zeros using the tile
            result = [row[:] for row in grid]
            for r in range(h):
                for c in range(w):
                    if result[r][c] == 0:
                        result[r][c] = tile[r % ph][c % pw]
            return result

    # No consistent tile found — return original
    return grid


# --- Binary transforms ---

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


def mask_by(grid1: Grid, grid2: Grid) -> Grid:
    """Keep pixels from grid1 where grid2 is non-zero, zero elsewhere."""
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    return [[grid1[r][c] if grid2[r][c] != 0 else 0
             for c in range(w)] for r in range(h)]


# =============================================================================
# Build functions
# =============================================================================

def build_atomic_primitives() -> list[Primitive]:
    """Build atomic transformation primitives.

    Each primitive is justified by specific ARC tasks.
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
        ("trim_rows",                   trim_rows),
        ("trim_cols",                   trim_cols),
        ("crop_half_top",               crop_half_top),
        ("crop_half_bottom",            crop_half_bottom),
        ("crop_half_left",              crop_half_left),
        ("crop_half_right",             crop_half_right),
        # Color (2)
        ("binarize",                    binarize),
        ("invert_colors",               invert_colors),
        # Physics (4)
        ("gravity_down",                gravity_down),
        ("gravity_up",                  gravity_up),
        ("gravity_left",                gravity_left),
        ("gravity_right",               gravity_right),
        # Fill (1)
        ("fill_enclosed",               fill_enclosed),
        # Morphological (2)
        ("dilate",                      dilate),
        ("erode",                       erode),
        # Sorting (2)
        ("sort_rows_by_nonzero",        sort_rows_by_nonzero),
        ("sort_cols_by_nonzero",        sort_cols_by_nonzero),
        # Extraction (2)
        ("extract_largest_cc",          extract_largest_cc),
        ("extract_unique_color_region", extract_unique_color_region),
        # Inpainting (1)
        ("inpaint_periodic",            inpaint_periodic),
    ]
    prims = [Primitive(name=name, arity=1, fn=fn, domain="arc")
             for name, fn in unary_ops]
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))
    prims.append(Primitive(name="mask_by", arity=2, fn=mask_by, domain="arc"))
    return prims


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
    from collections import Counter
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
    from collections import Counter
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


def _recolor_foreground_factory(color: int):
    """Factory: replace all non-background colors with given color."""
    from collections import Counter
    def recolor(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
        bg = Counter(flat).most_common(1)[0][0]
        return [[color if cell != bg else cell for cell in row] for row in grid]
    return recolor


def build_parameterized_primitives() -> list[Primitive]:
    """Build parameterized action primitives (factory functions)."""
    return [
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
        Primitive(name="recolor_foreground", arity=1,
                  fn=_recolor_foreground_factory, domain="arc", kind="parameterized"),
        Primitive(name="scale", arity=1,
                  fn=_scale_factory, domain="arc", kind="parameterized"),
        Primitive(name="tile", arity=1,
                  fn=_tile_factory, domain="arc", kind="parameterized"),
        Primitive(name="downscale", arity=1,
                  fn=_downscale_factory, domain="arc", kind="parameterized"),
    ]


ATOMIC_ESSENTIAL_PAIR_CONCEPTS: frozenset = frozenset([
    "trim_rows",
    "trim_cols",
    "overlay",
    "mirror_horizontal",
    "mirror_vertical",
    "binarize",
    "dilate",
    "erode",
])
