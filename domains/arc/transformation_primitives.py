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


# --- Periodicity transforms ---

def extract_period_tile(grid: Grid) -> Grid:
    """Find the smallest tile that, when repeated, reconstructs the grid.

    Justified by task 7b7f7511.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            tile = [grid[r][:pw] for r in range(ph)]
            match = all(grid[r][c] == tile[r % ph][c % pw]
                        for r in range(h) for c in range(w))
            if match and (ph < h or pw < w):
                return tile
    return grid


# --- Compression transforms ---

def compress_rows(grid: Grid) -> Grid:
    """Remove duplicate consecutive rows. Justified by tasks eb5a1d5d, 746b3537."""
    if not grid:
        return grid
    result = [grid[0]]
    for r in range(1, len(grid)):
        if grid[r] != grid[r - 1]:
            result.append(grid[r])
    return result


def unique_cols(grid: Grid) -> Grid:
    """Remove duplicate columns (keep first occurrence). Justified by task 2dee498d."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, ...]] = set()
    keep: list[int] = []
    for c in range(w):
        col = tuple(grid[r][c] for r in range(h))
        if col not in seen:
            seen.add(col)
            keep.append(c)
    if not keep:
        return grid
    return [[grid[r][c] for c in keep] for r in range(h)]


def compress_cols(grid: Grid) -> Grid:
    """Remove duplicate consecutive columns."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    keep = [0]
    for c in range(1, w):
        col = tuple(grid[r][c] for r in range(h))
        prev = tuple(grid[r][c - 1] for r in range(h))
        if col != prev:
            keep.append(c)
    return [[grid[r][c] for c in keep] for r in range(h)]


# --- Ray extension transforms ---

def extend_diag_rays(grid: Grid) -> Grid:
    """Each non-zero pixel shoots diagonal rays until hitting another non-zero.

    Justified by task 623ea044.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            color = grid[r][c]
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                    result[nr][nc] = color
                    nr += dr
                    nc += dc
    return result


def extend_down(grid: Grid) -> Grid:
    """Each non-zero pixel extends downward until hitting another non-zero.

    Justified by task d037b0a7.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            nr = r + 1
            while nr < h and grid[nr][c] == 0:
                result[nr][c] = grid[r][c]
                nr += 1
    return result


# --- Connection transforms ---

def connect_same_color_h(grid: Grid) -> Grid:
    """Fill zeros between same-color pixels horizontally.

    For each row, if two pixels of the same color have only zeros
    between them, fill the gap with that color.

    Justified by tasks 22eb0ac0, 22168020.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        color_positions: dict[int, list[int]] = {}
        for c in range(w):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append(c)
        for color, positions in color_positions.items():
            if len(positions) >= 2:
                for c in range(min(positions), max(positions) + 1):
                    if result[r][c] == 0:
                        result[r][c] = color
    return result


def connect_same_color_v(grid: Grid) -> Grid:
    """Fill zeros between same-color pixels vertically.

    For each column, if two pixels of the same color have only zeros
    between them, fill the gap with that color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        color_positions: dict[int, list[int]] = {}
        for r in range(h):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append(r)
        for color, positions in color_positions.items():
            if len(positions) >= 2:
                for r in range(min(positions), max(positions) + 1):
                    if result[r][c] == 0:
                        result[r][c] = color
    return result


# --- Edge detection transforms ---

def outline(grid: Grid) -> Grid:
    """Extract edges: keep non-zero pixels that have at least one zero 4-neighbor.

    Equivalent to diff(input, erode(input)). Justified by task 4347f46a.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            is_edge = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == 0:
                    is_edge = True
                    break
            if is_edge:
                result[r][c] = grid[r][c]
    return result


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


# --- Tiling transforms ---

def mirror_tile_h(grid: Grid) -> Grid:
    """Tile horizontally: original | horizontally-mirrored.

    Justified by tasks 6d0aefbc, c9e6f938.
    """
    if not grid or not grid[0]:
        return grid
    return [row + row[::-1] for row in grid]


def mirror_tile_v(grid: Grid) -> Grid:
    """Tile vertically: original on top, vertically-mirrored on bottom.

    Justified by tasks 6fa7a44f, 8be77c9e.
    """
    if not grid:
        return grid
    return grid + grid[::-1]


def mirror_tile_both(grid: Grid) -> Grid:
    """2x2 mirror tile: orig|mirH / mirV|mirHV.

    Justified by tasks 67e8384a, 3af2c5a8, 62c24649.
    """
    if not grid or not grid[0]:
        return grid
    top = [row + row[::-1] for row in grid]
    bottom = [row + row[::-1] for row in grid[::-1]]
    return top + bottom


def tile_h(grid: Grid) -> Grid:
    """Repeat grid horizontally: grid | grid.

    Justified by task a416b8f3.
    """
    if not grid or not grid[0]:
        return grid
    return [row + row for row in grid]


def rotate_tile_cw(grid: Grid) -> Grid:
    """2x2 rotation tile: orig|rot90 / rot270|rot180. Square grids only.

    Justified by tasks 46442a0e, 7fe24cdd.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h != w:
        return grid
    r90 = [[grid[h - 1 - c][r] for c in range(h)] for r in range(w)]
    r180 = [[grid[h - 1 - r][w - 1 - c] for c in range(w)] for r in range(h)]
    r270 = [[grid[c][w - 1 - r] for c in range(h)] for r in range(w)]
    top = [grid[r] + r90[r] for r in range(h)]
    bottom = [r270[r] + r180[r] for r in range(h)]
    return top + bottom


# --- Inpainting transforms ---

def inpaint_by_symmetry(grid: Grid) -> Grid:
    """Inpaint masked cells using the grid's inherent symmetry.

    Auto-detects the mask color (the one that breaks symmetry most),
    then fills masked cells from their symmetric counterparts using
    H-mirror, V-mirror, 180° rotation, and transpose.

    Justified by task b8825c91.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    h, w = len(grid), len(grid[0])

    # Find the best mask color: try each non-zero color, pick the one
    # where replacing it via symmetry produces the most consistent result
    colors = set(c for row in grid for c in row if c != 0)
    if not colors:
        return grid

    best_result = None
    best_remaining = h * w  # fewer remaining mask cells = better

    for mask_color in colors:
        result = [row[:] for row in grid]
        # Apply symmetry fills iteratively until stable
        for _ in range(4):
            changed = False
            for r in range(h):
                for c in range(w):
                    if result[r][c] != mask_color:
                        continue
                    # Try mirrors in priority order
                    for mr, mc in [(r, w-1-c), (h-1-r, c), (h-1-r, w-1-c)]:
                        if 0 <= mr < h and 0 <= mc < w and result[mr][mc] != mask_color:
                            result[r][c] = result[mr][mc]
                            changed = True
                            break
                    else:
                        if h == w and result[c][r] != mask_color:
                            result[r][c] = result[c][r]
                            changed = True
            if not changed:
                break

        remaining = sum(1 for r in range(h) for c in range(w) if result[r][c] == mask_color)
        if remaining < best_remaining:
            best_remaining = remaining
            best_result = result

    return best_result if best_result is not None else grid


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

def extend_right_and_down(grid: Grid) -> Grid:
    """Each non-zero pixel extends right to edge, then down along the right edge.

    Forms an L-shape: horizontal ray rightward + vertical ray downward
    from the rightmost point of the horizontal ray. Later (lower) pixels
    overwrite earlier ones.

    Justified by task 99fa7670.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    # Collect source pixels sorted top-to-bottom
    sources = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    sources.sort()  # top-to-bottom, left-to-right

    result = [row[:] for row in grid]
    for r, c, color in sources:
        # Extend right to edge
        for cc in range(c + 1, w):
            result[r][cc] = color
        # Extend down from the right edge
        for rr in range(r + 1, h):
            result[rr][w - 1] = color
    return result


def extract_unique_quadrant(grid: Grid) -> Grid:
    """Split grid by separator lines and extract the unique quadrant.

    Detects horizontal and vertical separator lines (rows/cols of one color),
    splits into sections, and returns the one that differs from the majority.

    Justified by tasks 2dc579da, 88a62173.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Find separator rows (all same non-zero color)
    sep_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and 0 not in vals:
            sep_rows.append(r)

    # Find separator cols
    sep_cols = []
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and 0 not in vals:
            sep_cols.append(c)

    # Also try: separator = row where all pixels match AND that color appears in every position
    # Find potential separator color
    if not sep_rows and not sep_cols:
        # Try rows where all pixels are the same color
        for r in range(h):
            if len(set(grid[r])) == 1:
                sep_rows.append(r)
        for c in range(w):
            col_vals = set(grid[r][c] for r in range(h))
            if len(col_vals) == 1:
                sep_cols.append(c)

    if not sep_rows and not sep_cols:
        return grid

    # Build row boundaries and col boundaries
    row_bounds = []
    prev = 0
    for r in sorted(set(sep_rows)):
        if r > prev:
            row_bounds.append((prev, r))
        prev = r + 1
    if prev < h:
        row_bounds.append((prev, h))

    col_bounds = []
    prev = 0
    for c in sorted(set(sep_cols)):
        if c > prev:
            col_bounds.append((prev, c))
        prev = c + 1
    if prev < w:
        col_bounds.append((prev, w))

    if not row_bounds or not col_bounds:
        return grid

    # Extract all sections
    sections = []
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            section = tuple(tuple(grid[r][c] for c in range(c0, c1)) for r in range(r0, r1))
            sections.append((r0, c0, r1, c1, section))

    if len(sections) < 2:
        return grid

    # Find the unique section (differs from majority)
    section_data = [s[4] for s in sections]
    from collections import Counter
    counts = Counter(section_data)
    if len(counts) < 2:
        return grid  # All sections are the same

    # Find the section that appears least (the unique one)
    least_common = counts.most_common()[-1][0]
    for r0, c0, r1, c1, section in sections:
        if section == least_common:
            return [list(grid[r][c0:c1]) for r in range(r0, r1)]

    return grid


def overlay_all_sections(grid: Grid) -> Grid:
    """Split grid by separator lines, overlay all sections (non-zero wins).

    Combines all quadrants/sections by keeping non-zero pixels from any.
    Justified by task a68b268e.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    sep_rows = sorted(set(r for r in range(h) if len(set(grid[r])) == 1))
    sep_cols = sorted(set(c for c in range(w) if len(set(grid[r][c] for r in range(h))) == 1))

    row_bounds = []
    prev = 0
    for r in sep_rows:
        if r > prev:
            row_bounds.append((prev, r))
        prev = r + 1
    if prev < h:
        row_bounds.append((prev, h))

    col_bounds = []
    prev = 0
    for c in sep_cols:
        if c > prev:
            col_bounds.append((prev, c))
        prev = c + 1
    if prev < w:
        col_bounds.append((prev, w))

    if not row_bounds or not col_bounds or len(row_bounds) * len(col_bounds) < 2:
        return grid

    sh = row_bounds[0][1] - row_bounds[0][0]
    sw = col_bounds[0][1] - col_bounds[0][0]
    for r0, r1 in row_bounds:
        if r1 - r0 != sh:
            return grid
    for c0, c1 in col_bounds:
        if c1 - c0 != sw:
            return grid

    result = [[0] * sw for _ in range(sh)]
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            for dr in range(sh):
                for dc in range(sw):
                    v = grid[r0 + dr][c0 + dc]
                    if v != 0 and result[dr][dc] == 0:
                        result[dr][dc] = v
    return result


def remove_separators(grid: Grid) -> Grid:
    """Remove separator rows and columns (rows/cols where all pixels are same color).

    Keeps only rows and columns with mixed content.
    Justified by task 68b67ca3.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    kept_rows = [r for r in range(h) if len(set(grid[r])) > 1]
    if not kept_rows:
        return grid
    intermediate = [grid[r] for r in kept_rows]
    h2 = len(intermediate)
    w2 = len(intermediate[0])
    kept_cols = [c for c in range(w2) if len(set(intermediate[r][c] for r in range(h2))) > 1]
    if not kept_cols:
        return intermediate
    return [[intermediate[r][c] for c in kept_cols] for r in range(h2)]


def crop_to_content(grid: Grid) -> Grid:
    """Crop to minimal bounding box containing all non-zero pixels.

    Combines trim_rows and trim_cols in one operation.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    # Find bounds
    top, bot, left, right = h, -1, w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                top = min(top, r)
                bot = max(bot, r)
                left = min(left, c)
                right = max(right, c)
    if bot < 0:
        return grid
    return [grid[r][left:right + 1] for r in range(top, bot + 1)]


def flood_fill_by_neighbor(grid: Grid) -> Grid:
    """Fill each enclosed zero-region with its surrounding border color.

    Unlike fill_enclosed (which uses the object's own color), this uses
    the most common color bordering each enclosed region.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    h, w = len(grid), len(grid[0])

    # Find exterior zeros via flood fill from border
    exterior = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                if (r, c) not in exterior:
                    exterior.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in exterior and grid[nr][nc] == 0:
                exterior.add((nr, nc))
                queue.append((nr, nc))

    # Find connected interior regions and fill each with neighbor color
    visited = set()
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in exterior and (r, c) not in visited:
                region = set()
                q = [(r, c)]
                while q:
                    cr, cc = q.pop()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] != 0 or (cr, cc) in exterior:
                        continue
                    visited.add((cr, cc))
                    region.add((cr, cc))
                    q.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                # Find majority border color
                border_colors = Counter()
                for pr, pc in region:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = pr+dr, pc+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                            border_colors[grid[nr][nc]] += 1
                if border_colors:
                    fc = border_colors.most_common(1)[0][0]
                    for pr, pc in region:
                        result[pr][pc] = fc
    return result


def subtract_grid(grid1: Grid, grid2: Grid) -> Grid:
    """Subtract grid2 from grid1: keep grid1 where grid2 is zero.

    Where grid2 has non-zero pixels, the result becomes 0.
    Like an inverse mask.
    """
    if not grid1 or not grid2:
        return grid1 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    return [[grid1[r][c] if grid2[r][c] == 0 else 0
             for c in range(w)] for r in range(h)]


def xor_grid(grid1: Grid, grid2: Grid) -> Grid:
    """XOR of two grids: keep pixels that exist in one but not both.

    If both grids have non-zero at (r,c), result is 0.
    If only one has non-zero, keep that value.
    """
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            a, b = grid1[r][c], grid2[r][c]
            if a != 0 and b == 0:
                result[r][c] = a
            elif b != 0 and a == 0:
                result[r][c] = b
    return result


def overlay_and(grid1: Grid, grid2: Grid) -> Grid:
    """Logical AND of two grids: keep pixels non-zero in BOTH, using grid1's color.

    output[r][c] = grid1[r][c] if both grid1[r][c] != 0 and grid2[r][c] != 0, else 0.
    If sizes differ, return grid1 unchanged.
    """
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return grid1
    h, w = len(grid1), len(grid1[0])
    return [[grid1[r][c] if (grid1[r][c] != 0 and grid2[r][c] != 0) else 0
             for c in range(w)] for r in range(h)]


def overlay_xor(grid1: Grid, grid2: Grid) -> Grid:
    """XOR overlay: keep pixels non-zero in exactly ONE grid (not both).

    Uses whichever grid has the non-zero value.
    If both are non-zero or both are zero, output is 0.
    If sizes differ, return grid1 unchanged.
    """
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return grid1
    h, w = len(grid1), len(grid1[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            a, b = grid1[r][c], grid2[r][c]
            if a != 0 and b == 0:
                result[r][c] = a
            elif b != 0 and a == 0:
                result[r][c] = b
            # both non-zero or both zero → 0
    return result


def extrapolate_growth(grid: Grid) -> Grid:
    """Grow each connected component (4-connected, non-zero) by 1 pixel in cardinal directions.

    Uses the component's color. Does not overwrite existing non-zero pixels.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                        result[nr][nc] = color
    return result


def shrink_objects(grid: Grid) -> Grid:
    """Erode connected non-zero components by removing boundary pixels.

    A pixel is removed (set to 0) if it has fewer than 2 non-zero 4-neighbors
    within the same component color. Interior pixels with >= 2 neighbors are kept.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                neighbor_count = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                        neighbor_count += 1
                if neighbor_count < 2:
                    result[r][c] = 0
    return result


def tile_v(grid: Grid) -> Grid:
    """Repeat grid vertically: grid on top, grid on bottom."""
    if not grid:
        return grid
    return grid + [row[:] for row in grid]


def tile_both(grid: Grid) -> Grid:
    """2x2 tile: repeat grid in both dimensions.

    Top-left = original, top-right = original,
    bottom-left = original, bottom-right = original.
    """
    if not grid or not grid[0]:
        return grid
    top = [row + row for row in grid]
    return top + [row[:] for row in top]


def scale_up_2x(grid: Grid) -> Grid:
    """Scale grid 2x: each pixel becomes a 2x2 block."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        new_row = []
        for v in row:
            new_row.extend([v, v])
        result.append(new_row)
        result.append(new_row[:])
    return result


def scale_up_3x(grid: Grid) -> Grid:
    """Scale grid 3x: each pixel becomes a 3x3 block."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        new_row = []
        for v in row:
            new_row.extend([v, v, v])
        for _ in range(3):
            result.append(new_row[:])
    return result


def densest_subgrid(grid: Grid) -> Grid:
    """Extract the densest rectangular subgrid.

    Finds the subgrid with the highest ratio of non-zero pixels,
    trying all connected component bounding boxes.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    best_grid = grid
    best_density = -1.0

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
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
                if len(comp) < 2:
                    continue
                r0 = min(p[0] for p in comp)
                c0 = min(p[1] for p in comp)
                r1 = max(p[0] for p in comp)
                c1 = max(p[1] for p in comp)
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                density = len(comp) / area
                if density > best_density:
                    best_density = density
                    best_grid = [grid[rr][c0:c1 + 1] for rr in range(r0, r1 + 1)]
    return best_grid


def most_colorful_subgrid(grid: Grid) -> Grid:
    """Extract the connected component with the most distinct colors in its bbox."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    best_grid = grid
    best_colors = -1

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                r0 = min(p[0] for p in comp)
                c0 = min(p[1] for p in comp)
                r1 = max(p[0] for p in comp)
                c1 = max(p[1] for p in comp)
                colors = set()
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        if grid[rr][cc] != 0:
                            colors.add(grid[rr][cc])
                if len(colors) > best_colors:
                    best_colors = len(colors)
                    best_grid = [grid[rr][c0:c1 + 1] for rr in range(r0, r1 + 1)]
    return best_grid


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
# Inpainting primitives
# =============================================================================

def inpaint_by_neighbors(grid: Grid) -> Grid:
    """Fill zeros with majority color of non-zero 4-neighbors.

    Iterates until no more changes or max(h,w) iterations.
    """
    from collections import Counter
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    max_iters = max(h, w)
    for _ in range(max_iters):
        changed = False
        new = [row[:] for row in result]
        for r in range(h):
            for c in range(w):
                if result[r][c] == 0:
                    neighbors = []
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and result[nr][nc] != 0:
                            neighbors.append(result[nr][nc])
                    if neighbors:
                        new[r][c] = Counter(neighbors).most_common(1)[0][0]
                        changed = True
        result = new
        if not changed:
            break
    return result


def symmetry_complete(grid: Grid) -> Grid:
    """Detect nearest symmetry axis, fill zeros with symmetric counterpart.

    Scores horizontal and vertical mirror axes by matching non-zero pairs.
    Uses the better axis if score ratio > 0.5.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    def score_horizontal():
        """Score horizontal (left-right) mirror symmetry."""
        matches = total = 0
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if mc <= c:
                    continue
                a, b = grid[r][c], grid[r][mc]
                if a != 0 or b != 0:
                    total += 1
                    if a == b:
                        matches += 1
        return matches / total if total > 0 else 0.0

    def score_vertical():
        """Score vertical (top-bottom) mirror symmetry."""
        matches = total = 0
        for r in range(h):
            mr = h - 1 - r
            if mr <= r:
                continue
            for c in range(w):
                a, b = grid[r][c], grid[mr][c]
                if a != 0 or b != 0:
                    total += 1
                    if a == b:
                        matches += 1
        return matches / total if total > 0 else 0.0

    h_score = score_horizontal()
    v_score = score_vertical()
    result = [row[:] for row in grid]

    if h_score >= v_score and h_score > 0.5:
        # Fill using horizontal mirror
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if result[r][c] == 0 and result[r][mc] != 0:
                    result[r][c] = result[r][mc]
    elif v_score > h_score and v_score > 0.5:
        # Fill using vertical mirror
        for r in range(h):
            mr = h - 1 - r
            for c in range(w):
                if result[r][c] == 0 and result[mr][c] != 0:
                    result[r][c] = result[mr][c]
    return result


def fill_by_row_col_pattern(grid: Grid) -> Grid:
    """Find dominant non-zero color per row and col, fill zeros preferring row color."""
    from collections import Counter
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    row_color = []
    for r in range(h):
        nz = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        row_color.append(Counter(nz).most_common(1)[0][0] if nz else 0)

    col_color = []
    for c in range(w):
        nz = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        col_color.append(Counter(nz).most_common(1)[0][0] if nz else 0)

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                if row_color[r] != 0:
                    result[r][c] = row_color[r]
                elif col_color[c] != 0:
                    result[r][c] = col_color[c]
    return result


def inpaint_diagonal(grid: Grid) -> Grid:
    """Fill zeros on diagonals where all non-zero values share the same color.

    Collects top-left to bottom-right diagonals; if all non-zero values on
    a diagonal are the same color, fills zeros on that diagonal with it.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    # Diagonals indexed by (r - c), ranging from -(w-1) to (h-1)
    for diag in range(-(w - 1), h):
        cells = []
        for r in range(h):
            c = r - diag
            if 0 <= c < w:
                cells.append((r, c))
        nz_colors = set(grid[r][c] for r, c in cells if grid[r][c] != 0)
        if len(nz_colors) == 1:
            color = next(iter(nz_colors))
            for r, c in cells:
                if result[r][c] == 0:
                    result[r][c] = color
    return result


def inpaint_from_template(grid: Grid) -> Grid:
    """Find most common NxN pattern in non-zero regions, stamp into zero regions.

    Tries 2x2 then 3x3 patterns. A candidate pattern is stamped into a zero
    region when its non-zero cells match the existing non-zero cells at that
    location.
    """
    from collections import Counter
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    for n in (2, 3):
        if h < n or w < n:
            continue
        # Collect all NxN patches from non-zero regions
        patch_counts: Counter = Counter()
        for r in range(h - n + 1):
            for c in range(w - n + 1):
                patch = tuple(grid[r + dr][c + dc]
                              for dr in range(n) for dc in range(n))
                if all(v != 0 for v in patch):
                    patch_counts[patch] += 1
        if not patch_counts:
            continue
        best_patch = patch_counts.most_common(1)[0][0]
        # Stamp pattern into regions where existing non-zero cells match
        for r in range(h - n + 1):
            for c in range(w - n + 1):
                positions = [(r + dr, c + dc) for dr in range(n) for dc in range(n)]
                current = [grid[rr][cc] for rr, cc in positions]
                # Check: non-zero cells must match the pattern
                matches = all(cur == 0 or cur == pat
                              for cur, pat in zip(current, best_patch))
                has_zero = any(cur == 0 for cur in current)
                if matches and has_zero:
                    for (rr, cc), pat in zip(positions, best_patch):
                        if result[rr][cc] == 0:
                            result[rr][cc] = pat
        break  # Stop after first successful size
    return result


# =============================================================================
# Denoising primitives
# =============================================================================

def remove_isolated(grid: Grid) -> Grid:
    """Remove non-zero pixels with no non-zero 4-neighbors."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                has_neighbor = False
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    result[r][c] = 0
    return result


def majority_filter_3x3(grid: Grid) -> Grid:
    """Replace each pixel with the majority color in its 3x3 neighborhood."""
    from collections import Counter
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            neighborhood = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighborhood.append(grid[nr][nc])
            if neighborhood:
                result[r][c] = Counter(neighborhood).most_common(1)[0][0]
    return result


def morphological_close(grid: Grid) -> Grid:
    """Morphological closing: dilate then erode."""
    return erode(dilate(grid))


# =============================================================================
# Grid structure primitives
# =============================================================================

def remove_border(grid: Grid) -> Grid:
    """Strip outermost row and column on all sides."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h < 3 or w < 3:
        return grid
    return [row[1:-1] for row in grid[1:-1]]


# =============================================================================
# Cardinal extension primitives
# =============================================================================

def extend_up(grid: Grid) -> Grid:
    """Each non-zero pixel extends upward, filling zeros until hitting another non-zero.

    Scan bottom-to-top so upper pixels don't overwrite each other.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h - 2, -1, -1):
        for c in range(w):
            if result[r][c] == 0 and result[r + 1][c] != 0:
                result[r][c] = result[r + 1][c]
    return result


def extend_left(grid: Grid) -> Grid:
    """Each non-zero pixel extends leftward, filling zeros until hitting another non-zero."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if result[r][c] == 0 and result[r][c + 1] != 0:
                result[r][c] = result[r][c + 1]
    return result


def extend_right(grid: Grid) -> Grid:
    """Each non-zero pixel extends rightward, filling zeros until hitting another non-zero."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(1, w):
            if result[r][c] == 0 and result[r][c - 1] != 0:
                result[r][c] = result[r][c - 1]
    return result


# =============================================================================
# Object relationship primitives
# =============================================================================

def draw_line_between_objects(grid: Grid) -> Grid:
    """For each pair of same-color non-zero pixels on same row or column, fill the gap.

    Fills the gap between same-color pixels on the same row or column with
    that color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    # Horizontal: same row, same color
    for r in range(h):
        color_positions: dict[int, list[int]] = {}
        for c in range(w):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append(c)
        for color, positions in color_positions.items():
            if len(positions) >= 2:
                lo, hi = min(positions), max(positions)
                for c in range(lo, hi + 1):
                    result[r][c] = color
    # Vertical: same column, same color
    for c in range(w):
        color_positions = {}
        for r in range(h):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append(r)
        for color, positions in color_positions.items():
            if len(positions) >= 2:
                lo, hi = min(positions), max(positions)
                for r in range(lo, hi + 1):
                    result[r][c] = color
    return result


def color_by_object_rank(grid: Grid) -> Grid:
    """Find connected components, rank by size (largest=1), recolor with rank."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS flood fill
                comp = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    # Sort by size descending; rank 1 = largest
    components.sort(key=lambda comp: len(comp), reverse=True)
    result = [row[:] for row in grid]
    for rank, comp in enumerate(components, start=1):
        for r, c in comp:
            result[r][c] = rank
    return result


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
        # Periodicity (1)
        ("extract_period_tile",         extract_period_tile),
        # Compression (3)
        ("compress_rows",               compress_rows),
        ("compress_cols",               compress_cols),
        ("unique_cols",                 unique_cols),
        # Ray extension (2)
        ("extend_diag_rays",            extend_diag_rays),
        ("extend_down",                 extend_down),
        # Connection (2)
        ("connect_same_color_h",        connect_same_color_h),
        ("connect_same_color_v",        connect_same_color_v),
        # Edge detection (1)
        ("outline",                     outline),
        # Extraction (2)
        ("extract_largest_cc",          extract_largest_cc),
        ("extract_unique_color_region", extract_unique_color_region),
        # Tiling (5)
        ("mirror_tile_h",              mirror_tile_h),
        ("mirror_tile_v",              mirror_tile_v),
        ("mirror_tile_both",           mirror_tile_both),
        ("tile_h",                     tile_h),
        ("rotate_tile_cw",             rotate_tile_cw),
        # Inpainting (2)
        ("inpaint_by_symmetry",         inpaint_by_symmetry),
        ("inpaint_periodic",            inpaint_periodic),
        # Content extraction (3)
        ("crop_to_content",             crop_to_content),
        ("densest_subgrid",             densest_subgrid),
        ("most_colorful_subgrid",       most_colorful_subgrid),
        # Fill variants (1)
        ("flood_fill_by_neighbor",      flood_fill_by_neighbor),
        # L-shape extension (1)
        ("extend_right_and_down",       extend_right_and_down),
        # Quadrant/separator operations (3)
        ("extract_unique_quadrant",     extract_unique_quadrant),
        ("overlay_all_sections",        overlay_all_sections),
        ("remove_separators",           remove_separators),
        # Tiling (1)
        ("tile_v",                      tile_v),
        # Inpainting — new Tier 1 (5)
        ("inpaint_by_neighbors",        inpaint_by_neighbors),
        ("symmetry_complete",           symmetry_complete),
        ("fill_by_row_col_pattern",     fill_by_row_col_pattern),
        ("inpaint_diagonal",            inpaint_diagonal),
        ("inpaint_from_template",       inpaint_from_template),
        # Denoising — new Tier 1 (3)
        ("remove_isolated",             remove_isolated),
        ("majority_filter_3x3",         majority_filter_3x3),
        ("morphological_close",         morphological_close),
        # Grid structure — new Tier 1 (1)
        ("remove_border",               remove_border),
        # Cardinal extensions — new Tier 1 (3)
        ("extend_up",                   extend_up),
        ("extend_left",                 extend_left),
        ("extend_right",                extend_right),
        # Object relationships — new Tier 1 (2)
        ("draw_line_between_objects",   draw_line_between_objects),
        ("color_by_object_rank",        color_by_object_rank),
        # Morphological growth/shrink — Tier 3 (2)
        ("extrapolate_growth",          extrapolate_growth),
        ("shrink_objects",              shrink_objects),
    ]
    prims = [Primitive(name=name, arity=1, fn=fn, domain="arc")
             for name, fn in unary_ops]
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))
    prims.append(Primitive(name="mask_by", arity=2, fn=mask_by, domain="arc"))
    prims.append(Primitive(name="subtract_grid", arity=2, fn=subtract_grid, domain="arc"))
    prims.append(Primitive(name="xor_grid", arity=2, fn=xor_grid, domain="arc"))
    # Tier 2 binary primitives
    prims.append(Primitive(name="overlay_and", arity=2, fn=overlay_and, domain="arc"))
    prims.append(Primitive(name="overlay_xor", arity=2, fn=overlay_xor, domain="arc"))
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
    "extract_largest_cc",
    "extract_unique_color_region",
    "inpaint_periodic",
    "inpaint_by_symmetry",
    "tile_h",
    "mirror_tile_h",
    "mirror_tile_v",
    "mirror_tile_both",
])
