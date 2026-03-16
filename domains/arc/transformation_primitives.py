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
Compositional operations (like crop_to_content = trim_rows(trim_cols(x)))
are NOT primitives — they must be discovered through composition.

DISCOVERY GOALS — compound operations to discover through composition:
- crop_to_content = trim_cols(trim_rows(x)) or trim_rows(trim_cols(x))
- extract_largest_object = label_components + largest_object_color + keep_color + mask_by
- keep_largest/smallest_component = label_components + rank + mask
"""

from __future__ import annotations

from collections import Counter

from core import Primitive

Grid = list[list[int]]


# =============================================================================
# Geometric transforms (6) — each is one intuitive visual concept
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
# Spatial transforms (7)
# =============================================================================

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


# =============================================================================
# Sorting transforms (2)
# =============================================================================

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


# =============================================================================
# Fill transforms (1)
# =============================================================================

def border_extend(grid: Grid) -> Grid:
    """Extend non-zero border pixels into adjacent zero cells on the grid edges."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    # Top and bottom edges
    for c in range(w):
        if result[0][c] == 0 and h > 1 and grid[1][c] != 0:
            result[0][c] = grid[1][c]
        if result[h - 1][c] == 0 and h > 1 and grid[h - 2][c] != 0:
            result[h - 1][c] = grid[h - 2][c]
    # Left and right edges
    for r in range(h):
        if result[r][0] == 0 and w > 1 and grid[r][1] != 0:
            result[r][0] = grid[r][1]
        if result[r][w - 1] == 0 and w > 1 and grid[r][w - 2] != 0:
            result[r][w - 1] = grid[r][w - 2]
    return result


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
# Ray extension transforms (4) — extend colored pixels in a direction
# =============================================================================

def extend_rays_right(grid: Grid) -> Grid:
    """Extend each non-zero pixel rightward until hitting another non-zero pixel or edge."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                for cc in range(c + 1, w):
                    if grid[r][cc] != 0:
                        break
                    result[r][cc] = grid[r][c]
    return result


def extend_rays_left(grid: Grid) -> Grid:
    """Extend each non-zero pixel leftward until hitting another non-zero pixel or edge."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w - 1, -1, -1):
            if grid[r][c] != 0:
                for cc in range(c - 1, -1, -1):
                    if grid[r][cc] != 0:
                        break
                    result[r][cc] = grid[r][c]
    return result


def extend_rays_down(grid: Grid) -> Grid:
    """Extend each non-zero pixel downward until hitting another non-zero pixel or edge."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        for r in range(h):
            if grid[r][c] != 0:
                for rr in range(r + 1, h):
                    if grid[rr][c] != 0:
                        break
                    result[rr][c] = grid[r][c]
    return result


def extend_rays_up(grid: Grid) -> Grid:
    """Extend each non-zero pixel upward until hitting another non-zero pixel or edge."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        for r in range(h - 1, -1, -1):
            if grid[r][c] != 0:
                for rr in range(r - 1, -1, -1):
                    if grid[rr][c] != 0:
                        break
                    result[rr][c] = grid[r][c]
    return result


# =============================================================================
# Flood fill from markers (1) — paint bucket from each non-zero seed
# =============================================================================

def flood_fill_from_markers(grid: Grid) -> Grid:
    """Flood fill connected zero regions from adjacent non-zero seed pixels.

    For each non-zero pixel, fill all connected zero-valued neighbors with
    that pixel's color (like paint bucket tool). Processes seeds in
    reading order (top-left to bottom-right).
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                # BFS into adjacent zeros
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and result[nr][nc] == 0):
                        # Flood fill this connected zero region
                        queue = [(nr, nc)]
                        result[nr][nc] = color
                        qi = 0
                        while qi < len(queue):
                            cr, cc = queue[qi]
                            qi += 1
                            for d2r, d2c in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                n2r, n2c = cr + d2r, cc + d2c
                                if (0 <= n2r < h and 0 <= n2c < w
                                        and result[n2r][n2c] == 0):
                                    result[n2r][n2c] = color
                                    queue.append((n2r, n2c))
    return result


# =============================================================================
# Connected component labeling (1) — truly atomic, single BFS
# =============================================================================

def extract_largest_object(grid: Grid) -> Grid:
    """Extract the largest connected foreground component, crop to its bounding box.

    Returns the bounding-box crop of the largest component (by pixel count),
    with background pixels set to 0.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]

    visited = set()
    best_pixels = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                pixels = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    pixels.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                if len(pixels) > len(best_pixels):
                    best_pixels = pixels
    if not best_pixels:
        return grid
    min_r = min(r for r, c in best_pixels)
    max_r = max(r for r, c in best_pixels)
    min_c = min(c for r, c in best_pixels)
    max_c = max(c for r, c in best_pixels)
    pixel_set = set(best_pixels)
    return [[grid[r][c] if (r, c) in pixel_set else 0
             for c in range(min_c, max_c + 1)]
            for r in range(min_r, max_r + 1)]


def extract_smallest_object(grid: Grid) -> Grid:
    """Extract the smallest connected foreground component, crop to its bounding box."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]

    visited = set()
    best_pixels = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                pixels = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    pixels.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                if best_pixels is None or len(pixels) < len(best_pixels):
                    best_pixels = pixels
    if not best_pixels:
        return grid
    min_r = min(r for r, c in best_pixels)
    max_r = max(r for r, c in best_pixels)
    min_c = min(c for r, c in best_pixels)
    max_c = max(c for r, c in best_pixels)
    pixel_set = set(best_pixels)
    return [[grid[r][c] if (r, c) in pixel_set else 0
             for c in range(min_c, max_c + 1)]
            for r in range(min_r, max_r + 1)]


def hollow_objects(grid: Grid) -> Grid:
    """Remove interior pixels from objects, keeping only outlines.

    A pixel is interior if all 4 neighbors are the same color (non-background).
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                is_interior = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == bg:
                        is_interior = False
                        break
                if is_interior:
                    result[r][c] = bg
    return result


def fill_diagonal_symmetry(grid: Grid) -> Grid:
    """Complete diagonal symmetry: reflect across the main diagonal.

    For non-square grids, takes the minimum dimension.
    Copies grid[r][c] to grid[c][r] where current value is background.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    n = min(h, w)
    for r in range(n):
        for c in range(n):
            if result[r][c] == bg and r < h and c < w and c < h and r < w:
                result[r][c] = grid[c][r]
            if result[c][r] == bg and c < h and r < w and r < h and c < w:
                result[c][r] = grid[r][c]
    return result


def complete_horizontal_symmetry(grid: Grid) -> Grid:
    """Complete horizontal (left-right) symmetry.

    For each pixel that is background, check if its mirror position is non-background.
    If so, copy the mirror pixel to fill the gap.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r][c] == bg and grid[r][mc] != bg:
                result[r][c] = grid[r][mc]
    return result


def complete_vertical_symmetry(grid: Grid) -> Grid:
    """Complete vertical (top-bottom) symmetry.

    For each pixel that is background, check if its mirror position is non-background.
    If so, copy the mirror pixel to fill the gap.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        mr = h - 1 - r
        for c in range(w):
            if result[r][c] == bg and grid[mr][c] != bg:
                result[r][c] = grid[mr][c]
    return result


def remove_border(grid: Grid) -> Grid:
    """Remove 1-pixel border from the grid (inverse of pad_border)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h <= 2 or w <= 2:
        return grid
    return [row[1:-1] for row in grid[1:-1]]


def denoise_majority(grid: Grid) -> Grid:
    """Denoise: replace each pixel with the majority of its 3x3 neighborhood.

    Useful for cleaning up noise in ARC grids where most pixels agree
    with their neighbors.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(grid[nr][nc])
            if neighbors:
                result[r][c] = Counter(neighbors).most_common(1)[0][0]
    return result


def keep_unique_rows(grid: Grid) -> Grid:
    """Keep only unique rows (remove duplicate rows)."""
    if not grid:
        return grid
    seen = []
    result = []
    for row in grid:
        key = tuple(row)
        if key not in seen:
            seen.append(key)
            result.append(row[:])
    return result if result else grid


def keep_unique_cols(grid: Grid) -> Grid:
    """Keep only unique columns (remove duplicate columns)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    cols = [tuple(grid[r][c] for r in range(h)) for c in range(w)]
    seen = []
    keep_indices = []
    for i, col in enumerate(cols):
        if col not in seen:
            seen.append(col)
            keep_indices.append(i)
    if not keep_indices:
        return grid
    return [[grid[r][c] for c in keep_indices] for r in range(h)]


def extract_repeating_pattern(grid: Grid) -> Grid:
    """Extract the smallest repeating tile from a grid.

    Finds the smallest rectangle that, when tiled, reproduces the grid.
    Returns the tile. If no period is found, returns the grid unchanged.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Find vertical period
    best_ph = h
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        valid = True
        for r in range(h):
            for c in range(w):
                if grid[r][c] != grid[r % ph][c]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_ph = ph
            break

    # Find horizontal period
    best_pw = w
    for pw in range(1, w + 1):
        if w % pw != 0:
            continue
        valid = True
        for r in range(h):
            for c in range(w):
                if grid[r][c] != grid[r][c % pw]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_pw = pw
            break

    if best_ph == h and best_pw == w:
        return grid
    return [row[:best_pw] for row in grid[:best_ph]]


def xor_grids(grid1: Grid, grid2: Grid) -> Grid:
    """XOR two grids: non-zero where exactly one input is non-zero."""
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            a, b = grid1[r][c], grid2[r][c]
            if a != 0 and b == 0:
                row.append(a)
            elif a == 0 and b != 0:
                row.append(b)
            else:
                row.append(0)
        result.append(row)
    return result


def subtract_grids(grid1: Grid, grid2: Grid) -> Grid:
    """Subtract grid2 from grid1: keep grid1 pixels where grid2 is zero."""
    if not grid1 or not grid2:
        return grid1 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    return [[grid1[r][c] if grid2[r][c] == 0 else 0
             for c in range(w)] for r in range(h)]


def label_components(grid: Grid) -> Grid:
    """Label each connected foreground component with a unique color (1, 2, 3, ...).

    Background (most common color) stays 0. Each connected 4-neighbor
    region of non-background pixels gets a unique label. This is the
    atomic foundation for object-level operations — composition with
    keep_color/erase_color can isolate specific components.

    DISCOVERY GOAL: compositions like keep_color(largest_object_color)(label_components(x))
    should achieve extract_largest_object.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]

    result = [[0] * w for _ in range(h)]
    visited = set()
    label = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                label += 1
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    result[cr][cc] = label
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return result


# =============================================================================
# Binary transforms (1)
# =============================================================================

def mask_by(grid1: Grid, grid2: Grid) -> Grid:
    """Keep pixels from grid1 where grid2 is non-zero, zero elsewhere.

    This is the dual of overlay: overlay combines, mask_by filters.
    Enables compositions like mask_by(original, keep_color(largest)(label_components(original)))
    to extract the largest connected component with original colors.
    """
    if not grid1 or not grid2:
        return grid1 or grid2 or []
    h = min(len(grid1), len(grid2))
    w = min(len(grid1[0]), len(grid2[0])) if grid1[0] and grid2[0] else 0
    return [[grid1[r][c] if grid2[r][c] != 0 else 0
             for c in range(w)] for r in range(h)]


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


def _repeat_rows_factory(n: int):
    """Factory: repeat each row n times."""
    if not isinstance(n, int) or n < 1 or n > 10:
        return lambda grid: grid
    def repeat_rows(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        result = []
        for row in grid:
            for _ in range(n):
                result.append(row[:])
        return result
    return repeat_rows


def _repeat_cols_factory(n: int):
    """Factory: repeat each column n times."""
    if not isinstance(n, int) or n < 1 or n > 10:
        return lambda grid: grid
    def repeat_cols(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        return [[cell for cell in row for _ in range(n)] for row in grid]
    return repeat_cols


def _recolor_foreground_factory(color: int):
    """Factory: replace all non-background colors with given color."""
    def recolor(grid: Grid) -> Grid:
        if not grid or not grid[0]:
            return grid
        flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
        bg = Counter(flat).most_common(1)[0][0]
        return [[color if cell != bg else cell for cell in row] for row in grid]
    return recolor


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

    31 unary transforms + 2 binary (overlay, mask_by) = 33 total.
    Each performs exactly ONE visual concept.

    crop_to_content is NOT a primitive — it's compositional:
    trim_cols(trim_rows(x)) or trim_rows(trim_cols(x)).
    """
    unary_ops = [
        # Geometric (6) — each is one intuitive concept
        ("rotate_90_clockwise",         rotate_90_cw),
        ("rotate_90_counterclockwise",  rotate_90_ccw),
        ("rotate_180",                  rotate_180),
        ("mirror_horizontal",           mirror_horizontal),
        ("mirror_vertical",             mirror_vertical),
        ("transpose",                   transpose),
        # Spatial (7)
        ("trim_rows",                   trim_rows),
        ("trim_cols",                   trim_cols),
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
        # Physics (4) — directional gravity
        ("gravity_down",                gravity_down),
        ("gravity_up",                  gravity_up),
        ("gravity_left",                gravity_left),
        ("gravity_right",               gravity_right),
        # Sorting (2)
        ("sort_rows_by_nonzero",        sort_rows_by_nonzero),
        ("sort_cols_by_nonzero",        sort_cols_by_nonzero),
        # Fill (2)
        ("fill_enclosed",               fill_enclosed),
        ("border_extend",               border_extend),
        # Ray extension (4) — extend colored pixels in a direction
        ("extend_rays_right",           extend_rays_right),
        ("extend_rays_left",            extend_rays_left),
        ("extend_rays_down",            extend_rays_down),
        ("extend_rays_up",              extend_rays_up),
        # Flood fill from markers (1) — paint bucket from seeds
        ("flood_fill_from_markers",     flood_fill_from_markers),
        # Connected component labeling (1) — single BFS, truly atomic
        ("label_components",            label_components),
        # Object extraction (2) — extract largest/smallest by size
        ("extract_largest_object",      extract_largest_object),
        ("extract_smallest_object",     extract_smallest_object),
        # Object outline (1) — hollow interior
        ("hollow_objects",              hollow_objects),
        # Symmetry completion (3) — fill gaps to make symmetric
        ("complete_horizontal_symmetry", complete_horizontal_symmetry),
        ("complete_vertical_symmetry",  complete_vertical_symmetry),
        ("fill_diagonal_symmetry",      fill_diagonal_symmetry),
        # Spatial (2) — border removal, denoising
        ("remove_border",              remove_border),
        ("denoise_majority",           denoise_majority),
        # Deduplication (2) — remove repeated rows/columns
        ("keep_unique_rows",           keep_unique_rows),
        ("keep_unique_cols",           keep_unique_cols),
        # Pattern (1) — extract repeating tile
        ("extract_repeating_pattern",  extract_repeating_pattern),
    ]

    prims = [Primitive(name=name, arity=1, fn=fn, domain="arc")
             for name, fn in unary_ops]
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))
    prims.append(Primitive(name="mask_by", arity=2, fn=mask_by, domain="arc"))
    prims.append(Primitive(name="xor_grids", arity=2, fn=xor_grids, domain="arc"))
    prims.append(Primitive(name="subtract_grids", arity=2, fn=subtract_grids, domain="arc"))
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
        Primitive(name="recolor_foreground", arity=1,
                  fn=_recolor_foreground_factory, domain="arc", kind="parameterized"),
        # Scale/tile parameterized
        Primitive(name="scale", arity=1,
                  fn=_scale_factory, domain="arc", kind="parameterized"),
        Primitive(name="tile", arity=1,
                  fn=_tile_factory, domain="arc", kind="parameterized"),
        Primitive(name="downscale", arity=1,
                  fn=_downscale_factory, domain="arc", kind="parameterized"),
        # Row/column repetition parameterized
        Primitive(name="repeat_rows", arity=1,
                  fn=_repeat_rows_factory, domain="arc", kind="parameterized"),
        Primitive(name="repeat_cols", arity=1,
                  fn=_repeat_cols_factory, domain="arc", kind="parameterized"),
    ]


# =============================================================================
# Essential pair concepts for atomic vocabulary
# =============================================================================

ATOMIC_ESSENTIAL_PAIR_CONCEPTS: frozenset = frozenset([
    "trim_rows",
    "trim_cols",
    "overlay",
    "mirror_horizontal",
    "mirror_vertical",
    "binarize",
    "dilate",
    "erode",
    "extract_largest_object",
    "extract_smallest_object",
    "hollow_objects",
    "remove_border",
    "complete_horizontal_symmetry",
    "complete_vertical_symmetry",
])
