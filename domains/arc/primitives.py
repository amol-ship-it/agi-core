"""
ARC-AGI Grid Transformation Primitives.

All Grid→Grid functions used by the ARC domain.
Organized into categories: geometric, color, spatial, object-level,
grid partitioning, diagonal/line, morphological, etc.

Primitives adapted from vibhor-77/agi-mvp-general.
"""

from __future__ import annotations

from collections import Counter
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


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) of a grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid[0] else 0)


def valid_grid(grid: Grid) -> bool:
    """Check if a grid is valid (non-empty, rectangular, values 0-9)."""
    if not grid or not grid[0]:
        return False
    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            return False
        for v in row:
            if not (0 <= v <= 9):
                return False
    return True


def empty_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    """Create an empty grid filled with a single color."""
    return [[fill] * cols for _ in range(rows)]

# =============================================================================
# ARC Grid Primitives — the atomic building blocks
# =============================================================================
# Each primitive takes a Grid and returns a Grid.
# Adapted from agi-mvp-general/arc_agent/primitives.py

def identity(grid: Grid) -> Grid:
    """No-op: return the grid unchanged."""
    return [row[:] for row in grid]


# --- Geometric transforms ---

def rotate_90_cw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=-1))


def rotate_90_ccw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees counter-clockwise."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=1))


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=2))


def mirror_horizontal(grid: Grid) -> Grid:
    """Flip grid left-right."""
    return [row[::-1] for row in grid]


def mirror_vertical(grid: Grid) -> Grid:
    """Flip grid top-bottom."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Transpose grid (swap rows and columns)."""
    arr = to_np(grid)
    return from_np(arr.T)


# --- Color transforms ---

def invert_colors(grid: Grid) -> Grid:
    """Invert all non-zero colors: c -> 10 - c (0 stays 0)."""
    return [[0 if c == 0 else 10 - c for c in row] for row in grid]


def replace_bg_with_most_common(grid: Grid) -> Grid:
    """Replace background (0) with the most common non-zero color."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    mc = Counter(flat).most_common(1)[0][0]
    return [[mc if c == 0 else c for c in row] for row in grid]


def keep_color(grid: Grid, color: int) -> Grid:
    """Keep only pixels of the specified color, zero everything else."""
    return [[c if c == color else 0 for c in row] for row in grid]


def remove_color(grid: Grid, color: int) -> Grid:
    """Remove all pixels of the specified color (set to 0)."""
    return [[0 if c == color else c for c in row] for row in grid]


def most_common_color(grid: Grid) -> int:
    """Return the most common non-zero color in the grid."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return 0
    return Counter(flat).most_common(1)[0][0]


def fill_color(grid: Grid, color: int) -> Grid:
    """Fill entire grid with a single color."""
    rows, cols = grid_shape(grid)
    return empty_grid(rows, cols, color)


# --- Color-parameterized primitives (curried for colors 1-9) ---

def _make_replace_color(from_c: int, to_c: int):
    """Create a primitive that replaces from_c with to_c."""
    def fn(grid: Grid) -> Grid:
        return [[to_c if c == from_c else c for c in row] for row in grid]
    fn.__name__ = f"replace_{from_c}_with_{to_c}"
    return fn


def _make_color_remap(mapping: dict[int, int]):
    """Create a primitive that applies a multi-color remapping."""
    def fn(grid: Grid) -> Grid:
        a = to_np(grid)
        result = a.copy()
        for old_c, new_c in mapping.items():
            result[a == old_c] = new_c
        return result.tolist()
    fn.__name__ = f"color_remap_{'_'.join(f'{k}to{v}' for k, v in sorted(mapping.items()))}"
    return fn


def _make_keep_color(color: int):
    """Create a primitive that keeps only the given color."""
    def fn(grid: Grid) -> Grid:
        return [[c if c == color else 0 for c in row] for row in grid]
    fn.__name__ = f"keep_color_{color}"
    return fn


# --- Spatial / cropping ---

def crop_to_nonzero(grid: Grid) -> Grid:
    """Crop to the bounding box of all non-zero pixels."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        return [[0]]
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return from_np(arr[r_min:r_max + 1, c_min:c_max + 1])


def crop_to_color(grid: Grid, color: int) -> Grid:
    """Crop to the bounding box of a specific color."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr == color)
    if nonzero.size == 0:
        return [[0]]
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return from_np(arr[r_min:r_max + 1, c_min:c_max + 1])


def get_top_half(grid: Grid) -> Grid:
    """Return the top half of the grid."""
    h = len(grid)
    return [row[:] for row in grid[:h // 2]]


def get_bottom_half(grid: Grid) -> Grid:
    """Return the bottom half of the grid."""
    h = len(grid)
    return [row[:] for row in grid[h // 2:]]


def get_left_half(grid: Grid) -> Grid:
    """Return the left half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[:w // 2] for row in grid]


def get_right_half(grid: Grid) -> Grid:
    """Return the right half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[w // 2:] for row in grid]


# --- Tiling / scaling ---

def tile_2x2(grid: Grid) -> Grid:
    """Tile the grid 2x2."""
    arr = to_np(grid)
    return from_np(np.tile(arr, (2, 2)))


def tile_3x3(grid: Grid) -> Grid:
    """Tile the grid 3x3."""
    arr = to_np(grid)
    return from_np(np.tile(arr, (3, 3)))


def scale_2x(grid: Grid) -> Grid:
    """Scale each pixel to a 2x2 block."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1))


def scale_3x(grid: Grid) -> Grid:
    """Scale each pixel to a 3x3 block."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 3, axis=0), 3, axis=1))


# --- Gravity ---

def gravity_down(grid: Grid) -> Grid:
    """Drop all non-zero pixels to the bottom of each column."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for c in range(cols):
        col = arr[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[rows - len(nonzero):, c] = nonzero
    return from_np(result)


def gravity_up(grid: Grid) -> Grid:
    """Push all non-zero pixels to the top of each column."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for c in range(cols):
        col = arr[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[:len(nonzero), c] = nonzero
    return from_np(result)


def gravity_left(grid: Grid) -> Grid:
    """Push all non-zero pixels to the left of each row."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for r in range(rows):
        row = arr[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, :len(nonzero)] = nonzero
    return from_np(result)


def gravity_right(grid: Grid) -> Grid:
    """Push all non-zero pixels to the right of each row."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for r in range(rows):
        row = arr[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, cols - len(nonzero):] = nonzero
    return from_np(result)


# --- Pattern detection / filling ---

def outline(grid: Grid) -> Grid:
    """Draw outline around non-zero regions (mark boundary pixels)."""
    arr = to_np(grid)
    nonzero = arr != 0
    # A pixel is boundary if it's non-zero AND at least one 4-neighbor is zero or OOB
    padded = np.pad(arr, 1, mode='constant', constant_values=0)
    neighbor_zero = (
        (padded[:-2, 1:-1] == 0) |  # top
        (padded[2:, 1:-1] == 0) |    # bottom
        (padded[1:-1, :-2] == 0) |   # left
        (padded[1:-1, 2:] == 0)      # right
    )
    result = np.where(nonzero & neighbor_zero, arr, 0)
    return from_np(result)


def fill_enclosed(grid: Grid) -> Grid:
    """Fill enclosed background regions with the surrounding color."""
    arr = to_np(grid).copy()
    rows, cols = arr.shape
    # Flood fill from borders to find non-enclosed background
    visited = np.zeros((rows, cols), dtype=bool)
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and arr[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Fill enclosed zeros: use most common non-zero neighbor color
    enclosed = (arr == 0) & ~visited
    if not np.any(enclosed):
        return from_np(arr)

    # For enclosed cells, find the most common adjacent non-zero color
    mc = most_common_color(grid)
    padded = np.pad(arr, 1, mode='constant', constant_values=0)
    enc_rows, enc_cols = np.where(enclosed)
    for r, c in zip(enc_rows, enc_cols):
        # Check 4-neighbors in padded array (offset by 1)
        neighbors = []
        for nr, nc in [(r, c+1), (r+2, c+1), (r+1, c), (r+1, c+2)]:
            v = padded[nr, nc]
            if v != 0:
                neighbors.append(v)
        arr[r, c] = Counter(neighbors).most_common(1)[0][0] if neighbors else mc
    return from_np(arr)


def denoise_3x3(grid: Grid) -> Grid:
    """Replace each pixel with the majority color in its 3x3 neighborhood."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = arr.copy()
    # For each color 0-9, count occurrences in each cell's 3x3 neighborhood
    # using convolution-like approach with numpy
    padded = np.pad(arr, 1, mode='edge')
    counts = np.zeros((rows, cols, 10), dtype=np.int32)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            shifted = padded[1 + dr:rows + 1 + dr, 1 + dc:cols + 1 + dc]
            for color in range(10):
                counts[:, :, color] += (shifted == color).astype(np.int32)
    result = np.argmax(counts, axis=2).astype(np.int32)
    return from_np(result)


# --- Logical combinations of halves ---

def xor_halves_v(grid: Grid) -> Grid:
    """XOR top and bottom halves (keep pixel if only in one half)."""
    h = len(grid)
    half = h // 2
    top = to_np(grid[:half])
    bot = to_np(grid[half:half + half])
    if top.shape != bot.shape:
        return grid
    result = np.where((top != 0) ^ (bot != 0), np.maximum(top, bot), 0)
    return from_np(result)


def or_halves_v(grid: Grid) -> Grid:
    """OR top and bottom halves (overlay non-zero pixels)."""
    h = len(grid)
    half = h // 2
    top = to_np(grid[:half])
    bot = to_np(grid[half:half + half])
    if top.shape != bot.shape:
        return grid
    result = np.where(top != 0, top, bot)
    return from_np(result)


def xor_halves_h(grid: Grid) -> Grid:
    """XOR left and right halves."""
    arr = to_np(grid)
    w = arr.shape[1]
    half = w // 2
    left = arr[:, :half]
    right = arr[:, half:half + half]
    if left.shape != right.shape:
        return grid
    result = np.where((left != 0) ^ (right != 0), np.maximum(left, right), 0)
    return from_np(result)


def or_halves_h(grid: Grid) -> Grid:
    """OR left and right halves."""
    arr = to_np(grid)
    w = arr.shape[1]
    half = w // 2
    left = arr[:, :half]
    right = arr[:, half:half + half]
    if left.shape != right.shape:
        return grid
    result = np.where(left != 0, left, right)
    return from_np(result)


# --- Extract / isolate objects ---

def extract_largest_object(grid: Grid) -> Grid:
    """Extract the largest connected non-zero region (4-connected), cropped."""
    arr = to_np(grid)
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    best_mask = None
    best_size = 0
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] != 0 and not visited[r, c]:
                mask = np.zeros_like(arr, dtype=bool)
                queue = [(r, c)]
                visited[r, c] = True
                mask[r, c] = True
                size = 0
                while queue:
                    cr, cc = queue.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and arr[nr, nc] != 0:
                            visited[nr, nc] = True
                            mask[nr, nc] = True
                            queue.append((nr, nc))
                if size > best_size:
                    best_size = size
                    best_mask = mask
    if best_mask is None:
        return [[0]]
    result = np.where(best_mask, arr, 0)
    nz = np.argwhere(result != 0)
    r_min, c_min = nz.min(axis=0)
    r_max, c_max = nz.max(axis=0)
    return from_np(result[r_min:r_max + 1, c_min:c_max + 1])


def extract_smallest_object(grid: Grid) -> Grid:
    """Extract the smallest connected non-zero region (4-connected), cropped."""
    arr = to_np(grid)
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    best_mask = None
    best_size = float('inf')
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] != 0 and not visited[r, c]:
                mask = np.zeros_like(arr, dtype=bool)
                queue = [(r, c)]
                visited[r, c] = True
                mask[r, c] = True
                size = 0
                while queue:
                    cr, cc = queue.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and arr[nr, nc] != 0:
                            visited[nr, nc] = True
                            mask[nr, nc] = True
                            queue.append((nr, nc))
                if size < best_size:
                    best_size = size
                    best_mask = mask
    if best_mask is None:
        return [[0]]
    result = np.where(best_mask, arr, 0)
    nz = np.argwhere(result != 0)
    r_min, c_min = nz.min(axis=0)
    r_max, c_max = nz.max(axis=0)
    return from_np(result[r_min:r_max + 1, c_min:c_max + 1])


# --- Symmetry operations ---

def anti_diagonal_mirror(grid: Grid) -> Grid:
    """Mirror along the anti-diagonal."""
    arr = to_np(grid)
    return from_np(np.flip(np.flip(arr, 0), 1).T)


def make_symmetric_h(grid: Grid) -> Grid:
    """Make the grid horizontally symmetric by mirroring the left half."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = arr.copy()
    for c in range(cols // 2):
        result[:, cols - 1 - c] = result[:, c]
    return from_np(result)


def make_symmetric_v(grid: Grid) -> Grid:
    """Make the grid vertically symmetric by mirroring the top half."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = arr.copy()
    for r in range(rows // 2):
        result[rows - 1 - r, :] = result[r, :]
    return from_np(result)


# --- Pattern replication ---

def repeat_pattern_right(grid: Grid) -> Grid:
    """Repeat the grid once to the right."""
    arr = to_np(grid)
    return from_np(np.concatenate([arr, arr], axis=1))


def repeat_pattern_down(grid: Grid) -> Grid:
    """Repeat the grid once downward."""
    arr = to_np(grid)
    return from_np(np.concatenate([arr, arr], axis=0))


# --- Border operations ---

def add_border(grid: Grid) -> Grid:
    """Add a 1-pixel border of the most common non-zero color."""
    mc = most_common_color(grid)
    if mc == 0:
        mc = 1
    arr = to_np(grid)
    return from_np(np.pad(arr, 1, mode='constant', constant_values=mc))


def remove_border(grid: Grid) -> Grid:
    """Remove the 1-pixel border."""
    arr = to_np(grid)
    if arr.shape[0] <= 2 or arr.shape[1] <= 2:
        return grid
    return from_np(arr[1:-1, 1:-1])


# --- Sorting / reordering ---

def sort_rows_by_color_count(grid: Grid) -> Grid:
    """Sort rows by the number of non-zero colors (ascending)."""
    rows = [row[:] for row in grid]
    rows.sort(key=lambda r: sum(1 for c in r if c != 0))
    return rows


def sort_cols_by_color_count(grid: Grid) -> Grid:
    """Sort columns by the number of non-zero colors (ascending)."""
    arr = to_np(grid)
    counts = np.sum(arr != 0, axis=0)
    order = np.argsort(counts)
    return from_np(arr[:, order])


# --- Unique row/col operations ---

def unique_rows(grid: Grid) -> Grid:
    """Keep only unique rows (first occurrence)."""
    seen = []
    result = []
    for row in grid:
        key = tuple(row)
        if key not in seen:
            seen.append(key)
            result.append(row[:])
    return result if result else [[0]]


def unique_cols(grid: Grid) -> Grid:
    """Keep only unique columns (first occurrence)."""
    arr = to_np(grid)
    _, idx = np.unique(arr, axis=1, return_index=True)
    result = arr[:, sorted(idx)]
    return from_np(result) if result.size > 0 else [[0]]


# --- Color mapping ---

def recolor_by_size_rank(grid: Grid) -> Grid:
    """Recolor: most frequent non-zero color -> 1, next -> 2, etc."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    counts = Counter(flat)
    ranked = [c for c, _ in counts.most_common()]
    mapping = {c: i + 1 for i, c in enumerate(ranked)}
    return [[mapping.get(c, c) if c != 0 else 0 for c in row] for row in grid]


# --- Extend lines ---

def extend_lines_h(grid: Grid) -> Grid:
    """Extend non-zero pixels horizontally to fill their row."""
    arr = to_np(grid)
    result = arr.copy()
    rows, cols = arr.shape
    for r in range(rows):
        colors = arr[r, arr[r] != 0]
        if len(colors) > 0:
            mc = Counter(colors.tolist()).most_common(1)[0][0]
            for c in range(cols):
                if result[r, c] == 0:
                    result[r, c] = mc
    return from_np(result)


def extend_lines_v(grid: Grid) -> Grid:
    """Extend non-zero pixels vertically to fill their column."""
    arr = to_np(grid)
    result = arr.copy()
    rows, cols = arr.shape
    for c in range(cols):
        colors = arr[arr[:, c] != 0, c]
        if len(colors) > 0:
            mc = Counter(colors.tolist()).most_common(1)[0][0]
            for r in range(rows):
                if result[r, c] == 0:
                    result[r, c] = mc
    return from_np(result)


# --- Object detection helpers ---

def count_colors(grid: Grid) -> int:
    """Count unique non-zero colors in the grid."""
    return len(set(c for row in grid for c in row if c != 0))


def find_bounding_box(grid: Grid) -> tuple[int, int, int, int]:
    """Return (r_min, c_min, r_max, c_max) of non-zero pixels."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        return (0, 0, 0, 0)
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return (int(r_min), int(c_min), int(r_max), int(c_max))


# =============================================================================
# Connected component detection (internal utility)
# =============================================================================

def _find_connected_components(grid: Grid) -> list[dict]:
    """Find all connected components (objects) via 4-connectivity flood fill.

    Returns list of dicts with keys: color, pixels (set of (r,c)), bbox, size.
    Background (0) is excluded.
    """
    if not grid or not grid[0]:
        return []
    height, width = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                pixels: set[tuple[int, int]] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= height or cc < 0 or cc >= width:
                        continue
                    if grid[cr][cc] != color:
                        continue
                    visited.add((cr, cc))
                    pixels.add((cr, cc))
                    stack.extend([
                        (cr - 1, cc), (cr + 1, cc),
                        (cr, cc - 1), (cr, cc + 1),
                    ])
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                components.append({
                    "color": color,
                    "pixels": pixels,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "size": len(pixels),
                })
    return components


def _component_to_subgrid(comp: dict) -> Grid:
    """Extract a component as a minimal cropped sub-grid."""
    min_r, min_c, max_r, max_c = comp["bbox"]
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    result = [[0] * w for _ in range(h)]
    for r, c in comp["pixels"]:
        result[r - min_r][c - min_c] = comp["color"]
    return result


# =============================================================================
# Object-level primitives (connected component based)
# =============================================================================

def keep_largest_object_only(grid: Grid) -> Grid:
    """Keep only the largest connected component, zero everything else."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    largest = max(comps, key=lambda o: o["size"])
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r, c in largest["pixels"]:
        result[r][c] = grid[r][c]
    return result


def keep_smallest_object_only(grid: Grid) -> Grid:
    """Keep only the smallest connected component, zero everything else."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    smallest = min(comps, key=lambda o: o["size"])
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r, c in smallest["pixels"]:
        result[r][c] = grid[r][c]
    return result


def remove_largest_object(grid: Grid) -> Grid:
    """Remove the largest connected component (set its pixels to 0)."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    largest = max(comps, key=lambda o: o["size"])
    result = [row[:] for row in grid]
    for r, c in largest["pixels"]:
        result[r][c] = 0
    return result


def remove_smallest_object(grid: Grid) -> Grid:
    """Remove the smallest connected component (set its pixels to 0)."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    smallest = min(comps, key=lambda o: o["size"])
    result = [row[:] for row in grid]
    for r, c in smallest["pixels"]:
        result[r][c] = 0
    return result


def count_objects_as_grid(grid: Grid) -> Grid:
    """Return a 1x1 grid whose value is the number of connected components."""
    n = len(_find_connected_components(grid))
    return [[min(n, 9)]]


def recolor_each_object(grid: Grid) -> Grid:
    """Recolor each connected component with a unique color (1, 2, 3, ...)."""
    comps = _find_connected_components(grid)
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for i, comp in enumerate(comps):
        color = (i % 9) + 1
        for r, c in comp["pixels"]:
            result[r][c] = color
    return result


def mirror_objects_h(grid: Grid) -> Grid:
    """Mirror each connected component horizontally within its bounding box."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    # Copy background
    all_pixels: set[tuple[int, int]] = set()
    for comp in comps:
        all_pixels.update(comp["pixels"])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if (r, c) not in all_pixels:
                result[r][c] = grid[r][c]
    # Mirror each object within its bbox
    for comp in comps:
        min_r, min_c, max_r, max_c = comp["bbox"]
        for r, c in comp["pixels"]:
            mirrored_c = max_c - (c - min_c)
            result[r][mirrored_c] = comp["color"]
    return result


def mirror_objects_v(grid: Grid) -> Grid:
    """Mirror each connected component vertically within its bounding box."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    all_pixels: set[tuple[int, int]] = set()
    for comp in comps:
        all_pixels.update(comp["pixels"])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if (r, c) not in all_pixels:
                result[r][c] = grid[r][c]
    for comp in comps:
        min_r, min_c, max_r, max_c = comp["bbox"]
        for r, c in comp["pixels"]:
            mirrored_r = max_r - (r - min_r)
            result[mirrored_r][c] = comp["color"]
    return result


def sort_objects_by_size(grid: Grid) -> Grid:
    """Sort objects by size: rearrange from left-to-right, smallest to largest."""
    comps = _find_connected_components(grid)
    if len(comps) < 2:
        return [row[:] for row in grid]
    # Sort by size
    comps.sort(key=lambda o: o["size"])
    # Extract subgrids
    subgrids = [_component_to_subgrid(c) for c in comps]
    # Pack horizontally
    max_h = max(len(sg) for sg in subgrids)
    total_w = sum(len(sg[0]) for sg in subgrids) + len(subgrids) - 1
    result = [[0] * total_w for _ in range(max_h)]
    col_offset = 0
    for sg in subgrids:
        sh, sw = len(sg), len(sg[0])
        for r in range(sh):
            for c in range(sw):
                if sg[r][c] != 0:
                    result[r][col_offset + c] = sg[r][c]
        col_offset += sw + 1
    return result


def flood_fill_bg(grid: Grid) -> Grid:
    """Flood-fill background (0) regions enclosed by non-zero cells.

    Uses the most common non-zero color adjacent to each enclosed region.
    """
    arr = to_np(grid)
    h, w = arr.shape
    # Find background regions connected to the border (not enclosed)
    border_connected = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in [0, w - 1]:
            if arr[r, c] == 0 and not border_connected[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if arr[r, c] == 0 and not border_connected[r, c]:
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        if cr < 0 or cr >= h or cc < 0 or cc >= w:
            continue
        if border_connected[cr, cc] or arr[cr, cc] != 0:
            continue
        border_connected[cr, cc] = True
        stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])

    result = arr.copy()
    # Fill enclosed background cells with the most common adjacent non-zero color
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 and not border_connected[r, c]:
                # Find most common neighboring non-zero color
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] != 0:
                        neighbors.append(int(arr[nr, nc]))
                if neighbors:
                    result[r, c] = Counter(neighbors).most_common(1)[0][0]
                else:
                    result[r, c] = 1  # fallback
    return from_np(result)


# =============================================================================
# Grid partitioning primitives
# =============================================================================

def detect_grid_lines(grid: Grid) -> tuple[list[int], list[int]]:
    """Detect horizontal and vertical separator lines in the grid.

    A separator line is a full row or column of a single non-zero color.
    Returns (horizontal_line_rows, vertical_line_cols).
    """
    arr = to_np(grid)
    h, w = arr.shape
    h_lines = []
    v_lines = []

    for r in range(h):
        row = arr[r]
        vals = set(row.tolist())
        if len(vals) == 1 and 0 not in vals:
            h_lines.append(r)

    for c in range(w):
        col = arr[:, c]
        vals = set(col.tolist())
        if len(vals) == 1 and 0 not in vals:
            v_lines.append(c)

    return h_lines, v_lines


def extract_top_left_cell(grid: Grid) -> Grid:
    """If grid has separator lines, extract the top-left cell."""
    h_lines, v_lines = detect_grid_lines(grid)
    top = 0
    bottom = h_lines[0] if h_lines else len(grid)
    left = 0
    right = v_lines[0] if v_lines else len(grid[0])
    if bottom <= top or right <= left:
        return [row[:] for row in grid]
    return [row[left:right] for row in grid[top:bottom]]


def extract_bottom_right_cell(grid: Grid) -> Grid:
    """If grid has separator lines, extract the bottom-right cell."""
    h_lines, v_lines = detect_grid_lines(grid)
    top = (h_lines[-1] + 1) if h_lines else 0
    bottom = len(grid)
    left = (v_lines[-1] + 1) if v_lines else 0
    right = len(grid[0])
    if bottom <= top or right <= left:
        return [row[:] for row in grid]
    return [row[left:right] for row in grid[top:bottom]]


def remove_grid_lines(grid: Grid) -> Grid:
    """Remove separator lines (full rows/cols of one color), keeping cells."""
    h_lines, v_lines = detect_grid_lines(grid)
    if not h_lines and not v_lines:
        return [row[:] for row in grid]
    h_set = set(h_lines)
    v_set = set(v_lines)
    result = []
    for r in range(len(grid)):
        if r in h_set:
            continue
        new_row = [grid[r][c] for c in range(len(grid[0])) if c not in v_set]
        if new_row:
            result.append(new_row)
    return result if result else [[0]]


# =============================================================================
# Diagonal and line extension primitives
# =============================================================================

def shift_rows_right(grid: Grid) -> Grid:
    """Shift each row right by its row index (creating a diagonal staircase)."""
    arr = to_np(grid)
    h, w = arr.shape
    new_w = w + h - 1
    result = np.zeros((h, new_w), dtype=np.int32)
    for r in range(h):
        result[r, r:r+w] = arr[r]
    return from_np(result)


def shift_rows_left(grid: Grid) -> Grid:
    """Shift each row left by its row index (reverse diagonal staircase)."""
    arr = to_np(grid)
    h, w = arr.shape
    new_w = w + h - 1
    result = np.zeros((h, new_w), dtype=np.int32)
    for r in range(h):
        offset = h - 1 - r
        result[r, offset:offset+w] = arr[r]
    return from_np(result)


def extend_lines(grid: Grid) -> Grid:
    """Extend partial lines (>=2 consecutive cells) to grid boundaries.

    Detects horizontal and vertical runs of the same non-zero color
    and extends them in both directions until hitting another color or edge.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0:
        return grid
    result = [row[:] for row in grid]

    # Extend horizontal lines
    for r in range(h):
        c = 0
        while c < w:
            if result[r][c] != 0:
                color = result[r][c]
                start = c
                while c < w and result[r][c] == color:
                    c += 1
                end = c - 1
                if end - start + 1 >= 2:
                    for lc in range(start - 1, -1, -1):
                        if result[r][lc] != 0:
                            break
                        result[r][lc] = color
                    for rc in range(end + 1, w):
                        if result[r][rc] != 0:
                            break
                        result[r][rc] = color
            else:
                c += 1

    # Extend vertical lines
    for c in range(w):
        r = 0
        while r < h:
            if result[r][c] != 0:
                color = result[r][c]
                start = r
                while r < h and result[r][c] == color:
                    r += 1
                end = r - 1
                if end - start + 1 >= 2:
                    for lr in range(start - 1, -1, -1):
                        if result[lr][c] != 0:
                            break
                        result[lr][c] = color
                    for rr in range(end + 1, h):
                        if result[rr][c] != 0:
                            break
                        result[rr][c] = color
            else:
                r += 1

    return result


def extend_diagonal_lines(grid: Grid) -> Grid:
    """Extend isolated non-zero cells diagonally (both main and anti-diag)."""
    arr = to_np(grid)
    h, w = arr.shape
    result = arr.copy()

    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = int(arr[r, c])
                # Check if isolated (no same-color orthogonal neighbors)
                has_neighbor = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == color:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    # Extend along all 4 diagonals until hitting non-zero or edge
                    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = color
                            nr += dr
                            nc += dc

    return from_np(result)


def binarize(grid: Grid) -> Grid:
    """Convert grid to binary: non-zero → 1, zero → 0."""
    return [[1 if c != 0 else 0 for c in row] for row in grid]


def color_to_most_common(grid: Grid) -> Grid:
    """Replace all non-zero colors with the most common non-zero color."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    mc = Counter(flat).most_common(1)[0][0]
    return [[mc if c != 0 else 0 for c in row] for row in grid]


def upscale_pattern(grid: Grid) -> Grid:
    """If grid is small (<=5x5), upscale by treating each cell as the grid itself."""
    h, w = len(grid), len(grid[0])
    if h > 5 or w > 5 or h == 0 or w == 0:
        return [row[:] for row in grid]
    # Each non-zero cell becomes a copy of the entire grid
    new_h = h * h
    new_w = w * w
    result = [[0] * new_w for _ in range(new_h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                for ir in range(h):
                    for ic in range(w):
                        result[r * h + ir][c * w + ic] = grid[ir][ic]
    return result


# =============================================================================
# Near-miss-targeted primitives: anomaly removal, rectangle ops
# =============================================================================

def denoise_majority(grid: Grid) -> Grid:
    """Replace each cell with the majority color in its 3x3 neighborhood.

    More aggressive than denoise_3x3: replaces ANY cell that disagrees
    with its neighborhood majority, not just cells with a 3x3 mode.
    Effective for removing scattered noise within uniform objects.
    """
    arr = to_np(grid)
    h, w = arr.shape
    result = arr.copy()
    for r in range(h):
        for c in range(w):
            neighbors = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(int(arr[nr, nc]))
            majority = Counter(neighbors).most_common(1)[0][0]
            result[r, c] = majority
    return from_np(result)


def fill_rectangles(grid: Grid) -> Grid:
    """Find rectangular regions and fill holes within them.

    For each connected component, if it's roughly rectangular
    (compactness > 0.6), fill the entire bounding box with its color.
    """
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    result = [row[:] for row in grid]
    for comp in comps:
        min_r, min_c, max_r, max_c = comp["bbox"]
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        compactness = comp["size"] / bbox_area if bbox_area > 0 else 0
        if compactness > 0.6:
            # Fill entire bounding box
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    result[r][c] = comp["color"]
    return result


def extract_minority_color(grid: Grid) -> Grid:
    """Keep only the least common non-zero color, zero everything else."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    counts = Counter(flat)
    minority = counts.most_common()[-1][0]
    return [[c if c == minority else 0 for c in row] for row in grid]


def extract_majority_color(grid: Grid) -> Grid:
    """Keep only the most common non-zero color, zero everything else."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    majority = Counter(flat).most_common(1)[0][0]
    return [[c if c == majority else 0 for c in row] for row in grid]


def replace_noise_in_objects(grid: Grid) -> Grid:
    """For each connected component, replace any enclosed different-color
    cells with the component's color.

    Finds objects, then for each cell inside an object's bounding box
    that's a different non-zero color, replaces it with the object's color.
    """
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]

    # Sort by size descending — larger objects get priority
    comps.sort(key=lambda c: c["size"], reverse=True)
    result = [row[:] for row in grid]
    claimed: set[tuple[int, int]] = set()

    for comp in comps:
        min_r, min_c, max_r, max_c = comp["bbox"]
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        compactness = comp["size"] / bbox_area if bbox_area > 0 else 0
        if compactness > 0.5:
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if (r, c) not in claimed and result[r][c] != 0:
                        result[r][c] = comp["color"]
                        claimed.add((r, c))

    return result


def hollow_objects(grid: Grid) -> Grid:
    """Convert filled objects to their outlines only (hollow them out)."""
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]

    for comp in comps:
        for r, c in comp["pixels"]:
            # Check if this pixel is on the border (has a non-same-color neighbor)
            is_border = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] != comp["color"]:
                    is_border = True
                    break
            if is_border:
                result[r][c] = comp["color"]
    return result


# =============================================================================
# Cyclic shift primitives
# =============================================================================

def shift_down_1(grid: Grid) -> Grid:
    """Cyclically shift all rows down by 1 (bottom row wraps to top)."""
    if not grid:
        return grid
    return [grid[-1][:]] + [row[:] for row in grid[:-1]]


def shift_up_1(grid: Grid) -> Grid:
    """Cyclically shift all rows up by 1 (top row wraps to bottom)."""
    if not grid:
        return grid
    return [row[:] for row in grid[1:]] + [grid[0][:]]


def shift_left_1(grid: Grid) -> Grid:
    """Cyclically shift all columns left by 1 (left col wraps to right)."""
    if not grid or not grid[0]:
        return grid
    return [row[1:] + [row[0]] for row in grid]


def shift_right_1(grid: Grid) -> Grid:
    """Cyclically shift all columns right by 1 (right col wraps to left)."""
    if not grid or not grid[0]:
        return grid
    return [[row[-1]] + row[:-1] for row in grid]


# =============================================================================
# Symmetry completion primitives
# =============================================================================

def complete_symmetry_h(grid: Grid) -> Grid:
    """Complete horizontal symmetry: for each row, mirror the non-zero half.

    If the left half has more non-zero pixels, mirror it to the right;
    otherwise mirror the right half to the left.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    mid = w // 2
    for r in range(h):
        left_count = sum(1 for c in range(mid) if result[r][c] != 0)
        right_count = sum(1 for c in range(mid, w) if result[r][c] != 0)
        if left_count >= right_count:
            # Mirror left to right
            for c in range(mid):
                mc = w - 1 - c
                if mc < w and result[r][c] != 0:
                    result[r][mc] = result[r][c]
        else:
            # Mirror right to left
            for c in range(mid, w):
                mc = w - 1 - c
                if mc >= 0 and result[r][c] != 0:
                    result[r][mc] = result[r][c]
    return result


def complete_symmetry_v(grid: Grid) -> Grid:
    """Complete vertical symmetry: for each column, mirror the non-zero half.

    If the top half has more non-zero pixels, mirror it to the bottom;
    otherwise mirror the bottom half to the top.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    mid = h // 2
    for c in range(w):
        top_count = sum(1 for r in range(mid) if result[r][c] != 0)
        bot_count = sum(1 for r in range(mid, h) if result[r][c] != 0)
        if top_count >= bot_count:
            # Mirror top to bottom
            for r in range(mid):
                mr = h - 1 - r
                if mr < h and result[r][c] != 0:
                    result[mr][c] = result[r][c]
        else:
            # Mirror bottom to top
            for r in range(mid, h):
                mr = h - 1 - r
                if mr >= 0 and result[r][c] != 0:
                    result[mr][c] = result[r][c]
    return result


# =============================================================================
# Split-by-separator operations
# =============================================================================

def overlay_split_halves_h(grid: Grid) -> Grid:
    """Split grid horizontally at separator line, overlay top onto bottom.

    Finds the first full horizontal separator line (all same non-zero color),
    splits at it, and overlays the top half onto the bottom half.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    # Find horizontal separator line
    sep_row = -1
    for r in range(h):
        if grid[r][0] != 0 and all(grid[r][c] == grid[r][0] for c in range(w)):
            sep_row = r
            break
    if sep_row <= 0:
        return [row[:] for row in grid]
    top = [grid[r][:] for r in range(sep_row)]
    bottom = [grid[r][:] for r in range(sep_row + 1, h)]
    if not bottom:
        return [row[:] for row in grid]
    return overlay(bottom, top)


def overlay_split_halves_v(grid: Grid) -> Grid:
    """Split grid vertically at separator line, overlay left onto right.

    Finds the first full vertical separator line (all same non-zero color),
    splits at it, and overlays the left half onto the right half.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    # Find vertical separator line
    sep_col = -1
    for c in range(w):
        if grid[0][c] != 0 and all(grid[r][c] == grid[0][c] for r in range(h)):
            sep_col = c
            break
    if sep_col <= 0:
        return [row[:] for row in grid]
    left = [grid[r][:sep_col] for r in range(h)]
    right = [grid[r][sep_col + 1:] for r in range(h)]
    if not right or not right[0]:
        return [row[:] for row in grid]
    return overlay(right, left)


# =============================================================================
# Morphological operations
# =============================================================================

def erode(grid: Grid) -> Grid:
    """Erode: remove pixels that don't have all 4 neighbors of the same color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            color = grid[r][c]
            # Keep only if all 4 neighbors are the same color (or edge)
            keep = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr][nc] != color:
                        keep = False
                        break
            if keep:
                result[r][c] = color
    return result


def spread_colors(grid: Grid) -> Grid:
    """Spread/dilate: each non-zero pixel spreads to its 4-connected bg neighbors."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                        result[nr][nc] = grid[r][c]
    return result


# =============================================================================
# Color cycling primitives
# =============================================================================

def rotate_colors_up(grid: Grid) -> Grid:
    """Rotate non-zero colors up by 1: 1->2->3->...->9->1."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        new_row = []
        for v in row:
            if v == 0:
                new_row.append(0)
            else:
                new_row.append((v % 9) + 1)  # 1->2, 2->3, ..., 9->1
        result.append(new_row)
    return result


def rotate_colors_down(grid: Grid) -> Grid:
    """Rotate non-zero colors down by 1: 1->9, 2->1, 3->2, ..., 9->8."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        new_row = []
        for v in row:
            if v == 0:
                new_row.append(0)
            else:
                new_row.append(((v - 2) % 9) + 1)  # 1->9, 2->1, 3->2, ..., 9->8
        result.append(new_row)
    return result


# --- Composable binary operations (for composing two grids) ---

def overlay(base: Grid, top: Grid) -> Grid:
    """Overlay top grid onto base (non-zero pixels from top win)."""
    base_arr = to_np(base)
    top_arr = to_np(top)
    # Pad to same size if needed
    max_r = max(base_arr.shape[0], top_arr.shape[0])
    max_c = max(base_arr.shape[1], top_arr.shape[1])
    result = np.zeros((max_r, max_c), dtype=np.int32)
    result[:base_arr.shape[0], :base_arr.shape[1]] = base_arr
    tr, tc = top_arr.shape
    mask = top_arr != 0
    result[:tr, :tc][mask] = top_arr[mask]
    return from_np(result)



# =============================================================================
# Ported from agi-mvp-general — fill, pattern, structural, neighbor ops
# =============================================================================

def denoise_5x5(grid: Grid) -> Grid:
    """Replace each cell with 5x5 neighborhood majority."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            counts: dict[int, int] = {}
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = grid[nr][nc]
                        counts[v] = counts.get(v, 0) + 1
            majority = max(counts, key=lambda k: counts[k])
            total = sum(counts.values())
            if counts[majority] > total // 2:
                result[r][c] = majority
    return result


def fill_holes_per_color(grid: Grid) -> Grid:
    """Fill enclosed zero-regions for each color independently."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    colors = set()
    for row in grid:
        for c in row:
            if c != 0:
                colors.add(c)
    for color in colors:
        border_reachable: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    if grid[r][c] != color:
                        stack.append((r, c))
        while stack:
            r, c = stack.pop()
            if (r, c) in border_reachable:
                continue
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if grid[r][c] == color:
                continue
            border_reachable.add((r, c))
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        for r in range(h):
            for c in range(w):
                if grid[r][c] == 0 and (r, c) not in border_reachable:
                    result[r][c] = color
    return result


def fill_holes_in_objects(grid: Grid) -> Grid:
    """Fill enclosed zero-regions inside objects with surrounding color."""
    if not grid or not grid[0]:
        return grid
    from collections import deque
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    result = [row[:] for row in grid]
    reachable = [[False] * w for _ in range(h)]
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr][nc] and grid[nr][nc] == bg:
                reachable[nr][nc] = True
                queue.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not reachable[r][c]:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dist in range(1, max(h, w)):
                        nr, nc = r + dr * dist, c + dc * dist
                        if not (0 <= nr < h and 0 <= nc < w):
                            break
                        if grid[nr][nc] != bg:
                            result[r][c] = grid[nr][nc]
                            break
                    if result[r][c] != bg:
                        break
    return result


def _most_common_overall(grid: Grid) -> int:
    """Find the most common color overall (including bg)."""
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else 0


def fill_tile_pattern(grid: Grid) -> Grid:
    """Infer a repeating tile from visible cells and fill zeros with it."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            if th == h and tw == w:
                continue
            votes: list[list[dict[int, int]]] = [
                [{} for _ in range(tw)] for _ in range(th)
            ]
            for r in range(h):
                for c in range(w):
                    v = grid[r][c]
                    if v != 0:
                        d = votes[r % th][c % tw]
                        d[v] = d.get(v, 0) + 1
            tile = [[0] * tw for _ in range(th)]
            n_resolved = 0
            for tr in range(th):
                for tc in range(tw):
                    if votes[tr][tc]:
                        tile[tr][tc] = max(votes[tr][tc], key=lambda k: votes[tr][tc][k])
                        n_resolved += 1
            if n_resolved < th * tw * 0.5:
                continue
            n_agree = n_nz = 0
            for r in range(h):
                for c in range(w):
                    v = grid[r][c]
                    if v != 0:
                        n_nz += 1
                        if tile[r % th][c % tw] == v:
                            n_agree += 1
            if n_nz > 0 and n_agree / n_nz < 0.90:
                continue
            return [[tile[r % th][c % tw] for c in range(w)] for r in range(h)]
    return [row[:] for row in grid]


def fill_by_symmetry(grid: Grid) -> Grid:
    """Fill masked rectangular region using the grid's symmetry."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    def find_mask_rect(mask_c: int):
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == mask_c]
        if not cells:
            return None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] != mask_c:
                    return None
        return r0, c0, r1, c1

    for mask_c in range(1, 10):
        rect = find_mask_rect(mask_c)
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        for sym_fn in [
            lambda r, c: (h - 1 - r, w - 1 - c),  # 180 rot
            lambda r, c: (h - 1 - r, c),            # V mirror
            lambda r, c: (r, w - 1 - c),            # H mirror
        ]:
            result = [row[:] for row in grid]
            filled = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    sr, sc = sym_fn(r, c)
                    if 0 <= sr < h and 0 <= sc < w and grid[sr][sc] != mask_c:
                        result[r][c] = grid[sr][sc]
                    else:
                        filled = False
            if filled:
                return result
    return [row[:] for row in grid]


def deduplicate_rows(grid: Grid) -> Grid:
    """Remove duplicate consecutive rows."""
    if not grid:
        return []
    result = [grid[0][:]]
    for row in grid[1:]:
        if row != result[-1]:
            result.append(row[:])
    return result


def deduplicate_cols(grid: Grid) -> Grid:
    """Remove duplicate consecutive columns."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    t = _transpose_list(grid)
    deduped = deduplicate_rows(t)
    return _transpose_list(deduped)


def _transpose_list(grid: Grid) -> Grid:
    """Transpose a list-of-lists grid."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(h)] for c in range(w)]


def reverse_rows(grid: Grid) -> Grid:
    """Reverse the order of rows."""
    return [row[:] for row in grid[::-1]]


def reverse_cols(grid: Grid) -> Grid:
    """Reverse each row (mirror horizontally)."""
    return [row[::-1] for row in grid]


def repeat_rows_2x(grid: Grid) -> Grid:
    """Double the grid vertically by repeating rows."""
    return [row[:] for row in grid] + [row[:] for row in grid]


def repeat_cols_2x(grid: Grid) -> Grid:
    """Double the grid horizontally by repeating columns."""
    return [row[:] + row[:] for row in grid]


def stack_with_mirror_v(grid: Grid) -> Grid:
    """Stack grid with its vertical mirror below."""
    return [row[:] for row in grid] + [row[:] for row in reversed(grid)]


def stack_with_mirror_h(grid: Grid) -> Grid:
    """Stack grid with its horizontal mirror to the right."""
    return [row[:] + row[::-1] for row in grid]


def mirror_diagonal_main(grid: Grid) -> Grid:
    """Mirror along main diagonal (transpose for square grids)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h == w:
        return [[grid[c][r] for c in range(h)] for r in range(w)]
    n = max(h, w)
    result = [[0] * n for _ in range(n)]
    for r in range(h):
        for c in range(w):
            result[c][r] = grid[r][c]
    return result


def mirror_diagonal_anti(grid: Grid) -> Grid:
    """Mirror along anti-diagonal."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    n = max(h, w)
    result = [[0] * n for _ in range(n)]
    for r in range(h):
        for c in range(w):
            result[n - 1 - c][n - 1 - r] = grid[r][c]
    return [row[:h] for row in result[:w]]


def grid_difference(grid: Grid) -> Grid:
    """Subtract bottom half from top half (keep unique-to-top)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h < 2:
        return [row[:] for row in grid]
    mid = h // 2
    result = [[0] * w for _ in range(mid)]
    for r in range(mid):
        for c in range(w):
            a, b = grid[r][c], grid[mid + r][c]
            result[r][c] = a if (a != 0 and b == 0) else 0
    return result


def grid_difference_h(grid: Grid) -> Grid:
    """Subtract right half from left half (keep unique-to-left)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if w < 2:
        return [row[:] for row in grid]
    mid = w // 2
    result = [[0] * mid for _ in range(h)]
    for r in range(h):
        for c in range(mid):
            a, b = grid[r][c], grid[r][mid + c]
            result[r][c] = a if (a != 0 and b == 0) else 0
    return result


def and_halves_v(grid: Grid) -> Grid:
    """AND top and bottom halves (half-height output)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h < 2:
        return [row[:] for row in grid]
    mid = h // 2
    result = [[0] * w for _ in range(mid)]
    for r in range(mid):
        for c in range(w):
            a, b = grid[r][c], grid[mid + r][c]
            result[r][c] = a if (a != 0 and b != 0) else 0
    return result


def and_halves_h(grid: Grid) -> Grid:
    """AND left and right halves (half-width output)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if w < 2:
        return [row[:] for row in grid]
    mid = w // 2
    result = [[0] * mid for _ in range(h)]
    for r in range(h):
        for c in range(mid):
            a, b = grid[r][c], grid[r][mid + c]
            result[r][c] = a if (a != 0 and b != 0) else 0
    return result


def keep_only_largest_color(grid: Grid) -> Grid:
    """Keep only the most common non-zero color, zero everything else."""
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return [row[:] for row in grid]
    mc = max(counts, key=lambda k: counts[k])
    return [[v if v == mc else 0 for v in row] for row in grid]


def keep_only_smallest_color(grid: Grid) -> Grid:
    """Keep only the least common non-zero color, zero everything else."""
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return [row[:] for row in grid]
    lc = min(counts, key=lambda k: counts[k])
    return [[v if v == lc else 0 for v in row] for row in grid]


def swap_most_least(grid: Grid) -> Grid:
    """Swap the most and least common non-zero colors."""
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if len(counts) < 2:
        return [row[:] for row in grid]
    mc = max(counts, key=lambda k: counts[k])
    lc = min(counts, key=lambda k: counts[k])
    return [[lc if v == mc else (mc if v == lc else v) for v in row] for row in grid]


def extract_repeating_tile(grid: Grid) -> Grid:
    """Find the smallest tile that tiles to reconstruct the grid."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            if th == h and tw == w:
                continue
            match = True
            for r in range(h):
                if not match:
                    break
                for c in range(w):
                    if grid[r][c] != grid[r % th][c % tw]:
                        match = False
                        break
            if match:
                return [grid[r][:tw] for r in range(th)]
    return [row[:] for row in grid]


def extract_top_left_block(grid: Grid) -> Grid:
    """Extract block above/left of first separator line."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for r in range(1, h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return [row[:] for row in grid[:r]]
    for c in range(1, w):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            return [row[:c] for row in grid]
    return [row[:] for row in grid]


def extract_bottom_right_block(grid: Grid) -> Grid:
    """Extract block below/right of last separator line."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for r in range(h - 1, 0, -1):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return [row[:] for row in grid[r + 1:]] if r + 1 < h else [row[:] for row in grid]
    for c in range(w - 1, 0, -1):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            return [row[c + 1:] for row in grid] if c + 1 < w else [row[:] for row in grid]
    return [row[:] for row in grid]


def extract_unique_block(grid: Grid) -> Grid:
    """Find the sub-block that differs from the others (split by separators or equal division)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    sep_rows = [r for r in range(h) if len(set(grid[r])) == 1 and grid[r][0] != 0]
    if sep_rows:
        boundaries = [-1] + sep_rows + [h]
        blocks = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i] + 1, boundaries[i + 1]
            if start < end:
                blocks.append(tuple(tuple(grid[r]) for r in range(start, end)))
    else:
        blocks = []
        for n in [2, 3, 4]:
            if h % n == 0:
                bh = h // n
                blocks = [tuple(tuple(grid[r]) for r in range(i * bh, (i + 1) * bh)) for i in range(n)]
                break
        if not blocks:
            return [row[:] for row in grid]
    if len(blocks) < 2:
        return [row[:] for row in grid]
    block_counts: dict[tuple, int] = {}
    for b in blocks:
        block_counts[b] = block_counts.get(b, 0) + 1
    if len(block_counts) == 1:
        return [row[:] for row in grid]
    least = min(block_counts, key=lambda k: block_counts[k])
    return [list(row) for row in least]


def compress_rows(grid: Grid) -> Grid:
    """Remove duplicate rows (keep first occurrence of each unique row)."""
    if not grid:
        return []
    seen: list[tuple[int, ...]] = []
    result = []
    for row in grid:
        t = tuple(row)
        if t not in seen:
            seen.append(t)
            result.append(row[:])
    return result


def compress_cols(grid: Grid) -> Grid:
    """Remove duplicate columns."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    t = _transpose_list(grid)
    compressed = compress_rows(t)
    return _transpose_list(compressed)


def max_color_per_cell(grid: Grid) -> Grid:
    """Overlay blocks separated by colored lines, keeping max color per cell."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    sep_rows = [r for r in range(h) if len(set(grid[r])) == 1 and grid[r][0] != 0]
    if not sep_rows:
        return [row[:] for row in grid]
    boundaries = [-1] + sep_rows + [h]
    blocks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i] + 1, boundaries[i + 1]
        if start < end:
            blocks.append([grid[r][:] for r in range(start, end)])
    if len(blocks) < 2:
        return [row[:] for row in grid]
    bh = len(blocks[0])
    bw = len(blocks[0][0]) if blocks[0] else 0
    if any(len(b) != bh for b in blocks):
        return [row[:] for row in grid]
    result = [[0] * bw for _ in range(bh)]
    for block in blocks:
        for r in range(bh):
            for c in range(bw):
                if block[r][c] != 0:
                    result[r][c] = max(result[r][c], block[r][c])
    return result


def min_color_per_cell(grid: Grid) -> Grid:
    """Overlay blocks keeping min non-zero color per cell."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    sep_rows = [r for r in range(h) if len(set(grid[r])) == 1 and grid[r][0] != 0]
    if not sep_rows:
        return [row[:] for row in grid]
    boundaries = [-1] + sep_rows + [h]
    blocks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i] + 1, boundaries[i + 1]
        if start < end:
            blocks.append([grid[r][:] for r in range(start, end)])
    if len(blocks) < 2:
        return [row[:] for row in grid]
    bh = len(blocks[0])
    bw = len(blocks[0][0]) if blocks[0] else 0
    if any(len(b) != bh for b in blocks):
        return [row[:] for row in grid]
    result = [[0] * bw for _ in range(bh)]
    for block in blocks:
        for r in range(bh):
            for c in range(bw):
                v = block[r][c]
                if v != 0:
                    result[r][c] = min(result[r][c], v) if result[r][c] != 0 else v
    return result


def flatten_to_row(grid: Grid) -> Grid:
    """Flatten unique non-zero colors into a single row, sorted."""
    colors = sorted(set(c for row in grid for c in row if c != 0))
    return [colors] if colors else [[0]]


def flatten_to_column(grid: Grid) -> Grid:
    """Flatten unique non-zero colors into a single column, sorted."""
    colors = sorted(set(c for row in grid for c in row if c != 0))
    return [[c] for c in colors] if colors else [[0]]


def mode_color_per_row(grid: Grid) -> Grid:
    """Replace each row with its most common non-zero color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        counts: dict[int, int] = {}
        for c in range(w):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            dom = max(counts, key=lambda k: counts[k])
            result[r] = [dom] * w
    return result


def mode_color_per_col(grid: Grid) -> Grid:
    """Replace each column with its most common non-zero color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        counts: dict[int, int] = {}
        for r in range(h):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            dom = max(counts, key=lambda k: counts[k])
            for r in range(h):
                result[r][c] = dom
    return result


def extend_to_border_h(grid: Grid) -> Grid:
    """Extend each non-zero cell horizontally to fill its row."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        nz = [(c, grid[r][c]) for c in range(w) if grid[r][c] != 0]
        if not nz:
            continue
        if len(set(v for _, v in nz)) == 1:
            result[r] = [nz[0][1]] * w
        else:
            row = [0] * w
            last = 0
            for c in range(w):
                if grid[r][c] != 0:
                    last = grid[r][c]
                row[c] = last
            last = 0
            for c in range(w - 1, -1, -1):
                if row[c] == 0 and last != 0:
                    row[c] = last
                elif grid[r][c] != 0:
                    last = row[c]
            result[r] = row
    return result


def extend_to_border_v(grid: Grid) -> Grid:
    """Extend each non-zero cell vertically to fill its column."""
    return _transpose_list(extend_to_border_h(_transpose_list(grid)))


def spread_in_lanes_h(grid: Grid) -> Grid:
    """Spread non-separator colors horizontally within row lanes."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    sep_rows: set[int] = set()
    sep_cols: set[int] = set()
    color_counts: dict[int, int] = {}
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_rows.add(r)
            color_counts[grid[r][0]] = color_counts.get(grid[r][0], 0) + 1
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != 0:
            sep_cols.add(c)
            color_counts[grid[0][c]] = color_counts.get(grid[0][c], 0) + 1
    if not color_counts:
        return [row[:] for row in grid]
    sep_color = max(color_counts, key=lambda k: color_counts[k])
    result = [row[:] for row in grid]
    for r in range(h):
        if r in sep_rows:
            continue
        row_colors = [grid[r][c] for c in range(w)
                      if c not in sep_cols and grid[r][c] != 0 and grid[r][c] != sep_color]
        if not row_colors:
            continue
        counts: dict[int, int] = {}
        for v in row_colors:
            counts[v] = counts.get(v, 0) + 1
        fill = max(counts, key=lambda k: counts[k])
        for c in range(w):
            if c not in sep_cols and grid[r][c] != sep_color:
                result[r][c] = fill
    return result


def spread_in_lanes_v(grid: Grid) -> Grid:
    """Spread non-separator colors vertically within column lanes."""
    return _transpose_list(spread_in_lanes_h(_transpose_list(grid)))


def complete_pattern_4way(grid: Grid) -> Grid:
    """Complete partial pattern with 4-way (D4) symmetry."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            candidates = [grid[r][c]]
            if 0 <= h - 1 - r < h:
                candidates.append(grid[h - 1 - r][c])
            if 0 <= w - 1 - c < w:
                candidates.append(grid[r][w - 1 - c])
            if 0 <= h - 1 - r < h and 0 <= w - 1 - c < w:
                candidates.append(grid[h - 1 - r][w - 1 - c])
            non_bg = [v for v in candidates if v != bg]
            if non_bg:
                counts: dict[int, int] = {}
                for v in non_bg:
                    counts[v] = counts.get(v, 0) + 1
                val = max(counts, key=lambda k: counts[k])
                result[r][c] = val
                if 0 <= h - 1 - r < h:
                    result[h - 1 - r][c] = val
                if 0 <= w - 1 - c < w:
                    result[r][w - 1 - c] = val
                if 0 <= h - 1 - r < h and 0 <= w - 1 - c < w:
                    result[h - 1 - r][w - 1 - c] = val
    return result


def complete_symmetry_diagonal(grid: Grid) -> Grid:
    """Complete main diagonal symmetry."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    n = min(h, w)
    result = [row[:] for row in grid]
    for r in range(n):
        for c in range(r + 1, n):
            if grid[r][c] != 0 and grid[c][r] == 0:
                result[c][r] = grid[r][c]
            elif grid[c][r] != 0 and grid[r][c] == 0:
                result[r][c] = grid[c][r]
    return result


def mirror_h_merge(grid: Grid) -> Grid:
    """Mirror horizontally and overlay with OR logic."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                result[r][c] = grid[r][w - 1 - c]
    return result


def mirror_v_merge(grid: Grid) -> Grid:
    """Mirror vertically and overlay with OR logic."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                result[r][c] = grid[h - 1 - r][c]
    return result


def sort_rows_by_value(grid: Grid) -> Grid:
    """Sort values in each row ascending."""
    return [sorted(row) for row in grid]


def sort_cols_by_value(grid: Grid) -> Grid:
    """Sort values in each column ascending."""
    return _transpose_list(sort_rows_by_value(_transpose_list(grid)))


def sort_rows_by_sum(grid: Grid) -> Grid:
    """Sort rows by the sum of their values."""
    rows = [row[:] for row in grid]
    rows.sort(key=sum)
    return rows


def sort_cols_by_sum(grid: Grid) -> Grid:
    """Sort columns by the sum of their values."""
    return _transpose_list(sort_rows_by_sum(_transpose_list(grid)))


def fill_row_from_right(grid: Grid) -> Grid:
    """Propagate non-zero values rightward in each row."""
    if not grid or not grid[0]:
        return grid
    result = [row[:] for row in grid]
    for r in range(len(result)):
        last = 0
        for c in range(len(result[r])):
            if result[r][c] != 0:
                last = result[r][c]
            elif last != 0:
                result[r][c] = last
    return result


def fill_col_from_bottom(grid: Grid) -> Grid:
    """Propagate non-zero values upward in each column."""
    return _transpose_list(fill_row_from_right(_transpose_list(grid)))


def propagate_color_h(grid: Grid) -> Grid:
    """Extend non-zero colors rightward until hitting another non-zero cell."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        last_color = 0
        for c in range(w):
            if grid[r][c] != 0:
                last_color = grid[r][c]
            elif last_color != 0:
                result[r][c] = last_color
    return result


def propagate_color_v(grid: Grid) -> Grid:
    """Extend non-zero colors downward until hitting another non-zero cell."""
    return _transpose_list(propagate_color_h(_transpose_list(grid)))


def fill_stripe_gaps_h(grid: Grid) -> Grid:
    """Fill bg cells between same-color cells within each row."""
    if not grid or not grid[0]:
        return grid
    result = [row[:] for row in grid]
    for r in range(len(result)):
        row = result[r]
        w = len(row)
        for _ in range(w):
            changed = False
            for c in range(1, w - 1):
                if row[c] == 0:
                    if row[c - 1] != 0 and row[c - 1] == row[c + 1]:
                        row[c] = row[c - 1]
                        changed = True
            if not changed:
                break
    return result


def fill_stripe_gaps_v(grid: Grid) -> Grid:
    """Fill bg cells between same-color cells within each column."""
    return _transpose_list(fill_stripe_gaps_h(_transpose_list(grid)))


def complete_tile_from_modal_col(grid: Grid) -> Grid:
    """For each column, fill anomalous cells with the modal value."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        counts: dict[int, int] = {}
        for r in range(h):
            v = grid[r][c]
            counts[v] = counts.get(v, 0) + 1
        if counts:
            mode = max(counts, key=lambda k: counts[k])
            for r in range(h):
                result[r][c] = mode
    return result


def complete_tile_from_modal_row(grid: Grid) -> Grid:
    """For each row, fill anomalous cells with the modal value."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        counts: dict[int, int] = {}
        for v in row:
            counts[v] = counts.get(v, 0) + 1
        mode = max(counts, key=lambda k: counts[k])
        result.append([mode] * len(row))
    return result


def recolor_minority_in_rows(grid: Grid) -> Grid:
    """In each row, recolor single-occurrence cells to row's dominant color."""
    if not grid or not grid[0]:
        return grid
    result = []
    for row in grid:
        counts: dict[int, int] = {}
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            result.append(row[:])
            continue
        dom = max(counts, key=lambda k: counts[k])
        new_row = []
        for v in row:
            if v != 0 and counts.get(v, 0) == 1 and v != dom:
                new_row.append(dom)
            else:
                new_row.append(v)
        result.append(new_row)
    return result


def recolor_minority_in_cols(grid: Grid) -> Grid:
    """In each column, recolor single-occurrence cells to column's dominant color."""
    return _transpose_list(recolor_minority_in_rows(_transpose_list(grid)))


def remove_color_noise(grid: Grid) -> Grid:
    """Remove isolated single pixels (no same-color 4-way neighbors)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == grid[r][c]:
                    has_neighbor = True
                    break
            if not has_neighbor:
                result[r][c] = 0
    return result


def recolor_isolated_to_nearest(grid: Grid) -> Grid:
    """Recolor isolated pixels to nearest non-bg color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or grid[r][c] == 0:
                continue
            has_same = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == grid[r][c]:
                    has_same = True
                    break
            if has_same:
                continue
            best_d = h + w
            best_c = grid[r][c]
            for r2 in range(h):
                for c2 in range(w):
                    if (r2, c2) != (r, c) and grid[r2][c2] != bg and grid[r2][c2] != 0:
                        d = abs(r2 - r) + abs(c2 - c)
                        if d < best_d:
                            best_d = d
                            best_c = grid[r2][c2]
            result[r][c] = best_c
    return result


def fill_enclosed_wall_color(grid: Grid) -> Grid:
    """Fill enclosed bg regions with the most common adjacent wall color."""
    if not grid or not grid[0]:
        return grid
    from collections import deque
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    reachable = [[False] * w for _ in range(h)]
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr][nc] and grid[nr][nc] == bg:
                reachable[nr][nc] = True
                queue.append((nr, nc))
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not reachable[r][c]:
                adj_colors: dict[int, int] = {}
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != bg:
                        adj_colors[grid[nr][nc]] = adj_colors.get(grid[nr][nc], 0) + 1
                if adj_colors:
                    result[r][c] = max(adj_colors, key=lambda k: adj_colors[k])
    return result


def remove_border_objects(grid: Grid) -> Grid:
    """Remove all objects that touch the grid border."""
    if not grid or not grid[0]:
        return grid
    from collections import deque
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    to_remove: set[tuple[int, int]] = set()
    visited = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                touches_border = False
                queue: deque[tuple[int, int]] = deque([(r, c)])
                visited[r][c] = True
                color = grid[r][c]
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                        touches_border = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if touches_border:
                    to_remove.update(comp)
    result = [row[:] for row in grid]
    for r, c in to_remove:
        result[r][c] = bg
    return result


def keep_interior_objects(grid: Grid) -> Grid:
    """Keep only objects that don't touch the grid border."""
    if not grid or not grid[0]:
        return grid
    from collections import deque
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    to_keep: set[tuple[int, int]] = set()
    visited = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                touches_border = False
                queue: deque[tuple[int, int]] = deque([(r, c)])
                visited[r][c] = True
                color = grid[r][c]
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                        touches_border = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if not touches_border:
                    to_keep.update(comp)
    result = [[bg] * w for _ in range(h)]
    for r, c in to_keep:
        result[r][c] = grid[r][c]
    return result


def fill_object_bboxes(grid: Grid) -> Grid:
    """Fill the bounding box of each object with its color."""
    if not grid or not grid[0]:
        return grid
    from collections import deque
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    result = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                queue: deque[tuple[int, int]] = deque([(r, c)])
                visited[r][c] = True
                cells: list[tuple[int, int]] = []
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                for br in range(min(rs), max(rs) + 1):
                    for bc in range(min(cs), max(cs) + 1):
                        result[br][bc] = color
    return result


def crop_to_content_border(grid: Grid) -> Grid:
    """Crop to bounding box of non-bg cells, add 1-cell bg border."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    r0, r1, c0, c1 = h, 0, w, 0
    found = False
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                found = True
                r0 = min(r0, r)
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)
    if not found:
        return [[bg]]
    cropped = [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
    cw = c1 - c0 + 1
    result = [[bg] * (cw + 2)]
    for row in cropped:
        result.append([bg] + row[:] + [bg])
    result.append([bg] * (cw + 2))
    return result


def recolor_by_nearest_border(grid: Grid) -> Grid:
    """Recolor isolated pixels using nearest border stripe color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    border_rows: dict[int, int] = {}
    border_cols: dict[int, int] = {}
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            border_rows[r] = grid[r][0]
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != 0:
            border_cols[c] = grid[0][c]
    if not border_rows and not border_cols:
        return [row[:] for row in grid]
    border_colors = set(border_rows.values()) | set(border_cols.values())
    interior_counts: dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0 and v not in border_colors:
                interior_counts[v] = interior_counts.get(v, 0) + 1
    if not interior_counts:
        return [row[:] for row in grid]
    noise = min(interior_counts, key=lambda k: interior_counts[k])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == noise:
                best_d = h + w
                best_c = noise
                for br, bc in border_rows.items():
                    if abs(r - br) < best_d:
                        best_d = abs(r - br)
                        best_c = bc
                for ci, cc in border_cols.items():
                    if abs(c - ci) < best_d:
                        best_d = abs(c - ci)
                        best_c = cc
                result[r][c] = best_c
    return result


def _make_swap_colors(a: int, b: int):
    """Factory: create a function that swaps two colors."""
    def swap(grid: Grid) -> Grid:
        return [[b if v == a else (a if v == b else v) for v in row] for row in grid]
    return swap


def _make_fill_bg(color: int):
    """Factory: fill background (0) with a specific color."""
    def fill(grid: Grid) -> Grid:
        return [[color if v == 0 else v for v in row] for row in grid]
    return fill


def _make_erase_color(color: int):
    """Factory: replace a specific color with background (0)."""
    def erase(grid: Grid) -> Grid:
        return [[0 if v == color else v for v in row] for row in grid]
    return erase


def _make_recolor_nonzero(to_color: int):
    """Factory: recolor all non-zero cells to a specific color."""
    def recolor(grid: Grid) -> Grid:
        return [[to_color if v != 0 else 0 for v in row] for row in grid]
    return recolor


def project_markers_to_block(grid: Grid) -> Grid:
    """Draw lines from block edges to isolated marker cells."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    color_counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    bg_color = max(color_counts, key=lambda k: color_counts[k]) if color_counts else 0
    # Find largest connected component
    visited: set[tuple[int, int]] = set()
    block_cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or grid[r][c] == bg_color:
                continue
            comp: set[tuple[int, int]] = set()
            stack = [(r, c)]
            while stack:
                tr, tc = stack.pop()
                if (tr, tc) in comp or tr < 0 or tr >= h or tc < 0 or tc >= w:
                    continue
                if grid[tr][tc] != grid[r][c]:
                    continue
                comp.add((tr, tc))
                stack.extend([(tr+1, tc), (tr-1, tc), (tr, tc+1), (tr, tc-1)])
            visited.update(comp)
            if len(comp) > len(block_cells):
                block_cells = comp
    if not block_cells:
        return result
    rows = [r for r, c in block_cells]
    cols = [c for r, c in block_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    # Find isolated markers
    markers = []
    visited2: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in visited2 or (r, c) in block_cells or grid[r][c] == bg_color:
                continue
            comp2: set[tuple[int, int]] = set()
            stack2 = [(r, c)]
            while stack2:
                tr, tc = stack2.pop()
                if (tr, tc) in comp2 or tr < 0 or tr >= h or tc < 0 or tc >= w:
                    continue
                if grid[tr][tc] != grid[r][c]:
                    continue
                comp2.add((tr, tc))
                stack2.extend([(tr+1, tc), (tr-1, tc), (tr, tc+1), (tr, tc-1)])
            visited2.update(comp2)
            if len(comp2) <= 2:
                for mr, mc in comp2:
                    markers.append((grid[r][c], mr, mc))
    for mc, mr, mcc in markers:
        if mr < min_r and min_c <= mcc <= max_c:
            for r in range(mr, min_r):
                result[r][mcc] = mc
        elif mr > max_r and min_c <= mcc <= max_c:
            for r in range(max_r + 1, mr + 1):
                result[r][mcc] = mc
        elif mcc < min_c and min_r <= mr <= max_r:
            for c in range(mcc, min_c):
                result[mr][c] = mc
        elif mcc > max_c and min_r <= mr <= max_r:
            for c in range(max_c + 1, mcc + 1):
                result[mr][c] = mc
    return result


def fill_bg_from_border(grid: Grid) -> Grid:
    """Fill all bg cells with the most common non-bg border color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    border_vals = []
    for c in range(w):
        border_vals.extend([grid[0][c], grid[h - 1][c]])
    for r in range(1, h - 1):
        border_vals.extend([grid[r][0], grid[r][w - 1]])
    non_bg = [v for v in border_vals if v != bg]
    if not non_bg:
        return [row[:] for row in grid]
    counts: dict[int, int] = {}
    for v in non_bg:
        counts[v] = counts.get(v, 0) + 1
    fill = max(counts, key=lambda k: counts[k])
    return [[fill if v == bg else v for v in row] for row in grid]


def fill_grid_intersections(grid: Grid) -> Grid:
    """Fill bg cells at row/col intersections with matching colors."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                continue
            row_colors = set(grid[r][cc] for cc in range(w) if grid[r][cc] != 0)
            col_colors = set(grid[rr][c] for rr in range(h) if grid[rr][c] != 0)
            common = row_colors & col_colors
            if common:
                result[r][c] = min(common)
    return result


def tile_grid_2x1(grid: Grid) -> Grid:
    """Tile grid twice horizontally."""
    return [row[:] + row[:] for row in grid]


def tile_grid_1x2(grid: Grid) -> Grid:
    """Tile grid twice vertically."""
    return [row[:] for row in grid] + [row[:] for row in grid]


def mask_by_color_overlap(grid: Grid) -> Grid:
    """Keep only cells matching the most common non-bg color."""
    mc = 0
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if counts:
        mc = max(counts, key=lambda k: counts[k])
    return [[v if v == mc else 0 for v in row] for row in grid]


def fill_diagonal_stripes(grid: Grid) -> Grid:
    """Fill bg cells with diagonal stripe pattern using non-bg colors."""
    if not grid or not grid[0]:
        return grid
    colors = sorted(set(v for row in grid for v in row if v != 0))
    if not colors:
        return [row[:] for row in grid]
    result = [row[:] for row in grid]
    n = len(colors)
    for r in range(len(result)):
        for c in range(len(result[r])):
            if result[r][c] == 0:
                result[r][c] = colors[(r + c) % n]
    return result


def keep_border_only(grid: Grid) -> Grid:
    """Keep only the outermost ring of cells."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    bg = _most_common_overall(grid)
    result = [row[:] for row in grid]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            result[r][c] = bg
    return result


# =============================================================================
# Port batch 2: remaining primitives from agi-mvp-general
# =============================================================================

def connect_pixels_to_rect(grid: Grid) -> Grid:
    """Connect isolated single-pixel anomalies to the nearest rectangle border.

    Finds isolated non-bg pixels (surrounded entirely by bg), then draws a
    straight line (H or V) from that pixel to the nearest edge of any
    rectangle object, filling with the isolated pixel's color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]

    visited = [[False] * w for _ in range(h)]
    components = []

    def bfs(sr, sc):
        color = grid[sr][sc]
        cells = []
        queue = [(sr, sc)]
        visited[sr][sc] = True
        while queue:
            r, c = queue.pop(0)
            cells.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return color, cells

    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                color, cells = bfs(r, c)
                components.append((color, cells))

    if len(components) < 2:
        return grid

    isolated = []
    rects = []
    for color, cells in components:
        if len(cells) == 1:
            r, c = cells[0]
            neighbors = [grid[r + dr][c + dc] for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if 0 <= r + dr < h and 0 <= c + dc < w]
            if all(v == bg for v in neighbors):
                isolated.append((color, r, c))
            else:
                rects.append((color, cells))
        else:
            rects.append((color, cells))

    if not isolated or not rects:
        return grid

    rect_cells_set = set()
    for _, cells in rects:
        for rc in cells:
            rect_cells_set.add(rc)

    for iso_color, ir, ic in isolated:
        best_dist = float('inf')
        best_rc = None
        for rr, rc in rect_cells_set:
            if rr == ir or rc == ic:
                d = abs(rr - ir) + abs(rc - ic)
                if d < best_dist:
                    best_dist = d
                    best_rc = (rr, rc)
        if best_rc is None:
            for rr, rc in rect_cells_set:
                d = abs(rr - ir) + abs(rc - ic)
                if d < best_dist:
                    best_dist = d
                    best_rc = (rr, rc)
        if best_rc is None:
            continue
        rr, rc = best_rc
        if rr == ir:
            c_start, c_end = min(ic, rc), max(ic, rc)
            for c in range(c_start, c_end + 1):
                if result[ir][c] == bg:
                    result[ir][c] = iso_color
        elif rc == ic:
            r_start, r_end = min(ir, rr), max(ir, rr)
            for r in range(r_start, r_end + 1):
                if result[r][ic] == bg:
                    result[r][ic] = iso_color
        else:
            if abs(ic - rc) <= abs(ir - rr):
                c_start, c_end = min(ic, rc), max(ic, rc)
                for c in range(c_start, c_end + 1):
                    if result[ir][c] == bg:
                        result[ir][c] = iso_color
            else:
                r_start, r_end = min(ir, rr), max(ir, rr)
                for r in range(r_start, r_end + 1):
                    if result[r][ic] == bg:
                        result[r][ic] = iso_color
    return result


def gravity_toward_color(grid: Grid) -> Grid:
    """Pull all non-bg cells toward rows/cols containing solid bands."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    band_rows = []
    for r in range(h):
        row_vals = set(grid[r])
        if len(row_vals) == 1 and list(row_vals)[0] != bg:
            band_rows.append((r, list(row_vals)[0]))

    band_cols = []
    for c in range(w):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and list(col_vals)[0] != bg:
            band_cols.append((c, list(col_vals)[0]))

    if not band_rows and not band_cols:
        return grid

    result = [row[:] for row in grid]
    band_row_set = {r for r, _ in band_rows}
    band_col_set = {c for c, _ in band_cols}

    if band_rows:
        for c in range(w):
            vals_above, vals_below = [], []
            first_band = band_rows[0][0]
            last_band = band_rows[-1][0]
            for r in range(h):
                if r in band_row_set:
                    continue
                if grid[r][c] != bg:
                    if r < first_band:
                        vals_above.append(grid[r][c])
                    else:
                        vals_below.append(grid[r][c])
            for r in range(h):
                if r not in band_row_set:
                    result[r][c] = bg
            for i, val in enumerate(reversed(vals_above)):
                r = first_band - 1 - i
                if 0 <= r < h and r not in band_row_set:
                    result[r][c] = val
            for i, val in enumerate(vals_below):
                r = last_band + 1 + i
                if 0 <= r < h and r not in band_row_set:
                    result[r][c] = val

    if band_cols:
        for r in range(h):
            vals_left, vals_right = [], []
            first_band = band_cols[0][0]
            last_band = band_cols[-1][0]
            for c in range(w):
                if c in band_col_set:
                    continue
                if grid[r][c] != bg:
                    if c < first_band:
                        vals_left.append(grid[r][c])
                    else:
                        vals_right.append(grid[r][c])
            for c in range(w):
                if c not in band_col_set:
                    result[r][c] = bg
            for i, val in enumerate(reversed(vals_left)):
                c = first_band - 1 - i
                if 0 <= c < w and c not in band_col_set:
                    result[r][c] = val
            for i, val in enumerate(vals_right):
                c = last_band + 1 + i
                if 0 <= c < w and c not in band_col_set:
                    result[r][c] = val
    return result


def recolor_2nd_to_3rd(grid: Grid) -> Grid:
    """Replace the 2nd most common non-bg color with the 3rd most common."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 3:
        return grid
    src, dst = non_bg[1], non_bg[2]
    return [[dst if v == src else v for v in row] for row in grid]


def recolor_least_to_2nd_least(grid: Grid) -> Grid:
    """Replace the least common non-bg color with the 2nd least common."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg_sorted = [(v, c) for v, c in sorted(counts.items(), key=lambda x: x[1]) if v != bg]
    if len(non_bg_sorted) < 2:
        return grid
    src, dst = non_bg_sorted[0][0], non_bg_sorted[1][0]
    return [[dst if v == src else v for v in row] for row in grid]


def swap_most_and_2nd_color(grid: Grid) -> Grid:
    """Swap the most common and 2nd most common non-bg colors."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    c1, c2 = non_bg[0], non_bg[1]
    return [[c2 if v == c1 else (c1 if v == c2 else v) for v in row] for row in grid]


def keep_unique_rows(grid: Grid) -> Grid:
    """Remove duplicate rows, keeping only the first occurrence."""
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
    """Remove duplicate columns, keeping only the first occurrence."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    seen = []
    keep_cols = []
    for c in range(w):
        key = tuple(grid[r][c] for r in range(h))
        if key not in seen:
            seen.append(key)
            keep_cols.append(c)
    if not keep_cols:
        return grid
    return [[grid[r][c] for c in keep_cols] for r in range(h)]


def repeat_pattern_to_size(grid: Grid) -> Grid:
    """Find smallest repeating sub-pattern and tile it to fill original size."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    for th in range(1, h // 2 + 1):
        if h % th != 0:
            continue
        for tw in range(1, w // 2 + 1):
            if w % tw != 0:
                continue
            tile = [grid[r][:tw] for r in range(th)]
            valid = True
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != tile[r % th][c % tw]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and (th < h or tw < w):
                return [[tile[r % th][c % tw] for c in range(w)] for r in range(h)]
    return grid


def extend_lines_to_contact(grid: Grid) -> Grid:
    """Extend non-bg colored segments to fill gaps within their row or column."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    # Extend horizontally
    for r in range(h):
        non_bg_cols = [(c, grid[r][c]) for c in range(w) if grid[r][c] != bg]
        if len(non_bg_cols) >= 2:
            min_c, fill_color = non_bg_cols[0]
            max_c = non_bg_cols[-1][0]
            if all(grid[r][c] in (bg, fill_color) for c in range(min_c, max_c + 1)):
                for c in range(min_c, max_c + 1):
                    if result[r][c] == bg:
                        result[r][c] = fill_color
    # Extend vertically
    for c in range(w):
        non_bg_rows = [(r, grid[r][c]) for r in range(h) if grid[r][c] != bg]
        if len(non_bg_rows) >= 2:
            min_r, fill_color = non_bg_rows[0]
            max_r = non_bg_rows[-1][0]
            if all(grid[r][c] in (bg, fill_color) for r in range(min_r, max_r + 1)):
                for r in range(min_r, max_r + 1):
                    if result[r][c] == bg:
                        result[r][c] = fill_color
    return result


def recolor_2nd_to_dominant(grid: Grid) -> Grid:
    """Recolor the 2nd most common non-bg color to the dominant non-bg color."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    dominant, accent = non_bg[0], non_bg[1]
    return [[dominant if v == accent else v for v in row] for row in grid]


def erase_2nd_color(grid: Grid) -> Grid:
    """Erase (set to bg) the 2nd most common non-bg color."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    accent = non_bg[1]
    return [[bg if v == accent else v for v in row] for row in grid]


def recolor_bg_enclosed_by_dominant(grid: Grid) -> Grid:
    """Fill enclosed bg regions with the dominant non-bg color."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if not non_bg:
        return grid
    fill_color = non_bg[0]

    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr, nc))
                queue.append((nr, nc))

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r, c) not in reachable:
                result[r][c] = fill_color
    return result


def _fill_rect_interiors(grid: Grid, fill_color: int) -> Grid:
    """Fill interior of rectangular frames (enclosed bg not reachable from border)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr, nc))
                queue.append((nr, nc))

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r, c) not in reachable:
                result[r][c] = fill_color
    return result


def _recolor_cells_at_intersections(grid: Grid, new_color: int) -> Grid:
    """Recolor bg cells at row/col intersections of non-bg content."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    active_rows = {r for r in range(h) if any(grid[r][c] != bg for c in range(w))}
    active_cols = {c for c in range(w) if any(grid[r][c] != bg for r in range(h))}

    result = [row[:] for row in grid]
    for r in active_rows:
        for c in active_cols:
            if grid[r][c] == bg:
                result[r][c] = new_color
    return result


def _make_recolor_dominant_touching_accent(new_color: int):
    """Factory: recolor dominant-color cells adjacent to accent (2nd) color."""
    def _fn(grid):
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        from collections import Counter
        flat = [v for row in grid for v in row]
        bg = Counter(flat).most_common(1)[0][0]
        non_bg = [v for v, _ in Counter(flat).most_common() if v != bg]
        if len(non_bg) < 2:
            return grid
        dominant, accent = non_bg[0], non_bg[1]
        result = [row[:] for row in grid]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != dominant:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == accent:
                        result[r][c] = new_color
                        break
        return result
    return _fn


def _make_fill_smallest_hole(new_color: int):
    """Factory: fill the smallest enclosed bg region with new_color."""
    def _fn(grid):
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        from collections import Counter
        flat = [v for row in grid for v in row]
        bg = Counter(flat).most_common(1)[0][0]

        reachable = set()
        queue = []
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                    if (r, c) not in reachable:
                        reachable.add((r, c))
                        queue.append((r, c))
        while queue:
            r, c = queue.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

        visited = set()
        holes = []
        for sr in range(h):
            for sc in range(w):
                if grid[sr][sc] == bg and (sr, sc) not in reachable and (sr, sc) not in visited:
                    hole = []
                    q = [(sr, sc)]
                    visited.add((sr, sc))
                    while q:
                        r, c = q.pop()
                        hole.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and grid[nr][nc] == bg:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                    holes.append(hole)

        if not holes:
            return grid
        smallest = min(holes, key=len)
        result = [row[:] for row in grid]
        for r, c in smallest:
            result[r][c] = new_color
        return result
    return _fn


def _make_recolor_nonzero_inside_bbox(accent_color: int, new_color: int):
    """Factory: recolor non-zero non-accent cells inside accent_color's bounding box."""
    def _fn(grid):
        if not grid or not grid[0]:
            return grid
        h, w = len(grid), len(grid[0])
        accent_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == accent_color]
        if not accent_cells:
            return grid
        min_r = min(r for r, c in accent_cells)
        max_r = max(r for r, c in accent_cells)
        min_c = min(c for r, c in accent_cells)
        max_c = max(c for r, c in accent_cells)
        result = [row[:] for row in grid]
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                v = grid[r][c]
                if v != 0 and v != accent_color:
                    result[r][c] = new_color
        return result
    return _fn


# =============================================================================
# Build the ARC primitive registry
# =============================================================================

def _build_arc_primitives() -> list[Primitive]:
    """Build the complete list of ARC primitives."""
    prims = []

    # Arity-1 primitives (Grid -> Grid): the core set
    unary_ops = [
        ("identity",        identity),
        ("rot90cw",         rotate_90_cw),
        ("rot90ccw",        rotate_90_ccw),
        ("rot180",          rotate_180),
        ("mirror_h",        mirror_horizontal),
        ("mirror_v",        mirror_vertical),
        ("transpose",       transpose),
        ("invert_colors",   invert_colors),
        ("crop_nonzero",    crop_to_nonzero),
        ("top_half",        get_top_half),
        ("bottom_half",     get_bottom_half),
        ("left_half",       get_left_half),
        ("right_half",      get_right_half),
        ("tile_2x2",        tile_2x2),
        ("tile_3x3",        tile_3x3),
        ("scale_2x",        scale_2x),
        ("scale_3x",        scale_3x),
        ("gravity_down",    gravity_down),
        ("gravity_up",      gravity_up),
        ("gravity_left",    gravity_left),
        ("gravity_right",   gravity_right),
        ("outline",         outline),
        ("fill_enclosed",   fill_enclosed),
        ("denoise_3x3",     denoise_3x3),
        ("xor_halves_v",    xor_halves_v),
        ("or_halves_v",     or_halves_v),
        ("xor_halves_h",    xor_halves_h),
        ("or_halves_h",     or_halves_h),
        ("replace_bg_mc",   replace_bg_with_most_common),
        # --- New spatial/object primitives ---
        ("extract_largest",     extract_largest_object),
        ("extract_smallest",    extract_smallest_object),
        ("anti_diag_mirror",    anti_diagonal_mirror),
        ("make_sym_h",          make_symmetric_h),
        ("make_sym_v",          make_symmetric_v),
        ("repeat_right",        repeat_pattern_right),
        ("repeat_down",         repeat_pattern_down),
        ("add_border",          add_border),
        ("remove_border",       remove_border),
        ("sort_rows",           sort_rows_by_color_count),
        ("sort_cols",           sort_cols_by_color_count),
        ("unique_rows",         unique_rows),
        ("unique_cols",         unique_cols),
        ("recolor_by_rank",     recolor_by_size_rank),
        ("extend_lines_h",      extend_lines_h),
        ("extend_lines_v",      extend_lines_v),
        # --- Connected component / object-level primitives ---
        ("keep_largest_only",   keep_largest_object_only),
        ("keep_smallest_only",  keep_smallest_object_only),
        ("remove_largest_obj",  remove_largest_object),
        ("remove_smallest_obj", remove_smallest_object),
        ("count_objects",       count_objects_as_grid),
        ("recolor_each_obj",    recolor_each_object),
        ("mirror_objects_h",    mirror_objects_h),
        ("mirror_objects_v",    mirror_objects_v),
        ("flood_fill_bg",       flood_fill_bg),
        # --- Grid partitioning ---
        ("extract_tl_cell",     extract_top_left_cell),
        ("extract_br_cell",     extract_bottom_right_cell),
        ("remove_grid_lines",   remove_grid_lines),
        # --- Diagonal / line extension ---
        ("shift_rows_right",    shift_rows_right),
        ("shift_rows_left",     shift_rows_left),
        ("extend_lines",        extend_lines),
        ("extend_diagonals",    extend_diagonal_lines),
        # --- Color/pattern ops ---
        ("binarize",            binarize),
        ("color_to_mc",         color_to_most_common),
        ("upscale_pattern",     upscale_pattern),
        # --- Near-miss targeted: anomaly removal, rectangle ops ---
        ("denoise_majority",    denoise_majority),
        ("fill_rectangles",     fill_rectangles),
        ("extract_minority_c",  extract_minority_color),
        ("extract_majority_c",  extract_majority_color),
        ("replace_noise_objs",  replace_noise_in_objects),
        ("hollow_objects",      hollow_objects),
        # --- Cyclic shifts ---
        ("shift_down_1",        shift_down_1),
        ("shift_up_1",          shift_up_1),
        ("shift_left_1",        shift_left_1),
        ("shift_right_1",       shift_right_1),
        # --- Symmetry completion ---
        ("complete_sym_h",      complete_symmetry_h),
        ("complete_sym_v",      complete_symmetry_v),
        # --- Split-by-separator ---
        ("overlay_split_h",     overlay_split_halves_h),
        ("overlay_split_v",     overlay_split_halves_v),
        # --- Morphological ops ---
        ("erode",               erode),
        ("spread_colors",       spread_colors),
        # --- Color cycling ---
        ("rotate_colors_up",    rotate_colors_up),
        ("rotate_colors_down",  rotate_colors_down),
        # --- Ported from agi-mvp-general ---
        ("denoise_5x5",             denoise_5x5),
        ("fill_holes_per_color",    fill_holes_per_color),
        ("fill_holes_in_objects",   fill_holes_in_objects),
        ("fill_tile_pattern",       fill_tile_pattern),
        ("fill_by_symmetry",        fill_by_symmetry),
        ("deduplicate_rows",        deduplicate_rows),
        ("deduplicate_cols",        deduplicate_cols),
        ("reverse_rows",            reverse_rows),
        ("reverse_cols",            reverse_cols),
        ("repeat_rows_2x",          repeat_rows_2x),
        ("repeat_cols_2x",          repeat_cols_2x),
        ("stack_mirror_v",          stack_with_mirror_v),
        ("stack_mirror_h",          stack_with_mirror_h),
        ("mirror_diag_main",        mirror_diagonal_main),
        ("mirror_diag_anti",        mirror_diagonal_anti),
        ("grid_difference",         grid_difference),
        ("grid_difference_h",       grid_difference_h),
        ("and_halves_v",            and_halves_v),
        ("and_halves_h",            and_halves_h),
        ("keep_largest_color",      keep_only_largest_color),
        ("keep_smallest_color",     keep_only_smallest_color),
        ("swap_most_least",         swap_most_least),
        ("extract_rep_tile",        extract_repeating_tile),
        ("extract_tl_block",        extract_top_left_block),
        ("extract_br_block",        extract_bottom_right_block),
        ("extract_unique_block",    extract_unique_block),
        ("compress_rows",           compress_rows),
        ("compress_cols",           compress_cols),
        ("max_color_per_cell",      max_color_per_cell),
        ("min_color_per_cell",      min_color_per_cell),
        ("flatten_to_row",          flatten_to_row),
        ("flatten_to_col",          flatten_to_column),
        ("mode_per_row",            mode_color_per_row),
        ("mode_per_col",            mode_color_per_col),
        ("extend_border_h",         extend_to_border_h),
        ("extend_border_v",         extend_to_border_v),
        ("spread_lanes_h",          spread_in_lanes_h),
        ("spread_lanes_v",          spread_in_lanes_v),
        ("complete_4way",           complete_pattern_4way),
        ("complete_diag",           complete_symmetry_diagonal),
        ("mirror_h_merge",          mirror_h_merge),
        ("mirror_v_merge",          mirror_v_merge),
        ("sort_rows_val",           sort_rows_by_value),
        ("sort_cols_val",           sort_cols_by_value),
        ("sort_rows_sum",           sort_rows_by_sum),
        ("sort_cols_sum",           sort_cols_by_sum),
        ("fill_row_right",          fill_row_from_right),
        ("fill_col_bottom",         fill_col_from_bottom),
        ("propagate_h",             propagate_color_h),
        ("propagate_v",             propagate_color_v),
        ("fill_stripe_gaps_h",      fill_stripe_gaps_h),
        ("fill_stripe_gaps_v",      fill_stripe_gaps_v),
        ("tile_modal_col",          complete_tile_from_modal_col),
        ("tile_modal_row",          complete_tile_from_modal_row),
        ("recolor_minor_rows",      recolor_minority_in_rows),
        ("recolor_minor_cols",      recolor_minority_in_cols),
        ("remove_noise",            remove_color_noise),
        ("recolor_isolated",        recolor_isolated_to_nearest),
        ("fill_enclosed_wall",      fill_enclosed_wall_color),
        ("remove_border_objs",      remove_border_objects),
        ("keep_interior_objs",      keep_interior_objects),
        ("fill_obj_bboxes",         fill_object_bboxes),
        ("crop_content_border",     crop_to_content_border),
        ("recolor_nearest_border",  recolor_by_nearest_border),
        ("project_markers",         project_markers_to_block),
        ("fill_bg_border",          fill_bg_from_border),
        ("fill_grid_inters",        fill_grid_intersections),
        ("tile_2x1",                tile_grid_2x1),
        ("tile_1x2",                tile_grid_1x2),
        ("mask_color_overlap",      mask_by_color_overlap),
        ("fill_diag_stripes",       fill_diagonal_stripes),
        ("keep_border",             keep_border_only),
        # --- Port batch 2: agi-mvp-general ---
        ("connect_to_rect",         connect_pixels_to_rect),
        ("gravity_toward_color",    gravity_toward_color),
        ("recolor_2nd_3rd",         recolor_2nd_to_3rd),
        ("recolor_least_2nd",       recolor_least_to_2nd_least),
        ("swap_most_2nd",           swap_most_and_2nd_color),
        ("keep_unique_rows",        keep_unique_rows),
        ("keep_unique_cols",        keep_unique_cols),
        ("repeat_to_size",          repeat_pattern_to_size),
        ("extend_to_contact",       extend_lines_to_contact),
        ("recolor_2nd_dom",         recolor_2nd_to_dominant),
        ("erase_2nd_color",         erase_2nd_color),
        ("fill_enclosed_dominant",  recolor_bg_enclosed_by_dominant),
    ]

    for name, fn in unary_ops:
        prims.append(Primitive(name=name, arity=1, fn=fn, domain="arc"))

    # Color-specific keep/remove primitives for colors 1-9
    for color in range(1, 10):
        prims.append(Primitive(
            name=f"keep_c{color}",
            arity=1,
            fn=_make_keep_color(color),
            domain="arc",
        ))

    # Erase color (replace with bg)
    for color in range(1, 10):
        prims.append(Primitive(
            name=f"erase_{color}",
            arity=1,
            fn=_make_erase_color(color),
            domain="arc",
        ))

    # Fill background with color
    for color in range(1, 10):
        prims.append(Primitive(
            name=f"fill_bg_{color}",
            arity=1,
            fn=_make_fill_bg(color),
            domain="arc",
        ))

    # Recolor all non-zero to specific color
    for color in range(1, 10):
        prims.append(Primitive(
            name=f"recolor_to_{color}",
            arity=1,
            fn=_make_recolor_nonzero(color),
            domain="arc",
        ))

    # Color replacement pairs: replace each color with bg
    for from_c in range(1, 10):
        prims.append(Primitive(
            name=f"recolor_{from_c}_to_0",
            arity=1,
            fn=_make_replace_color(from_c, 0),
            domain="arc",
        ))

    # Pairwise color swaps (most common in ARC)
    for a in range(1, 6):
        for b in range(a + 1, 6):
            prims.append(Primitive(
                name=f"swap_{a}_{b}",
                arity=1,
                fn=_make_swap_colors(a, b),
                domain="arc",
            ))

    # Color swaps from->to for colors 1-4
    for from_c in range(1, 5):
        for to_c in range(1, 5):
            if from_c != to_c:
                prims.append(Primitive(
                    name=f"swap_{from_c}_to_{to_c}",
                    arity=1,
                    fn=_make_replace_color(from_c, to_c),
                    domain="arc",
                ))

    # Fill rect interiors with specific colors
    for color in [1, 2, 3, 4]:
        prims.append(Primitive(
            name=f"fill_rect_interior_{color}",
            arity=1, fn=lambda g, c=color: _fill_rect_interiors(g, c), domain="arc",
        ))

    # Mark row/col intersections with specific colors
    for color in [2, 3, 4]:
        prims.append(Primitive(
            name=f"mark_intersections_{color}",
            arity=1, fn=lambda g, c=color: _recolor_cells_at_intersections(g, c), domain="arc",
        ))

    # Recolor dominant-touching-accent to specific colors
    for color in [2, 3, 4, 6, 7, 8]:
        prims.append(Primitive(
            name=f"dom_touch_accent_{color}",
            arity=1, fn=_make_recolor_dominant_touching_accent(color), domain="arc",
        ))

    # Fill smallest rect hole with specific colors
    for color in [1, 4, 8]:
        prims.append(Primitive(
            name=f"fill_hole_{color}",
            arity=1, fn=_make_fill_smallest_hole(color), domain="arc",
        ))

    # Recolor nonzero inside accent bbox (most common accent/new pairs)
    for accent, new in [(8, 3), (8, 4), (8, 2), (2, 4), (2, 8), (2, 3),
                        (3, 4), (3, 8), (6, 4), (6, 8)]:
        prims.append(Primitive(
            name=f"recolor_in_{accent}_bbox_{new}",
            arity=1, fn=_make_recolor_nonzero_inside_bbox(accent, new), domain="arc",
        ))

    # Arity-2 primitives (compose two transforms): overlay is the main one
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))

    return prims


ARC_PRIMITIVES = _build_arc_primitives()
_PRIM_MAP = {p.name: p for p in ARC_PRIMITIVES}


def register_prim(p: Primitive) -> None:
    """Register a primitive in the lookup map (for task-specific prims)."""
    _PRIM_MAP[p.name] = p


def lookup_prim(name: str) -> Optional[Primitive]:
    """Look up a primitive by name."""
    return _PRIM_MAP.get(name)


# =============================================================================
# Predicates: Grid → bool functions for conditional branching
# =============================================================================

def _pred_is_symmetric_h(grid: Grid) -> bool:
    """Check if grid has horizontal (left-right) symmetry."""
    return all(row == row[::-1] for row in grid)


def _pred_is_symmetric_v(grid: Grid) -> bool:
    """Check if grid has vertical (top-bottom) symmetry."""
    h = len(grid)
    return all(grid[i] == grid[h - 1 - i] for i in range(h // 2))


def _pred_is_square(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    return h == w and h > 0


def _pred_has_single_color(grid: Grid) -> bool:
    colors = {c for row in grid for c in row if c != 0}
    return len(colors) <= 1


def _pred_is_tall(grid: Grid) -> bool:
    return len(grid) > (len(grid[0]) if grid else 0)


def _pred_is_wide(grid: Grid) -> bool:
    return (len(grid[0]) if grid else 0) > len(grid)


def _pred_has_many_colors(grid: Grid) -> bool:
    colors = {c for row in grid for c in row if c != 0}
    return len(colors) > 3


def _pred_is_small(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    return h * w < 50


def _pred_is_large(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    return h * w > 200


def _pred_has_bg_majority(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0:
        return True
    zero_count = sum(1 for row in grid for c in row if c == 0)
    return zero_count > h * w / 2


def _pred_is_mostly_empty(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0:
        return True
    zero_count = sum(1 for row in grid for c in row if c == 0)
    return zero_count > h * w * 0.8


def _pred_has_frame(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    if h < 3 or w < 3:
        return False
    border = set()
    interior = set()
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == 0:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                border.add(v)
            else:
                interior.add(v)
    return bool(border) and bool(interior) and not (border & interior)


def _pred_has_diag_sym(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    if h != w or h == 0:
        return False
    return all(grid[r][c] == grid[c][r] for r in range(h) for c in range(r + 1, w))


def _pred_is_odd_dims(grid: Grid) -> bool:
    h, w = len(grid), len(grid[0]) if grid else 0
    return h % 2 == 1 and w % 2 == 1 and h > 0


def _pred_has_two_colors(grid: Grid) -> bool:
    colors = {c for row in grid for c in row if c != 0}
    return len(colors) == 2


def _pred_has_h_stripe(grid: Grid) -> bool:
    for row in grid:
        non_zero = [c for c in row if c != 0]
        if len(non_zero) == len(row) and len(set(non_zero)) == 1:
            return True
    return False


def _pred_has_v_stripe(grid: Grid) -> bool:
    h = len(grid)
    w = len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return False
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        non_zero = [v for v in col if v != 0]
        if len(non_zero) == h and len(set(non_zero)) == 1:
            return True
    return False


# All predicates as (name, function) pairs
ARC_PREDICATES: list[tuple[str, callable]] = [
    ("is_symmetric_h", _pred_is_symmetric_h),
    ("is_symmetric_v", _pred_is_symmetric_v),
    ("is_square", _pred_is_square),
    ("has_single_color", _pred_has_single_color),
    ("is_tall", _pred_is_tall),
    ("is_wide", _pred_is_wide),
    ("has_many_colors", _pred_has_many_colors),
    ("is_small", _pred_is_small),
    ("is_large", _pred_is_large),
    ("has_bg_majority", _pred_has_bg_majority),
    ("is_mostly_empty", _pred_is_mostly_empty),
    ("has_frame", _pred_has_frame),
    ("has_diag_sym", _pred_has_diag_sym),
    ("is_odd_dims", _pred_is_odd_dims),
    ("has_two_colors", _pred_has_two_colors),
    ("has_h_stripe", _pred_has_h_stripe),
    ("has_v_stripe", _pred_has_v_stripe),
]
