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
import numba as nb

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
# Numba JIT kernels — accelerate hot inner loops for all expensive primitives.
# ARC grids have at most 10 colors (0-9), so dict-based color counting is
# replaced with fixed int[10] arrays.  BFS queues use pre-allocated arrays.
# =============================================================================

@nb.njit(cache=True)
def _jit_draw_cross(arr, result):
    """Fill entire row and column for each non-zero pixel."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = arr[r, c]
                for cc in range(w):
                    if result[r, cc] == 0:
                        result[r, cc] = color
                for rr in range(h):
                    if result[rr, c] == 0:
                        result[rr, c] = color
    return result


@nb.njit(cache=True)
def _jit_draw_cross_to_contact(arr, result):
    """Extend cross lines from pixels until hitting another non-zero pixel."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = arr[r, c]
                # 4 cardinal directions
                for d in range(4):
                    dr = (-1, 1, 0, 0)[d]
                    dc = (0, 0, -1, 1)[d]
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if arr[nr, nc] != 0:
                            break
                        result[nr, nc] = color
                        nr += dr
                        nc += dc
    return result


@nb.njit(cache=True)
def _jit_draw_diagonal(arr, result):
    """Extend diagonal lines from each non-zero pixel to grid edges."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = arr[r, c]
                for d in range(4):
                    dr = (-1, -1, 1, 1)[d]
                    dc = (-1, 1, -1, 1)[d]
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr, nc] == 0:
                            result[nr, nc] = color
                        nr += dr
                        nc += dc
    return result


@nb.njit(cache=True)
def _jit_draw_diagonal_nearest(arr, result, dist):
    """Diagonal lines with nearest-source priority (BFS-like)."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                dist[r, c] = 0.0
                color = arr[r, c]
                for d in range(4):
                    dr = (-1, -1, 1, 1)[d]
                    dc = (-1, 1, -1, 1)[d]
                    nr, nc = r + dr, c + dc
                    step = 1.0
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr, nc] == 0 and step < dist[nr, nc]:
                            result[nr, nc] = color
                            dist[nr, nc] = step
                        elif result[nr, nc] != 0:
                            break
                        step += 1.0
                        nr += dr
                        nc += dc
    return result


@nb.njit(cache=True)
def _jit_extend_diagonal_lines(arr, result):
    """Extend isolated non-zero cells diagonally."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = arr[r, c]
                # Check if isolated (no same-color orthogonal neighbors)
                has_neighbor = False
                for d in range(4):
                    dr = (-1, 1, 0, 0)[d]
                    dc = (0, 0, -1, 1)[d]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == color:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    for d in range(4):
                        dr = (-1, -1, 1, 1)[d]
                        dc = (-1, 1, -1, 1)[d]
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = color
                            nr += dr
                            nc += dc
    return result


@nb.njit(cache=True)
def _jit_surround_3x3(arr, result, ring_color):
    """Draw 3x3 ring around each non-zero pixel."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                            result[nr, nc] = ring_color
    return result


@nb.njit(cache=True)
def _jit_connect_color_h(arr, result):
    """Fill horizontal gaps between same-colored pixels using int[10] arrays."""
    h, w = arr.shape
    for r in range(h):
        # For each color 1-9, track min and max column
        for color in range(1, 10):
            c_min = w
            c_max = -1
            for c in range(w):
                if arr[r, c] == color:
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c
            if c_max > c_min:
                for c in range(c_min + 1, c_max):
                    if result[r, c] == 0:
                        result[r, c] = color
    return result


@nb.njit(cache=True)
def _jit_connect_color_v(arr, result):
    """Fill vertical gaps between same-colored pixels using int[10] arrays."""
    h, w = arr.shape
    for c in range(w):
        for color in range(1, 10):
            r_min = h
            r_max = -1
            for r in range(h):
                if arr[r, c] == color:
                    if r < r_min:
                        r_min = r
                    if r > r_max:
                        r_max = r
            if r_max > r_min:
                for r in range(r_min + 1, r_max):
                    if result[r, c] == 0:
                        result[r, c] = color
    return result


@nb.njit(cache=True)
def _jit_fill_between_h(arr, result):
    """Fill horizontal gaps between same-colored objects."""
    h, w = arr.shape
    for r in range(h):
        for color in range(1, 10):
            c_min = w
            c_max = -1
            for c in range(w):
                if arr[r, c] == color:
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c
            if c_max > c_min:
                for c in range(c_min, c_max + 1):
                    if result[r, c] == 0:
                        result[r, c] = color
    return result


@nb.njit(cache=True)
def _jit_fill_between_v(arr, result):
    """Fill vertical gaps between same-colored objects."""
    h, w = arr.shape
    for c in range(w):
        for color in range(1, 10):
            r_min = h
            r_max = -1
            for r in range(h):
                if arr[r, c] == color:
                    if r < r_min:
                        r_min = r
                    if r > r_max:
                        r_max = r
            if r_max > r_min:
                for r in range(r_min, r_max + 1):
                    if result[r, c] == 0:
                        result[r, c] = color
    return result


@nb.njit(cache=True)
def _jit_fill_between_diagonal(arr, result):
    """Fill diagonal lines between same-colored pixels.

    For each color and each diagonal index (r-c and r+c), find pixels
    on that diagonal and fill between adjacent pairs.
    """
    h, w = arr.shape
    max_diag = h + w  # max possible diagonal index offset

    # For r-c diagonals (backslash \): index range [-(w-1), h-1]
    # Offset by (w-1) so index is always >= 0
    # For r+c diagonals (forward /): index range [0, h+w-2]
    for pass_type in range(2):  # 0 = r-c, 1 = r+c
        for color in range(1, 10):
            # Collect pixels on each diagonal for this color
            # Use a flat buffer: for each diagonal, store (row, col) pairs
            # Max pixels per diagonal = min(h, w)
            for diag_idx in range(max_diag):
                # Collect pixels on this diagonal with this color
                n_pts = 0
                # Pre-allocated arrays for up to max(h,w) points
                buf_r = np.empty(max(h, w), dtype=np.int32)
                buf_c = np.empty(max(h, w), dtype=np.int32)

                for r in range(h):
                    for c in range(w):
                        if arr[r, c] == color:
                            if pass_type == 0:
                                d = r - c + (w - 1)
                            else:
                                d = r + c
                            if d == diag_idx:
                                buf_r[n_pts] = r
                                buf_c[n_pts] = c
                                n_pts += 1

                if n_pts < 2:
                    continue

                # Sort by row (already in order since we iterate r ascending)
                # Fill between adjacent pairs
                for k in range(n_pts - 1):
                    r1, c1 = buf_r[k], buf_c[k]
                    r2, c2 = buf_r[k + 1], buf_c[k + 1]
                    dr = abs(r2 - r1)
                    if dr <= 1:
                        continue
                    sr = 1 if r2 > r1 else -1
                    sc = 1 if c2 > c1 else -1
                    for step in range(1, dr):
                        rr = r1 + step * sr
                        cc = c1 + step * sc
                        if result[rr, cc] == 0:
                            result[rr, cc] = color

    return result


@nb.njit(cache=True)
def _jit_denoise_5x5(arr, result):
    """Replace each cell with 5x5 neighborhood majority using int[10] counts."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            counts = np.zeros(10, dtype=np.int32)
            total = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = arr[nr, nc]
                        if 0 <= v < 10:
                            counts[v] += 1
                        total += 1
            best = 0
            best_count = 0
            for v in range(10):
                if counts[v] > best_count:
                    best_count = counts[v]
                    best = v
            if best_count > total // 2:
                result[r, c] = best
    return result


@nb.njit(cache=True)
def _jit_try_tile(arr, h, w, th, tw):
    """Try a specific tile size. Returns (success, result_array)."""
    votes = np.zeros((th, tw, 10), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            v = arr[r, c]
            if v != 0 and 0 <= v < 10:
                votes[r % th, c % tw, v] += 1
    tile = np.zeros((th, tw), dtype=np.int32)
    for tr in range(th):
        for tc in range(tw):
            best = 0
            best_count = 0
            for v in range(10):
                if votes[tr, tc, v] > best_count:
                    best_count = votes[tr, tc, v]
                    best = v
            if best_count == 0:
                return False, arr.copy()
            tile[tr, tc] = best
    # Verify: all non-zero cells must agree exactly
    for r in range(h):
        for c in range(w):
            v = arr[r, c]
            if v != 0 and tile[r % th, c % tw] != v:
                return False, arr.copy()
    result = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            result[r, c] = tile[r % th, c % tw]
    return True, result


@nb.njit(cache=True)
def _jit_fill_tile_pattern(arr, h, w):
    """Infer repeating tile from visible cells and fill zeros.

    Fast path: try tiles that divide grid evenly (original behavior).
    Slow path: if no divisor tile works, try non-divisor tiles + wide/tall.
    """
    # Quick check: if no zeros, nothing to fill
    has_zero = False
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0:
                has_zero = True
                break
        if has_zero:
            break
    if not has_zero:
        return arr.copy()

    # Fast path: divisor tiles only (same cost as original)
    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            if th == h and tw == w:
                continue
            ok, result = _jit_try_tile(arr, h, w, th, tw)
            if ok:
                return result

    # Slow path: non-divisor small 2D tiles
    limit = min(h // 2 + 1, 11)
    limit_w = min(w // 2 + 1, 11)
    for th in range(1, limit):
        for tw in range(1, limit_w):
            if h % th == 0 and w % tw == 0:
                continue  # already tried
            ok, result = _jit_try_tile(arr, h, w, th, tw)
            if ok:
                return result

    # Wide tiles (row-periodic patterns where col period = full width)
    for th in range(1, h // 2 + 1):
        if h % th == 0:
            continue  # already tried as divisor
        ok, result = _jit_try_tile(arr, h, w, th, w)
        if ok:
            return result

    # Tall tiles (col-periodic patterns where row period = full height)
    for tw in range(1, w // 2 + 1):
        if w % tw == 0:
            continue  # already tried as divisor
        ok, result = _jit_try_tile(arr, h, w, h, tw)
        if ok:
            return result

    return arr.copy()


@nb.njit(cache=True)
def _jit_fill_holes_in_objects(arr, bg, h, w):
    """Fill enclosed zero-regions with surrounding color via BFS + raycast."""
    result = arr.copy()
    reachable = np.zeros((h, w), dtype=nb.boolean)

    # BFS queue (pre-allocated, max size = h*w)
    queue_r = np.empty(h * w, dtype=np.int32)
    queue_c = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0

    # Seed borders
    for r in range(h):
        for c_idx in range(2):
            c = 0 if c_idx == 0 else w - 1
            if c < w and arr[r, c] == bg and not reachable[r, c]:
                reachable[r, c] = True
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1
    for c in range(w):
        for r_idx in range(2):
            r = 0 if r_idx == 0 else h - 1
            if r < h and arr[r, c] == bg and not reachable[r, c]:
                reachable[r, c] = True
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    # BFS flood fill
    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        for d in range(4):
            dr = (-1, 1, 0, 0)[d]
            dc = (0, 0, -1, 1)[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and arr[nr, nc] == bg:
                reachable[nr, nc] = True
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

    # Fill unreachable bg cells by raycasting to find surrounding color
    for r in range(h):
        for c in range(w):
            if arr[r, c] == bg and not reachable[r, c]:
                for d in range(4):
                    dr = (-1, 1, 0, 0)[d]
                    dc = (0, 0, -1, 1)[d]
                    for dist in range(1, max(h, w)):
                        nr = r + dr * dist
                        nc = c + dc * dist
                        if not (0 <= nr < h and 0 <= nc < w):
                            break
                        if arr[nr, nc] != bg:
                            result[r, c] = arr[nr, nc]
                            break
                    if result[r, c] != bg:
                        break
    return result


@nb.njit(cache=True)
def _jit_extend_lines_h(arr, result):
    """Extend non-zero pixels horizontally to fill their row."""
    h, w = arr.shape
    for r in range(h):
        # Find most common non-zero color in this row
        counts = np.zeros(10, dtype=np.int32)
        for c in range(w):
            v = arr[r, c]
            if v != 0 and 0 <= v < 10:
                counts[v] += 1
        mc = 0
        best = 0
        for v in range(1, 10):
            if counts[v] > best:
                best = counts[v]
                mc = v
        if mc > 0:
            for c in range(w):
                if result[r, c] == 0:
                    result[r, c] = mc
    return result


@nb.njit(cache=True)
def _jit_extend_lines_v(arr, result):
    """Extend non-zero pixels vertically to fill their column."""
    h, w = arr.shape
    for c in range(w):
        counts = np.zeros(10, dtype=np.int32)
        for r in range(h):
            v = arr[r, c]
            if v != 0 and 0 <= v < 10:
                counts[v] += 1
        mc = 0
        best = 0
        for v in range(1, 10):
            if counts[v] > best:
                best = counts[v]
                mc = v
        if mc > 0:
            for r in range(h):
                if result[r, c] == 0:
                    result[r, c] = mc
    return result


@nb.njit(cache=True)
def _jit_most_common_overall(arr):
    """Find most common color overall (including bg=0)."""
    h, w = arr.shape
    counts = np.zeros(10, dtype=np.int32)
    for r in range(h):
        for c in range(w):
            v = arr[r, c]
            if 0 <= v < 10:
                counts[v] += 1
    best = 0
    best_count = 0
    for v in range(10):
        if counts[v] > best_count:
            best_count = counts[v]
            best = v
    return best


@nb.njit(cache=True)
def _jit_extend_lines_to_contact(arr, result, bg):
    """Extend colored segments to fill gaps within row/column, per color."""
    h, w = arr.shape
    # Horizontal
    for r in range(h):
        for color in range(10):
            if color == bg:
                continue
            c_min = w
            c_max = -1
            for c in range(w):
                if arr[r, c] == color:
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c
            if c_max > c_min:
                for c in range(c_min, c_max + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
    # Vertical
    for c in range(w):
        for color in range(10):
            if color == bg:
                continue
            r_min = h
            r_max = -1
            for r in range(h):
                if arr[r, c] == color:
                    if r < r_min:
                        r_min = r
                    if r > r_max:
                        r_max = r
            if r_max > r_min:
                for r in range(r_min, r_max + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
    return result


@nb.njit(cache=True)
def _jit_fill_grid_intersections(arr, result):
    """Fill bg cells at row/col intersections with matching colors."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                continue
            # Collect row and col colors using bitmask (colors 0-9)
            row_mask = 0
            col_mask = 0
            for cc in range(w):
                if arr[r, cc] != 0:
                    row_mask |= (1 << arr[r, cc])
            for rr in range(h):
                if arr[rr, c] != 0:
                    col_mask |= (1 << arr[rr, c])
            common = row_mask & col_mask
            if common:
                # Find minimum color in common
                for v in range(1, 10):
                    if common & (1 << v):
                        result[r, c] = v
                        break
    return result


@nb.njit(cache=True)
def _jit_complete_sym_90(arr):
    """Complete 90-degree rotational symmetry."""
    h, w = arr.shape
    result = arr.copy()
    if h != w:
        return result
    n = h
    for _iter in range(4):
        changed = False
        for r in range(n):
            for c in range(n):
                if result[r, c] != 0:
                    r2, c2 = c, n - 1 - r
                    if 0 <= r2 < n and 0 <= c2 < n and result[r2, c2] == 0:
                        result[r2, c2] = result[r, c]
                        changed = True
                    r3, c3 = n - 1 - r, n - 1 - c
                    if 0 <= r3 < n and 0 <= c3 < n and result[r3, c3] == 0:
                        result[r3, c3] = result[r, c]
                        changed = True
                    r4, c4 = n - 1 - c, r
                    if 0 <= r4 < n and 0 <= c4 < n and result[r4, c4] == 0:
                        result[r4, c4] = result[r, c]
                        changed = True
        if not changed:
            break
    return result


@nb.njit(cache=True)
def _jit_complete_sym_180(arr):
    """Complete 180-degree rotational symmetry."""
    h, w = arr.shape
    result = arr.copy()
    for _iter in range(2):
        changed = False
        for r in range(h):
            for c in range(w):
                if result[r, c] != 0:
                    r2, c2 = h - 1 - r, w - 1 - c
                    if 0 <= r2 < h and 0 <= c2 < w and result[r2, c2] == 0:
                        result[r2, c2] = result[r, c]
                        changed = True
        if not changed:
            break
    return result


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
    return from_np(_jit_extend_lines_h(arr, arr.copy()))


def extend_lines_v(grid: Grid) -> Grid:
    """Extend non-zero pixels vertically to fill their column."""
    arr = to_np(grid)
    return from_np(_jit_extend_lines_v(arr, arr.copy()))


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
    return from_np(_jit_extend_diagonal_lines(arr, arr.copy()))


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
    arr = to_np(grid)
    return from_np(_jit_denoise_5x5(arr, arr.copy()))


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
    arr = to_np(grid)
    h, w = arr.shape
    bg = np.int32(_jit_most_common_overall(arr))
    return from_np(_jit_fill_holes_in_objects(arr, bg, h, w))


def _most_common_overall(grid: Grid) -> int:
    """Find the most common color overall (including bg)."""
    return int(_jit_most_common_overall(to_np(grid)))


def fill_tile_pattern(grid: Grid) -> Grid:
    """Infer a repeating tile from visible cells and fill zeros with it."""
    if not grid or not grid[0]:
        return grid
    arr = to_np(grid)
    h, w = arr.shape
    return from_np(_jit_fill_tile_pattern(arr, h, w))


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


# --- Context-dependent color primitives ---
# These address the near-miss gap: programs that get geometry right but colors wrong.


def recolor_by_neighbor_vote(grid: Grid) -> Grid:
    """Recolor each non-background cell to the majority color of its 4-neighbors.

    Fixes "artifact" cells that have the wrong color relative to their
    neighborhood. Leaves background (0) untouched. Ties broken by keeping
    the original color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            neighbors: dict[int, int] = {}
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                    neighbors[grid[nr][nc]] = neighbors.get(grid[nr][nc], 0) + 1
            if neighbors:
                best = max(neighbors, key=lambda k: neighbors[k])
                if neighbors[best] > neighbors.get(grid[r][c], 0):
                    result[r][c] = best
    return result


def recolor_by_8neighbor_vote(grid: Grid) -> Grid:
    """Like recolor_by_neighbor_vote but uses 8-connected neighbors."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            neighbors: dict[int, int] = {}
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        neighbors[grid[nr][nc]] = neighbors.get(grid[nr][nc], 0) + 1
            if neighbors:
                best = max(neighbors, key=lambda k: neighbors[k])
                if neighbors[best] > neighbors.get(grid[r][c], 0):
                    result[r][c] = best
    return result


def swap_two_most_common(grid: Grid) -> Grid:
    """Swap the two most common non-background colors.

    Identifies the two most frequent non-zero colors and swaps them.
    Useful when a transformation gets the structure right but assigns
    the two main colors backwards.
    """
    flat = [c for row in grid for c in row if c != 0]
    if len(flat) < 2:
        return [row[:] for row in grid]
    counts = Counter(flat)
    top2 = [c for c, _ in counts.most_common(2)]
    if len(top2) < 2:
        return [row[:] for row in grid]
    a, b = top2
    return [[b if v == a else (a if v == b else v) for v in row] for row in grid]


def swap_two_least_common(grid: Grid) -> Grid:
    """Swap the two least common non-background colors."""
    flat = [c for row in grid for c in row if c != 0]
    if len(flat) < 2:
        return [row[:] for row in grid]
    counts = Counter(flat)
    ranked = [c for c, _ in counts.most_common()]
    if len(ranked) < 2:
        return [row[:] for row in grid]
    a, b = ranked[-1], ranked[-2]
    return [[b if v == a else (a if v == b else v) for v in row] for row in grid]


def fill_by_surround_color(grid: Grid) -> Grid:
    """Fill background cells with the color of the surrounding non-bg region.

    For each background (0) cell that has exactly one non-zero color in its
    4-neighbors, fill it with that color. Iterates until stable (flood-fill
    from edges inward).
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    changed = True
    max_iters = max(h, w)
    for _ in range(max_iters):
        if not changed:
            break
        changed = False
        for r in range(h):
            for c in range(w):
                if result[r][c] != 0:
                    continue
                neighbor_colors: set[int] = set()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] != 0:
                        neighbor_colors.add(result[nr][nc])
                if len(neighbor_colors) == 1:
                    result[r][c] = neighbor_colors.pop()
                    changed = True
    return result


def cleanup_isolated_cells(grid: Grid) -> Grid:
    """Remove non-background cells that have no same-colored 4-neighbors.

    These are often artifacts from imperfect structural transformations.
    Sets isolated colored cells to background (0).
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            has_same = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == grid[r][c]:
                    has_same = True
                    break
            if not has_same:
                result[r][c] = 0
    return result


def recolor_minority_to_majority(grid: Grid) -> Grid:
    """For each connected component, recolor minority cells to the majority.

    Uses 4-connectivity. For each component, finds the most common color
    and recolors all other cells in the component to that color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    result = [row[:] for row in grid]

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] == 0:
                continue
            # BFS to find connected non-zero component
            component = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if visited[cr][cc]:
                    continue
                visited[cr][cc] = True
                component.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                        stack.append((nr, nc))
            # Find majority color in component
            color_counts: dict[int, int] = {}
            for cr, cc in component:
                v = grid[cr][cc]
                color_counts[v] = color_counts.get(v, 0) + 1
            majority = max(color_counts, key=lambda k: color_counts[k])
            for cr, cc in component:
                result[cr][cc] = majority

    return result


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
    arr = to_np(grid)
    return from_np(_jit_fill_grid_intersections(arr, arr.copy()))


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
    arr = to_np(grid)
    bg = np.int32(_jit_most_common_overall(arr))
    return from_np(_jit_extend_lines_to_contact(arr, arr.copy(), bg))


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


def erase_rare(grid: Grid) -> Grid:
    """Erase (set to bg) the rarest non-bg color."""
    if not grid or not grid[0]:
        return grid
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in counts.items() if v != bg]
    if not non_bg:
        return grid
    rare = min(non_bg, key=lambda x: x[1])[0]
    return [[bg if v == rare else v for v in row] for row in grid]


def erase_3rd_color(grid: Grid) -> Grid:
    """Erase (set to bg) the 3rd most common non-bg color."""
    if not grid or not grid[0]:
        return grid
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 3:
        return grid
    target = non_bg[2]
    return [[bg if v == target else v for v in row] for row in grid]


def fill_hole_by_neighbor(grid: Grid) -> Grid:
    """Fill zero (bg) cells with their most common non-bg 4-neighbor color.

    Useful for tasks where holes should be filled with the surrounding color.
    Iterates once — for propagation, use with apply_until_stable.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                continue
            # Count non-bg 4-neighbors
            neighbor_counts: dict[int, int] = {}
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                    v = grid[nr][nc]
                    neighbor_counts[v] = neighbor_counts.get(v, 0) + 1
            if neighbor_counts:
                result[r][c] = max(neighbor_counts, key=lambda k: neighbor_counts[k])
    return result


def fill_hole_dominant(grid: Grid) -> Grid:
    """Fill zero (bg) cells that are enclosed by non-bg with the dominant color.

    Unlike fill_enclosed which uses a hard-coded color, this uses the most
    common non-bg color in the grid.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in counts.items() if v != bg]
    if not non_bg:
        return grid
    dominant = max(non_bg, key=lambda x: x[1])[0]

    # BFS from boundary to find non-enclosed bg
    reachable = set()
    queue = []
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    while queue:
        cr, cc = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < h and 0 <= nc < w
                    and (nr, nc) not in reachable and grid[nr][nc] == bg):
                reachable.add((nr, nc))
                queue.append((nr, nc))

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r, c) not in reachable:
                result[r][c] = dominant
    return result


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
# Batch 3: Additional high-value primitives
# =============================================================================

def _fill_bg_adjacent_to_color(target_color: int, fill_color: int):
    """Fill background cells adjacent to target_color with fill_color."""
    def fn(grid: Grid) -> Grid:
        a = to_np(grid)
        h, w = a.shape
        result = a.copy()
        for r in range(h):
            for c in range(w):
                if a[r, c] != 0:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and a[nr, nc] == target_color:
                        result[r, c] = fill_color
                        break
        return result.tolist()
    return fn


def color_by_row_position(grid: Grid) -> Grid:
    """Assign each non-zero cell a color based on its row index (mod 9 + 1)."""
    return [[(r % 9 + 1) if c != 0 else 0 for c in row] for r, row in enumerate(grid)]


def color_by_col_position(grid: Grid) -> Grid:
    """Assign each non-zero cell a color based on its column index (mod 9 + 1)."""
    return [[(ci % 9 + 1) if c != 0 else 0 for ci, c in enumerate(row)] for row in grid]


def fill_col_from_bottom(grid: Grid) -> Grid:
    """For each column, fill upward from the bottom-most non-zero cell."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for c in range(w):
        bottom_color = 0
        for r in range(h - 1, -1, -1):
            if a[r, c] != 0:
                bottom_color = a[r, c]
                break
        if bottom_color:
            for r in range(h):
                if result[r, c] == 0:
                    result[r, c] = bottom_color
    return result.tolist()


def fill_row_from_right(grid: Grid) -> Grid:
    """For each row, fill leftward from the right-most non-zero cell."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for r in range(h):
        right_color = 0
        for c in range(w - 1, -1, -1):
            if a[r, c] != 0:
                right_color = a[r, c]
                break
        if right_color:
            for c in range(w):
                if result[r, c] == 0:
                    result[r, c] = right_color
    return result.tolist()


def extend_color_within_row_bounds(grid: Grid) -> Grid:
    """Extend non-zero cells horizontally to fill gaps within the same row's bounds."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for r in range(h):
        nonzero = [(c, a[r, c]) for c in range(w) if a[r, c] != 0]
        if len(nonzero) < 2:
            continue
        min_c = nonzero[0][0]
        max_c = nonzero[-1][0]
        for c in range(min_c, max_c + 1):
            if result[r, c] == 0:
                # Fill with nearest non-zero in this row
                left_color = 0
                for cc in range(c - 1, -1, -1):
                    if a[r, cc] != 0:
                        left_color = a[r, cc]
                        break
                result[r, c] = left_color
    return result.tolist()


def extend_color_within_col_bounds(grid: Grid) -> Grid:
    """Extend non-zero cells vertically to fill gaps within the same column's bounds."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for c in range(w):
        nonzero = [(r, a[r, c]) for r in range(h) if a[r, c] != 0]
        if len(nonzero) < 2:
            continue
        min_r = nonzero[0][0]
        max_r = nonzero[-1][0]
        for r in range(min_r, max_r + 1):
            if result[r, c] == 0:
                top_color = 0
                for rr in range(r - 1, -1, -1):
                    if a[rr, c] != 0:
                        top_color = a[rr, c]
                        break
                result[r, c] = top_color
    return result.tolist()


def fill_diagonal_stripes(grid: Grid) -> Grid:
    """Fill background cells where diagonal neighbors have the same non-zero color."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0:
                continue
            diag_colors = []
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and a[nr, nc] != 0:
                    diag_colors.append(a[nr, nc])
            if len(diag_colors) >= 2 and len(set(diag_colors)) == 1:
                result[r, c] = diag_colors[0]
    return result.tolist()


def fill_rooms_with_new_color(grid: Grid) -> Grid:
    """Fill each enclosed room (connected bg region not touching border) with a unique color."""
    a = to_np(grid)
    h, w = a.shape
    visited = np.zeros((h, w), dtype=bool)

    # Mark border-connected background
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and a[r, c] == 0:
                visited[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and a[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Find and fill interior rooms
    result = a.copy()
    color = 1
    for r in range(h):
        for c in range(w):
            if a[r, c] == 0 and not visited[r, c]:
                # Flood fill this room
                room = [(r, c)]
                visited[r, c] = True
                idx = 0
                while idx < len(room):
                    cr, cc = room[idx]
                    idx += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and a[nr, nc] == 0:
                            visited[nr, nc] = True
                            room.append((nr, nc))
                # Assign color to this room (cycle through 1-9)
                for cr, cc in room:
                    result[cr, cc] = color
                color = (color % 9) + 1
    return result.tolist()


def count_per_row(grid: Grid) -> Grid:
    """Replace each row with a count of non-zero cells (as a 1-wide grid)."""
    return [[sum(1 for c in row if c != 0)] for row in grid]


def count_objects_grid(grid: Grid) -> Grid:
    """Return a 1×1 grid containing the number of connected components."""
    from .objects import _find_connected_components
    comps = _find_connected_components(grid)
    return [[len(comps)]]


def recolor_by_size_rank(grid: Grid) -> Grid:
    """Recolor objects by size rank: largest=1, next=2, etc."""
    from .objects import _find_connected_components
    comps = _find_connected_components(grid)
    if not comps:
        return [row[:] for row in grid]
    comps_sorted = sorted(comps, key=lambda c: -c["size"])
    result = [[0] * len(row) for row in grid]
    for rank, comp in enumerate(comps_sorted):
        color = rank % 9 + 1
        for r, c in comp["pixels"]:
            result[r][c] = color
    return result


def downscale_2x(grid: Grid) -> Grid:
    """Downscale grid by 2x, taking the most common non-zero value in each 2x2 block."""
    a = to_np(grid)
    h, w = a.shape
    nh, nw = h // 2, w // 2
    if nh == 0 or nw == 0:
        return grid
    result = np.zeros((nh, nw), dtype=np.int32)
    for r in range(nh):
        for c in range(nw):
            block = a[r*2:r*2+2, c*2:c*2+2].flatten()
            nonzero = [v for v in block if v != 0]
            if nonzero:
                result[r, c] = max(set(nonzero), key=nonzero.count)
    return result.tolist()


def downscale_3x(grid: Grid) -> Grid:
    """Downscale grid by 3x, taking the most common non-zero value in each 3x3 block."""
    a = to_np(grid)
    h, w = a.shape
    nh, nw = h // 3, w // 3
    if nh == 0 or nw == 0:
        return grid
    result = np.zeros((nh, nw), dtype=np.int32)
    for r in range(nh):
        for c in range(nw):
            block = a[r*3:r*3+3, c*3:c*3+3].flatten()
            nonzero = [v for v in block if v != 0]
            if nonzero:
                result[r, c] = max(set(nonzero), key=nonzero.count)
    return result.tolist()


def extend_nonzero_fill_row(grid: Grid) -> Grid:
    """For each row, extend non-zero values to fill entire row."""
    result = []
    for row in grid:
        nonzero = [c for c in row if c != 0]
        if nonzero:
            fill = max(set(nonzero), key=nonzero.count)
            result.append([fill] * len(row))
        else:
            result.append(row[:])
    return result


def extend_nonzero_fill_col(grid: Grid) -> Grid:
    """For each column, extend non-zero values to fill entire column."""
    a = to_np(grid)
    h, w = a.shape
    result = a.copy()
    for c in range(w):
        col = a[:, c]
        nonzero = [v for v in col if v != 0]
        if nonzero:
            fill = max(set(nonzero), key=nonzero.count)
            result[:, c] = fill
    return result.tolist()


# =============================================================================
# Batch 4: Grid partition, annotation, and scaling primitives
# =============================================================================

import functools

@functools.lru_cache(maxsize=64)
def _detect_any_separator_lines_cached(grid_key: tuple) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Cached separator line detection. Takes hashable grid tuple."""
    h = len(grid_key)
    w = len(grid_key[0]) if h > 0 else 0
    h_lines: list[int] = []
    v_lines: list[int] = []

    # First try non-zero separators (standard)
    for r in range(h):
        row = grid_key[r]
        vals = set(row)
        if len(vals) == 1 and 0 not in vals:
            h_lines.append(r)

    for c in range(w):
        col = tuple(grid_key[r][c] for r in range(h))
        vals = set(col)
        if len(vals) == 1 and 0 not in vals:
            v_lines.append(c)

    if h_lines or v_lines:
        return tuple(h_lines), tuple(v_lines)

    # Fall back: try zero-valued separators
    for r in range(h):
        if all(v == 0 for v in grid_key[r]):
            has_above = r > 0 and any(v != 0 for rr in range(r) for v in grid_key[rr])
            has_below = r < h - 1 and any(v != 0 for rr in range(r+1, h) for v in grid_key[rr])
            if has_above and has_below:
                h_lines.append(r)

    for c in range(w):
        col = [grid_key[r][c] for r in range(h)]
        if all(v == 0 for v in col):
            has_left = c > 0 and any(grid_key[r][cc] != 0 for r in range(h) for cc in range(c))
            has_right = c < w - 1 and any(grid_key[r][cc] != 0 for r in range(h) for cc in range(c+1, w))
            if has_left and has_right:
                v_lines.append(c)

    return tuple(h_lines), tuple(v_lines)


def _detect_any_separator_lines(grid: Grid) -> tuple[list[int], list[int]]:
    """Detect separator lines including zero-valued (background) separators."""
    grid_key = tuple(tuple(row) for row in grid)
    h_lines, v_lines = _detect_any_separator_lines_cached(grid_key)
    return list(h_lines), list(v_lines)


@functools.lru_cache(maxsize=64)
def _split_grid_cells_cached(grid_key: tuple) -> tuple:
    """Cached grid cell splitting. Returns tuple of cell tuples."""
    grid = [list(row) for row in grid_key]
    cells = _split_grid_cells_impl(grid)
    return tuple(tuple(tuple(row) for row in cell) for cell in cells)


def _split_grid_cells(grid: Grid) -> list[Grid]:
    """Split grid into cells (cached)."""
    grid_key = tuple(tuple(row) for row in grid)
    cached = _split_grid_cells_cached(grid_key)
    return [[list(row) for row in cell] for cell in cached]


def _split_grid_cells_impl(grid: Grid) -> list[Grid]:
    """Split grid into cells using both horizontal and vertical separator lines.

    Returns list of sub-grids (cells) extracted from the partition.
    Falls back to equal division if no separators found.
    """
    arr = to_np(grid)
    h, w = arr.shape

    h_lines, v_lines = _detect_any_separator_lines(grid)

    # Build row boundaries
    if h_lines:
        row_bounds = []
        prev = 0
        for rl in h_lines:
            if rl > prev:
                row_bounds.append((prev, rl))
            prev = rl + 1
        if prev < h:
            row_bounds.append((prev, h))
    else:
        # Try equal division
        for n in [2, 3, 4, 5]:
            if h % n == 0:
                bh = h // n
                row_bounds = [(i * bh, (i + 1) * bh) for i in range(n)]
                break
        else:
            row_bounds = [(0, h)]

    # Build col boundaries
    if v_lines:
        col_bounds = []
        prev = 0
        for cl in v_lines:
            if cl > prev:
                col_bounds.append((prev, cl))
            prev = cl + 1
        if prev < w:
            col_bounds.append((prev, w))
    else:
        for n in [2, 3, 4, 5]:
            if w % n == 0:
                bw = w // n
                col_bounds = [(i * bw, (i + 1) * bw) for i in range(n)]
                break
        else:
            col_bounds = [(0, w)]

    cells = []
    for r_start, r_end in row_bounds:
        for c_start, c_end in col_bounds:
            cell = arr[r_start:r_end, c_start:c_end]
            if cell.size > 0:
                cells.append(from_np(cell))
    return cells


def select_odd_one_out(grid: Grid) -> Grid:
    """Split grid by separators, return the cell that differs from the majority."""
    cells = _split_grid_cells(grid)
    if len(cells) < 2:
        return [row[:] for row in grid]

    # Convert to hashable tuples for comparison
    cell_tuples = [tuple(tuple(r) for r in c) for c in cells]
    counts: dict = {}
    for ct in cell_tuples:
        counts[ct] = counts.get(ct, 0) + 1

    if len(counts) == 1:
        return [row[:] for row in grid]

    # Return least common cell
    least = min(counts, key=lambda k: counts[k])
    return [list(r) for r in least]


def overlay_grid_cells(grid: Grid) -> Grid:
    """Split grid by separators, overlay all cells (non-zero takes priority)."""
    cells = _split_grid_cells(grid)
    if len(cells) < 2:
        return [row[:] for row in grid]

    # All cells must be same shape
    shapes = set((len(c), len(c[0])) for c in cells)
    if len(shapes) != 1:
        return cells[0] if cells else [row[:] for row in grid]

    ch, cw = shapes.pop()
    result = np.zeros((ch, cw), dtype=np.int32)
    for cell in cells:
        a = to_np(cell)
        mask = a != 0
        result[mask] = a[mask]
    return from_np(result)


def majority_vote_cells(grid: Grid) -> Grid:
    """Split grid by separators, take majority vote per pixel across cells."""
    cells = _split_grid_cells(grid)
    if len(cells) < 2:
        return [row[:] for row in grid]

    shapes = set((len(c), len(c[0])) for c in cells)
    if len(shapes) != 1:
        return cells[0] if cells else [row[:] for row in grid]

    ch, cw = shapes.pop()
    result = np.zeros((ch, cw), dtype=np.int32)
    stack = np.stack([to_np(c) for c in cells], axis=0)

    for r in range(ch):
        for c in range(cw):
            vals = stack[:, r, c]
            nonzero = [int(v) for v in vals if v != 0]
            if nonzero:
                result[r, c] = max(set(nonzero), key=nonzero.count)
    return from_np(result)


def xor_grid_cells(grid: Grid) -> Grid:
    """Split grid by separators, XOR cells: keep pixels that differ from majority."""
    cells = _split_grid_cells(grid)
    if len(cells) < 2:
        return [row[:] for row in grid]

    shapes = set((len(c), len(c[0])) for c in cells)
    if len(shapes) != 1:
        return cells[0] if cells else [row[:] for row in grid]

    ch, cw = shapes.pop()
    majority = np.zeros((ch, cw), dtype=np.int32)
    stack = np.stack([to_np(c) for c in cells], axis=0)

    for r in range(ch):
        for c in range(cw):
            vals = stack[:, r, c]
            nonzero = [int(v) for v in vals if v != 0]
            if nonzero:
                majority[r, c] = max(set(nonzero), key=nonzero.count)

    # For the odd-one-out cell, show where it differs
    cell_tuples = [tuple(tuple(r) for r in c) for c in cells]
    counts: dict = {}
    for ct in cell_tuples:
        counts[ct] = counts.get(ct, 0) + 1

    if len(counts) <= 1:
        return from_np(np.zeros((ch, cw), dtype=np.int32))

    least = min(counts, key=lambda k: counts[k])
    odd = np.array(least, dtype=np.int32)

    result = np.zeros((ch, cw), dtype=np.int32)
    diff_mask = odd != majority
    result[diff_mask] = odd[diff_mask]
    return from_np(result)


def extract_top_left_cell_2d(grid: Grid) -> Grid:
    """Extract the top-left cell from 2D grid partition (split by both h and v lines)."""
    cells = _split_grid_cells(grid)
    return cells[0] if cells else [row[:] for row in grid]


def select_most_colorful_cell(grid: Grid) -> Grid:
    """Split grid by separators, return cell with most distinct non-zero colors."""
    cells = _split_grid_cells(grid)
    if not cells:
        return [row[:] for row in grid]

    best = max(cells, key=lambda c: len(set(v for row in c for v in row if v != 0)))
    return best


def select_most_filled_cell(grid: Grid) -> Grid:
    """Split grid by separators, return cell with most non-zero pixels."""
    cells = _split_grid_cells(grid)
    if not cells:
        return [row[:] for row in grid]

    best = max(cells, key=lambda c: sum(1 for row in c for v in row if v != 0))
    return best


def select_least_filled_cell(grid: Grid) -> Grid:
    """Split grid by separators, return cell with fewest non-zero pixels (but > 0)."""
    cells = _split_grid_cells(grid)
    if not cells:
        return [row[:] for row in grid]

    nonempty = [c for c in cells if any(v != 0 for row in c for v in row)]
    if not nonempty:
        return cells[0]
    return min(nonempty, key=lambda c: sum(1 for row in c for v in row if v != 0))


# --- Pixel annotation primitives ---

def surround_pixels_3x3(grid: Grid) -> Grid:
    """Draw a 3x3 ring around each non-zero pixel using the next available color."""
    arr = to_np(grid)
    # Find the ring color: most common nonzero, or 1
    colors = set(arr[arr != 0].tolist())
    ring_color = max(colors) + 1 if colors else 1
    if ring_color > 9:
        ring_color = 1
    return from_np(_jit_surround_3x3(arr, arr.copy(), np.int32(ring_color)))


def draw_cross_from_pixels(grid: Grid) -> Grid:
    """From each non-zero pixel, draw cross lines (up/down/left/right) to grid edge."""
    arr = to_np(grid)
    return from_np(_jit_draw_cross(arr, arr.copy()))


def draw_cross_to_contact(grid: Grid) -> Grid:
    """From each non-zero pixel, draw cross lines until hitting another non-zero pixel."""
    arr = to_np(grid)
    return from_np(_jit_draw_cross_to_contact(arr, arr.copy()))


def draw_diagonal_from_pixels(grid: Grid) -> Grid:
    """From each non-zero pixel, draw diagonal lines to grid edge."""
    arr = to_np(grid)
    return from_np(_jit_draw_diagonal(arr, arr.copy()))


def connect_same_color_h(grid: Grid) -> Grid:
    """Draw horizontal lines between same-colored pixels in each row."""
    arr = to_np(grid)
    return from_np(_jit_connect_color_h(arr, arr.copy()))


def connect_same_color_v(grid: Grid) -> Grid:
    """Draw vertical lines between same-colored pixels in each column."""
    arr = to_np(grid)
    return from_np(_jit_connect_color_v(arr, arr.copy()))


# --- Additional scaling primitives ---

def fill_between_objects_h(grid: Grid) -> Grid:
    """Fill horizontal gaps between same-colored objects in each row."""
    arr = to_np(grid)
    return from_np(_jit_fill_between_h(arr, arr.copy()))


def fill_between_objects_v(grid: Grid) -> Grid:
    """Fill vertical gaps between same-colored objects in each column."""
    arr = to_np(grid)
    return from_np(_jit_fill_between_v(arr, arr.copy()))


def fill_bbox_per_object(grid: Grid) -> Grid:
    """Fill the bounding box of each object with its color (solid rectangles)."""
    arr = to_np(grid)
    h, w = arr.shape
    result = np.zeros_like(arr)

    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and not visited[r, c]:
                color = arr[r, c]
                # BFS to find component
                queue = [(r, c)]
                visited[r, c] = True
                pixels = []
                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                # Fill bounding box
                min_r = min(p[0] for p in pixels)
                max_r = max(p[0] for p in pixels)
                min_c = min(p[1] for p in pixels)
                max_c = max(p[1] for p in pixels)
                result[min_r:max_r+1, min_c:max_c+1] = color
    return from_np(result)


def draw_rect_around_objects(grid: Grid) -> Grid:
    """Draw a rectangle outline around each connected component."""
    arr = to_np(grid)
    h, w = arr.shape
    result = arr.copy()

    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and not visited[r, c]:
                color = arr[r, c]
                queue = [(r, c)]
                visited[r, c] = True
                pixels = []
                while queue:
                    cr, cc = queue.pop(0)
                    pixels.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                min_r = min(p[0] for p in pixels)
                max_r = max(p[0] for p in pixels)
                min_c = min(p[1] for p in pixels)
                max_c = max(p[1] for p in pixels)
                # Draw rectangle outline
                for rr in range(min_r, max_r + 1):
                    if rr in (min_r, max_r):
                        for cc in range(min_c, max_c + 1):
                            result[rr, cc] = color
                    else:
                        result[rr, min_c] = color
                        result[rr, max_c] = color
    return from_np(result)


def scale_4x(grid: Grid) -> Grid:
    """Scale grid up by 4x (each pixel becomes 4x4 block)."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1))


def scale_5x(grid: Grid) -> Grid:
    """Scale grid up by 5x."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 5, axis=0), 5, axis=1))


def _downscale_nx(grid: Grid, n: int) -> Grid:
    """Generic downscale by n, taking most common non-zero in each nxn block."""
    a = to_np(grid)
    h, w = a.shape
    nh, nw = h // n, w // n
    if nh == 0 or nw == 0:
        return grid
    result = np.zeros((nh, nw), dtype=np.int32)
    for r in range(nh):
        for c in range(nw):
            block = a[r*n:r*n+n, c*n:c*n+n].flatten()
            nonzero = [v for v in block if v != 0]
            if nonzero:
                result[r, c] = max(set(nonzero), key=nonzero.count)
    return result.tolist()


def downscale_4x(grid: Grid) -> Grid:
    """Downscale by 4x, majority non-zero per block."""
    return _downscale_nx(grid, 4)


def downscale_5x(grid: Grid) -> Grid:
    """Downscale by 5x, majority non-zero per block."""
    return _downscale_nx(grid, 5)


def downscale_7x(grid: Grid) -> Grid:
    """Downscale by 7x, majority non-zero per block."""
    return _downscale_nx(grid, 7)


# --- Majority/minority downscale (all values including 0) ---

def downscale_majority_2x(grid: Grid) -> Grid:
    """Downscale by 2x, taking majority vote including zeros."""
    a = to_np(grid)
    h, w = a.shape
    nh, nw = h // 2, w // 2
    if nh == 0 or nw == 0:
        return grid
    result = np.zeros((nh, nw), dtype=np.int32)
    for r in range(nh):
        for c in range(nw):
            block = a[r*2:r*2+2, c*2:c*2+2].flatten().tolist()
            result[r, c] = max(set(block), key=block.count)
    return result.tolist()


def downscale_majority_3x(grid: Grid) -> Grid:
    """Downscale by 3x, taking majority vote including zeros."""
    a = to_np(grid)
    h, w = a.shape
    nh, nw = h // 3, w // 3
    if nh == 0 or nw == 0:
        return grid
    result = np.zeros((nh, nw), dtype=np.int32)
    for r in range(nh):
        for c in range(nw):
            block = a[r*3:r*3+3, c*3:c*3+3].flatten().tolist()
            result[r, c] = max(set(block), key=block.count)
    return result.tolist()


# --- Conditional per-object operations ---

def recolor_objects_by_neighbor_count(grid: Grid) -> Grid:
    """Recolor each connected component based on how many distinct neighbor colors it touches."""
    arr = to_np(grid)
    h, w = arr.shape
    if h == 0 or w == 0:
        return grid

    # Find connected components
    visited = np.zeros((h, w), dtype=bool)
    components = []

    def bfs(start_r, start_c):
        color = arr[start_r, start_c]
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        cells = []
        while queue:
            r, c = queue.pop(0)
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return cells, color

    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and not visited[r, c]:
                cells, color = bfs(r, c)
                components.append((cells, color))

    result = arr.copy()
    for cells, color in components:
        # Count distinct neighbor colors
        neighbor_colors = set()
        for r, c in cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in set(cells):
                    v = arr[nr, nc]
                    if v != color:
                        neighbor_colors.add(v)
        # Recolor: new color = number of distinct neighbor colors (1-indexed)
        new_c = len(neighbor_colors)
        if new_c > 0 and new_c <= 9:
            for r, c in cells:
                result[r, c] = new_c
    return from_np(result)


def fill_convex_hull(grid: Grid) -> Grid:
    """Fill the convex hull of all non-zero pixels with the most common color."""
    arr = to_np(grid)
    h, w = arr.shape
    nz = list(zip(*np.where(arr != 0)))
    if len(nz) < 3:
        return [row[:] for row in grid]

    mc = _most_common_overall(grid)
    if mc == 0:
        mc = 1

    result = arr.copy()

    # Simple convex hull via bounding rows
    rows_with_nz = {}
    for r, c in nz:
        if r not in rows_with_nz:
            rows_with_nz[r] = (c, c)
        else:
            rows_with_nz[r] = (min(rows_with_nz[r][0], c), max(rows_with_nz[r][1], c))

    min_r = min(rows_with_nz.keys())
    max_r = max(rows_with_nz.keys())

    for r in range(min_r, max_r + 1):
        if r in rows_with_nz:
            c_min, c_max = rows_with_nz[r]
        else:
            # Interpolate from neighbors
            above = max(rr for rr in rows_with_nz if rr < r) if any(rr < r for rr in rows_with_nz) else min_r
            below = min(rr for rr in rows_with_nz if rr > r) if any(rr > r for rr in rows_with_nz) else max_r
            if above in rows_with_nz and below in rows_with_nz:
                t = (r - above) / max(1, below - above)
                c_min = int(rows_with_nz[above][0] + t * (rows_with_nz[below][0] - rows_with_nz[above][0]))
                c_max = int(rows_with_nz[above][1] + t * (rows_with_nz[below][1] - rows_with_nz[above][1]))
            else:
                continue
        for c in range(c_min, c_max + 1):
            if result[r, c] == 0:
                result[r, c] = mc
    return from_np(result)


# =============================================================================
# Batch 5: Targeted primitives from near-miss analysis
# =============================================================================


def complete_sym_180(grid: Grid) -> Grid:
    """Complete 180° rotational symmetry around the center of non-zero content.

    For each zero cell that has a non-zero 180°-symmetric counterpart
    (relative to the bounding box center of non-zero pixels), fill it
    with that counterpart's color. Also applies H and V mirror symmetry
    iteratively until stable.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Find bounding box of non-zero content
    nz_rows = [r for r in range(h) if any(grid[r][c] != 0 for c in range(w))]
    nz_cols = [c for c in range(w) if any(grid[r][c] != 0 for r in range(h))]
    if not nz_rows or not nz_cols:
        return grid

    r0, r1 = min(nz_rows), max(nz_rows)
    c0, c1 = min(nz_cols), max(nz_cols)

    result = [row[:] for row in grid]
    # Iterate: apply 180°, H-mirror, V-mirror until no changes
    for _ in range(3):
        changed = False
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if result[r][c] != 0:
                    continue
                # 180° rotation
                sr, sc = r0 + r1 - r, c0 + c1 - c
                if r0 <= sr <= r1 and c0 <= sc <= c1 and result[sr][sc] != 0:
                    result[r][c] = result[sr][sc]
                    changed = True
                    continue
                # H-mirror (flip across horizontal center)
                sr2 = r0 + r1 - r
                if r0 <= sr2 <= r1 and result[sr2][c] != 0:
                    result[r][c] = result[sr2][c]
                    changed = True
                    continue
                # V-mirror (flip across vertical center)
                sc2 = c0 + c1 - c
                if c0 <= sc2 <= c1 and result[r][sc2] != 0:
                    result[r][c] = result[r][sc2]
                    changed = True
        if not changed:
            break
    return result


def complete_sym_90(grid: Grid) -> Grid:
    """Complete 4-fold (90°) rotational symmetry around the bounding box center.

    For each zero cell, checks if any of its 90°, 180°, or 270° rotated
    counterparts (relative to the bounding box center) are non-zero, and
    fills it with that value. Iterates until stable.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    nz_rows = [r for r in range(h) if any(grid[r][c] != 0 for c in range(w))]
    nz_cols = [c for c in range(w) if any(grid[r][c] != 0 for r in range(h))]
    if not nz_rows or not nz_cols:
        return grid

    r0, r1 = min(nz_rows), max(nz_rows)
    c0, c1 = min(nz_cols), max(nz_cols)
    cr2 = r0 + r1  # 2x center row (avoid float)
    cc2 = c0 + c1  # 2x center col

    result = [row[:] for row in grid]
    for _ in range(4):
        changed = False
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if result[r][c] != 0:
                    continue
                # Try 90°, 180°, 270° rotations
                # 90° CW: (r,c) -> (c, cr2+cc2-r) relative to center
                # In terms of grid coords:
                dr, dc = 2 * r - cr2, 2 * c - cc2
                rotations = [
                    (-dc + cr2, dr + cc2),   # 90° CW
                    (-dr + cr2, -dc + cc2),  # 180°
                    (dc + cr2, -dr + cc2),   # 270° CW
                ]
                for sr2, sc2 in rotations:
                    if sr2 % 2 != 0 or sc2 % 2 != 0:
                        continue
                    sr, sc = sr2 // 2, sc2 // 2
                    if r0 <= sr <= r1 and c0 <= sc <= c1 and result[sr][sc] != 0:
                        result[r][c] = result[sr][sc]
                        changed = True
                        break
        if not changed:
            break
    return result


def remove_small_components(grid: Grid, max_size: int = 2) -> Grid:
    """Remove connected components with size <= max_size (set to 0).

    Uses 4-connectivity. Removes noise clusters larger than single pixels
    (which remove_color_noise handles) but smaller than real objects.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    result = [row[:] for row in grid]

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in visited:
                continue
            # BFS to find component
            color = grid[r][c]
            component: list[tuple[int, int]] = []
            stack = [(r, c)]
            visited.add((r, c))
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and (nr, nc) not in visited
                            and grid[nr][nc] == color):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            if len(component) <= max_size:
                for cr, cc in component:
                    result[cr][cc] = 0
    return result


def remove_components_lte3(grid: Grid) -> Grid:
    """Remove connected components with size <= 3."""
    return remove_small_components(grid, max_size=3)


def keep_solid_rectangle(grid: Grid) -> Grid:
    """Keep only the largest solid monochromatic rectangle, zero the rest.

    Finds all axis-aligned rectangles of uniform non-zero color and keeps
    only the largest one. Useful for extracting signal from noise.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    best_rect = None
    best_area = 0

    # For each possible top-left corner and color
    for r0 in range(h):
        for c0 in range(w):
            if grid[r0][c0] == 0:
                continue
            color = grid[r0][c0]
            # Extend right as far as possible
            max_c = c0
            while max_c + 1 < w and grid[r0][max_c + 1] == color:
                max_c += 1
            # Extend down, narrowing width as needed
            for r1 in range(r0, h):
                if grid[r1][c0] != color:
                    break
                # Find rightmost extent in this row
                row_max = c0
                while row_max + 1 <= max_c and grid[r1][row_max + 1] == color:
                    row_max += 1
                max_c = row_max
                area = (r1 - r0 + 1) * (max_c - c0 + 1)
                if area > best_area:
                    best_area = area
                    best_rect = (r0, c0, r1, max_c, color)

    if best_rect is None:
        return grid

    r0, c0, r1, c1, color = best_rect
    result = [[0] * w for _ in range(h)]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            result[r][c] = color
    return result


def propagate_in_subcells(grid: Grid) -> Grid:
    """Detect sub-grid structure and propagate markers across sub-cells.

    Identifies separator lines (full rows/cols of 0) that divide the grid
    into a regular arrangement of sub-cells. In each sub-cell, finds
    the dominant (most common) non-zero color and treats other non-zero
    colors as markers. Propagates markers across sub-cells in the same
    sub-grid row and column.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Find separator rows (all 0) and separator cols (all 0)
    sep_rows = [r for r in range(h) if all(grid[r][c] == 0 for c in range(w))]
    sep_cols = [c for c in range(w) if all(grid[r][c] == 0 for r in range(h))]

    if not sep_rows or not sep_cols:
        return grid

    # Find cell boundaries
    def _find_ranges(seps, total):
        ranges = []
        prev = 0
        for s in seps:
            if s > prev:
                ranges.append((prev, s))
            prev = s + 1
        if prev < total:
            ranges.append((prev, total))
        return ranges

    row_ranges = _find_ranges(sep_rows, h)
    col_ranges = _find_ranges(sep_cols, w)

    if len(row_ranges) < 2 or len(col_ranges) < 2:
        return grid

    # Check all sub-cells have the same size
    cell_h = row_ranges[0][1] - row_ranges[0][0]
    cell_w = col_ranges[0][1] - col_ranges[0][0]
    if not all(r1 - r0 == cell_h for r0, r1 in row_ranges):
        return grid
    if not all(c1 - c0 == cell_w for c0, c1 in col_ranges):
        return grid

    n_rows = len(row_ranges)
    n_cols = len(col_ranges)

    # Extract sub-cell content
    cells = [[None] * n_cols for _ in range(n_rows)]
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            cell = []
            for r in range(r0, r1):
                row = []
                for c in range(c0, c1):
                    row.append(grid[r][c])
                cell.append(row)
            cells[ri][ci] = cell

    # Find the dominant non-zero color across all sub-cells
    from collections import Counter
    all_nz = [v for ri in range(n_rows) for ci in range(n_cols)
              for row in cells[ri][ci] for v in row if v != 0]
    if not all_nz:
        return grid
    dominant = Counter(all_nz).most_common(1)[0][0]

    # A marker is any non-zero, non-dominant pixel in a sub-cell
    def _get_markers(cell):
        markers = []
        for dr in range(cell_h):
            for dc in range(cell_w):
                if cell[dr][dc] != 0 and cell[dr][dc] != dominant:
                    markers.append((dr, dc, cell[dr][dc]))
        return markers

    result = [row[:] for row in grid]

    # Propagate: if a cell has markers, copy those markers to all cells
    # in the same sub-grid row
    for ri in range(n_rows):
        for ci in range(n_cols):
            markers = _get_markers(cells[ri][ci])
            if not markers:
                continue
            for ci2 in range(n_cols):
                if ci2 == ci:
                    continue
                r0, _ = row_ranges[ri]
                c0, _ = col_ranges[ci2]
                for dr, dc, color in markers:
                    result[r0 + dr][c0 + dc] = color

    return result


def mark_intersections_exclude_axis(grid: Grid) -> Grid:
    """Mark row/col intersections with color 2, excluding the axes' own crossing.

    Finds non-bg pixels, splits them into a "row group" (sharing one row)
    and a "col group" (sharing one column). Fills bg cells at cross-product
    intersections, but skips the cell where the row-axis and col-axis cross.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if not non_bg:
        return grid

    # Find rows/cols that contain non-bg pixels
    row_counts = Counter(r for r, c in non_bg)
    col_counts = Counter(c for r, c in non_bg)

    # Identify "header" rows (rows with many non-bg pixels) and
    # "side" cols (cols with many non-bg pixels)
    header_rows = {r for r, cnt in row_counts.items() if cnt >= 2}
    side_cols = {c for c, cnt in col_counts.items() if cnt >= 2}

    # Get the columns from header rows and rows from side cols
    header_cols = {c for r, c in non_bg if r in header_rows}
    side_rows = {r for r, c in non_bg if c in side_cols}

    result = [row[:] for row in grid]
    for r in side_rows:
        for c in header_cols:
            # Skip positions where both axes overlap (the crossing point)
            if r in header_rows and c in side_cols:
                continue
            if grid[r][c] == bg:
                result[r][c] = 2
    return result


def flood_fill_enclosed_with_accent(grid: Grid) -> Grid:
    """Fill enclosed 0-regions with the accent (2nd most common) color.

    Finds 0-cells not reachable from the grid border (treating non-zero as
    walls). Fills those with the second most common non-zero color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Find border-reachable zeros
    border_reachable: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if (r, c) in border_reachable:
            continue
        border_reachable.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and (nr, nc) not in border_reachable:
                stack.append((nr, nc))

    # Find accent color (2nd most common non-zero)
    from collections import Counter
    color_counts = Counter(grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0)
    if len(color_counts) < 2:
        # If only 1 color, use it
        fill_color = color_counts.most_common(1)[0][0] if color_counts else 0
    else:
        fill_color = color_counts.most_common(2)[1][0]

    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in border_reachable:
                result[r][c] = fill_color
    return result


def draw_diagonal_nearest(grid: Grid) -> Grid:
    """Draw diagonal lines from non-zero pixels, nearest pixel wins ties.

    Unlike draw_diagonal_from_pixels which processes in arbitrary order,
    this uses BFS-like propagation so closer sources always win.
    """
    arr = to_np(grid)
    h, w = arr.shape
    dist = np.full((h, w), np.inf)
    return from_np(_jit_draw_diagonal_nearest(arr, arr.copy(), dist))


def keep_minority_color_only(grid: Grid) -> Grid:
    """Keep only objects of the least common non-zero color, zero rest."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    h, w = len(grid), len(grid[0])
    color_counts = Counter(grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0)
    if not color_counts:
        return [row[:] for row in grid]
    minority_color = color_counts.most_common()[-1][0]
    return [[grid[r][c] if grid[r][c] == minority_color else 0
             for c in range(w)] for r in range(h)]


def keep_majority_color_only(grid: Grid) -> Grid:
    """Keep only objects of the most common non-zero color, zero rest."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    h, w = len(grid), len(grid[0])
    color_counts = Counter(grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0)
    if not color_counts:
        return [row[:] for row in grid]
    majority_color = color_counts.most_common(1)[0][0]
    return [[grid[r][c] if grid[r][c] == majority_color else 0
             for c in range(w)] for r in range(h)]


def gravity_to_nearest_pixel(grid: Grid) -> Grid:
    """Move each isolated pixel toward its nearest same-color pixel."""
    arr = to_np(grid)
    h, w = arr.shape
    result = np.zeros_like(arr)

    # Find connected components
    comps = _find_connected_components(grid)
    placed = set()

    for comp in comps:
        pixels = comp["pixels"]
        color = comp["color"]
        if len(pixels) > 1:
            # Keep multi-pixel objects in place
            for r, c in pixels:
                result[r, c] = color
                placed.add((r, c))
        else:
            # Single pixel: find nearest same-color component
            r0, c0 = next(iter(pixels))
            best_dist = float("inf")
            best_target = (r0, c0)
            for other in comps:
                if other is comp or other["color"] != color:
                    continue
                for r2, c2 in other["pixels"]:
                    d = abs(r0 - r2) + abs(c0 - c2)
                    if d < best_dist:
                        best_dist = d
                        # Move toward this target
                        best_target = (r2, c2)
            if best_dist < float("inf"):
                # Move one step toward target
                tr, tc = best_target
                dr = (1 if tr > r0 else -1 if tr < r0 else 0)
                dc = (1 if tc > c0 else -1 if tc < c0 else 0)
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in placed:
                    result[nr, nc] = color
                    placed.add((nr, nc))
                else:
                    result[r0, c0] = color
                    placed.add((r0, c0))
            else:
                result[r0, c0] = color
                placed.add((r0, c0))
    return from_np(result)


def recolor_by_enclosed_count(grid: Grid) -> Grid:
    """Recolor each object based on how many cells it encloses.

    Objects enclosing 0 cells get recolored to color 2, those enclosing
    nonzero cells keep their color. Useful for classification tasks.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    comps = _find_connected_components(grid)
    result = [row[:] for row in grid]

    for comp in comps:
        pixels = set(comp["pixels"])
        if len(pixels) < 4:
            continue
        # Find bounding box
        rs = [r for r, c in pixels]
        cs = [c for r, c in pixels]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        # Count enclosed zeros (zeros inside bbox not reachable from outside)
        bbox_zeros = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] == 0 and (r, c) not in pixels:
                    bbox_zeros.append((r, c))
        if bbox_zeros:
            # This object encloses some zeros - recolor to 2
            for r, c in comp["pixels"]:
                result[r][c] = 2
    return result


def shift_nonzero_to_gravity_center(grid: Grid) -> Grid:
    """Move all non-zero pixels 1 step toward the grid center."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    cr, cc = h / 2, w / 2
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                dr = (1 if r < cr - 0.5 else -1 if r > cr + 0.5 else 0)
                dc = (1 if c < cc - 0.5 else -1 if c > cc + 0.5 else 0)
                nr, nc = r + dr, c + dc
                nr = max(0, min(h - 1, nr))
                nc = max(0, min(w - 1, nc))
                if result[nr][nc] == 0:
                    result[nr][nc] = grid[r][c]
                else:
                    result[r][c] = grid[r][c]  # keep in place if blocked
    return result


def fill_zero_regions_with_neighbor_color(grid: Grid) -> Grid:
    """Fill each enclosed 0-region with the color that borders it most."""
    if not grid or not grid[0]:
        return grid
    from collections import deque, Counter
    h, w = len(grid), len(grid[0])

    # Find connected 0-regions
    visited: set[tuple[int, int]] = set()
    result = [row[:] for row in grid]

    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] != 0 or (sr, sc) in visited:
                continue
            # BFS to find this 0-region
            region: list[tuple[int, int]] = []
            border_colors: list[int] = []
            touches_edge = False
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            while queue:
                r, c = queue.popleft()
                region.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_edge = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr][nc] == 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                        elif grid[nr][nc] != 0:
                            border_colors.append(grid[nr][nc])

            # Only fill enclosed regions (not touching edge)
            if not touches_edge and border_colors:
                fill_c = Counter(border_colors).most_common(1)[0][0]
                for r, c in region:
                    result[r][c] = fill_c
    return result


# =============================================================================
# Batch 6: Targeted near-miss improvements
# =============================================================================

def move_objects_toward_each_other(grid: Grid) -> Grid:
    """Move objects toward the nearest other object (gravity between objects).

    Find connected components, then move each toward its nearest neighbor
    by 1 step. Useful for tasks where objects need to be adjacent.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, int(arr[cr, cc])))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 2:
        return grid

    # For each object, find direction to nearest other object and shift by 1
    result = np.zeros_like(arr)
    for i, obj in enumerate(objects):
        # Object center of mass
        cr = sum(r for r, c, v in obj) / len(obj)
        cc = sum(c for r, c, v in obj) / len(obj)

        # Find nearest other object's center
        min_dist = float('inf')
        best_dr, best_dc = 0, 0
        for j, other in enumerate(objects):
            if i == j:
                continue
            or_ = sum(r for r, c, v in other) / len(other)
            oc = sum(c for r, c, v in other) / len(other)
            dist = abs(cr - or_) + abs(cc - oc)
            if dist < min_dist:
                min_dist = dist
                dr = 1 if or_ > cr else (-1 if or_ < cr else 0)
                dc = 1 if oc > cc else (-1 if oc < cc else 0)
                best_dr, best_dc = dr, dc

        # Shift object by 1 step toward nearest
        for r, c, v in obj:
            nr, nc = r + best_dr, c + best_dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = v
            else:
                result[r, c] = v  # keep in place if out of bounds

    return from_np(result)


def stamp_pattern_at_markers(grid: Grid) -> Grid:
    """Find a small pattern/template and stamp it at marker positions.

    Identifies the largest connected region as the template and isolated
    single pixels as markers. Places the template centered at each marker.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, int(arr[cr, cc])))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 2:
        return grid

    # Find template (largest object) and markers (single pixels)
    objects.sort(key=len, reverse=True)
    template = objects[0]
    markers = [(obj[0][0], obj[0][1], obj[0][2]) for obj in objects[1:] if len(obj) == 1]

    if not markers:
        return grid

    # Template relative to its center
    tr = sum(r for r, c, v in template) / len(template)
    tc = sum(c for r, c, v in template) / len(template)
    rel_template = [(r - tr, c - tc, v) for r, c, v in template]

    # Stamp at each marker position
    result = arr.copy()
    for mr, mc, mv in markers:
        for dr, dc, v in rel_template:
            nr, nc = int(round(mr + dr)), int(round(mc + dc))
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = v

    return from_np(result)


def fill_between_diagonal(grid: Grid) -> Grid:
    """Fill diagonal lines between same-colored pixels.

    For each pair of same-colored pixels on the same diagonal, fill between them.
    """
    arr = to_np(grid)
    h, w = arr.shape
    if h == 0 or w == 0:
        return grid
    return from_np(_jit_fill_between_diagonal(arr, arr.copy()))


def complete_border_pattern(grid: Grid) -> Grid:
    """Complete a partially filled border pattern.

    If the grid has non-zero pixels on some border positions,
    extend the pattern to fill the entire border consistently.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid

    result = [row[:] for row in grid]

    # Collect border pixels
    border_colors = []
    for c in range(w):
        border_colors.append(grid[0][c])
        border_colors.append(grid[h-1][c])
    for r in range(1, h-1):
        border_colors.append(grid[r][0])
        border_colors.append(grid[r][w-1])

    non_zero = [c for c in border_colors if c != 0]
    if not non_zero:
        return grid

    # Most common border color
    fill = Counter(non_zero).most_common(1)[0][0]

    # Fill border with the dominant color where it's bg
    for c in range(w):
        if result[0][c] == 0:
            result[0][c] = fill
        if result[h-1][c] == 0:
            result[h-1][c] = fill
    for r in range(1, h-1):
        if result[r][0] == 0:
            result[r][0] = fill
        if result[r][w-1] == 0:
            result[r][w-1] = fill

    return result


def replicate_small_object_to_large(grid: Grid) -> Grid:
    """Find the smallest object and tile it to fill the largest object's bbox.

    Common pattern: a small 2x2 or 3x3 template is replicated to fill
    a larger rectangular region.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 2:
        return grid

    objects.sort(key=len)
    small = objects[0]
    large = objects[-1]

    # Extract small object as a sub-grid
    s_min_r = min(r for r, c in small)
    s_max_r = max(r for r, c in small)
    s_min_c = min(c for r, c in small)
    s_max_c = max(c for r, c in small)
    sh = s_max_r - s_min_r + 1
    sw = s_max_c - s_min_c + 1
    if sh == 0 or sw == 0:
        return grid

    # Large object bbox
    l_min_r = min(r for r, c in large)
    l_max_r = max(r for r, c in large)
    l_min_c = min(c for r, c in large)
    l_max_c = max(c for r, c in large)

    # Tile small into large bbox
    result = arr.copy()
    for r in range(l_min_r, l_max_r + 1):
        for c in range(l_min_c, l_max_c + 1):
            sr = s_min_r + (r - l_min_r) % sh
            sc = s_min_c + (c - l_min_c) % sw
            result[r, c] = arr[sr, sc]

    return from_np(result)


# =============================================================================
# Batch 7: Near-miss targeted primitives
# =============================================================================

def move_objects_to_contact(grid: Grid) -> Grid:
    """Move smaller objects toward the largest until they touch.

    The largest connected component stays anchored. All other objects
    slide toward it along the dominant axis until adjacent.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, int(arr[cr, cc])))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 2:
        return grid

    # Largest object is the anchor
    objects.sort(key=len, reverse=True)
    anchor = objects[0]
    anchor_cells = {(r, c) for r, c, v in anchor}
    anchor_cr = sum(r for r, c, v in anchor) / len(anchor)
    anchor_cc = sum(c for r, c, v in anchor) / len(anchor)

    result = np.zeros_like(arr)
    # Place anchor
    for r, c, v in anchor:
        result[r, c] = v

    # Move each non-anchor object toward anchor until contact
    for obj in objects[1:]:
        obj_cr = sum(r for r, c, v in obj) / len(obj)
        obj_cc = sum(c for r, c, v in obj) / len(obj)
        dr = 1 if anchor_cr > obj_cr else (-1 if anchor_cr < obj_cr else 0)
        dc = 1 if anchor_cc > obj_cc else (-1 if anchor_cc < obj_cc else 0)

        # Only move in the dominant direction
        if abs(anchor_cr - obj_cr) >= abs(anchor_cc - obj_cc):
            dc = 0
        else:
            dr = 0

        # Slide until would overlap with anchor or go OOB
        shift = 0
        for step in range(1, max(h, w)):
            would_overlap = False
            for r, c, v in obj:
                nr, nc = r + dr * step, c + dc * step
                if (nr, nc) in anchor_cells:
                    would_overlap = True
                    break
                if not (0 <= nr < h and 0 <= nc < w):
                    would_overlap = True
                    break
            if would_overlap:
                shift = step - 1
                break
            shift = step

        for r, c, v in obj:
            nr, nc = r + dr * shift, c + dc * shift
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = v

    return from_np(result)


def project_color_to_object_face(grid: Grid) -> Grid:
    """Project isolated pixels' colors onto the nearest face of the main object.

    Single-pixel markers shoot their color onto the closest edge cell
    of the largest connected component.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, int(arr[cr, cc])))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 2:
        return grid

    # Find the largest object — all non-largest pixels are potential markers
    objects.sort(key=len, reverse=True)
    main_obj = objects[0]
    main_cells = {(r, c) for r, c, v in main_obj}

    # Collect individual marker pixels from all non-main objects
    markers = []
    for obj in objects[1:]:
        for r, c, v in obj:
            markers.append((r, c, v))

    if not markers:
        return grid

    result = arr.copy()

    # Find edge cells of main object (cells adjacent to empty space)
    edge_cells = set()
    for r, c, v in main_obj:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in main_cells:
                edge_cells.add((r, c))
                break

    # For each marker pixel, find the nearest edge cell in same row or column
    for mr, mc, mv in markers:
        best_r, best_c = -1, -1
        best_dist = float('inf')
        for er, ec in edge_cells:
            # Prefer same row or same column alignment
            if er == mr or ec == mc:
                dist = abs(er - mr) + abs(ec - mc)
                if dist < best_dist:
                    best_dist = dist
                    best_r, best_c = er, ec
        if best_r >= 0:
            result[best_r, best_c] = mv

    return from_np(result)


def align_objects_vertically(grid: Grid) -> Grid:
    """Align all objects to the vertical center of the grid.

    Move each connected component so its vertical center matches
    the grid's vertical center, keeping horizontal positions.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in visited:
                obj = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, int(arr[cr, cc])))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and arr[nr, nc] != 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj)

    if len(objects) < 1:
        return grid

    result = np.zeros_like(arr)
    grid_center_r = h / 2.0

    for obj in objects:
        obj_center_r = sum(r for r, c, v in obj) / len(obj)
        obj_h = max(r for r, c, v in obj) - min(r for r, c, v in obj) + 1
        # Move to center vertically
        shift_r = int(round(grid_center_r - obj_center_r))

        for r, c, v in obj:
            nr = r + shift_r
            if 0 <= nr < h:
                result[nr, c] = v

    return from_np(result)


def fill_grid_cells_between_markers(grid: Grid) -> Grid:
    """In a grid divided by lines, fill cells between same-colored markers.

    Detects grid structure (rows of a repeating color), identifies colored
    markers in cells, and fills all cells between same-colored markers
    in the same row with that color.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Detect grid lines: find the most common non-zero color on rows/cols
    # that span the full width or height
    line_color = 0
    for c_val in range(1, 10):
        # Check if any full row is this color
        for r in range(h):
            if all(arr[r, c] == c_val for c in range(w)):
                line_color = c_val
                break
        if line_color:
            break

    if line_color == 0:
        # Try full columns
        for c_val in range(1, 10):
            for c in range(w):
                if all(arr[r, c] == c_val for r in range(h)):
                    line_color = c_val
                    break
            if line_color:
                break

    if line_color == 0:
        return grid

    # Find grid line rows
    line_rows = [r for r in range(h) if all(arr[r, c] == line_color for c in range(w))]
    line_cols = [c for c in range(w) if all(arr[r, c] == line_color for r in range(h))]

    if not line_rows or not line_cols:
        return grid

    # Identify cell boundaries
    row_bands = []
    prev = 0
    for lr in line_rows:
        if lr > prev:
            row_bands.append((prev, lr))
        prev = lr + 1
    if prev < h:
        row_bands.append((prev, h))

    col_bands = []
    prev = 0
    for lc in line_cols:
        if lc > prev:
            col_bands.append((prev, lc))
        prev = lc + 1
    if prev < w:
        col_bands.append((prev, w))

    # For each cell, determine its color (majority non-zero, non-line color)
    cell_colors = {}  # (ri, ci) -> color
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            colors = []
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if arr[r, c] != 0 and arr[r, c] != line_color:
                        colors.append(int(arr[r, c]))
            if colors:
                cell_colors[(ri, ci)] = Counter(colors).most_common(1)[0][0]

    # Fill between same-colored cells in the same row
    result = arr.copy()
    for ri, (r0, r1) in enumerate(row_bands):
        by_color: dict[int, list[int]] = {}
        for ci in range(len(col_bands)):
            if (ri, ci) in cell_colors:
                color = cell_colors[(ri, ci)]
                if color not in by_color:
                    by_color[color] = []
                by_color[color].append(ci)

        for color, cols in by_color.items():
            if len(cols) < 2:
                continue
            min_ci, max_ci = min(cols), max(cols)
            for ci in range(min_ci, max_ci + 1):
                if (ri, ci) not in cell_colors:
                    c0, c1 = col_bands[ci]
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            if result[r, c] != line_color:
                                result[r, c] = color

    # Also fill between same-colored cells in the same column
    for ci, (c0, c1) in enumerate(col_bands):
        by_color2: dict[int, list[int]] = {}
        for ri in range(len(row_bands)):
            if (ri, ci) in cell_colors:
                color = cell_colors[(ri, ci)]
                if color not in by_color2:
                    by_color2[color] = []
                by_color2[color].append(ri)

        for color, rows in by_color2.items():
            if len(rows) < 2:
                continue
            min_ri, max_ri = min(rows), max(rows)
            for ri in range(min_ri, max_ri + 1):
                if (ri, ci) not in cell_colors:
                    r0, r1 = row_bands[ri]
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            if result[r, c] != line_color:
                                result[r, c] = color

    return from_np(result)


def absorb_noise_to_nearest_line(grid: Grid) -> Grid:
    """Move isolated pixels to extend the nearest same-color line.

    For each isolated pixel, find the nearest vertical/horizontal line
    of the same color. Extend that line by 1 pixel toward the isolated pixel.
    Remove the isolated pixel.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)

    # Find vertical lines (columns where most cells share a color)
    lines = []  # (orientation, position, color)
    for c in range(w):
        col = arr[:, c]
        non_zero = col[col != 0]
        if len(non_zero) >= h * 0.6:
            color = int(Counter(non_zero.tolist()).most_common(1)[0][0])
            if sum(1 for v in col if v == color) >= h * 0.6:
                lines.append(('v', c, color))

    for r in range(h):
        row = arr[r, :]
        non_zero = row[row != 0]
        if len(non_zero) >= w * 0.6:
            color = int(Counter(non_zero.tolist()).most_common(1)[0][0])
            if sum(1 for v in row if v == color) >= w * 0.6:
                lines.append(('h', r, color))

    if not lines:
        return grid

    # Find isolated pixels (not part of any line)
    line_cells = set()
    for orient, pos, color in lines:
        if orient == 'v':
            for r in range(h):
                if arr[r, pos] == color:
                    line_cells.add((r, pos))
        else:
            for c in range(w):
                if arr[pos, c] == color:
                    line_cells.add((pos, c))

    result = arr.copy()

    # Find isolated colored pixels not on lines
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0 and (r, c) not in line_cells:
                pixel_color = int(arr[r, c])
                # Find nearest same-color line
                best_line = None
                best_dist = float('inf')
                for orient, pos, lcolor in lines:
                    if lcolor != pixel_color:
                        continue
                    if orient == 'v':
                        dist = abs(c - pos)
                    else:
                        dist = abs(r - pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_line = (orient, pos, lcolor)

                # Remove the noise pixel regardless
                result[r, c] = 0
                if best_line:
                    orient, pos, lcolor = best_line
                    # Extend line by 1 toward the noise pixel
                    if orient == 'v':
                        dc = 1 if c > pos else -1
                        result[r, pos + dc] = lcolor
                    else:
                        dr = 1 if r > pos else -1
                        result[pos + dr, c] = lcolor

    return from_np(result)


def compact_shape(grid: Grid) -> Grid:
    """Make shapes more compact by closing diagonal gaps.

    For each connected component, fill in pixels that would make the
    bounding box more filled while preserving the shape's outline.
    Specifically, close 1-pixel diagonal gaps in contours.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid
    arr = to_np(grid)
    result = arr.copy()

    # For each non-zero pixel, check if it has diagonal neighbors
    # that share a color but no orthogonal path between them
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                color = arr[r, c]
                # Check diagonal neighbors
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == color:
                        # Check if there's NO orthogonal connection
                        mid1 = arr[r, nc] if 0 <= nc < w else -1
                        mid2 = arr[nr, c] if 0 <= nr < h else -1
                        if mid1 != color and mid2 != color:
                            # Fill one of the gap cells
                            if mid1 == 0 and 0 <= nc < w:
                                result[r, nc] = color
                            elif mid2 == 0 and 0 <= nr < h:
                                result[nr, c] = color

    return from_np(result)


# =============================================================================
# Build the ARC primitive registry
# =============================================================================

def _build_arc_primitives() -> list[Primitive]:
    """Build the complete list of ARC primitives."""
    prims = []

    # Arity-1 primitives (Grid -> Grid): the core set
    unary_ops = [
        ("identity",        identity),
        ("rotate_90_clockwise",         rotate_90_cw),
        ("rotate_90_counterclockwise",  rotate_90_ccw),
        ("rotate_180",                  rotate_180),
        ("mirror_horizontal",           mirror_horizontal),
        ("mirror_vertical",             mirror_vertical),
        ("transpose",       transpose),
        ("invert_colors",   invert_colors),
        ("crop_to_nonzero", crop_to_nonzero),
        ("top_half",        get_top_half),
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
        ("xor_halves_vertical",    xor_halves_v),
        ("or_halves_vertical",     or_halves_v),
        ("xor_halves_horizontal",  xor_halves_h),
        ("or_halves_horizontal",   or_halves_h),
        ("replace_background_most_common",   replace_bg_with_most_common),
        # --- New spatial/object primitives ---
        ("extract_largest_object",     extract_largest_object),
        ("extract_smallest_object",    extract_smallest_object),
        ("anti_diagonal_mirror",       anti_diagonal_mirror),
        ("make_symmetric_horizontal",  make_symmetric_h),
        ("make_symmetric_vertical",    make_symmetric_v),
        ("repeat_pattern_right",       repeat_pattern_right),
        ("repeat_pattern_down",        repeat_pattern_down),
        ("add_border",          add_border),
        ("remove_border",       remove_border),
        ("sort_rows_by_color_count",    sort_rows_by_color_count),
        ("unique_rows",         unique_rows),
        ("unique_columns",              unique_cols),
        ("recolor_by_size_rank",        recolor_by_size_rank),
        ("extend_lines_horizontal",     extend_lines_h),
        ("extend_lines_vertical",       extend_lines_v),
        # --- Connected component / object-level primitives ---
        ("keep_largest_object_only",    keep_largest_object_only),
        ("keep_smallest_object_only",   keep_smallest_object_only),
        ("remove_largest_object",       remove_largest_object),
        ("remove_smallest_object",      remove_smallest_object),
        ("count_objects_as_grid",       count_objects_as_grid),
        ("mirror_objects_horizontal",   mirror_objects_h),
        ("mirror_objects_vertical",     mirror_objects_v),
        # --- Grid partitioning ---
        ("extract_top_left_cell",     extract_top_left_cell),
        ("extract_bottom_right_cell", extract_bottom_right_cell),
        ("remove_grid_lines",   remove_grid_lines),
        # --- Diagonal / line extension ---
        ("extend_lines",        extend_lines),
        ("extend_diagonal_lines",    extend_diagonal_lines),
        # --- Color/pattern ops ---
        ("binarize",            binarize),
        ("color_to_most_common", color_to_most_common),
        ("upscale_pattern",     upscale_pattern),
        # --- Near-miss targeted: anomaly removal, rectangle ops ---
        ("denoise_majority",    denoise_majority),
        ("fill_rectangles",     fill_rectangles),
        ("extract_minority_color",  extract_minority_color),
        ("extract_majority_color",  extract_majority_color),
        ("replace_noise_in_objects",  replace_noise_in_objects),
        ("hollow_objects",      hollow_objects),
        # --- Cyclic shifts ---
        ("shift_down_1",        shift_down_1),
        ("shift_up_1",          shift_up_1),
        ("shift_left_1",        shift_left_1),
        ("shift_right_1",       shift_right_1),
        # --- Symmetry completion ---
        ("complete_symmetry_horizontal",      complete_symmetry_h),
        ("complete_symmetry_vertical",      complete_symmetry_v),
        # --- Split-by-separator ---
        ("overlay_split_halves_horizontal",     overlay_split_halves_h),
        ("overlay_split_halves_vertical",     overlay_split_halves_v),
        # --- Morphological ops ---
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
        ("deduplicate_columns",     deduplicate_cols),
        ("stack_with_mirror_vertical",   stack_with_mirror_v),
        ("stack_with_mirror_horizontal", stack_with_mirror_h),
        ("mirror_diagonal_main",        mirror_diagonal_main),
        ("grid_difference",         grid_difference),
        ("grid_difference_horizontal",  grid_difference_h),
        ("and_halves_horizontal",       and_halves_h),
        ("swap_most_least",         swap_most_least),
        ("extract_repeating_tile",      extract_repeating_tile),
        ("extract_bottom_right_block",  extract_bottom_right_block),
        ("compress_columns",        compress_cols),
        ("flatten_to_row",          flatten_to_row),
        ("flatten_to_column",           flatten_to_column),
        ("mode_color_per_row",          mode_color_per_row),
        ("mode_color_per_column",       mode_color_per_col),
        ("extend_to_border_horizontal", extend_to_border_h),
        ("extend_to_border_vertical",   extend_to_border_v),
        ("spread_in_lanes_horizontal",  spread_in_lanes_h),
        ("spread_in_lanes_vertical",    spread_in_lanes_v),
        ("complete_pattern_4way",       complete_pattern_4way),
        ("complete_symmetry_diagonal",  complete_symmetry_diagonal),
        ("mirror_horizontal_merge",  mirror_h_merge),
        ("mirror_vertical_merge",   mirror_v_merge),
        ("sort_columns_by_value",       sort_cols_by_value),
        ("fill_row_from_right",         fill_row_from_right),
        ("propagate_color_horizontal",  propagate_color_h),
        ("propagate_color_vertical",    propagate_color_v),
        ("fill_stripe_gaps_horizontal",  fill_stripe_gaps_h),
        ("fill_stripe_gaps_vertical",   fill_stripe_gaps_v),
        ("recolor_minority_in_rows",        recolor_minority_in_rows),
        ("recolor_minority_in_columns",     recolor_minority_in_cols),
        ("remove_color_noise",              remove_color_noise),
        ("recolor_isolated_to_nearest",     recolor_isolated_to_nearest),
        ("remove_border_objects",           remove_border_objects),
        ("fill_object_bounding_boxes",      fill_object_bboxes),
        ("recolor_by_nearest_border",       recolor_by_nearest_border),
        ("project_markers_to_block",        project_markers_to_block),
        ("fill_grid_intersections",         fill_grid_intersections),
        ("fill_diagonal_stripes",       fill_diagonal_stripes),
        ("keep_border_only",            keep_border_only),
        # --- Port batch 2: agi-mvp-general ---
        ("connect_pixels_to_rectangle", connect_pixels_to_rect),
        ("gravity_toward_color",    gravity_toward_color),
        ("recolor_2nd_to_3rd_most_common",  recolor_2nd_to_3rd),
        ("recolor_least_to_2nd_least",     recolor_least_to_2nd_least),
        ("swap_most_and_2nd_most_common",  swap_most_and_2nd_color),
        ("keep_unique_rows",        keep_unique_rows),
        ("extend_lines_to_contact",     extend_lines_to_contact),
        ("recolor_2nd_to_dominant",     recolor_2nd_to_dominant),
        ("erase_2nd_color",         erase_2nd_color),
        ("fill_enclosed_by_dominant",  recolor_bg_enclosed_by_dominant),
        # --- Port batch 3: new high-value primitives ---
        ("color_by_column_position",        color_by_col_position),
        ("extend_color_within_row_bounds",  extend_color_within_row_bounds),
        ("extend_color_within_column_bounds", extend_color_within_col_bounds),
        ("fill_rooms_with_new_color",   fill_rooms_with_new_color),
        ("downscale_2x",            downscale_2x),
        ("downscale_3x",            downscale_3x),
        # --- Port batch 4: grid partition + annotation ---
        ("select_odd_one_out",              select_odd_one_out),
        ("overlay_grid_cells",              overlay_grid_cells),
        ("majority_vote_cells",             majority_vote_cells),
        ("select_most_colorful_cell",       select_most_colorful_cell),
        ("select_most_filled_cell",         select_most_filled_cell),
        ("surround_pixels_3x3",             surround_pixels_3x3),
        ("draw_cross_from_pixels",          draw_cross_from_pixels),
        ("draw_cross_to_contact",           draw_cross_to_contact),
        ("draw_diagonal_from_pixels",       draw_diagonal_from_pixels),
        ("connect_same_color_horizontal",   connect_same_color_h),
        ("connect_same_color_vertical",     connect_same_color_v),
        ("downscale_4x",            downscale_4x),
        ("downscale_5x",            downscale_5x),
        ("downscale_7x",            downscale_7x),
        ("downscale_majority_2x",       downscale_majority_2x),
        ("downscale_majority_3x",       downscale_majority_3x),
        ("fill_convex_hull",        fill_convex_hull),
        ("draw_rectangle_around_objects",   draw_rect_around_objects),
        # --- Batch 5: targeted near-miss improvements ---
        ("complete_symmetry_90",                complete_sym_90),
        ("remove_small_components_2",           remove_small_components),
        ("remove_small_components_3",           remove_components_lte3),
        ("keep_solid_rectangle",                keep_solid_rectangle),
        ("mark_intersections_exclude_axis",     mark_intersections_exclude_axis),
        ("flood_fill_enclosed_with_accent",     flood_fill_enclosed_with_accent),
        ("draw_diagonal_nearest",               draw_diagonal_nearest),
        ("gravity_to_nearest_pixel",            gravity_to_nearest_pixel),
        ("shift_nonzero_to_gravity_center",     shift_nonzero_to_gravity_center),
        # --- Batch 6: targeted near-miss improvements ---
        ("stamp_pattern_at_markers",       stamp_pattern_at_markers),
        ("fill_between_diagonal",          fill_between_diagonal),
        ("complete_border_pattern", complete_border_pattern),
        ("replicate_small_object_to_large", replicate_small_object_to_large),
        # --- Batch 7: near-miss targeted primitives ---
        ("move_objects_to_contact",         move_objects_to_contact),
        ("project_color_to_object_face",    project_color_to_object_face),
        ("align_objects_vertically",        align_objects_vertically),
        ("fill_grid_cells_between_markers", fill_grid_cells_between_markers),
        ("absorb_noise_to_nearest_line",    absorb_noise_to_nearest_line),
        ("compact_shape",           compact_shape),
        # --- Batch 8: context-dependent color primitives (Decision 59) ---
        ("recolor_by_4_neighbor_vote",      recolor_by_neighbor_vote),
        ("recolor_by_8_neighbor_vote",      recolor_by_8neighbor_vote),
        ("swap_two_least_common_colors",    swap_two_least_common),
        ("fill_by_surround_color",          fill_by_surround_color),
        ("recolor_minority_to_majority",    recolor_minority_to_majority),
    ]

    for name, fn in unary_ops:
        prims.append(Primitive(name=name, arity=1, fn=fn, domain="arc"))

    # NOTE: Parameterized color primitives (keep_cN, erase_N, fill_bg_N,
    # swap_A_B, etc.) are NO LONGER built statically. They are generated
    # per-task in build_task_color_primitives() based on the colors actually
    # present in the task's training examples. This reduces the search space
    # from ~349 to ~150-200 primitives per task.

    # Arity-2 primitives (compose two transforms): overlay is the main one
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))

    return prims


def build_task_color_primitives(task_colors: set[int]) -> list[Primitive]:
    """Generate color-parameterized primitives for a specific task's palette.

    Instead of pre-building primitives for all 9 colors × all operations
    (~120 primitives), only instantiates those relevant to the current task.
    A typical task uses 3-5 colors, yielding ~20-40 color primitives.

    Args:
        task_colors: set of color values (0-9) appearing in the task's
                     training inputs and outputs.

    Returns:
        List of task-specific color primitives.
    """
    prims: list[Primitive] = []
    # Non-zero colors in the task
    colors = sorted(task_colors - {0})

    # Keep only pixels of a specific color, zero everything else
    for c in colors:
        prims.append(Primitive(
            name=f"keep_only_color_{c}", arity=1,
            fn=_make_keep_color(c), domain="arc",
        ))

    # Erase a specific color (replace with background 0)
    for c in colors:
        prims.append(Primitive(
            name=f"erase_color_{c}", arity=1,
            fn=_make_erase_color(c), domain="arc",
        ))

    # Fill background (0) with a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"fill_background_with_color_{c}", arity=1,
            fn=_make_fill_bg(c), domain="arc",
        ))

    # Recolor all non-zero pixels to a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"recolor_all_to_color_{c}", arity=1,
            fn=_make_recolor_nonzero(c), domain="arc",
        ))

    # Replace a specific color with background (0)
    for c in colors:
        prims.append(Primitive(
            name=f"replace_color_{c}_with_background", arity=1,
            fn=_make_replace_color(c, 0), domain="arc",
        ))

    # Swap two colors symmetrically (both directions at once)
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            prims.append(Primitive(
                name=f"swap_color_{a}_and_color_{b}", arity=1,
                fn=_make_swap_colors(a, b), domain="arc",
            ))

    # Replace one color with another (one direction only)
    for a in colors:
        for b in colors:
            if a != b:
                prims.append(Primitive(
                    name=f"replace_color_{a}_with_color_{b}", arity=1,
                    fn=_make_replace_color(a, b), domain="arc",
                ))

    # Fill rectangle interiors with a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"fill_rectangle_interior_color_{c}", arity=1,
            fn=lambda g, color=c: _fill_rect_interiors(g, color), domain="arc",
        ))

    # Mark row/column intersections with a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"mark_intersections_color_{c}", arity=1,
            fn=lambda g, color=c: _recolor_cells_at_intersections(g, color),
            domain="arc",
        ))

    # Recolor dominant-color pixels touching accent to a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"recolor_dominant_touching_accent_to_color_{c}", arity=1,
            fn=_make_recolor_dominant_touching_accent(c), domain="arc",
        ))

    # Fill the smallest rectangular hole with a specific color
    for c in colors:
        prims.append(Primitive(
            name=f"fill_smallest_hole_with_color_{c}", arity=1,
            fn=_make_fill_smallest_hole(c), domain="arc",
        ))

    # Recolor non-zero pixels inside an accent color's bounding box
    for accent in colors:
        for new in colors:
            if accent != new:
                prims.append(Primitive(
                    name=f"recolor_nonzero_in_color_{accent}_bbox_to_color_{new}",
                    arity=1,
                    fn=_make_recolor_nonzero_inside_bbox(accent, new),
                    domain="arc",
                ))

    # Fill background pixels adjacent to a target color with a fill color
    for target in colors:
        for fill in colors:
            if target != fill:
                prims.append(Primitive(
                    name=f"fill_adjacent_to_color_{target}_with_color_{fill}",
                    arity=1,
                    fn=_fill_bg_adjacent_to_color(target, fill), domain="arc",
                ))

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
# Fixed-point iteration combinator
# =============================================================================

def apply_until_stable(fn, grid: Grid, max_iters: int = 20) -> Grid:
    """Apply fn repeatedly until the grid stops changing (fixed point).

    Many ARC tasks need iterated application: fill propagation, growth,
    pattern completion. This applies fn up to max_iters times, stopping
    early when the output equals the input (convergence).
    """
    current = grid
    for _ in range(max_iters):
        try:
            result = fn(current)
            if not isinstance(result, list) or not result:
                return current
            if result == current:
                return current
            current = result
        except Exception:
            return current
    return current


def make_fixed_point_fn(fn):
    """Wrap a Grid→Grid function to apply until stable."""
    def fp(grid):
        return apply_until_stable(fn, grid)
    return fp


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
