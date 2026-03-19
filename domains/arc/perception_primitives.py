"""
Atomic perception primitives: Grid → Value.

Each extracts exactly ONE property from a grid. These are the "eyes"
that feed into parameterized action primitives.
"""

from __future__ import annotations

from collections import Counter

from core import Primitive


Grid = list[list[int]]


# =============================================================================
# Color perception
# =============================================================================

def background_color(grid: Grid) -> int:
    """Return the most common color (typically the background)."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    if not flat:
        return 0
    return Counter(flat).most_common(1)[0][0]


def dominant_color(grid: Grid) -> int:
    """Return the most common non-background color."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [c for c in flat if c != bg]
    if not non_bg:
        return bg
    return Counter(non_bg).most_common(1)[0][0]


def rarest_color(grid: Grid) -> int:
    """Return the least common non-background color."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [c for c in flat if c != bg]
    if not non_bg:
        return bg
    return Counter(non_bg).most_common()[-1][0]


def accent_color(grid: Grid) -> int:
    """Return the second most common non-background color."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = Counter(c for c in flat if c != bg)
    if not non_bg:
        return bg
    ranked = non_bg.most_common()
    return ranked[1][0] if len(ranked) > 1 else ranked[0][0]


def second_color(grid: Grid) -> int:
    """Return the second most common color overall."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    ranked = Counter(flat).most_common()
    return ranked[1][0] if len(ranked) > 1 else ranked[0][0]


def corner_color(grid: Grid) -> int:
    """Return the color of the top-left corner pixel."""
    if not grid or not grid[0]:
        return 0
    return grid[0][0]


# =============================================================================
# Counting perception
# =============================================================================

def n_colors(grid: Grid) -> int:
    """Return the number of distinct colors in the grid."""
    if not grid or not grid[0]:
        return 0
    return len(set(grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))))


def n_foreground_colors(grid: Grid) -> int:
    """Return the number of distinct non-background colors."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    return len(set(c for c in flat if c != bg))


# =============================================================================
# Geometry perception
# =============================================================================

def grid_height(grid: Grid) -> int:
    """Return the height (number of rows) of the grid."""
    return len(grid) if grid else 0


def grid_width(grid: Grid) -> int:
    """Return the width (number of columns) of the grid."""
    return len(grid[0]) if grid and grid[0] else 0


def grid_min_dim(grid: Grid) -> int:
    """Return the smaller dimension."""
    h = len(grid) if grid else 0
    w = len(grid[0]) if grid and grid[0] else 0
    return min(h, w) if h and w else 0


def grid_max_dim(grid: Grid) -> int:
    """Return the larger dimension."""
    h = len(grid) if grid else 0
    w = len(grid[0]) if grid and grid[0] else 0
    return max(h, w) if h and w else 0


# =============================================================================
# Object perception
# =============================================================================

def n_objects(grid: Grid) -> int:
    """Count connected components (4-connected, ignoring color 0)."""
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                count += 1
                # BFS flood fill
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
    return count


# =============================================================================
# Build functions
# =============================================================================

def build_perception_primitives() -> list[Primitive]:
    """Build perception primitives (Grid → Value)."""
    perceptions = [
        ("background_color",        background_color),
        ("dominant_color",          dominant_color),
        ("rarest_color",            rarest_color),
        ("accent_color",            accent_color),
        ("second_color",            second_color),
        ("corner_color",            corner_color),
        ("n_colors",                n_colors),
        ("n_foreground_colors",     n_foreground_colors),
        ("grid_height",             grid_height),
        ("grid_width",              grid_width),
        ("grid_min_dim",            grid_min_dim),
        ("grid_max_dim",            grid_max_dim),
        # Object perception — new Tier 1 (1)
        ("n_objects",               n_objects),
    ]
    return [
        Primitive(name=name, arity=0, fn=fn, domain="arc", kind="perception")
        for name, fn in perceptions
    ]
