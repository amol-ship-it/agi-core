"""
Atomic perception primitives: Grid → Value.

Each extracts exactly ONE property from a grid. These are the "eyes"
that feed into parameterized action primitives. They detect color roles,
object counts, and structural properties without transforming the grid.

Perception primitives compose with parameterized actions:
    swap_colors(background_color(grid), accent_color(grid))(grid)
    keep_color(dominant_color(grid))(grid)
"""

from __future__ import annotations

from collections import Counter

from core import Primitive


Grid = list[list[int]]


# =============================================================================
# Color perception: detect color roles
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
    """Return the second most common non-background color.

    If only one non-bg color exists, returns it. Useful for tasks
    where the "accent" or "marker" color differs from the dominant.
    """
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = Counter(c for c in flat if c != bg)
    if not non_bg:
        return bg
    ranked = non_bg.most_common()
    return ranked[1][0] if len(ranked) > 1 else ranked[0][0]


# =============================================================================
# Counting perception: detect quantities
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
# Dimension perception: detect grid geometry
# =============================================================================

def grid_height(grid: Grid) -> int:
    """Return the height (number of rows) of the grid."""
    return len(grid) if grid else 0


def grid_width(grid: Grid) -> int:
    """Return the width (number of columns) of the grid."""
    return len(grid[0]) if grid and grid[0] else 0


def grid_min_dim(grid: Grid) -> int:
    """Return the smaller dimension (min of height, width)."""
    h = len(grid) if grid else 0
    w = len(grid[0]) if grid and grid[0] else 0
    return min(h, w) if h and w else 0


# =============================================================================
# Object counting perception
# =============================================================================

def n_objects(grid: Grid) -> int:
    """Return the number of connected foreground components."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    h, w = len(grid), len(grid[0])
    visited = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                count += 1
                # BFS flood fill
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return count


# =============================================================================
# Structural perception: detect grid structure
# =============================================================================

def largest_object_color(grid: Grid) -> int:
    """Return the color of the largest connected foreground component."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    h, w = len(grid), len(grid[0])
    visited = set()
    largest_size = 0
    largest_color = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                color = grid[r][c]
                size = 0
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                if size > largest_size:
                    largest_size = size
                    largest_color = color
    return largest_color


def smallest_object_color(grid: Grid) -> int:
    """Return the color of the smallest connected foreground component."""
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    bg = Counter(flat).most_common(1)[0][0]
    h, w = len(grid), len(grid[0])
    visited = set()
    smallest_size = float('inf')
    smallest_color = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                color = grid[r][c]
                size = 0
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                if size < smallest_size:
                    smallest_size = size
                    smallest_color = color
    return smallest_color


# =============================================================================
# Spatial perception: detect grid spatial properties
# =============================================================================

def grid_max_dim(grid: Grid) -> int:
    """Return the larger dimension (max of height, width)."""
    h = len(grid) if grid else 0
    w = len(grid[0]) if grid and grid[0] else 0
    return max(h, w) if h and w else 0


def second_color(grid: Grid) -> int:
    """Return the second most common color overall.

    Useful for tasks where the grid has two dominant colors and
    you need to reference the non-majority one.
    """
    if not grid or not grid[0]:
        return 0
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    ranked = Counter(flat).most_common()
    return ranked[1][0] if len(ranked) > 1 else ranked[0][0]


def corner_color(grid: Grid) -> int:
    """Return the color of the top-left corner pixel.

    Many ARC tasks use corner pixels as reference colors.
    """
    if not grid or not grid[0]:
        return 0
    return grid[0][0]


def center_color(grid: Grid) -> int:
    """Return the color of the center pixel.

    For odd-dimensioned grids, this is the exact center.
    For even-dimensioned grids, uses floor(h/2), floor(w/2).
    """
    if not grid or not grid[0]:
        return 0
    return grid[len(grid) // 2][len(grid[0]) // 2]


def edge_color(grid: Grid) -> int:
    """Return the most common color on the grid border.

    Border = first/last row + first/last column. Useful for
    detecting frame colors in bordered grids.
    """
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    border = []
    for c in range(w):
        border.append(grid[0][c])
        if h > 1:
            border.append(grid[h - 1][c])
    for r in range(1, h - 1):
        border.append(grid[r][0])
        if w > 1:
            border.append(grid[r][w - 1])
    if not border:
        return 0
    return Counter(border).most_common(1)[0][0]


def interior_dominant_color(grid: Grid) -> int:
    """Return the most common non-background color in the interior.

    Interior = excluding first/last row and column. Useful for
    detecting the "content" color inside a bordered grid.
    """
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    if h <= 2 or w <= 2:
        return dominant_color(grid)
    flat = [grid[r][c] for r in range(1, h - 1) for c in range(1, w - 1)]
    if not flat:
        return 0
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [c for c in flat if c != bg]
    if not non_bg:
        return bg
    return Counter(non_bg).most_common(1)[0][0]


# =============================================================================
# Additional perception: spatial and structural
# =============================================================================

def has_horizontal_symmetry(grid: Grid) -> int:
    """Return 1 if the grid is horizontally symmetric, 0 otherwise."""
    if not grid or not grid[0]:
        return 0
    return 1 if all(row == row[::-1] for row in grid) else 0


def has_vertical_symmetry(grid: Grid) -> int:
    """Return 1 if the grid is vertically symmetric, 0 otherwise."""
    if not grid or not grid[0]:
        return 0
    return 1 if grid == grid[::-1] else 0


def nonzero_pixel_count(grid: Grid) -> int:
    """Return the count of non-zero pixels in the grid."""
    if not grid or not grid[0]:
        return 0
    return sum(1 for row in grid for c in row if c != 0)


def unique_color_count(grid: Grid) -> int:
    """Return the number of unique non-zero colors."""
    if not grid or not grid[0]:
        return 0
    return len(set(c for row in grid for c in row if c != 0))


def most_common_nonzero(grid: Grid) -> int:
    """Return the most common non-zero color (differs from dominant_color by not excluding bg)."""
    if not grid or not grid[0]:
        return 0
    nonzero = [c for row in grid for c in row if c != 0]
    if not nonzero:
        return 0
    return Counter(nonzero).most_common(1)[0][0]


def largest_object_size(grid: Grid) -> int:
    """Return the pixel count of the largest connected foreground component."""
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    visited = set()
    max_size = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                size = 0
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in visited
                                and grid[nr][nc] != bg):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                max_size = max(max_size, size)
    return max_size


def grid_density(grid: Grid) -> int:
    """Return the percentage of non-background pixels (0-100).

    Useful as a parameterized action input: e.g., scale by density ratio.
    """
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    total = h * w
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]
    fg = sum(1 for c in flat if c != bg)
    return (fg * 100) // total if total > 0 else 0


# =============================================================================
# Build functions
# =============================================================================

def build_perception_primitives() -> list[Primitive]:
    """Build the perception primitives (Grid → Value).

    Each is arity-0 in the composition sense — they have no program
    children. They receive the input grid through the execution context.

    25 total: 6 color role + 2 counting + 4 geometry + 6 structural
              + 2 symmetry + 2 pixel counting + 3 additional structural.
    """
    perceptions = [
        # Color role detection (6)
        ("background_color",        background_color),
        ("dominant_color",          dominant_color),
        ("rarest_color",            rarest_color),
        ("accent_color",            accent_color),
        ("second_color",            second_color),
        ("corner_color",            corner_color),
        # Counting (2)
        ("n_colors",                n_colors),
        ("n_foreground_colors",     n_foreground_colors),
        # Geometry (4)
        ("grid_height",             grid_height),
        ("grid_width",              grid_width),
        ("grid_min_dim",            grid_min_dim),
        ("grid_max_dim",            grid_max_dim),
        # Structural (6)
        ("n_objects",               n_objects),
        ("largest_object_color",    largest_object_color),
        ("smallest_object_color",   smallest_object_color),
        ("center_color",            center_color),
        ("edge_color",              edge_color),
        ("interior_dominant_color", interior_dominant_color),
        # Symmetry detection (2)
        ("has_horizontal_symmetry", has_horizontal_symmetry),
        ("has_vertical_symmetry",   has_vertical_symmetry),
        # Pixel counting (2)
        ("nonzero_pixel_count",     nonzero_pixel_count),
        ("unique_color_count",      unique_color_count),
        # Additional structural (3)
        ("most_common_nonzero",     most_common_nonzero),
        ("largest_object_size",     largest_object_size),
        ("grid_density",            grid_density),
    ]
    return [
        Primitive(name=name, arity=0, fn=fn, domain="arc", kind="perception")
        for name, fn in perceptions
    ]
