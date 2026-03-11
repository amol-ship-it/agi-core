"""
Connected component detection for ARC grids.

Provides flood-fill-based object extraction used by object-level primitives.
"""

from __future__ import annotations

from .primitives import Grid


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

