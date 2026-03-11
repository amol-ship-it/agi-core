"""
Connected component detection and object decomposition for ARC grids.

Provides:
- Flood-fill-based object extraction
- Per-object transform pipeline (perceive → transform → reassemble)
- Conditional recolor by object properties (size, shape, position, etc.)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Callable

from .primitives import Grid


# =============================================================================
# Connected component detection
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
# Object-level perception and reassembly
# =============================================================================

def find_foreground_shapes(grid: Grid) -> list[dict]:
    """Extract each object as a subgrid with metadata for reassembly.

    Returns list of dicts with: subgrid, bbox, color, size, position.
    """
    components = _find_connected_components(grid)
    shapes = []
    for comp in components:
        shapes.append({
            "subgrid": _component_to_subgrid(comp),
            "bbox": comp["bbox"],
            "color": comp["color"],
            "size": comp["size"],
            "position": (comp["bbox"][0], comp["bbox"][1]),
        })
    return shapes


def place_subgrid(
    canvas: Grid,
    subgrid: Grid,
    position: tuple[int, int],
    transparent_color: int = 0,
) -> Grid:
    """Place a subgrid onto a canvas at the given position.

    Non-destructive: returns a copy with subgrid placed. Cells matching
    transparent_color in subgrid don't overwrite the canvas.
    """
    result = [row[:] for row in canvas]
    pr, pc = position
    sh = len(subgrid)
    sw = len(subgrid[0]) if subgrid else 0
    ch = len(canvas)
    cw = len(canvas[0]) if canvas else 0

    for r in range(sh):
        for c in range(sw):
            tr, tc = pr + r, pc + c
            if 0 <= tr < ch and 0 <= tc < cw:
                if subgrid[r][c] != transparent_color:
                    result[tr][tc] = subgrid[r][c]
    return result


def _get_background_color(grid: Grid) -> int:
    """Most frequent value in the grid (typically 0)."""
    counts: Counter = Counter()
    for row in grid:
        for val in row:
            counts[val] += 1
    return max(counts, key=lambda k: counts[k])


# =============================================================================
# Per-object transform pipeline
# =============================================================================

def apply_transform_per_object(
    grid: Grid,
    transform: Callable[[Grid], Grid],
    bg_color: int = 0,
) -> Optional[Grid]:
    """Apply a transform to each object's subgrid and reassemble.

    1. Extract foreground shapes
    2. Create blank canvas with background color
    3. For each shape: transform subgrid, place back at same position
    """
    shapes = find_foreground_shapes(grid)
    if not shapes:
        return None

    h, w = len(grid), len(grid[0]) if grid else 0
    canvas = [[bg_color] * w for _ in range(h)]

    for shape in shapes:
        try:
            transformed = transform(shape["subgrid"])
            if transformed is None:
                return None
        except Exception:
            return None
        canvas = place_subgrid(canvas, transformed, shape["position"],
                               transparent_color=bg_color)
    return canvas


def try_object_decomposition(
    task_examples: list[tuple],
    primitives: list,
) -> Optional[tuple[str, Callable]]:
    """Try to solve a task by applying the same transform per object.

    For same-dims tasks, tries each primitive as a per-object transform.
    Returns (name, transform_fn) if pixel-perfect on all training examples.
    """
    if not task_examples:
        return None

    # Only same-dims tasks
    for inp, out in task_examples:
        if len(inp) != len(out):
            return None
        if inp and out and len(inp[0]) != len(out[0]):
            return None

    bg_color = _get_background_color(task_examples[0][0])

    for prim in primitives:
        if prim.arity != 1:
            continue

        def make_per_obj_fn(t=prim.fn, bg=bg_color):
            def fn(grid):
                result = apply_transform_per_object(grid, t, bg)
                return result if result is not None else grid
            return fn

        per_obj_fn = make_per_obj_fn()
        all_match = True
        for inp, expected in task_examples:
            try:
                result = per_obj_fn(inp)
                if result != expected:
                    all_match = False
                    break
            except Exception:
                all_match = False
                break

        if all_match:
            return (f"per_object({prim.name})", per_obj_fn)

    # Strategy 2: conditional recolor by object properties
    cond = _try_conditional_recolor(task_examples)
    if cond is not None:
        return cond

    return None


# =============================================================================
# Conditional per-object recolor strategies
# =============================================================================

def _match_objects_by_position(inp: Grid, out: Grid) -> Optional[list[tuple[dict, dict]]]:
    """Match input objects to output objects by pixel overlap."""
    shapes_in = find_foreground_shapes(inp)
    shapes_out = find_foreground_shapes(out)
    if not shapes_in:
        return None

    matches = []
    used = set()

    for si in shapes_in:
        si_pixels = {(si["position"][0] + r, si["position"][1] + c)
                     for r in range(len(si["subgrid"]))
                     for c in range(len(si["subgrid"][0]))
                     if si["subgrid"][r][c] != 0}

        best_idx, best_overlap = -1, 0
        for j, so in enumerate(shapes_out):
            if j in used:
                continue
            so_pixels = {(so["position"][0] + r, so["position"][1] + c)
                         for r in range(len(so["subgrid"]))
                         for c in range(len(so["subgrid"][0]))
                         if so["subgrid"][r][c] != 0}
            overlap = len(si_pixels & so_pixels)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = j

        if best_idx >= 0:
            used.add(best_idx)
            matches.append((si, shapes_out[best_idx]))
        else:
            # Fallback: match by proximity
            best_idx, best_dist = -1, float("inf")
            for j, so in enumerate(shapes_out):
                if j in used:
                    continue
                dist = abs(si["position"][0] - so["position"][0]) + \
                       abs(si["position"][1] - so["position"][1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx >= 0:
                used.add(best_idx)
                matches.append((si, shapes_out[best_idx]))
            else:
                return None

    return matches


def _shape_signature(shape: dict) -> tuple:
    """Translation/color-invariant shape signature."""
    positions = [(r, c) for r in range(len(shape["subgrid"]))
                 for c in range(len(shape["subgrid"][0]))
                 if shape["subgrid"][r][c] != 0]
    if not positions:
        return ()
    min_r = min(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    return tuple(sorted((r - min_r, c - min_c) for r, c in positions))


def _compactness(shape: dict) -> float:
    """Compactness = pixels / bbox area. Rectangle → 1.0."""
    r0, c0, r1, c1 = shape["bbox"]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return shape["size"] / area if area > 0 else 1.0


def _has_hole(shape: dict) -> bool:
    """Check if shape has an enclosed hole via flood fill from border."""
    sg = shape["subgrid"]
    h, w = len(sg), len(sg[0]) if sg else 0
    if h <= 2 or w <= 2:
        return False
    visited = [[False] * w for _ in range(h)]
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and sg[r][c] == 0:
                visited[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and sg[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))
    return any(sg[r][c] == 0 and not visited[r][c]
               for r in range(h) for c in range(w))


def _try_conditional_recolor(
    task_examples: list[tuple],
) -> Optional[tuple[str, Callable]]:
    """Try 6 property-based recolor strategies on objects."""
    strategies = [
        ("by_size", _learn_recolor_by_size),
        ("by_singleton", _learn_recolor_by_singleton),
        ("by_input_color", _learn_recolor_by_input_color),
        ("by_shape", _learn_recolor_by_shape),
        ("by_size_rank", _learn_recolor_by_size_rank),
        ("by_compactness", _learn_recolor_by_compactness),
        ("by_has_hole", _learn_recolor_by_has_hole),
    ]

    for strat_name, learn_fn in strategies:
        rule = learn_fn(task_examples)
        if rule is None:
            continue

        fn = _make_conditional_recolor_fn(rule, strat_name)

        # Verify pixel-perfect on all examples
        all_match = True
        for inp, expected in task_examples:
            try:
                if fn(inp) != expected:
                    all_match = False
                    break
            except Exception:
                all_match = False
                break
        if all_match:
            return (f"per_object_recolor({strat_name})", fn)

    return None


def _learn_recolor_by_size(examples: list[tuple]) -> Optional[dict]:
    size_to_color: dict[int, int] = {}
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            s = si["size"]
            if s in size_to_color and size_to_color[s] != so["color"]:
                return None
            size_to_color[s] = so["color"]
    if not any(True for inp, out in examples
               for si, so in (_match_objects_by_position(inp, out) or [])
               if si["color"] != so["color"]):
        return None
    return size_to_color


def _learn_recolor_by_singleton(examples: list[tuple]) -> Optional[dict]:
    singleton_colors: set[int] = set()
    multi_colors: set[int] = set()
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            if si["size"] == 1:
                singleton_colors.add(so["color"])
            else:
                multi_colors.add(so["color"])
    if len(singleton_colors) != 1 or len(multi_colors) != 1:
        return None
    s, m = next(iter(singleton_colors)), next(iter(multi_colors))
    return {True: m, False: s} if s != m else None


def _learn_recolor_by_input_color(examples: list[tuple]) -> Optional[dict]:
    color_map: dict[int, int] = {}
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            ic = si["color"]
            if ic in color_map and color_map[ic] != so["color"]:
                return None
            color_map[ic] = so["color"]
    if not any(k != v for k, v in color_map.items()):
        return None
    return color_map


def _learn_recolor_by_shape(examples: list[tuple]) -> Optional[dict]:
    shape_to_color: dict[tuple, int] = {}
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            sig = _shape_signature(si)
            if sig in shape_to_color and shape_to_color[sig] != so["color"]:
                return None
            shape_to_color[sig] = so["color"]
    if len(set(shape_to_color.values())) < 2:
        return None
    return shape_to_color


def _learn_recolor_by_size_rank(examples: list[tuple]) -> Optional[dict]:
    rank_to_color: dict[int, int] = {}
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        sized = sorted(matches, key=lambda x: -x[0]["size"])
        for rank, (si, so) in enumerate(sized):
            if rank in rank_to_color and rank_to_color[rank] != so["color"]:
                return None
            rank_to_color[rank] = so["color"]
    if len(set(rank_to_color.values())) < 2:
        return None
    return rank_to_color


def _learn_recolor_by_compactness(examples: list[tuple]) -> Optional[dict]:
    compact_colors: set[int] = set()
    non_compact_colors: set[int] = set()
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            if _compactness(si) >= 1.0 - 1e-9:
                compact_colors.add(so["color"])
            else:
                non_compact_colors.add(so["color"])
    if len(compact_colors) != 1 or len(non_compact_colors) != 1:
        return None
    cc, nc = next(iter(compact_colors)), next(iter(non_compact_colors))
    return {True: cc, False: nc} if cc != nc else None


def _learn_recolor_by_has_hole(examples: list[tuple]) -> Optional[dict]:
    hole_colors: set[int] = set()
    no_hole_colors: set[int] = set()
    for inp, out in examples:
        matches = _match_objects_by_position(inp, out)
        if matches is None:
            return None
        for si, so in matches:
            if _has_hole(si):
                hole_colors.add(so["color"])
            else:
                no_hole_colors.add(so["color"])
    if len(hole_colors) != 1 or len(no_hole_colors) != 1:
        return None
    hc, nhc = next(iter(hole_colors)), next(iter(no_hole_colors))
    return {True: hc, False: nhc} if hc != nhc else None


def _make_conditional_recolor_fn(rule: dict, strategy: str) -> Callable:
    """Build a Grid→Grid function from a learned conditional recolor rule."""
    def transform(grid: Grid) -> Grid:
        shapes = find_foreground_shapes(grid)
        if not shapes:
            return grid
        h, w = len(grid), len(grid[0]) if grid else 0
        result = [row[:] for row in grid]

        # Pre-compute size ranks if needed
        size_ranks: dict[int, int] = {}
        if strategy == "by_size_rank":
            for rank, idx in enumerate(sorted(range(len(shapes)),
                                              key=lambda i: -shapes[i]["size"])):
                size_ranks[idx] = rank

        for i, shape in enumerate(shapes):
            if strategy == "by_size":
                new_color = rule.get(shape["size"], shape["color"])
            elif strategy == "by_singleton":
                new_color = rule[shape["size"] > 1]
            elif strategy == "by_input_color":
                new_color = rule.get(shape["color"], shape["color"])
            elif strategy == "by_shape":
                new_color = rule.get(_shape_signature(shape), shape["color"])
            elif strategy == "by_size_rank":
                new_color = rule.get(size_ranks.get(i, -1), shape["color"])
            elif strategy == "by_compactness":
                new_color = rule[_compactness(shape) >= 1.0 - 1e-9]
            elif strategy == "by_has_hole":
                new_color = rule[_has_hole(shape)]
            else:
                new_color = shape["color"]

            pos = shape["position"]
            for r in range(len(shape["subgrid"])):
                for c in range(len(shape["subgrid"][0])):
                    if shape["subgrid"][r][c] != 0:
                        result[pos[0] + r][pos[1] + c] = new_color
        return result
    return transform

