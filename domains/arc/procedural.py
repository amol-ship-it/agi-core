"""
Procedural object DSL — learn per-object action rules from examples.

This module implements a pixel-diff engine that:
1. Computes what changed between input and output
2. Attributes changes to input objects
3. Matches action templates (fill_bbox, extend_ray, fill_between, project_to_border)
4. Learns which objects get which action via property-based rules
5. LOOCV-validates before returning

The approach is additive — it doesn't modify any existing code paths.
"""

from __future__ import annotations

from typing import Optional, Callable
from collections import Counter

from .primitives import Grid
from .objects import (
    _find_connected_components,
    _find_multicolor_objects,
    find_foreground_shapes,
    _shape_signature,
    _compactness,
    _has_hole,
    _object_center,
    _test_on_examples,
)


# =============================================================================
# Pixel-diff engine
# =============================================================================

def compute_diff(inp: Grid, out: Grid) -> dict[tuple[int, int], int]:
    """Return {(r, c): new_color} for all pixels that changed."""
    if not inp or not out:
        return {}
    h_in, w_in = len(inp), len(inp[0])
    h_out, w_out = len(out), len(out[0])
    if h_in != h_out or w_in != w_out:
        return {}  # Dimension-change tasks handled separately
    diff = {}
    for r in range(h_in):
        for c in range(w_in):
            if inp[r][c] != out[r][c]:
                diff[(r, c)] = out[r][c]
    return diff


def attribute_to_objects(
    objects: list[dict], diff: dict[tuple[int, int], int], grid: Grid,
) -> dict[int, dict[tuple[int, int], int]]:
    """Assign each diff pixel to its nearest input object.

    Returns {obj_index: {(r,c): color}} mapping.
    """
    if not objects or not diff:
        return {}

    # Build pixel→object index for fast lookup
    pixel_to_obj: dict[tuple[int, int], int] = {}
    for i, obj in enumerate(objects):
        for p in obj["pixels"]:
            pixel_to_obj[p] = i

    # For each diff pixel, find nearest object
    result: dict[int, dict[tuple[int, int], int]] = {}

    for (r, c), color in diff.items():
        # Check if this pixel is inside an object
        if (r, c) in pixel_to_obj:
            obj_idx = pixel_to_obj[(r, c)]
        else:
            # Find nearest object by manhattan distance to any pixel
            best_dist = float('inf')
            best_idx = 0
            for i, obj in enumerate(objects):
                for pr, pc in obj["pixels"]:
                    d = abs(r - pr) + abs(c - pc)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                        if d <= 1:
                            break
                if best_dist <= 1:
                    break
            obj_idx = best_idx

        if obj_idx not in result:
            result[obj_idx] = {}
        result[obj_idx][(r, c)] = color

    return result


# =============================================================================
# Object property extraction (for rule learning)
# =============================================================================

def _object_properties(obj: dict, all_objects: list[dict],
                       grid: Grid) -> dict:
    """Compute a property dict for an object, used as rule keys."""
    h, w = len(grid), len(grid[0])
    r0, c0, r1, c1 = obj["bbox"]
    center_r, center_c = (r0 + r1) / 2.0, (c0 + c1) / 2.0

    props = {
        "all": True,  # Constant key — matches if all objects get same action
        "color": obj["color"],
        "size": obj["size"],
        "size_gt_1": obj["size"] > 1,
        "is_largest": obj["size"] == max(o["size"] for o in all_objects),
        "is_smallest": obj["size"] == min(o["size"] for o in all_objects),
        "compactness_bin": round(_compactness(obj), 1),
        "has_hole": _has_hole(obj),
        "quadrant": (int(center_r >= h / 2), int(center_c >= w / 2)),
        "touches_top": r0 == 0,
        "touches_bottom": r1 == h - 1,
        "touches_left": c0 == 0,
        "touches_right": c1 == w - 1,
        "bbox_h": r1 - r0 + 1,
        "bbox_w": c1 - c0 + 1,
    }
    # Shape signature (translation/color invariant)
    if "subgrid" in obj:
        props["shape_sig"] = _shape_signature(obj)
    return props


# =============================================================================
# Action templates
# =============================================================================

def _check_fill_object_bbox(obj: dict, diff_pixels: dict[tuple[int, int], int],
                             grid: Grid) -> Optional[dict]:
    """Template 1: Fill zeros inside object's bounding box.

    Returns action params if this template explains ALL diff pixels for this object,
    or None if it doesn't match.
    """
    r0, c0, r1, c1 = obj["bbox"]
    fill_color = obj["color"]

    # Check: every diff pixel should be inside bbox and was 0 in input
    expected_fills = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == 0:
                expected_fills[(r, c)] = fill_color

    if not expected_fills:
        return None

    # All diff pixels must match expected fills
    if diff_pixels == expected_fills:
        return {"template": "fill_object_bbox", "fill_color": fill_color}

    # Also try: fill with a specific color (not necessarily object color)
    if diff_pixels:
        colors = set(diff_pixels.values())
        if len(colors) == 1:
            fc = colors.pop()
            expected_fc = {(r, c): fc for r, c in expected_fills if (r, c) in diff_pixels}
            if diff_pixels == {k: fc for k in expected_fills if k in diff_pixels}:
                # Check all expected fills are present
                if set(diff_pixels.keys()) == set(expected_fills.keys()):
                    return {"template": "fill_object_bbox", "fill_color": fc}

    return None


def _check_extend_ray(obj: dict, diff_pixels: dict[tuple[int, int], int],
                       grid: Grid) -> Optional[dict]:
    """Template 2: Extend rays from object edges in a direction.

    Checks if diff pixels form lines extending from object boundary pixels.
    """
    if not diff_pixels:
        return None

    h, w = len(grid), len(grid[0])
    obj_pixels = obj["pixels"]

    # Determine the fill color
    colors = set(diff_pixels.values())
    if len(colors) != 1:
        return None
    fill_color = colors.pop()

    # Try each direction
    for direction, dr, dc in [("up", -1, 0), ("down", 1, 0),
                               ("left", 0, -1), ("right", 0, 1)]:
        # Find boundary pixels (object pixels with no neighbor in direction)
        boundary = set()
        for pr, pc in obj_pixels:
            nr, nc = pr + dr, pc + dc
            if (nr, nc) not in obj_pixels:
                boundary.add((pr, pc))

        # Generate expected ray pixels
        expected = {}
        for pr, pc in boundary:
            r, c = pr + dr, pc + dc
            while 0 <= r < h and 0 <= c < w and grid[r][c] == 0:
                expected[(r, c)] = fill_color
                r += dr
                c += dc

        if expected and expected == diff_pixels:
            return {"template": "extend_ray", "direction": direction,
                    "fill_color": fill_color}

    return None


def _check_fill_between_aligned(obj: dict, obj_idx: int,
                                 all_objects: list[dict],
                                 diff_pixels: dict[tuple[int, int], int],
                                 grid: Grid) -> Optional[dict]:
    """Template 3: Fill between two aligned objects.

    Checks if diff pixels fill the gap between this object and another
    on the same row(s) or column(s).
    """
    if not diff_pixels:
        return None

    colors = set(diff_pixels.values())
    if len(colors) != 1:
        return None
    fill_color = colors.pop()

    r0, c0, r1, c1 = obj["bbox"]

    for other_idx, other in enumerate(all_objects):
        if other_idx == obj_idx:
            continue
        or0, oc0, or1, oc1 = other["bbox"]

        # Horizontal alignment: overlapping rows
        row_overlap = max(0, min(r1, or1) - max(r0, or0) + 1)
        if row_overlap > 0:
            # Fill between columns
            if c1 < oc0:  # obj is left of other
                expected = {}
                for r in range(max(r0, or0), min(r1, or1) + 1):
                    for c in range(c1 + 1, oc0):
                        if grid[r][c] == 0:
                            expected[(r, c)] = fill_color
                if expected and expected == diff_pixels:
                    return {"template": "fill_between", "axis": "h",
                            "fill_color": fill_color, "partner_color": other.get("color")}
            elif oc1 < c0:  # obj is right of other
                expected = {}
                for r in range(max(r0, or0), min(r1, or1) + 1):
                    for c in range(oc1 + 1, c0):
                        if grid[r][c] == 0:
                            expected[(r, c)] = fill_color
                if expected and expected == diff_pixels:
                    return {"template": "fill_between", "axis": "h",
                            "fill_color": fill_color, "partner_color": other.get("color")}

        # Vertical alignment: overlapping columns
        col_overlap = max(0, min(c1, oc1) - max(c0, oc0) + 1)
        if col_overlap > 0:
            if r1 < or0:  # obj is above other
                expected = {}
                for c in range(max(c0, oc0), min(c1, oc1) + 1):
                    for r in range(r1 + 1, or0):
                        if grid[r][c] == 0:
                            expected[(r, c)] = fill_color
                if expected and expected == diff_pixels:
                    return {"template": "fill_between", "axis": "v",
                            "fill_color": fill_color, "partner_color": other.get("color")}
            elif or1 < r0:  # obj is below other
                expected = {}
                for c in range(max(c0, oc0), min(c1, oc1) + 1):
                    for r in range(or1 + 1, r0):
                        if grid[r][c] == 0:
                            expected[(r, c)] = fill_color
                if expected and expected == diff_pixels:
                    return {"template": "fill_between", "axis": "v",
                            "fill_color": fill_color, "partner_color": other.get("color")}

    return None


def _check_project_to_border(obj: dict, diff_pixels: dict[tuple[int, int], int],
                              grid: Grid) -> Optional[dict]:
    """Template 4: Project object to grid border.

    Checks if diff pixels extend the object to the grid edge.
    """
    if not diff_pixels:
        return None

    h, w = len(grid), len(grid[0])
    colors = set(diff_pixels.values())
    if len(colors) != 1:
        return None
    fill_color = colors.pop()

    obj_pixels = obj["pixels"]

    for direction, dr, dc in [("up", -1, 0), ("down", 1, 0),
                               ("left", 0, -1), ("right", 0, 1)]:
        expected = {}
        for pr, pc in obj_pixels:
            r, c = pr + dr, pc + dc
            while 0 <= r < h and 0 <= c < w:
                if grid[r][c] == 0 and (r, c) not in obj_pixels:
                    expected[(r, c)] = fill_color
                r += dr
                c += dc

        if expected and expected == diff_pixels:
            return {"template": "project_to_border", "direction": direction,
                    "fill_color": fill_color}

    return None


def _check_fill_enclosed(obj: dict, diff_pixels: dict[tuple[int, int], int],
                          grid: Grid) -> Optional[dict]:
    """Template 5: Fill enclosed zeros (holes) within an object.

    Uses flood fill from grid border — any zero NOT reachable from border
    that is within/near the object is considered enclosed.
    """
    if not diff_pixels:
        return None

    h, w = len(grid), len(grid[0])
    colors = set(diff_pixels.values())
    if len(colors) != 1:
        return None
    fill_color = colors.pop()

    # Flood fill from border to find all exterior zeros
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

    # Interior zeros = all zeros not in exterior
    # Filter to zeros near this object's bbox (expanded by 1)
    r0, c0, r1, c1 = obj["bbox"]
    expected = {}
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if grid[r][c] == 0 and (r, c) not in exterior:
                expected[(r, c)] = fill_color

    if expected and expected == diff_pixels:
        return {"template": "fill_enclosed", "fill_color": fill_color}

    return None


def _check_gravity(obj: dict, diff_pixels: dict[tuple[int, int], int],
                    grid: Grid, all_objects: list[dict],
                    obj_idx: int) -> Optional[dict]:
    """Template 6: Object moves in a direction (gravity) until hitting obstacle.

    Diff should show: object pixels at old position become 0,
    and new object pixels appear at new position.
    """
    if not diff_pixels:
        return None

    h, w = len(grid), len(grid[0])
    obj_pixels = obj["pixels"]

    # Check if the diff removes the object (old pixels → 0) and adds it elsewhere
    removed = {(r, c) for (r, c), v in diff_pixels.items() if v == 0 and (r, c) in obj_pixels}
    added = {(r, c) for (r, c), v in diff_pixels.items() if v == obj["color"] and (r, c) not in obj_pixels}

    if not removed or not added:
        return None
    if len(removed) != len(obj_pixels) or len(removed) != len(added):
        return None

    # Compute displacement
    old_min_r = min(r for r, c in obj_pixels)
    old_min_c = min(c for r, c in obj_pixels)
    new_min_r = min(r for r, c in added)
    new_min_c = min(c for r, c in added)
    dr = new_min_r - old_min_r
    dc = new_min_c - old_min_c

    # Verify all pixels shifted consistently
    expected_new = {(r + dr, c + dc) for r, c in obj_pixels}
    if expected_new != added:
        return None

    # Verify this is a valid gravity direction (single axis)
    if dr != 0 and dc != 0:
        return None

    if dr > 0:
        direction = "down"
    elif dr < 0:
        direction = "up"
    elif dc > 0:
        direction = "right"
    else:
        direction = "left"

    # Verify gravity: object should stop at first obstacle
    dir_dr = 1 if dr > 0 else (-1 if dr < 0 else 0)
    dir_dc = 1 if dc > 0 else (-1 if dc < 0 else 0)

    # Check that the object can't move one more step
    other_pixels = set()
    for i, o in enumerate(all_objects):
        if i != obj_idx:
            other_pixels.update(o["pixels"])

    for r, c in expected_new:
        nr, nc = r + dir_dr, c + dir_dc
        if not (0 <= nr < h and 0 <= nc < w):
            break  # Hit wall
        if (nr, nc) in other_pixels or grid[nr][nc] != 0:
            break  # Hit obstacle
    else:
        # Could move further — not gravity
        return None

    return {"template": "gravity", "direction": direction, "fill_color": obj["color"]}


# =============================================================================
# Action application (for generating output)
# =============================================================================

def _apply_fill_object_bbox(obj: dict, grid: Grid, params: dict) -> dict[tuple[int, int], int]:
    """Apply fill_object_bbox action, return pixels to set."""
    r0, c0, r1, c1 = obj["bbox"]
    result = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == 0:
                result[(r, c)] = params["fill_color"]
    return result


def _apply_extend_ray(obj: dict, grid: Grid, params: dict) -> dict[tuple[int, int], int]:
    """Apply extend_ray action, return pixels to set."""
    h, w = len(grid), len(grid[0])
    dir_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = dir_map[params["direction"]]
    fill_color = params["fill_color"]

    boundary = set()
    for pr, pc in obj["pixels"]:
        nr, nc = pr + dr, pc + dc
        if (nr, nc) not in obj["pixels"]:
            boundary.add((pr, pc))

    result = {}
    for pr, pc in boundary:
        r, c = pr + dr, pc + dc
        while 0 <= r < h and 0 <= c < w and grid[r][c] == 0:
            result[(r, c)] = fill_color
            r += dr
            c += dc
    return result


def _apply_fill_between(obj: dict, obj_idx: int, all_objects: list[dict],
                         grid: Grid, params: dict) -> dict[tuple[int, int], int]:
    """Apply fill_between action, return pixels to set."""
    fill_color = params["fill_color"]
    axis = params["axis"]
    partner_color = params.get("partner_color")
    r0, c0, r1, c1 = obj["bbox"]

    result = {}
    for other_idx, other in enumerate(all_objects):
        if other_idx == obj_idx:
            continue
        if partner_color is not None and other.get("color") != partner_color:
            continue
        or0, oc0, or1, oc1 = other["bbox"]

        if axis == "h":
            row_overlap = max(0, min(r1, or1) - max(r0, or0) + 1)
            if row_overlap > 0:
                left_c = min(c1, oc1) + 1
                right_c = max(c0, oc0)
                for r in range(max(r0, or0), min(r1, or1) + 1):
                    for c in range(left_c, right_c):
                        if grid[r][c] == 0:
                            result[(r, c)] = fill_color
        elif axis == "v":
            col_overlap = max(0, min(c1, oc1) - max(c0, oc0) + 1)
            if col_overlap > 0:
                top_r = min(r1, or1) + 1
                bottom_r = max(r0, or0)
                for c in range(max(c0, oc0), min(c1, oc1) + 1):
                    for r in range(top_r, bottom_r):
                        if grid[r][c] == 0:
                            result[(r, c)] = fill_color
    return result


def _apply_project_to_border(obj: dict, grid: Grid,
                              params: dict) -> dict[tuple[int, int], int]:
    """Apply project_to_border action, return pixels to set."""
    h, w = len(grid), len(grid[0])
    dir_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = dir_map[params["direction"]]
    fill_color = params["fill_color"]

    result = {}
    for pr, pc in obj["pixels"]:
        r, c = pr + dr, pc + dc
        while 0 <= r < h and 0 <= c < w:
            if grid[r][c] == 0 and (r, c) not in obj["pixels"]:
                result[(r, c)] = fill_color
            r += dr
            c += dc
    return result


def _apply_fill_enclosed(obj: dict, grid: Grid, params: dict) -> dict[tuple[int, int], int]:
    """Apply fill_enclosed action, return pixels to set."""
    h, w = len(grid), len(grid[0])
    fill_color = params["fill_color"]

    # Flood fill from border
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

    r0, c0, r1, c1 = obj["bbox"]
    result = {}
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if grid[r][c] == 0 and (r, c) not in exterior:
                result[(r, c)] = fill_color
    return result


def _apply_gravity(obj: dict, grid: Grid, params: dict,
                    all_objects: list[dict] = None,
                    obj_idx: int = -1) -> dict[tuple[int, int], int]:
    """Apply gravity action, return pixels to set (including clearing old position)."""
    h, w = len(grid), len(grid[0])
    dir_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = dir_map[params["direction"]]
    color = params["fill_color"]

    # Collect obstacle pixels (other objects)
    obstacles = set()
    if all_objects:
        for i, o in enumerate(all_objects):
            if i != obj_idx:
                obstacles.update(o["pixels"])

    # Move object in direction until hitting wall or obstacle
    shift = 0
    while True:
        shift += 1
        # Check if all pixels can move to shift position
        can_move = True
        for pr, pc in obj["pixels"]:
            nr, nc = pr + dr * shift, pc + dc * shift
            if not (0 <= nr < h and 0 <= nc < w):
                can_move = False
                break
            if (nr, nc) in obstacles:
                can_move = False
                break
            if grid[nr][nc] != 0 and (nr, nc) not in obj["pixels"]:
                can_move = False
                break
        if not can_move:
            shift -= 1
            break

    if shift == 0:
        return {}

    result = {}
    # Clear old pixels
    for r, c in obj["pixels"]:
        result[(r, c)] = 0
    # Place at new position
    for r, c in obj["pixels"]:
        result[(r + dr * shift, c + dc * shift)] = color
    return result


_APPLY_FNS = {
    "fill_object_bbox": _apply_fill_object_bbox,
    "extend_ray": _apply_extend_ray,
    "fill_between": _apply_fill_between,
    "project_to_border": _apply_project_to_border,
    "fill_enclosed": _apply_fill_enclosed,
    "gravity": _apply_gravity,
}


# =============================================================================
# Rule learning: which objects get which action
# =============================================================================

def _try_all_templates(obj, obj_idx, all_objects, obj_diff, grid):
    """Try all action templates on an object's diff. Returns action or None."""
    action = _check_fill_object_bbox(obj, obj_diff, grid)
    if action is None:
        action = _check_fill_enclosed(obj, obj_diff, grid)
    if action is None:
        action = _check_extend_ray(obj, obj_diff, grid)
    if action is None:
        action = _check_fill_between_aligned(obj, obj_idx, all_objects, obj_diff, grid)
    if action is None:
        action = _check_project_to_border(obj, obj_diff, grid)
    if action is None:
        action = _check_gravity(obj, obj_diff, grid, all_objects, obj_idx)
    return action


def _learn_object_action_rules(
    examples: list[tuple[Grid, Grid]],
) -> Optional[tuple[str, Callable]]:
    """Learn per-object action rules from training examples.

    For each example:
    1. Detect objects in input
    2. Compute diff (input → output)
    3. Attribute diff pixels to nearest objects
    4. Try action templates on each object's attributed diff
    5. Learn property→action mapping across examples
    6. LOOCV validate

    Returns (name, fn) if a consistent rule is found, None otherwise.
    """
    if not examples or len(examples) < 2:
        return None

    # Only same-dimension tasks for now
    for inp, out in examples:
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

    # Phase 1: Analyze each example
    per_example_actions = []  # list of list of (obj_props, action_params)

    for inp, out in examples:
        objects = _find_connected_components(inp)
        if not objects:
            return None

        # Add subgrid to each object for property computation
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            h_obj = r1 - r0 + 1
            w_obj = c1 - c0 + 1
            sg = [[0] * w_obj for _ in range(h_obj)]
            for r, c in obj["pixels"]:
                sg[r - r0][c - c0] = obj["color"]
            obj["subgrid"] = sg

        diff = compute_diff(inp, out)
        if not diff:
            return None  # No changes → nothing to learn

        attributed = attribute_to_objects(objects, diff, inp)

        # For objects with no attributed diff, their action is "none"
        example_actions = []
        for i, obj in enumerate(objects):
            props = _object_properties(obj, objects, inp)
            obj_diff = attributed.get(i, {})

            if not obj_diff:
                example_actions.append((props, {"template": "none"}))
                continue

            # Try each template
            action = _try_all_templates(obj, i, objects, obj_diff, inp)
            if action is None:
                return None  # Diff not explained by any template

            example_actions.append((props, action))

        per_example_actions.append(example_actions)

    # Phase 2: Find a property key that consistently predicts action template
    # Try various property keys to distinguish which objects get which action
    property_keys = [
        "all",  # Every object gets the same action
        "color", "is_largest", "is_smallest", "has_hole",
        "compactness_bin", "quadrant", "size",
        "touches_top", "touches_bottom", "touches_left", "touches_right",
        "size_gt_1",  # Distinguishes singletons from multi-pixel objects
    ]

    for prop_key in property_keys:
        # Build mapping: prop_value → action_template
        prop_to_action: dict = {}
        consistent = True

        for ex_actions in per_example_actions:
            for props, action in ex_actions:
                pv = props.get(prop_key)
                template = action["template"]
                key = (pv, template)

                # Check: same property value → same template + same params
                if pv in prop_to_action:
                    if prop_to_action[pv] != action:
                        consistent = False
                        break
                else:
                    prop_to_action[pv] = action
            if not consistent:
                break

        if not consistent:
            continue

        # Check that the rule actually produces the correct output
        def _make_applicator(rule_map, prop_key_inner):
            def apply_rule(grid):
                objects = _find_connected_components(grid)
                if not objects:
                    return grid

                # Add subgrids for property computation
                for obj in objects:
                    r0, c0, r1, c1 = obj["bbox"]
                    h_obj = r1 - r0 + 1
                    w_obj = c1 - c0 + 1
                    sg = [[0] * w_obj for _ in range(h_obj)]
                    for r, c in obj["pixels"]:
                        sg[r - r0][c - c0] = obj["color"]
                    obj["subgrid"] = sg

                result = [row[:] for row in grid]
                for i, obj in enumerate(objects):
                    props = _object_properties(obj, objects, grid)
                    pv = props.get(prop_key_inner)
                    if pv not in rule_map:
                        continue
                    action = rule_map[pv]
                    if action["template"] == "none":
                        continue

                    apply_fn = _APPLY_FNS.get(action["template"])
                    if apply_fn is None:
                        continue

                    if action["template"] in ("fill_between",):
                        pixels = apply_fn(obj, i, objects, grid, action)
                    elif action["template"] == "gravity":
                        pixels = apply_fn(obj, grid, action, all_objects=objects, obj_idx=i)
                    else:
                        pixels = apply_fn(obj, grid, action)

                    for (r, c), color in pixels.items():
                        result[r][c] = color
                return result
            return apply_rule

        fn = _make_applicator(prop_to_action, prop_key)

        # Verify on all examples
        if not _test_on_examples(fn, examples):
            continue

        # LOOCV validation
        loocv_pass = True
        for hold_idx in range(len(examples)):
            train_sub = [ex for i, ex in enumerate(examples) if i != hold_idx]
            # Re-learn on subset
            sub_result = _learn_rule_for_key(train_sub, prop_key)
            if sub_result is None:
                loocv_pass = False
                break
            sub_map, _ = sub_result
            sub_fn = _make_applicator(sub_map, prop_key)
            held_inp, held_exp = examples[hold_idx]
            try:
                if sub_fn(held_inp) != held_exp:
                    loocv_pass = False
                    break
            except Exception:
                loocv_pass = False
                break

        if not loocv_pass:
            continue

        # Build descriptive name
        templates_used = set(a["template"] for a in prop_to_action.values()
                            if a["template"] != "none")
        name = f"procedural({prop_key}:{'+'.join(sorted(templates_used))})"
        return (name, fn)

    return None


def _learn_rule_for_key(
    examples: list[tuple[Grid, Grid]], prop_key: str,
) -> Optional[tuple[dict, str]]:
    """Learn a property→action mapping for a specific property key.

    Returns (prop_to_action_map, prop_key) or None.
    """
    for inp, out in examples:
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

    prop_to_action: dict = {}

    for inp, out in examples:
        objects = _find_connected_components(inp)
        if not objects:
            return None

        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            h_obj = r1 - r0 + 1
            w_obj = c1 - c0 + 1
            sg = [[0] * w_obj for _ in range(h_obj)]
            for r, c in obj["pixels"]:
                sg[r - r0][c - c0] = obj["color"]
            obj["subgrid"] = sg

        diff = compute_diff(inp, out)
        if not diff:
            return None

        attributed = attribute_to_objects(objects, diff, inp)

        for i, obj in enumerate(objects):
            props = _object_properties(obj, objects, inp)
            obj_diff = attributed.get(i, {})

            if not obj_diff:
                action = {"template": "none"}
            else:
                action = _try_all_templates(obj, i, objects, obj_diff, inp)
                if action is None:
                    return None

            pv = props.get(prop_key)
            if pv in prop_to_action:
                if prop_to_action[pv] != action:
                    return None
            else:
                prop_to_action[pv] = action

    return (prop_to_action, prop_key)


# =============================================================================
# Public API
# =============================================================================

def try_procedural(
    task,
) -> Optional[tuple[str, Callable]]:
    """Main entry point: try to learn procedural object rules for a task.

    Called from environment.try_procedural().
    Returns (name, fn) or None.
    """
    examples = task.train_examples
    if not examples or len(examples) < 2:
        return None

    return _learn_object_action_rules(examples)
