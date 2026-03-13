"""
ARC-AGI Grammar: composition rules for grid transformation programs.
"""

from __future__ import annotations

import copy
import random
from collections import Counter

from typing import Any

from core import Grammar, Primitive, Program, Task, Decomposition
from .primitives import (
    ARC_PRIMITIVES, ARC_PREDICATES, _PRIM_MAP, register_prim,
    _detect_any_separator_lines, _split_grid_cells, Grid,
    build_task_color_primitives,
)
from .objects import (
    find_foreground_shapes, find_multicolor_objects, place_subgrid,
    _get_background_color,
)


# Structural transforms that score low individually but are critical as
# second/third steps in multi-step programs. Ported from agi-mvp-general.
# Only includes concepts that exist in our primitives registry.
_ARC_ESSENTIAL_PAIR_CONCEPTS: frozenset = frozenset([
    "identity",
    "fill_enclosed",
    "crop_to_nonzero",
    "compress_rows",
    "compress_cols",
    "max_color_per_cell",
    "min_color_per_cell",
    "fill_by_symmetry",
    "fill_tile_pattern",
    "spread_in_lanes_h",
    "spread_in_lanes_v",
    "fill_holes_in_objects",
    "complete_pattern_4way",
    "recolor_isolated_to_nearest",
    "mirror_h_merge",
    "mirror_v_merge",
    "complete_symmetry_diagonal",
    "sort_rows_by_value",
    "remove_color_noise",
    "fill_stripe_gaps_h",
    "fill_stripe_gaps_v",
    "propagate_color_v",
    "complete_tile_from_modal_row",
    "fill_enclosed_wall_color",
    # Batch 2 additions
    "connect_to_rect",
    "gravity_toward_color",
    "extend_to_contact",
    "keep_unique_rows",
    "keep_unique_cols",
    "fill_enclosed_dominant",
    # Batch 4 additions
    "select_odd_cell",
    "overlay_cells",
    "majority_cells",
    "draw_cross_contact",
    "connect_h",
    "connect_v",
    "surround_3x3",
    "fill_convex_hull",
    # Batch 5 additions
    "draw_diag_nearest",
    "draw_cross",
    "flood_fill_accent",
    "fill_enclosed_neighbor",
])


class ARCGrammar(Grammar):
    """
    Grammar for composing ARC grid transformation programs.

    Programs are trees where:
    - Leaves are unary primitives applied directly to the input grid
    - Internal nodes compose the outputs of their children

    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._task_prims: list[Primitive] = []

    def get_predicates(self) -> list[tuple[str, callable]]:
        return list(ARC_PREDICATES)

    def essential_pair_concepts(self) -> frozenset[str]:
        return _ARC_ESSENTIAL_PAIR_CONCEPTS

    def base_primitives(self) -> list[Primitive]:
        return list(ARC_PRIMITIVES) + self._task_prims

    def prepare_for_task(self, task: Task) -> None:
        """Generate task-specific primitives from training examples.

        Two categories:
        1. Color-parameterized primitives: keep_cN, erase_N, swap_A_B, etc.
           Only instantiated for colors present in this task's grids.
           Reduces search space from ~350 to ~150-200 per task.
        2. Structural role primitives: param_role_recolor, param_rank_recolor,
           etc. Learn generalizable role-based mappings from training pairs.
        """
        self._task_prims = []

        # 1. Task-specific color primitives
        task_colors = _extract_task_colors(task)
        color_prims = build_task_color_primitives(task_colors)
        for p in color_prims:
            self._task_prims.append(p)
            register_prim(p)

        # 2. Learned structural primitives
        prims = _learn_parameterized_prims(task)
        for p in prims:
            self._task_prims.append(p)
            register_prim(p)

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def _pick(self, candidates: list[Primitive], parent_op: str,
              transition_matrix) -> Primitive:
        """Pick a primitive, biased by transition matrix if available."""
        if transition_matrix and transition_matrix.size > 0 and parent_op:
            return transition_matrix.weighted_choice(parent_op, candidates, self._rng)
        return self._rng.choice(candidates)

    def mutate(self, program: Program, primitives: list[Primitive],
               transition_matrix=None) -> Program:
        """Mutate a program: point (swap label), grow (leaf→subtree), or shrink (subtree→leaf).

        Point-only mutations can never change tree structure, so grow/shrink
        are essential for discovering programs that require different depths
        than the initial random beam provides.

        When transition_matrix is provided, primitive choices are biased toward
        known-good compositions from the DreamCoder-style prior.
        """
        prog = copy.deepcopy(program)
        nodes = self._collect_nodes(prog)
        if not nodes:
            return prog

        r = self._rng.random()
        if r < 0.20:
            # GROW: replace a leaf with a small subtree (adds depth)
            leaves = [n for n in nodes if not n.children]
            if leaves:
                target = self._rng.choice(leaves)
                higher = [p for p in primitives if p.arity >= 1]
                if higher:
                    new_op = self._pick(higher, target.root, transition_matrix)
                    leaf_prims = [p for p in primitives if p.arity <= 1]
                    if not leaf_prims:
                        leaf_prims = primitives
                    children = [
                        Program(root=self._pick(leaf_prims, new_op.name, transition_matrix).name)
                        for _ in range(new_op.arity)
                    ]
                    target.root = new_op.name
                    target.children = children
                    target.params = {}
            return prog
        elif r < 0.30:
            # SHRINK: replace a non-leaf subtree with a leaf (removes depth)
            internals = [n for n in nodes if n.children]
            if internals:
                target = self._rng.choice(internals)
                leaf_prims = [p for p in primitives if p.arity <= 1]
                if leaf_prims:
                    new_leaf = self._pick(leaf_prims, target.root, transition_matrix)
                    target.root = new_leaf.name
                    target.children = []
                    target.params = {}
            return prog
        else:
            # POINT: swap label with same-arity primitive (preserves structure)
            target = self._rng.choice(nodes)
            prim = _PRIM_MAP.get(target.root)
            current_arity = prim.arity if prim else 1

            # For point mutations, use parent context if available
            parent_op = ""
            if program.children:
                # Find parent of target in the tree
                parent_op = program.root
            same_arity = [p for p in primitives if p.arity == current_arity]
            if same_arity:
                new_prim = self._pick(same_arity, parent_op, transition_matrix)
                target.root = new_prim.name

            return prog

    def crossover(self, a: Program, b: Program) -> Program:
        """Replace a random subtree in a with a random subtree from b."""
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)

        a_nodes = self._collect_nodes(a_copy)
        b_nodes = self._collect_nodes(b_copy)

        if not a_nodes or not b_nodes:
            return a_copy

        target = self._rng.choice(a_nodes)
        donor = self._rng.choice(b_nodes)

        target.root = donor.root
        target.children = donor.children
        target.params = donor.params

        return a_copy

    def _collect_nodes(self, program: Program) -> list[Program]:
        result = [program]
        for child in program.children:
            result.extend(self._collect_nodes(child))
        return result

    # --- Decomposition (inverse of composition) ---

    def decompose(self, input_data: Any, task: Task) -> list[Decomposition]:
        """Decompose an ARC grid into objects for independent transformation.

        Returns multiple decomposition strategies:
        1. Same-color objects (4-connectivity) — standard ARC objects
        2. Multi-color objects (8-connectivity) — for multi-colored patterns

        Each Decomposition contains the subgrids and reassembly context
        (positions, background color, grid dimensions).
        """
        grid = input_data
        if not grid or not grid[0]:
            return []

        bg_color = _get_background_color(grid)
        h, w = len(grid), len(grid[0])
        decompositions = []

        # Strategy 1: Same-color objects (4-connectivity)
        shapes = find_foreground_shapes(grid)
        if shapes and len(shapes) >= 2:
            decompositions.append(Decomposition(
                strategy="same_color_objects",
                parts=[s["subgrid"] for s in shapes],
                context={
                    "positions": [s["position"] for s in shapes],
                    "bg_color": bg_color,
                    "grid_h": h,
                    "grid_w": w,
                },
            ))

        # Strategy 2: Multi-color objects (8-connectivity)
        mc_objects = find_multicolor_objects(grid, bg_color)
        if mc_objects and len(mc_objects) >= 2:
            # Only add if different from strategy 1
            mc_parts = [o["subgrid"] for o in mc_objects]
            if len(mc_objects) != len(shapes):
                decompositions.append(Decomposition(
                    strategy="multicolor_objects",
                    parts=mc_parts,
                    context={
                        "positions": [o["position"] for o in mc_objects],
                        "bg_color": bg_color,
                        "grid_h": h,
                        "grid_w": w,
                    },
                ))

        # Strategy 3: Grid partition (separator lines)
        # Many ARC tasks have a grid divided by separator lines into cells.
        # Each cell can be independently transformed.
        try:
            h_lines, v_lines = _detect_any_separator_lines(grid)
            if h_lines or v_lines:
                cells = _split_grid_cells(grid)
                if cells and len(cells) >= 2:
                    # Determine separator color and line positions for recompose
                    sep_color = 0
                    if h_lines:
                        sep_color = grid[h_lines[0]][0]
                    elif v_lines:
                        sep_color = grid[0][v_lines[0]]
                    decompositions.append(Decomposition(
                        strategy="grid_partition",
                        parts=cells,
                        context={
                            "h_lines": h_lines,
                            "v_lines": v_lines,
                            "sep_color": sep_color,
                            "bg_color": bg_color,
                            "grid_h": h,
                            "grid_w": w,
                        },
                    ))
        except Exception:
            pass  # separator detection can fail on unusual grids

        return decompositions

    def recompose(self, decomposition: Decomposition,
                  transformed_parts: list[Any]) -> Any:
        """Reassemble transformed ARC subgrids back onto a canvas."""
        ctx = decomposition.context
        strategy = decomposition.strategy

        if strategy == "grid_partition":
            return self._recompose_grid_partition(ctx, transformed_parts)

        # Default: object-based recomposition
        bg = ctx.get("bg_color", 0)
        h = ctx.get("grid_h", 0)
        w = ctx.get("grid_w", 0)
        positions = ctx.get("positions", [])

        if not h or not w:
            return transformed_parts[0] if transformed_parts else None

        canvas = [[bg] * w for _ in range(h)]
        for part, pos in zip(transformed_parts, positions):
            if part is not None:
                canvas = place_subgrid(canvas, part, pos, transparent_color=bg)
        return canvas

    def _recompose_grid_partition(self, ctx: dict, parts: list) -> Any:
        """Reassemble grid cells separated by lines."""
        h = ctx.get("grid_h", 0)
        w = ctx.get("grid_w", 0)
        h_lines = ctx.get("h_lines", [])
        v_lines = ctx.get("v_lines", [])
        sep_color = ctx.get("sep_color", 0)

        if not h or not w:
            return parts[0] if parts else None

        canvas = [[0] * w for _ in range(h)]

        # Fill separator lines
        for r in h_lines:
            for c in range(w):
                if 0 <= r < h:
                    canvas[r][c] = sep_color
        for c in v_lines:
            for r in range(h):
                if 0 <= c < w:
                    canvas[r][c] = sep_color

        # Compute cell positions from separator lines
        row_boundaries = [0] + sorted(h_lines) + [h]
        col_boundaries = [0] + sorted(v_lines) + [w]

        cell_idx = 0
        for ri in range(len(row_boundaries) - 1):
            r_start = row_boundaries[ri]
            r_end = row_boundaries[ri + 1]
            if r_start in h_lines:
                r_start += 1
            for ci in range(len(col_boundaries) - 1):
                c_start = col_boundaries[ci]
                c_end = col_boundaries[ci + 1]
                if c_start in v_lines:
                    c_start += 1
                if cell_idx < len(parts) and parts[cell_idx] is not None:
                    cell = parts[cell_idx]
                    for r in range(min(len(cell), r_end - r_start)):
                        for c in range(min(len(cell[0]) if cell else 0, c_end - c_start)):
                            canvas[r_start + r][c_start + c] = cell[r][c]
                cell_idx += 1

        return canvas


# =============================================================================
# Parameterized primitive learning (structural color roles)
# =============================================================================

def _extract_task_colors(task: Task) -> set[int]:
    """Extract all colors appearing in a task's training inputs and outputs."""
    colors: set[int] = set()
    for inp, out in task.train_examples:
        if inp:
            for row in inp:
                colors.update(row)
        if out:
            for row in out:
                colors.update(row)
    return colors


def _assign_color_roles(grid: Grid) -> dict[int, str]:
    """Assign structural roles to colors based on frequency.

    Roles: 'bg' (most frequent), 'rare' (least frequent),
    'dominant' (second most frequent), 'accent' (everything else).
    Returns {color: role} mapping.
    """
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    freq = Counter(flat)
    if not freq:
        return {}

    sorted_colors = sorted(freq.items(), key=lambda x: -x[1])
    roles = {}
    for rank, (color, _) in enumerate(sorted_colors):
        if rank == 0:
            roles[color] = "bg"
        elif rank == 1:
            roles[color] = "dominant"
        elif rank == len(sorted_colors) - 1 and rank >= 2:
            roles[color] = "rare"
        else:
            roles[color] = f"accent_{rank}"
    return roles


def _learn_structural_recolor(task: Task) -> list[tuple[str, str]]:
    """Learn structural role-to-role recolor mappings from training examples.

    Key fix: uses INPUT roles for BOTH source and target colors. The output
    color is looked up in the INPUT's role map. This avoids role drift caused
    by the transformation itself changing frequency distributions.

    Returns list of (src_role, dst_role) pairs that are consistent.
    """
    if not task.train_examples:
        return []

    # Collect per-example role transitions (using input roles for both)
    all_transitions: list[dict[str, Counter]] = []

    for inp, out in task.train_examples:
        if not inp or not out:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            continue  # size-changing tasks don't apply

        inp_roles = _assign_color_roles(inp)

        transitions: dict[str, Counter] = {}
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != out[r][c]:
                    src_role = inp_roles.get(inp[r][c])
                    # Use INPUT roles for target color too — avoids role drift
                    dst_role = inp_roles.get(out[r][c])
                    if src_role and dst_role:
                        if src_role not in transitions:
                            transitions[src_role] = Counter()
                        transitions[src_role][dst_role] += 1

        if transitions:
            all_transitions.append(transitions)

    if not all_transitions:
        return []

    # Find role transitions consistent across all examples
    consistent = []
    first = all_transitions[0]
    for src_role, dst_counts in first.items():
        dst_role = dst_counts.most_common(1)[0][0]
        all_agree = True
        for ex_trans in all_transitions[1:]:
            if src_role not in ex_trans:
                all_agree = False
                break
            ex_dst = ex_trans[src_role].most_common(1)[0][0]
            if ex_dst != dst_role:
                all_agree = False
                break
        if all_agree:
            consistent.append((src_role, dst_role))

    return consistent


def _learn_recolor_by_frequency(task: Task) -> list[tuple[int, int]]:
    """Learn frequency-rank-based color mapping from training examples.

    Returns list of (input_rank, output_color) if consistent across examples.
    The mapping is: color at frequency rank N in input → specific output color.

    This generalizes because rank is structural, not absolute.
    """
    if not task.train_examples:
        return []

    all_rank_maps: list[dict[int, int]] = []

    for inp, out in task.train_examples:
        if not inp or not out:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            continue

        inp_flat = [inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))]
        out_flat = [out[r][c] for r in range(len(out)) for c in range(len(out[0]))]

        inp_freq = Counter(inp_flat)
        out_freq = Counter(out_flat)

        inp_sorted = sorted(inp_freq.items(), key=lambda x: -x[1])
        out_sorted = sorted(out_freq.items(), key=lambda x: -x[1])

        if len(inp_sorted) != len(out_sorted):
            return []  # different color counts → not a rank-based recolor

        rank_map = {}
        for rank in range(len(out_sorted)):
            rank_map[rank] = out_sorted[rank][0]

        # Verify this mapping actually works for this example
        inp_rank = {color: rank for rank, (color, _) in enumerate(inp_sorted)}
        ok = True
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                expected_out = rank_map.get(inp_rank.get(inp[r][c], -1), inp[r][c])
                if expected_out != out[r][c]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            return []

        all_rank_maps.append(rank_map)

    if not all_rank_maps:
        return []

    # Check consistency: same rank → same output color across examples?
    # No: rank maps use different absolute colors per example, that's the point.
    # The structural part is: use the output's frequency order.
    # So we just need to verify the STRUCTURE is consistent:
    # "rank N in input → rank N in output" for all examples.
    # If that's the case, return a sentinel indicating rank-based recolor.
    return [(r, c) for r, c in all_rank_maps[0].items()]


def _learn_parameterized_prims(task: Task) -> list[Primitive]:
    """Learn parameterized primitives from training examples.

    Returns a list of task-specific Primitive objects with structural
    (role-based) closures that generalize to unseen color palettes.
    """
    prims = []

    # --- 1. Structural role-based recolor ---
    role_maps = _learn_structural_recolor(task)
    if role_maps:
        # Build a closure that applies the structural mapping
        captured_maps = list(role_maps)

        def _make_role_recolor(maps):
            def role_recolor(grid: Grid) -> Grid:
                if not grid or not grid[0]:
                    return grid
                roles = _assign_color_roles(grid)
                # Invert: role → color
                role_to_color = {role: color for color, role in roles.items()}
                # Build color→color map from role transitions
                color_map = {}
                for src_role, dst_role in maps:
                    src_color = role_to_color.get(src_role)
                    dst_color = role_to_color.get(dst_role)
                    if src_color is not None and dst_color is not None:
                        color_map[src_color] = dst_color
                if not color_map:
                    return grid
                return [[color_map.get(cell, cell) for cell in row]
                        for row in grid]
            return role_recolor

        fn = _make_role_recolor(captured_maps)
        name = "param_role_recolor"
        prims.append(Primitive(name=name, arity=0, fn=fn))

    # --- 2. Frequency-rank recolor ---
    rank_maps = _learn_recolor_by_frequency(task)
    if rank_maps:
        def _make_rank_recolor(rmap):
            def rank_recolor(grid: Grid) -> Grid:
                if not grid or not grid[0]:
                    return grid
                flat = [grid[r][c] for r in range(len(grid))
                        for c in range(len(grid[0]))]
                freq = Counter(flat)
                sorted_colors = sorted(freq.items(), key=lambda x: -x[1])
                # Build color map: color at rank N → output color at rank N
                out_colors = [oc for _, oc in rmap]
                color_map = {}
                for rank, (color, _) in enumerate(sorted_colors):
                    if rank < len(out_colors):
                        color_map[color] = out_colors[rank]
                return [[color_map.get(cell, cell) for cell in row]
                        for row in grid]
            return rank_recolor

        fn = _make_rank_recolor(rank_maps)
        name = "param_rank_recolor"
        prims.append(Primitive(name=name, arity=0, fn=fn))

    # --- 3. Global color map (depth-0 full solution) ---
    global_map = _learn_global_color_map(task)
    if global_map:
        def _make_global_color_map(cmap):
            def global_color_map(grid: Grid) -> Grid:
                if not grid or not grid[0]:
                    return grid
                return [[cmap.get(cell, cell) for cell in row] for row in grid]
            return global_color_map

        fn = _make_global_color_map(global_map)
        name = "task_global_color_map"
        prims.append(Primitive(name=name, arity=0, fn=fn))

    # --- 4. Fill enclosed with learned color role ---
    fill_role = _learn_fill_enclosed_role(task)
    if fill_role:
        def _make_fill_enclosed(role):
            def fill_enclosed_param(grid: Grid) -> Grid:
                if not grid or not grid[0]:
                    return grid
                roles = _assign_color_roles(grid)
                role_to_color = {r: c for c, r in roles.items()}
                fill_color = role_to_color.get(role)
                if fill_color is None:
                    return grid

                rows, cols = len(grid), len(grid[0])
                result = [row[:] for row in grid]

                # BFS from boundary to find non-enclosed zeros
                reachable = set()
                bg_color = role_to_color.get("bg", 0)
                queue = []
                for r in range(rows):
                    for c in [0, cols - 1]:
                        if result[r][c] == bg_color and (r, c) not in reachable:
                            reachable.add((r, c))
                            queue.append((r, c))
                for c in range(cols):
                    for r in [0, rows - 1]:
                        if result[r][c] == bg_color and (r, c) not in reachable:
                            reachable.add((r, c))
                            queue.append((r, c))

                while queue:
                    cr, cc = queue.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and (nr, nc) not in reachable
                                and result[nr][nc] == bg_color):
                            reachable.add((nr, nc))
                            queue.append((nr, nc))

                # Fill enclosed bg pixels with learned color
                for r in range(rows):
                    for c in range(cols):
                        if result[r][c] == bg_color and (r, c) not in reachable:
                            result[r][c] = fill_color
                return result
            return fill_enclosed_param

        fn = _make_fill_enclosed(fill_role)
        name = "param_fill_enclosed"
        prims.append(Primitive(name=name, arity=0, fn=fn))

    return prims


def _learn_global_color_map(task: Task) -> dict[int, int] | None:
    """Learn a consistent global color→color mapping from training examples.

    For each training pair, computes per-pixel color→color transitions.
    If the same mapping is consistent across ALL training examples (every
    pixel of color X becomes color Y), returns the mapping dict.

    This is different from color_remap correction — it's a depth-0 primitive
    that solves tasks where the transformation is purely a color substitution.
    Only returns if the mapping perfectly reproduces all training outputs.
    """
    if not task.train_examples:
        return None

    # Collect votes across all examples
    global_map: dict[int, int] | None = None

    for inp, out in task.train_examples:
        if not inp or not out:
            return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

        # Build per-example color mapping
        example_map: dict[int, set[int]] = {}
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                src = inp[r][c]
                dst = out[r][c]
                if src not in example_map:
                    example_map[src] = set()
                example_map[src].add(dst)

        # Each source color must map to exactly one destination
        example_unique: dict[int, int] = {}
        for src, dsts in example_map.items():
            if len(dsts) != 1:
                return None  # ambiguous mapping
            example_unique[src] = next(iter(dsts))

        # Check consistency with previous examples
        if global_map is None:
            global_map = example_unique
        else:
            for src, dst in example_unique.items():
                if src in global_map and global_map[src] != dst:
                    return None  # inconsistent across examples
                global_map[src] = dst

    if global_map is None:
        return None

    # Must actually change something (not all identity)
    has_change = any(s != d for s, d in global_map.items())
    if not has_change:
        return None

    # Verify: applying the map reproduces all training outputs exactly
    for inp, out in task.train_examples:
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if global_map.get(inp[r][c], inp[r][c]) != out[r][c]:
                    return None

    return global_map


def _learn_fill_enclosed_role(task: Task) -> str | None:
    """Learn which color role fills enclosed bg regions in training examples.

    Returns the role string (e.g. 'dominant', 'rare') or None.
    """
    if not task.train_examples:
        return None

    fill_roles = []
    for inp, out in task.train_examples:
        if not inp or not out:
            continue
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            continue

        out_roles = _assign_color_roles(out)
        inp_roles = _assign_color_roles(inp)
        bg_role_color = None
        for color, role in inp_roles.items():
            if role == "bg":
                bg_role_color = color
                break

        if bg_role_color is None:
            continue

        # Find pixels that are bg in input but non-bg in output
        fill_colors = Counter()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] == bg_role_color and out[r][c] != bg_role_color:
                    fill_colors[out[r][c]] += 1

        if not fill_colors:
            continue

        fill_color = fill_colors.most_common(1)[0][0]
        role = out_roles.get(fill_color)
        if role:
            fill_roles.append(role)

    if not fill_roles:
        return None

    # Check consistency
    if len(set(fill_roles)) == 1:
        return fill_roles[0]
    return None

