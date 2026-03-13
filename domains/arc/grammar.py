"""
ARC-AGI Grammar: composition rules for grid transformation programs.
"""

from __future__ import annotations

import copy
import random

from typing import Any

from core import Grammar, Primitive, Program, Task, Decomposition
from .primitives import (
    ARC_PRIMITIVES, ARC_PREDICATES, _PRIM_MAP,
    _make_replace_color, _make_swap_colors,
    _detect_any_separator_lines, _split_grid_cells,
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

    Key feature: prepare_for_task() analyzes training examples to create
    task-specific color primitives inferred from input/output pairs.
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
        """Analyze training examples to create task-specific color primitives.

        Generates three categories of task-specific primitives:
        1. Color introduction/removal: for colors that appear/disappear between I/O
        2. Color replacement: for pairs where one color consistently replaces another
        3. Dominant-color operations: recolor to the task's most common output color
        """
        self._task_prims = []
        if not task.train_examples:
            return

        all_in_colors: set[int] = set()
        all_out_colors: set[int] = set()
        for inp, out in task.train_examples:
            all_in_colors.update(c for row in inp for c in row)
            all_out_colors.update(c for row in out for c in row)

        new_colors = all_out_colors - all_in_colors - {0}
        removed_colors = all_in_colors - all_out_colors - {0}
        prim_names = {p.name for p in ARC_PRIMITIVES}

        for c in new_colors:
            name = f"task_fill_bg_{c}"
            if name not in prim_names:
                self._task_prims.append(Primitive(
                    name=name, arity=1, fn=_make_replace_color(0, c), domain="arc"))
                prim_names.add(name)

        for c in removed_colors:
            name = f"task_remove_{c}"
            if name not in prim_names:
                self._task_prims.append(Primitive(
                    name=name, arity=1, fn=_make_replace_color(c, 0), domain="arc"))
                prim_names.add(name)

        for old_c in removed_colors:
            for new_c in new_colors:
                name = f"task_swap_{old_c}_to_{new_c}"
                if name not in prim_names:
                    self._task_prims.append(Primitive(
                        name=name, arity=1, fn=_make_replace_color(old_c, new_c), domain="arc"))
                    prim_names.add(name)

        # --- Pixel-level color transition analysis ---
        # Collect per-pixel color changes across all training examples.
        # If color A consistently becomes color B (>70% of transitions),
        # create a task-specific replacement primitive.
        from collections import Counter
        transitions: Counter = Counter()
        for inp, out in task.train_examples:
            h_i, w_i = len(inp), len(inp[0]) if inp else 0
            h_o, w_o = len(out), len(out[0]) if out else 0
            if h_i != h_o or w_i != w_o:
                continue  # skip size-changing tasks for this analysis
            for r in range(h_i):
                for c in range(w_i):
                    if inp[r][c] != out[r][c]:
                        transitions[(inp[r][c], out[r][c])] += 1

        # Group by source color
        by_src: dict[int, Counter] = {}
        for (src, dst), count in transitions.items():
            if src not in by_src:
                by_src[src] = Counter()
            by_src[src][dst] += count

        for src, tally in by_src.items():
            best_dst, best_count = tally.most_common(1)[0]
            total = sum(tally.values())
            if best_count / total >= 0.70 and best_count >= 2:
                name = f"task_recolor_{src}_to_{best_dst}"
                if name not in prim_names:
                    self._task_prims.append(Primitive(
                        name=name, arity=1,
                        fn=_make_replace_color(src, best_dst), domain="arc"))
                    prim_names.add(name)

        # --- Atomic color swap detection ---
        # When transitions show A→B AND B→A consistently, generate a swap
        # primitive. Swaps are atomic (read from original, write to copy)
        # so they don't corrupt pixels the way sequential remaps would.
        swap_pairs_seen: set[tuple[int, int]] = set()
        for src, tally in by_src.items():
            best_dst, best_count = tally.most_common(1)[0]
            total = sum(tally.values())
            if best_count / total < 0.70 or best_count < 2:
                continue
            # Check for reverse direction
            if best_dst in by_src:
                rev_tally = by_src[best_dst]
                rev_best, rev_count = rev_tally.most_common(1)[0]
                rev_total = sum(rev_tally.values())
                if (rev_best == src and rev_count / rev_total >= 0.70
                        and rev_count >= 2):
                    pair = (min(src, best_dst), max(src, best_dst))
                    if pair not in swap_pairs_seen:
                        swap_pairs_seen.add(pair)
                        a, b = pair
                        name = f"task_swap_{a}_and_{b}"
                        if name not in prim_names:
                            self._task_prims.append(Primitive(
                                name=name, arity=1,
                                fn=_make_swap_colors(a, b), domain="arc"))
                            prim_names.add(name)

        # Register task prims in _PRIM_MAP for execution
        for p in self._task_prims:
            _PRIM_MAP[p.name] = p

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

