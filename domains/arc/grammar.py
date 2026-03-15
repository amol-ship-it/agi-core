"""
ARC-AGI Grammar: composition rules for grid transformation programs.
"""

from __future__ import annotations

import copy
import random
from collections import Counter

from typing import Any

from core import Grammar, Primitive, Program, Task, Decomposition
from .primitives import _PRIM_MAP, register_prim, Grid


# Structural transforms that score low individually but are critical as
# second/third steps in multi-step programs. Ported from agi-mvp-general.
# Only includes concepts that exist in our primitives registry.
class ARCGrammar(Grammar):
    """Grammar for composing ARC grid transformation programs.

    Uses atomic vocabulary: truly atomic transforms + perception + parameterized.
    Programs are trees where leaves are primitives and internal nodes compose them.
    """

    def __init__(self, seed: int = 42, vocabulary: str = "full"):
        self._rng = random.Random(seed)
        self._task_prims: list[Primitive] = []
        self._vocabulary = vocabulary

    def allow_structural_phases(self) -> bool:
        return self._vocabulary != "atomic"

    def get_predicates(self) -> list[tuple[str, callable]]:
        return []  # predicates not used in atomic mode (structural phases disabled)

    def essential_pair_concepts(self) -> frozenset[str]:
        from .transformation_primitives import ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        return ATOMIC_ESSENTIAL_PAIR_CONCEPTS

    def task_priority_primitives(self, task: Task) -> list[str]:
        """Return primitives likely relevant for this task based on structure."""
        if not task.train_examples:
            return []
        first_inp = task.train_examples[0][0]
        first_out = task.train_examples[0][1]
        if not first_inp or not first_inp[0] or not first_out:
            return []

        hints: list[str] = []
        h, w = len(first_inp), len(first_inp[0])
        oh, ow = len(first_out), len(first_out[0]) if first_out[0] else 0

        # Output smaller → cropping
        if oh < h or ow < w:
            hints.extend(["crop_to_content", "crop_half_top", "crop_half_left"])
        # Output larger → scaling
        if oh > h or ow > w:
            hints.append("pad_border")
        # Same size → transforms
        if oh == h and ow == w:
            hints.extend(["fill_enclosed", "gravity_down", "label_components"])

        return hints

    def base_primitives(self) -> list[Primitive]:
        from .transformation_primitives import build_atomic_primitives, build_parameterized_primitives
        from .perception_primitives import build_perception_primitives
        from .primitives import register_atomic_primitives
        register_atomic_primitives()
        return (build_atomic_primitives()
                + build_perception_primitives()
                + build_parameterized_primitives()
                + self._task_prims)

    def prepare_for_task(self, task: Task) -> None:
        """Reset task-specific state. Atomic mode uses parameterized prims
        + perception for colors — no task-specific primitives needed."""
        self._task_prims = []

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

