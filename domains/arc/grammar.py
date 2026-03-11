"""
ARC-AGI Grammar: composition rules for grid transformation programs.
"""

from __future__ import annotations

import copy
import random

from core import Grammar, Primitive, Program, Task
from .primitives import (
    ARC_PRIMITIVES, ARC_PREDICATES, _PRIM_MAP, register_prim,
    _make_replace_color,
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
        """Analyze training examples to create task-specific color primitives."""
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

        # Register task prims in _PRIM_MAP for execution
        for p in self._task_prims:
            _PRIM_MAP[p.name] = p

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def mutate(self, program: Program, primitives: list[Primitive]) -> Program:
        """Mutate a program: point (swap label), grow (leaf→subtree), or shrink (subtree→leaf).

        Point-only mutations can never change tree structure, so grow/shrink
        are essential for discovering programs that require different depths
        than the initial random beam provides.
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
                    new_op = self._rng.choice(higher)
                    leaf_prims = [p for p in primitives if p.arity <= 1]
                    if not leaf_prims:
                        leaf_prims = primitives
                    children = [
                        Program(root=self._rng.choice(leaf_prims).name)
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
                    new_leaf = self._rng.choice(leaf_prims)
                    target.root = new_leaf.name
                    target.children = []
                    target.params = {}
            return prog
        else:
            # POINT: swap label with same-arity primitive (preserves structure)
            target = self._rng.choice(nodes)
            prim = _PRIM_MAP.get(target.root)
            current_arity = prim.arity if prim else 1

            same_arity = [p for p in primitives if p.arity == current_arity]
            if same_arity:
                new_prim = self._rng.choice(same_arity)
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

