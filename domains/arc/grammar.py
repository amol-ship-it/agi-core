"""
ARC-AGI Grammar: composition rules for grid transformation programs.

Composition model:
  Programs are trees. Internal nodes are primitives; children are sub-programs.
  Evaluation is recursive: _eval_tree in environment.py interprets the tree.

  - Transform nodes (arity 1-2): evaluate children as grids, apply transform
  - Parameterized nodes: evaluate perception children as values, build transform
  - Perception nodes (arity 0): extract a value from the input grid

Search operators: mutate (point/grow/shrink) and crossover (subtree swap).

STRIPPED TO CORE: Empty base primitives. Predicates, task priority, and
analysis integration removed. Will be added back when justified by tasks.
"""

from __future__ import annotations

import copy
import random

from typing import Any

from core import Grammar, Primitive, Program, Task
from .primitives import _PRIM_MAP, Grid


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
        """Structural phases disabled until justified by specific tasks."""
        return False

    def get_predicates(self) -> list[tuple[str, callable]]:
        """Return input→bool predicates. Currently empty."""
        return []

    def essential_pair_concepts(self) -> frozenset[str]:
        from .transformation_primitives import ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        return ATOMIC_ESSENTIAL_PAIR_CONCEPTS

    def task_priority_primitives(self, task: Task) -> list[str]:
        """Return primitives likely relevant for this task. Currently empty."""
        return []

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
        """Reset task-specific state."""
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
        """Mutate a program: point (swap label), grow (leaf→subtree), or shrink (subtree→leaf)."""
        prog = copy.deepcopy(program)
        nodes = self._collect_nodes(prog)
        if not nodes:
            return prog

        r = self._rng.random()
        if r < 0.20:
            # GROW: replace a leaf with a small subtree
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
            # SHRINK: replace a non-leaf subtree with a leaf
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
            # POINT: swap label with same-arity primitive
            target = self._rng.choice(nodes)
            prim = _PRIM_MAP.get(target.root)
            current_arity = prim.arity if prim else 1

            parent_op = ""
            if program.children:
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
