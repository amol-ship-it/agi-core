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
from core.types import SearchStratum
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
        """Enable structural search phases (per-object, cross-reference, etc.)."""
        return True

    def get_predicates(self) -> list[tuple[str, callable]]:
        """Return input->bool predicates for conditional search."""
        from collections import Counter
        from .objects import _find_connected_components

        def _safe_components(grid):
            try:
                return _find_connected_components(grid)
            except Exception:
                return []

        def _n_foreground_colors(g):
            if not g or not g[0]:
                return 0
            bg = Counter(c for row in g for c in row).most_common(1)[0][0]
            return len({c for row in g for c in row if c != bg})

        return [
            ("has_single_object", lambda g: len(_safe_components(g)) == 1),
            ("has_many_objects", lambda g: len(_safe_components(g)) > 3),
            ("is_square", lambda g: len(g) == len(g[0]) if g and g[0] else False),
            ("is_tall", lambda g: len(g) > len(g[0]) if g and g[0] else False),
            ("is_wide", lambda g: len(g) < len(g[0]) if g and g[0] else False),
            ("has_symmetry_h", lambda g: all(
                g[r] == g[r][::-1] for r in range(len(g))) if g else False),
            ("is_mostly_bg", lambda g: sum(
                1 for row in g for c in row if c == 0) > len(g) * len(g[0]) * 0.7
                if g and g[0] else False),
            ("has_symmetry_v", lambda g: all(
                g[r][c] == g[len(g) - 1 - r][c]
                for r in range(len(g)) for c in range(len(g[0])))
                if g and g[0] else False),
            ("is_small_grid", lambda g: len(g) * len(g[0]) < 100
                if g and g[0] else True),
            ("has_few_colors", lambda g: _n_foreground_colors(g) <= 2),
            ("has_many_colors", lambda g: _n_foreground_colors(g) > 4),
            ("all_objects_same_size", lambda g: len(set(
                comp["size"] for comp in _safe_components(g))) <= 1),
        ]

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

    def propose_strata(
        self, task: Task, primitives: list[Primitive]
    ) -> list[SearchStratum]:
        """Propose search strata based on task fingerprint analysis.

        Returns focused strata with primitive subsets and budget allocations.
        Always includes exhaustive_core (40% budget, all primitives).
        Triggered strata share the remaining 60% equally.
        """
        from .fingerprint import fingerprint_task

        fp = fingerprint_task(task)
        all_names = [p.name for p in primitives]

        # --- Define stratum triggers and their primitive filters ---
        # Each entry: (name, condition, name_substrings, metadata)
        _STRATUM_DEFS: list[tuple[str, bool, list[str], dict]] = [
            (
                "inpainting",
                fp.has_holes or fp.symmetry_broken or fp.has_periodic,
                ["inpaint", "symmetry", "fill", "mirror"],
                {"run_cross_ref": True},
            ),
            (
                "separator_algebra",
                fp.has_separators,
                ["separator", "section", "overlay", "split", "remove"],
                {"run_cross_ref": True},
            ),
            (
                "object_transform",
                fp.n_objects >= 2 and fp.dim_change == "same",
                ["object", "rotate", "mirror", "flip", "transpose", "move", "gravity"],
                {"run_object_decomp": True, "run_for_each_object": True, "run_procedural": True},
            ),
            (
                "object_extraction",
                fp.n_objects >= 2 and fp.dim_change == "shrink",
                ["extract", "crop", "largest", "object", "densest", "unique", "content"],
                {},
            ),
            (
                "local_rules",
                fp.dim_change == "same" and 0.0 <= fp.pixel_diff_ratio < 0.5,
                ["fill", "flood", "neighbor", "connect", "extend", "dilate", "erode", "outline"],
                {"run_local_rules": True},
            ),
            (
                "tiling_scaling",
                fp.dim_change == "grow" or fp.output_is_subgrid,
                ["tile", "scale", "mirror_tile", "repeat", "period"],
                {},
            ),
            (
                "color_logic",
                fp.is_recoloring or fp.colors_added > 0,
                ["color", "recolor", "swap", "replace", "keep", "erase", "binarize", "invert"],
                {},
            ),
            (
                "pattern_completion",
                fp.symmetry_broken or fp.has_periodic,
                ["symmetry", "inpaint", "mirror", "period", "tile", "fill"],
                {},
            ),
            (
                "line_drawing",
                fp.n_objects >= 2 and fp.colors_added > 0,
                ["connect", "extend", "line", "ray", "draw"],
                {},
            ),
            (
                "template_stamping",
                fp.n_sections >= 2 or (fp.n_objects >= 2 and fp.object_size_var < 1.0),
                ["overlay", "stamp", "tile", "mask", "section", "template"],
                {},
            ),
            (
                "denoising",
                0.0 <= fp.pixel_diff_ratio < 0.1 and fp.dim_change == "same",
                ["fill", "erode", "dilate", "flood", "compress", "unique", "sort"],
                {},
            ),
        ]

        # --- Build triggered strata ---
        triggered: list[tuple[str, list[str], dict]] = []
        for name, condition, substrings, metadata in _STRATUM_DEFS:
            if condition:
                # Filter primitives by substring match
                filtered = [
                    n for n in all_names
                    if any(sub in n.lower() for sub in substrings)
                ]
                # Include at least some primitives; fall back to all if filter empty
                if filtered:
                    triggered.append((name, filtered, metadata))

        # --- Build strata list ---
        strata: list[SearchStratum] = []

        # 1. exhaustive_core: always present, 40% budget, all primitives
        core_budget = 0.40
        strata.append(SearchStratum(
            name="exhaustive_core",
            primitive_names=list(all_names),
            budget_fraction=core_budget,
        ))

        # 2. Triggered strata share remaining 60%
        remaining = 1.0 - core_budget
        if triggered:
            raw_share = remaining / len(triggered)
            # Clamp to [0.05, 0.30]
            clamped = max(0.05, min(0.30, raw_share))
            total_clamped = clamped * len(triggered)
            # Normalize so total = remaining
            scale = remaining / total_clamped if total_clamped > 0 else 1.0

            for name, prim_names, metadata in triggered:
                budget = clamped * scale
                strata.append(SearchStratum(
                    name=name,
                    primitive_names=prim_names,
                    budget_fraction=budget,
                    metadata=metadata,
                ))
        else:
            # No triggers: give remaining budget to core
            strata[0] = SearchStratum(
                name="exhaustive_core",
                primitive_names=list(all_names),
                budget_fraction=1.0,
            )

        return strata

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
