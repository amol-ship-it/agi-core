"""
Domain plugin: List Operations.

A domain designed to validate the compounding hypothesis:
does the library actually help solve harder tasks after solving easier ones?

Lists of integers are transformed by compositions of simple operations.
Task difficulty increases with composition depth:
  - Level 1: single ops (reverse, sort, double_all, ...)
  - Level 2: two-step compositions (sort then reverse = reverse sort)
  - Level 3: three-step compositions

This file implements all 4 interfaces for this domain.
It imports ONLY from core — no domain cross-contamination.
"""

from __future__ import annotations
import copy
import random
from typing import Any, Optional

from core import (
    Environment, Grammar, DriveSignal,
    Primitive, Program, Task, Observation,
)


# =============================================================================
# Primitives: atomic list operations
# =============================================================================

def _reverse(lst: list) -> list:
    return list(reversed(lst))

def _sort_asc(lst: list) -> list:
    return sorted(lst)

def _sort_desc(lst: list) -> list:
    return sorted(lst, reverse=True)

def _double_all(lst: list) -> list:
    return [x * 2 for x in lst]

def _increment_all(lst: list) -> list:
    return [x + 1 for x in lst]

def _decrement_all(lst: list) -> list:
    return [x - 1 for x in lst]

def _negate_all(lst: list) -> list:
    return [-x for x in lst]

def _abs_all(lst: list) -> list:
    return [abs(x) for x in lst]

def _square_all(lst: list) -> list:
    return [x * x for x in lst]

def _head(lst: list) -> list:
    """First half of the list."""
    n = (len(lst) + 1) // 2
    return lst[:n]

def _tail(lst: list) -> list:
    """Second half of the list."""
    n = (len(lst) + 1) // 2
    return lst[n:]

def _filter_positive(lst: list) -> list:
    return [x for x in lst if x > 0]

def _filter_negative(lst: list) -> list:
    return [x for x in lst if x < 0]

def _filter_even(lst: list) -> list:
    return [x for x in lst if x % 2 == 0]

def _filter_odd(lst: list) -> list:
    return [x for x in lst if x % 2 != 0]

def _unique(lst: list) -> list:
    """Remove duplicates, preserving order."""
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def _cumsum(lst: list) -> list:
    """Cumulative sum."""
    result = []
    s = 0
    for x in lst:
        s += x
        result.append(s)
    return result

def _diff(lst: list) -> list:
    """Pairwise differences: [a,b,c] → [b-a, c-b]."""
    return [lst[i+1] - lst[i] for i in range(len(lst) - 1)] if len(lst) > 1 else lst

def _deduplicate_consecutive(lst: list) -> list:
    """Remove consecutive duplicates: [1,1,2,2,1] → [1,2,1]."""
    if not lst:
        return lst
    result = [lst[0]]
    for x in lst[1:]:
        if x != result[-1]:
            result.append(x)
    return result

def _identity(lst: list) -> list:
    return list(lst)

def _min_to_front(lst: list) -> list:
    """Move minimum element to front."""
    if not lst:
        return lst
    result = list(lst)
    idx = result.index(min(result))
    result.insert(0, result.pop(idx))
    return result

def _max_to_front(lst: list) -> list:
    """Move maximum element to front."""
    if not lst:
        return lst
    result = list(lst)
    idx = result.index(max(result))
    result.insert(0, result.pop(idx))
    return result


LIST_PRIMITIVES = [
    Primitive("identity",     1, _identity,     domain="list_ops"),
    Primitive("reverse",      1, _reverse,      domain="list_ops"),
    Primitive("sort_asc",     1, _sort_asc,     domain="list_ops"),
    Primitive("sort_desc",    1, _sort_desc,    domain="list_ops"),
    Primitive("double_all",   1, _double_all,   domain="list_ops"),
    Primitive("increment_all",1, _increment_all,domain="list_ops"),
    Primitive("decrement_all",1, _decrement_all,domain="list_ops"),
    Primitive("negate_all",   1, _negate_all,   domain="list_ops"),
    Primitive("abs_all",      1, _abs_all,      domain="list_ops"),
    Primitive("square_all",   1, _square_all,   domain="list_ops"),
    Primitive("head",         1, _head,         domain="list_ops"),
    Primitive("tail",         1, _tail,         domain="list_ops"),
    Primitive("filter_pos",   1, _filter_positive, domain="list_ops"),
    Primitive("filter_neg",   1, _filter_negative, domain="list_ops"),
    Primitive("filter_even",  1, _filter_even,  domain="list_ops"),
    Primitive("filter_odd",   1, _filter_odd,   domain="list_ops"),
    Primitive("unique",       1, _unique,       domain="list_ops"),
    Primitive("cumsum",       1, _cumsum,       domain="list_ops"),
    Primitive("diff",         1, _diff,         domain="list_ops"),
    Primitive("dedup_consec", 1, _deduplicate_consecutive, domain="list_ops"),
    Primitive("min_to_front", 1, _min_to_front, domain="list_ops"),
    Primitive("max_to_front", 1, _max_to_front, domain="list_ops"),
]

_PRIM_MAP = {p.name: p for p in LIST_PRIMITIVES}


# =============================================================================
# Environment
# =============================================================================

class ListEnv(Environment):
    """Execute list transformation programs."""

    def __init__(self):
        self._dynamic: dict[str, Primitive] = {}

    def load_task(self, task: Task) -> Observation:
        return Observation(data=task.train_examples[0][0])

    def execute(self, program: Program, input_data: Any) -> Any:
        if not isinstance(input_data, list):
            return input_data

        # Execute children first (innermost operation runs first)
        result = input_data
        for child in program.children:
            result = self.execute(child, result)

        # Apply this node's operation
        name = program.root
        prim = _PRIM_MAP.get(name) or self._dynamic.get(name)
        if prim and prim.fn:
            try:
                # Library entries have fn=Program (a stored sub-tree).
                # Execute the stored program recursively.
                if isinstance(prim.fn, Program):
                    result = self.execute(prim.fn, result)
                else:
                    result = prim.fn(result)
            except (ValueError, IndexError, ZeroDivisionError):
                return input_data  # safe fallback
        return result

    def register_primitive(self, primitive: Primitive) -> None:
        self._dynamic[primitive.name] = primitive

    def reset(self) -> None:
        self._dynamic.clear()


# =============================================================================
# Grammar
# =============================================================================

class ListGrammar(Grammar):
    """Composition rules for list operations."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def base_primitives(self) -> list[Primitive]:
        return list(LIST_PRIMITIVES)

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def mutate(self, program: Program, primitives: list[Primitive],
               transition_matrix: Any = None) -> Program:
        """Random structural mutation."""
        unary = [p for p in primitives if p.arity <= 1]
        roll = self._rng.random()

        if roll < 0.4:
            # Point mutation: swap root
            new_root = self._rng.choice(unary).name
            return Program(root=new_root, children=list(program.children))
        elif roll < 0.7:
            # Grow: wrap in a new operation
            wrapper = self._rng.choice(unary).name
            return Program(root=wrapper, children=[
                Program(root=program.root, children=list(program.children))])
        else:
            # Shrink: unwrap one level
            if program.children:
                return Program(root=program.children[0].root,
                             children=list(program.children[0].children))
            return Program(root=self._rng.choice(unary).name)

    def crossover(self, a: Program, b: Program) -> Program:
        """Take root from a, children from b (or vice versa)."""
        if self._rng.random() < 0.5:
            return Program(root=a.root, children=list(b.children) if b.children else [])
        return Program(root=b.root, children=list(a.children) if a.children else [])


# =============================================================================
# Drive signal
# =============================================================================

class ListDrive(DriveSignal):
    """Score list predictions by element-wise accuracy."""

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        if not isinstance(predicted, list) or not isinstance(expected, list):
            return 1.0 if predicted != expected else 0.0

        if predicted == expected:
            return 0.0

        # Length mismatch contributes 50% error
        if len(predicted) != len(expected):
            # Partial credit: compare what overlaps
            max_len = max(len(predicted), len(expected), 1)
            min_len = min(len(predicted), len(expected))
            length_penalty = 0.5 * (1 - min_len / max_len)
            # Element accuracy on the overlap
            matches = sum(1 for a, b in zip(predicted, expected) if a == b)
            element_score = matches / max_len if max_len > 0 else 0
            return min(1.0, length_penalty + 0.5 * (1 - element_score))

        # Same length: fraction of wrong elements
        n = len(expected)
        if n == 0:
            return 0.0
        wrong = sum(1 for a, b in zip(predicted, expected) if a != b)
        return wrong / n


# =============================================================================
# Task generation: compositions of known depth
# =============================================================================

# Operations that are safe to compose (don't change list length unpredictably)
_LENGTH_PRESERVING = [
    "reverse", "sort_asc", "sort_desc", "double_all", "increment_all",
    "decrement_all", "negate_all", "abs_all", "square_all",
    "min_to_front", "max_to_front",
]

# All composable ops (including length-changing ones)
_ALL_COMPOSABLE = _LENGTH_PRESERVING + [
    "head", "tail", "filter_pos", "filter_neg", "filter_even",
    "filter_odd", "unique", "cumsum", "diff", "dedup_consec",
]


def _apply_op(name: str, lst: list) -> list:
    """Apply a named primitive to a list."""
    return _PRIM_MAP[name].fn(lst)


def _generate_examples(ops: list[str], n_examples: int = 4,
                        rng: random.Random = None) -> list[tuple[list, list]]:
    """Generate (input, output) pairs for a composition of ops.

    ops is [innermost, ..., outermost], so the program is
    outermost(... (innermost(input))).
    """
    if rng is None:
        rng = random.Random(42)

    examples = []
    for _ in range(n_examples):
        # Generate a random input list
        length = rng.randint(4, 8)
        inp = [rng.randint(-10, 10) for _ in range(length)]

        # Apply operations innermost to outermost
        result = list(inp)
        for op in ops:
            result = _apply_op(op, result)

        # Skip degenerate examples (empty output)
        if len(result) == 0:
            continue
        examples.append((inp, result))

    return examples


def get_sample_tasks(seed: int = 42) -> list[Task]:
    """Generate a curriculum of list tasks at increasing difficulty.

    Level 1 (depth 1): 8 single-operation tasks
    Level 2 (depth 2): 12 two-step composition tasks
    Level 3 (depth 3): 8 three-step composition tasks

    Total: 28 tasks. With exhaustive_depth=2, level 1 and 2 should be
    solvable. Level 3 tests whether the library (from solved level 1+2 tasks)
    enables solving deeper compositions.
    """
    rng = random.Random(seed)
    tasks = []

    # Level 1: single operations
    level1_ops = [
        ["reverse"], ["sort_asc"], ["sort_desc"], ["double_all"],
        ["increment_all"], ["negate_all"], ["abs_all"], ["unique"],
    ]
    for i, ops in enumerate(level1_ops):
        examples = _generate_examples(ops, n_examples=4, rng=rng)
        if len(examples) < 2:
            continue
        train = examples[:3]
        test_in = [examples[-1][0]]
        test_out = [examples[-1][1]]
        tasks.append(Task(
            task_id=f"list_L1_{ops[0]}",
            train_examples=train,
            test_inputs=test_in,
            test_outputs=test_out,
            difficulty=1.0,
        ))

    # Level 2: two-step compositions
    level2_ops = [
        ["reverse", "sort_asc"],         # sort then reverse = sort_desc (interesting: same result as sort_desc)
        ["double_all", "sort_asc"],       # double all then sort
        ["increment_all", "reverse"],     # increment then reverse
        ["abs_all", "sort_asc"],          # abs then sort
        ["negate_all", "reverse"],        # negate then reverse
        ["filter_pos", "sort_asc"],       # filter positive then sort
        ["unique", "reverse"],            # unique then reverse
        ["sort_asc", "head"],             # sort then take first half
        ["double_all", "increment_all"],  # double then increment (2x+1)
        ["negate_all", "abs_all"],        # negate then abs (= abs)
        ["sort_desc", "diff"],            # sort descending then diff
        ["cumsum", "reverse"],            # cumsum then reverse
    ]
    for i, ops in enumerate(level2_ops):
        examples = _generate_examples(ops, n_examples=5, rng=rng)
        if len(examples) < 3:
            continue
        train = examples[:4]
        test_in = [examples[-1][0]]
        test_out = [examples[-1][1]]
        name = "_then_".join(ops)
        tasks.append(Task(
            task_id=f"list_L2_{name}",
            train_examples=train,
            test_inputs=test_in,
            test_outputs=test_out,
            difficulty=2.0,
        ))

    # Level 3: three-step compositions (the compounding test)
    # These require depth-3 programs or a library entry + depth-1/2
    level3_ops = [
        ["reverse", "sort_asc", "double_all"],       # reverse, sort, double
        ["abs_all", "sort_asc", "reverse"],           # abs, sort, reverse
        ["increment_all", "double_all", "sort_asc"],  # inc, double, sort
        ["filter_pos", "sort_asc", "reverse"],        # filter+, sort, rev
        ["unique", "sort_asc", "double_all"],         # unique, sort, double
        ["negate_all", "abs_all", "sort_desc"],       # negate, abs, sort_desc
        ["double_all", "increment_all", "reverse"],   # double, inc, reverse
        ["abs_all", "sort_desc", "head"],             # abs, sort_desc, head
    ]
    for i, ops in enumerate(level3_ops):
        examples = _generate_examples(ops, n_examples=5, rng=rng)
        if len(examples) < 3:
            continue
        train = examples[:4]
        test_in = [examples[-1][0]]
        test_out = [examples[-1][1]]
        name = "_then_".join(ops)
        tasks.append(Task(
            task_id=f"list_L3_{name}",
            train_examples=train,
            test_inputs=test_in,
            test_outputs=test_out,
            difficulty=3.0,
        ))

    return tasks
