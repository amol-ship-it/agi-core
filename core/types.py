"""
Foundational data types for the Universal Learning Loop.

These are the vocabulary shared by all modules in core/.
Pure dataclasses with no dependencies on ABCs or algorithms.
This module is the leaf of the import DAG — it imports nothing from core/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Primitive:
    """An atomic operation in a grammar. The smallest unit of composition."""
    name: str
    arity: int  # how many arguments it takes
    fn: Any     # the callable that implements it
    domain: str = ""  # which domain contributed this primitive
    learned: bool = False  # True if discovered by sleep phase, False if hand-coded

    def __repr__(self):
        tag = " [learned]" if self.learned else ""
        return f"Primitive({self.name}/{self.arity}{tag})"


@dataclass
class Program:
    """
    A composed expression: a tree of primitives applied to arguments.

    This is the universal representation of a hypothesis, a skill,
    a transformation, a policy — anything the system synthesizes.
    """
    root: str                          # name of the primitive at this node
    children: list[Program] = field(default_factory=list)  # sub-expressions
    params: dict[str, float] = field(default_factory=dict) # fitted constants

    @property
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        """Total number of nodes. Used as complexity measure."""
        return 1 + sum(c.size for c in self.children)

    def __repr__(self):
        if not self.children:
            return self.root
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.root}({args})"


@dataclass
class Observation:
    """What the environment gives back after an action (or at the start)."""
    data: Any           # domain-specific: a grid, a number, game text, etc.
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """
    A single problem to solve.

    For ARC: input/output grid pairs (train) + test inputs.
    For symbolic regression: (x, y) data points.
    For Zork: initial game state + goal description.
    """
    task_id: str
    train_examples: list[tuple[Any, Any]]  # (input, expected_output) pairs
    test_inputs: list[Any]                 # inputs to predict
    test_outputs: list[Any] = field(default_factory=list)  # ground truth if available
    difficulty: float = 0.0                # estimated difficulty for curriculum
    metadata: dict = field(default_factory=dict)


@dataclass
class Decomposition:
    """A structured decomposition of an input into parts.

    Decomposition is the inverse of composition: given a complex input,
    break it into simpler parts that can be independently transformed
    and then reassembled.

    This is the core abstraction for "divide and conquer" — the universal
    principle that complex problems can be solved by decomposing into
    sub-problems, solving each, and recomposing.

    For ARC: grid → [object_subgrids], with reassembly info
    For symbolic regression: expression → sub-expressions
    For planning: goal → subgoals
    """
    strategy: str                    # name of the decomposition strategy
    parts: list[Any]                 # the sub-problems (subgrids, sub-expressions, etc.)
    context: dict = field(default_factory=dict)  # reassembly info (positions, background, etc.)

    @property
    def n_parts(self) -> int:
        return len(self.parts)


@dataclass
class ScoredProgram:
    """A program together with its evaluation metrics."""
    program: Program
    energy: float           # total energy = error + complexity cost
    prediction_error: float
    complexity_cost: float
    task_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LibraryEntry:
    """A learned abstraction stored in the library."""
    name: str
    program: Program        # the sub-tree this entry compresses
    usefulness: float       # how much it reduced search cost
    reuse_count: int = 0    # times reused in subsequent solutions
    source_tasks: list[str] = field(default_factory=list)  # which tasks it came from
    domain: str = ""        # which domain it was learned in ("" = cross-domain)
