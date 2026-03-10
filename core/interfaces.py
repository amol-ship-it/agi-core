"""
The 4 pluggable interfaces of the Universal Learning Loop.

These are the ONLY abstractions that domain-specific code must implement.
The core loop (learner.py) depends on nothing else.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Data types shared across all interfaces
# =============================================================================

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


# =============================================================================
# Interface 1: Environment (sensors + actuators)
# =============================================================================

class Environment(ABC):
    """
    The world the agent interacts with.

    Provides observations, accepts programs/actions, returns consequences.
    This is the feedback loop — predictions are tested against reality.
    """

    @abstractmethod
    def load_task(self, task: Task) -> Observation:
        """Present a task to the environment, get initial observation."""
        ...

    @abstractmethod
    def execute(self, program: Program, input_data: Any) -> Any:
        """
        Run a candidate program on an input and return the output.

        For ARC: apply a grid transformation program to an input grid.
        For symbolic regression: evaluate a formula on input x values.
        For Zork: execute an action sequence in the game state.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset environment state between tasks."""
        ...


# =============================================================================
# Interface 2: Grammar (primitives + composition rules)
# =============================================================================

class Grammar(ABC):
    """
    The building blocks and composition rules.

    Defines what primitives exist and how they can be combined.
    This is the abstraction & composability pillar.
    """

    @abstractmethod
    def base_primitives(self) -> list[Primitive]:
        """
        Return the hand-coded primitives for this domain.

        ARC: rotate90, flip_y, crop_to_box, replace_color, ...
        Symbolic math: sin, cos, add, multiply, square, ...
        Zork: move, take, use, attack, open, ...
        """
        ...

    @abstractmethod
    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        """Build a new program by applying a primitive to sub-programs."""
        ...

    @abstractmethod
    def mutate(self, program: Program, primitives: list[Primitive]) -> Program:
        """
        Produce a variant of a program by random structural change.

        E.g., swap a node, add a node, remove a node, change a constant.
        """
        ...

    @abstractmethod
    def crossover(self, a: Program, b: Program) -> Program:
        """Combine sub-trees from two programs into a new program."""
        ...

    def prepare_for_task(self, task: "Task") -> None:
        """Called before the wake phase begins on a task.

        Grammars can use this to cache task-specific data (e.g. training
        examples for constant optimization).  Default: no-op.
        """

    def inject_library(self, entries: list[LibraryEntry]) -> list[Primitive]:
        """
        Convert library entries into primitives usable by the grammar.

        This is how the library expands the effective vocabulary.
        Default: wrap each entry's program as a new Primitive.
        """
        learned = []
        for entry in entries:
            p = Primitive(
                name=entry.name,
                arity=0,  # learned primitives are pre-composed
                fn=entry.program,
                domain=entry.domain,
                learned=True,
            )
            learned.append(p)
        return learned


# =============================================================================
# Interface 3: DriveSignal (energy / scoring function)
# =============================================================================

class DriveSignal(ABC):
    """
    Tells the system if it's getting warmer or colder.

    Combines prediction accuracy with complexity cost.
    This is the approximability pillar.
    """

    @abstractmethod
    def prediction_error(self, predicted: Any, expected: Any) -> float:
        """
        How wrong is the prediction?

        ARC: pixel edit distance (fraction of cells that differ).
        Symbolic math: mean squared error.
        Zork: negative game score / distance from goal.
        """
        ...

    def complexity_cost(self, program: Program) -> float:
        """
        How complex is the program? Default: node count.

        Operationalizes Occam's Razor / MDL.
        Override for domain-specific complexity measures.
        """
        return float(program.size)

    def energy(
        self,
        program: Program,
        predicted: Any,
        expected: Any,
        alpha: float = 1.0,
        beta: float = 0.001,
    ) -> tuple[float, float, float]:
        """
        E(candidate) = α · prediction_error + β · complexity_cost

        Returns (total_energy, pred_error, complexity).
        """
        pred_err = self.prediction_error(predicted, expected)
        comp_cost = self.complexity_cost(program)
        total = alpha * pred_err + beta * comp_cost
        return total, pred_err, comp_cost


# =============================================================================
# Interface 4: Memory (episodic + library + program store)
# =============================================================================

class Memory(ABC):
    """
    The organism's memory systems.

    Stores episodic experiences, the abstraction library,
    and the program/policy library.
    """

    # --- Episodic memory ---

    @abstractmethod
    def record_episode(self, task_id: str, observation: Any, program: Optional[Program], score: float) -> None:
        """Store a task attempt for later replay."""
        ...

    @abstractmethod
    def replay_episodes(self, n: int = 10) -> list[dict]:
        """Retrieve recent episodes for offline learning."""
        ...

    # --- Abstraction library ---

    @abstractmethod
    def get_library(self) -> list[LibraryEntry]:
        """Return all learned abstractions."""
        ...

    @abstractmethod
    def add_to_library(self, entry: LibraryEntry) -> None:
        """Add a new learned abstraction."""
        ...

    @abstractmethod
    def update_usefulness(self, name: str, delta: float) -> None:
        """Update the usefulness score of a library entry after reuse."""
        ...

    # --- Program library (solved tasks) ---

    @abstractmethod
    def store_solution(self, task_id: str, scored: ScoredProgram) -> None:
        """Store a successful solution for a task."""
        ...

    @abstractmethod
    def get_solutions(self) -> dict[str, ScoredProgram]:
        """Return all stored solutions, keyed by task_id."""
        ...

    # --- Persistence ---

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist memory to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load memory from disk."""
        ...
