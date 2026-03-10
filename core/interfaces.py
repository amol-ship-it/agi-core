"""
The 4 pluggable interfaces of the Universal Learning Loop.

These are the ONLY abstractions that domain-specific code must implement.
The core loop (learner.py) depends on nothing else.

Data types (Primitive, Program, Task, etc.) live in core/types.py.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

from .types import (
    Primitive,
    Program,
    Observation,
    Task,
    ScoredProgram,
    LibraryEntry,
)


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

    def prepare_for_task(self, task: Task) -> None:
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
