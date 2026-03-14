"""
The 4 pluggable interfaces of the Universal Learning Loop.

These are the ONLY abstractions that domain-specific code must implement.
The core loop (learner.py) depends on nothing else.

Data types (Primitive, Program, Task, etc.) live in core/types.py.

The Grammar interface embodies Pillar 3 (Abstraction & Composition).
Decomposition is the flip side of composition: complex inputs are broken
into parts, each part is solved independently, and results are recomposed.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

from .types import (
    Primitive,
    Program,
    Observation,
    Task,
    Decomposition,
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

    def register_primitive(self, primitive: "Primitive") -> None:
        """Register a dynamically created primitive for execution.

        Used by the learner to register conditional or other synthesized
        primitives so execute() can resolve them.

        Default: no-op (environments that don't need dynamic registration).
        """

    def try_object_decomposition(
        self,
        task: "Task",
        primitives: list["Primitive"],
    ) -> Optional[tuple[str, Any]]:
        """Try solving a task by applying transforms per-object.

        For domains with discrete objects (like ARC grids), this tries
        applying each primitive to individual objects and reassembling.

        Returns (name, transform_fn) if a pixel-perfect decomposition
        is found, or None.

        Default: no decomposition (returns None).
        """
        return None

    def try_for_each_object(
        self,
        task: "Task",
        candidate_programs: list["ScoredProgram"],
        top_k: int = 10,
    ) -> Optional[tuple[str, Any]]:
        """Try applying top-K candidate programs per-object.

        Unlike try_object_decomposition (which tries individual primitives),
        this takes already-scored programs (including depth-2+ compositions)
        and applies each per-object. This enables compositions like
        for_each_object(mirror_h(crop_to_nonzero)).

        Returns (name, transform_fn) if a pixel-perfect decomposition
        is found, or None.

        Default: no per-object support (returns None).
        """
        return None

    def try_conditional_per_object(
        self,
        task: "Task",
        candidate_programs: list["ScoredProgram"],
        predicates: list,
        top_k: int = 8,
    ) -> Optional[tuple[str, Any]]:
        """Try per-object conditional transforms: if(pred, A, B) per object.

        Default: not supported (returns None).
        """
        return None

    def try_cross_reference(
        self,
        task: "Task",
        primitives: list["Primitive"],
    ) -> Optional[tuple[str, Any]]:
        """Try solving a task using cross-reference: one grid part informs another.

        Common ARC patterns:
        - A small pattern acts as a mask/template for a larger region
        - Grid divided by separators; one cell is the "key" that transforms others
        - An object's color/shape determines how another object is modified

        Returns (name, transform_fn) or None.

        Default: no cross-reference support (returns None).
        """
        return None

    def infer_output_correction(
        self,
        program_outputs: list[Any],
        expected_outputs: list[Any],
        **kwargs,
    ) -> Optional[Program]:
        """Given a program's outputs and expected outputs, try to infer a
        simple correction transform (e.g., color remapping for ARC grids).

        Returns a Program node that should be composed on top of the original
        program, or None if no consistent correction is found.

        Keyword args are domain-specific (e.g., max_rules for ARC).

        Default: no correction (returns None).
        """
        return None


# =============================================================================
# Interface 2: Grammar (primitives + composition rules)
# =============================================================================

class Grammar(ABC):
    """
    The building blocks, composition rules, and decomposition strategies.

    Defines what primitives exist, how they can be combined (composition),
    and how complex inputs can be broken into simpler parts (decomposition).

    Composition and decomposition are duals:
    - Composition: simple parts → complex whole (synthesis)
    - Decomposition: complex whole → simple parts (analysis)

    This is Pillar 3: Abstraction & Composability.
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
    def mutate(self, program: Program, primitives: list[Primitive],
               transition_matrix: "Any | None" = None) -> Program:
        """
        Produce a variant of a program by random structural change.

        E.g., swap a node, add a node, remove a node, change a constant.

        If transition_matrix is provided, use it to bias primitive choices
        toward known-good compositions (DreamCoder-style prior).
        """
        ...

    @abstractmethod
    def crossover(self, a: Program, b: Program) -> Program:
        """Combine sub-trees from two programs into a new program."""
        ...

    def get_predicates(self) -> list[tuple[str, Any]]:
        """Return (name, callable) pairs of predicate functions.

        Predicates are input→bool functions used for conditional branching:
        if pred(input) then A(input) else B(input).

        Default: no predicates.
        """
        return []

    def essential_pair_concepts(self) -> frozenset[str]:
        """Return names of primitives that should always be included in pair/triple
        exhaustive search, even if they rank low individually.

        These are structural transforms (crop, fill, compress, etc.) that rarely
        score well alone but are critical as second or third steps. Each domain
        should override this with its own set of known-important compositions.

        Default: empty (no forced inclusions).
        """
        return frozenset()

    def task_priority_primitives(self, task: Task) -> list[str]:
        """Return primitives likely relevant for this specific task.

        Based on structural analysis of the task's inputs, returns primitive
        names that should be boosted in pool construction. Unlike
        essential_pair_concepts (static), these are task-adaptive.

        Default: empty (no task-specific priorities).
        """
        return []

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

    # --- Decomposition (inverse of composition) ---

    def decompose(self, input_data: Any, task: Task) -> list[Decomposition]:
        """Decompose an input into structured parts for independent solving.

        This is the inverse of composition: given a complex input, propose
        ways to break it into simpler sub-problems. Each Decomposition
        contains the parts and context needed for reassembly.

        Multiple decomposition strategies may be returned (e.g., for ARC:
        same-color objects, multi-color objects, grid partitions). The
        learner tries each and picks the best.

        This embodies the universal principle: complex problems are solved
        by decomposing into sub-problems, solving each, and recomposing.

        Default: no decomposition (returns empty list).

        Examples:
          ARC: grid → [object_subgrids] with positions for reassembly
          Symbolic regression: expression → [sub-expressions]
          Planning: goal → [subgoals]
        """
        return []

    def recompose(self, decomposition: Decomposition,
                  transformed_parts: list[Any]) -> Any:
        """Reassemble transformed parts back into a complete output.

        Given a decomposition and the independently-transformed parts,
        produce the final output. This is the composition step that
        follows decomposition.

        Default: returns the first part (identity).
        """
        return transformed_parts[0] if transformed_parts else None


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

    def prune_library(self, min_usefulness: float = 0.01) -> int:
        """Remove library entries with usefulness below threshold and no reuse.

        Returns the number of entries pruned.
        Default: no-op (returns 0).
        """
        return 0

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
