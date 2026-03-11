"""
Result dataclasses returned by wake, sleep, and curriculum phases.

Depends only on core/types.py — no imports from learner or interfaces.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .types import Program, ScoredProgram, LibraryEntry


@dataclass
class ParetoEntry:
    """One point on the Pareto front: best program at a given complexity."""
    complexity: int           # node count
    prediction_error: float
    energy: float
    program: Program

    def __repr__(self):
        return (f"ParetoEntry(complexity={self.complexity}, "
                f"error={self.prediction_error:.6g}, program={self.program})")


@dataclass
class WakeResult:
    """What comes out of one wake phase."""
    task_id: str
    train_solved: bool  # matched all training examples (error < threshold)
    best: Optional[ScoredProgram]
    generations_used: int
    evaluations: int = 0    # total program evaluations (deterministic compute measure)
    wall_time: float = 0.0
    pareto_front: list[ParetoEntry] = field(default_factory=list)
    dedup_count: int = 0    # how many duplicates were removed
    test_error: Optional[float] = None  # error on held-out test examples (None if unavailable)
    test_solved: Optional[bool] = None  # did the program solve the held-out test examples?

    @property
    def solved(self) -> bool:
        """A task is 'solved' only when its test examples are solved.

        Falls back to train_solved when no test data is available.
        """
        if self.test_solved is not None:
            return self.test_solved
        return self.train_solved


@dataclass
class SleepResult:
    """What comes out of one sleep phase."""
    new_entries: list[LibraryEntry]
    library_size_before: int
    library_size_after: int
    wall_time: float


@dataclass
class RoundResult:
    """What comes out of one full wake-sleep round."""
    round_number: int
    wake_results: list[WakeResult]
    sleep_result: SleepResult
    train_solved: int      # tasks matching training examples
    tasks_total: int
    train_solve_rate: float
    cumulative_library_size: int

    @property
    def solved(self) -> int:
        """Tasks solved = test-verified (falls back to train when no test data)."""
        return sum(1 for w in self.wake_results if w.solved)

    @property
    def solve_rate(self) -> float:
        """Solve rate = test-verified (falls back to train when no test data)."""
        return self.solved / self.tasks_total if self.tasks_total > 0 else 0.0
