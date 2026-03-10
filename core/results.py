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
    solved: bool
    best: Optional[ScoredProgram]
    generations_used: int
    evaluations: int = 0    # total program evaluations (deterministic compute measure)
    wall_time: float = 0.0
    pareto_front: list[ParetoEntry] = field(default_factory=list)
    dedup_count: int = 0    # how many duplicates were removed


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
    tasks_solved: int
    tasks_total: int
    solve_rate: float
    cumulative_library_size: int
