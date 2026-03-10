"""
Configuration dataclasses for the Universal Learning Loop.

Pure data — no dependencies on other core modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchConfig:
    """Knobs for the beam search. Domain-agnostic."""
    beam_width: int = 200         # candidates kept per generation
    mutations_per_candidate: int = 3
    crossover_fraction: float = 0.3
    max_generations: int = 100    # compute budget = beam_width × max_generations
    energy_alpha: float = 1.0     # weight on prediction error
    energy_beta: float = 0.001    # weight on complexity cost
    early_stop_energy: float = 0.0  # stop if energy <= this (perfect solve)
    solve_threshold: float = 1e-4   # prediction_error <= this counts as solved
    seed: Optional[int] = None
    semantic_dedup: bool = True     # deduplicate beam by output vector
    dedup_precision: int = 6        # decimal places for output hashing


@dataclass
class SleepConfig:
    """Knobs for the sleep/consolidation phase."""
    min_occurrences: int = 2      # sub-tree must appear in >= N solutions
    min_size: int = 2             # sub-tree must have >= N nodes
    max_library_size: int = 500   # cap on total library entries
    usefulness_decay: float = 0.95  # decay old entries each sleep cycle


@dataclass
class CurriculumConfig:
    """Knobs for curriculum-ordered learning."""
    sort_by_difficulty: bool = True
    wake_sleep_rounds: int = 3
    workers: int = 0  # 0 = auto-detect (performance cores), 1 = sequential
