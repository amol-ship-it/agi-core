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

    # Exhaustive enumeration: try ALL programs up to this depth before beam search.
    # depth 1 = all single primitives, depth 2 = all pairs, depth 3 = all triples.
    # Set to 0 to disable. Enumeration is cheap for depth <= 2 (N + N² programs).
    exhaustive_depth: int = 3
    # For depth 2+, limit inner primitives to top-K ranked by performance.
    # Depth-3 cost is N×K (not K³) due to smart subtree reuse.
    exhaustive_top_k: int = 20


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
    # Within-run sequential compounding: process tasks one at a time,
    # immediately promoting solved programs to the library so the next
    # task benefits. When True, overrides workers to 1.
    sequential_compounding: bool = False
