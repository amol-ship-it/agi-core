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
    # Near-miss refinement: try appending/prepending primitives to programs
    # with prediction_error < this threshold. High-ROI: catches "almost right"
    # programs that need one more step (e.g. a color fix or crop).
    near_miss_threshold: float = 0.20

    # Exhaustive enumeration: try ALL programs up to this depth before beam search.
    # depth 1 = all single primitives, depth 2 = all pairs, depth 3 = all triples.
    # Set to 0 to disable.
    exhaustive_depth: int = 3
    # Pair exhaustion: top-K singles (by individual score) + essential structural
    # concepts form the pair pool. Both steps drawn from this pool → K² combos.
    # Wider K catches solutions where the first step scores low individually.
    exhaustive_pair_top_k: int = 40
    # Triple exhaustion: top-K singles + essential concepts → K³ combos.
    # Smaller K (15) keeps cost manageable: ~15% of ARC needs exactly 3 steps.
    exhaustive_triple_top_k: int = 15

    # Per-task eval budget (0 = unlimited). When > 0, expensive phases
    # (beam search, near-miss refinement) are skipped once n_evals exceeds
    # this. Set by the benchmark runner using cell-normalized compute cap.
    eval_budget: int = 0

    # Base cell size for per-task compute cap normalization.
    # 800 = median ARC grid size. Domains with different scale should override.
    eval_budget_base_cells: int = 800

    # Verbose worker output (per-task diagnostic prints). Set False in batch mode.
    verbose: bool = True


@dataclass
class SleepConfig:
    """Knobs for the sleep/consolidation phase."""
    min_occurrences: int = 1      # sub-tree must appear in >= N programs
    min_size: int = 2             # sub-tree must have >= N nodes
    max_library_size: int = 200   # cap on total library entries
    usefulness_decay: float = 0.90  # decay old entries each sleep cycle
    reuse_bonus: float = 2.0       # scoring bonus per reuse for eviction ranking
    unsolved_weight: float = 0.5  # quality discount for unsolved vs solved programs


@dataclass
class CurriculumConfig:
    """Knobs for curriculum-ordered learning."""
    sort_by_difficulty: bool = False
    wake_sleep_rounds: int = 3
    workers: int = 0  # 0 = auto-detect (performance cores), 1 = sequential
    # Within-run sequential compounding: process tasks one at a time,
    # immediately promoting solved programs to the library so the next
    # task benefits. When True, overrides workers to 1.
    sequential_compounding: bool = False
    # Adaptive compute reallocation: after each round, re-run close-to-solved
    # tasks (error < SearchConfig.near_miss_threshold) with boosted budget.
    adaptive_realloc: bool = False
    adaptive_realloc_budget_multiplier: float = 3.0
    adaptive_realloc_pair_top_k_boost: int = 20  # added to base pair_top_k
    adaptive_realloc_triple_top_k_boost: int = 10  # added to base triple_top_k
