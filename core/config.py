"""
Configuration dataclasses for the Universal Learning Loop.

Pure data — no dependencies on other core modules.
"""

import math
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Auto-derivation: compute budget → search parameters
# =============================================================================

def derive_search_params(eval_budget: int, n_prims: int = 48) -> dict:
    """Derive search parameters from per-task eval budget.

    Allocates budget to highest-ROI phases first:
    1. Depth-1 exhaustive (always, ~n_prims evals)
    2. Structural phases (always, ~1000 evals)
    3. Depth-2 pairs (pair_top_k² × 0.6 evals)
    4. Depth-3 triples (triple_top_k³ evals)
    """
    # Fixed costs (always allocated)
    fixed = n_prims + 1000  # depth-1 + structural
    remaining = max(0, eval_budget - fixed)

    # Depth-2: pair_top_k scales with sqrt(remaining)
    pair_top_k = min(n_prims, max(15, int(math.sqrt(remaining / 0.6))))
    remaining -= int(pair_top_k ** 2 * 0.6)
    remaining = max(0, remaining)

    # Depth-3: triple_top_k scales with cube root
    if remaining > 500:
        triple_top_k = min(20, max(8, int(remaining ** (1 / 3))))
        remaining -= triple_top_k ** 3
        remaining = max(0, remaining)
    else:
        triple_top_k = 8

    return {
        "exhaustive_pair_top_k": pair_top_k,
        "exhaustive_triple_top_k": triple_top_k,
        "beam_width": 1,
        "max_generations": 1,
    }


def derive_rounds(compute_cap: int) -> int:
    """Auto-derive rounds from compute budget.

    Round 2 gives +50% solves (huge ROI, always worth it).
    Round 3 gives +15% solves (only worth it at high budget).
    Rounds 4-5 (contest mode, 10M+): search only unsolved tasks,
    compounding library benefits on the hardest remaining problems.
    """
    if compute_cap >= 10_000_000:
        return 5  # contest mode: 5 rounds
    if compute_cap >= 200_000:
        return 2
    return 1


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
    # These defaults are fallbacks; the benchmark runner auto-derives optimal
    # values from compute_cap via derive_search_params().
    exhaustive_pair_top_k: int = 40
    # Triple exhaustion: top-K singles + essential concepts → K³ combos.
    # Auto-derived from compute_cap in practice (see derive_search_params).
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

    # Guided depth-4/5 search: after exhaustive depth-3 fails, search deeper
    # with a pruned primitive set ranked by depth-1/2/3 results.
    guided_depth4_top_k: int = 20       # primitives for depth-4 enumeration
    guided_depth5_top_k: int = 10       # primitives for depth-5 enumeration
    guided_nearmiss_top_k: int = 5      # near-miss programs to extend by 1 step
    guided_budget_fraction: float = 0.30  # max fraction of original budget for guided phase


@dataclass
class SleepConfig:
    """Knobs for the sleep/consolidation phase."""
    min_occurrences: int = 2      # sub-tree must appear in >= N programs
    min_size: int = 2             # sub-tree must have >= N nodes
    max_library_size: int = 50    # cap on total library entries
    usefulness_decay: float = 0.90  # decay old entries each sleep cycle
    reuse_bonus: float = 2.0       # scoring bonus per reuse for eviction ranking
    unsolved_weight: float = 0.10  # quality discount for unsolved vs solved programs
    example_solve_exponent: float = 2.0  # exponent for (k/n)^e per-example scoring


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
