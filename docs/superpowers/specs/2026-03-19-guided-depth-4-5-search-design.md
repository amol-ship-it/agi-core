# Guided Depth-4/5 Search — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Goal:** Break the depth-3 ceiling to increase ARC-AGI-1 eval from 53/400 (13.2%) toward 100+/400 (25%+)

## Problem

The exhaustive depth-3 search enumerates all ~950K programs (98 primitives). This ceiling is hit: adding more primitives (54->98) yielded only +1 eval. 279/347 unsolved eval tasks have prediction_error < 0.5, meaning they're amenable — they just need longer programs. Depth-4 exhaustive (98^4 = 92M) is 1,844x over the per-task compute budget.

## Solution: Adaptive Guided Deep Search

After depth-3 exhaustive finishes unsolved, use **leftover budget** to search depth-4/5 with a **pruned primitive set** ranked by depth-1/2/3 results.

### Architecture

```
Current flow:
  Stratum enumeration (depth 1->2->3)  ->  Structural hooks

New flow:
  Stratum enumeration (depth 1->2->3)  ->  GUIDED DEPTH-4/5  ->  Structural hooks
```

Zero regression: depth-3 exhaustive is unchanged. Guided search only runs on unsolved tasks with remaining budget.

### Three Search Strategies

**1. Guided depth-4 enumeration**
- Select top-K primitives by depth-1/2/3 error ranking (default K=20)
- Enumerate all A(B(C(D(x)))) compositions: 20^4 = 160,000 programs
- Primitive pool: union of (a) top-K by depth-1 error, (b) primitives appearing in top depth-2/3 results

**2. Guided depth-5 enumeration**
- Select top-K' primitives (default K'=10)
- Enumerate depth-5 chains: 10^5 = 100,000 programs
- Only runs if depth-4 didn't solve and budget remains

**3. Near-miss extension**
- Take top-M best unsolved programs from depth-3 (default M=5)
- Try wrapping each with all 98 primitives: `prim(near_miss)` = 5 x 98 = 490 candidates
- Also try prepending: `near_miss_outer(prim(near_miss_inner))` for key depth-2 near-misses
- Runs first (cheapest, highest expected value)

### Budget Management

- Depth-3 exhaustive runs with full budget as before
- After depth-3: `remaining = original_budget - evals_consumed`
- Guided phase gets: `min(remaining, original_budget * guided_budget_fraction)`
- Default `guided_budget_fraction = 0.30` (reserves 30% of original budget)
- Strategy priority: near-miss extension -> depth-4 -> depth-5 (early exit on solve)

### Primitive Selection Algorithm

```python
def select_guided_pool(ctx, top_k):
    """Select top-K primitives for guided deep search."""
    # Source 1: Best depth-1 performers (lowest single-primitive error)
    depth1_ranked = sorted(ctx.depth1_scores.items(), key=lambda x: x[1])

    # Source 2: Primitives appearing in best depth-2/3 programs
    depth23_names = set()
    for sp in ctx.pareto_front[:top_k * 2]:
        depth23_names.update(extract_primitive_names(sp.program))

    # Merge: prioritize depth-1 ranking, supplement with depth-2/3 appearances
    pool = []
    seen = set()
    for name, _ in depth1_ranked:
        if name not in seen:
            pool.append(name)
            seen.add(name)
        if len(pool) >= top_k:
            break

    # Fill remaining slots from depth-2/3 contributors
    for name in depth23_names:
        if name not in seen and len(pool) < top_k:
            pool.append(name)
            seen.add(name)

    return pool
```

### Where Code Lives

| Component | File | Change |
|-----------|------|--------|
| `_guided_deep_search()` | `core/learner.py` | New method, ~150 lines |
| `_wake_core()` | `core/learner.py` | Add call between strata and hooks |
| `SearchConfig` | `core/config.py` | 4 new fields |
| `_WakeContext` | `core/learner.py` | Store depth-1 scores for reuse |
| Tests | `tests/test_guided_search.py` | New test file |

### Config Additions

```python
# core/config.py — SearchConfig dataclass
guided_depth4_top_k: int = 20      # Primitives for depth-4 guided search
guided_depth5_top_k: int = 10      # Primitives for depth-5 guided search
guided_nearmiss_top_k: int = 5     # Near-miss programs to extend
guided_budget_fraction: float = 0.30  # Max fraction of original budget for guided phase
```

### Hyperparameter Optimization Plan

Run 3 values (low/mid/high) on a 50-task subset (25 unsolved train + 25 unsolved eval):

| Parameter | Low | Mid | High |
|-----------|-----|-----|------|
| `guided_depth4_top_k` | 15 | 20 | 30 |
| `guided_depth5_top_k` | 8 | 10 | 15 |
| `guided_budget_fraction` | 0.20 | 0.30 | 0.50 |
| `guided_nearmiss_top_k` | 3 | 5 | 10 |

Tune one parameter at a time, holding others at mid. Measure: new solves, wall time, evals consumed.

### Invariants Preserved

- Core loop (`core/`) never imports domain-specific code
- All new code is in `core/learner.py` and `core/config.py` (domain-agnostic)
- Depth-3 exhaustive is byte-identical — zero regression risk
- Structural hooks still run after guided search
- Sleep/library learning untouched
- Grammar/strata system untouched

### Expected Impact

| Metric | Current | Conservative | Optimistic |
|--------|---------|-------------|------------|
| Eval solved | 53/400 | 68/400 (+15) | 90/400 (+37) |
| Train solved | 120/400 | 135/400 (+15) | 155/400 (+35) |

Primary targets: 48 very-close misses (error < 0.05), 151 close misses (error 0.05-0.2).

### Risks

1. **Wall time increase**: Depth-4/5 enumeration adds compute to every unsolved task. Mitigated by budget fraction cap.
2. **Diminishing returns**: If close misses need qualitatively different programs (not just deeper), depth-4/5 won't help. Near-miss extension hedges this.
3. **Memory**: Storing depth-1 scores adds ~98 floats per task. Negligible.
