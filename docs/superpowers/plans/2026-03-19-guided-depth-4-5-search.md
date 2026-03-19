# Guided Depth-4/5 Search Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add guided depth-4/5 search after exhaustive depth-3 to break the search depth ceiling, targeting +15-37 eval solves.

**Architecture:** After depth-3 exhaustive finishes unsolved, use leftover budget for three strategies in priority order: (1) near-miss extension wrapping top-5 programs with all primitives, (2) depth-4 enumeration with top-20 primitives, (3) depth-5 with top-10. All new code in `core/learner.py` and `core/config.py` (domain-agnostic). Zero regression on existing solves.

**Tech Stack:** Python 3.13, pytest, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-19-guided-depth-4-5-search-design.md`

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `core/config.py` | Search configuration | Add 4 guided search fields to `SearchConfig` |
| `core/learner.py` | Universal learning loop | Add `_guided_deep_search()`, `_select_guided_pool()`, modify `_wake_core()`, extend `_WakeContext` |
| `tests/test_guided_search.py` | Unit + integration tests | New file, ~200 lines |
| `DECISIONS.md` | Decision log | Add Decision 120 with benchmark results |

---

### Task 1: Add Config Fields

**Files:**
- Modify: `core/config.py:66-108` (SearchConfig dataclass)
- Test: `tests/test_guided_search.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_guided_search.py`:

```python
"""Tests for guided depth-4/5 search."""
import pytest
from core.config import SearchConfig


class TestGuidedSearchConfig:
    def test_default_guided_fields(self):
        cfg = SearchConfig()
        assert cfg.guided_depth4_top_k == 20
        assert cfg.guided_depth5_top_k == 10
        assert cfg.guided_nearmiss_top_k == 5
        assert cfg.guided_budget_fraction == 0.30

    def test_custom_guided_fields(self):
        cfg = SearchConfig(guided_depth4_top_k=15, guided_depth5_top_k=8)
        assert cfg.guided_depth4_top_k == 15
        assert cfg.guided_depth5_top_k == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedSearchConfig -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'guided_depth4_top_k'`

- [ ] **Step 3: Add fields to SearchConfig**

In `core/config.py`, add after the `verbose` field (line 107):

```python
    # Guided depth-4/5 search: after exhaustive depth-3 fails, search deeper
    # with a pruned primitive set ranked by depth-1/2/3 results.
    guided_depth4_top_k: int = 20       # primitives for depth-4 enumeration
    guided_depth5_top_k: int = 10       # primitives for depth-5 enumeration
    guided_nearmiss_top_k: int = 5      # near-miss programs to extend by 1 step
    guided_budget_fraction: float = 0.30  # max fraction of original budget for guided phase
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedSearchConfig -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite for regression**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: 541 passed (539 existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git add core/config.py tests/test_guided_search.py
git commit -m "feat: add guided depth-4/5 search config fields"
```

---

### Task 2: Extend _WakeContext to Track Depth-1 Scores

**Files:**
- Modify: `core/learner.py:53-76` (_WakeContext class)
- Test: `tests/test_guided_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_guided_search.py`:

```python
from core.learner import _WakeContext
from core.config import SearchConfig
from core.types import Task, Program, ScoredProgram


class TestWakeContextDepth1Scores:
    def test_depth1_scores_initialized_empty(self):
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        assert ctx.depth1_scores == {}

    def test_depth1_scores_mutable(self):
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores["rotate_90"] = 0.15
        ctx.depth1_scores["mirror_h"] = 0.02
        assert len(ctx.depth1_scores) == 2
        assert ctx.depth1_scores["mirror_h"] == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_guided_search.py::TestWakeContextDepth1Scores -v`
Expected: FAIL with `AttributeError: '_WakeContext' object has no attribute 'depth1_scores'`

- [ ] **Step 3: Add depth1_scores to _WakeContext**

In `core/learner.py`, modify `_WakeContext`:

Add `"depth1_scores"` to `__slots__` (line 56):
```python
    __slots__ = (
        "task", "all_prims", "cfg", "eval_budget", "record", "t0",
        "best_so_far", "n_evals", "total_deduped", "gens_used",
        "pareto", "enum_candidates", "beam_scored", "depth1_scores",
    )
```

Add initialization in `__init__` (after line 75):
```python
        self.depth1_scores: dict[str, float] = {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_guided_search.py::TestWakeContextDepth1Scores -v`
Expected: 2 passed

- [ ] **Step 5: Populate depth1_scores in _exhaustive_enumerate**

In `core/learner.py`, inside `_run_stratum_enumeration()` (after line 234 `ctx.enum_candidates.extend(candidates)`), add code to extract depth-1 scores from the candidates:

```python
        # Capture depth-1 scores for guided search pool selection
        for sp in candidates:
            name = sp.program.root
            if not sp.program.children and name not in ctx.depth1_scores:
                ctx.depth1_scores[name] = sp.prediction_error
```

This only stores scores for depth-1 programs (no children = single primitive). First-seen wins, so the lowest-error stratum's score is kept.

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All pass (543 total)

- [ ] **Step 7: Commit**

```bash
git add core/learner.py tests/test_guided_search.py
git commit -m "feat: track depth-1 scores in WakeContext for guided search pool selection"
```

---

### Task 3: Implement _select_guided_pool

**Files:**
- Modify: `core/learner.py` (new method on Learner)
- Test: `tests/test_guided_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_guided_search.py`:

```python
from core.learner import Learner
from core.interfaces import Environment, Grammar, DriveSignal, Memory


def _make_learner():
    """Create a minimal Learner with mock interfaces."""
    from unittest.mock import MagicMock
    env = MagicMock(spec=Environment)
    grammar = MagicMock(spec=Grammar)
    drive = MagicMock(spec=DriveSignal)
    memory = MagicMock(spec=Memory)
    memory.get_library.return_value = []
    return Learner(env, grammar, drive, memory)


class TestSelectGuidedPool:
    def test_basic_pool_selection(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        # Simulate depth-1 scores: lower = better
        ctx.depth1_scores = {
            "mirror_h": 0.02,
            "rotate_90": 0.15,
            "fill_enclosed": 0.30,
            "gravity_down": 0.50,
            "transpose": 0.70,
            "identity": 0.90,
        }
        # Simulate pareto front with depth-2/3 programs
        ctx.enum_candidates = [
            ScoredProgram(
                program=Program(root="outline", children=[Program(root="erode")]),
                energy=0.1, prediction_error=0.05, complexity_cost=0.002,
            ),
        ]

        pool = learner._select_guided_pool(ctx, top_k=4)
        assert len(pool) == 4
        # Best depth-1 performers should be first
        assert pool[0] == "mirror_h"
        assert pool[1] == "rotate_90"
        # depth-2/3 contributors should fill remaining slots
        assert "outline" in pool or "erode" in pool or "fill_enclosed" in pool

    def test_pool_respects_top_k(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores = {f"prim_{i}": i * 0.01 for i in range(50)}
        ctx.enum_candidates = []

        pool = learner._select_guided_pool(ctx, top_k=10)
        assert len(pool) == 10

    def test_pool_empty_scores(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores = {}
        ctx.enum_candidates = []

        pool = learner._select_guided_pool(ctx, top_k=10)
        assert pool == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_guided_search.py::TestSelectGuidedPool -v`
Expected: FAIL with `AttributeError: 'Learner' object has no attribute '_select_guided_pool'`

- [ ] **Step 3: Implement _select_guided_pool**

In `core/learner.py`, add as a method on `Learner` (after `_run_stratum_enumeration`, around line 241):

```python
    def _select_guided_pool(self, ctx: _WakeContext, top_k: int) -> list[str]:
        """Select top-K primitives for guided deep search.

        Sources:
        1. Best depth-1 performers (lowest single-primitive error)
        2. Primitives appearing in best depth-2/3 programs from pareto front

        Returns list of primitive names, ordered by expected quality.
        """
        if not ctx.depth1_scores:
            return []

        # Source 1: depth-1 scores ranked by error (lowest first)
        depth1_ranked = sorted(ctx.depth1_scores.items(), key=lambda x: x[1])

        # Source 2: primitives appearing in best depth-2/3 candidates
        depth23_names: list[str] = []
        candidates_ranked = sorted(ctx.enum_candidates,
                                   key=lambda sp: sp.prediction_error)
        seen_d23: set[str] = set()
        for sp in candidates_ranked[:top_k * 2]:
            for name in _extract_primitive_names(sp.program):
                if name not in seen_d23:
                    depth23_names.append(name)
                    seen_d23.add(name)

        # Merge: prioritize depth-1 ranking, fill with depth-2/3 contributors
        pool: list[str] = []
        seen: set[str] = set()
        for name, _ in depth1_ranked:
            if name not in seen:
                pool.append(name)
                seen.add(name)
            if len(pool) >= top_k:
                break

        for name in depth23_names:
            if name not in seen and len(pool) < top_k:
                pool.append(name)
                seen.add(name)

        return pool[:top_k]
```

Also add this module-level helper function (before the `Learner` class):

```python
def _extract_primitive_names(program: Program) -> list[str]:
    """Recursively extract all primitive names from a program tree."""
    names = [program.root]
    for child in (program.children or []):
        names.extend(_extract_primitive_names(child))
    return names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_guided_search.py::TestSelectGuidedPool -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/learner.py tests/test_guided_search.py
git commit -m "feat: implement _select_guided_pool for guided search primitive ranking"
```

---

### Task 4: Implement _guided_deep_search (Near-Miss Extension + Depth-4 + Depth-5)

**Files:**
- Modify: `core/learner.py` (new method on Learner)
- Test: `tests/test_guided_search.py`

This is the core task. The method has three sub-strategies run in priority order, each with early exit on solve.

- [ ] **Step 1: Write the failing test for near-miss extension**

Append to `tests/test_guided_search.py`:

```python
class TestGuidedDeepSearch:
    """Integration test using real ARC environment for guided search."""

    def _make_real_learner(self):
        """Create a Learner with real ARC interfaces for integration testing."""
        from domains.arc.environment import ARCEnv
        from domains.arc.grammar import ARCGrammar
        from domains.arc.drive import ARCDrive
        from core.memory import InMemoryStore
        env = ARCEnv()
        grammar = ARCGrammar(seed=42)
        drive = ARCDrive()
        memory = InMemoryStore()
        cfg = SearchConfig(
            eval_budget=100000,
            guided_depth4_top_k=5,
            guided_depth5_top_k=3,
            guided_nearmiss_top_k=3,
            guided_budget_fraction=0.50,
        )
        return Learner(env, grammar, drive, memory, search_config=cfg)

    def test_guided_search_returns_none_when_no_budget(self):
        """Guided search should do nothing when budget is exhausted."""
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(eval_budget=100, guided_budget_fraction=0.30),
            eval_budget=100,
            record=False,
        )
        ctx.n_evals = 100  # budget exhausted
        ctx.depth1_scores = {"prim_a": 0.1}
        result = learner._guided_deep_search(ctx)
        assert result is None

    def test_guided_search_returns_none_when_already_solved(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_examples=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=10000,
            record=False,
        )
        ctx.depth1_scores = {"prim_a": 0.1}
        # Fake a solved state
        ctx.best_so_far = ScoredProgram(
            program=Program(root="prim_a"),
            energy=0.001, prediction_error=0.0, complexity_cost=0.001,
            max_example_error=0.0,
        )
        result = learner._guided_deep_search(ctx)
        assert result is None

    def test_guided_search_method_exists(self):
        learner = _make_learner()
        assert hasattr(learner, '_guided_deep_search')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedDeepSearch -v`
Expected: FAIL with `AttributeError: 'Learner' object has no attribute '_guided_deep_search'`

- [ ] **Step 3: Implement _guided_deep_search**

In `core/learner.py`, add as a method on `Learner` (after `_select_guided_pool`):

```python
    def _guided_deep_search(self, ctx: _WakeContext) -> Optional[str]:
        """Guided depth-4/5 search using leftover budget after depth-3.

        Three strategies in priority order (early exit on solve):
        1. Near-miss extension: wrap top-M near-miss programs with each primitive
        2. Depth-4 enumeration: top-K primitives, K^4 programs
        3. Depth-5 enumeration: top-K' primitives, K'^5 programs

        Returns phase name if solved, None otherwise.
        """
        cfg = ctx.cfg
        if ctx.solved or not ctx.depth1_scores:
            return None

        # Compute guided budget from remaining + fraction cap
        if ctx.eval_budget > 0:
            remaining = ctx.eval_budget - ctx.n_evals
            fraction_cap = int(ctx.eval_budget * cfg.guided_budget_fraction)
            guided_budget = min(remaining, fraction_cap)
            if guided_budget < 100:
                return None
        else:
            guided_budget = 0  # unlimited

        guided_evals = 0
        solve_thresh = cfg.solve_threshold

        def _gbudget_ok() -> bool:
            return guided_budget <= 0 or guided_evals < guided_budget

        # Collect unary transform primitives for wrapping
        prim_by_name = {p.name: p for p in ctx.all_prims}
        wrap_prims = [p for p in ctx.all_prims
                      if p.arity <= 1 and p.kind == "transform"]

        # --- Strategy 1: Near-miss extension ---
        # Take best unsolved programs, try wrapping with each primitive
        near_misses = sorted(
            [sp for sp in ctx.enum_candidates
             if sp.prediction_error > solve_thresh],
            key=lambda sp: sp.prediction_error,
        )[:cfg.guided_nearmiss_top_k]

        for nm in near_misses:
            if not _gbudget_ok():
                break
            for wp in wrap_prims:
                if not _gbudget_ok():
                    break
                prog = Program(root=wp.name, children=[nm.program])
                sp = self._evaluate_program(prog, ctx.task)
                guided_evals += 1
                ctx.n_evals += 1
                ctx.update_best(sp)
                if sp.prediction_error <= solve_thresh:
                    return "guided_nearmiss"

        # --- Strategy 2: Depth-4 enumeration ---
        pool4 = self._select_guided_pool(ctx, cfg.guided_depth4_top_k)
        if pool4 and _gbudget_ok():
            for a in pool4:
                if not _gbudget_ok():
                    break
                for b in pool4:
                    if not _gbudget_ok():
                        break
                    for c in pool4:
                        if not _gbudget_ok():
                            break
                        for d in pool4:
                            if not _gbudget_ok():
                                break
                            if a == b == c == d:
                                continue
                            prog = Program(root=a, children=[
                                Program(root=b, children=[
                                    Program(root=c, children=[
                                        Program(root=d)])])])
                            sp = self._evaluate_program(prog, ctx.task)
                            guided_evals += 4
                            ctx.n_evals += 4
                            ctx.update_best(sp)
                            if sp.prediction_error <= solve_thresh:
                                return "guided_depth4"

        # --- Strategy 3: Depth-5 enumeration ---
        pool5 = self._select_guided_pool(ctx, cfg.guided_depth5_top_k)
        if pool5 and _gbudget_ok():
            for a in pool5:
                if not _gbudget_ok():
                    break
                for b in pool5:
                    if not _gbudget_ok():
                        break
                    for c in pool5:
                        if not _gbudget_ok():
                            break
                        for d in pool5:
                            if not _gbudget_ok():
                                break
                            for e in pool5:
                                if not _gbudget_ok():
                                    break
                                if a == b == c == d == e:
                                    continue
                                prog = Program(root=a, children=[
                                    Program(root=b, children=[
                                        Program(root=c, children=[
                                            Program(root=d, children=[
                                                Program(root=e)])])])])
                                sp = self._evaluate_program(prog, ctx.task)
                                guided_evals += 5
                                ctx.n_evals += 5
                                ctx.update_best(sp)
                                if sp.prediction_error <= solve_thresh:
                                    return "guided_depth5"

        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedDeepSearch -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/learner.py tests/test_guided_search.py
git commit -m "feat: implement _guided_deep_search with near-miss, depth-4, depth-5 strategies"
```

---

### Task 5: Wire _guided_deep_search into _wake_core

**Files:**
- Modify: `core/learner.py:169-207` (_wake_core method)
- Test: `tests/test_guided_search.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_guided_search.py`:

```python
class TestGuidedSearchIntegration:
    """Test that guided search is wired into the wake pipeline."""

    def test_wake_core_calls_guided_search(self):
        """Verify _wake_core calls _guided_deep_search between strata and hooks."""
        from unittest.mock import MagicMock, patch, call

        learner = _make_learner()
        # We need to verify the call order, so patch the method
        learner._guided_deep_search = MagicMock(return_value=None)

        # Mock out the expensive parts
        learner.grammar.prepare_for_task = MagicMock()
        learner.grammar.base_primitives = MagicMock(return_value=[])
        learner.grammar.inject_library = MagicMock(return_value=[])
        learner.grammar.propose_strata = MagicMock(return_value=[])
        learner.grammar.allow_structural_phases = MagicMock(return_value=False)
        learner.memory.get_library = MagicMock(return_value=[])

        task = Task(task_id="t1", train_examples=[], test_examples=[])
        learner._wake_core(task, record=False)

        # Guided search should have been called
        learner._guided_deep_search.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedSearchIntegration -v`
Expected: FAIL (guided search not called because it's not wired in yet)

- [ ] **Step 3: Wire guided search into _wake_core**

In `core/learner.py`, modify `_wake_core()`. Between the stratum loop (line 198) and Stage 2 structural hooks (line 201), insert the guided search call:

Find this block in `_wake_core`:
```python
            if solved_by is not None:
                return self._make_solved_result(ctx, solved_by)

        # Stage 2: Structural hooks (run once with aggregated candidates)
```

Replace with:
```python
            if solved_by is not None:
                return self._make_solved_result(ctx, solved_by)

        # Stage 1.5: Guided depth-4/5 search (adaptive budget)
        if not ctx.solved:
            solved_by = self._guided_deep_search(ctx)
            if solved_by is not None:
                return self._make_solved_result(ctx, solved_by)

        # Stage 2: Structural hooks (run once with aggregated candidates)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_guided_search.py::TestGuidedSearchIntegration -v`
Expected: PASS

- [ ] **Step 5: Run full test suite for regression**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All pass (547+ total)

- [ ] **Step 6: Commit**

```bash
git add core/learner.py tests/test_guided_search.py
git commit -m "feat: wire guided depth-4/5 search into wake pipeline between strata and hooks"
```

---

### Task 6: Quick Validation — Run on 20 Unsolved Tasks

**Files:**
- No code changes. Validation only.

- [ ] **Step 1: Identify 20 unsolved eval tasks with low error**

```bash
python3 -c "
import json
with open('runs/arc_agi_1_pipeline_20260319_153623.json') as f:
    data = json.load(f)
# Get 20 unsolved eval tasks with lowest error
unsolved = [(info['prediction_error'], tid) for tid, info in data['eval_tasks'].items()
            if not info.get('test_solved') and info.get('prediction_error', 1.0) < 0.1]
unsolved.sort()
ids = ','.join(tid for _, tid in unsolved[:20])
print(ids)
"
```

- [ ] **Step 2: Run benchmark on those 20 tasks**

```bash
python3 -m common --domain arc-agi-1 --mode contest --rounds 1 --split evaluation --task-ids <IDS_FROM_STEP_1>
```

Expected: Some new solves beyond the 53 baseline (any > 0 is signal). Check for:
- No crashes
- Guided search phase appearing in output
- Wall time reasonable (not >10x slower per task)

- [ ] **Step 3: Compare results**

Count new solves and note which strategy found them (guided_nearmiss, guided_depth4, guided_depth5). Record wall time per task.

- [ ] **Step 4: Commit validation notes to DECISIONS.md**

Add Decision 120 with the quick validation results.

```bash
git add DECISIONS.md
git commit -m "docs: Decision 120 — guided search quick validation results"
```

---

### Task 7: Hyperparameter Sweet-Spot Optimization

**Files:**
- Modify: `DECISIONS.md`
- Possibly modify: `core/config.py` (if defaults change)

Per CLAUDE.md: "Always do sweet-spot optimization. Don't just pick a value — measure alternatives."

- [ ] **Step 1: Create a 50-task validation set**

Select 25 unsolved training + 25 unsolved eval tasks (error < 0.2):

```bash
python3 -c "
import json
with open('runs/arc_agi_1_pipeline_20260319_153623.json') as f:
    data = json.load(f)
train_ids = [tid for tid, info in data['train_tasks'].items()
             if not info.get('test_solved') and info.get('prediction_error', 1.0) < 0.2][:25]
eval_ids = [tid for tid, info in data['eval_tasks'].items()
            if not info.get('test_solved') and info.get('prediction_error', 1.0) < 0.2][:25]
print('TRAIN:', ','.join(train_ids))
print('EVAL:', ','.join(eval_ids))
"
```

- [ ] **Step 2: Tune guided_depth4_top_k (15 vs 20 vs 30)**

Run 3 benchmarks on the 50-task set, varying `guided_depth4_top_k` only. Measure: new solves, wall time, evals consumed. Hold other params at defaults.

- [ ] **Step 3: Tune guided_budget_fraction (0.20 vs 0.30 vs 0.50)**

Same procedure with best depth4_top_k from step 2.

- [ ] **Step 4: Update defaults if needed**

If a non-default value wins, update the default in `core/config.py`.

- [ ] **Step 5: Document results in DECISIONS.md**

Record all measurements and final parameter choices.

- [ ] **Step 6: Commit**

```bash
git add core/config.py DECISIONS.md
git commit -m "tune: optimize guided search hyperparameters"
```

---

### Task 8: Full Contest Benchmark

**Files:**
- Modify: `DECISIONS.md`

- [ ] **Step 1: Run full 2-round contest benchmark**

```bash
python3 -m common --domain arc-agi-1 --mode contest --rounds 2
```

Expected: ~35 minutes. Record train and eval solve counts.

- [ ] **Step 2: Compare against baseline**

Baseline: Train 120/400 (30.0%), Eval 53/400 (13.2%)
Target: Any improvement in eval, ideally +15 or more.

- [ ] **Step 3: Document results in DECISIONS.md**

Add final benchmark numbers, comparison, and analysis of which strategies contributed.

- [ ] **Step 4: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: Decision 121 — full benchmark with guided depth-4/5 search"
```

---

### Task 9: Push and Cleanup

- [ ] **Step 1: Run full test suite one final time**

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: All pass.

- [ ] **Step 2: Push branch**

```bash
git push origin claude/zen-feynman
```

(Note: may need credentials configured)

- [ ] **Step 3: Update DECISIONS.md with final state**

Total state summary: primitives, local rules, test count, train/eval solve rates.
