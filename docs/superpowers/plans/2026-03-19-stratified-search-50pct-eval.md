# Stratified Search with Primitive Expansion — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve ARC-AGI-1 eval score from 12.2% (49/400) to >50% (200+/400) via search stratification, new primitives, deeper composition, and generalization guards.

**Architecture:** Replace the monolithic 10-phase wake pipeline with a two-stage model: (1) stratum-based enumeration where each stratum searches a focused primitive subset, (2) global structural hooks (object decomp, cross-ref, local rules, procedural). Task fingerprinting drives stratum selection. 25 new primitives are added but only activated within relevant strata to prevent search dilution.

**Tech Stack:** Python 3.10+, numpy, pytest, existing core/ framework

**Spec:** `docs/superpowers/specs/2026-03-19-stratified-search-50pct-eval-design.md`

---

## Chunk 1: Core Types & Interfaces (Phase 1a)

### Task 1: Add SearchStratum to core/types.py

**Files:**
- Modify: `core/types.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_interfaces.py — add at the end
def test_search_stratum_defaults():
    from core.types import SearchStratum
    s = SearchStratum(name="test", primitive_names=["rotate_90_cw", "mirror_h"])
    assert s.name == "test"
    assert s.primitive_names == ["rotate_90_cw", "mirror_h"]
    assert s.max_depth == 3
    assert s.budget_fraction == 0.1
    assert s.try_corrections is True
    assert s.metadata == {}


def test_search_stratum_custom():
    from core.types import SearchStratum
    s = SearchStratum(
        name="inpainting",
        primitive_names=["inpaint_diagonal"],
        max_depth=4,
        budget_fraction=0.3,
        try_corrections=False,
        metadata={"run_local_rules": True},
    )
    assert s.max_depth == 4
    assert s.budget_fraction == 0.3
    assert s.try_corrections is False
    assert s.metadata["run_local_rules"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py::test_search_stratum_defaults -v`
Expected: FAIL with "cannot import name 'SearchStratum'"

- [ ] **Step 3: Write minimal implementation**

Add to `core/types.py` after the `LibraryEntry` class:

```python
@dataclass
class SearchStratum:
    """A focused search context with its own primitive subset and budget.

    The core learner iterates over strata proposed by the Grammar.
    Each stratum runs exhaustive enumeration over its primitive subset.
    """
    name: str
    primitive_names: list[str]   # subset of primitives to search over
    max_depth: int = 3
    budget_fraction: float = 0.1  # share of task compute budget
    try_corrections: bool = True
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interfaces.py::test_search_stratum_defaults tests/test_interfaces.py::test_search_stratum_custom -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/types.py tests/test_interfaces.py
git commit -m "feat: add SearchStratum dataclass to core/types.py"
```

---

### Task 2: Add propose_strata() to Grammar interface

**Files:**
- Modify: `core/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_interfaces.py — add
def test_grammar_propose_strata_default():
    """Default propose_strata returns single stratum with all primitives."""
    from core.types import Primitive, Task, SearchStratum
    from core.interfaces import Grammar

    class MinGrammar(Grammar):
        def base_primitives(self):
            return [Primitive("a", 1, lambda x: x), Primitive("b", 1, lambda x: x)]
        def compose(self, outer, inner_programs):
            from core.types import Program
            return Program(root=outer.name, children=inner_programs)
        def mutate(self, program, primitives, transition_matrix=None):
            return program
        def crossover(self, a, b):
            return a

    g = MinGrammar()
    task = Task("t1", [(1, 2)], [3])
    prims = g.base_primitives()
    strata = g.propose_strata(task, prims)
    assert len(strata) == 1
    assert strata[0].name == "default"
    assert set(strata[0].primitive_names) == {"a", "b"}
    assert strata[0].budget_fraction == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py::test_grammar_propose_strata_default -v`
Expected: FAIL with "MinGrammar has no attribute 'propose_strata'"

- [ ] **Step 3: Write minimal implementation**

Add to `Grammar` class in `core/interfaces.py`, after `prepare_for_task()`:

```python
def propose_strata(
    self, task: Task, primitives: list[Primitive]
) -> list["SearchStratum"]:
    """Propose search strata for this task based on structural analysis.

    Each stratum is a focused search over a primitive subset.
    The core learner runs each stratum independently and keeps the best
    program found across all strata.

    Default: single stratum with all primitives (backward compatible).
    """
    from .types import SearchStratum
    return [SearchStratum(
        name="default",
        primitive_names=[p.name for p in primitives],
        budget_fraction=1.0,
    )]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interfaces.py::test_grammar_propose_strata_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add propose_strata() to Grammar interface with backward-compatible default"
```

---

### Task 3: Add inverse_primitives() to Grammar and promote try_local_rules/try_procedural to Environment

**Files:**
- Modify: `core/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_interfaces.py — add
def test_grammar_inverse_primitives_default():
    """Default inverse_primitives returns empty dict."""
    from core.types import Primitive
    from core.interfaces import Grammar

    class MinGrammar(Grammar):
        def base_primitives(self): return []
        def compose(self, outer, inner_programs):
            from core.types import Program
            return Program(root=outer.name, children=inner_programs)
        def mutate(self, program, primitives, tm=None): return program
        def crossover(self, a, b): return a

    g = MinGrammar()
    assert g.inverse_primitives() == {}


def test_env_try_local_rules_default():
    """Default try_local_rules returns None."""
    from core.interfaces import Environment

    class MinEnv(Environment):
        def load_task(self, task): return None
        def execute(self, program, input_data): return None
        def reset(self): pass

    env = MinEnv()
    from core.types import Task
    task = Task("t1", [(1, 2)], [3])
    assert env.try_local_rules(task) is None
    assert env.try_procedural(task) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py::test_grammar_inverse_primitives_default tests/test_interfaces.py::test_env_try_local_rules_default -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Add to `Grammar` in `core/interfaces.py`, after `essential_pair_concepts()`:

```python
def inverse_primitives(self) -> dict[str, str]:
    """Map primitive names to their inverses for bidirectional search.

    Used by bidirectional search: if f(x)=y, then f_inv(y)=x.
    Only invertible primitives need entries.
    Default: empty (no invertible primitives).
    """
    return {}
```

Add to `Environment` in `core/interfaces.py`, after `infer_output_correction()`:

```python
def try_local_rules(
    self, task: "Task",
) -> Optional[tuple[str, Any]]:
    """Try solving via learned pixel-level neighborhood rules.

    Learns (center, neighbor_features) -> output_color mappings
    from training examples, validated by LOOCV.

    Default: not supported (returns None).
    """
    return None

def try_procedural(
    self, task: "Task",
) -> Optional[tuple[str, Any]]:
    """Try solving via procedural object DSL (per-object action learning).

    Learns per-object actions (fill, move, extract) from pixel diffs.

    Default: not supported (returns None).
    """
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interfaces.py::test_grammar_inverse_primitives_default tests/test_interfaces.py::test_env_try_local_rules_default -v`
Expected: PASS

- [ ] **Step 5: Update learner.py to remove hasattr checks**

In `core/learner.py`, replace lines 285-286:
```python
# Before:
if not hasattr(self.env, 'try_local_rules'):
    return None
# After: (remove these 2 lines — method is now on the interface)
```

Similarly replace lines 303-304:
```python
# Before:
if not hasattr(self.env, 'try_procedural'):
    return None
# After: (remove these 2 lines)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add core/interfaces.py core/learner.py tests/test_interfaces.py
git commit -m "feat: add inverse_primitives() to Grammar, promote try_local_rules/try_procedural to Environment"
```

---

### Task 4: Add get_primitive_generality() to Memory

**Files:**
- Modify: `core/interfaces.py`
- Modify: `core/memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_memory.py — add
def test_primitive_generality_default():
    from core.memory import InMemoryStore
    store = InMemoryStore()
    assert store.get_primitive_generality() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py::test_primitive_generality_default -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Add to `Memory` in `core/interfaces.py`, after `update_primitive_score()`:

```python
def get_primitive_generality(self) -> dict[str, float]:
    """Per-primitive generality: n_distinct_tasks_solved / total_solved.

    Higher = primitive solves diverse tasks (more transferable).
    Used to prioritize primitives during eval search.
    Default: empty (uniform generality).
    """
    return {}
```

Add to `InMemoryStore` in `core/memory.py`, after `update_primitive_score()`:

```python
def get_primitive_generality(self) -> dict[str, float]:
    """Compute generality from solutions: how many distinct tasks use each primitive."""
    if not self._solutions:
        return {}
    total = len(self._solutions)
    prim_tasks: dict[str, set[str]] = {}
    for task_id, sp in self._solutions.items():
        for name in self._extract_primitive_names(sp.program):
            prim_tasks.setdefault(name, set()).add(task_id)
    return {name: len(tasks) / total for name, tasks in prim_tasks.items()}

@staticmethod
def _extract_primitive_names(prog: "Program") -> list[str]:
    """Recursively extract all primitive names from a program tree."""
    names = [prog.root]
    for child in prog.children:
        names.extend(InMemoryStore._extract_primitive_names(child))
    return names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory.py::test_primitive_generality_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/interfaces.py core/memory.py tests/test_memory.py
git commit -m "feat: add get_primitive_generality() to Memory interface"
```

---

## Chunk 2: Task Fingerprinting (Phase 1b)

### Task 5: Create fingerprint module

**Files:**
- Create: `domains/arc/fingerprint.py`
- Test: `tests/test_fingerprint.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fingerprint.py
"""Tests for ARC task fingerprinting."""
import pytest
from core.types import Task


def _make_task(pairs):
    """Helper: make a Task from (input, output) grid pairs."""
    return Task(
        task_id="test",
        train_examples=pairs,
        test_inputs=[p[0] for p in pairs],
    )


def test_fingerprint_same_dim():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
    ])
    fp = fingerprint_task(task)
    assert fp.dim_change == "same"


def test_fingerprint_shrink():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1]]),
    ])
    fp = fingerprint_task(task)
    assert fp.dim_change == "shrink"


def test_fingerprint_grow():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1]], [[1, 1], [1, 1]]),
    ])
    fp = fingerprint_task(task)
    assert fp.dim_change == "grow"


def test_fingerprint_has_holes():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 1, 1], [1, 2, 1], [1, 1, 1]]),
    ])
    fp = fingerprint_task(task)
    assert fp.has_holes is True


def test_fingerprint_colors():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 2], [3, 0]], [[1, 2], [3, 4]]),
    ])
    fp = fingerprint_task(task)
    assert fp.n_colors_in == 4  # 0,1,2,3
    assert fp.n_colors_out == 4  # 1,2,3,4
    assert fp.colors_added >= 1  # color 4 added
    assert fp.colors_removed >= 1  # color 0 removed


def test_fingerprint_pixel_diff():
    from domains.arc.fingerprint import fingerprint_task
    # 1 of 4 pixels changed
    task = _make_task([
        ([[1, 2], [3, 4]], [[1, 2], [3, 5]]),
    ])
    fp = fingerprint_task(task)
    assert 0.2 <= fp.pixel_diff_ratio <= 0.3


def test_fingerprint_is_recoloring():
    from domains.arc.fingerprint import fingerprint_task
    # Same structure, just colors swapped
    task = _make_task([
        ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
    ])
    fp = fingerprint_task(task)
    assert fp.is_recoloring is True


def test_fingerprint_separators():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 5, 2], [5, 5, 5], [3, 5, 4]], [[1, 5, 2], [5, 5, 5], [3, 5, 4]]),
    ])
    fp = fingerprint_task(task)
    assert fp.has_separators is True
    assert fp.n_sections >= 2


def test_fingerprint_n_objects():
    from domains.arc.fingerprint import fingerprint_task
    task = _make_task([
        ([[1, 0, 2], [0, 0, 0], [3, 0, 0]], [[1, 0, 2], [0, 0, 0], [3, 0, 0]]),
    ])
    fp = fingerprint_task(task)
    assert fp.n_objects == 3


def test_fingerprint_speed(benchmark):
    """Fingerprinting must be fast (< 1ms for typical grids)."""
    from domains.arc.fingerprint import fingerprint_task
    grid = [[i % 10 for i in range(30)] for _ in range(30)]
    task = _make_task([(grid, grid)])
    # Just ensure it runs without error; benchmark plugin measures time
    fp = fingerprint_task(task)
    assert fp is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fingerprint.py -v`
Expected: FAIL with "No module named 'domains.arc.fingerprint'"

- [ ] **Step 3: Write implementation**

Create `domains/arc/fingerprint.py`:

```python
"""Task fingerprinting for ARC-AGI search stratification.

Computes structural features from a task's training examples to determine
which search strata are relevant. Must be fast (< 1ms per task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from core.types import Task

Grid = list[list[int]]


@dataclass
class TaskFingerprint:
    """Structural features of an ARC task, computed from training examples."""
    dim_change: str = "same"        # "same", "shrink", "grow", "variable"
    has_separators: bool = False
    n_sections: int = 1
    symmetry: set = field(default_factory=set)
    symmetry_broken: bool = False
    n_objects: int = 0
    object_size_var: float = 0.0
    has_periodic: bool = False
    has_holes: bool = False
    n_colors_in: int = 0
    n_colors_out: int = 0
    colors_added: int = 0
    colors_removed: int = 0
    pixel_diff_ratio: float = 0.0
    output_is_subgrid: bool = False
    is_recoloring: bool = False


def _count_objects(grid: Grid) -> int:
    """Count 4-connected non-zero components."""
    if not grid:
        return 0
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr][cc]:
                        continue
                    visited[cr][cc] = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                            stack.append((nr, nc))
    return count


def _has_enclosed_zeros(grid: Grid) -> bool:
    """Check if grid has zero-valued pixels enclosed by non-zero pixels."""
    if not grid:
        return False
    h, w = len(grid), len(grid[0])
    # Flood fill from border zeros
    border_connected = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        if border_connected[cr][cc]:
            continue
        border_connected[cr][cc] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not border_connected[nr][nc] and grid[nr][nc] == 0:
                stack.append((nr, nc))
    # Any zero not connected to border = enclosed
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not border_connected[r][c]:
                return True
    return False


def _detect_separators(grid: Grid) -> tuple[bool, int]:
    """Detect uniform-color separator rows/cols. Returns (has_sep, n_sections)."""
    if not grid or len(grid) < 3:
        return False, 1
    h, w = len(grid), len(grid[0])

    # Check for separator rows (all same non-zero color)
    sep_rows = []
    for r in range(h):
        row = grid[r]
        if len(set(row)) == 1 and row[0] != 0:
            sep_rows.append(r)

    # Check for separator cols
    sep_cols = []
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] != 0:
            sep_cols.append(c)

    if sep_rows or sep_cols:
        # Count sections: rows create horizontal sections, cols create vertical
        n_row_sections = len(sep_rows) + 1 if sep_rows else 1
        n_col_sections = len(sep_cols) + 1 if sep_cols else 1
        n_sections = max(n_row_sections, n_col_sections)
        return True, n_sections
    return False, 1


def _shapes_match(grid_a: Grid, grid_b: Grid) -> bool:
    """Check if two grids have the same non-zero pixel positions."""
    if len(grid_a) != len(grid_b) or len(grid_a[0]) != len(grid_b[0]):
        return False
    for r in range(len(grid_a)):
        for c in range(len(grid_a[0])):
            if (grid_a[r][c] != 0) != (grid_b[r][c] != 0):
                return False
    return True


def fingerprint_task(task: Task) -> TaskFingerprint:
    """Compute structural fingerprint from training examples.

    Aggregates features across all examples, using majority voting for booleans.
    """
    fp = TaskFingerprint()
    examples = task.train_examples
    if not examples:
        return fp

    # --- Dimension change ---
    dim_votes = []
    for inp, out in examples:
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if ih == oh and iw == ow:
            dim_votes.append("same")
        elif oh * ow < ih * iw:
            dim_votes.append("shrink")
        else:
            dim_votes.append("grow")
    if len(set(dim_votes)) == 1:
        fp.dim_change = dim_votes[0]
    else:
        fp.dim_change = "variable"

    # --- Colors ---
    all_colors_in = set()
    all_colors_out = set()
    for inp, out in examples:
        for row in inp:
            all_colors_in.update(row)
        for row in out:
            all_colors_out.update(row)
    fp.n_colors_in = len(all_colors_in)
    fp.n_colors_out = len(all_colors_out)
    fp.colors_added = len(all_colors_out - all_colors_in)
    fp.colors_removed = len(all_colors_in - all_colors_out)

    # --- Pixel diff ratio (only for same-dim) ---
    if fp.dim_change == "same":
        total_pixels = 0
        diff_pixels = 0
        for inp, out in examples:
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    total_pixels += 1
                    if inp[r][c] != out[r][c]:
                        diff_pixels += 1
        fp.pixel_diff_ratio = diff_pixels / max(total_pixels, 1)

    # --- Recoloring detection (same shapes, different colors) ---
    if fp.dim_change == "same":
        fp.is_recoloring = all(_shapes_match(inp, out) for inp, out in examples)

    # --- Holes ---
    hole_votes = [_has_enclosed_zeros(inp) for inp, _ in examples]
    fp.has_holes = sum(hole_votes) > len(hole_votes) / 2

    # --- Objects ---
    obj_counts = [_count_objects(inp) for inp, _ in examples]
    fp.n_objects = max(obj_counts) if obj_counts else 0
    if len(obj_counts) > 1:
        mean_obj = sum(obj_counts) / len(obj_counts)
        fp.object_size_var = sum((x - mean_obj) ** 2 for x in obj_counts) / len(obj_counts)

    # --- Separators ---
    sep_votes = []
    section_counts = []
    for inp, _ in examples:
        has_sep, n_sec = _detect_separators(inp)
        sep_votes.append(has_sep)
        section_counts.append(n_sec)
    fp.has_separators = sum(sep_votes) > len(sep_votes) / 2
    fp.n_sections = max(section_counts) if section_counts else 1

    return fp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fingerprint.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add domains/arc/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: add task fingerprinting for ARC search stratification"
```

---

### Task 6: Implement propose_strata() in ARCGrammar

**Files:**
- Modify: `domains/arc/grammar.py`
- Test: `tests/test_arc.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_arc.py — add
def test_arc_grammar_propose_strata_always_includes_core():
    """Every task gets exhaustive_core stratum."""
    from domains.arc.grammar import ARCGrammar
    from core.types import Task

    g = ARCGrammar()
    task = Task("t1", [([[1, 0], [0, 1]], [[0, 1], [1, 0]])], [[[1, 0], [0, 1]]])
    prims = g.base_primitives()
    strata = g.propose_strata(task, prims)
    names = [s.name for s in strata]
    assert "exhaustive_core" in names
    # Budget fractions should sum to ~1.0
    total = sum(s.budget_fraction for s in strata)
    assert 0.95 <= total <= 1.05


def test_arc_grammar_propose_strata_triggers_local_rules():
    """Same-dim task with low pixel diff should trigger local_rules stratum."""
    from domains.arc.grammar import ARCGrammar
    from core.types import Task

    g = ARCGrammar()
    # Same-dim, only 1 pixel changed out of 4
    task = Task("t1", [
        ([[1, 2], [3, 4]], [[1, 2], [3, 5]]),
        ([[5, 6], [7, 8]], [[5, 6], [7, 9]]),
    ], [[[1, 2], [3, 4]]])
    prims = g.base_primitives()
    strata = g.propose_strata(task, prims)
    names = [s.name for s in strata]
    assert "local_rules" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arc.py::test_arc_grammar_propose_strata_always_includes_core -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Add to `ARCGrammar` in `domains/arc/grammar.py`:

```python
def propose_strata(self, task, primitives):
    """Propose search strata based on task fingerprint."""
    from core.types import SearchStratum
    from .fingerprint import fingerprint_task

    fp = fingerprint_task(task)

    # All primitive names for reference
    all_names = [p.name for p in primitives]

    strata = []
    triggered = []

    # Always: exhaustive_core with all current primitives
    core = SearchStratum(
        name="exhaustive_core",
        primitive_names=all_names,
        budget_fraction=0.4,  # set below after counting triggered
        try_corrections=True,
        metadata={"run_object_decomp": True, "run_for_each_object": True},
    )
    strata.append(core)

    # Conditional strata based on fingerprint
    if fp.has_holes or fp.symmetry_broken or fp.has_periodic:
        triggered.append(SearchStratum(
            name="inpainting",
            primitive_names=[n for n in all_names if "inpaint" in n or "symmetry" in n or "fill" in n],
            metadata={"run_cross_ref": True},
        ))

    if fp.has_separators:
        triggered.append(SearchStratum(
            name="separator_algebra",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "section", "separator", "overlay", "quadrant", "subgrid"])],
            metadata={"run_cross_ref": True},
        ))

    if fp.n_objects >= 2 and fp.dim_change == "same":
        triggered.append(SearchStratum(
            name="object_transform",
            primitive_names=all_names,  # full set for per-object
            metadata={"run_object_decomp": True, "run_for_each_object": True,
                      "run_procedural": True},
        ))

    if fp.n_objects >= 2 and fp.dim_change == "shrink":
        triggered.append(SearchStratum(
            name="object_extraction",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "extract", "crop", "densest", "colorful", "largest", "smallest"])],
        ))

    if fp.dim_change == "same" and fp.pixel_diff_ratio < 0.5:
        triggered.append(SearchStratum(
            name="local_rules",
            primitive_names=[],  # local rules don't use primitive enumeration
            metadata={"run_local_rules": True},
        ))

    if fp.dim_change == "grow" or fp.output_is_subgrid:
        triggered.append(SearchStratum(
            name="tiling_scaling",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "tile", "scale", "mirror_tile", "border"])],
        ))

    if fp.is_recoloring or fp.colors_added > 0:
        triggered.append(SearchStratum(
            name="color_logic",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "color", "swap", "replace", "keep", "erase", "recolor"])],
        ))

    if fp.n_objects >= 2 and fp.colors_added > 0:
        triggered.append(SearchStratum(
            name="line_drawing",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "connect", "extend", "draw", "line"])],
        ))

    if fp.symmetry_broken or fp.has_periodic:
        triggered.append(SearchStratum(
            name="pattern_completion",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "symmetry", "extrapolate", "pattern", "inpaint", "complete"])],
        ))

    if fp.n_sections >= 2 or (fp.n_objects >= 2 and fp.object_size_var < 1.0):
        triggered.append(SearchStratum(
            name="template_stamping",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "template", "stamp", "section", "apply", "overlay"])],
        ))

    if fp.pixel_diff_ratio < 0.1 and fp.dim_change == "same":
        triggered.append(SearchStratum(
            name="denoising",
            primitive_names=[n for n in all_names if any(k in n for k in [
                "remove", "isolated", "majority", "morpho", "close", "erode", "dilate"])],
        ))

    # Budget allocation
    n_triggered = len(triggered)
    if n_triggered == 0:
        core.budget_fraction = 1.0
    else:
        core.budget_fraction = 0.4
        per_stratum = 0.6 / n_triggered
        per_stratum = max(0.05, min(0.30, per_stratum))
        # Normalize so total = 1.0
        total_triggered = per_stratum * n_triggered
        core.budget_fraction = 1.0 - total_triggered
        for s in triggered:
            s.budget_fraction = per_stratum

    strata.extend(triggered)
    return strata
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arc.py::test_arc_grammar_propose_strata_always_includes_core tests/test_arc.py::test_arc_grammar_propose_strata_triggers_local_rules -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add domains/arc/grammar.py tests/test_arc.py
git commit -m "feat: implement propose_strata() in ARCGrammar using task fingerprinting"
```

---

## Chunk 3: Learner Wake Loop Refactor (Phase 1c)

### Task 7: Refactor wake loop to two-stage model

**Files:**
- Modify: `core/learner.py`
- Test: `tests/test_learner.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_learner.py — add
def test_wake_uses_strata_from_grammar():
    """Wake loop should call grammar.propose_strata() and iterate over them."""
    from core.types import Task, Primitive, Program, SearchStratum
    from core.interfaces import Environment, Grammar, DriveSignal
    from core.memory import InMemoryStore
    from core.config import SearchConfig
    from core.learner import Learner

    call_log = []

    class TrackingGrammar(Grammar):
        def base_primitives(self):
            return [Primitive("identity", 0, lambda x: x)]
        def compose(self, outer, inner_programs):
            return Program(root=outer.name, children=inner_programs)
        def mutate(self, program, primitives, tm=None):
            return program
        def crossover(self, a, b):
            return a
        def propose_strata(self, task, primitives):
            call_log.append("propose_strata")
            return [SearchStratum(
                name="test_stratum",
                primitive_names=["identity"],
                budget_fraction=1.0,
            )]

    class SimpleEnv(Environment):
        def load_task(self, task): return None
        def execute(self, program, input_data): return input_data
        def reset(self): pass

    class SimpleDrive(DriveSignal):
        def prediction_error(self, predicted, expected):
            return 0.0 if predicted == expected else 1.0

    env = SimpleEnv()
    grammar = TrackingGrammar()
    drive = SimpleDrive()
    memory = InMemoryStore()
    cfg = SearchConfig(exhaustive_depth=1, eval_budget=100)

    learner = Learner(env, grammar, drive, memory, cfg)
    task = Task("t1", [(1, 1)], [1], test_outputs=[1])
    result = learner.wake_on_task(task)

    assert "propose_strata" in call_log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learner.py::test_wake_uses_strata_from_grammar -v`
Expected: FAIL (current wake loop doesn't call propose_strata)

- [ ] **Step 3: Refactor _wake_core in core/learner.py**

The key change: replace `_wake_phases()` iteration with two stages.

In `_wake_core()`, replace:
```python
for phase_fn in self._wake_phases():
    solved_by = phase_fn(ctx)
    if solved_by is not None:
        return self._make_solved_result(ctx, solved_by)
```

With:
```python
# Stage 1: Stratum enumeration
strata = self.grammar.propose_strata(task, all_prims)
for stratum in strata:
    if ctx.solved:
        break
    solved_by = self._run_stratum_enumeration(ctx, stratum, all_prims)
    if solved_by is not None:
        return self._make_solved_result(ctx, solved_by)

# Stage 2: Structural hooks (run once with aggregated candidates)
if not ctx.solved and self.grammar.allow_structural_phases():
    for hook_fn in self._structural_hooks():
        solved_by = hook_fn(ctx)
        if solved_by is not None:
            return self._make_solved_result(ctx, solved_by)
```

Add new method `_run_stratum_enumeration`:
```python
def _run_stratum_enumeration(self, ctx: _WakeContext, stratum, all_prims) -> Optional[str]:
    """Run exhaustive enumeration for a single stratum."""
    # Filter primitives to this stratum's subset
    stratum_prims = [p for p in all_prims if p.name in set(stratum.primitive_names)]
    if not stratum_prims and stratum.name != "local_rules":
        return None

    # Scale eval budget by stratum's fraction
    original_budget = ctx.eval_budget
    if ctx.eval_budget > 0:
        ctx.eval_budget = int(original_budget * stratum.budget_fraction)

    # Run exhaustive enumeration with stratum's primitives
    if stratum_prims:
        t = time.time()
        candidates, n_evals = self._exhaustive_enumerate(
            stratum_prims, ctx.task, min(ctx.cfg.exhaustive_depth, stratum.max_depth),
            eval_budget=ctx.eval_budget)
        ctx.n_evals += n_evals
        for sp in candidates:
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
        ctx.enum_candidates.extend(candidates)
        logger.debug(f"  [wake] Stratum '{stratum.name}': {time.time()-t:.2f}s, {n_evals} evals, {len(candidates)} candidates")

    # Restore budget
    ctx.eval_budget = original_budget

    return stratum.name if ctx.solved else None
```

Rename `_wake_phases` to `_structural_hooks` and remove exhaustive from it:
```python
def _structural_hooks(self):
    """Structural hook methods run once after all strata."""
    return [
        self._phase_object_decomposition,
        self._phase_for_each_object,
        self._phase_cross_reference,
        self._phase_local_rules,
        self._phase_procedural,
        self._phase_conditional_search,
        self._phase_color_fix,
        self._phase_input_pred_correction,
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_learner.py::test_wake_uses_strata_from_grammar -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add core/learner.py tests/test_learner.py
git commit -m "refactor: two-stage wake loop — stratum enumeration + structural hooks"
```

---

### Task 8: Run baseline benchmark to verify zero regression

- [ ] **Step 1: Run quick mode benchmark**

Run: `python -m common --domain arc-agi-1 --mode quick`
Expected: ~14/50 train, ~4/50 eval (within ± 1 of previous)

- [ ] **Step 2: Run full default benchmark**

Run: `python -m common --domain arc-agi-1 --mode default`
Expected: ~118/400 train, ~49/400 eval (within ± 2 of previous)

- [ ] **Step 3: Record results in DECISIONS.md**

```markdown
## Decision 125: Search Stratification Foundation (Phase 1a-1c)

**Date:** 2026-03-19
**Change:** Refactored wake loop to two-stage model (stratum enumeration + structural hooks).
Added SearchStratum type, propose_strata() on Grammar, fingerprinting.
**Results:** Train [X]/400, Eval [X]/400 (baseline: 118/49)
**Regression:** [none / details]
```

- [ ] **Step 4: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: record Phase 1 baseline results in DECISIONS.md"
```

---

## Chunk 4: Extend LOOCV & New Primitives Tier 1 (Phase 1d + 2a)

### Task 9: Extend LOOCV to colormaps and procedural learning

**Files:**
- Modify: `domains/arc/environment.py`
- Test: `tests/test_arc.py`

- [ ] **Step 1: Identify colormap and procedural learning code paths**

Read `domains/arc/environment.py` and locate:
- `try_cross_reference()` — contains half_colormap, transform_colormap learning
- `try_procedural()` — contains per-object action learning
- Find where these produce solutions WITHOUT LOOCV validation

- [ ] **Step 2: Write failing test for LOOCV on colormap**

```python
# In tests/test_arc.py — add
def test_colormap_rejects_overfit_with_loocv():
    """Colormap that perfectly fits 1 example but fails on holdout should be rejected."""
    # This test verifies LOOCV is applied to colormaps
    # Specific implementation depends on how colormaps are currently validated
    # The test should show that a colormap learned from N-1 examples
    # is validated on the held-out example before being accepted
    pass  # Will be filled based on code reading
```

- [ ] **Step 3: Add LOOCV validation to colormap learning**

In the half_colormap section of `try_cross_reference()`, wrap the learning loop with:
```python
# For each candidate colormap, validate with LOOCV:
# 1. For i in range(n_train):
#    a. Learn colormap from all examples EXCEPT i
#    b. Apply to example i
#    c. If prediction != expected for example i, reject
# 2. Only accept colormaps that pass LOOCV
```

- [ ] **Step 4: Apply same pattern to procedural learning**

In `try_procedural()`, add LOOCV validation before accepting any learned rule.

- [ ] **Step 5: Run tests**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Run quick benchmark**

Run: `python -m common --domain arc-agi-1 --mode quick`
Expected: Same or slightly better train, potentially better eval (fewer overfitting solutions)

- [ ] **Step 7: Commit**

```bash
git add domains/arc/environment.py tests/test_arc.py
git commit -m "feat: extend LOOCV validation to colormaps and procedural learning"
```

---

### Task 10: Add Tier 1 inpainting primitives (5 new)

**Files:**
- Modify: `domains/arc/transformation_primitives.py`
- Test: `tests/test_new_primitives.py`

- [ ] **Step 1: Write failing tests for inpainting primitives**

```python
# tests/test_new_primitives.py
"""Tests for new primitives added in stratified search expansion."""
import pytest


def test_inpaint_by_neighbors():
    from domains.arc.transformation_primitives import inpaint_by_neighbors
    # Grid with a hole at (1,1) surrounded by 2s
    grid = [[2, 2, 2], [2, 0, 2], [2, 2, 2]]
    result = inpaint_by_neighbors(grid)
    assert result[1][1] == 2  # hole filled with majority neighbor


def test_inpaint_by_neighbors_no_holes():
    from domains.arc.transformation_primitives import inpaint_by_neighbors
    grid = [[1, 2], [3, 4]]
    result = inpaint_by_neighbors(grid)
    assert result == grid  # no zeros, no change


def test_remove_isolated():
    from domains.arc.transformation_primitives import remove_isolated
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    result = remove_isolated(grid)
    assert result[1][1] == 0  # isolated pixel removed


def test_remove_isolated_keeps_connected():
    from domains.arc.transformation_primitives import remove_isolated
    grid = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
    result = remove_isolated(grid)
    assert result[0][0] == 1  # connected, kept
    assert result[0][1] == 1  # connected, kept


def test_symmetry_complete_horizontal():
    from domains.arc.transformation_primitives import symmetry_complete
    # Left half has pattern, right half has holes
    grid = [[1, 2, 0, 0], [3, 4, 0, 0]]
    result = symmetry_complete(grid)
    # Should complete to mirror: [[1,2,2,1],[3,4,4,3]]
    assert result[0][2] != 0 or result[0][3] != 0  # at least some filling


def test_majority_filter_3x3():
    from domains.arc.transformation_primitives import majority_filter_3x3
    # Single noise pixel
    grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    result = majority_filter_3x3(grid)
    assert result[1][1] == 1  # noise smoothed to majority


def test_morphological_close():
    from domains.arc.transformation_primitives import morphological_close
    # Small gap in a line
    grid = [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
    result = morphological_close(grid)
    # Dilate then erode should fill the gap
    assert result[0][1] == 1  # gap filled


def test_remove_border():
    from domains.arc.transformation_primitives import remove_border
    grid = [[5, 5, 5], [5, 1, 5], [5, 5, 5]]
    result = remove_border(grid)
    assert result == [[1]]


def test_extend_up():
    from domains.arc.transformation_primitives import extend_up
    grid = [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    result = extend_up(grid)
    assert result[0][1] == 1
    assert result[1][1] == 1
    assert result[2][1] == 1


def test_extend_left():
    from domains.arc.transformation_primitives import extend_left
    grid = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    result = extend_left(grid)
    assert result[1][0] == 1
    assert result[1][1] == 1


def test_extend_right():
    from domains.arc.transformation_primitives import extend_right
    grid = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    result = extend_right(grid)
    assert result[1][1] == 1
    assert result[1][2] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_new_primitives.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the primitives**

Add to `domains/arc/transformation_primitives.py`:

```python
# --- Inpainting ---

def inpaint_by_neighbors(grid: Grid) -> Grid:
    """Fill zeros with majority color of non-zero 4-neighbors. Iterative."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    changed = True
    max_iters = max(h, w)
    for _ in range(max_iters):
        if not changed:
            break
        changed = False
        for r in range(h):
            for c in range(w):
                if result[r][c] != 0:
                    continue
                nbrs = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] != 0:
                        nbrs.append(result[nr][nc])
                if nbrs:
                    from collections import Counter
                    result[r][c] = Counter(nbrs).most_common(1)[0][0]
                    changed = True
    return result


def symmetry_complete(grid: Grid) -> Grid:
    """Detect nearest symmetry axis and fill zeros with symmetric counterpart."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Try horizontal axis (left-right mirror)
    h_score = 0
    h_total = 0
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            if result[r][c] != 0 and result[r][mc] != 0:
                h_total += 1
                if result[r][c] == result[r][mc]:
                    h_score += 1

    # Try vertical axis (top-bottom mirror)
    v_score = 0
    v_total = 0
    for r in range(h // 2):
        mr = h - 1 - r
        for c in range(w):
            if result[r][c] != 0 and result[mr][c] != 0:
                v_total += 1
                if result[r][c] == result[mr][c]:
                    v_score += 1

    h_ratio = h_score / max(h_total, 1)
    v_ratio = v_score / max(v_total, 1)

    if h_ratio >= v_ratio and h_ratio > 0.5:
        # Fill using horizontal symmetry
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if result[r][c] == 0 and result[r][mc] != 0:
                    result[r][c] = result[r][mc]
    elif v_ratio > 0.5:
        # Fill using vertical symmetry
        for r in range(h):
            mr = h - 1 - r
            for c in range(w):
                if result[r][c] == 0 and result[mr][c] != 0:
                    result[r][c] = result[mr][c]

    return result


def fill_by_row_col_pattern(grid: Grid) -> Grid:
    """Detect per-row and per-col color patterns, fill zeros at intersections."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # For each row, find the dominant non-zero color
    row_colors = []
    for r in range(h):
        colors = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        if colors:
            from collections import Counter
            row_colors.append(Counter(colors).most_common(1)[0][0])
        else:
            row_colors.append(0)

    # For each col, find the dominant non-zero color
    col_colors = []
    for c in range(w):
        colors = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        if colors:
            from collections import Counter
            col_colors.append(Counter(colors).most_common(1)[0][0])
        else:
            col_colors.append(0)

    # Fill zeros: prefer row color, fallback to col color
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                if row_colors[r] != 0:
                    result[r][c] = row_colors[r]
                elif col_colors[c] != 0:
                    result[r][c] = col_colors[c]

    return result


def inpaint_diagonal(grid: Grid) -> Grid:
    """Fill zeros by extrapolating diagonal color sequences."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Collect diagonals (top-left to bottom-right)
    for d in range(-(h - 1), w):
        diag = []
        for r in range(h):
            c = d + r
            if 0 <= c < w:
                diag.append((r, c, result[r][c]))
        if not diag:
            continue
        # Find non-zero values and their positions
        nz = [(i, v) for i, (r, c, v) in enumerate(diag) if v != 0]
        if len(nz) < 2:
            continue
        # Check if non-zero values form a repeating pattern
        vals = [v for _, v in nz]
        if len(set(vals)) == 1:
            # Constant diagonal — fill zeros with that color
            fill_color = vals[0]
            for i, (r, c, v) in enumerate(diag):
                if v == 0:
                    result[r][c] = fill_color

    return result


def inpaint_from_template(grid: Grid) -> Grid:
    """Find most common NxN pattern in non-zero regions, stamp into zero regions."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Try template sizes 2x2 and 3x3
    for ts in [2, 3]:
        if h < ts or w < ts:
            continue
        from collections import Counter
        patterns = Counter()
        for r in range(h - ts + 1):
            for c in range(w - ts + 1):
                block = tuple(grid[r + dr][c + dc] for dr in range(ts) for dc in range(ts))
                if 0 not in block:
                    patterns[block] += 1
        if not patterns:
            continue
        best_pattern = patterns.most_common(1)[0][0]
        # Stamp into zero regions where it fits
        for r in range(h - ts + 1):
            for c in range(w - ts + 1):
                block = [grid[r + dr][c + dc] for dr in range(ts) for dc in range(ts)]
                if any(v == 0 for v in block):
                    # Check if non-zero cells match the pattern
                    match = True
                    for dr in range(ts):
                        for dc in range(ts):
                            v = grid[r + dr][c + dc]
                            if v != 0 and v != best_pattern[dr * ts + dc]:
                                match = False
                                break
                        if not match:
                            break
                    if match:
                        for dr in range(ts):
                            for dc in range(ts):
                                if result[r + dr][c + dc] == 0:
                                    result[r + dr][c + dc] = best_pattern[dr * ts + dc]
        break  # use first successful template size
    return result


# --- Denoising ---

def remove_isolated(grid: Grid) -> Grid:
    """Remove non-zero pixels with no non-zero 4-neighbors."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                    has_neighbor = True
                    break
            if not has_neighbor:
                result[r][c] = 0
    return result


def majority_filter_3x3(grid: Grid) -> Grid:
    """Replace each pixel with majority color in its 3x3 neighborhood."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    from collections import Counter
    for r in range(h):
        for c in range(w):
            colors = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        colors.append(grid[nr][nc])
            result[r][c] = Counter(colors).most_common(1)[0][0]
    return result


def morphological_close(grid: Grid) -> Grid:
    """Dilate then erode (fills small gaps in non-zero regions)."""
    return erode(dilate(grid))


# --- Grid structure ---

def remove_border(grid: Grid) -> Grid:
    """Strip outermost row and column on all sides."""
    if not grid or len(grid) < 3:
        return grid
    if len(grid[0]) < 3:
        return grid
    return [row[1:-1] for row in grid[1:-1]]


# --- Cardinal extensions ---

def extend_up(grid: Grid) -> Grid:
    """Each non-zero pixel extends upward until hitting another non-zero."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        for r in range(h - 1, -1, -1):
            if grid[r][c] != 0:
                color = grid[r][c]
                for r2 in range(r - 1, -1, -1):
                    if grid[r2][c] != 0:
                        break
                    result[r2][c] = color
    return result


def extend_left(grid: Grid) -> Grid:
    """Each non-zero pixel extends leftward until hitting another non-zero."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                for c2 in range(c - 1, -1, -1):
                    if grid[r][c2] != 0:
                        break
                    result[r][c2] = color
    return result


def extend_right(grid: Grid) -> Grid:
    """Each non-zero pixel extends rightward until hitting another non-zero."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w - 1, -1, -1):
            if grid[r][c] != 0:
                color = grid[r][c]
                for c2 in range(c + 1, w):
                    if grid[r][c2] != 0:
                        break
                    result[r][c2] = color
    return result
```

Register all new primitives in `domains/arc/primitives.py` `_register_atomics()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_new_primitives.py -v`
Expected: PASS

- [ ] **Step 5: Register primitives and run quick benchmark**

Run: `python -m common --domain arc-agi-1 --mode quick`
Expected: >= 14/50 train (no regression)

- [ ] **Step 6: Commit**

```bash
git add domains/arc/transformation_primitives.py domains/arc/primitives.py tests/test_new_primitives.py
git commit -m "feat: add Tier 1 primitives — inpainting, denoising, extensions, grid structure"
```

---

### Task 11: Add Tier 1 object relationship primitives (5 new)

**Files:**
- Modify: `domains/arc/transformation_primitives.py`
- Modify: `domains/arc/perception_primitives.py`
- Test: `tests/test_new_primitives.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_new_primitives.py

def test_n_objects_perception():
    from domains.arc.perception_primitives import n_objects
    grid = [[1, 0, 2], [0, 0, 0], [3, 0, 0]]
    assert n_objects(grid) == 3


def test_n_objects_empty():
    from domains.arc.perception_primitives import n_objects
    grid = [[0, 0], [0, 0]]
    assert n_objects(grid) == 0


def test_draw_line_between_objects():
    from domains.arc.transformation_primitives import draw_line_between_objects
    # Two same-color pixels separated by gap
    grid = [[1, 0, 0, 1], [0, 0, 0, 0]]
    result = draw_line_between_objects(grid)
    assert result[0][1] == 1  # gap filled
    assert result[0][2] == 1  # gap filled


def test_color_by_object_rank():
    from domains.arc.transformation_primitives import color_by_object_rank
    # 3 objects of different sizes
    grid = [[1, 1, 0, 2, 0, 3, 3, 3]]
    result = color_by_object_rank(grid)
    # Largest (3-cell) gets rank 1, medium (2-cell) gets rank 2, smallest (1-cell) gets rank 3
    assert result[0][5] == 1  # largest object
    assert result[0][0] == 2  # medium object
    assert result[0][3] == 3  # smallest object
```

- [ ] **Step 2: Run to verify failure, then implement**

- [ ] **Step 3: Implement n_objects, draw_line_between_objects, color_by_object_rank, align_objects_horizontal, align_objects_vertical, sort_objects_by_size**

- [ ] **Step 4: Run tests, run quick benchmark, commit**

```bash
git commit -m "feat: add Tier 1 object relationship primitives"
```

---

### Task 12: Run full benchmark after Tier 1

- [ ] **Step 1: Run default mode**

Run: `python -m common --domain arc-agi-1 --mode default`
Expected: Train 130+ (32%), Eval 55+ (14%)

- [ ] **Step 2: Record in DECISIONS.md**

- [ ] **Step 3: Commit results**

---

## Chunk 5: Tier 2 Primitives + New Local Rules (Phase 2b-2c)

### Task 13: Add Tier 2 primitives (overlay_and, color_intersection)

- [ ] **Step 1: Write tests, implement, verify, commit**

These are arity-2 primitives following the existing `overlay`, `mask_by` pattern.

### Task 14: Add 5 new local rule types

**Files:**
- Modify: `domains/arc/environment.py`
- Test: `tests/test_arc.py`

- [ ] **Step 1: Write failing tests for diagonal_nbr_rule**

```python
def test_diagonal_nbr_local_rule():
    """Diagonal neighbor rule should learn from (center, NW, NE, SW, SE) features."""
    # Test that the rule type is available and produces valid output
    pass  # Specific test depends on try_local_rules implementation
```

- [ ] **Step 2: Implement 5 new rule types in the local rules section of environment.py**

Each follows the existing pattern:
1. Define key function: `(center, features...) -> key_tuple`
2. Learn mapping from training examples
3. LOOCV validate
4. Return if valid

- [ ] **Step 3: Run tests, benchmark, commit**

### Task 15: Add Tier 3 speculative primitives

- [ ] **Step 1: Implement extrapolate_growth and stamp_at_colored_pixels**
- [ ] **Step 2: Run benchmark, decide if they help or hurt**
- [ ] **Step 3: Keep or remove based on results, commit**

### Task 16: Full Phase 2 benchmark

- [ ] **Step 1: Run default mode**

Expected: Train 140+ (35%), Eval 65+ (16%)

- [ ] **Step 2: Run contest mode**

Expected: Train 150+ (37%), Eval 75+ (19%)

- [ ] **Step 3: Record all results in DECISIONS.md, commit**

---

## Chunk 6: Search Evolution (Phase 3)

### Task 17: Correction-as-composition

**Files:**
- Modify: `core/learner.py`
- Test: `tests/test_learner.py`

- [ ] **Step 1: Write test for correction composition**

```python
def test_correction_as_composition():
    """After finding best candidate, correction should be tried automatically."""
    # Verify that infer_output_correction is called on the best candidate
    # when stratum.try_corrections is True
    pass
```

- [ ] **Step 2: In the wake loop, after all strata complete, try correction on best candidates**
- [ ] **Step 3: Run tests, benchmark, commit**

### Task 18: Bidirectional search

**Files:**
- Modify: `core/learner.py`
- Modify: `domains/arc/grammar.py`
- Test: `tests/test_bidirectional.py`

- [ ] **Step 1: Implement inverse_primitives() in ARCGrammar**

```python
def inverse_primitives(self):
    return {
        "rotate_90_cw": "rotate_90_ccw",
        "rotate_90_ccw": "rotate_90_cw",
        "rotate_180": "rotate_180",
        "mirror_horizontal": "mirror_horizontal",
        "mirror_vertical": "mirror_vertical",
        "transpose": "transpose",
    }
```

- [ ] **Step 2: Implement meet-in-the-middle in learner**
- [ ] **Step 3: Write tests, run benchmark, commit**

### Task 19: 5-round compounding

**Files:**
- Modify: `core/config.py`
- Modify: `core/learner.py` (or `common/benchmark.py` for pipeline logic)

- [ ] **Step 1: Update derive_rounds() for 5-round support**

```python
def derive_rounds(compute_cap: int) -> int:
    if compute_cap >= 10_000_000:
        return 5  # replaces old 3-round threshold at 20M
    if compute_cap >= 200_000:
        return 2
    return 1
```

- [ ] **Step 2: Implement unsolved-only search for rounds 3-5**
- [ ] **Step 3: Run contest mode benchmark**

Expected: Train 180+ (45%), Eval 120+ (30%)

- [ ] **Step 4: Record in DECISIONS.md, commit**

---

## Chunk 7: Generalization & Culture Transfer (Phase 4)

### Task 20: Stratum-level culture transfer

**Files:**
- Modify: `core/memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Extend save_culture/load_culture to include stratum_stats**
- [ ] **Step 2: Test serialization round-trip, commit**

### Task 21: Overfit detection

**Files:**
- Modify: `core/learner.py`
- Test: `tests/test_learner.py`

- [ ] **Step 1: Add minimum example coverage check**
- [ ] **Step 2: Add complexity-gated acceptance for task-specific rules**
- [ ] **Step 3: Test, benchmark, commit**

### Task 22: Primitive generality scoring in eval

**Files:**
- Modify: `domains/arc/grammar.py`

- [ ] **Step 1: Use generality scores to order primitives within strata during eval**
- [ ] **Step 2: Benchmark, commit**

### Task 23: Final benchmark — target 50%+ eval

- [ ] **Step 1: Run contest mode pipeline**

Run: `python -m common --domain arc-agi-1 --mode contest`
Expected: Train 220+ (55%), Eval 200+ (50%)

- [ ] **Step 2: If below target, analyze which strata underperform and iterate**
- [ ] **Step 3: Record final results in DECISIONS.md**
- [ ] **Step 4: Update README.md with new performance numbers**
- [ ] **Step 5: Final commit and push**

```bash
git push origin claude/zen-feynman
```
