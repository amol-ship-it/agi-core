# Stratified Search with Primitive Expansion — Design Spec

**Goal:** ARC-AGI-1 eval score from 12.2% (49/400) to >50% (200+/400)

**Date:** 2026-03-19

**Baseline:** Train 118/400 (29.5%), Eval 49/400 (12.2%), 2 rounds, ~2 min wall time

---

## 1. Problem Analysis

### 1.1 Root Causes

**Training gap (71% unsolved):** 97% of unsolved training tasks produce ZERO valid prediction. The search space doesn't contain the right programs. This is a primitive coverage problem.

**Generalization gap (train 29.5% vs eval 12.2%):** Task-specific strategies (local rules, colormaps, per-object recolor) learn mappings that don't transfer. LOOCV is only applied to local rules, not to colormaps or procedural learning.

**Search dilution:** DECISIONS.md documents that adding 3 non-solving primitives caused -3 task regression. More primitives without focus = worse results.

**Depth ceiling:** Depth-3 exhaustive enumeration is the practical ceiling. Compounding via library partially bridges to depth-4+ but saturates after 2 rounds.

### 1.2 Failure Categorization (Eval, 348 unsolved)

| Error Range | Count | % |
|-------------|-------|---|
| Near-miss (< 0.15) | 153 | 44% |
| Moderate (0.15 - 0.50) | 128 | 37% |
| Far (> 0.50) | 67 | 19% |

| Dimension Change | Count | % |
|-----------------|-------|---|
| Same-dim | 250 | 72% |
| Shrink | 70 | 20% |
| Grow | 25 | 7% |
| Variable | 3 | 1% |

### 1.3 Key Gaps by Category

- **Inpainting / hole-filling:** Diagonal extrapolation, template-based fill, symmetry completion
- **Object relationships:** Connecting, aligning, sorting, stamping objects
- **Grid structure:** Separator-based section extraction, template application across sections
- **Denoising:** Isolated pixel removal, majority filtering
- **Cardinal extensions:** Only extend_down exists; missing up/left/right
- **Color logic:** Object-rank coloring, grid intersection overlay

---

## 2. Architecture: Search Stratification

### 2.1 Core Concept: SearchStratum

Replace the monolithic 10-phase wake pipeline with a data-driven stratum system. Each stratum is a focused search context with its own primitive subset and compute budget.

**New type in `core/types.py`:**

```python
@dataclass
class SearchStratum:
    name: str
    primitive_names: list[str]   # subset to search over
    max_depth: int = 3
    budget_fraction: float = 0.1  # share of task compute budget
    try_corrections: bool = True
    metadata: dict = field(default_factory=dict)
    # Expected metadata keys (domain-specific, core ignores unknown keys):
    #   "run_object_decomp": bool — run try_object_decomposition in structural stage
    #   "run_cross_ref": bool — run try_cross_reference in structural stage
    #   "run_local_rules": bool — run try_local_rules in structural stage
    #   "run_procedural": bool — run try_procedural in structural stage
    #   "run_for_each_object": bool — run try_for_each_object in structural stage
    #   "run_conditional": bool — run try_conditional_per_object in structural stage
```

**New method on `Grammar` in `core/interfaces.py`:**

```python
def propose_strata(
    self, task: Task, primitives: list[Primitive]
) -> list[SearchStratum]:
    """Propose search strata for this task based on structural analysis.

    Each stratum is a focused search over a primitive subset.
    The core learner runs each stratum independently and keeps the best
    program found across all strata.

    Default: single stratum with all primitives (backward compatible).
    """
    return [SearchStratum(
        name="default",
        primitive_names=[p.name for p in primitives],
        budget_fraction=1.0,
    )]
```

**Rationale for Grammar (not Environment):** `Grammar` already owns search-planning concerns — `task_priority_primitives()`, `essential_pair_concepts()`, and `prepare_for_task()` all live there. Strata are a natural extension of "what to search with." Placing `propose_strata()` on `Environment` would split search-planning across two interfaces.

**Core learner change — explicit wake loop structure:**

The wake loop has two stages per task:

1. **Stratum enumeration stage:** For each stratum from `grammar.propose_strata(task, primitives)`, run exhaustive enumeration (depth 1..max_depth) over that stratum's primitive subset with that stratum's budget fraction. Collect best candidates across all strata into a unified candidate pool.

2. **Structural hooks stage:** Run the existing structural methods once globally using the aggregated candidates from stage 1:
   - `env.try_object_decomposition(task, stratum_primitives)` — per-object transforms
   - `env.try_for_each_object(task, candidates)` — top-K per-object
   - `env.try_cross_reference(task, stratum_primitives)` — separator/colormap ops
   - `env.try_local_rules(task)` — local neighborhood rules (promoted to interface)
   - `env.try_procedural(task)` — procedural object DSL (promoted to interface)
   - `env.try_conditional_per_object(task, candidates, predicates)` — conditional per-object
   - If `stratum.try_corrections`: `env.infer_output_correction()` on best candidate

This preserves current behavior while allowing strata to focus the enumeration stage. The structural hooks are NOT duplicated per stratum — they run once with the combined candidate pool.

**Interface cleanup:** `try_local_rules()` and `try_procedural()` are promoted from ad-hoc `hasattr` checks to first-class optional methods on `Environment` with default `return None`:

```python
def try_local_rules(self, task: Task) -> Optional[tuple[str, Any]]:
    """Try solving via learned pixel-level neighborhood rules.
    Default: not supported (returns None).
    """
    return None

def try_procedural(self, task: Task) -> Optional[tuple[str, Any]]:
    """Try solving via procedural object DSL (per-object action learning).
    Default: not supported (returns None).
    """
    return None
```

**Invariant preserved:** Core never imports domain code. `SearchStratum` is a pure data type. The domain's Grammar decides what strata to propose; the core decides how to search within each.

### 2.2 Backward Compatibility

The default `propose_strata()` returns a single stratum with all primitives, reproducing current behavior. Domains that don't implement fingerprinting get the same results as before.

---

## 3. Task Fingerprinting (Domain-Side)

### 3.1 TaskFingerprint

**New file: `domains/arc/fingerprint.py`**

```python
@dataclass
class TaskFingerprint:
    dim_change: str          # "same", "shrink", "grow", "variable"
    has_separators: bool
    n_sections: int
    symmetry: set[str]       # {"h", "v", "diag", "rot"}
    symmetry_broken: bool
    n_objects: int
    object_size_var: float
    has_periodic: bool
    has_holes: bool
    n_colors_in: int
    n_colors_out: int
    colors_added: int
    colors_removed: int
    pixel_diff_ratio: float
    output_is_subgrid: bool
    is_recoloring: bool

def fingerprint_task(task: Task) -> TaskFingerprint:
    """Analyze training examples to compute structural fingerprint.

    Must be fast (< 1ms per task) and conservative.
    """
    ...
```

### 3.2 Design Principles

- **Fast:** < 1ms per task. Fingerprinting must not add measurable overhead.
- **Conservative:** Fingerprinting only ADDS strata, never removes `exhaustive_core`. If fingerprinting is wrong, baseline behavior is preserved.
- **Aggregated:** Features are computed across ALL training examples and majority-voted. A feature is True only if it holds for most examples.

---

## 4. Stratum Definitions

### 4.1 The 12 Strata

| # | Stratum | Trigger | Primitives | Est. Train | Est. Eval |
|---|---------|---------|------------|------------|-----------|
| 1 | `exhaustive_core` | Always | All current 48 transforms + learned | Baseline | Baseline |
| 2 | `inpainting` | `has_holes` or `symmetry_broken` or `has_periodic` | inpaint_* family (existing + 5 new) | 10-15 | 5-10 |
| 3 | `separator_algebra` | `has_separators` | extract_section, overlay_sections, separator ops, cross-ref hooks | 12-18 | 8-15 |
| 4 | `object_transform` | `n_objects >= 2` and `dim_change == "same"` | per-object transforms, recolor strategies, sort/align objects | 25-35 | 15-25 |
| 5 | `object_extraction` | `n_objects >= 2` and `dim_change == "shrink"` | extract_by_property, filter objects, crop prims | 12-18 | 8-15 |
| 6 | `local_rules` | `dim_change == "same"` and `pixel_diff_ratio < 0.5` | all 12 + 5 new local rule types | 18-25 | 10-15 |
| 7 | `tiling_scaling` | `dim_change == "grow"` or `input_is_subgrid` | tile/scale/mirror_tile + border ops | 6-10 | 4-8 |
| 8 | `color_logic` | `is_recoloring` or `colors_added > 0` | swap/replace/keep/erase + sequential coloring | 8-12 | 6-10 |
| 9 | `pattern_completion` | `symmetry_broken` or `has_periodic` | symmetry_complete, extrapolate_pattern | 6-10 | 4-8 |
| 10 | `line_drawing` | `n_objects >= 2` and `colors_added > 0` | connect objects, draw lines, extend rays | 6-10 | 4-8 |
| 11 | `template_stamping` | `n_sections >= 2` or same-size objects | template extraction + application | 8-12 | 6-10 |
| 12 | `denoising` | `pixel_diff_ratio < 0.1` and `dim_change == "same"` | remove_isolated, majority_filter, morph_close | 3-6 | 2-5 |

### 4.2 Budget Allocation

- `exhaustive_core`: 40% of task compute budget (proven workhorse)
- Triggered strata share remaining 60% equally
- Minimum floor: 5% per stratum (prevents starvation)
- Maximum cap: no single triggered stratum gets more than 30% (prevents one stratum starving others)
- Example: if 4 strata trigger → core gets 40%, each triggered stratum gets 15%
- Example: if 8 strata trigger → core gets 40%, each gets max(5%, 60%/8) = 7.5%

**Relationship to `task_priority_primitives()`:** `task_priority_primitives()` provides priority ordering WITHIN a stratum's primitive set (which primitives to try first in exhaustive enumeration). Strata provide the partitioning (which primitives are in scope at all). These are complementary, not redundant.

### 4.3 Mapping from Current Phases

| Current Phase | Maps To |
|--------------|---------|
| 1-3 (exhaustive, object decomp, for-each) | `exhaustive_core` + `object_transform` |
| 4 (cross-reference) | `separator_algebra` |
| 5 (local rules) | `local_rules` |
| 6 (procedural) | `object_transform` + `template_stamping` |
| 7 (conditional) | Composition strategy within any stratum |
| 8-10 (corrections) | `try_corrections` flag per stratum |

---

## 5. New Primitives (25 Total)

### 5.1 Tier 1 — High ROI (implement first)

**Inpainting (5):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `inpaint_diagonal` | transform | Fill zeros by extrapolating diagonal color sequences |
| `inpaint_by_neighbors` | transform | Fill zeros with majority of non-zero neighbors (iterative) |
| `symmetry_complete` | transform | Detect symmetry axis, fill pixels that break symmetry |
| `inpaint_from_template` | transform | Find common NxN pattern, stamp into zero regions |
| `fill_by_row_col_pattern` | transform | Detect row/col sequences, fill zeros at intersections |

**Object Relationship (5):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `n_objects` | perception | Count of 4-connected non-zero components |
| `draw_line_between_objects` | transform | Connect centers of same-color objects with lines |
| `align_objects_horizontal` | transform | Stack objects horizontally by x-position |
| `align_objects_vertical` | transform | Stack objects vertically by y-position |
| `sort_objects_by_size` | transform | Rearrange objects spatially by size |

**Grid Structure (4):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `extract_section` | parameterized(1) | Split by separators, return Nth section |
| `apply_template_to_sections` | transform | Apply "key" section pattern to all others |
| `add_border` | parameterized(1) | Add 1-pixel border of given color |
| `remove_border` | transform | Strip outermost row/col on all sides |

### 5.2 Tier 2 — Medium ROI

**Extension (3):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `extend_up` | transform | Extend non-zero pixels upward until hitting another |
| `extend_left` | transform | Extend non-zero pixels leftward until hitting another |
| `extend_right` | transform | Extend non-zero pixels rightward until hitting another |

**Denoising (3):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `remove_isolated` | transform | Remove pixels with no non-zero 4-neighbors |
| `majority_filter_3x3` | transform | Replace each pixel with 3x3 neighborhood majority |
| `morphological_close` | transform | Dilate then erode |

**Color Logic (3):**

| Primitive | Kind | Description |
|-----------|------|-------------|
| `color_by_object_rank` | transform | Recolor objects by size rank |
| `overlay_and` | transform (arity-2) | Keep pixels non-zero in both grids |
| `color_intersection` | transform (arity-2) | Keep grid1 color where both non-zero |

**Note on arity-2 primitives:** These compose as `Program(root="overlay_and", children=[prog_A, prog_B])` where both children are evaluated independently on the same input. The grammar's `compose()` handles arity-2 nodes by evaluating each child program on the input grid, then applying the binary operation. The environment's `_eval_tree()` already supports this pattern via the existing `overlay`, `mask_by`, `subtract_grid`, and `xor_grid` primitives.

### 5.3 Tier 3 — Speculative

| Primitive | Kind | Description |
|-----------|------|-------------|
| `extrapolate_growth` | transform | Detect growing pattern, generate next step |
| `stamp_at_colored_pixels` | parameterized(1) | Use smallest object as stamp at color-N pixels |

### 5.4 Search Dilution Guard

New primitives are NOT added to `exhaustive_core`. Each is tagged with its stratum. Only activated when that stratum fires for a task. This is the critical design decision — stratification makes it safe to grow the primitive vocabulary.

---

## 6. Local Rule Expansion (5 New Types)

| Rule Type | Features | What It Captures |
|-----------|----------|-----------------|
| `diagonal_nbr_rule` | (center, NW, NE, SW, SE) | Diagonal pattern propagation |
| `cross_context_rule` | (center, up, down, left, right) | Full cardinal neighborhood |
| `distance_to_border_rule` | (center, min_dist_to_edge) | Border-proximity behavior |
| `object_membership_rule` | (center, obj_id_hash, obj_size_bucket) | Object-aware pixel rules |
| `row_col_position_rule` | (center, row%P, col%P) for P in 2..6 | Multi-period positional patterns |

**Validation:** All new rule types use LOOCV. No exceptions.

**Transfer improvement:**
1. Sleep extracts common local rule structures as library entries when same rule type solves 3+ tasks
2. Culture transfer includes rule type effectiveness signals (which types solved how many tasks)

---

## 7. Composition & Search Evolution

### 7.1 Correction-as-Composition (Core)

Formalize the current ad-hoc correction phases (8-10) as a first-class composition pattern:

```python
Program(root="correct", children=[base_program, correction_program])
```

After finding the best program per stratum, the core automatically tries `infer_output_correction()` on top. Controlled by `SearchStratum.try_corrections`. Replaces 3 separate phases with one unified mechanism.

### 7.2 Bidirectional Search (Core)

Meet-in-the-middle: search depth-2 forward from input AND depth-2 backward from output. Match = depth-4 solution via depth-2 cost.

**New optional method on `Grammar`:**

```python
def inverse_primitives(self) -> dict[str, str]:
    """Map primitive names to their inverses for bidirectional search.
    Default: empty.
    """
    return {}
```

**ARC inversions:** `rotate_90_cw ↔ rotate_90_ccw`, `rotate_180 ↔ rotate_180`, `mirror_h ↔ mirror_h`, `mirror_v ↔ mirror_v`, `transpose ↔ transpose`.

Activated when `SearchStratum.max_depth >= 4`.

### 7.3 Multi-Round Compounding (Core)

Increase from 2 to 5 rounds with progressive budget.

**Budget model:** The `compute_cap` parameter defines total compute across ALL rounds (not per-round). The `derive_rounds()` function in `config.py` is updated to return 5 rounds when `compute_cap >= 10M`. Each round's per-task budget is `compute_cap * round_share / n_tasks_in_round`.

| Round | Budget Share | Tasks Searched | Purpose |
|-------|-------------|----------------|---------|
| 1 | 40% | All 400 | Initial sweep, build base library |
| 2 | 25% | All 400 | Compound: depth-1 over learned = depth-3+ |
| 3 | 15% | Unsolved only (~280) | Deep compound: depth-2 over learned = depth-5+ |
| 4 | 10% | Unsolved only (~240) | Refinement: corrections, rare strata |
| 5 | 10% | Unsolved only (~220) | Final sweep with full library |

**Key optimization:** Rounds 3-5 only search unsolved tasks. Since each later round has fewer tasks to search, the per-task budget actually INCREASES despite the smaller total budget share. E.g., Round 3 has 15% of total budget but only ~280 tasks → more compute per task than Round 1 at 40%/400.

**Concrete budget example (contest mode, compute_cap=50M):**

| Round | Total Ops | Tasks | Per-Task Ops |
|-------|-----------|-------|-------------|
| 1 | 20M | 400 | 50K |
| 2 | 12.5M | 400 | 31K |
| 3 | 7.5M | ~280 | ~27K |
| 4 | 5M | ~240 | ~21K |
| 5 | 5M | ~220 | ~23K |

Library pruning between rounds removes entries with zero reuse.

---

## 8. Generalization — Closing the Train/Eval Gap

### 8.1 Stratum-Level Culture Transfer

Culture JSON enhanced with stratum effectiveness:

```json
{
    "library": ["...existing..."],
    "stratum_stats": {
        "inpainting": {"tasks_solved": 12, "tasks_tried": 45},
        "separator_algebra": {"tasks_solved": 8, "tasks_tried": 30}
    }
}
```

Eval allocates more budget to strata that were productive on train.

### 8.2 Overfit Detection

1. **Minimum example coverage:** Require `n_examples_perfect >= max(2, n_train - 1)`. Programs fitting only 1 of 3+ examples are rejected.
2. **Complexity-gated acceptance:** Task-specific rules (local rules, colormaps) must have lookup table size < `grid_cells / 4`.

### 8.3 Primitive Generality Scoring

**New optional method on `Memory`:**

```python
def get_primitive_generality(self) -> dict[str, float]:
    """n_distinct_tasks_solved / total_solved per primitive.
    Used to prioritize high-generality primitives in eval search.
    Default: uniform.
    """
    return {}
```

### 8.4 Universal LOOCV

Extend LOOCV validation (currently only on local rules) to ALL task-specific learning:
- Colormaps (half_colormap, transform_colormap)
- Procedural rules (per-object action learning)
- Input-pred corrections

This is the single biggest lever for closing the generalization gap.

---

## 9. Implementation Phases

### Phase 1: Foundation (Fingerprinting + Stratification + LOOCV)

**Changes:**
- `core/types.py`: Add `SearchStratum` dataclass
- `core/interfaces.py`: Add `propose_strata()` to `Grammar`, `inverse_primitives()` to `Grammar`, `get_primitive_generality()` to `Memory`, promote `try_local_rules()` and `try_procedural()` to `Environment` interface
- `core/learner.py`: Refactor wake loop to two-stage model (stratum enumeration + structural hooks)
- `domains/arc/fingerprint.py`: New file — `TaskFingerprint` + `fingerprint_task()`
- `domains/arc/grammar.py`: Implement `propose_strata()` using fingerprint
- `domains/arc/environment.py`: Extend LOOCV to colormaps, procedural, and input-pred corrections (moved from Phase 4 — low risk, high impact on generalization gap, independent of other changes)

**Checkpoint:** Full benchmark = same score as current (118/49 ± 2). Zero regression allowed. LOOCV extension may cause small eval improvement even without new primitives.

### Phase 2: New Primitives + Local Rules

**Changes:**
- `domains/arc/transformation_primitives.py`: Add 25 new primitives (tiered)
- `domains/arc/environment.py`: Add 5 new local rule types
- Each primitive tagged with stratum

**Sub-phases (measure after each):**
- 2a: Tier 1 (14 primitives) → target train 130+ (32%), eval 55+ (14%)
- 2b: Tier 2 (9 primitives) → target train 140+ (35%), eval 65+ (16%)
- 2c: Tier 3 (2 primitives) → target train 145+ (36%), eval 70+ (17%)

**Checkpoint:** Train 140+ (35%), Eval 65+ (16%). Note: primitives alone cannot reach 50% — they provide the building blocks that Phases 3-4 will compose and generalize.

### Phase 3: Search Evolution

**Changes:**
- `core/learner.py`: Correction-as-composition, bidirectional search, 5-round compounding
- `core/config.py`: Update `derive_rounds()` for 5 rounds at compute_cap >= 10M
- `domains/arc/grammar.py`: Implement `inverse_primitives()`

**Checkpoint:** Train 180+ (45%), Eval 120+ (30%). Multi-round compounding should show measurable gains in rounds 3-5. Bidirectional search may contribute 5-10 additional solves (limited by small invertible primitive set).

### Phase 4: Generalization & Culture Transfer

**Changes:**
- `core/learner.py`: Stratum culture transfer, overfit detection
- `core/memory.py`: Primitive generality tracking
- Culture JSON format extended with stratum_stats

**Checkpoint:** Train 220+ (55%), Eval 200+ (50%). The train/eval gap should shrink from ~17% to ~5%.

**Documentation:** Every phase checkpoint is recorded in DECISIONS.md with measured before/after numbers, rationale for any deviations from targets, and hyperparameter values used.

---

## 10. Measurement Protocol

| Scope | Mode | Tasks | Time | When |
|-------|------|-------|------|------|
| Quick check | `quick` | 50 | ~18s | Every code change |
| Full measure | `default` | 400 | ~5 min | After each sub-phase |
| Max accuracy | `contest` | 400 | ~15 min | After each phase |

Every result logged in DECISIONS.md with before/after numbers.

**Regression policy:** If any change causes net loss of > 2 tasks, revert and investigate before proceeding.

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Search dilution from new primitives | Stratification — new prims only in relevant strata |
| Fingerprint errors misclassifying tasks | `exhaustive_core` always runs; strata are additive |
| Bidirectional search compute cost | Only activated when `max_depth >= 4`; small invertible subset. Expected yield: 5-10 tasks (limited by only 5 invertible primitives). May be deprioritized if ROI is too low. |
| Regression from refactoring | Phase 1 checkpoint requires zero regression |
| Overfit detection too aggressive | Complexity gate tunable; start conservative, relax if needed |
| 5 rounds too slow | Progressive budget — later rounds search only unsolved tasks; total ~10-15 min |
| Many strata trigger simultaneously | Budget floor (5%) and cap (30%) per stratum. With 8 strata: core=40%, each=7.5%. Worst case with all 11 triggered: core=40%, each=5.4%. Budget arithmetic always adds to 100%. |
| Phase 1 refactoring causes subtle bugs in later phases | Branch per phase. If Phase 2 reveals Phase 1 issues, fix on Phase 1 branch and merge forward. |

---

## 12. Files Changed

### Core (domain-agnostic)
| File | Change |
|------|--------|
| `core/types.py` | Add `SearchStratum` dataclass |
| `core/interfaces.py` | Add `propose_strata()` and `inverse_primitives()` to Grammar; promote `try_local_rules()` and `try_procedural()` to Environment; add `get_primitive_generality()` to Memory |
| `core/learner.py` | Refactor wake loop to two-stage model, add correction-as-composition, bidirectional search, 5-round compounding |
| `core/config.py` | Update `derive_rounds()` for 5-round support |
| `core/memory.py` | Add primitive generality tracking |

### Domain (ARC-specific)
| File | Change |
|------|--------|
| `domains/arc/fingerprint.py` | **New** — TaskFingerprint + fingerprint_task() |
| `domains/arc/transformation_primitives.py` | Add 25 new primitives |
| `domains/arc/environment.py` | 5 new local rules, extend LOOCV to colormaps/procedural/corrections, remove hasattr checks for try_local_rules/try_procedural |
| `domains/arc/grammar.py` | Implement `propose_strata()` using fingerprint, implement `inverse_primitives()` |

### Tests
| File | Change |
|------|--------|
| `tests/test_fingerprint.py` | **New** — fingerprinting unit tests |
| `tests/test_strata.py` | **New** — stratification integration tests |
| `tests/test_new_primitives.py` | **New** — new primitive unit tests |
| `tests/test_bidirectional.py` | **New** — bidirectional search tests |
| Existing test files | Update for new interface methods |

---

## 13. 4 Pillars Alignment

| Pillar | How This Design Serves It |
|--------|--------------------------|
| **Feedback Loops** | Each stratum provides focused feedback; 5-round compounding = more learning cycles |
| **Approximability** | Fingerprinting creates smoother search landscape; corrections give partial credit |
| **Abstraction & Composability** | Strata are composable search strategies; bidirectional enables deeper compositions |
| **Exploration** | Multiple strata explore different structural hypotheses in parallel per task |

---

## 14. Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Train solve rate | 29.5% (118/400) | 55% (220/400) | 65% (260/400) |
| Eval solve rate | 12.2% (49/400) | **50% (200/400)** | 55% (220/400) |
| Train/eval gap | 17.3% | < 5% | < 3% |
| Wall time (default) | ~2 min | < 15 min | < 10 min |
| Library size | ~30 | 50-80 | 100+ |
