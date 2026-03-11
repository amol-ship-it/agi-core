# Decisions & Judgements — Chronological Log

**Author:** Claude (working with vibhor-77)
**Purpose:** Living document of all technical decisions, judgements, trade-offs, and rationale made during development. Newest entries at the bottom.

---

## Session 1 — Claude Mobile App (Early March 2026)

### Analysis: Repository Landscape

After analyzing all repositories under vibhor-77:

| Repository | What It Contains | Status |
|---|---|---|
| `agi-mvp0` | Early prototype | Superseded |
| `agi-mvp-codex` | Codex-based approach | Superseded |
| `agi-mvp-claude` | Symbolic regression composer + grid worlds | Useful components |
| `agi-mvp-no-noise` | Cleaned-up variant | Superseded |
| `agi-sota-prototypes` | UniversalSolver with Wake-Sleep across ARC + Zork | Key reference |
| `agi-mvp-arc-agi-1` | Production ARC solver with beam search + DreamCoder-style library | Key reference |
| `agi-mvp-general` | Four Pillars AGI agent, 287 primitives, most mature | Primary source |
| `agi-core` | **The canonical monorepo** (this project) | Active |

**Judgement:** The evolution across repos shows clear convergence toward the architecture described in the manifesto. Each repo explored a different facet (beam search, library learning, cross-domain transfer, evolutionary synthesis). `agi-core` should be the consolidation point.

### Understanding of the Vision

Vibhor's core claim: **There is one general learning algorithm.** Differences between learners (nematode, dog, human, AI) are in hardware and data stream, not in the algorithm itself. The 4 pillars:

1. **Feedback Loops** — Act in environment, observe consequences, compare predictions to reality
2. **Approximability** — Candidates approximate the true generating function with quantified error
3. **Abstraction & Composability** — Primitives compose into programs; recurring compositions compress into reusable library entries
4. **Exploration** — Balance exploitation of known strategies with exploration of novel ones

These interact in a compounding cycle: better abstractions shrink search, better exploration discovers higher-value abstractions, better approximation scores candidates reliably, feedback grounds everything in reality.

### Response to Skeptics

**"Aren't researchers already working on this?"**
Yes, in pieces. DreamCoder (Ellis et al., 2021) does wake-sleep library learning within single domains. Chollet's ARC framework defines the benchmark. Friston's Free Energy Principle provides theoretical grounding. LLM-based synthesis (Greenblatt) achieves high ARC scores. But nobody has built a single clean system that: (a) separates invariant loop from domain plugins, (b) demonstrates compounding across multiple unrelated domains, (c) shows transfer. The contribution is the integration and the empirical test.

**"Too high level, how will it work?"**
The manifesto provides a concrete 6-phase experimental roadmap with specific deliverables. Phase 0 (extract core) is done. Phase 1 (ARC-AGI-1 curriculum training) is in progress. Each phase tests a specific claim about generality.

**"Don't LLMs already do this?"**
No, for structural reasons: (1) No explicit compounding — LLMs can't permanently learn a new abstraction from a single interaction. (2) No inspectable library — knowledge is distributed across billions of parameters. (3) No closed-loop interaction at inference. (4) Supervision assumption — next-token prediction assumes training corpus is truth. (5) Resource intensity — training costs hundreds of millions. LLMs can serve as a component (heuristic guide, perceptual front-end) but should not be the entire architecture.

---

## Session 2 — Claude Code on Web (March 10, 2026)

### Decision: Repository Restructuring

**Context:** Files were flat in root directory. Manifesto describes a structured layout.
**Decision:** Restructure into `core/`, `grammars/`, `environments/`, `drives/`, `library/`, `experiments/`, `tests/`.
**Rationale:** The architecture should be visible in the file structure itself. The invariant/pluggable separation must be enforced at the directory level. This prevents accidental cross-contamination.

### Decision: Don't Use PySR or DreamCoder as Dependencies

**Context:** User asked whether to leverage existing PySR and DreamCoder packages.
**Decision:** Don't depend on either. Port ideas, not code.
**Rationale:**
- **PySR** is for symbolic regression specifically. It has its own search loop, which would bypass the universal core and break the "one algorithm" principle. Great for comparison baseline, wrong as a dependency.
- **DreamCoder** (original codebase) is OCaml+Python, poorly maintained, hard to integrate. The key ideas (wake-sleep library learning, transition matrix prior, compression) are better reimplemented cleanly within the existing architecture.
- The manifesto's whole point is that intelligence lives in the *loop and library*, not in a specific library's implementation. Dependencies would obscure this.

**What we ported instead:**
- DreamCoder's **transition matrix prior** P(child_op | parent_op) — biases program generation toward compositions observed in successful solutions
- DreamCoder's **library compression** — extract recurring sub-programs from solved tasks
- These are implemented directly in `core/learner.py` (the `TransitionMatrix` class and enhanced `sleep()` method), staying within the invariant core with no domain imports

### Decision: ARC-AGI Primitive Set (48 Primitives)

**Context:** `agi-mvp-general` has 287 primitives. How many to include in the clean `agi-core`?
**Decision:** Start with 48 carefully chosen primitives covering the most common ARC transformation categories.
**Rationale:** The manifesto's claim is about compounding from a *small* initial set. Starting with 287 would be testing search, not learning. 48 provides enough coverage for basic geometric, color, spatial, gravity, and pattern operations while leaving room for the library to discover compositions. Can always add more if the compounding curve plateaus due to primitive poverty rather than algorithmic limitation.

**Categories included:**
- Geometric (7): identity, rot90cw/ccw, rot180, mirror_h/v, transpose
- Color (11): invert, replace_bg, keep_c1-c9
- Spatial/Cropping (5): crop_nonzero, top/bottom/left/right_half
- Tiling/Scaling (4): tile_2x2/3x3, scale_2x/3x
- Gravity (4): down/up/left/right
- Pattern (4): outline, fill_enclosed, denoise_3x3, replace_bg_mc
- Logical (4): xor/or_halves_v/h
- Color removal (9): recolor_1-9_to_0
- Binary (1): overlay

### Decision: Living Documentation Strategy

**Context:** User wants chronological record of all prompts and decisions.
**Decision:** Maintain two living documents:
- `PROMPTS.md` — All user instructions in chronological order (the "what was asked")
- `DECISIONS.md` — All technical decisions and rationale (the "what was decided and why")
**Rationale:** This creates an inspectable reasoning trail, consistent with the manifesto's emphasis on explainability. It also means any future Claude session (or human reader) can understand the full trajectory of the project without access to ephemeral chat logs.

### Decision: Core Loop Must Never Import Domain Code

**Context:** User explicitly reminded this constraint.
**Verification:** Confirmed that `core/learner.py`, `core/interfaces.py`, `core/memory.py`, and `core/metrics.py` import only from Python standard library and from each other (`from .interfaces import ...`). Zero domain-specific imports. The `TransitionMatrix` class added to learner.py operates purely on `Program` and `Primitive` types defined in `interfaces.py` — these are domain-agnostic data structures.

### Judgement: Expected Baseline Performance on ARC-AGI-1

Based on analysis of the existing repos:
- `agi-mvp-arc-agi-1` achieved ~10% with pure beam search (as noted in the manifesto)
- `agi-mvp-general` with 287 primitives + evolutionary search + DSL synthesis achieves higher
- `agi-core` with 48 primitives + basic beam search should achieve roughly **5-15%** on the 400 training tasks in the first round
- The key metric is not the absolute number but whether it **increases across rounds** as the library grows
- Even a modest improvement (e.g., 8% round 1 -> 12% round 3) without new hand-coded primitives would validate the compounding claim

### Decision: Numpy-Optimized Grid Primitives

**Context:** `outline()`, `denoise_3x3()`, and `fill_enclosed()` had nested Python loops — O(rows × cols × neighbors).
**Decision:** Rewrite all three as vectorized numpy operations.
**Rationale:** These are in the hot path (called for every candidate evaluation). Pure Python nested loops on grids are 10-100x slower than numpy vectorized equivalents.
- `outline`: replaced with `np.pad` + boolean array operations
- `denoise_3x3`: replaced with shifted-window counting over 10 colors + `np.argmax`
- `fill_enclosed`: kept flood fill (inherently sequential) but vectorized the neighbor-color fill step

### Decision: Parallel Wake Phase with ProcessPoolExecutor

**Context:** Tasks are independent during wake phase — natural parallelism opportunity.
**Decision:** Use `ProcessPoolExecutor` with automatic fallback to sequential if pickling fails.
**Rationale:** Each worker gets a snapshot of the learner state, solves independently, and results merge back. Falls back gracefully to sequential on any error. Auto-detects CPU core count.

### Decision: Simplified Compute Budget

**Context:** User found the `eval_budget` concept in agi-mvp-general confusing (cell-normalized budgeting with proportional ceilings).
**Decision:** Remove `eval_budget` as a separate knob. The compute budget is simply `beam_width × max_generations`. Presets define it. Early stopping saves unused compute.
**Rationale:** The simplest correct thing. beam × gens IS the compute. Presets map to use cases:
- `quick` = 3,200 evals/task — fast iteration
- `default` = 12,000 evals/task — balanced
- `contest` = 100,000 evals/task — max accuracy

If we ever need finer control (e.g., spending more on hard tasks, less on easy ones), we can add adaptive allocation later as a separate improvement.

### Benchmark Results: Phase 1 Baseline

**Run config:** default mode, 50 real ARC-AGI-1 tasks, 3 rounds, beam 150, 80 gens, 4 cores.
**Results:**
- Round 1: 2/50 solved (4.0%)
- Round 2: 2/50 solved (4.0%)
- Round 3: 2/50 solved (4.0%)
- Total wall time: ~10 min

**Sample tasks (built-in, no dataset):** 8/8 solved (100%) in ~50s.

**Interpretation:** The 4% baseline on real tasks is expected — these are hard puzzles and our search is pure beam search without heuristic guidance. The fact that it's plateauing at 4% across rounds tells us the library isn't growing yet — we need to solve more tasks before sleep can extract useful abstractions. Next steps: improve search quality (not just quantity), possibly add heuristic-guided mutation.

## Session 2 — Claude Code Web (March 10, 2026)

### Decision: Port Three Improvements from agi-mvp-no-noise

**Context:** The agi-mvp-no-noise repo (THOUGHTS.md, NEXT_STEPS.md) identified three concrete improvements. All three are ported into agi-core.

**1. Semantic Deduplication (core/learner.py)**
- **Problem:** `cos(π/2 + x²)` and `sin(x²)` are algebraically identical but have different tree structures, wasting beam slots on duplicates.
- **Solution:** Hash each program by its rounded output vector on training inputs. Two programs producing identical outputs are the same function. Keep the lowest-energy one.
- **Location:** `core/learner.py` — domain-agnostic. Uses `env.execute()` to compute outputs, so it works for any domain.
- **Config:** `SearchConfig.semantic_dedup` (default True), `dedup_precision` (default 6 decimal places).
- **Trade-off:** Extra evaluations per generation (one hash computation per candidate). Accepted because dedup saves far more compute by eliminating redundant beam members.

**2. Pareto Front Tracking (core/learner.py)**
- **Problem:** Beam search returns only the single best program. But a user may want the best program *at each complexity level* — the accuracy-complexity tradeoff.
- **Solution:** Track best prediction_error per program size across all generations. Filter to the true Pareto front (no entry is dominated in both error and complexity). Return in `WakeResult.pareto_front`.
- **Location:** `core/learner.py` — domain-agnostic. `ParetoEntry` dataclass added.
- **Inspiration:** PySR's Pareto front output, which shows the "knee" where adding complexity stops helping.

**3. Constant Optimization via scipy (grammars/symbolic_math.py)**
- **Problem:** Constants evolve by Gaussian mutation, which is slow for deep compositions. The loss landscape over coefficients is non-convex with many local minima.
- **Solution:** After structural mutation, extract all `const` nodes, pack their values into a vector, and run `scipy.optimize.minimize` (Nelder-Mead) to fit them. This decouples structure search (evolutionary) from coefficient search (gradient-free local optimization).
- **Location:** `grammars/symbolic_math.py` — domain-specific. Only symbolic math has fittable constants.
- **Interface:** Added `Grammar.prepare_for_task(task)` hook to `core/interfaces.py` so the grammar can cache training data. Default is no-op; SymbolicMathGrammar uses it to feed (x, y) pairs to the optimizer.
- **Fallback:** If scipy is not installed, constant optimization is silently skipped (graceful degradation).
- **Trade-off:** scipy is now a dependency, but it's lightweight and commonly available. The optimizer runs with max 200 function evaluations (fast).

**Tests:** 19 new tests added (5 semantic dedup, 8 Pareto front, 7 constant optimization). Total: 205 tests, all passing.

---

## Session 3 — Claude Code Web (March 10, 2026)

### Decision: Exhaustive Enumeration Before Beam Search

**Context:** agi-mvp-general solves 24.3% on training by using exhaustive search up to depth-3, not evolution. Beam search with width=150 explores a tiny fraction of the space; most random programs produce garbage grids.

**Key insight:** Exhaustive enumeration for short programs IS beam search with beam_width = vocabulary_size^depth. It's the same algorithm, different budget.

**Decision:** Add `SearchConfig.exhaustive_depth` (default 2). Before beam search:
- Depth 1: try ALL single primitives (N programs)
- Depth 2: try ALL top-K pairs outer(inner(x)) (N×K programs)

**Result:** 12/400 (3%) → 33/400 (8.2%) on training. 32 of 33 solved by enumeration, 1 by beam search. Confirms enumeration IS the primary solver for ARC.

**Compounding insight:** Learned library entries are 0-arity primitives. A depth-1 program using a learned concept IS a depth-3+ program in disguise. As vocabulary grows via sleep/promotion, depth-1 enumeration covers what previously required depth-3+.

### Decision: 16 New ARC Primitives

**Context:** Gap between 48 primitives and agi-mvp-general's 304. But most of those 304 are parameterized color ops.

**Decision:** Add 16 high-value spatial/object primitives:
- Object isolation: extract_largest, extract_smallest (3 tasks solved)
- Symmetry: make_symmetric_h/v, anti_diagonal_mirror (6 tasks solved via symmetry+repeat combos)
- Pattern: repeat_right/down, add/remove_border
- Sorting: sort_rows/cols, unique_rows/cols (2 tasks solved)
- Color: recolor_by_rank, extend_lines_h/v

**Result:** 14 of 33 solved tasks use new primitives. Not dead weight.

### Decision: Task-Specific Color Primitives

**Context:** Fixed `keep_c1`..`keep_c9` are rarely the right color ops. Most ARC tasks involve task-specific color mappings.

**Decision:** `ARCGrammar.prepare_for_task()` analyzes training examples to generate dynamic color primitives (fill_bg_X, remove_X, swap_X_to_Y) based on which colors appear/disappear.

**Rationale:** This keeps the core algorithm generic (the Grammar interface already has `prepare_for_task`) while giving ARC-specific color intelligence. 3 tasks solved using task-specific color prims.

### Decision: Sequential Compounding Mode

**Context:** Tasks processed in parallel can't share knowledge within a round. Easy tasks should seed concepts for hard tasks.

**Decision:** `CurriculumConfig.sequential_compounding=True`. Process tasks one at a time; after each solve, immediately promote non-trivial subtrees to the library.

**Result:** In practice, with depth-1 and depth-2 programs, subtrees are too small (size < 2) to promote useful concepts. Compounding via sequential processing added 0 new solves beyond parallel. The bottleneck is that unsolved tasks need fundamentally different operations (object-level reasoning, conditional programs), not more compositions of existing primitives.

**Lessons:** Sequential compounding will become valuable when: (a) deeper programs are found (more subtrees to promote), or (b) object-level primitives enable compositions that weren't possible before.

### Decision: Culture Persistence (Cross-Run Knowledge Transfer)

**Context:** agi-mvp-general's culture.py saves/loads learned concepts across runs. Training produces culture; evaluation loads it.

**Decision:** `InMemoryStore.save_culture()` / `load_culture()` with proper JSON serialization of Program trees (not just repr strings). Solutions are also saved for culture transfer.

**Rationale:** Proper round-trip serialization is essential for the train→eval pipeline. Using `_program_to_dict` / `_program_from_dict` instead of repr/eval for safety and correctness.

### Decision: Train/Eval Pipeline

**Context:** agi-mvp-general gets 35/400 on the evaluation set. We need to measure on eval but never use eval data for development decisions.

**Decision:** `experiments/phase1_arc.py` supports `--pipeline` (train→eval), `--eval` (eval only with `--culture`), and proper data split detection. Training data for all development; evaluation data only for final scoring.

### Benchmark Results: After This Session

| Config | Training (400) | Time |
|--------|---------------|------|
| Baseline (beam only, 48 prims) | 12/400 (3.0%) | 26s |
| + exhaustive depth=2 + 64 prims | 33/400 (8.2%) | 155s |
| + sequential compounding (2 rounds) | 32/400 (8.0%) | 304s |

### Evaluation Set Results (Scoring Only — Not for Development)

Pipeline run: `python -m experiments.phase1_arc --pipeline --mode quick`

| Split | Solved | Rate | Time | Library |
|-------|--------|------|------|---------|
| Training (2 rounds) | 32/400 | 8.0% | 3m00s | 2 abstractions |
| Evaluation (2 rounds, with culture) | 3/400 | 0.75% | 3m50s | 3 abstractions |

**Comparison with agi-mvp-general:** 35/400 (8.8%) on evaluation set.

**Analysis:** The eval-to-train ratio (0.75% vs 8.0%) shows significant overfitting to training distribution. The evaluation set tasks require more complex transformations than our depth-2 exhaustive search can produce. Key bottleneck: our 64 primitives + depth-2 compositions express ~4,160 unique programs. Most eval tasks need object-level reasoning, conditional logic, or deeper compositions that cannot be expressed in 2 operations.

**Bug fix:** `NameError: name 'runs_dir' is not defined` in `core/runner.py` line 724. Fixed by deriving culture path from library_path instead of using undefined variable.

---

## Session 4 — Claude Code Web (March 10, 2026)

### Decision: 25 New Primitives — Object-Level, Grid Partitioning, Diagonal

**Context:** Session 3 solved 33/400 (8.2%) training with 64 primitives and depth-2 exhaustive search. Analysis of near-miss tasks showed the system lacked object-level reasoning, grid partitioning, and anomaly removal capabilities.

**Research methodology:** Studied agi-mvp-general's `objects.py` (connected components), `decompose.py` (grid partitioning), and `spatial/` (line extension). Analyzed 15 ARC tasks to identify missing operation categories. Examined 8 near-miss tasks (error < 0.03) to find targeted primitives.

**Decision:** Add 25 new primitives (89 total) in three batches:

**Batch 1: Connected components (9 primitives)**
- `keep_largest_only`, `keep_smallest_only` — isolate objects by size
- `remove_largest_obj`, `remove_smallest_obj` — remove objects by size
- `count_objects`, `recolor_each_obj` — object analysis
- `mirror_objects_h`, `mirror_objects_v` — per-object mirroring within bbox
- `flood_fill_bg` — fill enclosed background regions

**Batch 2: Grid partitioning & structural (7 primitives)**
- `extract_tl_cell`, `extract_br_cell`, `remove_grid_lines` — grid structure ops
- `shift_rows_right`, `shift_rows_left` — diagonal staircase patterns
- `extend_lines`, `extend_diagonals` — line/ray completion

**Batch 3: Color/pattern & anomaly removal (9 primitives)**
- `binarize`, `color_to_mc`, `upscale_pattern` — color transforms
- `denoise_majority`, `fill_rectangles` — noise removal
- `extract_minority_c`, `extract_majority_c` — color isolation
- `replace_noise_objs`, `hollow_objects` — object cleanup

**Result:** 39/400 (9.8%) training — 6 new tasks solved using new primitives.

### Decision: Depth-3 Exhaustive Enumeration with Smart Pruning

**Context:** Previous depth-3 used K³ evaluations (brute-force triple combinations), which was expensive and explored many redundant combinations.

**Decision:** New depth-3 approach: take top-K depth-2 programs as complete subtrees, wrap each with every unary outer. Cost: N×K evaluations instead of K³. Includes:
- Early exit: stop enumeration immediately when a perfect solve is found
- Semantic dedup: filter duplicate outputs between depth levels
- Default depth increased from 2 to 3 (affordable with N×K cost)

**Rationale:** An N×K depth-3 search evaluates ~1,780 additional programs per task (89 prims × 20 top-K). This is far cheaper than K³ = 8,000 and produces better results because the depth-2 subtrees are pre-filtered by quality.

### Benchmark Results: After This Session

| Config | Training (400) | Eval (400) | Time |
|--------|---------------|------------|------|
| Session 3 baseline (depth-2, 64 prims) | 33/400 (8.2%) | 3/400 (0.75%) | 5m |
| Session 4 (depth-3, 89 prims) | 39/400 (9.8%) | 4/400 (1.0%) | 6m train + 10m eval |

**New training tasks solved by new primitives (6 of 39):**
- `007bbfb7: upscale_pattern` — self-similar tiling
- `08ed6ac7: recolor_each_obj` — assign unique colors to objects
- `0b148d64: crop_nonzero(extract_minority_c)` — isolate rare color
- `a87f7484: crop_nonzero(extract_majority_c)` — isolate dominant color
- `e26a3af2: fill_rectangles(denoise_3x3)` — rectangle completion + denoising
- `623ea044: extend_diagonals` — diagonal ray tracing

**Eval tasks solved (4):**
- `5b6cbef5: upscale_pattern` — NEW (from new primitive)
- `60c09cac: scale_2x` — existing
- `e1baa8a4: unique_rows(unique_cols)` — NEW (from depth-3 composition)
- `fc754716: outline(replace_bg_mc)` — existing

**Comparison with agi-mvp-general:** 35/400 (8.8%) on evaluation set. Gap remains significant — agi-mvp-general uses 304 primitives, 13 specialized search phases, and object decomposition pipeline.

**Analysis of remaining gap:** The eval-to-train ratio improved slightly (1.0% vs 9.8%) compared to session 3 (0.75% vs 8.0%). The bottleneck remains: most unsolved tasks require multi-step conditional reasoning (if object has property X, apply transform Y) or complex object interactions that can't be expressed as simple primitive compositions. Next steps would be: (a) object decomposition pipeline (perceive→transform-per-object→reassemble), (b) input-adaptive primitives that analyze training examples to infer task-specific operations.

---

## Session 5 — Modular Restructuring + Scoring Improvement (March 10-11, 2026)

### Decision: Restructure `grammars/` → `domains/` package

**Rationale:** The monolithic `grammars/arc.py` (2240 lines) mixed primitives, environment, grammar, drive signal, and dataset loading in one file. This violated the principle that each domain's primitives, composition grammar, and interfaces should be cleanly separated.

**New layout:**
```
domains/arc/
  primitives.py   - All Grid→Grid transforms (101 primitives)
  objects.py      - Connected component detection
  environment.py  - ARCEnv (program execution)
  grammar.py      - ARCGrammar (composition, mutation, crossover)
  drive.py        - ARCDrive (structural similarity scoring)
  dataset.py      - Task loading + sample tasks
domains/symbolic_math/
  __init__.py     - Full symbolic regression domain
```

`grammars/` retained as backward-compatible shims. All 305 tests pass unchanged.

### Decision: Port structural similarity scorer from agi-mvp-general

**Problem:** Binary pixel-match scoring creates a flat fitness landscape — programs either match or don't. Beam search can't make incremental progress.

**Solution:** Weighted composite scorer:
- 0.60 × pixel_accuracy
- 0.15 × dimension_match
- 0.15 × color_overlap (Jaccard on non-bg palettes)
- 0.10 × nonzero_density_similarity

**Result:** Smoother landscape enables beam search evolution to find depth-3 programs.

### Decision: Add near-miss refinement (Phase 1.5)

**Problem:** Many programs are "almost right" (prediction_error < 0.20) but need one more step.

**Solution:** After exhaustive enumeration, try appending/prepending each primitive to the top-10 near-miss programs. Cost: O(10 × N_prims × 2) = ~2000 extra evals per task.

### Decision: Add 12 new primitives (batch 2)

Cyclic shifts (4), symmetry completion (2), split-by-separator (2), morphological (2), color cycling (2).

### Benchmark Results

| Metric | Session 4 | Session 5 | Change |
|--------|-----------|-----------|--------|
| Training (quick, 1 round) | 39/400 (9.75%) | 52/400 (13.0%) | **+33%** |
| Primitives | 89 | 101 | +12 |
| Tests | 285 | 305 | +20 |
| Depth-3 solves | 0 | 7 | **first ever** |

**13 new tasks solved, 0 regressions.** Notable new solves:
- `shift_down_1` — new cyclic shift primitive
- `complete_sym_h(recolor_4_to_0)` — new symmetry completion + color op
- `overlay_split_h` — new split-by-separator
- `make_sym_v(make_sym_h(tile_2x2))` — depth-3 symmetry tiling (first depth-3 solve!)
- `left_half(top_half(crop_nonzero))` — depth-3 spatial extraction

**Key insight:** The structural similarity scorer unlocked depth-3 solutions by giving beam search enough signal to navigate toward them incrementally.

---

## Session — Codebase Audit & Refactoring (March 11, 2026)

### Audit: Self-Review of Entire Codebase

Performed a full audit as if looking at the repository for the first time.

**Bugs fixed:**
1. **`_wake_on_task_no_record` was a 175-line copy-paste of `wake_on_task`** — refactored into shared `_wake_core(task, record=bool)` method. Any future change to wake logic now only needs to be made once.
2. **`_near_miss_refine` prepend was a no-op** — `node.root = old_root` wrote the same value back. Fixed to correctly wrap deepest leaf: `leaf → prim(leaf)`.
3. **`_evaluate_program` wastefully called `drive.energy(program, None, None)`** then discarded the result and recomputed everything. Removed the dead call.

**Architecture cleanup:**
4. **Removed `grammars/` backward-compat shim files** — tests migrated to import directly from `domains/`. Per CLAUDE.md: "Avoid backwards-compatibility hacks."
5. **Added test accuracy (generalization) tracking** — `WakeResult` now includes `test_error` and `test_solved` computed on held-out test examples. Runner displays train vs test accuracy in final results and compounding table.

**Documentation fixes:**
6. README test count updated (205 → 323), structure diagram updated to reflect `domains/` directory, demo commands fixed.

**Test coverage:** 64% → 70% overall. `learner.py` 66% → 79%. Added 18 new tests covering test accuracy, near-miss refinement, runner helpers, and edge cases.

**Decision: Why `_wake_core(record=bool)` over other patterns.**
Alternatives considered: (a) decorator pattern, (b) inheritance. Chose simple boolean parameter because the recording behavior is a single if-check at 3 callsites. A decorator or inheritance would add complexity for no benefit.

---

## Session — Porting agi-mvp-general Solver (March 11, 2026)

### Decision: Port exhaustive enumeration strategy from agi-mvp-general

**Problem:** agi-core's exhaustive search used top-20 inner prims with N×K enumeration (all outers × top-K inners). This missed solutions where the first step scored low individually but was structurally critical (e.g. crop, fill, compress).

**Solution:** Adapted agi-mvp-general's proven approach:
- **Pair search:** top-40 singles + 30 essential structural concepts → K² combos (both steps from same pool)
- **Triple search:** top-15 + essential concepts → K³ exhaustive (guaranteed to find all 3-step solutions in pool)
- **Grammar.essential_pair_concepts():** domain-agnostic interface for structural prims
- **Adaptive beam search:** reduce generations when enumeration best error > 0.3 (beam rarely recovers)

**Primitive porting:** 101 → 222 → 260 primitives across two batches:
- Batch 1: fill, pattern, grid arithmetic, symmetry, color, propagation, object-level (121 new)
- Batch 2: connectivity, gravity, line extension, color reordering, factory-generated variants (38 new)

### Benchmark Results (50-task quick test)

| Version | Primitives | Train Solved | Test Solved |
|---------|-----------|-------------|-------------|
| Session 5 baseline | 101 | 52/400 (13.0%) | — |
| + 121 primitives | 222 | 9/50 (18.0%) | 8/50 (16.0%) |
| + enumeration + 38 prims | 260 | **12/50 (24.0%)** | **10/50 (20.0%)** |

**Key observation:** 3 new solves from wider enumeration + new primitives. The improvement from 18% → 24% validates that both wider search AND more primitives contribute. The remaining 76% unsolved tasks likely need conditionals, object decomposition, or DSL synthesis.

### Decision: Default rounds to 1

Wake-sleep rounds haven't shown accuracy improvements in practice. The library extraction phase adds abstractions but they don't measurably help subsequent rounds. Defaulted all presets to rounds=1.

**Rationale:** Until the sleep phase's extraction quality improves (better subtree scoring, cross-task transfer), multiple rounds just waste compute. The flag is preserved for experimentation.

---

## Session 7 — Color Fix, Conditional Branching, Object Decomposition (2026-03-11)

### Feature: Post-hoc color fixing (Phase 1.75)

Many ARC near-misses differ from the target by a consistent color substitution (e.g., all 3s should be 5s). The color fix phase:

1. Collects near-miss programs (prediction_error < 0.30)
2. Executes each on all training inputs, compares pixel-by-pixel
3. Builds a consistent (got→want) color remap with 80% consistency threshold
4. Wraps the original program with a color_remap primitive

**Architecture:** `Environment.infer_output_correction()` interface. ARCEnv overrides with pixel-level color remap detection. Domain-agnostic — other domains could implement their own correction inference.

### Feature: Conditional branching (Phase 1.25)

Implements if-then-else programs: `if pred(input) then A(input) else B(input)`.

- 17 predicates ported from agi-mvp-general: symmetric_h/v, square, tall, wide, single_color, many_colors, small, large, bg_majority, mostly_empty, frame, diag_sym, odd_dims, two_colors, h_stripe, v_stripe
- `Grammar.get_predicates()` interface for domain-agnostic predicate access
- Search strategy: partition training inputs by predicate, score top-K per group, try best 5×5 combos per non-trivial predicate
- Cost: O(P' × top_k × N_examples + P' × 25) where P' = non-trivial predicates

### Feature: Object decomposition (Phase 1.1)

Per-object transform pipeline: perceive → transform-per-object → reassemble.

- Connected component extraction via 4-connectivity flood fill
- `apply_transform_per_object()`: applies same primitive to each object's subgrid
- 7 conditional recolor strategies: by_size, by_singleton, by_input_color, by_shape, by_size_rank, by_compactness, by_has_hole
- `Environment.try_object_decomposition()` interface

### Current solver pipeline phases

1. **Phase 1**: Exhaustive enumeration (depth 1/2/3)
2. **Phase 1.1**: Object decomposition (per-object transforms + conditional recolor)
3. **Phase 1.25**: Conditional search (if-then-else with predicates)
4. **Phase 1.5**: Near-miss refinement (append/prepend primitives)
5. **Phase 1.75**: Color fix (learn color remap from mismatches)
6. **Phase 2**: Beam search (adaptive generations, seeded with Phase 1 results)

### Benchmark Results (50-task quick test)

| Version | Primitives | Train Solved | Test Solved |
|---------|-----------|-------------|-------------|
| Session 5 baseline | 101 | 52/400 (13.0%) | — |
| + 121 primitives | 222 | 9/50 (18.0%) | 8/50 (16.0%) |
| + enumeration + 38 prims | 260 | 12/50 (24.0%) | 10/50 (20.0%) |
| + color fix + conditionals + obj decomp | 260 | **13/50 (26.0%)** | **11/50 (22.0%)** |

**New solve:** Task 0d3d703e solved by `per_object_recolor(by_input_color)` — object decomposition feature correctly learned an input_color→output_color mapping and applied it per-object. No regressions.

### Test coverage

- 365 total tests, all passing
- 8 color fix tests, 18 conditional/predicate tests, 16 object decomposition tests

---

## Session 8 — Performance Optimization (March 11, 2026)

### Performance profiling results

Profiled per-phase timing on 5 representative tasks:

| Phase | Worst Case (before) | Worst Case (after) | Speedup |
|-------|--------------------|--------------------|---------|
| Phase 1 (enum) | 4.24s | 4.24s (unchanged) | 1x |
| Phase 1.1 (obj decomp) | 0.03s | 0.03s | 1x |
| Phase 1.25 (conditional) | 0.11s | 0.11s | 1x |
| Phase 1.5 (near-miss) | **11.25s** | **1.35s** | **8.3x** |
| Phase 2 (beam) | varies | varies | 1x |
| Phase 3 (post-beam) | ~11s | ~0.1s | **~100x** |

**Root cause:** Near-miss refinement tried 10 near-misses × 280 unary primitives × 2 directions = 5,600 evaluations per task.

### Optimizations applied

1. **Near-miss refinement**: Top-5 near-misses × top-50 primitives (by depth-1 score) + essential pair concepts. Reduces from 5,600 to ~550 evaluations.
2. **Phase 3**: Removed redundant second near-miss pass on beam results. Phase 1.5 already covers enum near-misses. Only color fix runs post-beam.
3. **Per-task speedup**: 28.7s → 6.2s for unsolved tasks (4.6x end-to-end).

### Bug fix: test > train accuracy

`test_solved` was evaluated for ALL tasks including unsolved ones. A program failing on training (bad average across 3 examples) could pass test (only 1 example). Fixed: only evaluate test when training is solved.

### Task ordering: shuffle instead of sort

Changed default from sorted-by-difficulty to seeded shuffle for parallel benchmarks. Sorting creates biased progress (easy tasks solve first, giving inflated early metrics). Shuffle gives honest progress estimates throughout the run. Sorting retained for sequential compounding mode where easy→hard ordering helps library build up.

### NumPy/Numba analysis

agi-mvp-general uses targeted numpy (scoring) and numba @njit (flood fill). Our scoring already uses numpy. Numba-ing flood fill would help for large grids but profiling showed object decomposition is only 0.03s — not the bottleneck. The real bottleneck was near-miss refinement (now fixed).

### Benchmark (in progress)

Running 400-task quick benchmark with all optimizations, shuffled order, 4 workers.

## Session 9 — Performance Fixes & Compute Budget (March 2026)

### Triple pool bloat fix — root cause of slowness

**Context:** Quick mode became very slow after porting 281 primitives + 29 essential pair concepts.
**Root cause:** `_exhaustive_enumerate` depth-3 triple pool was built as `top_k (15) + ALL essential concepts (29)`, giving pool sizes of 30-44 entries. Cost: K³ = 27,000-85,000 evals per task just for triple enumeration!
**Fix:** Cap triple pool at `triple_top_k` total entries. Essentials compete for slots instead of being added on top. Same fix applied to pair pool.
**Result:** Triple cost drops from ~27K-85K to ~3,375 evals (15³). 50-task benchmark: median 3.84s/task, down from 15-30s/task.

### Cell-normalized per-task compute budget

**Context:** `--compute-cap` flag was reducing max_generations globally, treating all tasks equally regardless of grid size. agi-mvp-general uses cell-normalized budgets.
**Decision:** Adopt agi-mvp-general's formula: `min(max(cap/cells, 500), cap/DEFAULT_CELLS)` where DEFAULT_CELLS=800 (median ARC grid size). Small grids get more evals (cheap), large grids get fewer. Budget enforced per-task via `_budget_ok()` gating on expensive phases (near-miss, beam search).
**Result:** `eval_budget` field added to `SearchConfig`, phases gated with `_budget_ok()`.

### Ctrl-C worker cleanup

**Context:** User reported ^C doesn't kill the job completely — CPU stays high.
**Root cause:** `ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)` only cancels pending futures; running workers continue as orphan processes.
**Fix:** On KeyboardInterrupt, explicitly `os.kill(pid, SIGTERM)` all worker processes before calling `shutdown()`.

### Semantic dedup was broken for grids

**Context:** `_semantic_hash` used `round(float(val), precision)` on grid outputs (list of lists), which throws TypeError. Every program hashed to `str([None, None, None])`, so dedup kept only ONE program per generation — beam_width was effectively 1.
**Fix:** Handle grid outputs via tuple conversion: `tuple(tuple(row) for row in val)`. Numeric outputs still use float rounding.
**Impact:** Beam search can now maintain actual diversity. This should improve solve rate on tasks where beam search matters (harder tasks that enumeration doesn't catch).

### Benchmark results after fixes

Quick preset, 281 primitives, 2 workers, shuffled order:
- **84/400 = 21.0% train accuracy** (up from 13% with 101 prims in Session 7)
- Median task time: 2.3s (~7x faster than before pool cap fix)
- Total wall time: ~19 minutes (1,140s sum of task times)
- Total evaluations: 2,290,000+ across all tasks

Comparison across sessions:
| Session | Primitives | Preset  | Train acc | Median/task | Notes |
|---------|-----------|---------|-----------|-------------|-------|
| 7       | 101       | quick   | 52/400 = 13.0% | 2.30s | Baseline |
| 8       | 281       | default | 18/400 = 4.5%  | 15-30s | Regression (bloated pool) |
| 9a      | 281       | quick   | 84/400 = 21.0% | 2.3s  | Pool fix, broken dedup |
| 9b      | 281       | quick   | 86/400 = 21.5% | 2.8s  | Pool fix + dedup fix + reduced beam |

The 281 primitives now help (21.5% vs 13%) instead of hurting (4.5%). Semantic dedup fix adds 2 more solves with ~0.5s/task overhead. Presets reduced (beam 80→30, gens 40→15) to compensate for proper beam diversity.

**Key insight:** Beam search contributes minimally to solve rate (~2 tasks out of 86). The exhaustive enumeration (depth 1-3) does the heavy lifting. This suggests future work should focus on better enumeration (richer primitives, smarter pool selection) rather than deeper beam search.

## Session 10 — Batch 4 Primitives: Grid Partition, Annotation, Scaling

### Analysis of Unsolved Tasks

Systematic analysis of 314 unsolved tasks from session 9 revealed:

| Pattern | Count | Description |
|---------|-------|-------------|
| Object annotation | 96 | Modify pixels around/between objects |
| Grid-partitioned | 51 | Input split by separator lines into regions |
| Same-size small diff | 99 | Few cells changed (filling, recoloring) |
| Subgrid selection | 26 | Extract one subgrid from structured input |
| Scaling | 27 | Up/downscale by various factors |
| Recoloring only | 29 | Same positions, different colors |

Most unsolved tasks (210/314) have same-size input/output. The dominant change type is filling background cells (138 tasks).

### New Primitives Added (302 total, up from 281)

**Grid partition (7):** `select_odd_cell`, `overlay_cells`, `majority_cells`, `xor_cells`, `most_colorful_cell`, `most_filled_cell`, `least_filled_cell`. Also improved separator detection to handle zero-valued grid lines (many ARC tasks use bg=0 as separator).

**Pixel annotation (5):** `surround_3x3`, `draw_cross`, `draw_cross_contact`, `draw_diag`, `fill_convex_hull`.

**Line connection (2):** `connect_h`, `connect_v`.

**Scaling (7):** `scale_4x`, `scale_5x`, `downscale_4x`, `downscale_5x`, `downscale_7x`, `downscale_maj_2x`, `downscale_maj_3x`.

**Other (1):** `recolor_objects_by_neighbor_count`.

### Benchmark Results

| Session | Primitives | Train acc | Eval acc | Median/task | Notes |
|---------|-----------|-----------|----------|-------------|-------|
| 9b | 281 | 86/400 = 21.5% | 7/122 = 5.7% | 2.8s | Baseline |
| 10 | 302 | 93/400 = 23.2% | 33/400 = 8.2% | 6.4s | +21 new prims |

Net +7 train tasks (+8 new, -1 regression). The 8 newly solved tasks:
- `select_odd_cell`: directly solved 2 partition tasks
- `downscale_7x`: solved 1 task
- `connect_h(connect_v)`: composition solved 1 task
- `binarize(surround_3x3)`: composition solved 1 task
- `downscale_4x(keep_smallest_only)`, `crop_nonzero(select_odd_cell(left_half))`: deeper compositions solved 2 tasks

**Speed tradeoff:** Median time doubled (2.8s → 6.4s) due to 302 primitives in exhaustive search. The depth-2 search space grew from 281² ≈ 79K to 302² ≈ 91K programs per task.

**Eval improvement:** From 5.7% to 8.2% on the evaluation set (400 tasks with culture transfer).

---

## Decision: Quick Preset — 50 Tasks Instead of All 400 (2026-03-11)

**Problem:** Quick mode ran all 400 tasks with smaller beam/gens, taking ~32 minutes. This defeats the purpose of a "quick" iteration mode.

**Change:** Set `max_tasks: 50` in the quick preset (was `0` = all tasks).

**Rationale:**
- 50 tasks × ~3s/task ÷ 4 workers ≈ ~40 seconds per phase. Full pipeline (train + eval) completes in ~2 minutes.
- Tasks are shuffled with a deterministic seed (42), so any subset is a representative random sample.
- Extrapolation works: if 12/50 (24%) solve in quick mode, expect ~96/400 (24%) on the full dataset.
- Users who want quick search settings on all 400 tasks can use `--mode quick --max-tasks 0`.

**Trade-off:** Quick mode no longer produces a full 400-task result. But the purpose of quick mode is fast iteration, not final benchmarking — that's what default/contest modes are for.

---

*This document will be updated with each new session and major decision.*
