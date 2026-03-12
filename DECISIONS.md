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

### Decision 30: Fix pool selection to guarantee essential concepts (Session 5)

**Date:** 2026-03-11
**Context:** Investigating why near-miss tasks weren't being solved despite having relevant essential concepts like `fill_enclosed`, `crop_to_nonzero`, `complete_diag` in the grammar.

**Problem:** The pair/triple pool building in `_exhaustive_enumerate()` filled all slots with top-scoring singles first, then added essentials only if room remained. With 324 primitives, top-scoring singles always filled all 40 pair slots and 15 triple slots, leaving **zero room for essential concepts**. Essentials — structural building blocks that score poorly alone but are critical in compositions — were never explored in depth-2 or depth-3 programs.

**Fix:** Changed pool building to add essential concepts first (up to half the pool), then fill remaining slots with top-scoring singles. This guarantees essentials are always explored while keeping the total pool size (and compute cost) unchanged.

**Result:** Quick mode went from 11/50 (22%) to 13/50 (26%). Task 23581191 (previously a near-miss at err=0.065) now solved by `dom_touch_accent_2(draw_cross)` — a composition only possible because `draw_cross` was now included in the pair pool as an essential concept.

---

### Decision 31: Add mark_intersections_exclude_axis primitive (Session 5)

**Date:** 2026-03-11
**Context:** Near-miss analysis showed task 2281f1f4 was 1 pixel off with `mark_intersections_2`. The error was always at the crossing point of the two perpendicular marker axes.

**Solution:** Created `mark_inters_excl_axis` that identifies header rows and side columns separately, fills their cross-product intersections, but excludes the cell where the axes themselves cross.

**Result:** 0 errors on all 3 training examples AND the test example. Confirmed +1 solve.

---

### Decision 32: Wire transition matrix into beam search mutations (Session 6)

**Date:** 2026-03-12
**Context:** The DreamCoder-style transition matrix was already built and observed from solutions, but beam search mutations used uniform random primitive selection. Free performance was being left on the table.

**Solution:** Added optional `transition_matrix` parameter to `Grammar.mutate()`. When provided, all three mutation types (point, grow, shrink) use `TransitionMatrix.weighted_choice()` to bias primitive selection toward known-good compositions. The Learner passes the transition matrix during beam search when it has observed data.

**Result:** Backward-compatible (default None). All domains updated. 420 tests pass.

---

### Decision 33: Improve sleep phase compounding with diversity bonus and pruning (Session 6)

**Date:** 2026-03-12
**Context:** Sleep phase was extracting only 1-2 library entries per 400 tasks, and solve rate didn't improve across rounds. Root causes: (1) scoring didn't reward structural diversity, (2) dead entries accumulated and crowded out better abstractions.

**Solution:**
- **Diversity bonus**: Subtrees appearing across solutions with different root operations score higher. Formula: `usefulness = tasks_used × log(size+1) × (1 + 0.5 × log(unique_roots))`. This rewards general-purpose compositions over task-specific ones.
- **Library pruning**: After decay, entries with usefulness < 0.01 AND reuse_count == 0 are removed. Added `Memory.prune_library()` method. This prevents the library from filling with stale abstractions.

**Result:** Library now self-cleans. General abstractions preferred over narrow ones. 420 tests pass.

---

### Decision 34: Add Zork text adventure domain (Session 6)

**Date:** 2026-03-12
**Context:** The architecture claimed domain-agnosticism but only had 2 domains (ARC grids, symbolic math). Both are stateless input→output transforms. Need to prove the core loop handles sequential, stateful, goal-directed domains.

**Solution:** New `domains/zork/` with:
- **Game engine**: Room graph with items, locked doors, inventory, flags
- **30 primitives**: 4 movement + 8 items × 3 verbs + wait + look
- **16 predicates**: has_item, room_has_item for conditional branching
- **Drive signal**: Weighted composite (40% room match, 30% inventory Jaccard, 15% score, 15% flags)
- **4 sample tasks**: navigation, take+move, locked door puzzle, simple traverse
- **36 tests**: Engine, primitives, environment, grammar, drive, integration

**Key insight:** Programs compose as sequential actions: `go_north(take_lamp(state))`. This is the same tree structure as ARC programs, proving the Program representation handles both stateless transforms and stateful action sequences.

**Result:** Core learner runs on Zork tasks without modification. 420 tests pass (380 → 420).

---

### Decision 35: List Operations Domain + Compounding Validation

**Date:** 2026-03-12

**Context:** Needed to validate the core compounding hypothesis: does library learning (sleep) actually help solve harder tasks (wake)? ARC is too complex to isolate the compounding mechanism. Need a simpler domain where the expected behavior is clear.

**Architecture:** New domain `domains/list_ops/` with 22 primitives (reverse, sort, double_all, filter_pos, cumsum, etc.), 28 tasks at 3 difficulty levels:
- Level 1 (8 tasks): single operations
- Level 2 (12 tasks): two-step compositions
- Level 3 (8 tasks): three-step compositions

Experiment script `experiments/list_compounding.py` runs multiple wake-sleep rounds with sequential compounding enabled.

**Result:** Domain works correctly, 51 tests pass. Experiment runs in <1 second.

---

### Decision 36: Fix Critical Library Execution Bug (All Domains)

**Date:** 2026-03-12

**Context:** Compounding experiment showed flat solve rate across rounds (75%→75%→78%). Library was growing but NOT helping solve new tasks. Diagnosed the root cause:

**The bug:** `inject_library()` creates 0-arity Primitives with `fn=Program` (a stored sub-tree). But ALL four environments silently ignored these:
- **ARCEnv**: `if prim.arity == 0: return grid` (line 122) — returns input unchanged
- **ListEnv**: No lookup for dynamic primitives with Program fn
- **SymbolicMathEnv**: Unknown primitives return 0.0
- **ZorkEnv**: Same pattern

This meant **library entries were NEVER executed in any domain**. The entire compounding mechanism was broken since the beginning.

**Fix:** When a primitive's fn is a `Program` (library entry), recursively execute it:
```python
if isinstance(prim.fn, Program):
    return self.execute(prim.fn, input_data)
```
Applied to all 4 environments. Also register library primitives with the environment in the learner so they can be resolved during execution.

**Before fix:** List domain: R1=75%, R5=78% (flat)
**After fix:** List domain: R1=89%, R5=96.4% (compounding!)

Key evidence of compounding:
- `list_L3_increment_all_then_double_all_then_sort_asc`: R1 needed 1505 evals (beam search), R2 solved in 32 evals (depth-1 via library entry) — **47x speedup**
- 7/8 L3 tasks solved by R5 vs 3/8 before fix

471 tests pass.

---

### Decision 38: Disable Beam Search in Quick/Default Presets (A/B Tested)

**Date:** 2026-03-12

**Context:** Beam search parameters in presets had never been scientifically validated. The DECISIONS.md noted "beam search contributes ~2 tasks out of 86" but this was an observation, not a controlled experiment.

**Experiment:** Ran A/B test on 49 training tasks (seed 42, quick preset):
- **Test A:** Exhaustive-only (beam_width=1, max_generations=1)
- **Test B:** Current quick preset (beam_width=20, max_generations=10)

**Results:**
| Config | Solved | Overfit | Wall Time | Beam Overhead |
|--------|--------|---------|-----------|---------------|
| No beam | **17/49** | 2 | 157.3s | — |
| Beam=20 | **17/49** | 2 | 178.3s | +21.0s (+13%) |

- **Exact same 17 tasks solved** in both runs (set intersection = 17)
- **Zero additional solves from beam search**
- Beam adds 0.65s overhead per unsolved task (pure waste)

**Decision:** Set beam_width=1, max_generations=1 in quick and default presets. Contest keeps beam=30, gens=15 as a safety net for harder tasks. Updated README presets table, options table, and expected performance to match.

**New presets:**
| Mode | Beam | Compute Cap |
|------|------|-------------|
| quick | off (1×1) | 5M |
| default | off (1×1) | 20M |
| contest | 30×15 | 50M |

---

## Session — Cell-Normalized Compute Cap (2026-03-12)

### Decision: Aggressive compute cap at 2M ops (~3x median)

**Problem:** Task `0dfd9992` (21×21=441 cells) consumed 69s and 6,580 evals — pure waste on an unsolved task. Large-grid tasks dominated wall time while contributing zero solves.

**Key insight — bimodal solve distribution (400-task training set):**
- **72 "fast" solves** (depth 0-1, <1K evals): direct primitives or simple pairs
- **23 "slow" solves** (depth 1+, >1K evals): `per_object_recolor`, `per_object`, or depth-3 triples
- **305 unsolved**: exhausted full search (5K-7.4K evals each), never found it
- Grid size is NOT the bottleneck: 30×30 task `1f85a75f` solves in 30 evals (0.0s) with `extract_largest`; 29×29 task `484b58aa` burns 146s unsolved
- Solved tasks: median 22K ops, max 3.25M ops
- Overall median ops: 714K

**Philosophy:** If a task needs >3x median ops to solve, the primitives aren't good enough. Brute-forcing deeper search is the wrong investment — better to add the right primitive.

**Implementation:** `compute_cap=2_000_000` for quick/default presets (~3x median ops).
- Loses 2 solves (both depth-3 compositions on 441-cell grids: `90c28cc7`, `0b148d64`)
- Caps 49 pathological tasks, saves ~17% of total compute ops
- Verified on 50-task quick run: still 17/50 solved (no regression)
- Contest preset remains at 50M for maximum effort
- Override with `--compute-cap 0` for unlimited

### Decision: Add --task-ids flag for targeted runs

**Rationale:** Debugging specific tasks required running the full dataset. Added `--task-ids` to `make_parser()` (all experiments inherit it) with prefix-match support (e.g., `--task-ids 0dfd` matches `0dfd9992`). Filtering happens in `run_experiment()` so it's domain-agnostic.

---

## Session — JIT Compilation, Compute Budget, Smart Search (March 2026)

### Decision: Numba JIT compilation for ARC primitives

**Problem:** Primitive cost variance was 1000x+ (0.03ms to 37ms/call). Dense grids caused O(n×p²) blowup in drawing/connecting primitives. Task `1190e5a7` took 600s; `0dfd9992` took 35s.

**Solution:** JIT-compile 18 hot primitives with `@nb.njit(cache=True)`. For dict-based color operations, replaced with fixed `int[10]` arrays (ARC has 10 colors). For BFS, replaced deque with pre-allocated numpy arrays.

**Results:**
| Task | Before | After | Speedup |
|---|---|---|---|
| 1190e5a7 | 600s | 2.4s | **250x** |
| 0dfd9992 | 35s | 3.1s | **11x** |
| 400-task full | ~40min+ | 7m32s | **~5x** |
| Median task | ~5-6s | 2.3s | **~2.5x** |

No solves lost: 85/400 (21.2%) before and after.

### Decision: Depth-weighted compute cost proxy

**Problem:** `evals × cells` treats all programs equally, but depth-3 programs apply 3 primitives while depth-1 applies 1. Budget was not a true proxy for compute.

**Solution:** Count depth-weighted ops in exhaustive enumeration: depth-1 = 1 op, depth-2 = 2 ops, depth-3 = 3 ops. Budget is now in "ops" not raw eval count. This makes budget enforcement proportional to actual work.

**Adjusted presets:** quick/default: 2M → 3M ops, contest: 50M → 100M ops (accounts for ~2.6x depth multiplier).

### Decision: Compute cap = 3M ops (ROI-optimized)

**ROI sweep on 50-task training set:**
| Cap | Solved | Time | ROI |
|---|---|---|---|
| 1M | 18/50 | 30s | Best efficiency (1.66s/solve) |
| **3M** | **19/50** | **80s** | **Best absolute solves** |
| 5M+ | 19/50 | 91-107s | Zero additional solves |

**Judgement:** Beyond 3M, exhaustive search hits hard diminishing returns. The path to more solves is smarter search, not more compute.

### Decision: Smart search pruning (inner-step filter + adaptive depth skip)

**Problem:** Most depth-2/3 combinations are wasteful. Exhaustive K² pairs enumerate many useless inner steps.

**Two pruning strategies (deterministic, no solve loss):**
1. **Inner-step quality filter**: Only use depth-1 primitives with error < 0.70 as inner steps. Programs that produce garbage alone rarely improve as intermediate steps.
2. **Adaptive depth-3 skip**: If best depth-2 error > 0.50, skip depth-3 entirely.

**Impact:** 13% fewer ops, slowest task 10.8s → 6.1s, median 2.3s → 1.6s, same 19/50 solves.

### Analysis: Path to higher solve rates

Near-miss analysis of unsolved tasks (50-task set):
- 3 tasks with <5% error (almost solved — need color fix or small adjustment)
- 15 tasks with 5-15% error (found partial structure)
- 9 tasks with 15-30% error
- 4 tasks with >30% error

**Next steps for increasing solves:**
1. **Per-example error vectors**: Track which examples each program solves. Compose programs that solve complementary examples.
2. **Wider near-miss refinement**: Current refinement tries ±1 step on top-5 near misses. Could try deeper refinement chains.
3. **More primitives**: The 3 almost-solved tasks likely need a specific primitive we don't have.
4. **Cross-task transfer**: Library learning across tasks (the compounding loop).

---

## Decision 46: Compute Cap Sweet Spot Experiments — 2026-03-12

### Context
Ran systematic experiments to find the lowest compute cap that preserves solve quality, testing caps from 100 to unlimited on 400 training tasks.

### Experiment Results (400 training tasks, 2 workers)

| Cap | Solved | Rate | Wall Time | Efficiency |
|-----|--------|------|-----------|------------|
| 100 | 76/400 | 19.0% | 72s | 1.06 solves/s |
| 500K | 77/400 | 19.2% | 86s | 0.90 solves/s |
| 1M | 78/400 | 19.5% | 152s | 0.51 solves/s |
| 2M | 79/400 | 19.8% | 277s | 0.29 solves/s |
| 2.5M | 80/400 | 20.0% | 315s | 0.25 solves/s |
| 2.8M | 85/400 | 21.2% | 362s | 0.23 solves/s |
| 3M | 85/400 | 21.2% | 379s | 0.22 solves/s |
| unlimited | 85/400 | 21.2% | ~379s | 0.22 solves/s |

### Key Findings
1. **Bimodal distribution**: 76 "fast" tasks solve in <500 evals (any cap works). 9 "slow" tasks (mostly per_object_recolor) need ~13K evals.
2. **500-eval floor dominates**: The `max(..., 500)` floor in cell-normalization means caps from 100 to ~400K all give identical results.
3. **Sharp threshold at 2.8M**: All 9 slow tasks appear between 2.5M and 2.8M — no gradual progression.
4. **Cap=100 captures 89% of solves** (76/85) in 19% of the time.

### Decision
- **Quick preset**: Changed from 3M to 500K. Same solve count as cap=100 but with headroom for future primitives. ~5x faster than 3M.
- **Default preset**: Keep 3M. Captures all 85 solves including per_object_recolor.
- **Contest preset**: Keep 100M. Safety net for beam search.

### User's M1 Max Benchmarks (for reference)
- Quick mode (50 tasks): ~25s, 17/50 train, 2/50 eval
- Quick mode (400 tasks): 3m09s, 86/400 train, 20/400 eval
- Default (400 tasks, cap=500M): 4m30s, 87/400 train, 22/400 eval

---

## Decision 47: Pipeline Summary & Combined Output Files — 2026-03-12

### Context
Users had to scroll up through train+eval output to find key results. No single file captured the full pipeline run.

### Changes
1. **Pipeline summary**: At the end of a pipeline run, print a comprehensive summary with all parameters, train results, eval results, and total wall time.
2. **Combined output files**: Save `phase1_pipeline.json` (parameters + train/eval summaries + all task records + library) and `phase1_pipeline.jsonl` (all task records with phase tags).
3. **ExperimentResult dataclass**: `run_experiment()` now returns an `ExperimentResult` with culture_path, results_path, jsonl_path, and results_data dict.

### Rationale
- The summary eliminates scrolling — all key information visible at the end.
- The combined JSON/JSONL files enable single-file analysis of full pipeline runs.
- The richer return type enables pipeline mode to access results data without re-reading files.

---

## Decision 48: Multi-Domain Baselines & ARC-AGI-2 Experiment — 2026-03-12

### Context
Need baseline benchmarks for new domains (ARC-AGI-2, Zork) to track progress and validate the "one algorithm" claim across domains.

### Results

| Domain | Tasks | Solved | Rate |
|--------|-------|--------|------|
| ARC-AGI-1 Train | 400 | 85 | 21.2% |
| ARC-AGI-1 Eval | 400 | ~20 | ~5% |
| ARC-AGI-2 Train (100/1000) | 100 | 10 | 10.0% |
| ARC-AGI-2 Eval (cold) | 120 | 0 | 0.0% |
| Zork | 4 | 2 | 50% |

### Changes
1. **experiments/phase2_arc.py**: ARC-AGI-2 experiment script with pipeline mode, auto-detection of AGI-2 data, fallback to AGI-1 training data.
2. **experiments/zork_baseline.py**: Zork baseline experiment.
3. **README.md**: Added ARC-AGI-2 clone instructions, updated experiment commands.

### Key Insight: Why Compounding Fails on ARC
78/80 ARC solves are depth-1 (single primitive). Library entries compress depth-2+ compositions that exhaustive depth-3 search already covers. Compounding works on list_ops because exhaustive_depth=2 forces reliance on library for depth-3+. The solution space is wide-and-shallow (342 primitives, depth 1-2), not narrow-and-deep.

---

## Decision 49: Honest README & Compounding Tests — 2026-03-12

### Context
External review revealed that the README overstated claims, had stale numbers, and the test suite had zero tests verifying the core compounding hypothesis.

### README Fixes
1. **Removed "NumPy is the only dependency"** — actually requires numpy, scipy, numba, pytest
2. **Fixed test count** — was "420", now dynamically accurate (482)
3. **Fixed quick mode compute cap** — was "8M", actually 500K
4. **Added honest "Current status" section**: explicitly states compounding works on list_ops but NOT on ARC, acknowledges the 4:1 train-eval gap, and notes that ARC performance depends primarily on 342 hand-crafted primitives
5. **Updated roadmap** — marked completed phases, reframed Phase 4 as "make compounding work on ARC"
6. **Added multi-domain results table** — ARC-AGI-2, Zork, list_ops baselines alongside ARC-AGI-1

### New Tests (test_compounding.py — 9 tests)
1. **Library reuse**: sequential compounding grows library, immediate_promote adds entries
2. **Multi-round compounding**: solve rate improves across rounds on list_ops, library doesn't shrink
3. **Cross-domain**: same algorithm runs on list_ops and Zork, core/ verified to have zero domain imports
4. **Generalization**: train_solved implies test_solved on list_ops (no overfitting)

### Rationale
Credibility requires honesty about what works and what doesn't. The framework's architecture is genuinely clean and generic — but the compounding claim is only demonstrated on a synthetic domain. The README now says this explicitly. Tests now verify the core hypothesis rather than just checking code doesn't crash.

---

## Session 8 — Claude Code Web (March 12, 2026)

### Decision 50: Custom Zork over Jericho

**Question:** Should we use Jericho (Python wrapper for real Infocom Z-machine games) instead of our custom Zork domain?

**Decision:** Keep custom Zork for now, plan Jericho as a future "hard mode" domain.

**Rationale:**
- Jericho is a heavyweight dependency (compiled C library + ROM files with licensing issues)
- Custom domain gives full control over task design for testing specific compounding depths
- Need to prove compounding on simple domain before scaling to real Zork
- Can add Jericho later as `domains/zork_jericho/` without touching core

### Decision 51: Distance-Based Room Matching in ZorkDrive

**Problem:** Zork drive signal used binary room matching (correct=0, wrong=0.40). Programs getting 2/3 of the way to the goal room scored identically to programs 1 step away. Depth-3 exhaustive search couldn't distinguish promising depth-2 partial solutions from useless ones.

**Fix:** BFS graph distance with partial credit: `room_match = 1/(1+dist)`. Distance=1→0.5, distance=2→0.33, etc.

**Impact:** Zork solve rate 7/20 (35%) → 10/20 (50%). Three new depth-3 solves including `go_north(go_north(go_north))` and `go_west(take_sword(go_east))`.

### Decision 52: Fix Library Primitive Execution in ZorkEnv

**Problem:** `ZorkEnv.register_primitive()` was a no-op (inherited default). Library entries like `promoted_0` were silently ignored during execution — the environment couldn't find them in `_ZORK_PRIM_MAP`.

**Fix:** Added `__init__` with `_dynamic_prims` dict, `register_primitive()` stores there, `execute()` checks both `_ZORK_PRIM_MAP` and `_dynamic_prims`.

**Impact:** Compounding now works on Zork. Library entries reused 5-11x across rounds. Hierarchical composition demonstrated: `promoted_2 = take_treasure(go_north(go_north))`.

### Decision 53: ARC Compounding A/B Test Results

**Results on 50 ARC tasks with --compounding flag (depth-2, 3 rounds, sequential):**
- Training: 17/50 (34%) — similar to baseline
- Eval: 1/50 (2%) — train-eval gap persists
- Library: 3-5 entries with 2x reuse each

**Analysis:** Compounding produces library entries on ARC, but:
1. Most ARC solves are depth-1 (single primitive), so library entries rarely help
2. The train-eval gap (34% vs 2%) is the bigger problem — primitives are engineering-biased toward training tasks
3. Compounding works much better on Zork where tasks naturally require multi-step solutions

### Updated Results Table

| Domain | Baseline | With Compounding | Library Entries | Reuse |
|--------|----------|-----------------|-----------------|-------|
| ARC-AGI-1 Train (50) | ~21% | 34% | 3-5 | 2x |
| ARC-AGI-1 Eval (50) | ~5% | 2% | 5 | 2x |
| Zork (20 tasks) | 35% → 50%* | 50% | 5 | 5-11x |
| List Ops (28) | ~71% | ~78% | 4-8 | 4-11x |

*Drive signal fix (binary→distance-based) accounts for 35%→50% improvement.

### Decision 54: Eval Gap Analysis — Root Cause is Overfitting, Not Missing Primitives

**Analysis methodology:** All insights derived from training set data only. Eval set used only for scoring, not for understanding task patterns or tuning the algorithm.

**Training set breakdown (400 tasks):**
- Truly solved (train+test): 85 (21%)
- Overfit (train only): 16 (4%)
- Unsolved: 299 (75%)

**Key finding 1: Depth strongly predicts overfitting.**
- Truly solved depth distribution: 62% depth-0, 36% depth-1, 1% depth-2
- Overfit depth distribution: 12% depth-0, 44% depth-1, 25% depth-2, 19% depth-3
- Deeper programs are 4-5x more likely to overfit. This makes sense: more composition steps = more degrees of freedom = easier to match by coincidence.

**Key finding 2: The eval gap is NOT about missing primitives.**
The initial hypothesis was that primitives were engineered for training tasks. But the real issue is that programs matching eval training examples don't generalize to eval test examples. The previous eval "33 solved" was likely 0 truly solved (the earlier run didn't compute test_error). A fresh 50-task run confirms: 2/50 eval, 0 overfit.

**Key finding 3: 160 training near-misses exist.**
160 unsolved training tasks have error < 0.15. These are tasks where the search found *almost* the right answer. Many use `identity` (15 tasks), `complete_diag` compositions, or `fill_hole_*` variants.

**Key finding 4: Primitive generalization rates (training set).**
Best generalizers (100% gen rate, 2+ uses): `stack_mirror_v` (6), `extend_to_contact` (6), `stack_mirror_h` (5), `color_to_mc` (3), `transpose` (2), `outline` (2), `mirror_v` (2), `repeat_right` (2).
Worst: 26 primitives appear only in overfit solutions.

**Proposed fixes (train-data-derived, no eval leakage):**
1. **Occam's razor**: Penalize program depth more aggressively in energy function. Deeper programs need proportionally lower error to be selected.
2. **Per-example verification**: Require programs to score below threshold on ALL training examples individually, not just average error. This catches programs that match one example perfectly but fail on others.
3. **Near-miss refinement**: The 160 near-miss tasks are the highest-ROI targets. Many need slight improvements to existing depth-1/2 programs rather than entirely new primitives.

### Decision 55: Max-Error Blending — Full 400-Task Validation

**Implementation:** `effective_error = max(avg_error, max_error * 0.5)` in `_evaluate_program()`.

**Full 400-task results:**

| Metric | Before (avg error) | After (max-error blend) |
|--------|-------------------|------------------------|
| Train solved | 101/400 (25%) | 85/400 (21%) |
| Train test_solved | 85/400 (21%) | 77/400 (19%) |
| Train overfit | 16 (16%) | 8 (9%) |
| Eval test_solved | N/A (no test data) | 15/400 (4%) |
| Eval overfit | N/A | 1 |

**Verdict:** Overfitting halved (16→8, 16%→9%), but true solves also dropped (85→77). The 0.5 blending coefficient is too aggressive — it rejects some genuinely correct deeper programs. The coefficient needs tuning (probably 0.3 or lower). But the approach is directionally correct and now produces reliable eval numbers (15/400 truly solved with proper test evaluation).

### Decision 56: Max-Error Coefficient is Binary — 0.3 Chosen as Default

**Experiment:** Swept max-error blending coefficient across {0.15, 0.3, 0.5} on full 400-task ARC-AGI-1 training set.

**Results:**

| Coefficient | Truly Solved | Overfit | Overfit Rate |
|-------------|-------------|---------|-------------|
| None (avg only) | 85/400 (21%) | 16 | 16% |
| 0.15 | 77/400 (19%) | 8 | 9% |
| 0.30 | 77/400 (19%) | 8 | 9% |
| 0.50 | 77/400 (19%) | 8 | 9% |

**Full pipeline with 0.3 (train + eval, 400 tasks each):**
- Train: 77/400 (19.2%) truly solved, 8 overfit (9%)
- Eval: 15/400 (3.8%) truly solved, 1 overfit
- Library: 2 abstractions learned

**Key insight:** The max-error blending acts as a **binary filter**, not a gradient. Overfit solutions have very high max_error values relative to avg_error (catastrophic failure on one or more examples), so any coefficient in [0.15, 0.5] blocks the same set of programs. The coefficient choice within this range doesn't matter.

**Decision:** Keep coefficient at 0.3 — a moderate default that's robust across the tested range. The original 0.5 was not "too aggressive" as hypothesized in Decision 55; rather, all coefficient values produce the same outcome because the overfitting programs fail dramatically on at least one example.

**Implication:** To recover the 8 lost true solves (85→77), we need a different approach than coefficient tuning. Options:
1. Per-example thresholding (flag only if max_error > k * avg_error for some k)
2. Leave-one-out validation within training examples
3. Accept the 8-solve cost as the price of halving overfitting

### Decision 57: Adaptive Compute Reallocation — Negative Result

**Hypothesis:** Near-miss tasks (152 tasks with error < 0.15) might convert to solves with more compute and wider search breadth.

**Implementation:** `--adaptive-realloc` flag in `CurriculumConfig`. After the first wake pass, re-runs near-miss tasks with:
- 3x eval budget
- +20 pair top-K (40→60)
- +10 triple top-K (15→25)

**400-task results:**

| Metric | Without realloc | With realloc |
|--------|----------------|-------------|
| Truly solved | 77/400 (19.2%) | 77/400 (19.2%) |
| Overfit | 8 | 9 |
| Extra compute | 0 | ~152 tasks re-run |

**Verdict: No improvement.** The near-misses are NOT budget-constrained or breadth-constrained. The exhaustive search already covers all depth-1, depth-2, and most depth-3 compositions in the first pass. More compute just re-does the same work.

**Root cause confirmed: The bottleneck is primitive coverage, not search compute.**
- 76 depth-0 near-misses: No single primitive in the 342 available solves these
- 73 depth-1 near-misses: No 2-primitive composition works either
- 3 depth-2 near-misses: Even 3-primitive chains aren't enough

**Near-miss pattern analysis (training set only):**
- `identity` appears 19 times (search found nothing useful)
- `draw_diag(complete_diag)` appears 3 times at err=0.017 (very close)
- Top root primitives: identity(19), complete_diag(4), mark_inters_excl_axis(3)
- Error distribution: 18 tasks under 0.03, 45 under 0.05, 103 under 0.10

**Next step:** Analyze training near-miss input/output pairs to identify what primitives are missing. The 18 tasks with error < 0.03 are the highest priority — the search is *almost* there, suggesting a small primitive gap.

### Decision 58: Eval Generalization Strategy — No Data Leakage

**Principle:** Eval set is scoring-only. All primitive design, algorithm tuning, and analysis use training data exclusively.

**Leakage-free approaches to improving eval:**
1. **Training near-miss analysis**: Inspect training I/O pairs for near-miss tasks to identify missing primitives. These primitives are domain-general (grid transformations), not task-specific.
2. **Primitive generalization filtering**: Only promote primitives with 100% gen rate on training (Decision 54). New primitives must meet this bar.
3. **Algorithm improvements**: Search optimizations (like adaptive realloc) apply uniformly to all tasks. If they help training, they help eval.
4. **Structural primitives**: Adding general grid operations (symmetry, color remapping, object manipulation) is domain knowledge, not data leakage.

**What we will NOT do:**
- Inspect eval task patterns to design primitives
- Tune thresholds to maximize eval scores
- Cherry-pick eval results

### Decision 59: Near-Miss Deep Analysis — The Primitive Gap is Color-Context

**Methodology:** Analyzed 45 closest near-misses (err < 0.05) on training set by executing the best program found and diffing output vs expected output cell-by-cell. All analysis uses training data only.

**Error type distribution (45 tasks, err < 0.05):**

| Category | Count | Description |
|----------|-------|-------------|
| multi_recolor | 32 | 3+ color transitions needed — structure right, colors wrong |
| color_swap | 6 | Two colors need to be swapped |
| two_recolors | 4 | Two distinct color changes needed |
| single_recolor | 3 | One color→color change needed |

**Detailed near-miss patterns (top 10, err < 0.02):**

| Task | Best Program | Wrong Cells | Issue |
|------|-------------|-------------|-------|
| 29ec7d0e | draw_diag(complete_diag) | 4/324 (1%) | Wrong colors at diagonal endpoints |
| ba97ae07 | recolor_minor_cols | 6/169 (4%) | Recolors wrong region (8→3) |
| e50d258f | extract_smallest(fill_tile) | 1/20 (5%) | Single cell edge case |
| 7f4411dc | remove_noise | 1/169 (1%) | Removes too much/too little |
| 0dfd9992 | fill_grid_inters(complete_diag) | 10/441 (2%) | Wrong colors at intersections |
| 98cf29f8 | mirror_objects_v(mirror_objects_h) | 4/238 (2%) | Artifacts after mirroring |
| 50846271 | fill_hole_8 | 10/440 (2%) | Fills with wrong color (5 not 8) |
| a48eeaf7 | move_to_contact | 2/100 (2%) | Shifts wrong object |
| 776ffc46 | identity | 10/400 (2%) | Needs contextual recoloring |
| 484b58aa | draw_diag(complete_diag) | 12/841 (1%) | Wrong diagonal colors |

**Key finding: The #1 missing capability is context-dependent color assignment.**

The search finds programs that get the **geometry** right — correct shapes, positions, sizes. But 32/45 closest near-misses have the wrong colors in 1-7% of cells. Current color primitives are hard-coded (`recolor_to_3`, `fill_hole_8`) and can't adapt to the specific color mapping a task requires.

**What the primitives can't do:**
1. Determine correct color from spatial context (neighbors, region membership)
2. Learn a color mapping from training examples and apply it
3. Post-process to fix color artifacts after structural transformations

**Proposed primitive additions (highest ROI):**
1. **`recolor_by_neighbor_vote`** — set each non-bg cell's color to the majority of its neighbors. Fixes many "artifact cleanup" cases.
2. **`auto_color_map`** — learns the dominant color mapping from training pairs and applies it. Parameterized primitive.
3. **`swap_two_colors`** — automatically identifies the two non-bg colors that differ between examples and swaps them. Fixes the 6 color_swap cases.
4. **`fill_by_surround`** — fill cells based on the color of the surrounding region. Fixes fill_hole cases where the wrong fill color is chosen.

**Estimated impact:** If these 4 primitives convert even 30% of the 45 closest near-misses, that's ~13 new solves → 77→90 training (22.5%), with proportional eval improvement expected.

### Decision 60: Context-Dependent Color Primitives — Negative Result

**Implementation:** 7 new primitives added (349 total):
- `neighbor_vote_4` / `neighbor_vote_8` — recolor by majority of 4/8-neighbors
- `swap_top2_colors` / `swap_bottom2_colors` — swap most/least common colors
- `fill_surround` — flood-fill bg cells from surrounding color
- `cleanup_isolated` — remove cells with no same-colored neighbor
- `recolor_min_to_maj` — per-component minority→majority recoloring

**400-task results:**

| Metric | Before (342 prims) | After (349 prims) | With wider search (60/25) |
|--------|-------------------|--------------------|---------------------------|
| Truly solved | 77/400 (19.2%) | 77/400 (19.2%) | 76/400 (19.0%) |
| Overfit | 8 | 7 | 8 |

**Targeted composition test:** Wrapped each of the 30 closest near-misses with each new primitive. Zero improvements. The color errors are task-specific and can't be fixed by generic color cleanup.

**Why the primitives failed:**
1. **Budget dilution**: 7 new prims add ~280 depth-2 combos and ~3000+ depth-3 combos competing for the same budget. Wider search (60/25 vs 40/15) actually *lost* 1 solve.
2. **Not in top-K**: New primitives score poorly individually on near-miss tasks (they're cleanup ops, not structural). They don't make the top-40 cut for depth-2 composition.
3. **Wrong abstraction level**: The color errors are task-specific (e.g., "color region based on position in grid pattern"). Generic neighbor-voting/swapping can't derive the correct mapping from grid structure alone.

**The real bottleneck:** The 152 near-misses need **task-conditioned** color assignment — determining the right color from the training examples themselves, not from grid spatial context. This requires either:
1. **Parameterized primitives** that fit color mappings from training I/O pairs
2. **Task-specific primitive generation** (expanding `prepare_for_task`)
3. **A fundamentally different search approach** for the color-assignment sub-problem

**Decision:** Keep the 7 new primitives (they're useful at depth-0 for 3 tasks, and will compose better as the library grows). But the next breakthrough requires parameterized/learned primitives, not more hand-coded ones.

### Decision 61: Pixel-Transition Primitives + Color Remap Safety

**Two changes:**

1. **Pixel-transition analysis in `prepare_for_task`**: For same-sized I/O pairs, analyze pixel-level color transitions. If color A consistently becomes color B (≥70%, ≥2 occurrences), generate `task_recolor_A_to_B` primitive. These are task-specific and composable.

2. **Color remap safety check**: `infer_output_correction` now verifies that remapping a color fixes more pixels than it corrupts. Previously, a remap like `{3→2}` would destroy all correct color-3 pixels to fix a few wrong ones. Also: ambiguous colors are now skipped instead of rejecting the entire remap.

**400-task results:**

| Metric | Before | After |
|--------|--------|-------|
| Train solved | 77/400 (19.2%) | 77/400 (19.2%) |
| Overfit | 8 | 6 (-2) |
| Eval solved | 15/400 (3.8%) | 15/400 (3.8%) |
| Near-misses improved | — | 15 tasks |
| Near-misses worsened | — | 7 tasks |

**Notable improvements:**
- 0d3d703e: 0.46→0.07 (using `task_recolor_2_to_6`)
- ea32f347: 0.11→0.03 (safer remap)
- 63613498: 0.05→0.04 (using `task_recolor_9_to_5`)

**Verdict:** No new solves but overfitting reduced (8→6) and 15 near-misses improved. The task-recolor primitives are being composed effectively. The remaining gap requires spatial (per-region) color assignment, not global remapping.

### Decision 62: Architecture Roadmap — Grammar Evolution

**Current bottleneck:** The 152 near-misses need task-conditioned, spatially-aware color assignment. Global remaps destroy correct pixels. Context-dependent cleanup (Decision 60) doesn't capture task semantics.

**Roadmap for breaking the plateau (in order of expected impact):**

1. **Map-over-objects**: Decompose grid → apply transform per-object → reassemble. The object decomposition infrastructure exists but is underutilized. Many tasks apply the same transform to each object independently but with per-object parameters.

2. **Recursive/iterative application**: Apply a transform until stable (fixed point). Many ARC patterns involve repeated application: fill, propagate, grow.

3. **Parameterized programs**: Programs with fitted constants (e.g., "recolor to the color of the nearest object"). Currently all primitives are zero-parameter. Adding even one fitted parameter (color choice) would dramatically expand expressiveness.

4. **Grammar evolution**: In the long term, the composition rules themselves should evolve. The sleep phase currently promotes sub-trees to primitives, but true grammar evolution means discovering new meta-operations (map, fold, iterate, condition) and adding them to the vocabulary.

---

### Decision 63: Extended Per-Object Decomposition — Pairs + Multi-Color

**Date:** 2026-03-12
**Context:** Phase 1.1 object decomposition only tried single primitives per-object. Many tasks need composed per-object transforms (e.g., crop then rotate each object).

**Changes:**
1. **Composed per-object transforms** (`objects.py`): Try top-15 × top-15 pairs of primitives applied per-object. Scoring function ranks prims by per-object pixel error to avoid O(n²) on all prims.
2. **Multi-color object segmentation** (`objects.py`): 8-connectivity flood fill groups adjacent non-background pixels regardless of color. Enables per-object transforms on multi-colored objects.
3. **`apply_transform_per_multicolor_object`**: New function paralleling `apply_transform_per_object` but using 8-connectivity segmentation.
4. **Test fix** (`test_exhaustive_enum.py`): `test_exhaustive_disabled` was brittle — expected beam search to run but new object decomp solves the task earlier. Changed to assert evaluations > 0.

**Results (400 training tasks):**
- Train: 110/400 (27.5%) — up from 77/400 (19.2%), **+33 new solves**
- Eval: 23/400 (5.8%) — up from 15/400 (3.8%), **+8 new solves**
- Combined: 93/400 (23.3%) — up from 77/400 (19.2%)

**Key insight:** The composed per-object search (Strategy 2) is where most gains come from. Many ARC tasks apply two-step transforms to individual objects.

---

### Decision 64: Decomposition as a Core Principle (Pillar 3 Dual)

**Date:** 2026-03-12
**Context:** User insight: "Decomposition is the flip side of composition" — it should be a first-class operation in the core loop, not just an ARC-specific hack. Complex problems are universally solved by decomposing into sub-problems, solving each, and recomposing.

**Architectural change:**

1. **New data type `Decomposition`** (`core/types.py`): Represents a structured decomposition of an input into parts with reassembly context. Fields: `strategy` (name), `parts` (sub-problems), `context` (reassembly info).

2. **Grammar gains `decompose()` and `recompose()` methods** (`core/interfaces.py`):
   - `decompose(input, task) → list[Decomposition]` — proposes multiple decomposition strategies
   - `recompose(decomposition, transformed_parts) → output` — reassembles transformed parts
   - Default: no decomposition (returns empty list)

3. **ARCGrammar implements both** (`domains/arc/grammar.py`):
   - Strategy 1: Same-color objects (4-connectivity) — standard ARC objects
   - Strategy 2: Multi-color objects (8-connectivity) — for multi-colored patterns
   - Recompose: place subgrids back at original positions on background canvas

4. **Phase 1.15 in learner** (`core/learner.py`): Generic decomposition phase that uses `grammar.decompose()` + `grammar.recompose()`. Tries each primitive as a per-part transform. Domain-agnostic — works for any Grammar that implements decompose/recompose.

**Design rationale:** Decomposition belongs on the Grammar (not Environment) because:
- Grammar defines "how things compose" — it should also define "how they decompose"
- Composition and decomposition are duals of the same abstraction
- Both are domain-specific but structurally universal

**Two levels of decomposition in ARC (user's framework):**
1. **Input decomposition** (perception): "How was this grid generated?" — detecting background, objects, patterns. This is inverse rendering.
2. **Transform decomposition** (program synthesis): "What operations map input to output?" — operating on the objects from level 1.

The key relationship: transform primitives operate on object primitives. You can't correctly express "rotate each object" without first decomposing the grid into objects.

**Future directions:**
- Recursive decomposition: decompose → solve → if stuck, decompose parts further
- Learned decomposition strategies: the sleep phase should discover new decomposition patterns from solved tasks
- Grammar evolution: decomposition strategies themselves should be primitives that can be composed and evolved

---

### Decision 65: Fixed-Point Iteration + Grid Partition Decomposition

**Date:** 2026-03-12
**Context:** Many ARC tasks need iterated application (fill propagation, pattern growth). Also, tasks with grids divided by separator lines need per-cell decomposition.

**Changes:**
1. **Fixed-point iteration** (`primitives.py`): `apply_until_stable(fn, grid, max_iters=20)` — applies fn repeatedly until output equals input (convergence). `make_fixed_point_fn` wrapper.
2. **Phase 1.6 in learner**: For near-miss depth-1 programs, tries `iterate(program)` — applying the program until stable. Checks if iterated version improves over single application.
3. **Grid partition decomposition** (`grammar.py`): New strategy in `decompose()` — detects separator lines, splits into cells, with `recompose` that reassembles cells with separator lines restored.

**Results:** 110/400 train (27.5%), 93/400 combined (23.2%) — same as Decision 63. The new features are structurally correct (506 tests pass, 10 new) but don't add immediate solves. They target task types (iterative propagation, grid-cell operations) that will compound with future work.

---

### Decision 66: Experimental Validation of Decomposition, Fixed-Point, Grid Partition

**Date:** 2026-03-12
**Context:** Three features were prototyped and experimentally validated:
1. Grammar decompose/recompose (Phase 1.15) — generic map-over-parts
2. Fixed-point iteration (Phase 1.6) — apply-until-stable
3. Grid partition decomposition — per-cell transforms for separator-line grids

**Experiments run:**

| Experiment | Method | Target | Result |
|---|---|---|---|
| A: Grammar decomp | decompose + single prim per part + recompose | 10 identity-best tasks | 0 solves |
| A2: Grammar decomp vs Phase 1.1 | Compare coverage | 400 tasks | Phase 1.15 adds 0 beyond Phase 1.1 |
| B: Fixed-point | iterate(prim) on near-misses | 20 nearest misses × 38 key prims | 0 solves |
| C: Grid partition | output == cell? | 50 near-misses with separators | 0 cell-sized outputs |

**Analysis:**
- **Phase 1.15 is redundant with Phase 1.1**: `try_object_decomposition` already covers decompose-apply-recompose, and does it more efficiently (includes pairs, conditional recolor).
- **Fixed-point doesn't converge**: Near-miss programs that use repeated primitives (e.g., `fill_hole_4³`) are already found by depth-3 enumeration. Iterating to convergence doesn't produce correct answers because the fix isn't convergent.
- **Grid partition tasks don't need per-cell transforms**: 200/290 unsolved tasks have separator lines, but none of the top-50 near-misses have cell-sized outputs. The separators are structural features, not decomposition boundaries.

**Decision:** Code removed. The architecturally sound abstractions (Decomposition type, Grammar.decompose/recompose) should be re-added when there's a concrete use case that validates them. The principle of decomposition-as-dual-of-composition remains correct but the current implementation doesn't find tasks where it helps.

**Lesson:** Apply scientific method — hypothesize, experiment, measure — BEFORE committing. Don't add speculative code; don't remove without evidence either.

---
*This document will be updated with each new session and major decision.*
