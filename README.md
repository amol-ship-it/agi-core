# agi-core: The Universal Learning Loop

**One algorithm. Pluggable domains. Compounding intelligence.**

```
WAKE:   observe → hypothesize → execute → score → store
SLEEP:  analyze solutions → extract recurring sub-programs → compress → add to library
REPEAT: library grows → search shrinks → harder problems → compounding
```

Based on the research and principles proposed by [Vibhor Jain](https://github.com/vibhor-77).

## Quick Start

```bash
# Clone and install
git clone https://github.com/vibhor-77/agi-core.git
cd agi-core

# Optional: create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Clone the ARC-AGI datasets
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI
git clone https://github.com/arcprizeorg/ARC-AGI-2.git data/ARC-AGI-2

# Reproduce our results — one command does train + eval with culture transfer
python -m experiments.phase1_arc

# Run the test suite (~1 second)
python -m pytest tests/ -v
```

The default command runs all 400 training tasks, saves the learned culture, then runs all 400 evaluation tasks using that culture. Results, logs, and culture snapshots are auto-saved with timestamps. Output file paths are printed at the start so you can `tail -f` them in another terminal.

**Requirements:** Python 3.10+, NumPy, SciPy, Numba. See `requirements.txt`.

### Syncing an existing clone

```bash
git pull origin main
pip install -r requirements.txt
git -C data/ARC-AGI pull
git -C data/ARC-AGI-2 pull
```

## Usage

```bash
# Full pipeline: train → save culture → eval (default, recommended)
python -m experiments.phase1_arc

# Quick subset for development (50 tasks, 500K compute cap, ~25s on M1 Max)
python -m experiments.phase1_arc --mode quick

# Train only, save culture for later
python -m experiments.phase1_arc --train-only
python -m experiments.phase1_arc --train-only --save-culture my_culture.json

# Eval only with a pre-trained culture file
python -m experiments.phase1_arc --eval-only --culture runs/20260311_120000_phase1_train_culture.json

# Single-process for debugging
python -m experiments.phase1_arc --workers 1
```

### Running a subset of tasks

Tasks are **shuffled by default** using a deterministic seed (`--seed 42`), so any subset is a representative random sample. This means you can run fewer tasks and extrapolate to the full dataset:

```bash
# Quick mode already uses 50 tasks — fastest way to iterate
python -m experiments.phase1_arc --mode quick

# Custom subset: 100 tasks with default search depth
python -m experiments.phase1_arc --max-tasks 100

# Tiny smoke test: 10 tasks
python -m experiments.phase1_arc --mode quick --max-tasks 10

# Full 400-task benchmark with quick search settings
python -m experiments.phase1_arc --mode quick --max-tasks 0
```

**Extrapolation:** If you solve 12/50 tasks (24%) in quick mode, you can expect roughly 96/400 (24%) on the full dataset. The seeded shuffle ensures the subset is unbiased.

### Other experiments

```bash
# ARC-AGI-2 (1000 training + 120 eval tasks, harder than AGI-1)
python -m experiments.phase2_arc --mode quick

# Zork text adventure — navigate rooms, collect items, unlock doors
python -m experiments.zork_baseline --mode quick

# Symbolic regression — discover mathematical formulas (y=2x+1, y=x², y=sin(x)+x, ...)
python -m domains.symbolic_math
```

The Zork domain is fully self-contained (custom game engine, no external dependencies). ARC-AGI-2 requires the dataset clone above.

### Auto-saved artifacts

Every run automatically saves timestamped files. Paths are printed at the start so you can monitor progress live:

```
runs/20260311_164939_phase1_train.log          — full console output (tee'd)
runs/20260311_164939_phase1_train.jsonl        — live per-task results (tail -f friendly)
runs/20260311_164939_phase1_train.json         — final results: meta + summary + per-task + library
runs/20260311_164939_phase1_train_culture.json — learned culture snapshot (for eval / cross-run transfer)
runs/20260311_164939_phase1_train_library.json — learned abstractions (legacy format)
runs/20260311_164939_phase1_train_metrics.json — compounding curve per round
runs/20260311_164939_phase1_train_metrics.csv  — same, for spreadsheets
```

Pipeline mode (default) also saves combined files with a summary printed at the end:

```
runs/20260311_164939_phase1_pipeline.json      — combined: parameters + train/eval summaries + all tasks
runs/20260311_164939_phase1_pipeline.jsonl     — all task records (train + eval) with phase tags
```

Monitor a running benchmark in another terminal:
```bash
tail -f runs/*_phase1_train.jsonl    # watch task results as they complete
tail -f runs/*_phase1_train.log      # watch full console output
```

### Verifying individual solves

The console output ends with a **SOLVED TASKS** section listing every solved task and its program:

```
  SOLVED TASKS (24 total)
    ✓ 007bbfb7                 program: upscale_pattern
    ✓ 00d62c1b                 program: fill_rect_interior_4
    ...

  OVERFIT TASKS (13 matched training but failed test)
    ~ 22168020                 program: fill_by_symmetry
    ...
```

"Solved" means the program passes held-out test examples — the real metric. "Overfit" means it matched training examples but failed test.

To verify a specific solve:

```bash
# View the original task
cat data/ARC-AGI/data/training/007bbfb7.json | python -m json.tool

# Search for a task in the JSONL results
grep "007bbfb7" runs/*_phase1_train.jsonl | python -m json.tool

# Search in the final JSON
python -c "import json; d=json.load(open('runs/TIMESTAMP_phase1_train.json')); print(json.dumps(d['tasks']['007bbfb7'], indent=2))"
```

Each task record includes: `task_id`, `solved` (test-verified), `train_solved`, `test_solved`, `test_error`, `energy`, `prediction_error`, `program`, `evaluations`, `wall_time`.

## Presets

Three modes. Pick one. That's the only knob most users need.

| Mode | Tasks | Beam | Compute Cap | Use case |
|------|-------|------|-------------|----------|
| `quick` | 50 | off | 500K | Fast dev loop (~25s) |
| `default` | all (400) | off | 3M | Full benchmark (~3 min) |
| `contest` | all (400) | 30×15 | 100M | Maximum accuracy (~30 min) |

All presets run **1 round** with **seed 42** by default. Results are fully deterministic.

**Why no beam search?** A/B testing on 49 tasks showed beam search (width=20, gens=10) solves **exactly the same tasks** as exhaustive-only, while adding +13% wall time. All solves come from exhaustive enumeration (depth 1-3), object decomposition, conditional search, near-miss refinement, and color fix. Beam is kept in contest mode as a safety net.

**Compute cap** is cell-normalized (larger grids get proportionally fewer evals). Experiments show solves are **bimodal**: 76 "fast" tasks solve in <500 evals (any cap works), while 9 "slow" tasks (per_object_recolor) need ~13K evals (cap ≥ 2.8M). Quick mode uses 500K for speed; default uses 3M to capture all solves. Override with `--compute-cap`:

```bash
python -m experiments.phase1_arc --compute-cap 100M    # override preset cap
```

### Expected performance

**ARC-AGI-1:**

| Mode | Training | Eval (culture transfer) | Wall time |
|------|----------|------------------------|-----------|
| `quick` | ~17/50 (~34%) | ~2/50 (~4%) | **~25s** |
| `default` | ~85/400 (~21%) | ~20/400 (~5%) | **~3 min** |
| `contest` | higher | TBD | ~30 min |

**Other domains:**

| Domain | Tasks | Solved | Rate | Notes |
|--------|-------|--------|------|-------|
| ARC-AGI-2 Train | 400 | 56 | 14% | Harder than AGI-1 |
| ARC-AGI-2 Eval | 120 | 0 | 0% | Cold start, no culture transfer |
| Zork | 20 | 10 | 50% | Compounding works: library entries reused 5-11x |
| List Ops | 28 | ~20 | ~71% | Compounding demonstrated here |

**349 hand-crafted ARC primitives** including grid partitioning, object decomposition, symmetry completion, connected components, diagonal ops, sub-grid propagation, and per-object conditional recoloring.
**Depth-3 exhaustive enumeration** with smart pool selection finds 1-4 step programs efficiently.
**Object decomposition** automatically detects per-object transform patterns and recolors by size, shape, or position.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `default` | Preset: `quick`, `default`, or `contest` |
| `--data-dir` | auto-detect | Path to ARC-AGI data directory |
| `--train-only` | off | Train only, no eval phase |
| `--eval-only` | off | Eval only (requires `--culture`) |
| `--culture` | none | Culture file to load (cross-run knowledge transfer) |
| `--save-culture` | auto | Override auto culture save path |
| `--max-tasks` | from preset | Limit tasks (0 = all). Quick: `50`, default/contest: all |
| `--rounds` | `1` | Wake-sleep rounds |
| `--beam-width` | from preset | Beam search width. Quick/default: `1` (off), contest: `30` |
| `--max-generations` | from preset | Beam generations. Quick/default: `1` (off), contest: `15` |
| `--workers` | `0` (perf cores) | Parallel workers. `0` = auto-detect performance cores |
| `--seed` | `42` | Random seed for deterministic, reproducible runs |
| `--compute-cap` | from preset | Per-task eval budget (cell-normalized). Quick/default: `0` (unlimited), contest: `50M`. `0` = unlimited |
| `--exhaustive-depth` | `3` | Exhaustive enumeration depth (`0`=off, `2`=pairs, `3`=triples) |
| `--exhaustive-pair-top-k` | `40` | Top-K singles for pair enumeration pool |
| `--exhaustive-triple-top-k` | `15` | Top-K singles for triple enumeration pool |
| `--sequential-compounding` | off | Process tasks sequentially with immediate concept promotion |
| `--runs-dir` | `runs` | Directory for all run artifacts |
| `--no-log` | off | Disable log file (console only) |
| `--verbose` | off | Enable debug logging |

## How It Works

### The wake-sleep loop

1. **WAKE**: For each task, search for a program that transforms input to output.
   All search phases respect the per-task compute budget — large grids get fewer evaluations.
   - **Exhaustive enumeration** (depth 1-3): systematically tries all single primitives, top-K pairs, and top-K triples. Solves ~97% of solvable tasks.
   - **Object decomposition**: detects per-object transform patterns via connected components, with conditional recoloring by size, shape, or position.
   - **Conditional branching**: partitions inputs by predicates (symmetric, tall, square, etc.) and finds per-group transforms.
   - **Near-miss refinement**: takes programs with error < 20% and tries appending/prepending each primitive.
   - **Color fix**: infers pixel-level color remappings from near-perfect programs.
   - **Beam search**: seeded with top enumeration results, mutates and crosses programs with semantic deduplication.
2. **SLEEP**: Analyze all solved programs. Extract recurring sub-programs.
   Add them to the library as new reusable abstractions.
3. **REPEAT**: The grown library biases future search toward proven compositions.
   This is the compounding mechanism.

### The 4 interfaces

Every domain implements exactly 4 things:

| Interface | What it does | Symbolic Math | ARC-AGI | Zork |
|-----------|-------------|---------------|---------|------|
| **Environment** | Execute programs | Evaluate formula on x | Apply grid transform | Execute game action |
| **Grammar** | Define primitives, compose/mutate | sin, cos, +, × | rotate, flip, crop | move, take, use |
| **DriveSignal** | Score: error + complexity | MSE + node count | Pixel distance + size | Game score + novelty |
| **Memory** | Store episodes, library, solutions | InMemoryStore | InMemoryStore | InMemoryStore |

The core loop (`core/learner.py`) depends **only** on these interfaces. It never imports anything domain-specific. The same search framework works for grid puzzles, symbolic math, text adventures, and (eventually) robotics — but performance on each domain depends on the quality of its primitives and domain engineering.

### Terminology

- **solved** — program passes held-out test examples (the real metric)
- **train_solved** — program matches training examples within a task (may overfit)
- **overfit** — train_solved but NOT solved (matched training, failed test)
- **solve_rate** — fraction of tasks solved (test-verified)

### The key metric: the compounding curve

```
Round  Solved     Rate  Library  New  Avg Energy   Wake(s)  Sleep(s)
---------------------------------------------------------------------
    1    2/4     50.0%        3    3      0.0012       4.2       0.1
    2    3/4     75.0%        5    2      0.0008       3.8       0.1
    3    3/4     75.0%        6    1      0.0005       3.1       0.0
```

If solve rate increases across rounds without new hand-coded primitives, the framework is working.

### Current status: what works and what doesn't

**Compounding is demonstrated on list_ops and Zork.** On list_ops (22 primitives, depth-limited to 2), the library learns depth-2 compositions that enable depth-3+ solutions in subsequent rounds, with library reuse counts of 4-11. On Zork (20 tasks across 5 difficulty levels), compounding produces 5 library entries reused 5-11x across rounds, with hierarchical composition (e.g., `take_treasure(go_north(go_north))` built from a promoted depth-2 entry).

**Compounding produces library entries on ARC but has limited impact.** With `--compounding` flag (depth-2 + 3 rounds + sequential), the system creates 3-5 library entries with 2x reuse. However, most ARC solves are depth-1 (single primitives), so the library provides little additional coverage beyond what depth-3 exhaustive search already finds.

**The training-eval gap is significant.** ARC-AGI-1 training: 85/400 (21%) truly solved (train+test), 16 overfit. Eval: ~2/50 (4%) on quick mode. Analysis shows the gap is driven by **overfitting, not missing primitives**: deeper programs (depth-2+) are 4-5x more likely to overfit than depth-0. The fix is better generalization (Occam's razor, per-example verification), not more primitives.

**Where the ARC solve rate comes from:** The 349 hand-crafted primitives (6,500 lines of domain code) encode substantial human knowledge about grid transformations. The core algorithm provides the search framework (exhaustive enumeration, beam search, object decomposition), but the primitives are the primary source of ARC performance. This is honest: the architecture is generic, but ARC results depend on domain engineering.

## Structure

```
agi-core/
├── core/                    # THE INVARIANT CORE — never imports domain code
│   ├── __init__.py          # Public API (re-exports everything)
│   ├── types.py             # Data types: Primitive, Program, Task, ScoredProgram, LibraryEntry
│   ├── interfaces.py        # 4 abstract interfaces (Environment, Grammar, DriveSignal, Memory)
│   ├── config.py            # SearchConfig, SleepConfig, CurriculumConfig
│   ├── results.py           # ParetoEntry, WakeResult, SleepResult, RoundResult
│   ├── transition_matrix.py # DreamCoder-style generative prior P(child|parent)
│   ├── learner.py           # Wake-sleep loop + beam search (the algorithm)
│   ├── runner.py            # Generic experiment runner (TeeWriter, ProgressTracker, presets)
│   ├── memory.py            # Default in-memory store
│   └── metrics.py           # Compounding curve measurement
│
├── experiments/             # Thin domain-specific wrappers over core/runner.py
│   ├── phase1_arc.py        # ARC-AGI-1 training & evaluation pipeline
│   ├── phase2_arc.py        # ARC-AGI-2 baseline experiment
│   ├── zork_baseline.py     # Zork text adventure baseline
│   └── list_compounding.py  # List ops compounding demonstration
│
├── domains/                 # Domain implementations (all 4 interfaces)
│   ├── arc/                 # ARC-AGI grid transformations (349 primitives)
│   │   ├── primitives.py    # Grid→Grid transform functions + registry
│   │   ├── objects.py       # Connected component detection
│   │   ├── environment.py   # ARCEnv
│   │   ├── grammar.py       # ARCGrammar
│   │   ├── drive.py         # ARCDrive
│   │   └── dataset.py       # Task loading + sample tasks
│   ├── symbolic_math/       # 1D symbolic regression (15 math primitives)
│   │   └── __init__.py      # All 4 interfaces in one file
│   ├── list_ops/            # List operations (22 primitives, compounding demo)
│   │   └── __init__.py      # All 4 interfaces in one file
│   └── zork/                # Text adventure (30 action primitives, 16 predicates)
│       └── __init__.py      # Game engine + all 4 interfaces
│
├── tests/                   # Test suite (513 tests, 14 files)
│   ├── test_arc.py
│   ├── test_color_fix.py
│   ├── test_compounding.py
│   ├── test_conditional_search.py
│   ├── test_exhaustive_enum.py
│   ├── test_interfaces.py
│   ├── test_learner.py
│   ├── test_list_ops.py
│   ├── test_memory.py
│   ├── test_metrics.py
│   ├── test_object_decomposition.py
│   ├── test_symbolic_math.py
│   └── test_zork.py
│
├── runs/                    # Run artifacts — timestamped, git-ignored
├── data/                    # External datasets (git-ignored)
│
├── CLAUDE.md                # Persistent instructions for Claude Code sessions
├── PROMPTS.md               # Chronological log of all prompts
├── DECISIONS.md             # Chronological log of all decisions
├── requirements.txt         # Python dependencies (numpy, scipy, numba, pytest)
└── README.md                # This file
```

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=core --cov=domains --cov-report=term-missing
```

**Current coverage (513 tests):** 72% overall. Core modules: learner 66%, runner 27% (mostly CLI/pipeline code), all other core modules 95-100%. Domain modules: ARC primitives 73%, ARC objects 57%, ARC environment 89%, Zork 95%, list_ops 94%.

## Documentation

- **[PROMPTS.md](PROMPTS.md)** — Every instruction given to Claude, in chronological order
- **[DECISIONS.md](DECISIONS.md)** — Every technical decision, rationale, and result
- **[CLAUDE.md](CLAUDE.md)** — Persistent rules for Claude Code sessions

These documents allow anyone to reproduce the exact trajectory of this project.

## Roadmap

- **Phase 0** ✅ Extract invariant core with pluggable interfaces
- **Phase 1** ✅ ARC-AGI-1 training (349 primitives, exhaustive enumeration, wake-sleep) — 85/400 (21%)
- **Phase 2** ✅ ARC-AGI-1 eval with culture transfer — ~20/400 (5%)
- **Phase 3** ✅ Additional domains (Zork 20 tasks, list_ops), same core — compounding demonstrated on list_ops and Zork
- **Phase 4** ✅ Compounding infrastructure: `--compounding` flag, distance-based drive signals, library primitive execution. Zork: 7/20→10/20 with library reuse 5-11x. ARC: library entries produced but limited impact.
- **Phase 5** 🔧 Close the ARC train-eval gap (34% train vs 2% eval) — the critical bottleneck
- **Phase 6** 🔧 ARC-AGI-2 baseline established (10% train, 0% eval) — improve toward parity with AGI-1
- **Phase 6** Cross-domain library transfer
- **Phase 7** Continuous mixed-domain learning

## License

MIT
