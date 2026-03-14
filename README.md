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

The default command runs all 400 training tasks, saves the learned culture, then runs all 400 evaluation tasks using that culture. Results, logs, visualizations, and culture snapshots are auto-saved with timestamps. Output file paths are printed at the start so you can `tail -f` them in another terminal.

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

# Quick subset for development (50 tasks, 500K compute cap, ~5s)
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

Tasks are **shuffled by default** using a deterministic seed (`--seed 42`), so any subset is a representative random sample:

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
runs/phase1_arc_pipeline_TIMESTAMP.json          — combined: parameters + train/eval summaries + all tasks
runs/phase1_arc_pipeline_TIMESTAMP.jsonl         — all task records (train + eval) with phase tags
runs/phase1_arc_train_TIMESTAMP_culture.json     — learned culture snapshot (for eval / cross-run transfer)
runs/phase1_arc_pipeline_TIMESTAMP_train_viz.html — train results visualization (index)
runs/phase1_arc_pipeline_TIMESTAMP_eval_viz.html  — eval results visualization (index)
runs/phase1_arc_pipeline_TIMESTAMP_train_viz/     — per-task detail pages (train)
runs/phase1_arc_pipeline_TIMESTAMP_eval_viz/      — per-task detail pages (eval)
```

In standalone mode (`--train-only` or `--eval-only`), each phase writes its own files:

```
runs/phase1_arc_train_TIMESTAMP.log      — full console output (tee'd)
runs/phase1_arc_train_TIMESTAMP.jsonl    — live per-task results (tail -f friendly)
runs/phase1_arc_train_TIMESTAMP.json     — final results: meta + summary + per-task + library
runs/phase1_arc_train_TIMESTAMP_viz.html — results visualization (index)
runs/phase1_arc_train_TIMESTAMP_viz/     — per-task detail pages
```

Monitor a running benchmark in another terminal:
```bash
tail -f runs/phase1_arc_*.jsonl    # watch task results as they complete
tail -f runs/phase1_arc_*.log      # watch full console output
```

### Visualization

HTML visualizations are auto-generated after every run. The index page shows all tasks with colored status indicators and grid previews (Input | Expected | Predicted for each example). Click any task to see the full detail page with step-by-step primitive execution showing each intermediate transformation.

```bash
# Regenerate visualization from a previous run
python -m experiments.visualize_results runs/phase1_arc_pipeline_TIMESTAMP.json

# Filter to only show overfit tasks
python -m experiments.visualize_results runs/phase1_arc_pipeline_TIMESTAMP.json --filter overfit
```

### Verifying individual solves

The console output ends with a **SOLVED TASKS** section listing every solved task and its program:

```
  SOLVED TASKS (107 total)
    ✓ 007bbfb7                 program: upscale_pattern
    ✓ 00d62c1b                 program: fill_rect_interior_4
    ...

  OVERFIT TASKS (170 matched training but failed test)
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

Each task record includes: `task_id`, `solved` (test-verified), `train_solved`, `test_solved`, `test_error`, `energy`, `prediction_error`, `program`, `evaluations`, `wall_time`, `train_predictions`, `test_predictions`.

## Presets

Three modes. Pick one. That's the only knob most users need.

| Mode | Tasks | Beam | Compute Cap | Vocabulary | Use case |
|------|-------|------|-------------|------------|----------|
| `quick` | 50 | off | 500K | full/minimal | Fast dev loop (~5s) |
| `default` | all (400) | off | 3M | full/minimal | Full benchmark (~4 min) |
| `contest` | all (400) | 30×15 | 100M | full/minimal | Maximum accuracy |

Two vocabulary modes (`--vocabulary`):
- `full`: 180 hand-crafted primitives (max single-round coverage)
- `minimal`: 60 fundamental primitives (action + perception + composition rules — cleaner, less overfit, designed for compounding)

All presets run **1 round** with **seed 42** by default. Results are fully deterministic (`PYTHONHASHSEED=0` is enforced automatically).

**Compute cap** is cell-normalized (larger grids get proportionally fewer evals). Override with `--compute-cap`:

```bash
python -m experiments.phase1_arc --compute-cap 100M    # override preset cap
```

### Expected performance

**ARC-AGI-1** — eval accuracy (test-verified solves on held-out evaluation set):

| Vocabulary | All 400 (quick cap) | All 400 (default cap) |
|-----------|--------------------|-----------------------|
| `full` (180 prims) | 25/400 (6.2%) | **36/400 (9.0%)** |
| `minimal` (60 prims) | 26/400 (6.5%) | — |

Note: training accuracy is higher (default: 110/400 = 27.5%) with overfit rate reduced to 24% (was 61% before cleanup). Cross-reference composition rule added +5 eval solves with zero overfit.

**Other domains:**

| Domain | Eval Tasks | Solved | Rate | Notes |
|--------|-----------|--------|------|-------|
| ARC-AGI-2 | 120 | 0 | 0.0% | With culture transfer from 1000 training tasks |
| Zork | 20 | 10 | 50% | 5 library entries, reuse 2-6x (5 rounds) |
| List Ops | 28 | 20 | 71.4% | 8 library entries, reuse 2-6x (3 rounds) |

**Two vocabulary modes:** `full` (180 hand-crafted primitives) or `minimal` (60 fundamental action+perception primitives designed for composition and compounding). Both include task-specific color primitives (~30-40 per task). The minimal set beats full at the same compute budget by covering more of the program space.
**Depth-3 exhaustive enumeration** with smart pool selection finds 1-4 step programs efficiently.
**Object decomposition** automatically detects per-object transform patterns and recolors by size, shape, or position.
**Simple correction** learns color remappings and small (3x3) neighborhood patches on near-miss programs.

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
| `--compute-cap` | from preset | Per-task eval budget (cell-normalized). `0` = unlimited |
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
   - **Exhaustive enumeration** (depth 1-3): systematically tries all single primitives, top-K pairs, and top-K triples.
   - **Object decomposition**: detects per-object transform patterns via connected components, with conditional recoloring by size, shape, or position.
   - **Conditional branching**: partitions inputs by predicates (symmetric, tall, square, etc.) and finds per-group transforms.
   - **Near-miss refinement**: takes programs with error < 20% and tries appending/prepending each primitive.
   - **Correction**: infers color remappings and small neighborhood patches (3x3) from near-miss programs.
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

**Compounding is demonstrated on list_ops and Zork.** On list_ops (22 primitives, depth-limited to 2), the library learns depth-2 compositions that enable depth-3+ solutions in subsequent rounds — 8 library entries with reuse 2-6x across 3 rounds. On Zork (20 tasks, 5 difficulty levels), compounding produces 5 library entries reused 2-6x across rounds, with hierarchical composition (e.g., `take_treasure(go_north(go_north))` built from a promoted depth-2 entry).

**Compounding produces library entries on ARC but has limited impact.** Most ARC solves are depth-1 (single primitives), so the library provides little additional coverage beyond what depth-3 exhaustive search already finds.

**Overfitting is reduced but still present.** In default mode, 141/400 programs match training but only 106/400 pass test (29% overfit rate). The aggressive correction cascade (5x5-11x11 neighborhoods, identity-seeded corrections) was removed in favor of clean, generalizable corrections only.

**Where the ARC solve rate comes from:** 180 base primitives plus task-specific additions (~9,000 lines of domain code) encode human knowledge about grid transformations. The core algorithm provides the search framework (exhaustive enumeration, beam search, object decomposition, correction), but ARC results depend on domain engineering — the architecture is generic, but the primitives are essential.

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
│   ├── visualize_results.py # HTML visualization generator
│   ├── zork_baseline.py     # Zork text adventure baseline
│   └── list_compounding.py  # List ops compounding demonstration
│
├── domains/                 # Domain implementations (all 4 interfaces)
│   ├── arc/                 # ARC-AGI grid transformations (60 minimal / 180 full)
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
├── tests/                   # Test suite (545 tests)
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

**Current coverage (545 tests):** 73% overall. Core modules: learner 68%, runner 25% (mostly CLI/pipeline code), all other core modules 95-100%. Domain modules: ARC primitives 75%, ARC grammar 78%, ARC objects 54%, ARC environment 94%, Zork 95%, list_ops 94%.

## Documentation

- **[PROMPTS.md](PROMPTS.md)** — Every instruction given to Claude, in chronological order
- **[DECISIONS.md](DECISIONS.md)** — Every technical decision, rationale, and result
- **[CLAUDE.md](CLAUDE.md)** — Persistent rules for Claude Code sessions

These documents allow anyone to reproduce the exact trajectory of this project.

## Roadmap

- **Phase 0** ✅ Extract invariant core with pluggable interfaces
- **Phase 1** ✅ ARC-AGI-1 training (exhaustive enumeration, object decomposition, wake-sleep)
- **Phase 2** ✅ ARC-AGI-1 eval with culture transfer
- **Phase 3** ✅ Additional domains (Zork 20 tasks, list_ops), same core — compounding demonstrated
- **Phase 4** ✅ Compounding infrastructure: Zork 10/20, list_ops 20/28 with library reuse 2-6x
- **Phase 5** ✅ Cleanup: removed overfit correction cascade, pruned dead primitives (235→180)
- **Phase 6** ✅ Minimal vocabulary (60 fundamental primitives: action + perception)
- **Phase 7** ✅ Composition rules: FOR_EACH, CROSS_REFERENCE (+10 zero-overfit solves)
- **Phase 8** ✅ Current: ARC-AGI-1 eval 36/400 (9.0%), minimal vocab 26/400 (6.5%)
- **Phase 9** 🔧 More composition rules (mask application, counting-based repetition)
- **Phase 10** 🔧 Compounding with richer compositions across domains
- **Phase 11** Cross-domain library transfer
- **Phase 12** Continuous mixed-domain learning

## License

MIT
