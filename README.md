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
python -m common --domain arc-agi-1 --rounds 3

# Run the test suite (~4 seconds)
python -m pytest tests/ -v
```

The default mode is pipeline: interleaved train → eval rounds, saving the learned culture after each training round and evaluating with it. Results, logs, visualizations, and culture snapshots are auto-saved with timestamps.

**Requirements:** Python 3.10+, NumPy, SciPy, Numba. See `requirements.txt`.

### Syncing an existing clone

```bash
git pull origin main
pip install -r requirements.txt
git -C data/ARC-AGI pull
git -C data/ARC-AGI-2 pull
```

## Usage

All domains run through a single CLI entry point:

```bash
# Full pipeline: train → eval × N rounds (default)
python -m common --domain arc-agi-1 --rounds 3

# Quick subset for development (50 tasks, 500K compute cap, ~4s)
python -m common --domain arc-agi-1 --mode quick

# Train only, save culture for later
python -m common --domain arc-agi-1 --run-mode single --split training
python -m common --domain arc-agi-1 --run-mode single --split training --save-culture my_culture.json

# Eval only with a pre-trained culture file
python -m common --domain arc-agi-1 --run-mode single --split evaluation --culture runs/TIMESTAMP_culture.json

# Single-process for debugging
python -m common --domain arc-agi-1 --workers 1

# Batch mode for hyperparameter sweeps (no visualization, minimal console output)
python -m common --domain arc-agi-1 --mode quick --batch
```

### Running a subset of tasks

Tasks are **shuffled by default** using a deterministic seed (`--seed 42`), so any subset is a representative random sample:

```bash
# Quick mode already uses 50 tasks — fastest way to iterate
python -m common --domain arc-agi-1 --mode quick

# Custom subset: 100 tasks with default search depth
python -m common --domain arc-agi-1 --max-tasks 100

# Tiny smoke test: 10 tasks
python -m common --domain arc-agi-1 --mode quick --max-tasks 10

# Multi-round compounding on quick subset
python -m common --domain arc-agi-1 --mode quick --rounds 3
```

### Other domains

```bash
# ARC-AGI-2 (1000 training + 120 eval tasks, harder than AGI-1)
python -m common --domain arc-agi-2 --mode quick

# Zork text adventure — navigate rooms, collect items, unlock doors
python -m common --domain zork --mode quick

# List operations — compounding demonstration
python -m common --domain list-ops --mode quick

# Symbolic regression — discover mathematical formulas (y=2x+1, y=x², y=sin(x)+x, ...)
python -m domains.symbolic_math
```

### Auto-saved artifacts

Every run automatically saves timestamped files:

```
runs/arc_agi_1_training_TIMESTAMP.jsonl        — live per-task results (tail -f friendly)
runs/arc_agi_1_training_TIMESTAMP.json         — final results: meta + summary + per-task + library
runs/arc_agi_1_training_TIMESTAMP_culture.json — learned culture snapshot (for eval / cross-run transfer)
runs/arc_agi_1_training_TIMESTAMP_viz.html     — results visualization (index)
runs/arc_agi_1_training_TIMESTAMP.log          — full console output
```

Pipeline mode adds:
```
runs/arc_agi_1_pipeline_TIMESTAMP_culture.jsonl — live learning events across all rounds
runs/arc_agi_1_pipeline_TIMESTAMP.json         — combined train + eval results + compounding curve
```

Monitor a running benchmark:
```bash
tail -f runs/arc_agi_1_*.jsonl    # watch task results as they complete
tail -f runs/*_culture.jsonl      # watch library growth across rounds
```

### Visualization

HTML visualizations are auto-generated after every run. The index page shows all tasks with colored status indicators and grid previews. Click any task to see step-by-step primitive execution showing each intermediate transformation. Learned abstractions are expanded inline.

```bash
# Regenerate visualization from a previous run
python -m experiments.visualize_results runs/arc_agi_1_training_TIMESTAMP.json

# Filter to only show overfit tasks
python -m experiments.visualize_results runs/arc_agi_1_training_TIMESTAMP.json --filter overfit
```

### Verifying individual solves

The console output ends with a **SOLVED TASKS** section listing every solved task and its program:

```
  SOLVED TASKS (4 total)
    ✓ 08ed6ac7                 program: label_components
    ✓ 1cf80156                 program: crop_to_content
    ✓ 1e0a9b12                 program: gravity_down
    ✓ 2013d3e2                 program: crop_half_left(learned_8=crop_half_top(crop_to_content))
```

"Solved" means the program passes held-out test examples — the real metric. Learned abstractions show their expansion inline.

## Presets

Three modes. Pick one. That's the only knob most users need.

| Mode | Tasks | Compute Cap | Use case |
|------|-------|-------------|----------|
| `quick` | 50 | 500K | Fast dev loop (~5s) |
| `default` | all (400) | 3M | Full benchmark (~2 min) |
| `contest` | all (400) | 50M | Maximum accuracy (~12 min) |

Presets differ only in compute budget. All search parameters — rounds, pair/triple pool sizes, beam width — are **auto-derived** from the compute cap via `derive_search_params()`. Higher budget → wider search pools → more solves on a single diminishing-returns curve. CLI flags like `--exhaustive-pair-top-k` override auto-derived values when explicitly set.

All runs use **atomic vocabulary** — 48 truly atomic primitives (22 transforms + 18 perception + 8 parameterized). Each primitive is one intuitive visual concept. 12 predicates enable conditional branching. Compositional operations like crop_to_content = trim_cols(trim_rows(x)) must be discovered through composition. Structural search strategies (per-object, cross-reference, conditional, scale/tile detection) compose these same primitives in structurally different ways.

Rounds are auto-derived: 2 for budget ≥200K, 3 for ≥20M. Results are fully deterministic with **seed 42** (`PYTHONHASHSEED=0` is enforced automatically).

**Rounds sweet spot** (measured 2026-03-15):

| Mode | 1 round | 2 rounds | 3 rounds |
|------|---------|----------|----------|
| quick (50 tasks) | 3/50 (6%) 3s | **4/50 (8%) 5s** | 6/50 (12%) 10s |
| default (400 tasks) | 23/400 (5.8%) 39s | **34/400 (8.5%) 97s** | — |
| contest (400 tasks) | 35/400 (8.8%) 187s | 39/400 (9.8%) 194s | **40/400 (10.0%) 214s** |

Default: round 2 gives +36% solves. Contest: +50% over default R1 from wider search; 3 rounds compounds further. Contest trades ~6× wall time for +30% more solves.

**Compute cap** is cell-normalized (larger grids get proportionally fewer evals). Override with `--compute-cap`:

```bash
python -m common --domain arc-agi-1 --compute-cap 100M    # override preset cap
```

### Expected performance

**ARC-AGI-1** (measured 2026-03-15):

| Mode | Round | Training (400) | Eval (400) | Library | Overfit |
|------|-------|---------------|------------|---------|---------|
| default | 1 | 23/400 (5.8%) | 9/400 (2.2%) | 200 | 1 |
| default | 2 | 34/400 (8.5%) | 9/400 (2.2%) | 200 | 1 |
| contest | 1 | 35/400 (8.8%) | 9/400 (2.2%) | 200 | 3 |
| contest | 2 | 39/400 (9.8%) | 11/400 (2.8%) | 200 | 6 |
| contest | 3 | 40/400 (10.0%) | 9/400 (2.2%) | 200 | 7 |

Contest mode finds +6 more training solutions (+18%) than default via wider exhaustive search and beam search. Eval peaks at 11 in R2 (vs 9 default). Overfit increases with more compute — wider search finds more task-specific solutions that don't generalize to test.

Quick mode (50 training tasks, ~5s): 6/50 (12%) with 3 rounds.

**Other domains:**

| Domain | Eval Tasks | Solved | Rate | Notes |
|--------|-----------|--------|------|-------|
| ARC-AGI-2 | 120 | 0 | 0.0% | With culture transfer from 1000 training tasks |
| Zork | 20 | 10 | 50% | 5 library entries, reuse 2-6x (5 rounds) |
| List Ops | 28 | 20 | 71.4% | 8 library entries, reuse 2-6x (3 rounds) |

**Three primitive kinds:** transforms (Grid→Grid), perception (Grid→Value), and parameterized ((Value,...) → Grid→Grid factory). Parameterized prims like `swap_colors(background_color, dominant_color)` are fully transferable — same program works on any task regardless of specific colors.

**Depth-3 exhaustive enumeration** with no-op pruning and binary near-miss refinement.

**Sleep learning** extracts subtrees from ALL programs (solved + unsolved), quality-weighted by accuracy. Promotes transferable compositions to a bounded library with eviction — reused entries are immune, weak entries displaced by better ones.

**Interleaved pipeline** runs train → eval per round, so each eval shows the value of compounding so far. The compounding curve (train/eval per round) is printed at the end and saved in the pipeline JSON.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | (required) | Domain: `arc-agi-1`, `arc-agi-2`, `zork`, or `list-ops` |
| `--mode` | `default` | Preset: `quick`, `default`, or `contest` |
| `--run-mode` | `pipeline` | `pipeline` (train → eval per round) or `single` (train or eval only) |
| `--split` | `training` | Data split for single mode: `training` or `evaluation` |
| `--rounds` | `1` | Wake-sleep rounds (use 3+ for compounding) |
| `--sequential-compounding` | off | Process tasks sequentially with immediate concept promotion |
| `--culture` | none | Culture file to load (cross-run knowledge transfer) |
| `--save-culture` | auto | Override auto culture save path |
| `--max-tasks` | from preset | Limit tasks (0 = all). Quick: `50`, default: all |
| `--workers` | `0` (perf cores) | Parallel workers. `0` = auto-detect performance cores |
| `--seed` | `42` | Random seed for deterministic, reproducible runs |
| `--compute-cap` | from preset | Per-task eval budget (cell-normalized). `0` = unlimited |
| `--exhaustive-depth` | `3` | Exhaustive enumeration depth (`0`=off, `2`=pairs, `3`=triples) |
| `--exhaustive-pair-top-k` | auto | Top-K singles for pair enumeration pool (auto-derived from compute cap) |
| `--exhaustive-triple-top-k` | auto | Top-K singles for triple enumeration pool (auto-derived from compute cap) |
| `--task-ids` | none | Comma-separated task IDs to run (prefix match) |
| `--data-dir` | auto-detect | Path to data directory |
| `--runs-dir` | `runs` | Directory for all run artifacts |
| `--no-log` | off | Disable log file (console only) |
| `--batch` | off | Batch mode: skip visualization + per-task output (for hyperparameter sweeps) |

## How It Works

### The wake-sleep loop

1. **WAKE**: For each task, search for a program that transforms input to output.
   - **Exhaustive enumeration** (depth 1-3): tries all single primitives, top-K pairs, and top-K triples. Parameterized prims are tried with all perception children.
   - **Near-miss refinement**: takes programs with error < 20% and tries appending, prepending, or wrapping with binary ops (overlay, mask_by).
   - No-op pruning: primitives that don't change the grid are skipped at depth 2+.
2. **SLEEP**: Analyze all solved programs and best unsolved attempts.
   Extract recurring sub-programs, quality-weighted by accuracy (solved=1.0, unsolved=max(exp(-error)×0.5, solve_score×0.5)).
   Per-example solve scoring: `(k/n)^2` where k=examples solved perfectly — a 2/3-solver always beats a uniformly mediocre program.
   **Primitive ROI tracking**: credit all primitives in program trees with quality weight; scores accumulate across tasks and decay each round.
   Promote transferable compositions (depth 2+) directly as library entries.
   Train composition priors (transition matrix) on all programs.
   Bounded library with eviction: reused entries are immune, weak entries displaced by better ones.
3. **REPEAT**: The grown library expands the effective vocabulary.
   Depth-1 search over a promoted depth-2 composition reaches depth-3 effectively.
   This is the compounding mechanism — each round builds on the last.

### The 4 interfaces

Every domain implements exactly 4 things:

| Interface | What it does | Symbolic Math | ARC-AGI | Zork |
|-----------|-------------|---------------|---------|------|
| **Environment** | Execute programs | Evaluate formula on x | Apply grid transform | Execute game action |
| **Grammar** | Define primitives, compose/mutate | sin, cos, +, × | rotate, flip, crop | move, take, use |
| **DriveSignal** | Score: error + complexity | MSE + node count | -log(similarity) + size | Game score + novelty |
| **Memory** | Store episodes, library, solutions | InMemoryStore | InMemoryStore | InMemoryStore |

The core loop (`core/learner.py`) depends **only** on these interfaces. It never imports anything domain-specific.

### Terminology

- **solved** — program passes held-out test examples (the real metric)
- **train_solved** — program matches training examples within a task (may overfit)
- **overfit** — train_solved but NOT solved (matched training, failed test)
- **solve_rate** — fraction of tasks solved (test-verified)

### The key metric: the compounding curve

```
COMPOUNDING CURVE (train / eval per round):
Round         Train  Overfit  Library          Eval  Overfit
─────  ────────────  ───────  ───────  ────────────  ───────
    1   23/400 (5.8%)        1      200     9/400 (2.2%)        2
    2   34/400 (8.5%)        1      200     9/400 (2.2%)        2
```

If solve rate increases across rounds without new hand-coded primitives, the framework is working.

### Current status

**Compounding demonstrated on ARC with atomic vocabulary + structural search.** Training compounds: 24→33 (+9) across 3 rounds. Per-object recolor (10 strategies) contributes ~30% of training solves. Library entries transfer to eval. Color fix + cell-wise patches on near-misses adds 2-3 more. Eval R1 reaches 10/400 (2.5%).

**Five search strategies** compose the same atomic primitives differently:
1. **Exhaustive enumeration** — depth 1-3 sequential pipelines + mixed parameterized/transform compositions
2. **Object decomposition** — per-object transforms, conditional recolor by properties (size, shape, hole, position)
3. **Cross-reference** — boolean ops on grid halves, separator-based cell extraction, scale/tile detection
4. **Color fix** — learn color remapping from near-miss program outputs
5. **Cell-wise patch** — learn fixed pixel corrections for near-miss outputs (<15% difference)

**Three primitive kinds:** transforms (Grid→Grid), perception (Grid→Value), and parameterized ((Value,...) → Grid→Grid factory). All compositions are fully transferable across tasks. **12 predicates** enable conditional branching (if/else programs).

### Current limitations

- **Composition depth bottleneck.** Depth-4+ compositions are verified to work manually but can't be found by depth-3 exhaustive search. Compounding across rounds builds up to depth-4+ but saturates quickly.
- **Library diversity.** 200 library entries learned from training. Most entries are task-specific. Cross-task transfer remains the key challenge.
- **Eval-train gap.** Training 8.5% vs eval 2.2% — structural strategies (per-object recolor) don't transfer to eval because the learned rules are task-specific.

## Structure

```
agi-core/
├── core/                    # THE INVARIANT CORE — never imports domain code
│   ├── __init__.py          # Public API (re-exports everything)
│   ├── types.py             # Data types: Primitive, Program, Task, ScoredProgram, LibraryEntry
│   ├── interfaces.py        # 5 abstract interfaces (Environment, Grammar, DriveSignal, Memory, DomainAdapter)
│   ├── config.py            # SearchConfig, SleepConfig, CurriculumConfig
│   ├── results.py           # ParetoEntry, WakeResult, SleepResult, RoundResult
│   ├── transition_matrix.py # DreamCoder-style generative prior P(child|parent)
│   ├── learner.py           # Wake-sleep loop (the algorithm)
│   ├── memory.py            # Default in-memory store
│   └── metrics.py           # Compounding curve measurement
│
├── common/                  # Benchmark infrastructure (runner, pipeline, CLI)
│   ├── __init__.py          # Public API
│   ├── benchmark.py         # ExperimentConfig, run_experiment, run_pipeline, presets, progress
│   └── __main__.py          # Unified CLI: python -m common --domain arc-agi-1 --mode quick
│
├── experiments/             # Diagnostic tools and visualization
│   └── visualize_results.py # HTML visualization generator (expands learned abstractions)
│
├── domains/                 # Domain implementations (4 interfaces + DomainAdapter)
│   ├── arc/                 # ARC-AGI grid transformations (48 atomic primitives, 12 predicates)
│   │   ├── transformation_primitives.py # Atomic transforms + parameterized factories (self-contained)
│   │   ├── perception_primitives.py     # Atomic perception Grid→Value (self-contained)
│   │   ├── primitives.py    # Registry (_PRIM_MAP) + utilities (to_np, from_np)
│   │   ├── objects.py       # Connected component detection
│   │   ├── environment.py   # ARCEnv (handles transform, perception, parameterized execution)
│   │   ├── grammar.py       # ARCGrammar (atomic vocabulary, structural phase gating)
│   │   ├── drive.py         # ARCDrive
│   │   ├── dataset.py       # Task loading, sample tasks, data auto-detection
│   │   └── adapter.py       # ARCAdapter (hardcoded to atomic vocabulary)
│   ├── symbolic_math/       # 1D symbolic regression (15 math primitives)
│   │   └── __init__.py      # All 4 interfaces in one file
│   ├── list_ops/            # List operations (22 primitives, compounding demo)
│   │   ├── __init__.py      # All 4 interfaces in one file
│   │   └── adapter.py       # ListOpsAdapter
│   └── zork/                # Text adventure (30 action primitives, 16 predicates)
│       ├── __init__.py      # Game engine + all 4 interfaces
│       └── adapter.py       # ZorkAdapter
│
├── tests/                   # Test suite (402 tests)
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

**434 tests.** Core modules: learner, memory, config, types 95-100%. Domain: ARC atomic primitives, environment, grammar, drive. Integration: pipeline, compounding, visualization, batch mode.

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
- **Phase 5-8** ✅ Cleanup, minimal vocabulary, composition rules, eval 36/400
- **Phase 9** ✅ Atomic vocabulary, sleep learning from all programs, first library entries on ARC
- **Phase 10** ✅ Perception + parameterized architecture, truly atomic (41 prims), interleaved pipeline
- **Phase 11** 🔧 Deeper composition discovery, expressiveness gap (atomic → compound ops)
- **Phase 12** Cross-domain library transfer
- **Phase 13** Continuous mixed-domain learning

## License

MIT
