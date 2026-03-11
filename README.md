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
# Clone and install (NumPy is the only runtime dependency)
git clone https://github.com/vibhor-77/agi-core.git
cd agi-core
pip install -r requirements.txt

# Clone the ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI

# Reproduce our results — one command does train + eval with culture transfer
python -m experiments.phase1_arc

# Run the test suite (380 tests, ~1 second)
python -m pytest tests/ -v
```

The default command runs all 400 training tasks, saves the learned culture, then runs all 400 evaluation tasks using that culture. Results, logs, and culture snapshots are auto-saved with timestamps. Output file paths are printed at the start so you can `tail -f` them in another terminal.

**Requirements:** Python 3.10+, NumPy.

### Syncing an existing clone

```bash
cd agi-core && git pull origin main
pip install -r requirements.txt
cd data/ARC-AGI && git pull && cd ../..
```

## Usage

```bash
# Full pipeline: train → save culture → eval (default, recommended)
python -m experiments.phase1_arc

# Quick subset for development (fast iteration, still runs full pipeline)
python -m experiments.phase1_arc --mode quick

# Train only, save culture for later
python -m experiments.phase1_arc --train-only
python -m experiments.phase1_arc --train-only --save-culture my_culture.json

# Eval only with a pre-trained culture file
python -m experiments.phase1_arc --eval-only --culture runs/20260311_120000_phase1_train_culture.json

# Single-process for debugging
python -m experiments.phase1_arc --workers 1
```

### Other demos (no dataset needed)

These demonstrate the **same invariant core algorithm** on different domains:

```bash
# Symbolic regression — discover mathematical formulas (y=2x+1, y=x², y=sin(x)+x, ...)
python -m domains.symbolic_math

# ARC with built-in sample tasks (rotate, mirror, crop, gravity, fill, ...)
python -m experiments.phase1_arc --mode quick
```

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

Monitor a running benchmark in another terminal:
```bash
tail -f runs/*_phase1_train.jsonl    # watch task results as they complete
tail -f runs/*_phase1_train.log      # watch full console output
```

### Verifying individual solves

The console output ends with a **SOLVED TASKS** section listing every solved task and its program. Each line shows the task ID, program, and test status:

```
  SOLVED TASKS (95 total)
    ✓ 007bbfb7                 program: upscale_pattern [test:✓]
    ✓ 00d62c1b                 program: fill_rect_interior_4 [test:✓]
    ...
```

To verify a specific solve, look up the task in the ARC dataset and trace the program:

```bash
# View the original task
cat data/ARC-AGI/data/training/007bbfb7.json | python -m json.tool

# Search for a task in the JSONL results
grep "007bbfb7" runs/*_phase1_train.jsonl | python -m json.tool

# Search in the final JSON
python -c "import json; d=json.load(open('runs/TIMESTAMP_phase1_train.json')); print(json.dumps(d['tasks']['007bbfb7'], indent=2))"
```

Each task record (JSONL and JSON) includes: `task_id`, `solved`, `test_solved`, `test_error`, `energy`, `prediction_error`, `program`, `evaluations`, `wall_time`.

### Presets

Three modes. Pick one. That's the only knob most users need.

| Mode | Rounds | Beam | Gens | ~Evals/task | Use case |
|------|--------|------|------|-------------|----------|
| `quick` | 1 | 30 | 15 | ~450 | Fast iteration |
| `default` | 1 | 80 | 40 | ~3,200 | Balanced speed/accuracy |
| `contest` | 1 | 250 | 100 | ~25,000 | Maximum accuracy |

Compute budget = beam × gens. Early stopping saves unused compute on easy tasks.

### Expected performance

Benchmarked with 4 parallel workers (x86_64). Times scale inversely with workers.

| Mode | 400 training tasks | 400 eval tasks (culture transfer) |
|------|-------------------|----------------------------------|
| `quick` | **95/400 (23.8%)**, median 3.1s/task, ~32 min total | **33/400 (8.2%)** |
| `default` | ~25-28%, ~1 hr | ~10% |
| `contest` | higher, ~3-4 hr | TBD |

**317 primitives** including grid partitioning, object decomposition, symmetry completion, connected components, diagonal ops, and per-object conditional recoloring.
**Depth-3 exhaustive enumeration** with smart pool selection finds 1-4 step programs efficiently.
**Object decomposition** automatically detects per-object transform patterns and recolors by size, shape, or position.

The key metric is whether solve rate **increases across rounds** as the library grows — that validates compounding.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `default` | Preset: `quick`, `default`, or `contest` |
| `--data-dir` | auto-detect | Path to ARC-AGI data directory |
| `--train-only` | off | Train only, no eval phase |
| `--eval-only` | off | Eval only (requires `--culture`) |
| `--culture` | none | Culture file to load (cross-run knowledge transfer) |
| `--save-culture` | auto | Override auto culture save path |
| `--max-tasks` | from preset | Limit tasks (0 = all) |
| `--rounds` | from preset | Wake-sleep rounds |
| `--beam-width` | from preset | Candidates per generation |
| `--max-generations` | from preset | Generations per task |
| `--workers` | 0 (perf cores) | Parallel workers (0 = performance cores only) |
| `--seed` | 42 | Random seed (deterministic) |
| `--compute-cap` | 0 (unlimited) | Total eval budget. Accepts: `50M`, `500K`, `0` |
| `--runs-dir` | `runs` | Directory for all run artifacts |
| `--no-log` | off | Disable log file (console only) |

## How It Works

### The wake-sleep loop

1. **WAKE**: For each task, search for a program that transforms input to output.
   Uses beam search with mutation and crossover over a library of primitives.
   - **Semantic deduplication** removes algebraically-equivalent programs from the beam (e.g. `cos(π/2+x²)` = `sin(x²)`) by hashing output vectors.
   - **Pareto front tracking** records the best program at each complexity level, showing the full accuracy-complexity tradeoff.
   - **Constant optimization** (symbolic math only): after structural mutations, fits constants via `scipy.optimize.minimize` (Nelder-Mead), decoupling structure search from coefficient search.
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

The core loop (`core/learner.py`) depends **only** on these interfaces. It never imports anything domain-specific. This is the "one algorithm" claim — the same loop works for grid puzzles, symbolic math, text adventures, and (eventually) robotics.

### The key metric: the compounding curve

```
Round  Solved     Rate  Library  New  Avg Energy   Wake(s)  Sleep(s)
---------------------------------------------------------------------
    1    2/4     50.0%        3    3      0.0012       4.2       0.1
    2    3/4     75.0%        5    2      0.0008       3.8       0.1
    3    3/4     75.0%        6    1      0.0005       3.1       0.0
```

If solve rate increases across rounds without new hand-coded primitives, the framework is working.

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
│   └── phase1_arc.py        # ARC curriculum training (dataset loading + ARC wiring)
│
├── domains/                 # Domain implementations (all 4 interfaces)
│   ├── arc/                 # ARC-AGI grid transformations (317 primitives)
│   │   ├── primitives.py    # Grid→Grid transform functions + registry
│   │   ├── objects.py       # Connected component detection
│   │   ├── environment.py   # ARCEnv
│   │   ├── grammar.py       # ARCGrammar
│   │   ├── drive.py         # ARCDrive
│   │   └── dataset.py       # Task loading + sample tasks
│   └── symbolic_math/       # 1D symbolic regression (15 math primitives)
│       └── __init__.py      # All 4 interfaces in one file
│
├── tests/                   # Test suite (380 tests)
│   ├── test_arc.py
│   ├── test_exhaustive_enum.py
│   ├── test_interfaces.py
│   ├── test_learner.py
│   ├── test_memory.py
│   ├── test_metrics.py
│   └── test_symbolic_math.py
│
├── runs/                    # Run artifacts — timestamped, git-ignored
├── data/                    # External datasets (git-ignored)
│
├── CLAUDE.md                # Persistent instructions for Claude Code sessions
├── PROMPTS.md               # Chronological log of all prompts
├── DECISIONS.md             # Chronological log of all decisions
├── requirements.txt         # Python dependencies (numpy, scipy, pytest)
└── README.md                # This file
```

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=core --cov=domains --cov-report=term-missing
```

## Documentation

- **[PROMPTS.md](PROMPTS.md)** — Every instruction given to Claude, in chronological order
- **[DECISIONS.md](DECISIONS.md)** — Every technical decision, rationale, and result
- **[CLAUDE.md](CLAUDE.md)** — Persistent rules for Claude Code sessions

These documents allow anyone to reproduce the exact trajectory of this project.

## Roadmap

- **Phase 0** ✅ Extract invariant core with pluggable interfaces
- **Phase 1** 🔧 ARC-AGI-1 training, curriculum style (317 primitives, beam search, wake-sleep)
- **Phase 2** ARC-AGI-1 eval, zero-shot transfer
- **Phase 3** Second domain (Zork), same core, cold start
- **Phase 4** Cross-domain library transfer
- **Phase 5** ARC-AGI-2
- **Phase 6** Continuous mixed-domain learning

## License

MIT
