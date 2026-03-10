# agi-core: The Universal Learning Loop

**One algorithm. Pluggable domains. Compounding intelligence.**

```
WAKE:   observe → hypothesize → execute → score → store
SLEEP:  analyze solutions → extract recurring sub-programs → compress → add to library
REPEAT: library grows → search shrinks → harder problems → compounding
```

## Prerequisites

- **Python 3.10+** (tested on 3.11)
- **Git**
- **pip**

## Setup

```bash
# 1. Clone this repository
git clone https://github.com/vibhor-77/agi-core.git
cd agi-core

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone the ARC-AGI dataset (auto-detected by the experiment runner)
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI

# 5. Verify everything works
python -m pytest tests/ -v
```

### Syncing an existing clone

```bash
cd agi-core
git pull origin main
pip install -r requirements.txt
cd data/ARC-AGI && git pull && cd ../..
```

## Quick start

```bash
# Just run it. No flags needed.
# Auto-detects ARC data, uses all CPU cores, sensible defaults.
python -m experiments.phase1_arc
```

If the ARC dataset is cloned (step 4 above), it runs on all 400 training tasks.
If not, it falls back to built-in sample tasks.

Other demos (no dataset needed):

```bash
python -m grammars.symbolic_math   # symbolic regression demo
python -m grammars.arc             # ARC sample tasks directly
```

## Presets

Three modes. Pick one. That's the only knob most users need.

| Mode | Rounds | Beam | Gens | ~Evals/task | Use case |
|------|--------|------|------|-------------|----------|
| `quick` | 2 | 80 | 40 | 3,200 | Fast iteration |
| `default` | 3 | 150 | 80 | 12,000 | Balanced speed/accuracy |
| `contest` | 10 | 500 | 200 | 100,000 | Maximum accuracy |

Compute budget = beam × gens. Early stopping saves unused compute on easy tasks.

```bash
python -m experiments.phase1_arc                  # default
python -m experiments.phase1_arc --mode quick     # fast dev loop
python -m experiments.phase1_arc --mode contest   # full benchmark
```

### Expected performance

Benchmarked on 4 CPU cores (x86_64). Times scale linearly with tasks, inversely with cores.

| Mode | Sample tasks (8) | 50 real tasks | 400 real tasks (full) |
|------|-----------------|---------------|----------------------|
| `quick` | 8/8, ~5s | ~0/50, ~13s | ~2-4%, ~2 min |
| `default` | 8/8, ~50s | 2/50 (4%), ~3 min | ~4%, ~10 min |
| `contest` | 8/8, ~3 min | — | ~5-8%, ~1 hr |

**On M1 Max (10 cores):** roughly 2.5x faster than the times above.

The 4% solve rate on real ARC tasks is the Phase 1 baseline. The key metric is whether
solve rate **increases across rounds** as the library grows — that validates compounding.

### Overriding individual parameters

Any flag overrides the preset:

```bash
python -m experiments.phase1_arc --max-tasks 50         # subset
python -m experiments.phase1_arc --mode contest --rounds 5
python -m experiments.phase1_arc --workers 1             # sequential
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `default` | Preset: `quick`, `default`, or `contest` |
| `--data-dir` | auto-detect | Path to ARC-AGI training directory |
| `--max-tasks` | from preset | Limit tasks (0 = all) |
| `--rounds` | from preset | Wake-sleep rounds |
| `--beam-width` | from preset | Candidates per generation |
| `--max-generations` | from preset | Generations per task |
| `--workers` | 0 (all cores) | Parallel workers |
| `--seed` | 42 | Random seed |
| `--output-dir` | `experiments/results` | Output directory |
| `--verbose` | off | Debug logging |

### Live monitoring

While a run is in progress:

```bash
# Watch per-task results as they complete
tail -f experiments/results/phase1_progress.jsonl | python -m json.tool
```

## Running tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=core --cov=grammars --cov-report=term-missing
```

## How it works

### The wake-sleep loop

1. **WAKE**: For each task, search for a program that transforms input to output.
   Uses beam search with mutation and crossover over a library of primitives.
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

## Output files

Each run produces:

| File | Contents |
|------|----------|
| `phase1_metrics.json` | Compounding curve data (solve rate, library size per round) |
| `phase1_metrics.csv` | Same, as CSV for plotting |
| `phase1_progress.jsonl` | Per-task results — one JSON object per line, live-tail friendly |
| `phase1_config.json` | Full run configuration for reproducibility |
| `phase1_library.json` | Learned abstractions (the "culture" — carries across runs) |

## Structure

```
agi-core/
├── core/                    # THE INVARIANT LOOP — never changes per domain
│   ├── __init__.py          # Public API
│   ├── interfaces.py        # 4 abstract interfaces
│   ├── learner.py           # Wake-sleep loop + transition matrix prior
│   ├── memory.py            # Default in-memory store
│   └── metrics.py           # Compounding curve measurement
│
├── grammars/                # PLUGGABLE — one per domain
│   ├── symbolic_math.py     # 1D symbolic regression
│   ├── arc.py               # ARC-AGI grid transformations (48 primitives)
│   └── zork.py              # TODO: text adventure actions
│
├── experiments/             # Experiment scripts and results
│   └── phase1_arc.py        # Phase 1 curriculum training runner
│
├── tests/                   # Test suite (16 tests)
│   └── test_arc.py
│
├── data/                    # External datasets (git-ignored)
│   └── ARC-AGI/
│
├── CLAUDE.md                # Persistent instructions for Claude Code sessions
├── PROMPTS.md               # Chronological log of all prompts
├── DECISIONS.md             # Chronological log of all decisions
├── requirements.txt         # Python dependencies (numpy, pytest)
└── README.md                # This file
```

## Documentation

- **[PROMPTS.md](PROMPTS.md)** — Every instruction given to Claude, in chronological order
- **[DECISIONS.md](DECISIONS.md)** — Every technical decision, rationale, and result
- **[CLAUDE.md](CLAUDE.md)** — Persistent rules for Claude Code sessions

These documents allow anyone to reproduce the exact trajectory of this project.

## Roadmap

- **Phase 0** ✅ Extract invariant core with pluggable interfaces
- **Phase 1** 🔧 ARC-AGI-1 training, curriculum style (48 primitives, beam search, wake-sleep)
- **Phase 2** ARC-AGI-1 eval, zero-shot transfer
- **Phase 3** Second domain (Zork), same core, cold start
- **Phase 4** Cross-domain library transfer
- **Phase 5** ARC-AGI-2
- **Phase 6** Continuous mixed-domain learning

## License

MIT
