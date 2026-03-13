"""
Generic experiment runner for any domain.

This is part of the invariant core. It provides:
- TeeWriter (stdout → console + log file)
- ProgressTracker (live per-task streaming + scoreboard)
- Signal handling (Ctrl-C kills immediately)
- Presets (quick/default/contest)
- Output formatting and results saving

Domain-specific scripts only need to provide tasks + the 4 interfaces.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import TextIOBase

from .types import Task
from .interfaces import Environment, Grammar, DriveSignal
from .config import SearchConfig, SleepConfig, CurriculumConfig
from .results import WakeResult
from .learner import Learner
from .memory import InMemoryStore
from .metrics import extract_metrics, print_compounding_table, save_metrics_json, save_metrics_csv


# =============================================================================
# Presets — the ONLY knob most users need
# =============================================================================

PRESETS = {
    # Quick: fast dev loop. Aggressive cap for speed.
    # Experiments on 400-task training set (this machine, 2 workers):
    #   - cap=100:  76/400 (19.0%) in 72s  — 89% of all possible solves
    #   - cap=3M:   85/400 (21.2%) in 379s — all solves, 5.3x slower
    #   - Solves are bimodal: 76 "fast" (<500 evals, depth 1-2) vs 9 "slow"
    #     (per_object_recolor, 12-14K evals, need cap ≥ 2.8M).
    #   - The 500-eval floor means caps below 400K all give identical results.
    #   - Philosophy: quick mode optimizes for iteration speed, not max solves.
    "quick": {
        "rounds": 1,
        "beam_width": 1,
        "max_generations": 1,
        "max_tasks": 50,
        "compute_cap": 500_000,   # 500K: same solves as 100, small headroom, ~5x faster than 3M
    },
    # Default: full dataset. Higher cap to capture per_object_recolor solves.
    # 3M cap (3750 base evals/task, up to 15K for small grids) catches all
    # current solves including per_object_recolor tasks.
    "default": {
        "rounds": 1,
        "beam_width": 1,
        "max_generations": 1,
        "max_tasks": 0,
        "compute_cap": 3_000_000,   # 3M: captures all 85 solves on 400 tasks
    },
    # Contest: maximum effort. Keeps modest beam in case deeper search
    # helps on the hardest tasks. Still mainly exhaustive.
    "contest": {
        "rounds": 1,
        "beam_width": 30,
        "max_generations": 15,
        "max_tasks": 0,
        "compute_cap": 100_000_000,  # 100M ops — beam search safety net
    },
}

DEFAULT_SEED = 42
DEFAULT_RUNS_DIR = "runs"


# =============================================================================
# Tee writer — duplicates stdout to both console and a log file
# =============================================================================

class TeeWriter(TextIOBase):
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str, original_stdout):
        super().__init__()
        self._original = original_stdout
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log = open(log_path, "w", buffering=1)

    def write(self, s):
        self._original.write(s)
        self._log.write(s)
        return len(s)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    @property
    def encoding(self):
        return self._original.encoding


# =============================================================================
# Formatting helpers
# =============================================================================

def hline(char="─", width=72):
    print(char * width)


def fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '12.3s'."""
    s = seconds
    if s >= 3600:
        return f"{int(s) // 3600}h{(int(s) % 3600) // 60:02d}m"
    if s >= 60:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"
    return f"{s:.1f}s"


# =============================================================================
# Machine auto-detection
# =============================================================================

def detect_machine() -> dict:
    """Detect machine characteristics for metadata."""
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "performance_cores": Learner.performance_core_count(),
        "python": platform.python_version(),
    }
    if info["platform"] == "Darwin" and info["arch"] == "arm64":
        info["chip"] = "Apple Silicon"
    return info


# =============================================================================
# Signal handling — Ctrl-C kills the whole process tree immediately
# =============================================================================

_interrupted = False


def _handle_sigint(signum, frame):
    """First Ctrl-C raises KeyboardInterrupt. Second force-kills."""
    global _interrupted
    if _interrupted:
        os.kill(os.getpid(), signal.SIGKILL)
    _interrupted = True
    raise KeyboardInterrupt


def install_signal_handler():
    """Install the Ctrl-C signal handler. Call this early in main()."""
    signal.signal(signal.SIGINT, _handle_sigint)


# =============================================================================
# Human-readable number parsing
# =============================================================================

def parse_human_int(s: str) -> int:
    """Parse a human-readable integer with optional suffix or commas.

    Examples: "50M" → 50_000_000, "500K" → 500_000, "8,000,000" → 8_000_000
    """
    s = s.strip().replace(",", "").replace("_", "")
    if not s:
        raise argparse.ArgumentTypeError("empty value")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
    last = s[-1].lower()
    if last in suffixes:
        try:
            return int(float(s[:-1]) * suffixes[last])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid number: {s!r}. Examples: 50M, 8,000,000, 500K, 0")

    try:
        return int(float(s))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid number: {s!r}. Examples: 50M, 8,000,000, 500K, 0")


# =============================================================================
# Standard argument parser (reusable by any domain)
# =============================================================================

def make_parser(description: str, domain_name: str = "experiment") -> argparse.ArgumentParser:
    """Create a standard argument parser with presets + common flags."""
    perf_cores = Learner.performance_core_count()

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"""
Presets (the only knob most users need):
  quick     Fast dev loop (~25s, 50 tasks, 500K compute cap)
  default   Full dataset (all tasks, 3M compute cap)
  contest   Maximum accuracy (all tasks, 100M compute cap, beam search)

Compute cap examples:
  --compute-cap 50M          # 50 million (cell-normalized per task)
  --compute-cap 500K         # 500 thousand
  --compute-cap 0            # unlimited (no cap)

Examples:
  python -m experiments.{domain_name}                    # sensible defaults
  python -m experiments.{domain_name} --mode quick       # fast iteration
  python -m experiments.{domain_name} --mode contest     # full benchmark
""",
    )
    parser.add_argument("--mode", type=str, default="default",
                        choices=list(PRESETS.keys()), help="Preset configuration")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (0 = all, default: from preset)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Wake-sleep rounds (default: 1)")
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Beam width (default: from preset)")
    parser.add_argument("--max-generations", type=int, default=None,
                        help="Max generations per task (default: from preset)")
    parser.add_argument("--workers", type=int, default=0,
                        help=f"Parallel workers (0 = performance cores = {perf_cores})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--compute-cap", type=parse_human_int,
                        default=str(0),
                        help="Override preset compute cap (0=use preset default). Accepts: 50M, 500K, 0")
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Directory for all run artifacts")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable log file (console only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    parser.add_argument("--exhaustive-depth", type=int, default=3,
                        help="Exhaustive enumeration depth (0=disabled, 2=pairs, 3=triples)")
    parser.add_argument("--exhaustive-pair-top-k", type=int, default=40,
                        help="Top-K singles for pair exhaustion (default 40)")
    parser.add_argument("--exhaustive-triple-top-k", type=int, default=15,
                        help="Top-K singles for triple exhaustion (default 15)")
    parser.add_argument("--sequential-compounding", action="store_true",
                        help="Process tasks sequentially with immediate concept promotion")
    parser.add_argument("--compounding", action="store_true",
                        help="Enable compounding mode: depth-2 exhaustive + 3 rounds + "
                             "sequential compounding + relaxed library promotion. "
                             "Forces the system to rely on library for depth-3+ solutions.")
    parser.add_argument("--culture", type=str, default="",
                        help="Path to culture file to load (cross-run knowledge transfer)")
    parser.add_argument("--save-culture", type=str, default="",
                        help="Override auto culture save path (e.g. culture_train.json)")
    parser.add_argument("--task-ids", type=str, default="",
                        help="Comma-separated task IDs to run (e.g. '0dfd9992,1190e5a7'). "
                             "Overrides --max-tasks. Prefix match supported.")
    parser.add_argument("--adaptive-realloc", action="store_true",
                        help="Re-run near-miss tasks with boosted compute budget "
                             "(3x budget, wider search breadth)")
    return parser


# =============================================================================
# Live progress tracker with scoreboard
# =============================================================================

class ProgressTracker:
    """Live progress tracker with rolling scoreboard.

    Streams per-task results to console + JSONL file. Prints a rolling
    scoreboard summary every N tasks.
    """

    SCOREBOARD_INTERVAL = 10

    def __init__(self, jsonl_path: str, t0: float, split_label: str = ""):
        self._file = open(jsonl_path, "w")
        self._t0 = t0
        self._split_label = split_label  # e.g. "TRAIN" or "EVAL"

        # Cumulative (across all rounds)
        self.done = 0
        self.solved = 0
        self.overfit = 0
        self.total_evals = 0
        self.total_gens = 0
        self.scores: list[float] = []
        self.times: list[float] = []
        self.all_records: list[dict] = []

        # Per-round tracking (reset each round)
        self._current_round = -1
        self._round_done = 0
        self._round_solved = 0
        self._round_overfit = 0

    def on_task_done(
        self, round_num: int, task_index: int, total_tasks: int, wr: WakeResult,
    ) -> None:
        """Called after each task completes. Streams live output."""
        # Reset per-round counters on new round
        if round_num != self._current_round:
            self._current_round = round_num
            self._round_done = 0
            self._round_solved = 0
            self._round_overfit = 0

        self.done += 1
        self._round_done += 1
        if wr.solved:  # test-verified (falls back to train when no test data)
            self.solved += 1
            self._round_solved += 1
        elif wr.train_solved and wr.test_solved is False:
            self.overfit += 1
            self._round_overfit += 1
        self.total_evals += wr.evaluations
        self.total_gens += wr.generations_used
        if wr.best:
            self.scores.append(wr.best.energy)
        self.times.append(wr.wall_time)

        elapsed = time.time() - self._t0
        icon = "✓" if wr.solved else "✗"
        energy_str = f"{wr.best.energy:.4f}" if wr.best else "    N/A"
        program_str = repr(wr.best.program) if wr.best else ""

        # Overfit tag: matched training but failed test
        overfit_tag = ""
        if wr.train_solved and wr.test_solved is False:
            overfit_tag = " [overfit]"

        slow_tag = ""
        if len(self.times) >= 5:
            med = statistics.median(self.times)
            if wr.wall_time > med * 10:
                slow_tag = "  *** SLOW ***"

        split_tag = f" {self._split_label}" if self._split_label else ""
        print(
            f"  {icon}{split_tag} R{round_num} [{task_index:>3}/{total_tasks}] "
            f"{wr.task_id:<20s} "
            f"E={energy_str}  gens={wr.generations_used:<4d} "
            f"evals={wr.evaluations:<6d} {wr.wall_time:.1f}s"
            f"{overfit_tag}{slow_tag}",
            flush=True,
        )
        if wr.solved and program_str:
            print(f"       program: {program_str}")
        overfit_str = f"  overfit={self._round_overfit}" if self._round_overfit else ""
        print(
            f"       R{round_num}: solved={self._round_solved}/{self._round_done}"
            f"{overfit_str}  "
            f"evals={self.total_evals:,}  "
            f"[{fmt_duration(elapsed)} elapsed]",
        )

        record = {
            "round": round_num,
            "task_index": task_index,
            "task_id": wr.task_id,
            "solved": wr.solved,           # test-verified (the real metric)
            "train_solved": wr.train_solved,
            "test_solved": wr.test_solved,
            "test_error": round(wr.test_error, 6) if wr.test_error is not None else None,
            "energy": wr.best.energy if wr.best else None,
            "prediction_error": wr.best.prediction_error if wr.best else None,
            "generations": wr.generations_used,
            "evaluations": wr.evaluations,
            "wall_time": round(wr.wall_time, 3),
            "program": repr(wr.best.program) if wr.best else None,
            "n_train_perfect": wr.n_train_perfect,
            "solving_rank": wr.solving_rank,
            "elapsed": round(elapsed, 1),
        }
        self.all_records.append(record)

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        if self.done % self.SCOREBOARD_INTERVAL == 0:
            self._print_scoreboard(round_num, total_tasks, elapsed)

    def _print_scoreboard(
        self, round_num: int, total_tasks: int, elapsed: float,
    ) -> None:
        rate = self.done / max(elapsed, 0.001)
        pending = total_tasks - (self.done % total_tasks or total_tasks)
        eta = pending / rate if rate > 0 else 0
        mean_energy = statistics.mean(self.scores) if self.scores else float("inf")
        med_time = statistics.median(self.times) if self.times else 0

        print()
        print(
            f"  ┌── Progress: {self.done} tasks  "
            f"{fmt_duration(elapsed)} elapsed  "
            f"ETA {fmt_duration(eta)}  "
            f"{rate:.1f} tasks/s ──"
        )
        overfit_str = (f"  ~ overfit={self._round_overfit}"
                       if self._round_overfit else "")
        unsolved = self._round_done - self._round_solved - self._round_overfit
        print(
            f"  │  R{round_num}: ✓ solved={self._round_solved}/{self._round_done}  "
            f"✗ unsolved={unsolved}/{self._round_done}"
            f"{overfit_str}"
        )
        print(
            f"  │  Energy: mean={mean_energy:.4f}  "
            f"Time: median={med_time:.1f}s  "
            f"Evals: {self.total_evals:,}"
        )
        if len(self.all_records) >= 5:
            by_time = sorted(self.all_records, key=lambda r: -r["wall_time"])
            top3 = by_time[:3]
            slowest_str = "  ".join(
                f"{r['task_id'][:8]}({r['wall_time']:.1f}s)" for r in top3
            )
            print(f"  │  Slowest: {slowest_str}")
        print(f"  └{'─' * 60}")
        print()

    def close(self):
        self._file.close()


# =============================================================================
# The generic experiment runner
# =============================================================================

@dataclass
class ExperimentConfig:
    """Everything needed to run a domain-agnostic experiment."""
    title: str
    domain_tag: str  # short tag for output filenames, e.g. "arc", "symreg"
    tasks: list[Task]

    environment: Environment
    grammar: Grammar
    drive: DriveSignal

    # Search parameters (resolved from preset + overrides)
    rounds: int = 1
    beam_width: int = 150
    max_generations: int = 80
    workers: int = 0
    seed: int = DEFAULT_SEED
    compute_cap: int = 0

    # Search tuning
    mutations_per_candidate: int = 2
    crossover_fraction: float = 0.3
    energy_alpha: float = 1.0
    energy_beta: float = 0.001
    solve_threshold: float = 0.001

    # Sleep tuning
    min_occurrences: int = 2
    min_size: int = 2
    max_library_size: int = 500

    # Exhaustive enumeration
    exhaustive_depth: int = 3
    exhaustive_pair_top_k: int = 40
    exhaustive_triple_top_k: int = 15

    # Sequential compounding
    sequential_compounding: bool = False

    # Adaptive compute reallocation for near-miss tasks
    adaptive_realloc: bool = False

    # Culture file (load pre-trained library)
    culture_path: str = ""
    save_culture: str = ""  # override auto-generated culture output path

    # Output
    runs_dir: str = DEFAULT_RUNS_DIR
    no_log: bool = False
    verbose: bool = False
    mode: str = "default"

    # Task filtering
    task_ids: str = ""  # comma-separated task IDs (prefix match supported)

    # Shared timestamp (for pipeline mode — reuse across train+eval)
    timestamp: str = ""

    # Suppress result/metric file output (for pipeline mode where the
    # pipeline script writes its own combined output files)
    suppress_files: bool = False


def resolve_from_preset(args, preset: dict) -> dict:
    """Resolve argument values: explicit args override preset defaults."""
    # CLI --compute-cap > 0 overrides preset; 0 means "use preset default"
    preset_cap = preset.get("compute_cap", 0)
    args_cap = getattr(args, "compute_cap", 0)
    explicit_cap = args_cap if args_cap > 0 else preset_cap
    return {
        "rounds": args.rounds if args.rounds is not None else preset["rounds"],
        "beam_width": args.beam_width if args.beam_width is not None else preset["beam_width"],
        "max_generations": args.max_generations if args.max_generations is not None else preset["max_generations"],
        "max_tasks": args.max_tasks if args.max_tasks is not None else preset["max_tasks"],
        "workers": args.workers if args.workers > 0 else Learner.performance_core_count(),
        "compute_cap": explicit_cap,
    }


@dataclass
class ExperimentResult:
    """Result returned by run_experiment for pipeline chaining."""
    culture_path: str
    results_path: str
    jsonl_path: str
    results_data: dict  # the full results dict (meta + summary + tasks + library)


def _ensure_deterministic_hashing():
    """Ensure PYTHONHASHSEED=0 for reproducible results.

    Python randomizes string hashing by default (PYTHONHASHSEED=random),
    which makes dict/set iteration order non-deterministic for string keys.
    This causes beam search results to vary between runs even with the
    same seed, because semantic dedup and program ranking depend on
    dict ordering when energies are tied.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        import subprocess
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        # Must restart the process for PYTHONHASHSEED to take effect
        # (it's read at interpreter startup, so mid-process changes don't help).
        # Relaunch using -m with the module name derived from the script path,
        # since the original -m context is lost in sys.argv.
        print("  Re-launching with PYTHONHASHSEED=0 for reproducibility...")
        script = sys.argv[0]
        cwd = os.getcwd()
        # Convert absolute script path to module name (relative to CWD)
        if os.path.isabs(script):
            script = os.path.relpath(script, cwd)
        module = script.replace(".py", "").replace(os.sep, ".")
        result = subprocess.run(
            [sys.executable, "-m", module] + sys.argv[1:], env=env
        )
        sys.exit(result.returncode)


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run a complete wake-sleep experiment on any domain.

    Returns an ExperimentResult with culture path, results path, and data.

    This is the top-level entry point. It:
    1. Sets up output files (log, jsonl, json, metrics, library)
    2. Tees stdout to log file
    3. Creates the Learner with the domain's 4 interfaces
    4. Runs the curriculum with live progress tracking
    5. Prints final results and saves all artifacts
    """
    _ensure_deterministic_hashing()
    install_signal_handler()

    run_timestamp = cfg.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{cfg.domain_tag}_{run_timestamp}"

    runs_dir = cfg.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    log_path = os.path.join(runs_dir, f"{prefix}.log")
    jsonl_path = os.path.join(runs_dir, f"{prefix}.jsonl")
    results_path = os.path.join(runs_dir, f"{prefix}.json")
    library_path = os.path.join(runs_dir, f"{prefix}_library.json")
    metrics_json_path = os.path.join(runs_dir, f"{prefix}_metrics.json")
    metrics_csv_path = os.path.join(runs_dir, f"{prefix}_metrics.csv")

    culture_path = (cfg.save_culture if cfg.save_culture
                     else library_path.replace("_library.json", "_culture.json"))

    tee = None
    if not cfg.no_log and not cfg.suppress_files:
        tee = TeeWriter(log_path, sys.stdout)
        sys.stdout = tee

    results_data = {}
    try:
        results_data = _run_experiment(cfg, run_timestamp, log_path, jsonl_path,
                                       results_path, library_path,
                                       metrics_json_path, metrics_csv_path,
                                       culture_path)
    except KeyboardInterrupt:
        print("\n\nAborted by user — partial results above.\n")
        raise  # propagate so pipeline mode stops too
    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()

    return ExperimentResult(
        culture_path=culture_path,
        results_path=results_path,
        jsonl_path=jsonl_path,
        results_data=results_data,
    )


def _run_experiment(cfg, run_timestamp, log_path, jsonl_path, results_path,
                    library_path, metrics_json_path, metrics_csv_path,
                    culture_path):
    """Core run logic, separated so tee cleanup always happens."""
    machine = detect_machine()
    workers = cfg.workers if cfg.workers > 0 else Learner.performance_core_count()
    rounds = cfg.rounds
    beam_width = cfg.beam_width
    max_gens = cfg.max_generations
    tasks = cfg.tasks
    compute_cap = cfg.compute_cap

    # Filter by task IDs if specified (prefix match)
    if cfg.task_ids:
        id_prefixes = [t.strip() for t in cfg.task_ids.split(",") if t.strip()]
        tasks = [t for t in tasks
                 if any(t.task_id.startswith(p) for p in id_prefixes)]
        if not tasks:
            print(f"  ERROR: No tasks matched --task-ids '{cfg.task_ids}'")
            sys.exit(1)

    # --- Header ---
    hline("═")
    print(f"  {cfg.title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hline("═")

    print(f"\n  Mode:       {cfg.mode}")
    print(f"  Rounds:     {rounds}")
    print(f"  Beam:       {beam_width}")
    print(f"  Gens:       {max_gens}")
    print(f"  Workers:    {workers} / {machine['cpu_count']} cores "
          f"({machine.get('chip', machine['arch'])})")
    print(f"  Seed:       {cfg.seed}")
    if compute_cap > 0:
        print(f"  Compute cap: {compute_cap:,} ops (cell-normalized)")
    else:
        print(f"  Compute cap: unlimited")

    print()
    hline("─")
    print("  Output files (available now for tail -f):")
    hline("─")
    print(f"  Results (live):   {jsonl_path}")
    if not cfg.suppress_files:
        print(f"  Results (final):  {results_path}")
        print(f"  Metrics:          {metrics_json_path}")
        print(f"  Library:          {library_path}")
        if not cfg.no_log:
            print(f"  Console log:      {log_path}")
    print()

    if not tasks:
        print("  ERROR: No tasks loaded.")
        sys.exit(1)

    # --- Create Learner ---
    memory = InMemoryStore()

    evals_per_task = beam_width * max_gens
    total_budget = evals_per_task * len(tasks) * rounds

    # Cell-normalized compute cap (from agi-mvp-general).
    # Instead of globally reducing max_gens, compute a per-task eval budget
    # proportional to grid size. Small grids get more evals (they're cheap),
    # large grids get fewer. Budget is enforced per-task in wake_on_task.
    #
    # Formula: min(max(compute_cap / avg_cells, 500), compute_cap / DEFAULT_CELLS)
    #   DEFAULT_CELLS = 800 (median ARC grid size)
    #   Floor of 500 evals ensures even huge grids get basic search.
    eval_budget = 0  # 0 = unlimited (passed per-task via SearchConfig)
    DEFAULT_CELLS = 800
    if compute_cap > 0:
        # Per-task budget uses median grid size as baseline
        max_evals = max(compute_cap // DEFAULT_CELLS, 500)
        eval_budget = max_evals  # default for tasks without grid info
        print(f"\n  Compute cap: {compute_cap:,} ops → ~{max_evals:,} evals/task (cell-normalized)")

    # Load culture file if specified (cross-run knowledge transfer)
    if cfg.culture_path and os.path.isfile(cfg.culture_path):
        memory.load_culture(cfg.culture_path)
        print(f"\n  Culture loaded: {len(memory.get_library())} library entries, "
              f"{len(memory.get_solutions())} solutions")

    learner = Learner(
        environment=cfg.environment,
        grammar=cfg.grammar,
        drive=cfg.drive,
        memory=memory,
        search_config=SearchConfig(
            beam_width=beam_width,
            max_generations=max_gens,
            mutations_per_candidate=cfg.mutations_per_candidate,
            crossover_fraction=cfg.crossover_fraction,
            energy_alpha=cfg.energy_alpha,
            energy_beta=cfg.energy_beta,
            solve_threshold=cfg.solve_threshold,
            seed=cfg.seed,
            exhaustive_depth=cfg.exhaustive_depth,
            exhaustive_pair_top_k=cfg.exhaustive_pair_top_k,
            exhaustive_triple_top_k=cfg.exhaustive_triple_top_k,
            eval_budget=eval_budget,
        ),
        sleep_config=SleepConfig(
            min_occurrences=cfg.min_occurrences,
            min_size=cfg.min_size,
            max_library_size=cfg.max_library_size,
        ),
    )

    print(f"\n  Tasks:      {len(tasks)}")
    print(f"  Primitives: {len(cfg.grammar.base_primitives())}")
    print(f"  Budget:     ~{evals_per_task:,} evals/task, ~{total_budget:,} total")

    # --- Run ---
    # Derive split label from title for per-line output clarity
    _title_upper = cfg.title.upper()
    if "EVAL" in _title_upper:
        _split = "EVAL"
    elif "TRAIN" in _title_upper:
        _split = "TRAIN"
    else:
        _split = ""
    tracker = ProgressTracker(jsonl_path, time.time(), split_label=_split)

    hline("─")
    print(f"  Running {len(tasks)} tasks × {rounds} rounds on {workers} workers")
    hline("─")
    print()

    t0 = time.time()
    results = learner.run_curriculum(
        tasks,
        CurriculumConfig(
            sort_by_difficulty=cfg.sequential_compounding,  # sort for compounding, shuffle for parallel
            wake_sleep_rounds=rounds,
            workers=workers,
            sequential_compounding=cfg.sequential_compounding,
            adaptive_realloc=cfg.adaptive_realloc,
        ),
        on_task_done=tracker.on_task_done,
    )
    total_time = time.time() - t0
    tracker.close()

    # --- Final results ---
    metrics = extract_metrics(results)
    total_evals = sum(wr.evaluations for rr in results for wr in rr.wake_results)

    print()
    hline("═")
    print("  COMPOUNDING CURVE — THE KEY METRIC")
    hline("═")
    print_compounding_table(metrics)
    print()

    # Use a phase label if the title contains TRAINING or EVALUATION
    phase_label = ""
    title_upper = cfg.title.upper()
    if "TRAINING" in title_upper:
        phase_label = " — TRAINING"
    elif "EVALUATION" in title_upper:
        phase_label = " — EVALUATION"

    hline("═")
    print(f"  FINAL RESULTS{phase_label}")
    hline("═")

    # Use last round's metrics for the headline numbers (not cumulative)
    last = metrics[-1] if metrics else None
    n_tasks = len(tasks)

    print(f"  Tasks:             {n_tasks}")
    if last:
        print(f"  ✓ Solved:          {last.tasks_solved}/{n_tasks}  "
              f"({last.solve_rate:.1%})")
        if last.train_solved > last.tasks_solved:
            overfits = last.train_solved - last.tasks_solved
            print(f"    (+ {overfits} overfit: matched training examples "
                  f"but failed held-out test)")

    print(f"  Rounds:            {rounds}")
    print(f"  Total evaluations: {total_evals:,}")
    if tracker.times:
        print(f"  Median task time:  {statistics.median(tracker.times):.2f}s")
    print(f"  Wall-clock time:   {fmt_duration(total_time)}")
    throughput = tracker.done / max(total_time, 0.001)
    print(f"  Throughput:        {throughput:.1f} tasks/s ({workers} workers)")

    if len(metrics) >= 2:
        first_rate = metrics[0].solve_rate
        last_rate = metrics[-1].solve_rate
        if last_rate > first_rate:
            print(f"\n  >>> COMPOUNDING DETECTED: {first_rate:.1%} → {last_rate:.1%}")
        elif last_rate == first_rate:
            print(f"\n  >>> PLATEAU: solve rate stayed at {first_rate:.1%}")
        else:
            print(f"\n  >>> REGRESSION: {first_rate:.1%} → {last_rate:.1%}")

    library = memory.get_library()
    print(f"\n  Library: {len(library)} learned abstractions")
    for entry in library[:20]:
        print(f"    {entry.name}: {entry.program} "
              f"(useful={entry.usefulness:.1f}, reused={entry.reuse_count}x, "
              f"from {len(entry.source_tasks)} tasks)")
    if len(library) > 20:
        print(f"    ... and {len(library) - 20} more")

    if len(tracker.all_records) >= 5:
        by_time = sorted(tracker.all_records, key=lambda r: -r["wall_time"])
        print(f"\n  Slowest tasks:")
        for r in by_time[:5]:
            icon = "✓" if r["solved"] else "✗"
            print(f"    {icon} {r['task_id']}  {r['wall_time']:.1f}s  "
                  f"evals={r['evaluations']:,}  E={r['energy']}")

    # --- Solved tasks summary (for verification) ---
    solved_records = [r for r in tracker.all_records if r["solved"]]
    overfit_records = [r for r in tracker.all_records
                       if r.get("train_solved") and not r["solved"]]
    if solved_records:
        print()
        hline("─")
        print(f"  SOLVED TASKS ({len(solved_records)} total)")
        hline("─")
        for r in sorted(solved_records, key=lambda r: r["task_id"]):
            print(f"    ✓ {r['task_id']:<24s} program: {r['program']}")
    if overfit_records:
        print()
        hline("─")
        print(f"  OVERFIT TASKS ({len(overfit_records)} matched training but failed test)")
        hline("─")
        for r in sorted(overfit_records, key=lambda r: r["task_id"]):
            err_str = f" err={r.get('test_error', '?')}" if r.get("test_error") else ""
            print(f"    ~ {r['task_id']:<24s} program: {r['program']}{err_str}")

    # --- Near misses (for debugging unsolved tasks) ---
    near_misses = [r for r in tracker.all_records
                   if not r["solved"] and not r.get("train_solved")
                   and r.get("prediction_error") is not None
                   and r["prediction_error"] < 0.1]
    if near_misses:
        near_misses.sort(key=lambda r: r["prediction_error"])
        print()
        hline("─")
        print(f"  NEAR MISSES ({len(near_misses)} tasks with error < 0.1)")
        hline("─")
        for r in near_misses[:20]:
            print(f"    ✗ {r['task_id']:<24s} err={r['prediction_error']:.4f}  "
                  f"program: {r['program']}")
        if len(near_misses) > 20:
            print(f"    ... and {len(near_misses) - 20} more")

    # --- Save artifacts ---
    results_data = {
        "meta": {
            "timestamp": run_timestamp,
            "datetime": datetime.now().isoformat(),
            "title": cfg.title,
            "domain": cfg.domain_tag,
            "mode": cfg.mode,
            "rounds": rounds,
            "beam_width": beam_width,
            "max_generations": max_gens,
            "evals_per_task": evals_per_task,
            "workers": workers,
            "seed": cfg.seed,
            "n_tasks": len(tasks),
            "n_primitives": len(cfg.grammar.base_primitives()),
            "compute_cap": compute_cap,
            "machine": machine,
        },
        "summary": {
            "n_tasks": len(tasks),
            "rounds": rounds,
            "last_round_solved": last.tasks_solved if last else 0,
            "last_round_solve_rate": round(last.solve_rate, 4) if last else 0,
            "last_round_train_solved": last.train_solved if last else 0,
            "last_round_train_solve_rate": round(last.train_solve_rate, 4) if last else 0,
            "total_task_instances": tracker.done,
            "mean_energy": round(statistics.mean(tracker.scores), 4) if tracker.scores else None,
            "total_evaluations": total_evals,
            "total_generations": tracker.total_gens,
            "median_task_time": round(statistics.median(tracker.times), 3) if tracker.times else None,
            "mean_task_time": round(statistics.mean(tracker.times), 3) if tracker.times else None,
            "wall_clock_seconds": round(total_time, 1),
            "throughput_tasks_per_sec": round(throughput, 2),
            "library_size": len(library),
            "compounding": [
                {
                    "round": m.round_number,
                    "solve_rate": round(m.solve_rate, 4),
                    "tasks_solved": m.tasks_solved,
                    "train_solve_rate": round(m.train_solve_rate, 4),
                    "train_solved": m.train_solved,
                    "tasks_total": m.tasks_total,
                    "library_size": m.library_size,
                    "new_abstractions": m.new_abstractions,
                    "avg_reuse": round(m.avg_reuse_per_entry, 2),
                    "avg_energy": round(m.avg_energy_of_solutions, 4) if m.avg_energy_of_solutions != float("inf") else None,
                    "wake_time": round(m.wall_time_wake, 1),
                    "sleep_time": round(m.wall_time_sleep, 1),
                }
                for m in metrics
            ],
        },
        "tasks": {
            r["task_id"]: {
                "round": r["round"],
                "solved": r["solved"],
                "train_solved": r.get("train_solved"),
                "test_solved": r.get("test_solved"),
                "test_error": r.get("test_error"),
                "energy": r["energy"],
                "prediction_error": r["prediction_error"],
                "generations": r["generations"],
                "evaluations": r["evaluations"],
                "wall_time": r["wall_time"],
                "program": r["program"],
                "n_train_perfect": r.get("n_train_perfect", 0),
                "solving_rank": r.get("solving_rank"),
            }
            for r in tracker.all_records
        },
        "all_records": tracker.all_records,
        "library": [
            {
                "name": entry.name,
                "program": repr(entry.program),
                "usefulness": round(entry.usefulness, 2),
                "reuse_count": entry.reuse_count,
                "source_tasks": list(entry.source_tasks),
            }
            for entry in library
        ],
    }

    if not cfg.suppress_files:
        # Exclude all_records from JSON (it's only for pipeline JSONL assembly)
        json_data = {k: v for k, v in results_data.items() if k != "all_records"}
        with open(results_path, "w") as f:
            json.dump(json_data, f, indent=2)

        save_metrics_json(metrics, metrics_json_path)
        save_metrics_csv(metrics, metrics_csv_path)

        # Save legacy library format
        memory.save(library_path)

    # Always save culture file (needed for pipeline eval transfer)
    memory.save_culture(culture_path)

    print()
    hline("─")
    print("  Artifacts:")
    hline("─")
    print(f"  Results (live):   {jsonl_path}")
    if not cfg.suppress_files:
        print(f"  Results (final):  {results_path}")
        print(f"  Metrics JSON:     {metrics_json_path}")
        print(f"  Metrics CSV:      {metrics_csv_path}")
        print(f"  Library:          {library_path}")
    print(f"  Culture:          {culture_path}")
    if not cfg.no_log:
        print(f"  Console log:      {log_path}")

    hline("═")
    print("  Done.")
    hline("═")
    print()

    return results_data
