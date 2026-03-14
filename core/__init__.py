"""
agi-core: The Universal Learning Loop.

Public API — import everything from here.
"""

from .types import (
    Primitive,
    Program,
    Observation,
    Task,
    Decomposition,
    ScoredProgram,
    LibraryEntry,
)
from .interfaces import (
    Environment,
    Grammar,
    DriveSignal,
    Memory,
    DomainAdapter,
)
from .config import (
    SearchConfig,
    SleepConfig,
    CurriculumConfig,
)
from .results import (
    ParetoEntry,
    WakeResult,
    SleepResult,
    RoundResult,
)
from .transition_matrix import TransitionMatrix
from .learner import Learner
from .memory import InMemoryStore
from .metrics import (
    CompoundingMetrics,
    extract_metrics,
    print_compounding_table,
    save_metrics_json,
    save_metrics_csv,
)

# Runner symbols are lazy-loaded from common.benchmark to avoid circular
# imports when common/ is the entry point (python -m common).
_RUNNER_SYMBOLS = {
    "ExperimentConfig", "ExperimentResult", "run_experiment", "make_parser",
    "resolve_from_preset", "PRESETS", "ProgressTracker", "TeeWriter",
    "parse_human_int", "fmt_duration",
}


def __getattr__(name):
    if name in _RUNNER_SYMBOLS:
        import common.benchmark as _bm
        val = getattr(_bm, name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module 'core' has no attribute {name!r}")


__all__ = [
    # Data types
    "Primitive", "Program", "Observation", "Task", "Decomposition", "ScoredProgram", "LibraryEntry",
    # Interfaces
    "Environment", "Grammar", "DriveSignal", "Memory", "DomainAdapter",
    # Config
    "SearchConfig", "SleepConfig", "CurriculumConfig",
    # Results
    "ParetoEntry", "WakeResult", "SleepResult", "RoundResult",
    # Transition matrix
    "TransitionMatrix",
    # Learner
    "Learner",
    # Memory
    "InMemoryStore",
    # Metrics
    "CompoundingMetrics", "extract_metrics", "print_compounding_table",
    "save_metrics_json", "save_metrics_csv",
    # Runner (lazy-loaded from common.benchmark)
    "ExperimentConfig", "ExperimentResult", "run_experiment", "make_parser",
    "resolve_from_preset", "PRESETS", "ProgressTracker", "TeeWriter",
    "parse_human_int", "fmt_duration",
]
