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
]
