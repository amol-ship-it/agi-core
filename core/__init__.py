"""
agi-core: The Universal Learning Loop.

Public API — import everything from here.
"""

from .interfaces import (
    Primitive,
    Program,
    Observation,
    Task,
    ScoredProgram,
    LibraryEntry,
    Environment,
    Grammar,
    DriveSignal,
    Memory,
)
from .learner import (
    Learner,
    SearchConfig,
    SleepConfig,
    CurriculumConfig,
    WakeResult,
    SleepResult,
    RoundResult,
)
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
    "Primitive", "Program", "Observation", "Task", "ScoredProgram", "LibraryEntry",
    # Interfaces
    "Environment", "Grammar", "DriveSignal", "Memory",
    # Learner
    "Learner", "SearchConfig", "SleepConfig", "CurriculumConfig",
    "WakeResult", "SleepResult", "RoundResult",
    # Memory
    "InMemoryStore",
    # Metrics
    "CompoundingMetrics", "extract_metrics", "print_compounding_table",
    "save_metrics_json", "save_metrics_csv",
]
