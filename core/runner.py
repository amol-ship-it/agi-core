"""
Backward-compatibility shim: re-exports from common.benchmark.

All experiment infrastructure now lives in common/benchmark.py.
This module preserves ``from core.runner import ...`` for existing code.

Uses lazy imports to avoid circular dependency when common/ is the entry point.
"""

_EXPORTS = {
    "PRESETS", "DEFAULT_SEED", "DEFAULT_RUNS_DIR",
    "TeeWriter", "hline", "fmt_duration",
    "detect_machine", "install_signal_handler",
    "parse_human_int", "make_parser",
    "ProgressTracker", "ExperimentConfig",
    "resolve_from_preset", "ExperimentResult",
    "run_experiment",
}


def __getattr__(name):
    if name in _EXPORTS:
        import common.benchmark as _bm
        val = getattr(_bm, name)
        globals()[name] = val  # cache
        return val
    raise AttributeError(f"module 'core.runner' has no attribute {name!r}")
