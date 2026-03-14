"""
Benchmark infrastructure for the Universal Learning Loop.

This package provides the experiment runner, pipeline orchestration,
progress tracking, and CLI utilities. It is separate from `core/`
(which contains only the learning algorithm) so that benchmark
plumbing never pollutes the invariant core.

Public API:
    ExperimentConfig, ExperimentResult, run_experiment
    make_parser, resolve_from_preset, PRESETS
    ProgressTracker, TeeWriter, parse_human_int, fmt_duration
    run_pipeline, pipeline_tee, save_pipeline_results, print_pipeline_summary
"""

from .benchmark import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    make_parser,
    resolve_from_preset,
    PRESETS,
    ProgressTracker,
    TeeWriter,
    parse_human_int,
    fmt_duration,
    run_pipeline,
    pipeline_tee,
    save_pipeline_results,
    print_pipeline_summary,
)

__all__ = [
    "ExperimentConfig", "ExperimentResult", "run_experiment",
    "make_parser", "resolve_from_preset", "PRESETS",
    "ProgressTracker", "TeeWriter", "parse_human_int", "fmt_duration",
    "run_pipeline", "pipeline_tee", "save_pipeline_results", "print_pipeline_summary",
]
