"""
Metrics and reporting for the Universal Learning Loop.

The most important output is the COMPOUNDING CURVE:
    - X axis: wake-sleep round number
    - Y axis: tasks solved / library size / reuse frequency

If this curve bends upward, the framework is working.
If it plateaus, we've hit a ceiling. Both are valuable data.

Terminology:
    - "solved" = program passes held-out test examples (the real metric)
    - "train_matched" = program matches training examples (may overfit)
"""

from __future__ import annotations
import json
import csv
import logging
from dataclasses import dataclass, asdict

from .results import RoundResult

logger = logging.getLogger(__name__)


@dataclass
class CompoundingMetrics:
    """The numbers that matter for testing the core claim.

    Primary metric: solve_rate (test-verified).
    Secondary: train_solve_rate (training examples only, may overfit).
    """
    round_number: int
    # Primary: test-verified solve rate (falls back to train when no test data)
    solve_rate: float
    tasks_solved: int
    tasks_total: int
    # Secondary: training-only match rate
    train_solve_rate: float
    train_solved: int
    # Library / search stats
    library_size: int
    new_abstractions: int
    avg_reuse_per_entry: float  # are library entries actually being reused?
    avg_energy_of_solutions: float
    wall_time_wake: float
    wall_time_sleep: float


def extract_metrics(results: list[RoundResult]) -> list[CompoundingMetrics]:
    """Convert raw RoundResults into the metrics that test the claim."""
    metrics = []
    for rr in results:
        # Average energy of solved tasks (train-matched, since those have programs)
        solved_energies = [
            w.best.energy for w in rr.wake_results
            if w.train_solved and w.best is not None
        ]
        avg_energy = (
            sum(solved_energies) / len(solved_energies)
            if solved_energies else float("inf")
        )

        # Average reuse: computed from library entries in sleep result
        lib_entries = rr.sleep_result.new_entries
        all_reuse = [e.reuse_count for e in lib_entries] if lib_entries else []
        avg_reuse = sum(all_reuse) / len(all_reuse) if all_reuse else 0.0

        wake_time = sum(w.wall_time for w in rr.wake_results)
        sleep_time = rr.sleep_result.wall_time

        m = CompoundingMetrics(
            round_number=rr.round_number,
            solve_rate=rr.solve_rate,          # test-verified (property)
            tasks_solved=rr.solved,            # test-verified (property)
            tasks_total=rr.tasks_total,
            train_solve_rate=rr.train_solve_rate,
            train_solved=rr.train_solved,
            library_size=rr.cumulative_library_size,
            new_abstractions=len(rr.sleep_result.new_entries),
            avg_reuse_per_entry=avg_reuse,
            avg_energy_of_solutions=avg_energy,
            wall_time_wake=wake_time,
            wall_time_sleep=sleep_time,
        )
        metrics.append(m)
    return metrics


def print_compounding_table(metrics: list[CompoundingMetrics]) -> None:
    """Print the compounding curve as a text table."""
    has_overfit = any(m.train_solved > m.tasks_solved for m in metrics)
    if has_overfit:
        header = (
            f"{'Round':>5}  {'Solved':>10}  {'Rate':>7}  "
            f"{'TrMatch':>10}  {'TrRate':>7}  "
            f"{'Library':>7}  {'New':>4}  "
            f"{'Wake(s)':>8}  {'Sleep(s)':>8}"
        )
    else:
        header = (
            f"{'Round':>5}  {'Solved':>8}  {'Rate':>7}  "
            f"{'Library':>7}  {'New':>4}  {'Avg Energy':>10}  "
            f"{'Wake(s)':>8}  {'Sleep(s)':>8}"
        )
    print(header)
    print("-" * len(header))
    for m in metrics:
        if has_overfit:
            print(
                f"{m.round_number:>5}  "
                f"{m.tasks_solved:>4}/{m.tasks_total:<4}  "
                f"{m.solve_rate:>6.1%}  "
                f"{m.train_solved:>4}/{m.tasks_total:<4}  "
                f"{m.train_solve_rate:>6.1%}  "
                f"{m.library_size:>7}  "
                f"{m.new_abstractions:>4}  "
                f"{m.wall_time_wake:>8.1f}  "
                f"{m.wall_time_sleep:>8.1f}"
            )
        else:
            print(
                f"{m.round_number:>5}  "
                f"{m.tasks_solved:>3}/{m.tasks_total:<4}  "
                f"{m.solve_rate:>6.1%}  "
                f"{m.library_size:>7}  "
                f"{m.new_abstractions:>4}  "
                f"{m.avg_energy_of_solutions:>10.4f}  "
                f"{m.wall_time_wake:>8.1f}  "
                f"{m.wall_time_sleep:>8.1f}"
            )


def save_metrics_json(metrics: list[CompoundingMetrics], path: str) -> None:
    """Save metrics as JSON for plotting."""
    with open(path, "w") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2)
    logger.info(f"Metrics saved to {path}")


def save_metrics_csv(metrics: list[CompoundingMetrics], path: str) -> None:
    """Save metrics as CSV for spreadsheets."""
    if not metrics:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))
    logger.info(f"Metrics CSV saved to {path}")
