"""
Metrics and reporting for the Universal Learning Loop.

The most important output is the COMPOUNDING CURVE:
    - X axis: wake-sleep round number
    - Y axis: tasks solved / library size / reuse frequency

If this curve bends upward, the framework is working.
If it plateaus, we've hit a ceiling. Both are valuable data.
"""

from __future__ import annotations
import json
import csv
import logging
from dataclasses import dataclass, asdict
from typing import TextIO

from .learner import RoundResult

logger = logging.getLogger(__name__)


@dataclass
class CompoundingMetrics:
    """The numbers that matter for testing the core claim."""
    round_number: int
    solve_rate: float
    tasks_solved: int
    tasks_total: int
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
        # Average energy of solved tasks
        solved_energies = [
            w.best.energy for w in rr.wake_results
            if w.solved and w.best is not None
        ]
        avg_energy = (
            sum(solved_energies) / len(solved_energies)
            if solved_energies else float("inf")
        )

        # Average reuse across library (placeholder — needs memory access)
        avg_reuse = 0.0  # TODO: compute from memory.get_library()

        wake_time = sum(w.wall_time for w in rr.wake_results)
        sleep_time = rr.sleep_result.wall_time

        m = CompoundingMetrics(
            round_number=rr.round_number,
            solve_rate=rr.solve_rate,
            tasks_solved=rr.tasks_solved,
            tasks_total=rr.tasks_total,
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
    header = (
        f"{'Round':>5}  {'Solved':>8}  {'Rate':>7}  "
        f"{'Library':>7}  {'New':>4}  {'Avg Energy':>10}  "
        f"{'Wake(s)':>8}  {'Sleep(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
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
