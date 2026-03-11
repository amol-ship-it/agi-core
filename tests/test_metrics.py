"""
Tests for core/metrics.py — CompoundingMetrics, extract_metrics, save/print.

Verifies:
1. extract_metrics correctly computes from RoundResults
2. save_metrics_json produces valid JSON
3. save_metrics_csv produces valid CSV
4. print_compounding_table runs without error
"""

import csv
import json
import os
import tempfile
import unittest
from io import StringIO

from core.types import Program, ScoredProgram, LibraryEntry
from core.results import WakeResult, SleepResult, RoundResult
from core.metrics import (
    CompoundingMetrics,
    extract_metrics,
    print_compounding_table,
    save_metrics_json,
    save_metrics_csv,
)


def _make_round_result(round_num=1, n_solved=2, n_total=5, lib_size=3):
    """Build a synthetic RoundResult for testing."""
    wake_results = []
    for i in range(n_total):
        solved = i < n_solved
        best = ScoredProgram(
            program=Program(root="x"),
            energy=0.0 if solved else 1.0,
            prediction_error=0.0 if solved else 0.8,
            complexity_cost=0.001,
            task_id=f"t{i}",
        )
        wake_results.append(WakeResult(
            task_id=f"t{i}",
            train_solved=solved,
            best=best,
            generations_used=10,
            evaluations=200,
            wall_time=0.5,
        ))

    sleep_result = SleepResult(
        new_entries=[
            LibraryEntry(name="lib_0", program=Program(root="f"), usefulness=1.0),
        ],
        library_size_before=lib_size - 1,
        library_size_after=lib_size,
        wall_time=0.1,
    )

    return RoundResult(
        round_number=round_num,
        wake_results=wake_results,
        sleep_result=sleep_result,
        train_solved=n_solved,
        tasks_total=n_total,
        train_solve_rate=n_solved / n_total,
        cumulative_library_size=lib_size,
    )


class TestExtractMetrics(unittest.TestCase):

    def test_single_round(self):
        rr = _make_round_result(round_num=1, n_solved=2, n_total=5, lib_size=3)
        metrics = extract_metrics([rr])
        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertIsInstance(m, CompoundingMetrics)
        self.assertEqual(m.round_number, 1)
        self.assertAlmostEqual(m.solve_rate, 0.4)
        self.assertEqual(m.tasks_solved, 2)
        self.assertEqual(m.tasks_total, 5)
        self.assertEqual(m.library_size, 3)
        self.assertEqual(m.new_abstractions, 1)
        self.assertAlmostEqual(m.avg_energy_of_solutions, 0.0)

    def test_multiple_rounds(self):
        rr1 = _make_round_result(round_num=1, n_solved=1, n_total=3)
        rr2 = _make_round_result(round_num=2, n_solved=3, n_total=3)
        metrics = extract_metrics([rr1, rr2])
        self.assertEqual(len(metrics), 2)
        self.assertLess(metrics[0].solve_rate, metrics[1].solve_rate)

    def test_no_solved_tasks(self):
        rr = _make_round_result(round_num=1, n_solved=0, n_total=3)
        metrics = extract_metrics([rr])
        self.assertEqual(metrics[0].avg_energy_of_solutions, float("inf"))

    def test_wall_times_summed(self):
        rr = _make_round_result(round_num=1, n_solved=2, n_total=5)
        metrics = extract_metrics([rr])
        # 5 tasks × 0.5s each = 2.5s wake time
        self.assertAlmostEqual(metrics[0].wall_time_wake, 2.5)
        self.assertAlmostEqual(metrics[0].wall_time_sleep, 0.1)


class TestSaveMetricsJson(unittest.TestCase):

    def test_save_and_reload(self):
        rr = _make_round_result()
        metrics = extract_metrics([rr])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_metrics_json(metrics, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["round_number"], 1)
            self.assertAlmostEqual(data[0]["solve_rate"], 0.4)
        finally:
            os.unlink(path)


class TestSaveMetricsCsv(unittest.TestCase):

    def test_save_and_reload(self):
        rr = _make_round_result()
        metrics = extract_metrics([rr])

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_metrics_csv(metrics, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["round_number"]), 1)
        finally:
            os.unlink(path)

    def test_empty_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_metrics_csv([], path)
            # Should not crash, file should be empty or not written
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


class TestPrintCompoundingTable(unittest.TestCase):

    def test_prints_without_error(self):
        rr = _make_round_result()
        metrics = extract_metrics([rr])
        # Should not raise
        print_compounding_table(metrics)

    def test_empty_metrics(self):
        print_compounding_table([])


if __name__ == "__main__":
    unittest.main()
