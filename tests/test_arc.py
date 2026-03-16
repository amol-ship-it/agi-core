"""
Tests for the ARC-AGI domain plugin.

Verifies:
1. The drive signal scores correctly
2. ARC task loading from JSON
3. Numpy conversion utilities

NOTE: Primitive-specific tests removed during zero-strip. They will be
re-added as each primitive is justified and added back.
"""

import json
import math
import os
import tempfile
import unittest

from core import Program, Task
from domains.arc import ARCEnv, ARCGrammar, ARCDrive, to_np, from_np
from domains.arc.drive import MAX_LOG_ERROR, DIM_MISMATCH_CAP
from domains.arc.dataset import load_arc_task, load_arc_dataset


class TestARCDrive(unittest.TestCase):
    """Test the ARC drive signal."""

    def test_perfect_match(self):
        drive = ARCDrive()
        grid = [[1, 2], [3, 4]]
        error = drive.prediction_error(grid, grid)
        self.assertAlmostEqual(error, 0.0)

    def test_total_mismatch(self):
        drive = ARCDrive()
        pred = [[1, 1], [1, 1]]
        exp = [[2, 2], [2, 2]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 1.0)
        self.assertLess(error, MAX_LOG_ERROR)

    def test_shape_mismatch_penalty(self):
        drive = ARCDrive()
        pred = [[1, 2, 3]]
        exp = [[1, 2]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)

    def test_none_inputs(self):
        drive = ARCDrive()
        self.assertAlmostEqual(drive.prediction_error(None, [[1]]), MAX_LOG_ERROR)
        self.assertAlmostEqual(drive.prediction_error([[1]], None), MAX_LOG_ERROR)

    def test_invalid_type(self):
        drive = ARCDrive()
        error = drive.prediction_error("not a grid", [[1]])
        self.assertAlmostEqual(error, MAX_LOG_ERROR)

    def test_log_scale_partial(self):
        """Partial match gives intermediate -log error."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)
        self.assertLess(error, MAX_LOG_ERROR)

    def test_log_scale_near_perfect(self):
        """-log(similarity) is small for near-perfect matches."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        error_partial = drive.prediction_error(pred, exp)
        pred_bad = [[5, 6], [7, 8]]
        error_bad = drive.prediction_error(pred_bad, exp)
        self.assertLess(error_partial, error_bad)
        self.assertGreater(error_bad / max(error_partial, 1e-10), 2.0)

    def test_complexity_cost(self):
        drive = ARCDrive()
        prog = Program(root="f", children=[Program(root="x")])
        self.assertAlmostEqual(drive.complexity_cost(prog), 2.0)


class TestARCTaskLoading(unittest.TestCase):
    """Test loading ARC tasks from JSON files."""

    def test_load_arc_task(self):
        task_data = {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]},
            ],
            "test": [
                {"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]},
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(task_data, f)
            path = f.name

        try:
            task = load_arc_task(path)
            self.assertEqual(len(task.train_examples), 2)
            self.assertEqual(len(task.test_inputs), 1)
            self.assertEqual(len(task.test_outputs), 1)
            self.assertGreater(task.difficulty, 0.0)
            self.assertIn("path", task.metadata)
        finally:
            os.unlink(path)

    def test_load_arc_dataset(self):
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"task{i}.json"), "w") as f:
                    json.dump(task_data, f)
            tasks = load_arc_dataset(tmpdir)
            self.assertEqual(len(tasks), 3)

    def test_load_arc_dataset_max_tasks(self):
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                with open(os.path.join(tmpdir, f"task{i}.json"), "w") as f:
                    json.dump(task_data, f)
            tasks = load_arc_dataset(tmpdir, max_tasks=2)
            self.assertEqual(len(tasks), 2)


class TestNumpyConversion(unittest.TestCase):
    """Test to_np / from_np roundtrip."""

    def test_roundtrip(self):
        grid = [[1, 2], [3, 4]]
        arr = to_np(grid)
        self.assertEqual(arr.shape, (2, 2))
        back = from_np(arr)
        self.assertEqual(back, grid)


class TestDimMismatchCap(unittest.TestCase):
    """Test that dimension-mismatched programs are capped in similarity."""

    def test_dim_mismatch_cap_applied(self):
        drive = ARCDrive()
        pred = [[1, 2, 3], [4, 5, 6]]
        exp = [[1, 2], [3, 4], [5, 6]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 1.0)

    def test_dim_match_not_affected(self):
        drive = ARCDrive()
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2], [3, 4]]
        error = drive.prediction_error(pred, exp)
        self.assertAlmostEqual(error, 0.0)

    def test_partial_dim_gradient(self):
        drive = ARCDrive()
        exp = [[1, 2, 3], [4, 5, 6]]
        pred_partial = [[1, 2], [4, 5]]
        pred_full_mismatch = [[1, 2, 3]]
        err_partial = drive.prediction_error(pred_partial, exp)
        err_full = drive.prediction_error(pred_full_mismatch, exp)
        self.assertGreater(err_partial, 1.0)
        self.assertGreater(err_full, 1.0)


class TestEmptyVocabulary(unittest.TestCase):
    """Test that the system works with zero primitives."""

    def test_empty_base_primitives(self):
        grammar = ARCGrammar()
        prims = grammar.base_primitives()
        self.assertEqual(len(prims), 0)

    def test_env_execute_unknown_primitive(self):
        """Unknown primitive should return input unchanged."""
        env = ARCEnv()
        from core import Program
        prog = Program(root="nonexistent_prim")
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        self.assertEqual(result, grid)


if __name__ == "__main__":
    unittest.main()
