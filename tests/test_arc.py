"""
Tests for the ARC-AGI domain plugin.

Verifies:
1. The drive signal scores correctly
2. ARC task loading from JSON
3. Numpy conversion utilities
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
from domains.arc.transformation_primitives import (
    gravity_up, gravity_left, gravity_right,
    sort_rows_by_nonzero, sort_cols_by_nonzero,
    _repeat_rows_factory, _repeat_cols_factory,
)


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
        # -log of small similarity → large positive value
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
        exp = [[1, 2], [3, 4]]  # 3/4 pixels match
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)
        self.assertLess(error, MAX_LOG_ERROR)

    def test_log_scale_near_perfect(self):
        """-log(similarity) is small for near-perfect matches."""
        drive = ARCDrive()
        # 3/4 pixels match → high similarity → small -log
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        error_partial = drive.prediction_error(pred, exp)
        # 0/4 pixels match → low similarity → large -log
        pred_bad = [[5, 6], [7, 8]]
        error_bad = drive.prediction_error(pred_bad, exp)
        # Near-perfect should have much smaller error than bad
        self.assertLess(error_partial, error_bad)
        # -log scale: ratio should be large (exponential separation)
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
        """Wrong-dim program → error > 1.0 (well above near-miss threshold)."""
        drive = ARCDrive()
        pred = [[1, 2, 3], [4, 5, 6]]  # 2×3
        exp = [[1, 2], [3, 4], [5, 6]]   # 3×2
        error = drive.prediction_error(pred, exp)
        # -log(DIM_MISMATCH_CAP) ≈ 1.05, so error should be > 1.0
        self.assertGreater(error, 1.0)

    def test_dim_match_not_affected(self):
        """Right-dim program → no cap applied, can score very well."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2], [3, 4]]
        error = drive.prediction_error(pred, exp)
        self.assertAlmostEqual(error, 0.0)

    def test_partial_dim_gradient(self):
        """One dim correct scores better than both wrong (but both capped)."""
        drive = ARCDrive()
        exp = [[1, 2, 3], [4, 5, 6]]  # 2×3
        # Same height, wrong width
        pred_partial = [[1, 2], [4, 5]]  # 2×2
        # Wrong both
        pred_full_mismatch = [[1, 2, 3]]  # 1×3
        err_partial = drive.prediction_error(pred_partial, exp)
        err_full = drive.prediction_error(pred_full_mismatch, exp)
        # Both should be capped (> 1.0), but they're distinguishable
        self.assertGreater(err_partial, 1.0)
        self.assertGreater(err_full, 1.0)


class TestGravityDirectional(unittest.TestCase):
    """Test directional gravity primitives."""

    def test_gravity_up(self):
        grid = [
            [0, 0, 0],
            [1, 0, 2],
            [0, 3, 0],
        ]
        result = gravity_up(grid)
        # Column 0: [1] floats to top
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][0], 0)
        self.assertEqual(result[2][0], 0)
        # Column 1: [3] floats to top
        self.assertEqual(result[0][1], 3)
        self.assertEqual(result[1][1], 0)
        # Column 2: [2] floats to top
        self.assertEqual(result[0][2], 2)
        self.assertEqual(result[1][2], 0)

    def test_gravity_left(self):
        grid = [
            [0, 1, 0],
            [2, 0, 3],
        ]
        result = gravity_left(grid)
        # Row 0: [1] packs left
        self.assertEqual(result[0], [1, 0, 0])
        # Row 1: [2, 3] pack left
        self.assertEqual(result[1], [2, 3, 0])

    def test_gravity_right(self):
        grid = [
            [0, 1, 0],
            [2, 0, 3],
        ]
        result = gravity_right(grid)
        # Row 0: [1] packs right
        self.assertEqual(result[0], [0, 0, 1])
        # Row 1: [2, 3] pack right
        self.assertEqual(result[1], [0, 2, 3])


class TestSortPrimitives(unittest.TestCase):
    """Test row/column sorting primitives."""

    def test_sort_rows(self):
        grid = [
            [1, 1, 1],  # 3 nonzero
            [0, 0, 0],  # 0 nonzero
            [1, 0, 1],  # 2 nonzero
        ]
        result = sort_rows_by_nonzero(grid)
        # Ascending: 0, 2, 3
        self.assertEqual(result[0], [0, 0, 0])
        self.assertEqual(result[1], [1, 0, 1])
        self.assertEqual(result[2], [1, 1, 1])

    def test_sort_cols(self):
        grid = [
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
        # Col 0: 3 nonzero, Col 1: 0 nonzero, Col 2: 1 nonzero
        result = sort_cols_by_nonzero(grid)
        # Ascending: col1(0), col2(1), col0(3)
        self.assertEqual(result[0], [0, 1, 1])
        self.assertEqual(result[1], [0, 0, 1])
        self.assertEqual(result[2], [0, 0, 1])


class TestRepeatPrimitives(unittest.TestCase):
    """Test repeat_rows and repeat_cols parameterized primitives."""

    def test_repeat_rows_basic(self):
        repeat = _repeat_rows_factory(2)
        grid = [[1, 2], [3, 4]]
        result = repeat(grid)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [[1, 2], [1, 2], [3, 4], [3, 4]])

    def test_repeat_cols_basic(self):
        repeat = _repeat_cols_factory(3)
        grid = [[1, 2], [3, 4]]
        result = repeat(grid)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 6)
        self.assertEqual(result, [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]])

    def test_repeat_n1_identity(self):
        repeat_r = _repeat_rows_factory(1)
        repeat_c = _repeat_cols_factory(1)
        grid = [[1, 2], [3, 4]]
        self.assertEqual(repeat_r(grid), grid)
        self.assertEqual(repeat_c(grid), grid)


if __name__ == "__main__":
    unittest.main()
