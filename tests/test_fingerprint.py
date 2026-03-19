"""Tests for ARC task fingerprinting."""

import unittest
from core.types import Task
from domains.arc.fingerprint import fingerprint_task, TaskFingerprint


class TestDimChange(unittest.TestCase):
    """Test dim_change detection."""

    def test_same_dims(self):
        task = Task(
            task_id="t1",
            train_examples=[
                ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
                ([[0, 0], [0, 0]], [[1, 1], [1, 1]]),
            ],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.dim_change, "same")

    def test_shrink(self):
        task = Task(
            task_id="t2",
            train_examples=[
                ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2], [3, 4]]),
            ],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.dim_change, "shrink")

    def test_grow(self):
        task = Task(
            task_id="t3",
            train_examples=[
                ([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            ],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.dim_change, "grow")


class TestHasHoles(unittest.TestCase):
    """Test has_holes detection."""

    def test_grid_with_hole(self):
        # A ring of 1s with a 0 inside
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        task = Task(
            task_id="t4",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertTrue(fp.has_holes)

    def test_grid_without_hole(self):
        grid = [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
        task = Task(
            task_id="t5",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertFalse(fp.has_holes)


class TestColorCounting(unittest.TestCase):
    """Test n_colors_in, n_colors_out, colors_added, colors_removed."""

    def test_color_counts(self):
        inp = [[0, 1], [2, 3]]   # colors: {0,1,2,3} -> 4
        out = [[0, 1], [2, 5]]   # colors: {0,1,2,5} -> 4, added=1(5), removed=1(3)
        task = Task(
            task_id="t6",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.n_colors_in, 4)
        self.assertEqual(fp.n_colors_out, 4)
        self.assertEqual(fp.colors_added, 1)
        self.assertEqual(fp.colors_removed, 1)

    def test_no_color_changes(self):
        grid = [[0, 1], [2, 3]]
        task = Task(
            task_id="t7",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.colors_added, 0)
        self.assertEqual(fp.colors_removed, 0)


class TestPixelDiffRatio(unittest.TestCase):
    """Test pixel_diff_ratio calculation."""

    def test_no_diff(self):
        grid = [[1, 2], [3, 4]]
        task = Task(
            task_id="t8",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertAlmostEqual(fp.pixel_diff_ratio, 0.0)

    def test_all_diff(self):
        inp = [[1, 1], [1, 1]]
        out = [[2, 2], [2, 2]]
        task = Task(
            task_id="t9",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertAlmostEqual(fp.pixel_diff_ratio, 1.0)

    def test_half_diff(self):
        inp = [[1, 1], [1, 1]]
        out = [[1, 1], [2, 2]]
        task = Task(
            task_id="t10",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertAlmostEqual(fp.pixel_diff_ratio, 0.5)

    def test_different_dims_returns_negative(self):
        """When dims differ, pixel_diff_ratio should be -1."""
        inp = [[1, 2, 3]]
        out = [[1, 2]]
        task = Task(
            task_id="t11",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertAlmostEqual(fp.pixel_diff_ratio, -1.0)


class TestIsRecoloring(unittest.TestCase):
    """Test is_recoloring detection."""

    def test_recoloring(self):
        inp = [[0, 1, 0], [1, 0, 1]]
        out = [[0, 2, 0], [2, 0, 2]]  # same structure, different color
        task = Task(
            task_id="t12",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertTrue(fp.is_recoloring)

    def test_not_recoloring(self):
        inp = [[0, 1, 0], [1, 0, 1]]
        out = [[1, 0, 1], [0, 1, 0]]  # different structure
        task = Task(
            task_id="t13",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertFalse(fp.is_recoloring)


class TestSeparatorDetection(unittest.TestCase):
    """Test has_separators and n_sections."""

    def test_horizontal_separator(self):
        grid = [
            [1, 2, 3],
            [5, 5, 5],  # uniform row = separator
            [4, 6, 7],
        ]
        task = Task(
            task_id="t14",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertTrue(fp.has_separators)
        self.assertEqual(fp.n_sections, 2)

    def test_no_separator(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        task = Task(
            task_id="t15",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertFalse(fp.has_separators)


class TestNObjects(unittest.TestCase):
    """Test n_objects counting."""

    def test_two_objects(self):
        grid = [
            [1, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
        ]
        task = Task(
            task_id="t16",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.n_objects, 2)

    def test_no_objects(self):
        grid = [
            [0, 0],
            [0, 0],
        ]
        task = Task(
            task_id="t17",
            train_examples=[(grid, grid)],
            test_inputs=[],
        )
        fp = fingerprint_task(task)
        self.assertEqual(fp.n_objects, 0)


if __name__ == "__main__":
    unittest.main()
