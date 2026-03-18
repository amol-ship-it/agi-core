"""Tests for the procedural object DSL (domains/arc/procedural.py)."""

import unittest
from domains.arc.procedural import (
    compute_diff,
    attribute_to_objects,
    _check_fill_object_bbox,
    _check_extend_ray,
    _check_fill_between_aligned,
    _check_project_to_border,
    _object_properties,
    _learn_object_action_rules,
    try_procedural,
)
from domains.arc.objects import _find_connected_components


class TestComputeDiff(unittest.TestCase):
    """Test pixel diff computation."""

    def test_no_change(self):
        grid = [[1, 2], [3, 4]]
        self.assertEqual(compute_diff(grid, grid), {})

    def test_single_change(self):
        inp = [[0, 0], [0, 0]]
        out = [[0, 0], [0, 5]]
        self.assertEqual(compute_diff(inp, out), {(1, 1): 5})

    def test_multiple_changes(self):
        inp = [[1, 0, 0], [0, 0, 0]]
        out = [[1, 1, 0], [0, 0, 2]]
        diff = compute_diff(inp, out)
        self.assertEqual(diff, {(0, 1): 1, (1, 2): 2})

    def test_dimension_mismatch(self):
        inp = [[1, 2]]
        out = [[1], [2]]
        self.assertEqual(compute_diff(inp, out), {})

    def test_empty_grids(self):
        self.assertEqual(compute_diff([], []), {})


class TestAttributeToObjects(unittest.TestCase):
    """Test attributing diff pixels to objects."""

    def test_diff_inside_object(self):
        # Object at (0,0)-(1,1), color 1
        objects = [{"pixels": {(0, 0), (0, 1), (1, 0), (1, 1)},
                    "color": 1, "bbox": (0, 0, 1, 1), "size": 4}]
        diff = {(0, 0): 5}
        grid = [[1, 1], [1, 1]]
        result = attribute_to_objects(objects, diff, grid)
        self.assertEqual(result, {0: {(0, 0): 5}})

    def test_diff_near_object(self):
        # Object at (0,0), diff at (0,1) — should attribute to obj 0
        objects = [{"pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1}]
        diff = {(0, 1): 3}
        grid = [[1, 0]]
        result = attribute_to_objects(objects, diff, grid)
        self.assertEqual(result, {0: {(0, 1): 3}})

    def test_multiple_objects(self):
        # Two objects, diff pixels closer to each
        objects = [
            {"pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1},
            {"pixels": {(0, 4)}, "color": 2, "bbox": (0, 4, 0, 4), "size": 1},
        ]
        diff = {(0, 1): 1, (0, 3): 2}
        grid = [[1, 0, 0, 0, 2]]
        result = attribute_to_objects(objects, diff, grid)
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertIn((0, 1), result[0])
        self.assertIn((0, 3), result[1])

    def test_empty_diff(self):
        objects = [{"pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1}]
        self.assertEqual(attribute_to_objects(objects, {}, [[1]]), {})


class TestFillObjectBbox(unittest.TestCase):
    """Test fill_object_bbox template matching."""

    def test_perfect_fill(self):
        # 3x3 object with hollow center → center filled
        obj = {
            "pixels": {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
            "color": 1, "bbox": (0, 0, 2, 2), "size": 8,
        }
        diff = {(1, 1): 1}
        grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        result = _check_fill_object_bbox(obj, diff, grid)
        self.assertIsNotNone(result)
        self.assertEqual(result["template"], "fill_object_bbox")
        self.assertEqual(result["fill_color"], 1)

    def test_no_match(self):
        obj = {
            "pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1,
        }
        diff = {(2, 2): 3}  # Far from object
        grid = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = _check_fill_object_bbox(obj, diff, grid)
        self.assertIsNone(result)


class TestExtendRay(unittest.TestCase):
    """Test extend_ray template matching."""

    def test_extend_right(self):
        # Single pixel at (0,0), ray extends right filling (0,1) and (0,2)
        obj = {
            "pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1,
        }
        diff = {(0, 1): 1, (0, 2): 1}
        grid = [[1, 0, 0]]
        result = _check_extend_ray(obj, diff, grid)
        self.assertIsNotNone(result)
        self.assertEqual(result["template"], "extend_ray")
        self.assertEqual(result["direction"], "right")

    def test_extend_down(self):
        obj = {
            "pixels": {(0, 0)}, "color": 2, "bbox": (0, 0, 0, 0), "size": 1,
        }
        diff = {(1, 0): 2, (2, 0): 2}
        grid = [[2], [0], [0]]
        result = _check_extend_ray(obj, diff, grid)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "down")


class TestFillBetweenAligned(unittest.TestCase):
    """Test fill_between_aligned template matching."""

    def test_horizontal_fill(self):
        # Two objects on same row, gap filled
        obj0 = {"pixels": {(0, 0)}, "color": 1, "bbox": (0, 0, 0, 0), "size": 1}
        obj1 = {"pixels": {(0, 3)}, "color": 1, "bbox": (0, 3, 0, 3), "size": 1}
        all_objects = [obj0, obj1]
        diff = {(0, 1): 1, (0, 2): 1}
        grid = [[1, 0, 0, 1]]
        result = _check_fill_between_aligned(obj0, 0, all_objects, diff, grid)
        self.assertIsNotNone(result)
        self.assertEqual(result["template"], "fill_between")
        self.assertEqual(result["axis"], "h")


class TestProjectToBorder(unittest.TestCase):
    """Test project_to_border template matching."""

    def test_project_right(self):
        obj = {
            "pixels": {(1, 0)}, "color": 3, "bbox": (1, 0, 1, 0), "size": 1,
        }
        diff = {(1, 1): 3, (1, 2): 3, (1, 3): 3}
        grid = [[0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]
        result = _check_project_to_border(obj, diff, grid)
        self.assertIsNotNone(result)
        self.assertEqual(result["template"], "project_to_border")
        self.assertEqual(result["direction"], "right")


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for procedural rule learning."""

    def _make_task(self, examples):
        """Create a minimal task-like object."""
        class MockTask:
            def __init__(self, train):
                self.train_examples = train
                self.test_inputs = []
                self.test_outputs = []
        return MockTask(examples)

    def test_fill_bbox_learning(self):
        """Two examples: hollow square objects get filled."""
        # Example 1: 3x3 hollow square at (0,0)
        inp1 = [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        out1 = [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        # Example 2: 3x3 hollow square at (1,2)
        inp2 = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 1],
        ]
        out2 = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
        examples = [(inp1, out1), (inp2, out2)]
        result = _learn_object_action_rules(examples)
        self.assertIsNotNone(result, "Should learn fill_object_bbox rule")
        name, fn = result
        self.assertIn("procedural", name)
        # Verify it works on both examples
        self.assertEqual(fn(inp1), out1)
        self.assertEqual(fn(inp2), out2)

    def test_extend_ray_learning(self):
        """Objects extend rays rightward."""
        inp1 = [[1, 0, 0, 0],
                 [0, 0, 0, 0]]
        out1 = [[1, 1, 1, 1],
                 [0, 0, 0, 0]]
        inp2 = [[0, 0, 0, 0],
                 [1, 0, 0, 0]]
        out2 = [[0, 0, 0, 0],
                 [1, 1, 1, 1]]
        examples = [(inp1, out1), (inp2, out2)]
        result = _learn_object_action_rules(examples)
        self.assertIsNotNone(result, "Should learn extend_ray rule")
        name, fn = result
        self.assertEqual(fn(inp1), out1)
        self.assertEqual(fn(inp2), out2)

    def test_try_procedural_integration(self):
        """Test the public API via try_procedural."""
        inp1 = [
            [2, 2, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 0],
        ]
        out1 = [
            [2, 2, 2, 0],
            [2, 2, 2, 0],
            [2, 2, 2, 0],
        ]
        inp2 = [
            [0, 2, 2, 2],
            [0, 2, 0, 2],
            [0, 2, 2, 2],
        ]
        out2 = [
            [0, 2, 2, 2],
            [0, 2, 2, 2],
            [0, 2, 2, 2],
        ]
        task = self._make_task([(inp1, out1), (inp2, out2)])
        result = try_procedural(task)
        self.assertIsNotNone(result)
        name, fn = result
        self.assertEqual(fn(inp1), out1)
        self.assertEqual(fn(inp2), out2)

    def test_no_match_returns_none(self):
        """Random transformations should not match any template."""
        inp1 = [[1, 2], [3, 4]]
        out1 = [[5, 6], [7, 8]]
        inp2 = [[1, 0], [0, 1]]
        out2 = [[3, 3], [3, 3]]
        task = self._make_task([(inp1, out1), (inp2, out2)])
        result = try_procedural(task)
        self.assertIsNone(result)

    def test_single_example_returns_none(self):
        """Need at least 2 examples for LOOCV."""
        task = self._make_task([([[1]], [[2]])])
        self.assertIsNone(try_procedural(task))

    def test_dimension_change_returns_none(self):
        """Different-sized input/output should return None."""
        task = self._make_task([
            ([[1, 2]], [[1], [2]]),
            ([[3, 4]], [[3], [4]]),
        ])
        self.assertIsNone(try_procedural(task))


class TestObjectProperties(unittest.TestCase):
    """Test object property computation."""

    def test_basic_properties(self):
        obj = {
            "pixels": {(0, 0), (0, 1), (1, 0), (1, 1)},
            "color": 3, "bbox": (0, 0, 1, 1), "size": 4,
            "subgrid": [[3, 3], [3, 3]],
        }
        grid = [[3, 3, 0], [3, 3, 0], [0, 0, 0]]
        props = _object_properties(obj, [obj], grid)
        self.assertEqual(props["color"], 3)
        self.assertEqual(props["size"], 4)
        self.assertTrue(props["is_largest"])
        self.assertTrue(props["is_smallest"])
        self.assertFalse(props["has_hole"])
        self.assertTrue(props["touches_top"])
        self.assertTrue(props["touches_left"])


if __name__ == "__main__":
    unittest.main()
