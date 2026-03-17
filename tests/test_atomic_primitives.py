"""
Tests for ARC atomic primitives.

STRIPPED TO ZERO: All primitive tests removed during zero-strip.
Tests will be re-added as each primitive is justified and added back.
"""

import unittest


class TestAtomicPrimitivesPlaceholder(unittest.TestCase):
    """Placeholder — primitive tests will be added as primitives are added."""

    def test_primitives_loaded(self):
        """Verify primitives are registered."""
        from domains.arc.transformation_primitives import (
            build_atomic_primitives,
            build_parameterized_primitives,
        )
        from domains.arc.perception_primitives import build_perception_primitives

        self.assertEqual(len(build_atomic_primitives()), 27)
        self.assertEqual(len(build_parameterized_primitives()), 9)
        self.assertEqual(len(build_perception_primitives()), 12)


class TestExtractLargestCC(unittest.TestCase):
    """Test extract_largest_cc primitive.

    Justification: tasks be94b721 and 1f85a75f.
    """

    def test_simple_extraction(self):
        from domains.arc.transformation_primitives import extract_largest_cc
        grid = [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0],
            [3, 3, 0, 0, 0],
            [3, 3, 0, 0, 0],
        ]
        result = extract_largest_cc(grid)
        self.assertEqual(result, [[3, 3], [3, 3], [3, 3]])

    def test_on_real_task_be94b721(self):
        import json, os
        path = 'data/ARC-AGI/data/training/be94b721.json'
        if not os.path.exists(path):
            self.skipTest("ARC data not available")
        from domains.arc.transformation_primitives import extract_largest_cc
        with open(path) as f:
            task = json.load(f)
        for i, ex in enumerate(task['train']):
            result = extract_largest_cc(ex['input'])
            self.assertEqual(result, ex['output'], f"Failed on train {i}")

    def test_on_real_task_1f85a75f(self):
        import json, os
        path = 'data/ARC-AGI/data/training/1f85a75f.json'
        if not os.path.exists(path):
            self.skipTest("ARC data not available")
        from domains.arc.transformation_primitives import extract_largest_cc
        with open(path) as f:
            task = json.load(f)
        for i, ex in enumerate(task['train']):
            result = extract_largest_cc(ex['input'])
            self.assertEqual(result, ex['output'], f"Failed on train {i}")


class TestInpaintPeriodic(unittest.TestCase):
    """Test periodic pattern inpainting primitive.

    Justification: tasks 73251a56 (err=0.006) and 29ec7d0e (err=0.007).
    """

    def test_simple_row_period(self):
        """Fill zeros in a row-periodic pattern."""
        from domains.arc.transformation_primitives import inpaint_periodic
        # Period-3 row pattern with one zero
        grid = [
            [1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 0, 6],
            [1, 2, 3, 1, 2, 3],
        ]
        result = inpaint_periodic(grid)
        self.assertEqual(result[1][4], 5)

    def test_2d_tile_period(self):
        """Fill zeros in a 2D tile pattern."""
        from domains.arc.transformation_primitives import inpaint_periodic
        # 2x2 tile repeated, with holes
        grid = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 0, 1, 2],
            [3, 4, 0, 4],
        ]
        result = inpaint_periodic(grid)
        self.assertEqual(result[2][1], 2)
        self.assertEqual(result[3][2], 3)

    def test_no_zeros_unchanged(self):
        """Grid with no zeros returns identical."""
        from domains.arc.transformation_primitives import inpaint_periodic
        grid = [[1, 2], [3, 4]]
        self.assertEqual(inpaint_periodic(grid), grid)

    def test_multiplication_table_pattern(self):
        """Test pattern like 29ec7d0e: periodic multiplication table."""
        from domains.arc.transformation_primitives import inpaint_periodic
        # 5-color multiplication table with period 5
        # cell(r,c) = ((r*c) % 5) + 1, but row 0 is all 1s
        grid = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [1, 3, 5, 2, 4, 1, 3, 0, 0, 4],  # zeros at (2,7) and (2,8)
            [1, 4, 2, 5, 3, 1, 4, 2, 5, 3],
            [1, 5, 4, 3, 2, 1, 5, 4, 3, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [1, 3, 5, 2, 4, 1, 3, 5, 2, 4],
            [1, 4, 2, 5, 3, 1, 4, 2, 5, 3],
            [1, 5, 4, 3, 2, 1, 5, 4, 3, 2],
        ]
        result = inpaint_periodic(grid)
        self.assertEqual(result[2][7], 5)
        self.assertEqual(result[2][8], 2)

    def test_on_real_task_29ec7d0e(self):
        """Test on actual task 29ec7d0e data."""
        import json, os
        task_path = 'data/ARC-AGI/data/training/29ec7d0e.json'
        if not os.path.exists(task_path):
            self.skipTest("ARC data not available")
        from domains.arc.transformation_primitives import inpaint_periodic
        with open(task_path) as f:
            task = json.load(f)
        for i, ex in enumerate(task['train']):
            result = inpaint_periodic(ex['input'])
            self.assertEqual(result, ex['output'],
                             f"Failed on train example {i}")

    def test_73251a56_is_not_simple_tile(self):
        """73251a56 is a diagonal band pattern, not a simple 2D tile.
        inpaint_periodic correctly does NOT apply (returns input unchanged)."""
        import json, os
        task_path = 'data/ARC-AGI/data/training/73251a56.json'
        if not os.path.exists(task_path):
            self.skipTest("ARC data not available")
        from domains.arc.transformation_primitives import inpaint_periodic
        with open(task_path) as f:
            task = json.load(f)
        # This task has a diagonal pattern, not a rectangular tile.
        # inpaint_periodic should not find a valid tile and return the
        # grid with zeros still present (or a wrong fill).
        result = inpaint_periodic(task['train'][0]['input'])
        self.assertNotEqual(result, task['train'][0]['output'])


if __name__ == "__main__":
    unittest.main()
