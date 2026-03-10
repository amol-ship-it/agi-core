"""
Tests for the ARC-AGI domain plugin.

Verifies that:
1. All primitives produce valid grids
2. The environment can execute programs
3. The drive signal scores correctly
4. The full loop runs without errors
"""

import unittest
from core import Program, Task, InMemoryStore, Learner, SearchConfig, CurriculumConfig
from grammars.arc import (
    ARCEnv, ARCGrammar, ARCDrive, ARC_PRIMITIVES,
    rotate_90_cw, mirror_horizontal, mirror_vertical, transpose,
    crop_to_nonzero, gravity_down, fill_enclosed, identity,
    to_np, from_np, make_sample_tasks,
)


class TestARCPrimitives(unittest.TestCase):
    """Test individual ARC primitives."""

    def setUp(self):
        self.grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_identity(self):
        result = identity(self.grid)
        self.assertEqual(result, self.grid)

    def test_rotate_90_cw(self):
        result = rotate_90_cw(self.grid)
        expected = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
        self.assertEqual(result, expected)

    def test_mirror_horizontal(self):
        result = mirror_horizontal(self.grid)
        expected = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        self.assertEqual(result, expected)

    def test_mirror_vertical(self):
        result = mirror_vertical(self.grid)
        expected = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
        self.assertEqual(result, expected)

    def test_transpose(self):
        result = transpose(self.grid)
        expected = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        self.assertEqual(result, expected)

    def test_crop_to_nonzero(self):
        grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        result = crop_to_nonzero(grid)
        self.assertEqual(result, [[5]])

    def test_gravity_down(self):
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = gravity_down(grid)
        expected = [[0, 0, 0], [0, 0, 0], [1, 2, 3]]
        self.assertEqual(result, expected)

    def test_fill_enclosed(self):
        grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        result = fill_enclosed(grid)
        expected = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertEqual(result, expected)

    def test_all_primitives_return_valid_grids(self):
        """Every unary primitive should return a non-empty list of lists."""
        grid = [[1, 2], [3, 0]]
        for prim in ARC_PRIMITIVES:
            if prim.arity == 1:
                result = prim.fn(grid)
                self.assertIsInstance(result, list, f"{prim.name} didn't return a list")
                self.assertTrue(len(result) > 0, f"{prim.name} returned empty grid")
                self.assertIsInstance(result[0], list, f"{prim.name} didn't return list of lists")


class TestARCEnvironment(unittest.TestCase):
    """Test the ARC environment."""

    def test_execute_single_primitive(self):
        env = ARCEnv()
        prog = Program(root="rot90cw")
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        expected = rotate_90_cw(grid)
        self.assertEqual(result, expected)

    def test_execute_composition(self):
        env = ARCEnv()
        # mirror_h(rot90cw(input))
        prog = Program(root="mirror_h", children=[Program(root="rot90cw")])
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        expected = mirror_horizontal(rotate_90_cw(grid))
        self.assertEqual(result, expected)

    def test_execute_unknown_primitive_returns_grid(self):
        env = ARCEnv()
        prog = Program(root="nonexistent_op")
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        self.assertEqual(result, grid)


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
        self.assertAlmostEqual(error, 1.0)

    def test_shape_mismatch_penalty(self):
        drive = ARCDrive()
        pred = [[1, 2, 3]]
        exp = [[1, 2]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)


class TestARCFullLoop(unittest.TestCase):
    """Test the full wake-sleep loop on ARC tasks."""

    def test_sample_tasks_run(self):
        """The loop should run without errors on sample tasks."""
        tasks = make_sample_tasks()[:3]  # just first 3 for speed
        env = ARCEnv()
        grammar = ARCGrammar(seed=42)
        drive = ARCDrive()
        memory = InMemoryStore()

        learner = Learner(
            environment=env, grammar=grammar, drive=drive, memory=memory,
            search_config=SearchConfig(
                beam_width=30, max_generations=10,
                solve_threshold=0.001, seed=42,
            ),
        )
        results = learner.run_curriculum(
            tasks, CurriculumConfig(wake_sleep_rounds=1),
        )
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].tasks_solved, 0)


if __name__ == "__main__":
    unittest.main()
