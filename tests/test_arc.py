"""
Tests for the ARC-AGI domain plugin.

Verifies that:
1. All primitives produce valid grids
2. Utility functions work correctly (grid_shape, valid_grid, etc.)
3. The environment can execute programs (unary, binary, error paths)
4. The grammar can mutate and crossover programs
5. The drive signal scores correctly (match, mismatch, shape mismatch, None)
6. ARC task loading from JSON
7. The full loop runs without errors
"""

import json
import os
import tempfile
import unittest

from core import Program, Task, InMemoryStore, Learner, SearchConfig, CurriculumConfig
from grammars.arc import (
    ARCEnv, ARCGrammar, ARCDrive, ARC_PRIMITIVES,
    rotate_90_cw, mirror_horizontal, mirror_vertical, transpose,
    crop_to_nonzero, gravity_down, fill_enclosed, identity,
    to_np, from_np, make_sample_tasks,
    grid_shape, valid_grid, empty_grid, invert_colors,
    replace_bg_with_most_common, keep_color, remove_color,
    most_common_color, fill_color, crop_to_color,
    xor_halves_v, or_halves_v, xor_halves_h, or_halves_h,
    count_colors, find_bounding_box, overlay,
    load_arc_task, load_arc_dataset,
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


class TestARCUtilityFunctions(unittest.TestCase):
    """Test ARC utility functions that weren't covered by primitive tests."""

    def test_grid_shape(self):
        self.assertEqual(grid_shape([[1, 2], [3, 4]]), (2, 2))

    def test_grid_shape_empty(self):
        self.assertEqual(grid_shape([]), (0, 0))

    def test_grid_shape_empty_row(self):
        self.assertEqual(grid_shape([[]]), (1, 0))

    def test_valid_grid_true(self):
        self.assertTrue(valid_grid([[1, 2], [3, 4]]))

    def test_valid_grid_empty(self):
        self.assertFalse(valid_grid([]))
        self.assertFalse(valid_grid([[]]))

    def test_valid_grid_ragged(self):
        self.assertFalse(valid_grid([[1, 2], [3]]))

    def test_valid_grid_out_of_range(self):
        self.assertFalse(valid_grid([[10]]))
        self.assertFalse(valid_grid([[-1]]))

    def test_empty_grid(self):
        result = empty_grid(2, 3, fill=5)
        self.assertEqual(result, [[5, 5, 5], [5, 5, 5]])

    def test_empty_grid_default_fill(self):
        result = empty_grid(1, 2)
        self.assertEqual(result, [[0, 0]])

    def test_invert_colors(self):
        result = invert_colors([[0, 1, 9]])
        self.assertEqual(result, [[0, 9, 1]])

    def test_replace_bg_with_most_common(self):
        grid = [[0, 1, 1], [0, 2, 1]]
        result = replace_bg_with_most_common(grid)
        # Most common non-zero is 1, so 0s become 1
        self.assertEqual(result, [[1, 1, 1], [1, 2, 1]])

    def test_replace_bg_all_zeros(self):
        grid = [[0, 0], [0, 0]]
        result = replace_bg_with_most_common(grid)
        self.assertEqual(result, [[0, 0], [0, 0]])

    def test_keep_color(self):
        result = keep_color([[1, 2, 3], [1, 1, 2]], 1)
        self.assertEqual(result, [[1, 0, 0], [1, 1, 0]])

    def test_remove_color(self):
        result = remove_color([[1, 2, 3], [1, 1, 2]], 1)
        self.assertEqual(result, [[0, 2, 3], [0, 0, 2]])

    def test_most_common_color(self):
        self.assertEqual(most_common_color([[1, 2, 1], [0, 1, 0]]), 1)

    def test_most_common_color_all_zeros(self):
        self.assertEqual(most_common_color([[0, 0], [0, 0]]), 0)

    def test_fill_color(self):
        result = fill_color([[1, 2], [3, 4]], 5)
        self.assertEqual(result, [[5, 5], [5, 5]])

    def test_crop_to_nonzero_all_zero(self):
        result = crop_to_nonzero([[0, 0], [0, 0]])
        self.assertEqual(result, [[0]])

    def test_crop_to_color(self):
        grid = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
        result = crop_to_color(grid, 1)
        self.assertEqual(result, [[1], [1]])

    def test_crop_to_color_not_found(self):
        result = crop_to_color([[0, 0], [0, 0]], 5)
        self.assertEqual(result, [[0]])

    def test_count_colors(self):
        self.assertEqual(count_colors([[1, 2, 0], [3, 1, 0]]), 3)
        self.assertEqual(count_colors([[0, 0], [0, 0]]), 0)

    def test_find_bounding_box(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(find_bounding_box(grid), (1, 1, 1, 1))

    def test_find_bounding_box_empty(self):
        self.assertEqual(find_bounding_box([[0, 0], [0, 0]]), (0, 0, 0, 0))

    def test_overlay(self):
        base = [[1, 1], [1, 1]]
        top = [[0, 2], [2, 0]]
        result = overlay(base, top)
        self.assertEqual(result, [[1, 2], [2, 1]])

    def test_overlay_different_sizes(self):
        base = [[1, 1, 1], [1, 1, 1]]
        top = [[2]]
        result = overlay(base, top)
        self.assertEqual(result[0][0], 2)
        self.assertEqual(result[0][1], 1)

    def test_xor_halves_v(self):
        grid = [[1, 0], [0, 0], [0, 1], [0, 0]]
        result = xor_halves_v(grid)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_or_halves_v(self):
        grid = [[1, 0], [0, 0], [0, 1], [0, 0]]
        result = or_halves_v(grid)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_xor_halves_h(self):
        grid = [[1, 0, 0, 1]]
        result = xor_halves_h(grid)
        self.assertIsInstance(result, list)

    def test_or_halves_h(self):
        grid = [[1, 0, 0, 1]]
        result = or_halves_h(grid)
        self.assertIsInstance(result, list)

    def test_xor_halves_v_odd_height(self):
        # Odd height: top half != bottom half size
        grid = [[1, 0], [0, 1], [1, 1]]
        result = xor_halves_v(grid)
        self.assertIsInstance(result, list)

    def test_or_halves_h_odd_width(self):
        grid = [[1, 0, 1]]
        result = or_halves_h(grid)
        self.assertIsInstance(result, list)

    def test_xor_halves_v_mismatched_shape(self):
        """When top and bottom halves have different shapes, return original."""
        grid = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]]
        result = xor_halves_v(grid)
        self.assertIsInstance(result, list)

    def test_or_halves_v_mismatched_shape(self):
        grid = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]]
        result = or_halves_v(grid)
        self.assertIsInstance(result, list)

    def test_xor_halves_h_mismatched_shape(self):
        grid = [[1, 0, 1, 0, 1]]  # 5 cols -> left=2, right=2, mismatch
        result = xor_halves_h(grid)
        self.assertIsInstance(result, list)

    def test_or_halves_h_mismatched_shape(self):
        grid = [[1, 0, 1, 0, 1]]
        result = or_halves_h(grid)
        self.assertIsInstance(result, list)

    def test_fill_enclosed_zeros_on_border_columns(self):
        """Test fill_enclosed with zeros touching top/bottom of columns."""
        grid = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        result = fill_enclosed(grid)
        self.assertIsInstance(result, list)
        # Center cell is enclosed, border zeros are not
        self.assertNotEqual(result[1][1], 0)


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

    def test_load_task(self):
        env = ARCEnv()
        task = Task(
            task_id="t1",
            train_examples=[([[1]], [[2]]), ([[3]], [[4]])],
            test_inputs=[[[5]]],
        )
        obs = env.load_task(task)
        self.assertEqual(len(obs.data), 2)
        self.assertEqual(obs.metadata["task_id"], "t1")

    def test_reset(self):
        env = ARCEnv()
        task = Task(task_id="t1", train_examples=[([[1]], [[2]])], test_inputs=[])
        env.load_task(task)
        env.reset()
        self.assertIsNone(env._current_task)

    def test_execute_binary_primitive(self):
        """Test binary (arity-2) primitive execution via overlay."""
        env = ARCEnv()
        prog = Program(root="overlay", children=[
            Program(root="identity"),
            Program(root="identity"),
        ])
        grid = [[1, 0], [0, 1]]
        result = env.execute(prog, grid)
        self.assertIsInstance(result, list)

    def test_execute_nullary_primitive(self):
        """Nullary primitives should return the input grid."""
        env = ARCEnv()
        # Find a nullary primitive if any exist in ARC_PRIMITIVES
        nullary = [p for p in ARC_PRIMITIVES if p.arity == 0]
        if nullary:
            prog = Program(root=nullary[0].name)
            grid = [[1, 2], [3, 4]]
            result = env.execute(prog, grid)
            self.assertEqual(result, grid)

    def test_execute_primitive_that_crashes(self):
        """Exception in primitive should return input grid."""
        env = ARCEnv()
        # Execute a composition that might fail gracefully
        prog = Program(root="rot90cw", children=[Program(root="nonexistent")])
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        self.assertIsInstance(result, list)


class TestARCGrammarMethods(unittest.TestCase):
    """Test ARCGrammar mutate and crossover."""

    def test_mutate(self):
        grammar = ARCGrammar(seed=42)
        prog = Program(root="rot90cw")
        prims = grammar.base_primitives()
        mutated = grammar.mutate(prog, prims)
        self.assertIsInstance(mutated, Program)

    def test_mutate_tree(self):
        grammar = ARCGrammar(seed=42)
        prog = Program(root="mirror_h", children=[Program(root="rot90cw")])
        prims = grammar.base_primitives()
        mutated = grammar.mutate(prog, prims)
        self.assertIsInstance(mutated, Program)

    def test_crossover(self):
        grammar = ARCGrammar(seed=42)
        a = Program(root="mirror_h", children=[Program(root="rot90cw")])
        b = Program(root="transpose", children=[Program(root="mirror_v")])
        child = grammar.crossover(a, b)
        self.assertIsInstance(child, Program)

    def test_crossover_leaves(self):
        grammar = ARCGrammar(seed=42)
        a = Program(root="rot90cw")
        b = Program(root="mirror_h")
        child = grammar.crossover(a, b)
        self.assertIsInstance(child, Program)

    def test_compose(self):
        grammar = ARCGrammar(seed=42)
        prim = ARC_PRIMITIVES[0]
        prog = grammar.compose(prim, [Program(root="identity")])
        self.assertEqual(prog.root, prim.name)

    def test_mutate_unknown_primitive(self):
        """Mutating a node with an unknown primitive should still work."""
        grammar = ARCGrammar(seed=42)
        prog = Program(root="unknown_op", children=[Program(root="rot90cw")])
        prims = grammar.base_primitives()
        mutated = grammar.mutate(prog, prims)
        self.assertIsInstance(mutated, Program)

    def test_crossover_empty_nodes(self):
        """Crossover with no collectable nodes should return copy."""
        grammar = ARCGrammar(seed=42)
        a = Program(root="rot90cw")
        b = Program(root="mirror_h")
        # Both are single nodes, crossover should work
        child = grammar.crossover(a, b)
        self.assertIsInstance(child, Program)


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

    def test_none_inputs(self):
        drive = ARCDrive()
        self.assertAlmostEqual(drive.prediction_error(None, [[1]]), 1.0)
        self.assertAlmostEqual(drive.prediction_error([[1]], None), 1.0)

    def test_invalid_type(self):
        drive = ARCDrive()
        error = drive.prediction_error("not a grid", [[1]])
        self.assertAlmostEqual(error, 1.0)

    def test_empty_grid_zero_cells(self):
        drive = ARCDrive()
        # Shape mismatch leading to min_r/min_c = 0
        error = drive.prediction_error([[1]], [[1, 2], [3, 4]])
        self.assertGreater(error, 0.0)

    def test_shape_mismatch_overlap_error(self):
        """Shape mismatch with valid overlap region."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2, 5]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)
        self.assertLessEqual(error, 1.0)

    def test_zero_total_cells(self):
        drive = ARCDrive()
        error = drive.prediction_error([[]], [[]])
        # Should handle gracefully
        self.assertIsInstance(error, float)

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

    def test_load_arc_dataset_skips_bad_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.json"), "w") as f:
                f.write("not json")
            tasks = load_arc_dataset(tmpdir)
            self.assertEqual(len(tasks), 0)


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
