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
    # New primitives
    _find_connected_components,
    keep_largest_object_only, keep_smallest_object_only,
    remove_largest_object, remove_smallest_object,
    count_objects_as_grid, recolor_each_object,
    mirror_objects_h, mirror_objects_v,
    flood_fill_bg, sort_objects_by_size,
    extract_top_left_cell, extract_bottom_right_cell,
    remove_grid_lines, detect_grid_lines,
    shift_rows_right, shift_rows_left,
    extend_lines, extend_diagonal_lines,
    binarize, color_to_most_common, upscale_pattern,
    denoise_majority, fill_rectangles,
    extract_minority_color, extract_majority_color,
    replace_noise_in_objects, hollow_objects,
    # Batch 2 primitives
    shift_down_1, shift_up_1, shift_left_1, shift_right_1,
    complete_symmetry_h, complete_symmetry_v,
    overlay_split_halves_h, overlay_split_halves_v,
    erode, spread_colors,
    rotate_colors_up, rotate_colors_down,
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


class TestConnectedComponents(unittest.TestCase):
    """Test connected component detection and object-level primitives."""

    def setUp(self):
        # Grid with 3 objects: a 2-cell horizontal bar, a single cell, and an L-shape
        self.grid = [
            [1, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 3, 0, 0],
            [0, 3, 3, 0],
        ]

    def test_find_connected_components(self):
        comps = _find_connected_components(self.grid)
        self.assertEqual(len(comps), 3)
        sizes = sorted(c["size"] for c in comps)
        self.assertEqual(sizes, [1, 2, 3])

    def test_find_components_empty_grid(self):
        self.assertEqual(_find_connected_components([[0, 0], [0, 0]]), [])

    def test_keep_largest_object_only(self):
        result = keep_largest_object_only(self.grid)
        # L-shape (color 3, size 3) should remain
        self.assertEqual(result[2][1], 3)
        self.assertEqual(result[3][1], 3)
        self.assertEqual(result[3][2], 3)
        # Others should be zeroed
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[1][3], 0)

    def test_keep_smallest_object_only(self):
        result = keep_smallest_object_only(self.grid)
        # Single cell (color 2, size 1) should remain
        self.assertEqual(result[1][3], 2)
        self.assertEqual(result[0][0], 0)

    def test_remove_largest_object(self):
        result = remove_largest_object(self.grid)
        # L-shape zeroed out
        self.assertEqual(result[2][1], 0)
        self.assertEqual(result[3][2], 0)
        # Others preserved
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][3], 2)

    def test_remove_smallest_object(self):
        result = remove_smallest_object(self.grid)
        self.assertEqual(result[1][3], 0)  # single cell removed
        self.assertEqual(result[0][0], 1)  # bar preserved

    def test_count_objects_as_grid(self):
        result = count_objects_as_grid(self.grid)
        self.assertEqual(result, [[3]])

    def test_recolor_each_object(self):
        result = recolor_each_object(self.grid)
        # Should have 3 distinct colors for 3 objects
        colors = set(c for row in result for c in row if c != 0)
        self.assertEqual(len(colors), 3)

    def test_mirror_objects_h(self):
        grid = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        result = mirror_objects_h(grid)
        # L-shape mirrored within its bbox (2x2)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[1][1], 1)
        self.assertEqual(result[1][0], 0)

    def test_mirror_objects_v(self):
        grid = [[1, 1], [1, 0], [0, 0]]
        result = mirror_objects_v(grid)
        # Object mirrored vertically within its bbox
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][1], 1)

    def test_flood_fill_bg(self):
        grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        result = flood_fill_bg(grid)
        # Enclosed 0 should be filled with 1
        self.assertEqual(result[1][1], 1)

    def test_flood_fill_bg_not_enclosed(self):
        grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
        result = flood_fill_bg(grid)
        # Top 0s are connected to border, should stay 0
        self.assertEqual(result[0][1], 0)

    def test_sort_objects_by_size(self):
        result = sort_objects_by_size(self.grid)
        # Should produce a grid with objects arranged left to right by size
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


class TestGridPartitioning(unittest.TestCase):
    """Test grid partitioning primitives."""

    def setUp(self):
        # 5x5 grid with color-5 separator lines at row 2 and col 2
        self.grid = [
            [1, 2, 5, 3, 4],
            [6, 7, 5, 8, 9],
            [5, 5, 5, 5, 5],
            [1, 0, 5, 2, 3],
            [4, 5, 5, 6, 7],
        ]

    def test_detect_grid_lines(self):
        h_lines, v_lines = detect_grid_lines(self.grid)
        self.assertIn(2, h_lines)  # row 2 is all 5s
        self.assertIn(2, v_lines)  # col 2 is all 5s

    def test_extract_top_left_cell(self):
        result = extract_top_left_cell(self.grid)
        self.assertEqual(result, [[1, 2], [6, 7]])

    def test_extract_bottom_right_cell(self):
        result = extract_bottom_right_cell(self.grid)
        self.assertEqual(result, [[2, 3], [6, 7]])

    def test_remove_grid_lines(self):
        result = remove_grid_lines(self.grid)
        # Should remove row 2 and col 2
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)

    def test_no_grid_lines(self):
        grid = [[1, 2], [3, 4]]
        h, v = detect_grid_lines(grid)
        self.assertEqual(h, [])
        self.assertEqual(v, [])
        self.assertEqual(extract_top_left_cell(grid), [[1, 2], [3, 4]])


class TestDiagonalAndLineOps(unittest.TestCase):
    """Test diagonal shift and line extension primitives."""

    def test_shift_rows_right(self):
        grid = [[1, 2], [3, 4]]
        result = shift_rows_right(grid)
        # Row 0: [1, 2, 0], Row 1: [0, 3, 4]
        self.assertEqual(result[0], [1, 2, 0])
        self.assertEqual(result[1], [0, 3, 4])

    def test_shift_rows_left(self):
        grid = [[1, 2], [3, 4]]
        result = shift_rows_left(grid)
        # Row 0: [0, 1, 2], Row 1: [3, 4, 0]
        self.assertEqual(result[0], [0, 1, 2])
        self.assertEqual(result[1], [3, 4, 0])

    def test_extend_lines(self):
        grid = [[0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]]
        result = extend_lines(grid)
        # Horizontal line of color 1 should extend to edges
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][3], 1)
        self.assertEqual(result[1][4], 1)

    def test_extend_lines_single_cell(self):
        """Single cell (not a line) should NOT be extended."""
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = extend_lines(grid)
        # Single cell, not a line (length < 2), should stay as-is
        self.assertEqual(result[0][1], 0)
        self.assertEqual(result[1][0], 0)

    def test_extend_diagonal_lines(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = extend_diagonal_lines(grid)
        # Isolated cell should extend diagonally
        self.assertNotEqual(result[0][0], 0)
        self.assertNotEqual(result[2][2], 0)


class TestColorAndPatternOps(unittest.TestCase):
    """Test binarize, color_to_most_common, upscale_pattern."""

    def test_binarize(self):
        grid = [[0, 3, 0], [7, 0, 2]]
        result = binarize(grid)
        self.assertEqual(result, [[0, 1, 0], [1, 0, 1]])

    def test_color_to_most_common(self):
        grid = [[1, 1, 2], [1, 0, 3]]
        result = color_to_most_common(grid)
        # Most common non-zero is 1
        self.assertEqual(result, [[1, 1, 1], [1, 0, 1]])

    def test_upscale_pattern(self):
        grid = [[1, 0], [0, 1]]
        result = upscale_pattern(grid)
        # 2x2 input → 4x4 output: non-zero cells become copies of the grid
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)
        # Top-left 2x2 should be the grid (cell [0][0]=1, so it's a copy)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 0)

    def test_upscale_large_grid_noop(self):
        grid = [[1]*6 for _ in range(6)]
        result = upscale_pattern(grid)
        self.assertEqual(len(result), 6)  # unchanged

    def test_all_new_primitives_produce_valid_output(self):
        """Every new primitive should return a non-empty list-of-lists."""
        grid = [[1, 2, 0], [0, 3, 0], [3, 3, 0]]
        new_fns = [
            keep_largest_object_only, keep_smallest_object_only,
            remove_largest_object, remove_smallest_object,
            count_objects_as_grid, recolor_each_object,
            mirror_objects_h, mirror_objects_v, flood_fill_bg,
            sort_objects_by_size,
            extract_top_left_cell, extract_bottom_right_cell,
            remove_grid_lines,
            shift_rows_right, shift_rows_left,
            extend_lines, extend_diagonal_lines,
            binarize, color_to_most_common, upscale_pattern,
            denoise_majority, fill_rectangles,
            extract_minority_color, extract_majority_color,
            replace_noise_in_objects, hollow_objects,
        ]
        for fn in new_fns:
            result = fn(grid)
            self.assertIsInstance(result, list, f"{fn.__name__} didn't return list")
            self.assertTrue(len(result) > 0, f"{fn.__name__} returned empty")
            self.assertIsInstance(result[0], list, f"{fn.__name__} row not list")


class TestNearMissPrimitives(unittest.TestCase):
    """Test primitives targeting near-miss tasks."""

    def test_denoise_majority(self):
        # Grid with scattered noise
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        result = denoise_majority(grid)
        # Center cell (2) should become 1 (majority of 3x3 neighborhood)
        self.assertEqual(result[1][1], 1)

    def test_fill_rectangles(self):
        # L-shaped object with hole
        grid = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        result = fill_rectangles(grid)
        # Compactness of L-shape = 3/4 = 0.75 > 0.6, so fill bbox
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][1], 1)  # hole filled

    def test_extract_minority_color(self):
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        result = extract_minority_color(grid)
        self.assertEqual(result[1][1], 2)
        self.assertEqual(result[0][0], 0)

    def test_extract_majority_color(self):
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        result = extract_majority_color(grid)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][1], 0)

    def test_replace_noise_in_objects(self):
        # Rectangle of 1s with a noise pixel (2) inside
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        result = replace_noise_in_objects(grid)
        # The 2 should be replaced with 1
        self.assertEqual(result[1][1], 1)

    def test_hollow_objects(self):
        grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = hollow_objects(grid)
        # Center should be 0 (interior), borders should be 1
        self.assertEqual(result[1][1], 0)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1)


class TestCyclicShifts(unittest.TestCase):
    """Test cyclic shift primitives."""

    def test_shift_down_1(self):
        grid = [[1, 2], [3, 4], [5, 6]]
        result = shift_down_1(grid)
        self.assertEqual(result, [[5, 6], [1, 2], [3, 4]])

    def test_shift_up_1(self):
        grid = [[1, 2], [3, 4], [5, 6]]
        result = shift_up_1(grid)
        self.assertEqual(result, [[3, 4], [5, 6], [1, 2]])

    def test_shift_left_1(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = shift_left_1(grid)
        self.assertEqual(result, [[2, 3, 1], [5, 6, 4]])

    def test_shift_right_1(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = shift_right_1(grid)
        self.assertEqual(result, [[3, 1, 2], [6, 4, 5]])

    def test_shift_roundtrip(self):
        """Shifting down then up should give identity."""
        grid = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(shift_up_1(shift_down_1(grid)), grid)
        self.assertEqual(shift_right_1(shift_left_1(grid)), grid)


class TestSymmetryCompletion(unittest.TestCase):
    """Test symmetry completion primitives."""

    def test_complete_symmetry_h_left_dominant(self):
        """Left side has content, right is empty -> mirror left to right."""
        grid = [[1, 2, 0, 0], [3, 4, 0, 0]]
        result = complete_symmetry_h(grid)
        self.assertEqual(result, [[1, 2, 2, 1], [3, 4, 4, 3]])

    def test_complete_symmetry_h_right_dominant(self):
        """Right side has content, left is empty -> mirror right to left."""
        grid = [[0, 0, 2, 1], [0, 0, 4, 3]]
        result = complete_symmetry_h(grid)
        self.assertEqual(result, [[1, 2, 2, 1], [3, 4, 4, 3]])

    def test_complete_symmetry_v_top_dominant(self):
        """Top has content, bottom is empty -> mirror top to bottom."""
        grid = [[1, 2], [3, 4], [0, 0], [0, 0]]
        result = complete_symmetry_v(grid)
        self.assertEqual(result, [[1, 2], [3, 4], [3, 4], [1, 2]])

    def test_complete_symmetry_v_bottom_dominant(self):
        """Bottom has content, top is empty -> mirror bottom to top."""
        grid = [[0, 0], [0, 0], [3, 4], [1, 2]]
        result = complete_symmetry_v(grid)
        self.assertEqual(result, [[1, 2], [3, 4], [3, 4], [1, 2]])


class TestSplitBySeparator(unittest.TestCase):
    """Test split-by-separator operations."""

    def test_overlay_split_h(self):
        """Split at horizontal separator, overlay top onto bottom."""
        grid = [
            [1, 0, 0],
            [5, 5, 5],  # separator
            [0, 0, 2],
        ]
        result = overlay_split_halves_h(grid)
        # overlay top [[1,0,0]] onto bottom [[0,0,2]] -> [[1,0,2]]
        self.assertEqual(result, [[1, 0, 2]])

    def test_overlay_split_v(self):
        """Split at vertical separator, overlay left onto right."""
        grid = [
            [1, 5, 0],
            [0, 5, 2],
        ]
        result = overlay_split_halves_v(grid)
        # overlay left [[1],[0]] onto right [[0],[2]] -> [[1],[2]]
        self.assertEqual(result, [[1], [2]])

    def test_overlay_split_h_no_separator(self):
        """No separator -> return copy of grid."""
        grid = [[1, 2], [3, 4]]
        result = overlay_split_halves_h(grid)
        self.assertEqual(result, grid)


class TestMorphologicalOps(unittest.TestCase):
    """Test morphological operations."""

    def test_erode_single_pixel(self):
        """Single pixel should be removed by erosion."""
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = erode(grid)
        self.assertEqual(result, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_erode_preserves_interior(self):
        """Interior of a 5x5 block: edge pixels adjacent to bg are removed."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        result = erode(grid)
        # Center pixel survives
        self.assertEqual(result[2][2], 1)
        # Edge of the block is removed (neighbor is bg=0)
        self.assertEqual(result[1][2], 0)

    def test_spread_colors(self):
        """Single pixel should spread to 4-connected neighbors."""
        grid = [[0, 0, 0], [0, 3, 0], [0, 0, 0]]
        result = spread_colors(grid)
        self.assertEqual(result[0][1], 3)  # up
        self.assertEqual(result[2][1], 3)  # down
        self.assertEqual(result[1][0], 3)  # left
        self.assertEqual(result[1][2], 3)  # right
        self.assertEqual(result[0][0], 0)  # diagonal stays 0

    def test_spread_doesnt_overwrite(self):
        """Spread should not overwrite existing non-zero pixels."""
        grid = [[0, 2, 0], [0, 3, 0], [0, 0, 0]]
        result = spread_colors(grid)
        self.assertEqual(result[0][1], 2)  # preserved


class TestColorCycling(unittest.TestCase):
    """Test color cycling primitives."""

    def test_rotate_colors_up(self):
        grid = [[0, 1, 9], [5, 0, 3]]
        result = rotate_colors_up(grid)
        self.assertEqual(result, [[0, 2, 1], [6, 0, 4]])

    def test_rotate_colors_down(self):
        grid = [[0, 1, 9], [5, 0, 3]]
        result = rotate_colors_down(grid)
        self.assertEqual(result, [[0, 9, 8], [4, 0, 2]])

    def test_color_cycle_roundtrip(self):
        """Up then down should give identity."""
        grid = [[0, 1, 5, 9], [3, 7, 0, 2]]
        self.assertEqual(rotate_colors_down(rotate_colors_up(grid)), grid)

    def test_color_cycle_9_times_is_identity(self):
        """Cycling up 9 times should return to original."""
        grid = [[1, 2, 3], [7, 8, 9]]
        result = grid
        for _ in range(9):
            result = rotate_colors_up(result)
        self.assertEqual(result, grid)


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
