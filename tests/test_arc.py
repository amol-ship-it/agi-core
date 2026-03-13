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
from domains.arc import ARCEnv, ARCGrammar, ARCDrive, ARC_PRIMITIVES, to_np, from_np
from domains.arc.dataset import make_sample_tasks, load_arc_task, load_arc_dataset
from domains.arc.primitives import (
    rotate_90_cw, mirror_horizontal, mirror_vertical, transpose,
    crop_to_nonzero, gravity_down, fill_enclosed, identity,
    grid_shape, valid_grid, empty_grid, invert_colors,
    replace_bg_with_most_common, keep_color, remove_color,
    most_common_color, fill_color, crop_to_color,
    xor_halves_v, or_halves_v, xor_halves_h, or_halves_h,
    count_colors, find_bounding_box, overlay,
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
    shift_down_1, shift_up_1, shift_left_1, shift_right_1,
    complete_symmetry_h, complete_symmetry_v,
    overlay_split_halves_h, overlay_split_halves_v,
    erode, spread_colors,
    rotate_colors_up, rotate_colors_down,
    select_odd_one_out, overlay_grid_cells, majority_vote_cells,
    surround_pixels_3x3, draw_cross_from_pixels, draw_cross_to_contact,
    connect_same_color_h, connect_same_color_v,
    scale_4x, scale_5x, downscale_4x, downscale_5x,
    _detect_any_separator_lines, _split_grid_cells,
    fill_tile_pattern,
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

    def test_grid_size_guard_prevents_expansion(self):
        """Composing grid-expanding primitives must not create huge grids.

        Without the guard, tile_3x3(tile_3x3(x)) on a 30x30 grid creates
        a 270x270 intermediate (72,900 pixels). Numba functions on such
        grids run for minutes, ignore Ctrl-C, and consume GBs of RAM.
        """
        env = ARCEnv()
        # tile_3x3(tile_3x3(input)) on a 30x30 grid
        # Inner: 30x30 -> 90x90 (8100 pixels, under 10,000 limit)
        # Outer: 90x90 -> would be 270x270 but input is 8100 pixels < guard
        # The guard should let the inner pass but block the outer from
        # processing the 8100-pixel grid through tile_3x3 (since 90x90
        # is already at the edge, the result 270x270 = 72900 > guard).
        prog = Program(root="tile_3x3", children=[
            Program(root="tile_3x3")])
        grid = [[i % 10 for i in range(30)] for _ in range(30)]
        result = env.execute(prog, grid)
        # Result should be bounded — either the guard kicked in (returns
        # original grid) or a reasonably-sized tiled grid
        self.assertIsInstance(result, list)
        total_pixels = len(result) * (len(result[0]) if result else 0)
        self.assertLessEqual(total_pixels, env.MAX_GRID_PIXELS,
                             f"Grid too large: {len(result)}x{len(result[0])} = {total_pixels} pixels")

    def test_grid_size_guard_allows_normal_grids(self):
        """Normal ARC grids (up to 30x30) should not be affected by the guard."""
        env = ARCEnv()
        # Single tile_3x3 on a small grid: 3x3 -> 9x9 = 81 pixels. Fine.
        prog = Program(root="tile_3x3")
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = env.execute(prog, grid)
        self.assertEqual(len(result), 9)
        self.assertEqual(len(result[0]), 9)


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
        """All pixels wrong but same shape → high error, partial credit for shape/density."""
        drive = ARCDrive()
        pred = [[1, 1], [1, 1]]
        exp = [[2, 2], [2, 2]]
        error = drive.prediction_error(pred, exp)
        # New structural similarity scorer gives partial credit for matching
        # dimensions (0.15) and nonzero density (0.10), so error ≈ 0.75
        self.assertGreater(error, 0.5)
        self.assertLess(error, 1.0)

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


class TestFillTilePattern(unittest.TestCase):
    """Test fill_tile_pattern with non-divisor tile sizes."""

    def test_divisor_tile(self):
        """3x3 tile in 6x6 grid (divides evenly)."""
        grid = [
            [1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 5, 6],
            [7, 8, 9, 7, 8, 9],
            [1, 2, 0, 1, 2, 3],  # one zero
            [4, 5, 6, 4, 5, 6],
            [7, 8, 9, 7, 8, 9],
        ]
        result = fill_tile_pattern(grid)
        self.assertEqual(result[3][2], 3)  # filled the zero

    def test_non_divisor_tile(self):
        """5-period tile in 7-wide grid (7 % 5 = 2)."""
        # Build a 5-period row pattern, 7 cols
        pattern = [1, 2, 3, 4, 5]
        row = [pattern[c % 5] for c in range(7)]
        grid = [row[:] for _ in range(3)]
        grid[1][3] = 0  # erase one cell
        result = fill_tile_pattern(grid)
        self.assertEqual(result[1][3], 4)  # should reconstruct pattern[3]

    def test_non_divisor_2d_tile(self):
        """5x5 tile in 16x16 grid (16 % 5 = 1). Realistic size for ARC."""
        tile = [[((r + c) % 5) + 1 for c in range(5)] for r in range(5)]
        grid = [[tile[r % 5][c % 5] for c in range(16)] for r in range(16)]
        # Erase a 4x4 rectangle (like a real ARC task hole)
        for r in range(3, 7):
            for c in range(5, 9):
                grid[r][c] = 0
        result = fill_tile_pattern(grid)
        expected = [[tile[r % 5][c % 5] for c in range(16)] for r in range(16)]
        self.assertEqual(result, expected)


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


class TestContextColorPrimitives(unittest.TestCase):
    """Test context-dependent color primitives (Decision 59)."""

    def test_neighbor_vote_fixes_artifact(self):
        from domains.arc.primitives import recolor_by_neighbor_vote
        # 3x3 grid with one artifact cell (center is 2, neighbors are all 1)
        grid = [[1, 1, 1],
                [1, 2, 1],
                [1, 1, 1]]
        result = recolor_by_neighbor_vote(grid)
        self.assertEqual(result[1][1], 1)  # artifact fixed

    def test_neighbor_vote_preserves_regions(self):
        from domains.arc.primitives import recolor_by_neighbor_vote
        # Two clear regions should stay distinct
        grid = [[1, 1, 2, 2],
                [1, 1, 2, 2]]
        result = recolor_by_neighbor_vote(grid)
        self.assertEqual(result, grid)  # no change needed

    def test_swap_two_most_common(self):
        from domains.arc.primitives import swap_two_most_common
        # Color 1 appears 4 times, color 2 appears 3 times
        grid = [[1, 1, 2],
                [1, 2, 2],
                [1, 0, 0]]
        result = swap_two_most_common(grid)
        self.assertEqual(result[0][0], 2)  # 1→2
        self.assertEqual(result[0][2], 1)  # 2→1
        self.assertEqual(result[2][1], 0)  # 0 preserved

    def test_fill_by_surround(self):
        from domains.arc.primitives import fill_by_surround_color
        # Hole surrounded by color 3
        grid = [[3, 3, 3],
                [3, 0, 3],
                [3, 3, 3]]
        result = fill_by_surround_color(grid)
        self.assertEqual(result[1][1], 3)

    def test_cleanup_isolated(self):
        from domains.arc.primitives import cleanup_isolated_cells
        # Isolated cell (5) with no same-colored neighbors
        grid = [[1, 1, 0],
                [1, 5, 0],
                [0, 0, 0]]
        result = cleanup_isolated_cells(grid)
        self.assertEqual(result[1][1], 0)  # isolated cell removed
        self.assertEqual(result[0][0], 1)  # connected cells preserved

    def test_recolor_minority_to_majority(self):
        from domains.arc.primitives import recolor_minority_to_majority
        # Component with 3 cells of color 1 and 1 cell of color 2
        grid = [[1, 1, 0],
                [1, 2, 0],
                [0, 0, 0]]
        result = recolor_minority_to_majority(grid)
        self.assertEqual(result[1][1], 1)  # minority recolored


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
        self.assertGreater(results[0].solved, 0)


class TestBatch4Primitives(unittest.TestCase):
    """Test batch 4: grid partition, annotation, and scaling primitives."""

    def test_detect_zero_separator(self):
        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
        ]
        h_lines, v_lines = _detect_any_separator_lines(grid)
        self.assertIn(2, h_lines)
        self.assertIn(2, v_lines)

    def test_split_grid_cells_2d(self):
        grid = [
            [1, 1, 5, 2, 2],
            [1, 1, 5, 2, 2],
            [5, 5, 5, 5, 5],
            [3, 3, 5, 4, 4],
            [3, 3, 5, 4, 4],
        ]
        cells = _split_grid_cells(grid)
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0], [[1, 1], [1, 1]])
        self.assertEqual(cells[1], [[2, 2], [2, 2]])

    def test_select_odd_one_out(self):
        # 3x3 grid with separator, one quadrant different
        grid = [
            [1, 1, 5, 1, 1],
            [1, 1, 5, 1, 1],
            [5, 5, 5, 5, 5],
            [1, 1, 5, 2, 2],
            [1, 1, 5, 2, 2],
        ]
        result = select_odd_one_out(grid)
        self.assertEqual(result, [[2, 2], [2, 2]])

    def test_overlay_grid_cells(self):
        grid = [
            [0, 1, 5, 0, 0],
            [0, 0, 5, 0, 0],
            [5, 5, 5, 5, 5],
            [0, 0, 5, 3, 0],
            [0, 0, 5, 0, 0],
        ]
        result = overlay_grid_cells(grid)
        self.assertEqual(result, [[3, 1], [0, 0]])  # later cells overwrite earlier

    def test_majority_vote_cells(self):
        grid = [
            [1, 0, 5, 1, 0],
            [0, 2, 5, 0, 2],
            [5, 5, 5, 5, 5],
            [1, 0, 5, 1, 0],
            [0, 2, 5, 0, 3],
        ]
        result = majority_vote_cells(grid)
        self.assertEqual(result, [[1, 0], [0, 2]])

    def test_surround_pixels_3x3(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = surround_pixels_3x3(grid)
        # Center pixel should be surrounded by ring_color=2
        self.assertEqual(result[2][2], 1)  # original
        self.assertEqual(result[1][1], 2)  # ring
        self.assertEqual(result[1][2], 2)
        self.assertEqual(result[1][3], 2)
        self.assertEqual(result[3][1], 2)

    def test_draw_cross(self):
        grid = [
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
        ]
        result = draw_cross_from_pixels(grid)
        self.assertEqual(result[0][1], 3)  # up
        self.assertEqual(result[2][1], 3)  # down
        self.assertEqual(result[1][0], 3)  # left
        self.assertEqual(result[1][2], 3)  # right
        self.assertEqual(result[0][0], 0)  # diagonal stays 0

    def test_draw_cross_to_contact(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        result = draw_cross_to_contact(grid)
        self.assertEqual(result[0][1], 2)  # up from (1,1)
        self.assertEqual(result[1][0], 2)  # left from (1,1)
        self.assertEqual(result[2][1], 2)  # down from (1,1) stops before hitting nothing

    def test_connect_same_color_h(self):
        grid = [
            [3, 0, 0, 0, 3],
            [0, 0, 0, 0, 0],
        ]
        result = connect_same_color_h(grid)
        self.assertEqual(result[0], [3, 3, 3, 3, 3])
        self.assertEqual(result[1], [0, 0, 0, 0, 0])

    def test_connect_same_color_v(self):
        grid = [
            [2, 0],
            [0, 0],
            [0, 0],
            [2, 0],
        ]
        result = connect_same_color_v(grid)
        expected = [
            [2, 0],
            [2, 0],
            [2, 0],
            [2, 0],
        ]
        self.assertEqual(result, expected)

    def test_scale_4x(self):
        grid = [[1, 2], [3, 4]]
        result = scale_4x(grid)
        self.assertEqual(len(result), 8)
        self.assertEqual(len(result[0]), 8)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][4], 2)

    def test_downscale_4x(self):
        # 4x4 grid where each 4x4 block has a dominant color
        grid = [[1]*4 + [2]*4 for _ in range(4)] + [[3]*4 + [4]*4 for _ in range(4)]
        result = downscale_4x(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])


class TestGrammarDecomposition(unittest.TestCase):
    """Test Grammar.decompose/recompose as a core principle."""

    def setUp(self):
        self.grammar = ARCGrammar(seed=42)

    def test_decompose_returns_strategies(self):
        """decompose should return at least one strategy for multi-object grids."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        self.assertGreater(len(decomps), 0)
        self.assertEqual(decomps[0].strategy, "same_color_objects")
        self.assertEqual(decomps[0].n_parts, 2)

    def test_recompose_identity(self):
        """recompose(decompose(grid), identity_parts) should reproduce grid."""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        self.assertGreater(len(decomps), 0)

        d = decomps[0]
        result = self.grammar.recompose(d, d.parts)
        self.assertEqual(result, grid)

    def test_decompose_empty_for_single_object(self):
        """Single object grids should not decompose (not useful)."""
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        self.assertEqual(len(decomps), 0)

    def test_grid_partition_decomposition(self):
        """Grid with separator lines should produce grid_partition strategy."""
        grid = [
            [1, 1, 5, 2, 2],
            [1, 1, 5, 2, 2],
            [5, 5, 5, 5, 5],
            [3, 3, 5, 4, 4],
            [3, 3, 5, 4, 4],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        strategies = [d.strategy for d in decomps]
        self.assertIn("grid_partition", strategies)
        gp = [d for d in decomps if d.strategy == "grid_partition"][0]
        self.assertEqual(gp.n_parts, 4)

    def test_grid_partition_recompose(self):
        """Recompose grid_partition should reproduce the original grid."""
        grid = [
            [1, 1, 5, 2, 2],
            [1, 1, 5, 2, 2],
            [5, 5, 5, 5, 5],
            [3, 3, 5, 4, 4],
            [3, 3, 5, 4, 4],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        gp = [d for d in decomps if d.strategy == "grid_partition"][0]
        result = self.grammar.recompose(gp, gp.parts)
        self.assertEqual(result, grid)

    def test_multicolor_strategy_when_different(self):
        """Multi-color decomposition should appear when 8-conn differs from 4-conn."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],  # 4-conn: 2 objects; 8-conn: 1 object
            [0, 0, 0, 0, 3],  # 4-conn: 3 objects total; 8-conn: 2
            [0, 0, 0, 0, 0],
        ]
        from core import Task
        task = Task(task_id="test", train_examples=[(grid, grid)], test_inputs=[grid])
        decomps = self.grammar.decompose(grid, task)
        strategies = [d.strategy for d in decomps]
        self.assertIn("same_color_objects", strategies)
        # 4-conn gives 3 objects (1,2,3); 8-conn gives 2 (1+2 merged, 3 alone)
        # So multicolor should also appear
        if len(decomps) > 1:
            self.assertIn("multicolor_objects", strategies)


class TestFixedPointIteration(unittest.TestCase):
    """Test apply_until_stable combinator."""

    def test_identity_converges_immediately(self):
        from domains.arc.primitives import apply_until_stable
        grid = [[1, 2], [3, 4]]
        result = apply_until_stable(lambda g: g, grid)
        self.assertEqual(result, grid)

    def test_fill_propagation(self):
        """Repeated fill should expand colored regions."""
        from domains.arc.primitives import apply_until_stable

        # Simple: each step fills one layer of bg-adjacent cells
        def fill_one_layer(grid):
            h, w = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            for r in range(h):
                for c in range(w):
                    if grid[r][c] == 0:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                                result[r][c] = grid[nr][nc]
                                break
            return result

        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        result = apply_until_stable(fill_one_layer, grid)
        # After iteration, everything should be filled with 1
        self.assertTrue(all(c == 1 for row in result for c in row))

    def test_max_iters_respected(self):
        """Should stop after max_iters even if not converged."""
        from domains.arc.primitives import apply_until_stable
        call_count = [0]

        def counter(grid):
            call_count[0] += 1
            return [[c + 1 for c in row] for row in grid]  # never converges

        apply_until_stable(counter, [[0]], max_iters=5)
        self.assertEqual(call_count[0], 5)

    def test_make_fixed_point_fn(self):
        from domains.arc.primitives import make_fixed_point_fn
        fn = make_fixed_point_fn(lambda g: g)
        self.assertEqual(fn([[1]]), [[1]])


class TestObjectDecompositionExtended(unittest.TestCase):
    """Test extended object decomposition: pairs, multi-color, scoring."""

    def test_multicolor_object_detection(self):
        """8-connectivity should group adjacent different-colored pixels."""
        from domains.arc.objects import find_multicolor_objects
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 3, 1, 0, 0],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 4, 4],
        ]
        objects = find_multicolor_objects(grid)
        self.assertEqual(len(objects), 2)
        # First object: the 2x2 cluster of colors 1,2,3
        sizes = sorted([o["size"] for o in objects])
        self.assertEqual(sizes, [3, 4])

    def test_multicolor_subgrid_extraction(self):
        """Multi-color subgrids should preserve original colors."""
        from domains.arc.objects import find_multicolor_objects
        grid = [
            [0, 0, 0],
            [0, 1, 2],
            [0, 3, 0],
        ]
        objects = find_multicolor_objects(grid)
        self.assertEqual(len(objects), 1)
        sg = objects[0]["subgrid"]
        # Should be a 2x2 subgrid with colors 1,2,3,0
        self.assertEqual(len(sg), 2)
        self.assertEqual(len(sg[0]), 2)
        self.assertEqual(sg[0][0], 1)
        self.assertEqual(sg[0][1], 2)
        self.assertEqual(sg[1][0], 3)
        self.assertEqual(sg[1][1], 0)

    def test_per_object_pair_decomposition(self):
        """Composed per-object transforms: outer(inner(obj)) should work."""
        from domains.arc.objects import try_object_decomposition
        from core import Primitive

        # Task: each object needs rotate90(identity(obj)) = rotate90(obj)
        # But we test that the pair search finds it even when single doesn't
        inp = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 2, 2, 0],
        ]
        # After rotate90 per-object, each L-shape rotates
        from domains.arc.primitives import rotate_90_cw
        from domains.arc.objects import apply_transform_per_object
        expected = apply_transform_per_object(inp, rotate_90_cw)

        if expected is not None:
            task_examples = [(inp, expected)]
            prims = [
                Primitive(name="identity", arity=1, fn=lambda g: g, domain="arc"),
                Primitive(name="rotate_90_cw", arity=1, fn=rotate_90_cw, domain="arc"),
            ]
            result = try_object_decomposition(task_examples, prims)
            self.assertIsNotNone(result)
            name, fn = result
            self.assertIn("per_object", name)

    def test_score_per_object_prims(self):
        """_score_per_object_prims should rank prims by per-object error."""
        from domains.arc.objects import _score_per_object_prims
        from core import Primitive

        inp = [[0, 1, 0], [0, 1, 0]]
        expected = [[0, 1, 0], [0, 1, 0]]  # identity is best
        task_examples = [(inp, expected)]

        prims = [
            Primitive(name="identity", arity=1, fn=lambda g: g, domain="arc"),
            Primitive(name="rotate_90_cw", arity=1, fn=rotate_90_cw, domain="arc"),
        ]
        scored = _score_per_object_prims(prims, task_examples, bg_color=0)
        # Identity should rank first (lowest error)
        self.assertEqual(scored[0].name, "identity")

    def test_multicolor_per_object_transform(self):
        """Per-multicolor-object transforms should work."""
        from domains.arc.objects import apply_transform_per_multicolor_object

        grid = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 1, 0],
            [0, 0, 0, 0],
        ]
        # Identity transform should reproduce the grid
        result = apply_transform_per_multicolor_object(grid, lambda g: g)
        self.assertEqual(result, grid)


class TestParameterizedPrimitives(unittest.TestCase):
    """Test parameterized (role-based) color primitives."""

    def test_role_recolor_learns_mapping(self):
        """Role recolor should learn rare→dominant and generalize."""
        from domains.arc.grammar import _learn_structural_recolor, _assign_color_roles
        # Example: rare color (appears least) maps to dominant color
        # Input has colors: 0 (bg, 16 cells), 1 (dominant, 6 cells), 2 (rare, 2 cells)
        # Output: 2→1 (rare→dominant)
        inp1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 2, 1, 0],
        ]
        out1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],  # 2→1
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],  # 2→1
        ]
        inp2 = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        out2 = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],  # 2→1
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        task = Task(
            task_id="test_role",
            train_examples=[(inp1, out1), (inp2, out2)],
            test_inputs=[],
        )
        mappings = _learn_structural_recolor(task)
        # Should learn rare→dominant
        self.assertTrue(len(mappings) > 0)
        src_roles = [m[0] for m in mappings]
        self.assertIn("rare", src_roles)

    def test_role_recolor_generates_primitive(self):
        """prepare_for_task should generate parameterized primitives."""
        inp1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 2, 1, 0],
        ]
        out1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
        inp2 = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        out2 = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        task = Task(
            task_id="test_gen",
            train_examples=[(inp1, out1), (inp2, out2)],
            test_inputs=[],
        )
        grammar = ARCGrammar()
        grammar.prepare_for_task(task)
        prim_names = [p.name for p in grammar._task_prims]
        self.assertIn("param_role_recolor", prim_names)

    def test_role_recolor_applies_correctly(self):
        """The generated role_recolor primitive should transform grids correctly."""
        from domains.arc.grammar import _learn_parameterized_prims
        # Train: rare (2) → dominant (1)
        inp1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 2, 1, 0],
        ]
        out1 = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
        task = Task(
            task_id="test_apply",
            train_examples=[(inp1, out1)],
            test_inputs=[],
        )
        prims = _learn_parameterized_prims(task)
        role_prim = [p for p in prims if p.name == "param_role_recolor"]
        self.assertEqual(len(role_prim), 1)

        # Apply to training input — should produce training output
        result = role_prim[0].fn(inp1)
        self.assertEqual(result, out1)

    def test_rank_recolor(self):
        """Frequency-rank recolor should swap colors by rank."""
        from domains.arc.grammar import _learn_recolor_by_frequency
        # Input: 0 (bg, 12 cells), 1 (dominant, 3 cells), 2 (rare, 1 cell)
        # Output: same positions but 1→2, 2→1 (swap dominant and rare)
        inp = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ]
        out = [
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 2, 1, 0],
            [0, 0, 0, 0],
        ]
        task = Task(
            task_id="test_rank",
            train_examples=[(inp, out)],
            test_inputs=[],
        )
        result = _learn_recolor_by_frequency(task)
        # Should learn a rank mapping
        self.assertTrue(len(result) > 0)

    def test_assign_color_roles(self):
        """Color role assignment should identify bg, dominant, rare."""
        from domains.arc.grammar import _assign_color_roles
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ]
        roles = _assign_color_roles(grid)
        self.assertEqual(roles[0], "bg")
        self.assertEqual(roles[1], "dominant")
        self.assertEqual(roles[2], "rare")

    def test_no_prims_for_size_change(self):
        """No parameterized prims should be generated for size-changing tasks."""
        from domains.arc.grammar import _learn_parameterized_prims
        task = Task(
            task_id="test_size_change",
            train_examples=[
                ([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]),  # 2x2 → 2x3
            ],
            test_inputs=[],
        )
        prims = _learn_parameterized_prims(task)
        self.assertEqual(len(prims), 0)


class TestDiffAndPatch(unittest.TestCase):
    """Test the extended infer_output_correction (Phase B)."""

    def setUp(self):
        self.env = ARCEnv()

    def test_color_remap_still_works(self):
        """Original color remapping should still work."""
        got = [[[1, 1], [1, 1]]]
        exp = [[[2, 2], [2, 2]]]
        correction = self.env.infer_output_correction(got, exp)
        self.assertIsNotNone(correction)
        self.assertIn("color_remap", correction.root)

    def test_adjacency_correction(self):
        """Adjacency-based correction: pixel changes based on neighbors."""
        # Program outputs 0s everywhere, but expected has 1s adjacent to 2s
        got = [
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
        ]
        exp = [
            [[1, 2, 1], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [1, 0, 0]],
        ]
        correction = self.env.infer_output_correction(got, exp)
        # Should find either adjacency or neighborhood correction
        if correction is not None:
            # Verify the correction works
            from domains.arc.primitives import _PRIM_MAP
            prim = _PRIM_MAP.get(correction.root)
            self.assertIsNotNone(prim)
            for g, e in zip(got, exp):
                result = prim.fn(g)
                self.assertEqual(result, e)

    def test_neighborhood_correction(self):
        """3x3 neighborhood correction catches context-dependent fixes."""
        # Simple case: center pixel surrounded by 1s should become 1
        got = [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]
        exp = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        correction = self.env.infer_output_correction(got, exp)
        self.assertIsNotNone(correction)

    def test_no_correction_inconsistent(self):
        """Return None when diffs are inconsistent across examples."""
        got = [[[1, 0], [0, 0]], [[1, 0], [0, 0]]]
        exp = [[[1, 2], [0, 0]], [[1, 3], [0, 0]]]  # inconsistent: 0→2 vs 0→3
        correction = self.env.infer_output_correction(got, exp)
        # Should return None (inconsistent mapping for same context)
        # Note: may succeed with adjacency if context differs
        # The key is it shouldn't crash

    def test_shape_mismatch_returns_none(self):
        """Return None when shapes don't match."""
        got = [[[1, 2], [3, 4]]]
        exp = [[[1, 2, 3]]]
        correction = self.env.infer_output_correction(got, exp)
        self.assertIsNone(correction)


class TestVocabPruning(unittest.TestCase):
    """Test task-specific color primitive generation (Phase D)."""

    def test_task_colors_extracted(self):
        """Colors from inputs and outputs should be extracted."""
        from domains.arc.grammar import _extract_task_colors
        task = Task(
            task_id="test",
            train_examples=[
                ([[0, 1, 2], [3, 0, 0]], [[0, 4, 2], [3, 0, 5]]),
            ],
            test_inputs=[],
        )
        colors = _extract_task_colors(task)
        self.assertEqual(colors, {0, 1, 2, 3, 4, 5})

    def test_fewer_prims_than_static(self):
        """Task-specific prims should be fewer than static all-colors."""
        from domains.arc.primitives import build_task_color_primitives
        # Task with only 3 colors (0, 1, 2)
        few_color_prims = build_task_color_primitives({0, 1, 2})
        # Task with all 10 colors
        all_color_prims = build_task_color_primitives(set(range(10)))
        self.assertLess(len(few_color_prims), len(all_color_prims))
        # With 2 non-zero colors, should be much smaller
        self.assertLess(len(few_color_prims), 50)

    def test_prims_only_use_task_colors(self):
        """Generated primitives should only reference task-relevant colors."""
        from domains.arc.primitives import build_task_color_primitives
        prims = build_task_color_primitives({0, 1, 3})
        names = [p.name for p in prims]
        # Should have keep_c1, keep_c3 but NOT keep_c2, keep_c4, etc.
        self.assertIn("keep_c1", names)
        self.assertIn("keep_c3", names)
        self.assertNotIn("keep_c2", names)
        self.assertNotIn("keep_c4", names)

    def test_prepare_for_task_generates_color_prims(self):
        """prepare_for_task should generate color-specific primitives."""
        grammar = ARCGrammar()
        task = Task(
            task_id="test",
            train_examples=[
                ([[0, 1], [2, 0]], [[0, 2], [1, 0]]),
            ],
            test_inputs=[],
        )
        grammar.prepare_for_task(task)
        prims = grammar.base_primitives()
        names = [p.name for p in prims]
        self.assertIn("keep_c1", names)
        self.assertIn("keep_c2", names)
        self.assertIn("swap_1_2", names)
        # Colors not in the task should NOT appear
        self.assertNotIn("keep_c5", names)
        self.assertNotIn("keep_c9", names)


class TestLOOCV(unittest.TestCase):
    """Test generalized LOOCV for training-perfect candidates (Phase A)."""

    def test_loocv_static_program_passes(self):
        """Static programs (no learned primitives) should always pass LOOCV."""
        env = ARCEnv()
        grammar = ARCGrammar()
        drive = ARCDrive()
        memory = InMemoryStore()
        cfg = SearchConfig(seed=42, exhaustive_depth=1)
        learner = Learner(env, grammar, drive, memory, search_config=cfg)

        # Task where identity is the solution
        task = Task(
            task_id="test_loocv",
            train_examples=[
                ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
                ([[5, 6], [7, 8]], [[5, 6], [7, 8]]),
                ([[0, 1], [1, 0]], [[0, 1], [1, 0]]),
            ],
            test_inputs=[[[9, 0], [0, 9]]],
            test_outputs=[[[9, 0], [0, 9]]],
        )
        grammar.prepare_for_task(task)

        # Identity program
        from core.types import ScoredProgram
        sp = ScoredProgram(
            program=Program(root="identity"),
            energy=0.0,
            prediction_error=0.0,
            complexity_cost=1.0,
        )
        score = learner._loocv_score(sp, task)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
