"""Tests for experiments/visualize_results.py — learned-abstraction expansion,
grid rendering robustness (inhomogeneous grids, malformed predictions)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.types import Program
from experiments.visualize_results import (
    _build_library_map,
    _expand_learned,
    _format_expanded_program,
    _graft_children,
    _safe_grid_array,
    parse_program_tree,
    render_grid,
    _pred_border,
)


class TestParseProgram(unittest.TestCase):
    """Baseline: ensure parse_program_tree works correctly."""

    def test_leaf(self):
        p = parse_program_tree("identity")
        self.assertEqual(p.root, "identity")
        self.assertEqual(p.children, [])

    def test_one_child(self):
        p = parse_program_tree("mirror_h(rotate_90)")
        self.assertEqual(p.root, "mirror_h")
        self.assertEqual(len(p.children), 1)
        self.assertEqual(p.children[0].root, "rotate_90")

    def test_nested(self):
        p = parse_program_tree("a(b(c))")
        self.assertEqual(p.root, "a")
        self.assertEqual(p.children[0].root, "b")
        self.assertEqual(p.children[0].children[0].root, "c")

    def test_empty(self):
        self.assertIsNone(parse_program_tree(""))
        self.assertIsNone(parse_program_tree("   "))

    def test_dynamic_prim_leaf(self):
        """Dynamic primitives with parenthesized names should be leaf nodes."""
        cases = [
            "half_colormap(hsplit_sep)",
            "per_object_recolor(by_size)",
            "cell_patch(14_cells)",
            "procedural(color:fill_object_bbox)",
            "input_pred_correct(trim_cols(gravity_down))",
            "transform_colormap(flood_fill)",
            "cond_bbox_fill(compactness)",
            "nway_colormap(vsplit3_nosep)",
            "quad_colormap(quad)",
            "pixel_to_tile(2x2)",
        ]
        for name in cases:
            p = parse_program_tree(name)
            self.assertEqual(p.root, name, f"Expected leaf for {name}")
            self.assertEqual(p.children, [], f"Expected no children for {name}")

    def test_dynamic_prim_inside_composition(self):
        """Dynamic prims used as children of real compositions."""
        p = parse_program_tree("overlay(half_colormap(vsplit), extract_unique_quadrant)")
        self.assertEqual(p.root, "overlay")
        self.assertEqual(len(p.children), 2)
        self.assertEqual(p.children[0].root, "half_colormap(vsplit)")
        self.assertEqual(p.children[0].children, [])
        self.assertEqual(p.children[1].root, "extract_unique_quadrant")

    def test_conditional_leaf(self):
        """if_ programs are leaf nodes."""
        p = parse_program_tree("if_is_tall_trim_rows_else_trim_cols")
        self.assertEqual(p.root, "if_is_tall_trim_rows_else_trim_cols")
        self.assertEqual(p.children, [])

    def test_regular_composition_unchanged(self):
        """Standard compositions should still parse normally."""
        p = parse_program_tree("crop_half_left(learned_14)")
        self.assertEqual(p.root, "crop_half_left")
        self.assertEqual(len(p.children), 1)
        self.assertEqual(p.children[0].root, "learned_14")


class TestBuildLibraryMap(unittest.TestCase):
    """Test _build_library_map from results JSON structure."""

    def test_empty_results(self):
        self.assertEqual(_build_library_map({}), {})
        self.assertEqual(_build_library_map({"library": []}), {})

    def test_single_entry(self):
        results = {
            "library": [
                {"name": "learned_0", "program": "mirror_h(rotate_90)"}
            ]
        }
        lm = _build_library_map(results)
        self.assertIn("learned_0", lm)
        self.assertEqual(lm["learned_0"].root, "mirror_h")
        self.assertEqual(len(lm["learned_0"].children), 1)
        self.assertEqual(lm["learned_0"].children[0].root, "rotate_90")

    def test_multiple_entries(self):
        results = {
            "library": [
                {"name": "learned_0", "program": "mirror_h(rotate_90)"},
                {"name": "learned_1", "program": "crop_to_content"},
            ]
        }
        lm = _build_library_map(results)
        self.assertEqual(len(lm), 2)
        self.assertIn("learned_0", lm)
        self.assertIn("learned_1", lm)
        self.assertEqual(lm["learned_1"].root, "crop_to_content")
        self.assertEqual(lm["learned_1"].children, [])

    def test_invalid_entry_skipped(self):
        results = {
            "library": [
                {"name": "", "program": "mirror_h"},
                {"name": "learned_0", "program": ""},
                {"name": "learned_1", "program": "crop_to_content"},
            ]
        }
        lm = _build_library_map(results)
        self.assertEqual(len(lm), 1)
        self.assertIn("learned_1", lm)


class TestExpandLearned(unittest.TestCase):
    """Test _expand_learned: recursive inline expansion of learned entries."""

    def test_no_library(self):
        prog = Program(root="mirror_h", children=[Program(root="rotate_90")])
        result = _expand_learned(prog, {})
        self.assertEqual(repr(result), "mirror_h(rotate_90)")

    def test_leaf_expansion(self):
        """learned_0 as a leaf: crop_half_left(learned_0)
        where learned_0 = crop_half_top(crop_to_content)
        should become crop_half_left(crop_half_top(crop_to_content))
        """
        library_map = {
            "learned_0": Program(root="crop_half_top",
                                 children=[Program(root="crop_to_content")])
        }
        prog = Program(root="crop_half_left",
                       children=[Program(root="learned_0")])
        result = _expand_learned(prog, library_map)
        self.assertEqual(repr(result),
                         "crop_half_left(crop_half_top(crop_to_content))")

    def test_root_expansion(self):
        """learned_0 as the root: learned_0
        where learned_0 = mirror_h(rotate_90)
        should become mirror_h(rotate_90)
        """
        library_map = {
            "learned_0": Program(root="mirror_h",
                                 children=[Program(root="rotate_90")])
        }
        prog = Program(root="learned_0")
        result = _expand_learned(prog, library_map)
        self.assertEqual(repr(result), "mirror_h(rotate_90)")

    def test_nested_learned(self):
        """Learned entry containing another learned entry.
        learned_1 = crop_half_top(learned_0)
        learned_0 = crop_to_content
        crop_half_left(learned_1)
        should become crop_half_left(crop_half_top(crop_to_content))
        """
        library_map = {
            "learned_0": Program(root="crop_to_content"),
            "learned_1": Program(root="crop_half_top",
                                 children=[Program(root="learned_0")])
        }
        prog = Program(root="crop_half_left",
                       children=[Program(root="learned_1")])
        result = _expand_learned(prog, library_map)
        self.assertEqual(repr(result),
                         "crop_half_left(crop_half_top(crop_to_content))")

    def test_no_match_passthrough(self):
        """Primitives not in library_map should pass through unchanged."""
        library_map = {"learned_0": Program(root="mirror_h")}
        prog = Program(root="rotate_90")
        result = _expand_learned(prog, library_map)
        self.assertEqual(repr(result), "rotate_90")

    def test_learned_with_children(self):
        """learned_0(some_prim) where learned_0 = a(b)
        should become a(b(some_prim))
        """
        library_map = {
            "learned_0": Program(root="a", children=[Program(root="b")])
        }
        prog = Program(root="learned_0",
                       children=[Program(root="some_prim")])
        result = _expand_learned(prog, library_map)
        self.assertEqual(repr(result), "a(b(some_prim))")


class TestGraftChildren(unittest.TestCase):
    """Test _graft_children: attaching children to innermost leaf."""

    def test_leaf_target(self):
        tree = Program(root="a")
        children = [Program(root="x")]
        result = _graft_children(tree, children)
        self.assertEqual(repr(result), "a(x)")

    def test_one_level_deep(self):
        tree = Program(root="a", children=[Program(root="b")])
        children = [Program(root="x")]
        result = _graft_children(tree, children)
        self.assertEqual(repr(result), "a(b(x))")

    def test_two_levels_deep(self):
        tree = Program(root="a",
                       children=[Program(root="b",
                                         children=[Program(root="c")])])
        children = [Program(root="x")]
        result = _graft_children(tree, children)
        self.assertEqual(repr(result), "a(b(c(x)))")


class TestFormatExpandedProgram(unittest.TestCase):
    """Test _format_expanded_program: human-readable display string."""

    def test_no_library(self):
        result = _format_expanded_program("crop_half_left(crop_to_content)", {})
        self.assertEqual(result, "crop_half_left(crop_to_content)")

    def test_with_learned(self):
        library_map = {
            "learned_14": Program(root="crop_half_top",
                                  children=[Program(root="crop_to_content")])
        }
        result = _format_expanded_program("crop_half_left(learned_14)",
                                          library_map)
        self.assertEqual(result,
                         "crop_half_left(learned_14=crop_half_top(crop_to_content))")

    def test_learned_at_root(self):
        library_map = {
            "learned_0": Program(root="mirror_h",
                                 children=[Program(root="rotate_90")])
        }
        result = _format_expanded_program("learned_0", library_map)
        self.assertEqual(result, "learned_0=mirror_h(rotate_90)")

    def test_nested_learned_display(self):
        """Two levels of learned: outer shows both expansions."""
        library_map = {
            "learned_0": Program(root="crop_to_content"),
            "learned_1": Program(root="crop_half_top",
                                 children=[Program(root="learned_0")])
        }
        result = _format_expanded_program("some_op(learned_1)", library_map)
        self.assertEqual(
            result,
            "some_op(learned_1=crop_half_top(learned_0=crop_to_content))"
        )

    def test_empty_string(self):
        library_map = {"learned_0": Program(root="x")}
        result = _format_expanded_program("", library_map)
        self.assertEqual(result, "")

    def test_no_learned_in_program(self):
        library_map = {"learned_0": Program(root="x")}
        result = _format_expanded_program("mirror_h(rotate_90)", library_map)
        self.assertEqual(result, "mirror_h(rotate_90)")


class TestSafeGridArray(unittest.TestCase):
    """Test _safe_grid_array handles malformed grids gracefully."""

    def test_normal_grid(self):
        """Normal 2D grid converts correctly."""
        import numpy as np
        arr = _safe_grid_array([[1, 2], [3, 4]])
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr[0, 0], 1)

    def test_inhomogeneous_rows_padded(self):
        """Rows of different lengths are padded with 0."""
        import numpy as np
        arr = _safe_grid_array([[1, 2, 3], [4, 5]])
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr[1, 2], 0)  # padded

    def test_nested_cells_flattened(self):
        """Cells containing lists are flattened (take first element)."""
        import numpy as np
        arr = _safe_grid_array([[1, [2, 3]], [4, 5]])
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr[0, 1], 2)  # first element of [2, 3]

    def test_none_returns_none(self):
        self.assertIsNone(_safe_grid_array(None))

    def test_empty_returns_none(self):
        self.assertIsNone(_safe_grid_array([]))

    def test_non_list_returns_none(self):
        self.assertIsNone(_safe_grid_array("not a grid"))


class TestRenderGridRobust(unittest.TestCase):
    """Test render_grid handles edge cases without crashing."""

    def test_normal_grid(self):
        result = render_grid([[1, 2], [3, 4]])
        self.assertIn('grid-wrapper', result)
        self.assertNotIn('malformed', result)

    def test_inhomogeneous_grid_renders(self):
        """Inhomogeneous grid renders instead of crashing."""
        result = render_grid([[1, 2, 3], [4, 5]])
        self.assertIn('grid-wrapper', result)
        self.assertNotIn('malformed', result)  # padded, not malformed

    def test_empty_grid(self):
        result = render_grid([])
        self.assertIn('empty', result)

    def test_none_grid(self):
        result = render_grid(None)
        self.assertIn('empty', result)

    def test_deeply_nested_grid(self):
        """Grid with nested cells doesn't crash."""
        result = render_grid([[1, [2, 3]], [4, 5]])
        self.assertIn('grid-wrapper', result)


class TestPredBorderRobust(unittest.TestCase):
    """Test _pred_border handles malformed grids."""

    def test_matching_grids(self):
        border, diff = _pred_border([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        self.assertEqual(border, "match-border")
        self.assertIsNone(diff)

    def test_different_grids(self):
        border, diff = _pred_border([[1, 2], [3, 4]], [[1, 2], [3, 5]])
        self.assertEqual(border, "diff-border")

    def test_different_shapes(self):
        border, diff = _pred_border([[1, 2]], [[1, 2], [3, 4]])
        self.assertEqual(border, "diff-border")
        self.assertIsNone(diff)

    def test_inhomogeneous_prediction(self):
        """Inhomogeneous prediction doesn't crash."""
        border, diff = _pred_border([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(border, "diff-border")

    def test_none_prediction(self):
        border, diff = _pred_border(None, [[1, 2]])
        self.assertEqual(border, "")


class TestExecuteStepsWithLibrary(unittest.TestCase):
    """Test _execute_steps with library_map expansion.

    Uses real ARCEnv primitives to verify step-by-step execution
    produces correct intermediate grids.
    """

    def setUp(self):
        from domains.arc import ARCEnv
        self.env = ARCEnv()

    def test_steps_without_library(self):
        """Basic steps work without library."""
        from experiments.visualize_results import _execute_steps
        # identity is a known primitive
        prog = Program(root="identity")
        grid = [[1, 2], [3, 4]]
        steps = _execute_steps(prog, grid, self.env)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["name"], "identity")
        self.assertEqual(steps[0]["output"], [[1, 2], [3, 4]])

    def test_steps_with_composition(self):
        """Composed program shows each step."""
        from experiments.visualize_results import _execute_steps
        # mirror_horizontal(identity) should show 2 steps
        prog = Program(root="mirror_horizontal",
                       children=[Program(root="identity")])
        grid = [[1, 2], [3, 4]]
        steps = _execute_steps(prog, grid, self.env)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["name"], "identity")
        self.assertEqual(steps[1]["name"], "mirror_horizontal")

    def test_steps_with_library_expansion(self):
        """Learned entry should be expanded into individual steps."""
        from experiments.visualize_results import _execute_steps
        from domains.arc.primitives import _PRIM_MAP
        from core.types import Primitive

        # Register a learned primitive that wraps mirror_horizontal(identity)
        inner_prog = Program(root="mirror_horizontal",
                             children=[Program(root="identity")])
        _PRIM_MAP["learned_test_0"] = Primitive(
            name="learned_test_0", arity=0, fn=inner_prog,
            domain="arc", learned=True
        )

        library_map = {
            "learned_test_0": inner_prog,
        }

        # Without library_map: 1 step (opaque learned_test_0)
        prog = Program(root="learned_test_0")
        grid = [[1, 2], [3, 4]]
        steps_opaque = _execute_steps(prog, grid, self.env, library_map=None)
        self.assertEqual(len(steps_opaque), 1)
        self.assertEqual(steps_opaque[0]["name"], "learned_test_0")

        # With library_map: 2 steps (identity, then mirror_horizontal)
        steps_expanded = _execute_steps(prog, grid, self.env,
                                        library_map=library_map)
        self.assertEqual(len(steps_expanded), 2)
        self.assertEqual(steps_expanded[0]["name"], "identity")
        self.assertEqual(steps_expanded[1]["name"], "mirror_horizontal")

        # Clean up
        del _PRIM_MAP["learned_test_0"]

    def test_steps_learned_in_composition(self):
        """outer(learned_0) where learned_0 = a(b) expands to outer(a(b))."""
        from experiments.visualize_results import _execute_steps
        from domains.arc.primitives import _PRIM_MAP
        from core.types import Primitive

        inner_prog = Program(root="identity")
        _PRIM_MAP["learned_test_1"] = Primitive(
            name="learned_test_1", arity=0, fn=inner_prog,
            domain="arc", learned=True
        )

        library_map = {
            "learned_test_1": inner_prog,
        }

        # mirror_horizontal(learned_test_1) should expand to
        # mirror_horizontal(identity) -> 2 steps
        prog = Program(root="mirror_horizontal",
                       children=[Program(root="learned_test_1")])
        grid = [[1, 2], [3, 4]]
        steps = _execute_steps(prog, grid, self.env, library_map=library_map)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["name"], "identity")
        self.assertEqual(steps[1]["name"], "mirror_horizontal")

        # Clean up
        del _PRIM_MAP["learned_test_1"]


if __name__ == "__main__":
    unittest.main()
