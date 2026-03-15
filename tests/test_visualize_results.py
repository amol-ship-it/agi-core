"""Tests for experiments/visualize_results.py learned-abstraction expansion."""

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
    parse_program_tree,
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
        self.assertEqual(steps[0][0], "identity")
        self.assertEqual(steps[0][1], [[1, 2], [3, 4]])

    def test_steps_with_composition(self):
        """Composed program shows each step."""
        from experiments.visualize_results import _execute_steps
        # mirror_horizontal(identity) should show 2 steps
        prog = Program(root="mirror_horizontal",
                       children=[Program(root="identity")])
        grid = [[1, 2], [3, 4]]
        steps = _execute_steps(prog, grid, self.env)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][0], "identity")
        self.assertEqual(steps[1][0], "mirror_horizontal")

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
        self.assertEqual(steps_opaque[0][0], "learned_test_0")

        # With library_map: 2 steps (identity, then mirror_horizontal)
        steps_expanded = _execute_steps(prog, grid, self.env,
                                        library_map=library_map)
        self.assertEqual(len(steps_expanded), 2)
        self.assertEqual(steps_expanded[0][0], "identity")
        self.assertEqual(steps_expanded[1][0], "mirror_horizontal")

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
        self.assertEqual(steps[0][0], "identity")
        self.assertEqual(steps[1][0], "mirror_horizontal")

        # Clean up
        del _PRIM_MAP["learned_test_1"]


if __name__ == "__main__":
    unittest.main()
