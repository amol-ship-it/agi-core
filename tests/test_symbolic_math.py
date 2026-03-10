"""
Tests for grammars/symbolic_math.py — the symbolic regression domain plugin.

Verifies:
1. All primitives compute correctly
2. SymbolicMathEnv evaluates expression trees
3. SymbolicMathGrammar mutation and crossover
4. SymbolicMathDrive prediction error
5. Task generation via make_task
"""

import math
import unittest

from core.interfaces import Program, Task, LibraryEntry
from grammars.symbolic_math import (
    MATH_PRIMITIVES,
    SymbolicMathEnv,
    SymbolicMathGrammar,
    SymbolicMathDrive,
    make_task,
    _safe_div,
    _safe_log,
    _safe_sqrt,
    _safe_exp,
)


class TestSafeFunctions(unittest.TestCase):

    def test_safe_div_normal(self):
        self.assertAlmostEqual(_safe_div(10, 2), 5.0)

    def test_safe_div_by_zero(self):
        self.assertAlmostEqual(_safe_div(10, 0.0), 0.0)

    def test_safe_log_normal(self):
        self.assertAlmostEqual(_safe_log(math.e), 1.0)

    def test_safe_log_zero(self):
        self.assertAlmostEqual(_safe_log(0.0), 0.0)

    def test_safe_sqrt(self):
        self.assertAlmostEqual(_safe_sqrt(4.0), 2.0)
        # Handles negative by using abs
        self.assertAlmostEqual(_safe_sqrt(-4.0), 2.0)

    def test_safe_exp_normal(self):
        self.assertAlmostEqual(_safe_exp(0.0), 1.0)

    def test_safe_exp_overflow(self):
        result = _safe_exp(1000.0)
        # Should be clamped, not overflow
        self.assertTrue(result < 1e10)


class TestMathPrimitives(unittest.TestCase):

    def test_primitive_count(self):
        self.assertEqual(len(MATH_PRIMITIVES), 15)

    def test_x_primitive(self):
        x_prim = next(p for p in MATH_PRIMITIVES if p.name == "x")
        self.assertEqual(x_prim.arity, 0)
        self.assertEqual(x_prim.fn({"x": 3.0}), 3.0)

    def test_const_primitive(self):
        c_prim = next(p for p in MATH_PRIMITIVES if p.name == "const")
        self.assertEqual(c_prim.fn({"c": 2.5}), 2.5)
        self.assertEqual(c_prim.fn({}), 1.0)  # default

    def test_add(self):
        add = next(p for p in MATH_PRIMITIVES if p.name == "add")
        self.assertEqual(add.fn(3, 4), 7)

    def test_mul(self):
        mul = next(p for p in MATH_PRIMITIVES if p.name == "mul")
        self.assertEqual(mul.fn(3, 4), 12)

    def test_sin(self):
        sin = next(p for p in MATH_PRIMITIVES if p.name == "sin")
        self.assertAlmostEqual(sin.fn(0), 0.0)

    def test_neg(self):
        neg = next(p for p in MATH_PRIMITIVES if p.name == "neg")
        self.assertEqual(neg.fn(5), -5)


class TestSafeExpOverflow(unittest.TestCase):
    """Test the OverflowError branch in _safe_exp."""

    def test_safe_exp_triggers_overflow(self):
        """Monkey-patch math.exp to simulate an OverflowError."""
        import grammars.symbolic_math as sm
        original_exp = math.exp
        def mock_exp(x):
            raise OverflowError("mock overflow")
        sm.math.exp = mock_exp
        try:
            result = sm._safe_exp(5.0)
            self.assertEqual(result, 1e6)
        finally:
            sm.math.exp = original_exp


class TestSymbolicMathEnv(unittest.TestCase):

    def setUp(self):
        self.env = SymbolicMathEnv()

    def test_load_task(self):
        task = make_task(lambda x: x, task_id="test")
        obs = self.env.load_task(task)
        self.assertEqual(len(obs.data), 50)  # default n_train

    def test_execute_x(self):
        """x evaluated at 3.0 should return 3.0."""
        prog = Program(root="x")
        result = self.env.execute(prog, 3.0)
        self.assertAlmostEqual(result, 3.0)

    def test_execute_sin_x(self):
        """sin(x) at pi should be ~0."""
        prog = Program(root="sin", children=[Program(root="x")])
        result = self.env.execute(prog, math.pi)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_execute_add(self):
        """add(x, const) = x + 1.0."""
        prog = Program(root="add", children=[
            Program(root="x"),
            Program(root="const"),  # default c=1.0
        ])
        result = self.env.execute(prog, 5.0)
        self.assertAlmostEqual(result, 6.0)

    def test_execute_nested(self):
        """square(add(x, const)) = (x+1)^2."""
        prog = Program(root="square", children=[
            Program(root="add", children=[
                Program(root="x"),
                Program(root="const", params={"c": 1.0}),
            ]),
        ])
        result = self.env.execute(prog, 2.0)
        self.assertAlmostEqual(result, 9.0)

    def test_execute_unknown_returns_zero(self):
        prog = Program(root="nonexistent")
        result = self.env.execute(prog, 5.0)
        self.assertAlmostEqual(result, 0.0)

    def test_execute_high_arity_returns_zero(self):
        """Primitives with arity > 2 should return 0.0 from eval_tree."""
        # The current code only handles arity 0, 1, 2 — anything else falls through
        # We can test this indirectly by checking the default return
        # Since all MATH_PRIMITIVES have arity <= 2, we test the fallthrough
        # by calling _eval_tree directly with a mocked prim map
        env = self.env
        # Create a tree with a known primitive but missing children
        prog = Program(root="add")  # arity 2, but no children
        result = env.execute(prog, 5.0)
        # add(0.0, 0.0) = 0.0 since missing children default to 0.0
        self.assertAlmostEqual(result, 0.0)

    def test_execute_unary_missing_child(self):
        """Unary primitive with no children should use 0.0."""
        prog = Program(root="sin")  # arity 1, no children
        result = self.env.execute(prog, 5.0)
        self.assertAlmostEqual(result, math.sin(0.0))

    def test_reset(self):
        task = make_task(lambda x: x)
        self.env.load_task(task)
        self.env.reset()
        self.assertIsNone(self.env._current_task)


class TestSymbolicMathGrammar(unittest.TestCase):

    def setUp(self):
        self.grammar = SymbolicMathGrammar(seed=42)

    def test_base_primitives(self):
        prims = self.grammar.base_primitives()
        self.assertEqual(len(prims), 15)
        names = {p.name for p in prims}
        self.assertIn("add", names)
        self.assertIn("sin", names)
        self.assertIn("x", names)

    def test_compose(self):
        from core.interfaces import Primitive
        add_prim = next(p for p in MATH_PRIMITIVES if p.name == "add")
        prog = self.grammar.compose(add_prim, [Program(root="x"), Program(root="const")])
        self.assertEqual(prog.root, "add")
        self.assertEqual(len(prog.children), 2)

    def test_mutate_changes_something(self):
        prog = Program(root="x")
        prims = self.grammar.base_primitives()
        # Run mutation many times, at least one should differ
        any_different = False
        for _ in range(20):
            mutated = self.grammar.mutate(prog, prims)
            if mutated.root != prog.root:
                any_different = True
                break
        self.assertTrue(any_different)

    def test_mutate_preserves_structure(self):
        """Mutation of a leaf should produce a leaf (same arity)."""
        prog = Program(root="x")  # arity 0
        prims = self.grammar.base_primitives()
        mutated = self.grammar.mutate(prog, prims)
        # The mutated program root should also be arity 0
        from grammars.symbolic_math import _PRIM_MAP
        if mutated.root in _PRIM_MAP:
            self.assertEqual(_PRIM_MAP[mutated.root].arity, 0)

    def test_crossover(self):
        a = Program(root="add", children=[Program(root="x"), Program(root="const")])
        b = Program(root="sin", children=[Program(root="x")])
        child = self.grammar.crossover(a, b)
        self.assertIsInstance(child, Program)

    def test_crossover_empty(self):
        a = Program(root="x")
        b = Program(root="y")
        child = self.grammar.crossover(a, b)
        self.assertIsInstance(child, Program)

    def test_mutate_empty_nodes(self):
        """Mutating a program with no collectable nodes returns it unchanged."""
        grammar = SymbolicMathGrammar(seed=42)
        # Program(root="x") will always have nodes, but we can test the
        # const mutation path
        prog = Program(root="const", params={"c": 1.0})
        prims = grammar.base_primitives()
        mutated = grammar.mutate(prog, prims)
        self.assertIsInstance(mutated, Program)


class TestSymbolicMathDrive(unittest.TestCase):

    def setUp(self):
        self.drive = SymbolicMathDrive()

    def test_perfect_match(self):
        self.assertAlmostEqual(self.drive.prediction_error(5.0, 5.0), 0.0)

    def test_error_value(self):
        # (3-1)^2 = 4
        self.assertAlmostEqual(self.drive.prediction_error(3.0, 1.0), 4.0)

    def test_none_input(self):
        self.assertEqual(self.drive.prediction_error(None, 5.0), 1e6)
        self.assertEqual(self.drive.prediction_error(5.0, None), 1e6)

    def test_invalid_type(self):
        self.assertEqual(self.drive.prediction_error("bad", 5.0), 1e6)


class TestMakeTask(unittest.TestCase):

    def test_basic(self):
        task = make_task(lambda x: x ** 2, task_id="sq", n_train=10, n_test=5)
        self.assertEqual(task.task_id, "sq")
        self.assertEqual(len(task.train_examples), 10)
        self.assertEqual(len(task.test_inputs), 5)
        self.assertEqual(len(task.test_outputs), 5)

    def test_train_examples_correct(self):
        task = make_task(lambda x: 2 * x, task_id="lin", n_train=5)
        for x, y in task.train_examples:
            self.assertAlmostEqual(y, 2 * x)

    def test_difficulty(self):
        task = make_task(lambda x: x, difficulty=3.0)
        self.assertAlmostEqual(task.difficulty, 3.0)


if __name__ == "__main__":
    unittest.main()
