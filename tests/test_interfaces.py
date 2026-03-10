"""
Tests for core/interfaces.py — data types and abstract interface contracts.

Verifies:
1. Dataclass construction and properties (Primitive, Program, Task, etc.)
2. Program tree depth/size computation
3. Default implementations on abstract base classes (Grammar.inject_library, DriveSignal.energy)
"""

import unittest
from core.interfaces import (
    Primitive, Program, Observation, Task, ScoredProgram, LibraryEntry,
    Environment, Grammar, DriveSignal, Memory,
)


class TestPrimitive(unittest.TestCase):

    def test_construction(self):
        p = Primitive(name="rot90", arity=1, fn=lambda x: x, domain="arc")
        self.assertEqual(p.name, "rot90")
        self.assertEqual(p.arity, 1)
        self.assertEqual(p.domain, "arc")
        self.assertFalse(p.learned)

    def test_learned_flag(self):
        p = Primitive(name="lib_0", arity=0, fn=None, learned=True)
        self.assertTrue(p.learned)

    def test_repr(self):
        p = Primitive(name="add", arity=2, fn=lambda a, b: a + b)
        self.assertIn("add/2", repr(p))
        p2 = Primitive(name="lib_0", arity=0, fn=None, learned=True)
        self.assertIn("[learned]", repr(p2))

    def test_frozen(self):
        p = Primitive(name="x", arity=0, fn=None)
        with self.assertRaises(AttributeError):
            p.name = "y"


class TestProgram(unittest.TestCase):

    def test_leaf(self):
        p = Program(root="x")
        self.assertEqual(p.depth, 1)
        self.assertEqual(p.size, 1)
        self.assertEqual(repr(p), "x")

    def test_tree_depth_and_size(self):
        # add(x, mul(x, x)) -> depth=3, size=5
        tree = Program(root="add", children=[
            Program(root="x"),
            Program(root="mul", children=[
                Program(root="x"),
                Program(root="x"),
            ]),
        ])
        self.assertEqual(tree.depth, 3)
        self.assertEqual(tree.size, 5)

    def test_repr_composition(self):
        tree = Program(root="f", children=[Program(root="a"), Program(root="b")])
        self.assertEqual(repr(tree), "f(a, b)")

    def test_params(self):
        p = Program(root="const", params={"c": 3.14})
        self.assertAlmostEqual(p.params["c"], 3.14)

    def test_default_children_and_params(self):
        p = Program(root="x")
        self.assertEqual(p.children, [])
        self.assertEqual(p.params, {})


class TestObservation(unittest.TestCase):

    def test_construction(self):
        obs = Observation(data=[1, 2, 3])
        self.assertEqual(obs.data, [1, 2, 3])
        self.assertEqual(obs.metadata, {})


class TestTask(unittest.TestCase):

    def test_construction(self):
        t = Task(
            task_id="t1",
            train_examples=[(1, 2), (3, 4)],
            test_inputs=[5],
            difficulty=1.5,
        )
        self.assertEqual(t.task_id, "t1")
        self.assertEqual(len(t.train_examples), 2)
        self.assertAlmostEqual(t.difficulty, 1.5)
        self.assertEqual(t.test_outputs, [])  # default


class TestScoredProgram(unittest.TestCase):

    def test_construction(self):
        sp = ScoredProgram(
            program=Program(root="x"),
            energy=0.5,
            prediction_error=0.4,
            complexity_cost=0.1,
            task_id="t1",
        )
        self.assertAlmostEqual(sp.energy, 0.5)


class TestLibraryEntry(unittest.TestCase):

    def test_construction(self):
        entry = LibraryEntry(
            name="lib_0",
            program=Program(root="f", children=[Program(root="x")]),
            usefulness=3.5,
            source_tasks=["t1", "t2"],
        )
        self.assertEqual(entry.name, "lib_0")
        self.assertEqual(entry.reuse_count, 0)


class TestGrammarInjectLibrary(unittest.TestCase):
    """Test the default inject_library on Grammar base class."""

    def test_inject_library_default(self):
        # Create a minimal concrete Grammar to test the default method
        class StubGrammar(Grammar):
            def base_primitives(self): return []
            def compose(self, outer, inner_programs): return Program(root=outer.name)
            def mutate(self, program, primitives): return program
            def crossover(self, a, b): return a

        g = StubGrammar()
        entries = [
            LibraryEntry(name="lib_0", program=Program(root="f"), usefulness=1.0, domain="test"),
            LibraryEntry(name="lib_1", program=Program(root="g"), usefulness=2.0),
        ]
        prims = g.inject_library(entries)
        self.assertEqual(len(prims), 2)
        self.assertEqual(prims[0].name, "lib_0")
        self.assertTrue(prims[0].learned)
        self.assertEqual(prims[0].arity, 0)
        self.assertEqual(prims[0].domain, "test")
        self.assertEqual(prims[1].domain, "")


class TestDriveSignalDefaults(unittest.TestCase):
    """Test the default implementations on DriveSignal."""

    def test_complexity_cost(self):
        class StubDrive(DriveSignal):
            def prediction_error(self, predicted, expected):
                return abs(predicted - expected)

        d = StubDrive()
        tree = Program(root="f", children=[Program(root="x"), Program(root="y")])
        self.assertAlmostEqual(d.complexity_cost(tree), 3.0)

    def test_energy(self):
        class StubDrive(DriveSignal):
            def prediction_error(self, predicted, expected):
                return abs(predicted - expected)

        d = StubDrive()
        prog = Program(root="x")  # size=1
        total, pred_err, comp = d.energy(prog, 3.0, 1.0, alpha=1.0, beta=0.01)
        self.assertAlmostEqual(pred_err, 2.0)
        self.assertAlmostEqual(comp, 1.0)
        self.assertAlmostEqual(total, 1.0 * 2.0 + 0.01 * 1.0)


if __name__ == "__main__":
    unittest.main()
