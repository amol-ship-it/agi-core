"""
Tests for the List Operations domain.

Tests the primitives, environment, grammar, drive signal,
task generation, and integration with the core learner.
"""

import unittest

from core.types import Primitive, Program, Task, LibraryEntry
from core.config import SearchConfig
from core.learner import Learner
from core.memory import InMemoryStore

from domains.list_ops import (
    LIST_PRIMITIVES, _PRIM_MAP,
    ListEnv, ListGrammar, ListDrive,
    get_sample_tasks,
    _apply_op, _generate_examples,
)


# =============================================================================
# Primitive tests
# =============================================================================

class TestListPrimitives(unittest.TestCase):

    def test_primitives_count(self):
        self.assertEqual(len(LIST_PRIMITIVES), 22)

    def test_all_have_functions(self):
        for p in LIST_PRIMITIVES:
            self.assertIsNotNone(p.fn, f"{p.name} has no function")

    def test_reverse(self):
        self.assertEqual(_apply_op("reverse", [1, 2, 3]), [3, 2, 1])

    def test_sort_asc(self):
        self.assertEqual(_apply_op("sort_asc", [3, 1, 2]), [1, 2, 3])

    def test_sort_desc(self):
        self.assertEqual(_apply_op("sort_desc", [3, 1, 2]), [3, 2, 1])

    def test_double_all(self):
        self.assertEqual(_apply_op("double_all", [1, 2, 3]), [2, 4, 6])

    def test_increment_all(self):
        self.assertEqual(_apply_op("increment_all", [1, 2, 3]), [2, 3, 4])

    def test_decrement_all(self):
        self.assertEqual(_apply_op("decrement_all", [1, 2, 3]), [0, 1, 2])

    def test_negate_all(self):
        self.assertEqual(_apply_op("negate_all", [1, -2, 3]), [-1, 2, -3])

    def test_abs_all(self):
        self.assertEqual(_apply_op("abs_all", [-1, 2, -3]), [1, 2, 3])

    def test_square_all(self):
        self.assertEqual(_apply_op("square_all", [1, 2, 3]), [1, 4, 9])

    def test_head(self):
        self.assertEqual(_apply_op("head", [1, 2, 3, 4]), [1, 2])
        self.assertEqual(_apply_op("head", [1, 2, 3]), [1, 2])

    def test_tail(self):
        self.assertEqual(_apply_op("tail", [1, 2, 3, 4]), [3, 4])
        self.assertEqual(_apply_op("tail", [1, 2, 3]), [3])

    def test_filter_positive(self):
        self.assertEqual(_apply_op("filter_pos", [-1, 2, 0, 3, -4]), [2, 3])

    def test_filter_negative(self):
        self.assertEqual(_apply_op("filter_neg", [-1, 2, 0, 3, -4]), [-1, -4])

    def test_filter_even(self):
        self.assertEqual(_apply_op("filter_even", [1, 2, 3, 4, 5]), [2, 4])

    def test_filter_odd(self):
        self.assertEqual(_apply_op("filter_odd", [1, 2, 3, 4, 5]), [1, 3, 5])

    def test_unique(self):
        self.assertEqual(_apply_op("unique", [1, 2, 2, 3, 1]), [1, 2, 3])

    def test_cumsum(self):
        self.assertEqual(_apply_op("cumsum", [1, 2, 3, 4]), [1, 3, 6, 10])

    def test_diff(self):
        self.assertEqual(_apply_op("diff", [1, 3, 6, 10]), [2, 3, 4])

    def test_dedup_consecutive(self):
        self.assertEqual(_apply_op("dedup_consec", [1, 1, 2, 2, 1]), [1, 2, 1])

    def test_identity(self):
        self.assertEqual(_apply_op("identity", [1, 2, 3]), [1, 2, 3])

    def test_min_to_front(self):
        self.assertEqual(_apply_op("min_to_front", [3, 1, 2]), [1, 3, 2])

    def test_max_to_front(self):
        self.assertEqual(_apply_op("max_to_front", [3, 1, 2]), [3, 1, 2])

    def test_empty_list_safety(self):
        """All primitives should handle empty lists without crashing."""
        for p in LIST_PRIMITIVES:
            try:
                result = p.fn([])
                self.assertIsInstance(result, list)
            except Exception as e:
                self.fail(f"{p.name} crashed on empty list: {e}")


# =============================================================================
# Environment tests
# =============================================================================

class TestListEnv(unittest.TestCase):

    def test_execute_single(self):
        env = ListEnv()
        prog = Program(root="reverse")
        result = env.execute(prog, [1, 2, 3])
        self.assertEqual(result, [3, 2, 1])

    def test_execute_composition(self):
        """sort_asc(reverse(input)) should sort, not reverse-then-sort."""
        env = ListEnv()
        prog = Program(root="sort_asc", children=[Program(root="reverse")])
        result = env.execute(prog, [3, 1, 2])
        # reverse([3,1,2]) = [2,1,3], then sort_asc([2,1,3]) = [1,2,3]
        self.assertEqual(result, [1, 2, 3])

    def test_execute_three_step(self):
        """double_all(sort_asc(reverse(input)))"""
        env = ListEnv()
        prog = Program(root="double_all", children=[
            Program(root="sort_asc", children=[
                Program(root="reverse")])])
        result = env.execute(prog, [3, 1, 2])
        # reverse→[2,1,3], sort→[1,2,3], double→[2,4,6]
        self.assertEqual(result, [2, 4, 6])

    def test_non_list_input(self):
        env = ListEnv()
        prog = Program(root="reverse")
        result = env.execute(prog, "not_a_list")
        self.assertEqual(result, "not_a_list")

    def test_register_dynamic_primitive(self):
        env = ListEnv()
        custom = Primitive("triple_all", 1, lambda lst: [x * 3 for x in lst])
        env.register_primitive(custom)
        prog = Program(root="triple_all")
        result = env.execute(prog, [1, 2, 3])
        self.assertEqual(result, [3, 6, 9])

    def test_execute_library_entry(self):
        """Library entries (fn=Program) should be recursively executed."""
        env = ListEnv()
        # Simulate a library entry: sort_asc(reverse(x)) stored as a primitive
        stored_prog = Program(root="sort_asc", children=[Program(root="reverse")])
        lib_prim = Primitive("learned_0", 0, stored_prog, learned=True)
        env.register_primitive(lib_prim)

        # Execute a program that uses the library entry
        prog = Program(root="learned_0")
        result = env.execute(prog, [3, 1, 2])
        # reverse([3,1,2]) = [2,1,3], sort_asc([2,1,3]) = [1,2,3]
        self.assertEqual(result, [1, 2, 3])

    def test_execute_library_entry_in_composition(self):
        """Library entries should compose with other primitives."""
        env = ListEnv()
        # Library entry: sort_asc(x)
        stored_prog = Program(root="sort_asc")
        lib_prim = Primitive("learned_sort", 0, stored_prog, learned=True)
        env.register_primitive(lib_prim)

        # Compose: double_all(learned_sort(x))
        prog = Program(root="double_all", children=[Program(root="learned_sort")])
        result = env.execute(prog, [3, 1, 2])
        # sort([3,1,2]) = [1,2,3], double([1,2,3]) = [2,4,6]
        self.assertEqual(result, [2, 4, 6])


# =============================================================================
# Grammar tests
# =============================================================================

class TestListGrammar(unittest.TestCase):

    def test_base_primitives(self):
        grammar = ListGrammar()
        prims = grammar.base_primitives()
        self.assertEqual(len(prims), 22)

    def test_compose(self):
        grammar = ListGrammar()
        outer = _PRIM_MAP["sort_asc"]
        inner = [Program(root="reverse")]
        prog = grammar.compose(outer, inner)
        self.assertEqual(prog.root, "sort_asc")
        self.assertEqual(len(prog.children), 1)

    def test_mutate(self):
        grammar = ListGrammar(seed=42)
        prog = Program(root="reverse")
        prims = grammar.base_primitives()
        mutant = grammar.mutate(prog, prims)
        self.assertIsInstance(mutant, Program)

    def test_crossover(self):
        grammar = ListGrammar(seed=42)
        a = Program(root="reverse", children=[Program(root="sort_asc")])
        b = Program(root="double_all", children=[Program(root="negate_all")])
        child = grammar.crossover(a, b)
        self.assertIsInstance(child, Program)


# =============================================================================
# Drive signal tests
# =============================================================================

class TestListDrive(unittest.TestCase):

    def test_perfect_match(self):
        drive = ListDrive()
        error = drive.prediction_error([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(error, 0.0)

    def test_completely_wrong(self):
        drive = ListDrive()
        error = drive.prediction_error([4, 5, 6], [1, 2, 3])
        self.assertAlmostEqual(error, 1.0)

    def test_partial_match(self):
        drive = ListDrive()
        error = drive.prediction_error([1, 2, 99], [1, 2, 3])
        self.assertAlmostEqual(error, 1 / 3, places=5)

    def test_length_mismatch(self):
        drive = ListDrive()
        error = drive.prediction_error([1, 2], [1, 2, 3])
        self.assertGreater(error, 0.0)
        self.assertLessEqual(error, 1.0)

    def test_empty_lists(self):
        drive = ListDrive()
        error = drive.prediction_error([], [])
        self.assertAlmostEqual(error, 0.0)

    def test_non_list_types(self):
        drive = ListDrive()
        self.assertAlmostEqual(drive.prediction_error("a", "a"), 0.0)
        self.assertAlmostEqual(drive.prediction_error("a", "b"), 1.0)


# =============================================================================
# Task generation tests
# =============================================================================

class TestListTasks(unittest.TestCase):

    def test_sample_tasks_load(self):
        tasks = get_sample_tasks()
        self.assertGreater(len(tasks), 20)
        for task in tasks:
            self.assertIsInstance(task, Task)
            self.assertGreater(len(task.train_examples), 0)

    def test_difficulty_levels(self):
        tasks = get_sample_tasks()
        difficulties = set(t.difficulty for t in tasks)
        self.assertEqual(difficulties, {1.0, 2.0, 3.0})

    def test_level1_tasks_solvable(self):
        """Level 1 tasks should be solvable with a single primitive."""
        env = ListEnv()
        tasks = get_sample_tasks()
        level1 = [t for t in tasks if t.difficulty == 1.0]
        for task in level1:
            # Extract the expected op name from task_id
            op_name = task.task_id.replace("list_L1_", "")
            prog = Program(root=op_name)
            for inp, expected in task.train_examples:
                result = env.execute(prog, inp)
                self.assertEqual(result, expected,
                    f"Task {task.task_id}: {op_name}({inp}) = {result} != {expected}")

    def test_level2_tasks_solvable(self):
        """Level 2 tasks should be solvable with two-step programs."""
        env = ListEnv()
        tasks = get_sample_tasks()
        level2 = [t for t in tasks if t.difficulty == 2.0]
        for task in level2:
            # Parse ops from task_id: list_L2_op1_then_op2
            name = task.task_id.replace("list_L2_", "")
            ops = name.split("_then_")
            # ops[0] is inner (applied first), ops[1] is outer
            prog = Program(root=ops[1], children=[Program(root=ops[0])])
            for inp, expected in task.train_examples:
                result = env.execute(prog, inp)
                self.assertEqual(result, expected,
                    f"Task {task.task_id}: {ops[1]}({ops[0]}({inp})) = {result} != {expected}")

    def test_examples_are_non_trivial(self):
        """Inputs should differ from outputs (tasks aren't identity)."""
        tasks = get_sample_tasks()
        for task in tasks:
            any_different = any(inp != out for inp, out in task.train_examples)
            self.assertTrue(any_different,
                f"Task {task.task_id} has all identical input/output pairs")


# =============================================================================
# Learner integration tests
# =============================================================================

class TestListLearnerIntegration(unittest.TestCase):

    def _make_learner(self, **kwargs):
        defaults = dict(
            beam_width=10,
            max_generations=5,
            solve_threshold=0.001,
            seed=42,
            exhaustive_depth=2,
            exhaustive_pair_top_k=22,
        )
        defaults.update(kwargs)
        return Learner(
            environment=ListEnv(),
            grammar=ListGrammar(seed=42),
            drive=ListDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(**defaults),
        )

    def test_learner_solves_level1(self):
        """Core learner should solve level-1 (single-op) tasks."""
        learner = self._make_learner()
        tasks = get_sample_tasks()
        level1 = [t for t in tasks if t.difficulty == 1.0]
        solved = 0
        for task in level1:
            result = learner.wake_on_task(task)
            if result.train_solved:
                solved += 1
        self.assertGreater(solved, 0, "Should solve at least one level-1 task")

    def test_learner_solves_level2(self):
        """Core learner should solve level-2 (two-step) tasks with exhaustive_depth=2."""
        learner = self._make_learner(exhaustive_depth=2)
        tasks = get_sample_tasks()
        level2 = [t for t in tasks if t.difficulty == 2.0]
        solved = 0
        for task in level2[:3]:  # test a few for speed
            result = learner.wake_on_task(task)
            if result.train_solved:
                solved += 1
        self.assertGreater(solved, 0, "Should solve at least one level-2 task")

    def test_sleep_extracts_library(self):
        """After solving tasks, sleep should extract reusable sub-programs."""
        learner = self._make_learner(exhaustive_depth=2)
        tasks = get_sample_tasks()
        level1 = [t for t in tasks if t.difficulty == 1.0]
        for task in level1[:4]:
            learner.wake_on_task(task)
        result = learner.sleep()
        self.assertIsNotNone(result)

    def test_learner_runs_without_crash(self):
        """Basic smoke test: learner doesn't crash on list domain."""
        learner = self._make_learner()
        tasks = get_sample_tasks()
        result = learner.wake_on_task(tasks[0])
        self.assertIsNotNone(result)
        self.assertGreater(result.evaluations, 0)


if __name__ == "__main__":
    unittest.main()
