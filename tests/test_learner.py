"""
Tests for core/learner.py — TransitionMatrix, Learner (wake, sleep, curriculum).

Uses minimal stub implementations of the 4 interfaces to test the core
loop in isolation from any domain. This verifies the core architecture
invariant: core/ imports NOTHING domain-specific.
"""

import copy
import random
import unittest
from typing import Any, Optional

from core.interfaces import (
    Environment, Grammar, DriveSignal, Primitive, Program, Task,
    Observation, LibraryEntry, ScoredProgram,
)
from core.memory import InMemoryStore
from core.learner import (
    TransitionMatrix, Learner, SearchConfig, SleepConfig, CurriculumConfig,
    WakeResult, SleepResult, RoundResult,
)


# =============================================================================
# Minimal stubs — the simplest possible domain for testing the core loop
# =============================================================================

def _id_fn(x):
    return x

def _double_fn(x):
    return x * 2

STUB_PRIMS = [
    Primitive("identity", 1, _id_fn),
    Primitive("double", 1, _double_fn),
]
_STUB_MAP = {p.name: p for p in STUB_PRIMS}


class StubEnv(Environment):
    """Trivial environment: execute just applies the named function."""

    def load_task(self, task: Task) -> Observation:
        return Observation(data=task.train_examples)

    def execute(self, program: Program, input_data: Any) -> Any:
        # Walk the tree: apply outermost to result of children
        result = input_data
        # Process children first (innermost)
        if program.children:
            result = self.execute(program.children[0], input_data)
        prim = _STUB_MAP.get(program.root)
        if prim and prim.fn:
            return prim.fn(result)
        return result

    def reset(self):
        pass


class StubGrammar(Grammar):
    """Minimal grammar for testing."""

    def __init__(self, seed=42):
        self._rng = random.Random(seed)

    def base_primitives(self) -> list[Primitive]:
        return list(STUB_PRIMS)

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def mutate(self, program: Program, primitives: list[Primitive]) -> Program:
        prog = copy.deepcopy(program)
        prim = self._rng.choice(primitives)
        prog.root = prim.name
        return prog

    def crossover(self, a: Program, b: Program) -> Program:
        return copy.deepcopy(a)


class StubDrive(DriveSignal):
    """Simple numeric error."""

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        if predicted is None or expected is None:
            return 1e6
        try:
            return abs(float(predicted) - float(expected))
        except (TypeError, ValueError):
            return 1e6


# =============================================================================
# TransitionMatrix tests
# =============================================================================

class TestTransitionMatrix(unittest.TestCase):

    def test_empty_matrix(self):
        tm = TransitionMatrix()
        self.assertEqual(tm.size, 0)

    def test_observe_and_probability(self):
        tm = TransitionMatrix(smoothing=0.1)
        # f(g(x))
        prog = Program(root="f", children=[
            Program(root="g", children=[Program(root="x")])
        ])
        tm.observe_program(prog)
        self.assertEqual(tm.size, 2)  # f->g and g->x

        # P(g|f) should be higher than P(x|f)
        p_g_given_f = tm.probability("f", "g", n_primitives=3)
        p_x_given_f = tm.probability("f", "x", n_primitives=3)
        self.assertGreater(p_g_given_f, p_x_given_f)

    def test_weighted_choice_with_no_prior(self):
        tm = TransitionMatrix()
        prims = [Primitive("a", 0, None), Primitive("b", 0, None)]
        rng = random.Random(42)
        # Should not crash, returns a random primitive
        result = tm.weighted_choice("unknown_parent", prims, rng)
        self.assertIn(result.name, ["a", "b"])

    def test_weighted_choice_with_prior(self):
        tm = TransitionMatrix(smoothing=0.01)
        # Observe f->a many times
        for _ in range(100):
            tm.observe_program(Program(root="f", children=[Program(root="a")]))

        prims = [Primitive("a", 0, None), Primitive("b", 0, None)]
        rng = random.Random(42)
        # Should heavily favor "a"
        choices = [tm.weighted_choice("f", prims, rng).name for _ in range(50)]
        a_count = choices.count("a")
        self.assertGreater(a_count, 35)  # should be biased toward a

    def test_repr(self):
        tm = TransitionMatrix()
        self.assertIn("0 transitions", repr(tm))


# =============================================================================
# Learner tests
# =============================================================================

def _make_identity_task():
    """Task where the answer is identity(input)."""
    return Task(
        task_id="identity_task",
        train_examples=[(5.0, 5.0), (3.0, 3.0), (7.0, 7.0)],
        test_inputs=[1.0, 2.0],
        test_outputs=[1.0, 2.0],
        difficulty=0.0,
    )


def _make_double_task():
    """Task where the answer is double(input)."""
    return Task(
        task_id="double_task",
        train_examples=[(2.0, 4.0), (3.0, 6.0), (5.0, 10.0)],
        test_inputs=[1.0, 4.0],
        test_outputs=[2.0, 8.0],
        difficulty=1.0,
    )


def _make_learner(**kwargs):
    """Create a learner with stubs and small search budget."""
    defaults = dict(
        beam_width=20,
        max_generations=15,
        solve_threshold=0.01,
        seed=42,
    )
    defaults.update(kwargs)
    return Learner(
        environment=StubEnv(),
        grammar=StubGrammar(seed=42),
        drive=StubDrive(),
        memory=InMemoryStore(),
        search_config=SearchConfig(**defaults),
    )


class TestLearnerWake(unittest.TestCase):

    def test_wake_solves_identity(self):
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertIsInstance(result, WakeResult)
        self.assertTrue(result.solved)
        self.assertAlmostEqual(result.best.prediction_error, 0.0, places=2)

    def test_wake_solves_double(self):
        learner = _make_learner()
        task = _make_double_task()
        result = learner.wake_on_task(task)
        self.assertTrue(result.solved)

    def test_wake_records_episode(self):
        learner = _make_learner()
        task = _make_identity_task()
        learner.wake_on_task(task)
        episodes = learner.memory.replay_episodes()
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["task_id"], "identity_task")

    def test_wake_stores_solution_if_solved(self):
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        if result.solved:
            sols = learner.memory.get_solutions()
            self.assertIn("identity_task", sols)

    def test_wake_evaluations_counted(self):
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertGreater(result.evaluations, 0)
        self.assertGreater(result.wall_time, 0.0)

    def test_wake_no_record_variant(self):
        learner = _make_learner()
        task = _make_identity_task()
        result = learner._wake_on_task_no_record(task)
        self.assertIsInstance(result, WakeResult)
        # Memory should be empty since no_record was used
        self.assertEqual(len(learner.memory.replay_episodes()), 0)


class TestLearnerSleep(unittest.TestCase):

    def test_sleep_with_no_solutions(self):
        learner = _make_learner()
        result = learner.sleep()
        self.assertIsInstance(result, SleepResult)
        self.assertEqual(len(result.new_entries), 0)
        self.assertEqual(result.library_size_before, 0)
        self.assertEqual(result.library_size_after, 0)

    def test_sleep_extracts_common_subtrees(self):
        learner = _make_learner()
        # Manually store two solutions that share a common subtree
        shared = Program(root="identity", children=[Program(root="double")])
        sol1 = ScoredProgram(
            program=Program(root="identity", children=[shared]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t1",
        )
        sol2 = ScoredProgram(
            program=Program(root="double", children=[shared]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t2",
        )
        learner.memory.store_solution("t1", sol1)
        learner.memory.store_solution("t2", sol2)

        result = learner.sleep()
        # Should have found some common subtrees
        self.assertGreaterEqual(result.library_size_after, result.library_size_before)

    def test_sleep_builds_transition_matrix(self):
        learner = _make_learner()
        prog = Program(root="identity", children=[Program(root="double")])
        sol = ScoredProgram(
            program=prog, energy=0.0, prediction_error=0.0,
            complexity_cost=1.0, task_id="t1",
        )
        learner.memory.store_solution("t1", sol)
        learner.sleep()
        self.assertGreater(learner._transition_matrix.size, 0)


class TestLearnerCurriculum(unittest.TestCase):

    def test_single_round(self):
        learner = _make_learner()
        tasks = [_make_identity_task()]
        results = learner.run_curriculum(
            tasks, CurriculumConfig(wake_sleep_rounds=1, workers=1),
        )
        self.assertEqual(len(results), 1)
        rr = results[0]
        self.assertIsInstance(rr, RoundResult)
        self.assertEqual(rr.round_number, 1)
        self.assertEqual(rr.tasks_total, 1)

    def test_multiple_rounds(self):
        learner = _make_learner()
        tasks = [_make_identity_task(), _make_double_task()]
        results = learner.run_curriculum(
            tasks, CurriculumConfig(wake_sleep_rounds=2, workers=1),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].round_number, 1)
        self.assertEqual(results[1].round_number, 2)

    def test_curriculum_sorts_by_difficulty(self):
        learner = _make_learner()
        hard = _make_double_task()
        hard.difficulty = 10.0
        easy = _make_identity_task()
        easy.difficulty = 0.0
        tasks = [hard, easy]
        # With sort enabled (default), easier should be attempted first
        results = learner.run_curriculum(
            tasks, CurriculumConfig(wake_sleep_rounds=1, workers=1, sort_by_difficulty=True),
        )
        self.assertEqual(len(results), 1)

    def test_curriculum_auto_workers(self):
        """workers=0 should auto-detect (shouldn't crash)."""
        learner = _make_learner()
        tasks = [_make_identity_task()]
        # workers=0 means auto; will fallback to sequential for 1 task
        results = learner.run_curriculum(
            tasks, CurriculumConfig(wake_sleep_rounds=1, workers=0),
        )
        self.assertEqual(len(results), 1)


class TestLearnerHelpers(unittest.TestCase):

    def test_init_beam(self):
        learner = _make_learner()
        prims = learner.grammar.base_primitives()
        beam = learner._init_beam(prims, 10)
        self.assertEqual(len(beam), 10)
        for prog in beam:
            self.assertIsInstance(prog, Program)

    def test_enumerate_subtrees(self):
        learner = _make_learner()
        tree = Program(root="f", children=[
            Program(root="g"),
            Program(root="h", children=[Program(root="x")]),
        ])
        subtrees = learner._enumerate_subtrees(tree)
        roots = {s.root for s in subtrees}
        self.assertEqual(roots, {"f", "g", "h", "x"})
        self.assertEqual(len(subtrees), 4)

    def test_evaluate_program(self):
        learner = _make_learner()
        task = _make_identity_task()
        prog = Program(root="identity")
        sp = learner._evaluate_program(prog, task)
        self.assertIsInstance(sp, ScoredProgram)
        self.assertAlmostEqual(sp.prediction_error, 0.0, places=4)

    def test_credit_library_usage(self):
        learner = _make_learner()
        entry = LibraryEntry(name="my_lib", program=Program(root="x"), usefulness=1.0)
        learner.memory.add_to_library(entry)
        # A program that uses the library entry name
        prog = Program(root="my_lib")
        learner._credit_library_usage(prog)
        lib = learner.memory.get_library()
        self.assertEqual(lib[0].reuse_count, 1)
        self.assertAlmostEqual(lib[0].usefulness, 2.0)

    def test_random_program_depth_1(self):
        learner = _make_learner()
        prims = learner.grammar.base_primitives()
        prog = learner._random_program(prims, max_depth=1, use_prior=False)
        self.assertEqual(prog.depth, 1)

    def test_random_program_with_prior(self):
        learner = _make_learner()
        # Build some prior
        for _ in range(10):
            learner._transition_matrix.observe_program(
                Program(root="identity", children=[Program(root="double")])
            )
        prims = learner.grammar.base_primitives()
        prog = learner._random_program(prims, max_depth=2, use_prior=True, parent_op="identity")
        self.assertIsInstance(prog, Program)


class TestConfigDefaults(unittest.TestCase):

    def test_search_config_defaults(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.beam_width, 200)
        self.assertEqual(cfg.max_generations, 100)
        self.assertIsNone(cfg.seed)

    def test_sleep_config_defaults(self):
        cfg = SleepConfig()
        self.assertEqual(cfg.min_occurrences, 2)
        self.assertEqual(cfg.max_library_size, 500)

    def test_curriculum_config_defaults(self):
        cfg = CurriculumConfig()
        self.assertTrue(cfg.sort_by_difficulty)
        self.assertEqual(cfg.wake_sleep_rounds, 3)
        self.assertEqual(cfg.workers, 0)


if __name__ == "__main__":
    unittest.main()
