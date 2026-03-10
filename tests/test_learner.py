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


class TestWakeWorker(unittest.TestCase):
    """Test the module-level _wake_worker function used by multiprocessing."""

    def test_wake_worker_direct(self):
        from core.learner import _wake_worker
        task = _make_identity_task()
        env = StubEnv()
        grammar = StubGrammar(seed=42)
        drive = StubDrive()
        library = []
        search_cfg = SearchConfig(beam_width=10, max_generations=5, solve_threshold=0.01, seed=42)
        tm = TransitionMatrix()
        args = (task, env, grammar, drive, library, search_cfg, tm, 42)
        result = _wake_worker(args)
        self.assertIsInstance(result, WakeResult)
        self.assertEqual(result.task_id, "identity_task")

    def test_wake_worker_with_library(self):
        from core.learner import _wake_worker
        task = _make_identity_task()
        env = StubEnv()
        grammar = StubGrammar(seed=42)
        drive = StubDrive()
        library = [LibraryEntry(name="lib_0", program=Program(root="identity"), usefulness=1.0)]
        search_cfg = SearchConfig(beam_width=10, max_generations=5, solve_threshold=0.01, seed=42)
        tm = TransitionMatrix()
        args = (task, env, grammar, drive, library, search_cfg, tm, 42)
        result = _wake_worker(args)
        self.assertIsInstance(result, WakeResult)


class TestLearnerEarlyStop(unittest.TestCase):
    """Test early stopping behavior in wake phase."""

    def test_early_stop_on_perfect_solve(self):
        """With generous early_stop_energy and an identity task, should stop early."""
        # energy = alpha * pred_error + beta * complexity_cost
        # For identity: pred_error=0.0, complexity=1 node, so energy = 0.001
        # Set threshold above that to trigger early stop
        learner = _make_learner(
            beam_width=20, max_generations=100,
            early_stop_energy=0.01, energy_beta=0.001,
        )
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertTrue(result.solved)
        self.assertLess(result.generations_used, 100)

    def test_no_record_early_stop(self):
        learner = _make_learner(
            beam_width=20, max_generations=100,
            early_stop_energy=0.01, energy_beta=0.001,
        )
        task = _make_identity_task()
        result = learner._wake_on_task_no_record(task)
        self.assertTrue(result.solved)
        self.assertLess(result.generations_used, 100)


class TestLearnerSleepEdgeCases(unittest.TestCase):

    def test_sleep_deduplicates_library_entries(self):
        """Sleep should not add entries that are already in the library."""
        learner = _make_learner()
        shared = Program(root="identity", children=[Program(root="double")])

        # Pre-add an entry to the library with the same program repr
        existing = LibraryEntry(name="existing", program=shared, usefulness=1.0)
        learner.memory.add_to_library(existing)

        # Store solutions with the same subtree
        sol1 = ScoredProgram(
            program=Program(root="identity", children=[
                Program(root="identity", children=[Program(root="double")])
            ]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t1",
        )
        sol2 = ScoredProgram(
            program=Program(root="double", children=[
                Program(root="identity", children=[Program(root="double")])
            ]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t2",
        )
        learner.memory.store_solution("t1", sol1)
        learner.memory.store_solution("t2", sol2)

        result = learner.sleep()
        # The shared subtree should be deduplicated against the existing entry
        lib_names = [e.name for e in learner.memory.get_library()]
        self.assertIn("existing", lib_names)

    def test_sleep_respects_max_library_size(self):
        learner = _make_learner()
        learner.sleep_cfg.max_library_size = 1

        # Pre-fill library to max
        entry = LibraryEntry(name="full", program=Program(root="x"), usefulness=1.0)
        learner.memory.add_to_library(entry)

        # Store solutions that would generate new entries
        for i in range(3):
            sol = ScoredProgram(
                program=Program(root="identity", children=[Program(root="double")]),
                energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id=f"t{i}",
            )
            learner.memory.store_solution(f"t{i}", sol)

        result = learner.sleep()
        self.assertEqual(len(result.new_entries), 0)

    def test_sleep_decays_old_entries(self):
        learner = _make_learner()
        learner.sleep_cfg.usefulness_decay = 0.5

        entry = LibraryEntry(name="old_entry", program=Program(root="x"), usefulness=10.0)
        learner.memory.add_to_library(entry)

        # Sleep with no solutions — should still decay existing entries
        learner.sleep()
        lib = learner.memory.get_library()
        # Usefulness should have decreased
        self.assertLess(lib[0].usefulness, 10.0)


class TestLearnerEvaluateException(unittest.TestCase):

    def test_crashing_program_gets_penalty(self):
        """Programs that throw exceptions should get a high error penalty."""

        class CrashingEnv(StubEnv):
            def execute(self, program, input_data):
                if program.root == "crash":
                    raise RuntimeError("boom")
                return super().execute(program, input_data)

        learner = Learner(
            environment=CrashingEnv(),
            grammar=StubGrammar(seed=42),
            drive=StubDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(beam_width=5, max_generations=3, seed=42),
        )
        task = _make_identity_task()
        prog = Program(root="crash")
        sp = learner._evaluate_program(prog, task)
        self.assertGreater(sp.prediction_error, 1e5)


class TestRandomProgramEdgeCases(unittest.TestCase):

    def test_arity_zero_at_depth_greater_than_1(self):
        """When a random pick selects arity-0 at depth > 1, should return leaf."""
        learner = _make_learner()
        # Only arity-0 primitives — forces the arity==0 early return at depth > 1
        zero_prims = [Primitive("a", 0, None), Primitive("b", 0, None)]
        prog = learner._random_program(zero_prims, max_depth=3, use_prior=False)
        self.assertEqual(prog.depth, 1)

    def test_no_low_arity_prims_at_leaf(self):
        """When all primitives are arity > 1, should still pick one as leaf."""
        learner = _make_learner()
        high_arity = [Primitive("f", 2, None), Primitive("g", 3, None)]
        prog = learner._random_program(high_arity, max_depth=1, use_prior=False)
        # Falls back to using all primitives when no arity<=1 found
        self.assertIn(prog.root, ["f", "g"])


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


# =============================================================================
# Semantic deduplication tests
# =============================================================================

class TestSemanticDedup(unittest.TestCase):

    def test_dedup_removes_identical_outputs(self):
        """Two programs producing the same outputs should be deduplicated."""
        learner = _make_learner(semantic_dedup=True, dedup_precision=6)
        task = _make_identity_task()
        # Two identity programs = same outputs
        sp1 = ScoredProgram(
            program=Program(root="identity"), energy=0.5,
            prediction_error=0.0, complexity_cost=1.0,
        )
        sp2 = ScoredProgram(
            program=Program(root="identity"), energy=0.8,
            prediction_error=0.0, complexity_cost=2.0,
        )
        deduped, n_removed = learner._semantic_dedup([sp1, sp2], task)
        self.assertEqual(n_removed, 1)
        self.assertEqual(len(deduped), 1)
        # Should keep the lower-energy one
        self.assertAlmostEqual(deduped[0].energy, 0.5)

    def test_dedup_keeps_different_outputs(self):
        """Programs with different outputs should both be kept."""
        learner = _make_learner(semantic_dedup=True)
        task = _make_identity_task()
        sp1 = ScoredProgram(
            program=Program(root="identity"), energy=0.5,
            prediction_error=0.0, complexity_cost=1.0,
        )
        sp2 = ScoredProgram(
            program=Program(root="double"), energy=0.8,
            prediction_error=1.0, complexity_cost=1.0,
        )
        deduped, n_removed = learner._semantic_dedup([sp1, sp2], task)
        self.assertEqual(n_removed, 0)
        self.assertEqual(len(deduped), 2)

    def test_semantic_hash_deterministic(self):
        """Same program + same task should always produce the same hash."""
        learner = _make_learner()
        task = _make_identity_task()
        prog = Program(root="identity")
        h1 = learner._semantic_hash(prog, task)
        h2 = learner._semantic_hash(prog, task)
        self.assertEqual(h1, h2)

    def test_dedup_disabled(self):
        """When semantic_dedup=False, no dedup should happen."""
        learner = _make_learner(semantic_dedup=False)
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertEqual(result.dedup_count, 0)

    def test_wake_reports_dedup_count(self):
        """Wake result should report how many duplicates were removed."""
        learner = _make_learner(semantic_dedup=True)
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertIsInstance(result.dedup_count, int)
        self.assertGreaterEqual(result.dedup_count, 0)


# =============================================================================
# Pareto front tests
# =============================================================================

class TestParetoFront(unittest.TestCase):

    def test_pareto_front_returned(self):
        """Wake result should contain a non-empty Pareto front."""
        from core.learner import ParetoEntry
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        self.assertIsInstance(result.pareto_front, list)
        self.assertGreater(len(result.pareto_front), 0)
        for entry in result.pareto_front:
            self.assertIsInstance(entry, ParetoEntry)

    def test_pareto_front_sorted_by_complexity(self):
        """Pareto front entries should be in increasing complexity order."""
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        complexities = [e.complexity for e in result.pareto_front]
        self.assertEqual(complexities, sorted(complexities))

    def test_pareto_front_error_decreasing(self):
        """On the true Pareto front, error should decrease as complexity increases."""
        learner = _make_learner()
        task = _make_identity_task()
        result = learner.wake_on_task(task)
        if len(result.pareto_front) >= 2:
            errors = [e.prediction_error for e in result.pareto_front]
            for i in range(len(errors) - 1):
                self.assertGreaterEqual(errors[i], errors[i + 1])

    def test_update_pareto_front(self):
        """_update_pareto_front should keep the best error per complexity."""
        from core.learner import ParetoEntry
        learner = _make_learner()
        pareto: dict[int, ParetoEntry] = {}

        sp1 = ScoredProgram(
            program=Program(root="identity"), energy=1.0,
            prediction_error=0.5, complexity_cost=1.0,
        )
        learner._update_pareto_front(pareto, sp1)
        self.assertEqual(pareto[1].prediction_error, 0.5)

        # Better error at same complexity should replace
        sp2 = ScoredProgram(
            program=Program(root="double"), energy=0.5,
            prediction_error=0.1, complexity_cost=1.0,
        )
        learner._update_pareto_front(pareto, sp2)
        self.assertEqual(pareto[1].prediction_error, 0.1)

        # Worse error should not replace
        sp3 = ScoredProgram(
            program=Program(root="identity"), energy=2.0,
            prediction_error=0.9, complexity_cost=1.0,
        )
        learner._update_pareto_front(pareto, sp3)
        self.assertEqual(pareto[1].prediction_error, 0.1)

    def test_extract_pareto_front_filters_dominated(self):
        """_extract_pareto_front should remove dominated entries."""
        from core.learner import ParetoEntry
        learner = _make_learner()
        pareto = {
            1: ParetoEntry(1, 0.5, 1.0, Program(root="a")),
            2: ParetoEntry(2, 0.8, 1.0, Program(root="b")),  # dominated by complexity=1
            3: ParetoEntry(3, 0.1, 1.0, Program(root="c")),  # not dominated
        }
        front = learner._extract_pareto_front(pareto)
        # Only complexity=1 (error=0.5) and complexity=3 (error=0.1) are non-dominated
        self.assertEqual(len(front), 2)
        self.assertEqual(front[0].complexity, 1)
        self.assertEqual(front[1].complexity, 3)

    def test_pareto_entry_repr(self):
        from core.learner import ParetoEntry
        entry = ParetoEntry(3, 0.001, 0.5, Program(root="x"))
        r = repr(entry)
        self.assertIn("complexity=3", r)
        self.assertIn("x", r)

    def test_no_record_also_has_pareto(self):
        """_wake_on_task_no_record should also return a Pareto front."""
        learner = _make_learner()
        task = _make_identity_task()
        result = learner._wake_on_task_no_record(task)
        self.assertIsInstance(result.pareto_front, list)
        self.assertGreater(len(result.pareto_front), 0)


if __name__ == "__main__":
    unittest.main()
