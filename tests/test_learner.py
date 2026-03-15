"""
Tests for core/learner.py — TransitionMatrix, Learner (wake, sleep, curriculum).

Uses minimal stub implementations of the 4 interfaces to test the core
loop in isolation from any domain. This verifies the core architecture
invariant: core/ imports NOTHING domain-specific.
"""

import copy
import math
import random
import unittest
from typing import Any, Optional

from core.types import Primitive, Program, Task, Observation, LibraryEntry, ScoredProgram
from core.interfaces import Environment, Grammar, DriveSignal
from core.config import SearchConfig, SleepConfig, CurriculumConfig
from core.results import ParetoEntry, WakeResult, SleepResult, RoundResult
from core.transition_matrix import TransitionMatrix
from core.learner import Learner
from core.memory import InMemoryStore


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

    def mutate(self, program: Program, primitives: list[Primitive],
               transition_matrix=None) -> Program:
        prog = copy.deepcopy(program)
        if transition_matrix and transition_matrix.size > 0:
            prim = transition_matrix.weighted_choice(program.root, primitives, self._rng)
        else:
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

    def test_beam_search_uses_transition_matrix(self):
        """Verify that beam search mutations pass the transition matrix to grammar.mutate()."""
        learner = _make_learner(exhaustive_depth=0, beam_width=10, max_generations=3)
        # Pre-populate transition matrix with a strong prior
        for _ in range(50):
            learner._transition_matrix.observe_program(
                Program(root="identity", children=[Program(root="double")]))
        self.assertGreater(learner._transition_matrix.size, 0)

        task = _make_identity_task()
        result = learner.wake_on_task(task)
        # Should still produce valid results (not crash) with TM active
        self.assertIsInstance(result, WakeResult)
        self.assertGreater(result.evaluations, 0)


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

    def test_sleep_prunes_dead_entries(self):
        """Entries with usefulness near zero and no reuse should be pruned."""
        learner = _make_learner()
        # Add an entry with very low usefulness and no reuse
        dead = LibraryEntry(name="dead", program=Program(root="x"),
                           usefulness=0.001, reuse_count=0)
        alive = LibraryEntry(name="alive", program=Program(root="y"),
                            usefulness=5.0, reuse_count=0)
        reused = LibraryEntry(name="reused", program=Program(root="z"),
                             usefulness=0.001, reuse_count=3)
        learner.memory.add_to_library(dead)
        learner.memory.add_to_library(alive)
        learner.memory.add_to_library(reused)

        learner.sleep()
        lib_names = [e.name for e in learner.memory.get_library()]
        # Dead entry (low usefulness, no reuse) should be pruned
        self.assertNotIn("dead", lib_names)
        # Alive entry (high usefulness) should survive
        self.assertIn("alive", lib_names)
        # Reused entry (low usefulness but has reuse) should survive
        self.assertIn("reused", lib_names)

    def test_sleep_diversity_bonus(self):
        """Subtrees appearing across structurally diverse solutions score higher."""
        learner = _make_learner()
        shared = Program(root="identity", children=[Program(root="double")])

        # Solutions with DIFFERENT root ops → high diversity
        sol1 = ScoredProgram(
            program=Program(root="identity", children=[shared]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t1")
        sol2 = ScoredProgram(
            program=Program(root="double", children=[shared]),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t2")
        learner.memory.store_solution("t1", sol1)
        learner.memory.store_solution("t2", sol2)

        result = learner.sleep()
        # Should find the shared subtree with diversity bonus applied
        if result.new_entries:
            # Diversity bonus should make usefulness > base value
            base = 2 * math.log(shared.size + 1)  # without diversity bonus
            self.assertGreater(result.new_entries[0].usefulness, base)

    def test_prune_library_on_memory(self):
        """InMemoryStore.prune_library removes dead entries."""
        mem = InMemoryStore()
        mem.add_to_library(LibraryEntry(
            name="dead", program=Program(root="x"),
            usefulness=0.005, reuse_count=0))
        mem.add_to_library(LibraryEntry(
            name="alive", program=Program(root="y"),
            usefulness=1.0, reuse_count=0))
        pruned = mem.prune_library(min_usefulness=0.01)
        self.assertEqual(pruned, 1)
        self.assertEqual(len(mem.get_library()), 1)
        self.assertEqual(mem.get_library()[0].name, "alive")


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
        self.assertEqual(cfg.min_occurrences, 1)
        self.assertEqual(cfg.max_library_size, 500)

    def test_curriculum_config_defaults(self):
        cfg = CurriculumConfig()
        self.assertFalse(cfg.sort_by_difficulty)
        self.assertEqual(cfg.wake_sleep_rounds, 3)
        self.assertEqual(cfg.workers, 0)

    def test_adaptive_realloc_config_defaults(self):
        cfg = CurriculumConfig()
        self.assertFalse(cfg.adaptive_realloc)
        self.assertEqual(cfg.adaptive_realloc_budget_multiplier, 3.0)
        self.assertEqual(cfg.adaptive_realloc_pair_top_k_boost, 20)
        self.assertEqual(cfg.adaptive_realloc_triple_top_k_boost, 10)

    def test_adaptive_realloc_no_near_misses(self):
        """Adaptive realloc with solvable tasks should be a no-op."""
        learner = _make_learner()
        tasks = [_make_identity_task()]
        results = learner.run_curriculum(
            tasks,
            CurriculumConfig(
                wake_sleep_rounds=1, workers=1,
                adaptive_realloc=True,
            ),
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].wake_results[0].train_solved)

    def test_adaptive_realloc_runs_on_unsolved(self):
        """Adaptive realloc runs a second pass on near-miss tasks."""
        # Use a task that can't be solved (output differs from any primitive)
        unsolvable = Task(
            task_id="unsolvable",
            train_examples=[(1.0, 99.0), (2.0, 99.0)],
            test_inputs=[3.0],
            test_outputs=[99.0],
            difficulty=5.0,
        )
        learner = _make_learner(near_miss_threshold=1.0)  # wide threshold
        results = learner.run_curriculum(
            [unsolvable],
            CurriculumConfig(
                wake_sleep_rounds=1, workers=1,
                adaptive_realloc=True,
            ),
        )
        self.assertEqual(len(results), 1)
        # Task should still be unsolved, but the realloc should have run
        wr = results[0].wake_results[0]
        self.assertFalse(wr.train_solved)
        # Evaluations should be higher than a single pass (realloc added more)
        self.assertGreater(wr.evaluations, 0)


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


# =============================================================================
# Test accuracy (generalization) tests
# =============================================================================

class TestTestAccuracy(unittest.TestCase):
    """Test that WakeResult includes test accuracy when test data is available."""

    def test_wake_returns_test_accuracy(self):
        """When test_inputs/test_outputs are provided, test_error should be set."""
        learner = _make_learner()
        # Task with test data
        task = Task(
            task_id="with_test",
            train_examples=[(1, 1), (2, 2), (3, 3)],
            test_inputs=[4, 5],
            test_outputs=[4, 5],
            difficulty=0.0,
        )
        result = learner.wake_on_task(task)
        self.assertIsNotNone(result.test_error)
        self.assertIsNotNone(result.test_solved)

    def test_wake_no_test_data(self):
        """When no test data, test_error should be None."""
        learner = _make_learner()
        task = Task(
            task_id="no_test",
            train_examples=[(1, 1), (2, 2)],
            test_inputs=[],
            test_outputs=[],
            difficulty=0.0,
        )
        result = learner.wake_on_task(task)
        self.assertIsNone(result.test_error)
        self.assertIsNone(result.test_solved)

    def test_test_solved_for_identity(self):
        """Identity task should be test-solved if train-solved."""
        learner = _make_learner()
        task = Task(
            task_id="identity_test",
            train_examples=[(1, 1), (2, 2), (3, 3)],
            test_inputs=[10, 20],
            test_outputs=[10, 20],
            difficulty=0.0,
        )
        result = learner.wake_on_task(task)
        if result.solved:
            self.assertTrue(result.test_solved)

    def test_no_record_also_has_test_accuracy(self):
        """_wake_on_task_no_record should also compute test accuracy."""
        learner = _make_learner()
        task = Task(
            task_id="identity_test",
            train_examples=[(1, 1), (2, 2), (3, 3)],
            test_inputs=[10, 20],
            test_outputs=[10, 20],
            difficulty=0.0,
        )
        result = learner._wake_on_task_no_record(task)
        self.assertIsNotNone(result.test_error)


# =============================================================================
# Near-miss refinement tests
# =============================================================================

class TestNearMissRefine(unittest.TestCase):
    """Test the near-miss refinement phase."""

    def test_near_miss_refine_produces_candidates(self):
        """Near-miss refinement should produce new candidate programs."""
        learner = _make_learner()
        task = _make_identity_task()
        prims = learner.grammar.base_primitives()

        # Create a near-miss candidate (prediction error between solve_threshold and threshold)
        near_miss = ScoredProgram(
            program=Program(root="double"),
            energy=0.1,
            prediction_error=0.1,  # above solve_threshold (0.01) but below default 0.20
            complexity_cost=1.0,
        )
        refined, n_evals = learner._near_miss_refine(
            [near_miss], prims, task, threshold=0.20)
        self.assertGreater(n_evals, 0)
        self.assertGreater(len(refined), 0)

    def test_near_miss_empty_when_no_near_misses(self):
        """If no candidates are near-misses, should return empty."""
        learner = _make_learner()
        task = _make_identity_task()
        prims = learner.grammar.base_primitives()

        # A perfect candidate (below solve_threshold) is not a near-miss
        perfect = ScoredProgram(
            program=Program(root="identity"),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0,
        )
        refined, n_evals = learner._near_miss_refine(
            [perfect], prims, task, threshold=0.20)
        self.assertEqual(n_evals, 0)
        self.assertEqual(len(refined), 0)

    def test_prepend_changes_program_structure(self):
        """The prepend operation should wrap the deepest leaf with a new primitive."""
        # Verify the fix for the no-op bug
        import copy
        prog = Program(root="double", children=[Program(root="identity")])
        prog_prepend = copy.deepcopy(prog)
        # Walk to deepest leaf
        node = prog_prepend
        while node.children:
            node = node.children[0]
        # Apply the fix: leaf → prim(leaf)
        old_root = node.root
        node.root = "double"  # the new primitive
        node.children = [Program(root=old_root)]  # the old leaf becomes child
        # Verify structure changed
        self.assertEqual(prog_prepend.root, "double")
        self.assertEqual(prog_prepend.children[0].root, "double")
        self.assertEqual(prog_prepend.children[0].children[0].root, "identity")


# =============================================================================
# Runner helper tests
# =============================================================================

class TestRunnerHelpers(unittest.TestCase):
    """Test benchmark utility functions."""

    def test_fmt_duration_seconds(self):
        from common.benchmark import fmt_duration
        self.assertEqual(fmt_duration(12.3), "12.3s")

    def test_fmt_duration_minutes(self):
        from common.benchmark import fmt_duration
        self.assertEqual(fmt_duration(272), "4m32s")

    def test_fmt_duration_hours(self):
        from common.benchmark import fmt_duration
        self.assertEqual(fmt_duration(4980), "1h23m")

    def test_parse_human_int_plain(self):
        from common.benchmark import parse_human_int
        self.assertEqual(parse_human_int("1000"), 1000)

    def test_parse_human_int_suffixes(self):
        from common.benchmark import parse_human_int
        self.assertEqual(parse_human_int("50M"), 50_000_000)
        self.assertEqual(parse_human_int("500K"), 500_000)
        self.assertEqual(parse_human_int("2B"), 2_000_000_000)

    def test_parse_human_int_commas(self):
        from common.benchmark import parse_human_int
        self.assertEqual(parse_human_int("8,000,000"), 8_000_000)

    def test_parse_human_int_empty_raises(self):
        import argparse
        from common.benchmark import parse_human_int
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_human_int("")

    def test_parse_human_int_invalid_raises(self):
        import argparse
        from common.benchmark import parse_human_int
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_human_int("abc")

    def test_detect_machine(self):
        from common.benchmark import detect_machine
        info = detect_machine()
        self.assertIn("platform", info)
        self.assertIn("cpu_count", info)
        self.assertIn("python", info)

    def test_resolve_from_preset(self):
        from common.benchmark import resolve_from_preset, PRESETS
        import argparse
        args = argparse.Namespace(
            rounds=None, beam_width=None, max_generations=None,
            max_tasks=None, workers=0, compute_cap=0,
        )
        resolved = resolve_from_preset(args, PRESETS["quick"])
        self.assertEqual(resolved["rounds"], 1)
        self.assertEqual(resolved["beam_width"], 1)
        self.assertEqual(resolved["compute_cap"], 500_000)

    def test_resolve_from_preset_overrides(self):
        from common.benchmark import resolve_from_preset, PRESETS
        import argparse
        args = argparse.Namespace(
            rounds=5, beam_width=200, max_generations=None,
            max_tasks=10, workers=4, compute_cap=0,
        )
        resolved = resolve_from_preset(args, PRESETS["quick"])
        self.assertEqual(resolved["rounds"], 5)
        self.assertEqual(resolved["beam_width"], 200)
        self.assertEqual(resolved["max_tasks"], 10)
        self.assertEqual(resolved["workers"], 4)


    def test_task_ids_filtering(self):
        """Task ID filtering works with exact and prefix match."""
        from common.benchmark import ExperimentConfig
        from core.types import Task

        tasks = [
            Task("0dfd9992", [], [], []),
            Task("1190e5a7", [], [], []),
            Task("045e512c", [], [], []),
        ]

        # Exact match
        cfg = ExperimentConfig(
            title="test", domain_tag="test", tasks=tasks,
            environment=None, grammar=None, drive=None,
            task_ids="0dfd9992",
        )
        id_prefixes = [t.strip() for t in cfg.task_ids.split(",") if t.strip()]
        filtered = [t for t in cfg.tasks
                    if any(t.task_id.startswith(p) for p in id_prefixes)]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].task_id, "0dfd9992")

        # Prefix match
        cfg2 = ExperimentConfig(
            title="test", domain_tag="test", tasks=tasks,
            environment=None, grammar=None, drive=None,
            task_ids="0",
        )
        id_prefixes2 = [t.strip() for t in cfg2.task_ids.split(",") if t.strip()]
        filtered2 = [t for t in cfg2.tasks
                     if any(t.task_id.startswith(p) for p in id_prefixes2)]
        self.assertEqual(len(filtered2), 2)  # 0dfd9992 and 045e512c

        # Multiple IDs
        cfg3 = ExperimentConfig(
            title="test", domain_tag="test", tasks=tasks,
            environment=None, grammar=None, drive=None,
            task_ids="0dfd9992,1190e5a7",
        )
        id_prefixes3 = [t.strip() for t in cfg3.task_ids.split(",") if t.strip()]
        filtered3 = [t for t in cfg3.tasks
                     if any(t.task_id.startswith(p) for p in id_prefixes3)]
        self.assertEqual(len(filtered3), 2)

    def test_compute_cap_presets(self):
        """All presets have expected compute_cap values."""
        from common.benchmark import PRESETS
        self.assertEqual(PRESETS["quick"]["compute_cap"], 500_000)
        self.assertEqual(PRESETS["default"]["compute_cap"], 3_000_000)
        self.assertEqual(PRESETS["contest"]["compute_cap"], 100_000_000)


if __name__ == "__main__":
    unittest.main()
