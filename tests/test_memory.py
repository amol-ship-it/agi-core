"""
Tests for core/memory.py — InMemoryStore.

Verifies:
1. Episodic memory: record and replay
2. Library: add, get, update usefulness, eviction
3. Solutions: store and retrieve
4. Best attempts: store and retrieve unsolved programs
"""

import unittest

from core.types import Program, ScoredProgram, LibraryEntry
from core.memory import InMemoryStore


class TestInMemoryStoreEpisodic(unittest.TestCase):

    def test_record_and_replay(self):
        store = InMemoryStore()
        store.record_episode("t1", "obs1", Program(root="x"), 0.5)
        store.record_episode("t2", "obs2", Program(root="y"), 0.3)
        episodes = store.replay_episodes(10)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["task_id"], "t1")
        self.assertEqual(episodes[1]["task_id"], "t2")

    def test_replay_truncates(self):
        store = InMemoryStore()
        for i in range(20):
            store.record_episode(f"t{i}", None, None, float(i))
        episodes = store.replay_episodes(5)
        self.assertEqual(len(episodes), 5)
        self.assertEqual(episodes[0]["task_id"], "t15")

    def test_replay_empty(self):
        store = InMemoryStore()
        self.assertEqual(store.replay_episodes(), [])


class TestInMemoryStoreLibrary(unittest.TestCase):

    def _make_entry(self, name="lib_0", usefulness=1.0):
        return LibraryEntry(
            name=name,
            program=Program(root="f", children=[Program(root="x")]),
            usefulness=usefulness,
        )

    def test_add_and_get(self):
        store = InMemoryStore()
        e = self._make_entry()
        store.add_to_library(e)
        lib = store.get_library()
        self.assertEqual(len(lib), 1)
        self.assertEqual(lib[0].name, "lib_0")

    def test_get_library_returns_copy(self):
        store = InMemoryStore()
        store.add_to_library(self._make_entry())
        lib = store.get_library()
        lib.clear()
        self.assertEqual(len(store.get_library()), 1)

    def test_update_usefulness_positive(self):
        store = InMemoryStore()
        e = self._make_entry(usefulness=1.0)
        store.add_to_library(e)
        store.update_usefulness("lib_0", 2.0)
        lib = store.get_library()
        self.assertAlmostEqual(lib[0].usefulness, 3.0)
        self.assertEqual(lib[0].reuse_count, 1)

    def test_update_usefulness_negative(self):
        store = InMemoryStore()
        e = self._make_entry(usefulness=5.0)
        store.add_to_library(e)
        store.update_usefulness("lib_0", -1.0)
        lib = store.get_library()
        self.assertAlmostEqual(lib[0].usefulness, 4.0)
        self.assertEqual(lib[0].reuse_count, 0)  # negative delta doesn't increment

    def test_update_usefulness_unknown_name(self):
        store = InMemoryStore()
        store.add_to_library(self._make_entry())
        # Should not raise
        store.update_usefulness("nonexistent", 1.0)


class TestInMemoryStoreSolutions(unittest.TestCase):

    def test_store_and_get(self):
        store = InMemoryStore()
        sp = ScoredProgram(
            program=Program(root="x"),
            energy=0.1, prediction_error=0.09, complexity_cost=0.01, task_id="t1",
        )
        store.store_solution("t1", sp)
        sols = store.get_solutions()
        self.assertIn("t1", sols)
        self.assertAlmostEqual(sols["t1"].energy, 0.1)

    def test_overwrite_solution(self):
        store = InMemoryStore()
        sp1 = ScoredProgram(program=Program(root="a"), energy=1.0, prediction_error=1.0, complexity_cost=0.0)
        sp2 = ScoredProgram(program=Program(root="b"), energy=0.5, prediction_error=0.5, complexity_cost=0.0)
        store.store_solution("t1", sp1)
        store.store_solution("t1", sp2)
        sols = store.get_solutions()
        self.assertEqual(sols["t1"].program.root, "b")

    def test_get_solutions_returns_copy(self):
        store = InMemoryStore()
        sp = ScoredProgram(program=Program(root="x"), energy=0.0, prediction_error=0.0, complexity_cost=0.0)
        store.store_solution("t1", sp)
        sols = store.get_solutions()
        sols.clear()
        self.assertEqual(len(store.get_solutions()), 1)


class TestBestAttempts(unittest.TestCase):

    def _sp(self, root="x", error=0.10):
        return ScoredProgram(
            program=Program(root=root),
            energy=error, prediction_error=error, complexity_cost=0.0,
        )

    def test_store_and_get(self):
        store = InMemoryStore()
        store.store_best_attempt("t1", self._sp(error=0.10))
        ba = store.get_best_attempts()
        self.assertIn("t1", ba)
        self.assertAlmostEqual(ba["t1"].prediction_error, 0.10)

    def test_keeps_lowest_error(self):
        store = InMemoryStore()
        store.store_best_attempt("t1", self._sp(error=0.10))
        store.store_best_attempt("t1", self._sp(root="y", error=0.05))
        ba = store.get_best_attempts()
        self.assertEqual(ba["t1"].program.root, "y")

    def test_does_not_overwrite_with_worse(self):
        store = InMemoryStore()
        store.store_best_attempt("t1", self._sp(error=0.05))
        store.store_best_attempt("t1", self._sp(root="y", error=0.10))
        ba = store.get_best_attempts()
        self.assertEqual(ba["t1"].program.root, "x")

    def test_stores_high_error_programs(self):
        """No threshold — even high-error programs are stored."""
        store = InMemoryStore()
        store.store_best_attempt("t1", self._sp(error=0.80))
        ba = store.get_best_attempts()
        self.assertIn("t1", ba)

    def test_empty_by_default(self):
        store = InMemoryStore()
        self.assertEqual(store.get_best_attempts(), {})

    def test_culture_round_trip(self):
        import tempfile, os

        store = InMemoryStore()
        store.store_best_attempt("t1", self._sp(root="crop_to_nonzero", error=0.08))
        store.store_best_attempt("t2", self._sp(root="rotate90", error=0.60))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save_culture(path)
            store2 = InMemoryStore()
            store2.load_culture(path)
            ba = store2.get_best_attempts()
            self.assertEqual(len(ba), 2)
            self.assertEqual(ba["t1"].program.root, "crop_to_nonzero")
            self.assertAlmostEqual(ba["t2"].prediction_error, 0.60)
        finally:
            os.unlink(path)


class TestEviction(unittest.TestCase):
    """Tests for bounded library with eviction."""

    def _entry(self, name, usefulness=1.0, reuse_count=0):
        e = LibraryEntry(
            name=name,
            program=Program(root=name),
            usefulness=usefulness,
        )
        e.reuse_count = reuse_count
        return e

    def test_unbounded_always_accepts(self):
        store = InMemoryStore()  # capacity=0 = unbounded
        for i in range(200):
            self.assertTrue(store.add_to_library(self._entry(f"e{i}")))
        self.assertEqual(len(store.get_library()), 200)

    def test_under_capacity_accepts(self):
        store = InMemoryStore(capacity=5)
        for i in range(5):
            self.assertTrue(store.add_to_library(self._entry(f"e{i}")))
        self.assertEqual(len(store.get_library()), 5)

    def test_evicts_weakest_when_full(self):
        store = InMemoryStore(capacity=3)
        store.add_to_library(self._entry("weak", usefulness=0.1))
        store.add_to_library(self._entry("mid", usefulness=1.0))
        store.add_to_library(self._entry("strong", usefulness=5.0))
        # Full — new entry stronger than weakest
        result = store.add_to_library(self._entry("better", usefulness=0.5))
        self.assertTrue(result)
        names = {e.name for e in store.get_library()}
        self.assertNotIn("weak", names)  # evicted
        self.assertIn("better", names)
        self.assertEqual(len(store.get_library()), 3)

    def test_rejects_when_too_weak(self):
        store = InMemoryStore(capacity=2)
        store.add_to_library(self._entry("a", usefulness=5.0))
        store.add_to_library(self._entry("b", usefulness=3.0))
        # New entry weaker than both
        result = store.add_to_library(self._entry("tiny", usefulness=0.1))
        self.assertFalse(result)
        self.assertEqual(len(store.get_library()), 2)

    def test_reuse_immunity(self):
        store = InMemoryStore(capacity=2)
        store.add_to_library(self._entry("reused", usefulness=0.01, reuse_count=1))
        store.add_to_library(self._entry("fresh", usefulness=0.5))
        # Even though "reused" has lower usefulness, it's immune
        result = store.add_to_library(self._entry("new", usefulness=10.0))
        self.assertTrue(result)
        names = {e.name for e in store.get_library()}
        self.assertIn("reused", names)  # immune — still there
        self.assertNotIn("fresh", names)  # evicted instead

    def test_all_reused_rejects(self):
        store = InMemoryStore(capacity=2)
        store.add_to_library(self._entry("a", usefulness=0.1, reuse_count=1))
        store.add_to_library(self._entry("b", usefulness=0.1, reuse_count=2))
        # All entries reused — no evictable slot
        result = store.add_to_library(self._entry("new", usefulness=100.0))
        self.assertFalse(result)
        self.assertEqual(len(store.get_library()), 2)

    def test_eviction_score_includes_reuse_bonus(self):
        store = InMemoryStore(capacity=3, reuse_bonus=5.0)
        store.add_to_library(self._entry("low_use_high_reuse", usefulness=0.1, reuse_count=0))
        store.add_to_library(self._entry("high_use", usefulness=10.0))
        store.add_to_library(self._entry("mid", usefulness=2.0))
        # Eviction should pick "low_use_high_reuse" (score=0.1, lowest)
        result = store.add_to_library(self._entry("new", usefulness=1.0))
        self.assertTrue(result)
        names = {e.name for e in store.get_library()}
        self.assertNotIn("low_use_high_reuse", names)

    def test_add_returns_bool(self):
        store = InMemoryStore()
        result = store.add_to_library(self._entry("a"))
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_load_culture_truncates(self):
        """Loading culture with more entries than capacity truncates to top-N."""
        import tempfile, os

        # Create a store with no capacity limit, fill with entries
        store1 = InMemoryStore()
        for i in range(10):
            store1.add_to_library(self._entry(f"e{i}", usefulness=float(i)))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store1.save_culture(path)
            # Load into a bounded store
            store2 = InMemoryStore(capacity=3)
            store2.load_culture(path)
            lib = store2.get_library()
            self.assertEqual(len(lib), 3)
            # Should keep highest-usefulness entries
            names = {e.name for e in lib}
            self.assertIn("e9", names)
            self.assertIn("e8", names)
            self.assertIn("e7", names)
        finally:
            os.unlink(path)


class TestPrimitiveGenerality(unittest.TestCase):

    def test_primitive_generality_default(self):
        store = InMemoryStore()
        self.assertEqual(store.get_primitive_generality(), {})

    def test_primitive_generality_with_solutions(self):
        store = InMemoryStore()
        # Two tasks, both use "rotate_90_cw", only one uses "mirror_h"
        store.store_solution("t1", ScoredProgram(
            program=Program(root="rotate_90_cw", children=[Program(root="mirror_h")]),
            energy=0.0, prediction_error=0.0, complexity_cost=2.0, task_id="t1"))
        store.store_solution("t2", ScoredProgram(
            program=Program(root="rotate_90_cw"),
            energy=0.0, prediction_error=0.0, complexity_cost=1.0, task_id="t2"))
        gen = store.get_primitive_generality()
        self.assertAlmostEqual(gen["rotate_90_cw"], 1.0)  # used in 2/2 tasks
        self.assertAlmostEqual(gen["mirror_h"], 0.5)  # used in 1/2 tasks


class TestStratumStats(unittest.TestCase):
    """Tests for stratum-level culture transfer (Task 20)."""

    def test_record_stratum_solve_basic(self):
        store = InMemoryStore()
        store.record_stratum_solve("exhaustive_core", "task_001")
        stats = store.get_stratum_stats()
        self.assertIn("exhaustive_core", stats)
        self.assertEqual(stats["exhaustive_core"]["solved_count"], 1)
        self.assertIn("task_001", stats["exhaustive_core"]["task_ids"])

    def test_record_stratum_solve_accumulates(self):
        store = InMemoryStore()
        store.record_stratum_solve("color_logic", "task_001")
        store.record_stratum_solve("color_logic", "task_002")
        store.record_stratum_solve("object_transform", "task_003")
        stats = store.get_stratum_stats()
        self.assertEqual(stats["color_logic"]["solved_count"], 2)
        self.assertIn("task_002", stats["color_logic"]["task_ids"])
        self.assertEqual(stats["object_transform"]["solved_count"], 1)

    def test_get_stratum_stats_empty_by_default(self):
        store = InMemoryStore()
        self.assertEqual(store.get_stratum_stats(), {})

    def test_stratum_stats_culture_round_trip(self):
        import tempfile, os

        store = InMemoryStore()
        store.record_stratum_solve("exhaustive_core", "t1")
        store.record_stratum_solve("exhaustive_core", "t2")
        store.record_stratum_solve("color_logic", "t3")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save_culture(path)
            store2 = InMemoryStore()
            store2.load_culture(path)
            stats = store2.get_stratum_stats()
            self.assertIn("exhaustive_core", stats)
            self.assertEqual(stats["exhaustive_core"]["solved_count"], 2)
            self.assertIn("t1", stats["exhaustive_core"]["task_ids"])
            self.assertIn("t2", stats["exhaustive_core"]["task_ids"])
            self.assertIn("color_logic", stats)
            self.assertEqual(stats["color_logic"]["solved_count"], 1)
        finally:
            os.unlink(path)

    def test_stratum_stats_merge_on_load(self):
        """Loading culture merges stats (update, not replace)."""
        import tempfile, os

        store = InMemoryStore()
        store.record_stratum_solve("exhaustive_core", "t1")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save_culture(path)
            store2 = InMemoryStore()
            store2.record_stratum_solve("color_logic", "t2")
            store2.load_culture(path)
            stats = store2.get_stratum_stats()
            # Both pre-existing and loaded stats are present
            self.assertIn("exhaustive_core", stats)
            self.assertIn("color_logic", stats)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
