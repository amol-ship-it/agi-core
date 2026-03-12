"""
Tests for the Zork text adventure domain.

Tests the game engine, primitives, environment, grammar, drive signal,
and integration with the core learner.
"""

import copy
import unittest

from core.types import Primitive, Program, Task, LibraryEntry
from core.config import SearchConfig, SleepConfig
from core.learner import Learner
from core.memory import InMemoryStore

from domains.zork import (
    Room, GameState,
    _execute_action,
    ZorkEnv, ZorkGrammar, ZorkDrive,
    ZORK_PRIMITIVES, ZORK_PREDICATES,
    _ZORK_PRIM_MAP,
    get_sample_tasks,
    _make_simple_world, _make_locked_world,
)


# =============================================================================
# Game engine tests
# =============================================================================

class TestGameState(unittest.TestCase):

    def test_copy_is_independent(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        copy_state = state.copy()
        copy_state.player_room = "hallway"
        copy_state.inventory.append("lamp")
        self.assertEqual(state.player_room, "entrance")
        self.assertEqual(state.inventory, [])

    def test_go_valid_direction(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        new_state = _execute_action(state, "go_north")
        self.assertEqual(new_state.player_room, "hallway")
        self.assertEqual(state.player_room, "entrance")  # original unchanged

    def test_go_invalid_direction(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        new_state = _execute_action(state, "go_west")
        self.assertEqual(new_state.player_room, "entrance")  # no exit west

    def test_take_item(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        new_state = _execute_action(state, "take_lamp")
        self.assertIn("lamp", new_state.inventory)
        self.assertNotIn("lamp", new_state.rooms["entrance"].items)
        self.assertEqual(new_state.score, 1)

    def test_take_nonexistent_item(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        new_state = _execute_action(state, "take_sword")
        self.assertEqual(new_state.inventory, [])

    def test_drop_item(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance", inventory=["key"])
        new_state = _execute_action(state, "drop_key")
        self.assertNotIn("key", new_state.inventory)
        self.assertIn("key", new_state.rooms["entrance"].items)

    def test_use_item(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance", inventory=["lamp"])
        new_state = _execute_action(state, "use_lamp")
        self.assertIn("used_lamp", new_state.flags)
        self.assertEqual(new_state.score, 2)

    def test_locked_door_without_key(self):
        rooms = _make_locked_world()
        state = GameState(rooms=rooms, player_room="start")
        new_state = _execute_action(state, "go_north")
        self.assertEqual(new_state.player_room, "start")  # still locked

    def test_locked_door_with_key(self):
        rooms = _make_locked_world()
        state = GameState(rooms=rooms, player_room="start", inventory=["key"])
        new_state = _execute_action(state, "go_north")
        self.assertEqual(new_state.player_room, "locked_passage")
        self.assertIn("unlocked_north_start", new_state.flags)

    def test_look_is_noop(self):
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        new_state = _execute_action(state, "look")
        self.assertEqual(new_state.player_room, state.player_room)


# =============================================================================
# Primitive tests
# =============================================================================

class TestZorkPrimitives(unittest.TestCase):

    def test_primitives_count(self):
        # 4 directions + 8 items × 3 verbs + wait + look
        expected = 4 + 8 * 3 + 2
        self.assertEqual(len(ZORK_PRIMITIVES), expected)

    def test_all_primitives_have_functions(self):
        for prim in ZORK_PRIMITIVES:
            self.assertIsNotNone(prim.fn, f"{prim.name} has no function")

    def test_go_north_primitive(self):
        prim = _ZORK_PRIM_MAP["go_north"]
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        result = prim.fn(state)
        self.assertEqual(result.player_room, "hallway")

    def test_take_lamp_primitive(self):
        prim = _ZORK_PRIM_MAP["take_lamp"]
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        result = prim.fn(state)
        self.assertIn("lamp", result.inventory)

    def test_predicates_exist(self):
        self.assertGreater(len(ZORK_PREDICATES), 0)
        names = [name for name, _ in ZORK_PREDICATES]
        self.assertIn("has_key", names)
        self.assertIn("room_has_sword", names)

    def test_has_item_predicate(self):
        pred = dict(ZORK_PREDICATES)["has_key"]
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance", inventory=["key"])
        self.assertTrue(pred(state))
        state2 = GameState(rooms=_make_simple_world(), player_room="entrance")
        self.assertFalse(pred(state2))


# =============================================================================
# Environment tests
# =============================================================================

class TestZorkEnv(unittest.TestCase):

    def test_execute_single_action(self):
        env = ZorkEnv()
        prog = Program(root="go_north")
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        result = env.execute(prog, state)
        self.assertEqual(result.player_room, "hallway")

    def test_execute_sequence(self):
        """go_north(go_north(state)) should go entrance → hallway → treasure_room."""
        env = ZorkEnv()
        prog = Program(root="go_north", children=[
            Program(root="go_north")
        ])
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        result = env.execute(prog, state)
        self.assertEqual(result.player_room, "treasure_room")

    def test_execute_take_then_move(self):
        """go_north(take_lamp(state))"""
        env = ZorkEnv()
        prog = Program(root="go_north", children=[
            Program(root="take_lamp")
        ])
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        result = env.execute(prog, state)
        self.assertEqual(result.player_room, "hallway")
        self.assertIn("lamp", result.inventory)

    def test_non_gamestate_input(self):
        """Should handle non-GameState input gracefully."""
        env = ZorkEnv()
        prog = Program(root="go_north")
        result = env.execute(prog, "not_a_state")
        self.assertEqual(result, "not_a_state")


# =============================================================================
# Grammar tests
# =============================================================================

class TestZorkGrammar(unittest.TestCase):

    def test_base_primitives(self):
        grammar = ZorkGrammar()
        prims = grammar.base_primitives()
        self.assertGreater(len(prims), 20)

    def test_compose(self):
        grammar = ZorkGrammar()
        outer = _ZORK_PRIM_MAP["go_north"]
        inner = [Program(root="take_lamp")]
        prog = grammar.compose(outer, inner)
        self.assertEqual(prog.root, "go_north")
        self.assertEqual(len(prog.children), 1)

    def test_mutate(self):
        grammar = ZorkGrammar(seed=42)
        prog = Program(root="go_north")
        prims = grammar.base_primitives()
        mutant = grammar.mutate(prog, prims)
        self.assertIsInstance(mutant, Program)

    def test_crossover(self):
        grammar = ZorkGrammar(seed=42)
        a = Program(root="go_north", children=[Program(root="take_lamp")])
        b = Program(root="go_south", children=[Program(root="take_sword")])
        child = grammar.crossover(a, b)
        self.assertIsInstance(child, Program)

    def test_prepare_for_task_discovers_items(self):
        grammar = ZorkGrammar()
        rooms = {"room": Room(name="room", items=["magic_ring"])}
        state = GameState(rooms=rooms, player_room="room")
        task = Task(
            task_id="test", train_examples=[(state, state)],
            test_inputs=[], test_outputs=[], difficulty=1.0,
        )
        grammar.prepare_for_task(task)
        prim_names = [p.name for p in grammar.base_primitives()]
        self.assertIn("take_magic_ring", prim_names)

    def test_predicates(self):
        grammar = ZorkGrammar()
        preds = grammar.get_predicates()
        self.assertGreater(len(preds), 0)


# =============================================================================
# Drive signal tests
# =============================================================================

class TestZorkDrive(unittest.TestCase):

    def test_perfect_match(self):
        drive = ZorkDrive()
        rooms = _make_simple_world()
        state = GameState(rooms=rooms, player_room="entrance")
        error = drive.prediction_error(state, state)
        self.assertAlmostEqual(error, 0.0, places=5)

    def test_wrong_room(self):
        drive = ZorkDrive()
        predicted = GameState(rooms=_make_simple_world(), player_room="entrance")
        expected = GameState(rooms=_make_simple_world(), player_room="hallway")
        error = drive.prediction_error(predicted, expected)
        self.assertGreater(error, 0.0)
        # Room mismatch contributes up to 40% (with distance-based partial credit)
        self.assertGreater(error, 0.0)
        self.assertLessEqual(error, 0.40)

    def test_wrong_inventory(self):
        drive = ZorkDrive()
        predicted = GameState(rooms=_make_simple_world(), player_room="entrance")
        expected = GameState(rooms=_make_simple_world(), player_room="entrance",
                           inventory=["lamp"])
        error = drive.prediction_error(predicted, expected)
        self.assertGreater(error, 0.0)
        # Inventory mismatch contributes 30%
        self.assertAlmostEqual(error, 0.30, places=2)

    def test_partial_inventory_match(self):
        drive = ZorkDrive()
        predicted = GameState(rooms=_make_simple_world(), player_room="entrance",
                            inventory=["lamp", "sword"])
        expected = GameState(rooms=_make_simple_world(), player_room="entrance",
                           inventory=["lamp"])
        error = drive.prediction_error(predicted, expected)
        # Jaccard({lamp, sword}, {lamp}) = 1/2 → 0.30 * 0.5 = 0.15
        self.assertAlmostEqual(error, 0.15, places=2)

    def test_non_gamestate_input(self):
        drive = ZorkDrive()
        self.assertEqual(drive.prediction_error("a", "b"), 1.0)

    def test_score_progress(self):
        drive = ZorkDrive()
        predicted = GameState(rooms=_make_simple_world(), player_room="entrance",
                            score=1, max_score=2)
        expected = GameState(rooms=_make_simple_world(), player_room="entrance",
                           score=2, max_score=2)
        error = drive.prediction_error(predicted, expected)
        # Score contributes: 0.15 * (1 - 0.5) = 0.075
        self.assertGreater(error, 0.0)


# =============================================================================
# Sample tasks
# =============================================================================

class TestZorkSampleTasks(unittest.TestCase):

    def test_sample_tasks_load(self):
        tasks = get_sample_tasks()
        self.assertGreaterEqual(len(tasks), 4)
        for task in tasks:
            self.assertIsInstance(task, Task)
            self.assertGreater(len(task.train_examples), 0)

    def test_navigate_only_solvable(self):
        """The simplest task (go north twice) should be solvable by enumeration."""
        env = ZorkEnv()
        tasks = get_sample_tasks()
        nav_task = [t for t in tasks if t.task_id == "zork_navigate_only"][0]
        start, goal = nav_task.train_examples[0]

        # Manual solution: go_north(go_north(state))
        prog = Program(root="go_north", children=[Program(root="go_north")])
        result = env.execute(prog, start)
        self.assertEqual(result.player_room, goal.player_room)


# =============================================================================
# Integration with core learner
# =============================================================================

class TestZorkLearnerIntegration(unittest.TestCase):

    def _make_zork_learner(self, **kwargs):
        defaults = dict(
            beam_width=10,
            max_generations=5,
            solve_threshold=0.01,
            seed=42,
            exhaustive_depth=2,
            exhaustive_pair_top_k=10,
        )
        defaults.update(kwargs)
        return Learner(
            environment=ZorkEnv(),
            grammar=ZorkGrammar(seed=42),
            drive=ZorkDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(**defaults),
        )

    def test_learner_runs_on_zork(self):
        """Core learner should work with Zork domain without crashing."""
        learner = self._make_zork_learner()
        tasks = get_sample_tasks()
        # Try the simplest task
        nav_task = [t for t in tasks if t.task_id == "zork_navigate_only"][0]
        result = learner.wake_on_task(nav_task)
        self.assertIsNotNone(result)
        self.assertGreater(result.evaluations, 0)

    def test_sleep_works_after_zork_wake(self):
        learner = self._make_zork_learner()
        tasks = get_sample_tasks()
        nav_task = [t for t in tasks if t.task_id == "zork_navigate_only"][0]
        learner.wake_on_task(nav_task)
        result = learner.sleep()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
