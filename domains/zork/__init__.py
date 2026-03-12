"""
Zork-style text adventure domain for the Universal Learning Loop.

Proves the architecture handles sequential, stateful, goal-directed domains
(not just grid transformations or symbolic regression).

The "world" is a small graph of rooms with items and locked doors.
Programs are action sequences (move, take, use, etc.).
The drive signal measures progress toward a goal state.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from core import (
    Environment,
    Grammar,
    DriveSignal,
    Primitive,
    Program,
    Task,
    Observation,
    LibraryEntry,
)


# =============================================================================
# World model: rooms, items, doors
# =============================================================================

@dataclass
class Room:
    """A location in the text adventure."""
    name: str
    description: str = ""
    exits: dict[str, str] = field(default_factory=dict)  # direction → room_name
    items: list[str] = field(default_factory=list)
    locked_exits: dict[str, str] = field(default_factory=dict)  # direction → key_item


@dataclass
class GameState:
    """Complete snapshot of the game world."""
    rooms: dict[str, Room]
    player_room: str
    inventory: list[str] = field(default_factory=list)
    score: int = 0
    max_score: int = 0
    flags: set[str] = field(default_factory=set)  # persistent state flags

    def copy(self) -> GameState:
        return GameState(
            rooms={name: Room(
                name=r.name,
                description=r.description,
                exits=dict(r.exits),
                items=list(r.items),
                locked_exits=dict(r.locked_exits),
            ) for name, r in self.rooms.items()},
            player_room=self.player_room,
            inventory=list(self.inventory),
            score=self.score,
            max_score=self.max_score,
            flags=set(self.flags),
        )


def _execute_action(state: GameState, action: str) -> GameState:
    """Execute a single action and return the new state."""
    state = state.copy()
    parts = action.split("_", 1)
    verb = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    room = state.rooms.get(state.player_room)
    if room is None:
        return state

    if verb == "go":
        direction = arg
        # Check locked exits first
        if direction in room.locked_exits:
            key = room.locked_exits[direction]
            if key in state.inventory:
                # Unlock: remove lock, allow passage
                del room.locked_exits[direction]
                state.flags.add(f"unlocked_{direction}_{room.name}")
            else:
                return state  # can't go, locked
        if direction in room.exits:
            state.player_room = room.exits[direction]

    elif verb == "take":
        item = arg
        if item in room.items:
            room.items.remove(item)
            state.inventory.append(item)
            state.score += 1

    elif verb == "drop":
        item = arg
        if item in state.inventory:
            state.inventory.remove(item)
            room.items.append(item)

    elif verb == "use":
        item = arg
        if item in state.inventory:
            state.flags.add(f"used_{item}")
            state.score += 2

    elif verb == "look":
        pass  # no-op, useful for predicates

    return state


# =============================================================================
# Primitives: atomic actions
# =============================================================================

# Directions
DIRECTIONS = ["north", "south", "east", "west"]

# Core action primitives
ZORK_PRIMITIVES: list[Primitive] = []

# Movement: go_north, go_south, go_east, go_west
for d in DIRECTIONS:
    name = f"go_{d}"
    ZORK_PRIMITIVES.append(Primitive(
        name=name, arity=1,
        fn=lambda state, _d=d: _execute_action(state, f"go_{_d}"),
        domain="zork",
    ))

# Item interactions (parameterized at task time, but we define the common ones)
COMMON_ITEMS = ["key", "sword", "lamp", "food", "treasure", "potion", "map", "gem"]

for item in COMMON_ITEMS:
    ZORK_PRIMITIVES.append(Primitive(
        name=f"take_{item}", arity=1,
        fn=lambda state, _i=item: _execute_action(state, f"take_{_i}"),
        domain="zork",
    ))
    ZORK_PRIMITIVES.append(Primitive(
        name=f"use_{item}", arity=1,
        fn=lambda state, _i=item: _execute_action(state, f"use_{_i}"),
        domain="zork",
    ))
    ZORK_PRIMITIVES.append(Primitive(
        name=f"drop_{item}", arity=1,
        fn=lambda state, _i=item: _execute_action(state, f"drop_{_i}"),
        domain="zork",
    ))

# Identity (no-op / wait)
ZORK_PRIMITIVES.append(Primitive(
    name="wait", arity=1,
    fn=lambda state: state,
    domain="zork",
))

# Look (observation, no state change)
ZORK_PRIMITIVES.append(Primitive(
    name="look", arity=1,
    fn=lambda state: _execute_action(state, "look"),
    domain="zork",
))

_ZORK_PRIM_MAP = {p.name: p for p in ZORK_PRIMITIVES}


# =============================================================================
# Predicates for conditional branching
# =============================================================================

def _has_item(item: str):
    def pred(state: GameState) -> bool:
        return item in state.inventory
    return pred

def _in_room(room_name: str):
    def pred(state: GameState) -> bool:
        return state.player_room == room_name
    return pred

def _room_has_item(item: str):
    def pred(state: GameState) -> bool:
        room = state.rooms.get(state.player_room)
        return room is not None and item in room.items
    return pred

ZORK_PREDICATES: list[tuple[str, Any]] = []
for item in COMMON_ITEMS:
    ZORK_PREDICATES.append((f"has_{item}", _has_item(item)))
    ZORK_PREDICATES.append((f"room_has_{item}", _room_has_item(item)))


# =============================================================================
# Environment
# =============================================================================

class ZorkEnv(Environment):
    """Execute action-sequence programs in a text adventure world."""

    def load_task(self, task: Task) -> Observation:
        return Observation(data=task.train_examples)

    def execute(self, program: Program, input_data: Any) -> Any:
        """Execute a program tree as an action sequence on a GameState.

        Programs compose as sequential actions: outer(inner(state)).
        A tree f(g(x)) means: first do g, then do f on the result.
        """
        state = input_data
        if not isinstance(state, GameState):
            return state

        # Process children first (innermost actions execute first)
        if program.children:
            for child in program.children:
                state = self.execute(child, state)

        # Apply this node's action
        prim = _ZORK_PRIM_MAP.get(program.root)
        if prim and prim.fn:
            try:
                # Library entries have fn=Program (a stored sub-tree).
                # Execute the stored program recursively.
                if isinstance(prim.fn, Program):
                    return self.execute(prim.fn, state)
                result = prim.fn(state)
                if isinstance(result, GameState):
                    return result
            except Exception:
                pass
        return state

    def reset(self) -> None:
        pass


# =============================================================================
# Grammar
# =============================================================================

class ZorkGrammar(Grammar):
    """Grammar for composing text adventure action sequences."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._task_prims: list[Primitive] = []

    def base_primitives(self) -> list[Primitive]:
        return list(ZORK_PRIMITIVES) + self._task_prims

    def get_predicates(self) -> list[tuple[str, Any]]:
        return list(ZORK_PREDICATES)

    def prepare_for_task(self, task: Task) -> None:
        """Analyze task to discover items not in the common set."""
        self._task_prims = []
        if not task.train_examples:
            return

        # Extract items from initial game states
        seen_items: set[str] = set()
        for start_state, _ in task.train_examples:
            if isinstance(start_state, GameState):
                for room in start_state.rooms.values():
                    seen_items.update(room.items)
                seen_items.update(start_state.inventory)

        # Create primitives for items not in COMMON_ITEMS
        known = set(COMMON_ITEMS)
        for item in seen_items - known:
            for verb, fn_factory in [
                ("take", lambda s, _i=item: _execute_action(s, f"take_{_i}")),
                ("use", lambda s, _i=item: _execute_action(s, f"use_{_i}")),
                ("drop", lambda s, _i=item: _execute_action(s, f"drop_{_i}")),
            ]:
                name = f"{verb}_{item}"
                if name not in _ZORK_PRIM_MAP:
                    p = Primitive(name=name, arity=1, fn=fn_factory, domain="zork")
                    self._task_prims.append(p)
                    _ZORK_PRIM_MAP[name] = p

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def mutate(self, program: Program, primitives: list[Primitive],
               transition_matrix=None) -> Program:
        """Mutate an action sequence program."""
        prog = copy.deepcopy(program)
        nodes = self._collect_nodes(prog)
        if not nodes:
            return prog

        r = self._rng.random()
        if r < 0.25:
            # GROW: extend the sequence by adding a step
            leaves = [n for n in nodes if not n.children]
            if leaves:
                target = self._rng.choice(leaves)
                action_prims = [p for p in primitives if p.arity >= 1]
                if action_prims:
                    new_op = self._pick(action_prims, target.root, transition_matrix)
                    leaf_prims = [p for p in primitives if p.arity <= 1]
                    if not leaf_prims:
                        leaf_prims = primitives
                    children = [
                        Program(root=self._pick(leaf_prims, new_op.name, transition_matrix).name)
                        for _ in range(new_op.arity)
                    ]
                    target.root = new_op.name
                    target.children = children
            return prog
        elif r < 0.35:
            # SHRINK: remove a step from the sequence
            internals = [n for n in nodes if n.children]
            if internals:
                target = self._rng.choice(internals)
                leaf_prims = [p for p in primitives if p.arity <= 1]
                if leaf_prims:
                    new_leaf = self._pick(leaf_prims, target.root, transition_matrix)
                    target.root = new_leaf.name
                    target.children = []
            return prog
        else:
            # POINT: swap one action for another
            target = self._rng.choice(nodes)
            prim = _ZORK_PRIM_MAP.get(target.root)
            current_arity = prim.arity if prim else 1
            same_arity = [p for p in primitives if p.arity == current_arity]
            if same_arity:
                parent_op = program.root if program.children else ""
                new_prim = self._pick(same_arity, parent_op, transition_matrix)
                target.root = new_prim.name
            return prog

    def _pick(self, candidates: list[Primitive], parent_op: str,
              transition_matrix) -> Primitive:
        """Pick a primitive, biased by transition matrix if available."""
        if transition_matrix and transition_matrix.size > 0 and parent_op:
            return transition_matrix.weighted_choice(parent_op, candidates, self._rng)
        return self._rng.choice(candidates)

    def crossover(self, a: Program, b: Program) -> Program:
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)
        a_nodes = self._collect_nodes(a_copy)
        b_nodes = self._collect_nodes(b_copy)
        if not a_nodes or not b_nodes:
            return a_copy
        target = self._rng.choice(a_nodes)
        donor = self._rng.choice(b_nodes)
        target.root = donor.root
        target.children = donor.children
        return a_copy

    def _collect_nodes(self, program: Program) -> list[Program]:
        result = [program]
        for child in program.children:
            result.extend(self._collect_nodes(child))
        return result


# =============================================================================
# Drive signal
# =============================================================================

class ZorkDrive(DriveSignal):
    """Score progress toward a goal state.

    Measures:
    - Room match (is the player in the right room?)
    - Inventory match (does the player have the right items?)
    - Score match (game score vs max possible)
    - Flags match (have required actions been performed?)
    """

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        if not isinstance(predicted, GameState) or not isinstance(expected, GameState):
            return 1.0

        errors = []

        # Room match (0 or 1)
        room_match = 1.0 if predicted.player_room == expected.player_room else 0.0
        errors.append(0.40 * (1.0 - room_match))

        # Inventory match (Jaccard distance)
        pred_inv = set(predicted.inventory)
        exp_inv = set(expected.inventory)
        if pred_inv or exp_inv:
            jaccard = len(pred_inv & exp_inv) / len(pred_inv | exp_inv)
        else:
            jaccard = 1.0
        errors.append(0.30 * (1.0 - jaccard))

        # Score match (normalized)
        if expected.max_score > 0:
            score_ratio = min(predicted.score, expected.max_score) / expected.max_score
        else:
            score_ratio = 1.0 if predicted.score == expected.score else 0.0
        errors.append(0.15 * (1.0 - score_ratio))

        # Flags match (Jaccard)
        pred_flags = predicted.flags
        exp_flags = expected.flags
        if pred_flags or exp_flags:
            flag_jaccard = len(pred_flags & exp_flags) / len(pred_flags | exp_flags)
        else:
            flag_jaccard = 1.0
        errors.append(0.15 * (1.0 - flag_jaccard))

        return sum(errors)


# =============================================================================
# Sample tasks
# =============================================================================

def _make_simple_world() -> dict[str, Room]:
    """A 4-room world: entrance → hallway → treasure_room, hallway → armory."""
    return {
        "entrance": Room(
            name="entrance",
            description="A dusty entrance hall.",
            exits={"north": "hallway"},
            items=["lamp"],
        ),
        "hallway": Room(
            name="hallway",
            description="A long stone hallway.",
            exits={"south": "entrance", "north": "treasure_room", "east": "armory"},
        ),
        "armory": Room(
            name="armory",
            description="Weapons line the walls.",
            exits={"west": "hallway"},
            items=["sword"],
        ),
        "treasure_room": Room(
            name="treasure_room",
            description="Gold glitters everywhere!",
            exits={"south": "hallway"},
            items=["treasure"],
        ),
    }


def _make_locked_world() -> dict[str, Room]:
    """A world with a locked door requiring a key."""
    return {
        "start": Room(
            name="start",
            description="A small room.",
            exits={"east": "key_room", "north": "locked_passage"},
            locked_exits={"north": "key"},
        ),
        "key_room": Room(
            name="key_room",
            description="A closet.",
            exits={"west": "start"},
            items=["key"],
        ),
        "locked_passage": Room(
            name="locked_passage",
            description="Beyond the locked door.",
            exits={"south": "start", "north": "goal"},
        ),
        "goal": Room(
            name="goal",
            description="You've reached the goal!",
            exits={"south": "locked_passage"},
            items=["gem"],
        ),
    }


def get_sample_tasks() -> list[Task]:
    """Return sample Zork tasks for testing."""
    tasks = []

    # Task 1: Navigate to treasure room and take treasure
    world1 = _make_simple_world()
    start1 = GameState(rooms=world1, player_room="entrance", max_score=1)
    # Expected: player in treasure_room with treasure in inventory
    goal1_rooms = _make_simple_world()
    goal1_rooms["treasure_room"].items = []  # treasure taken
    goal1 = GameState(
        rooms=goal1_rooms, player_room="treasure_room",
        inventory=["treasure"], score=1, max_score=1,
    )
    tasks.append(Task(
        task_id="zork_navigate_take",
        train_examples=[(start1, goal1)],
        test_inputs=[start1],
        test_outputs=[goal1],
        difficulty=2.0,
    ))

    # Task 2: Take lamp, go to hallway
    world2 = _make_simple_world()
    start2 = GameState(rooms=world2, player_room="entrance", max_score=1)
    goal2_rooms = _make_simple_world()
    goal2_rooms["entrance"].items = []  # lamp taken
    goal2 = GameState(
        rooms=goal2_rooms, player_room="hallway",
        inventory=["lamp"], score=1, max_score=1,
    )
    tasks.append(Task(
        task_id="zork_take_and_move",
        train_examples=[(start2, goal2)],
        test_inputs=[start2],
        test_outputs=[goal2],
        difficulty=1.5,
    ))

    # Task 3: Unlock door, reach goal, take gem
    world3 = _make_locked_world()
    start3 = GameState(rooms=world3, player_room="start", max_score=2)
    goal3_rooms = _make_locked_world()
    goal3_rooms["key_room"].items = []  # key taken
    goal3_rooms["goal"].items = []  # gem taken
    del goal3_rooms["start"].locked_exits["north"]  # unlocked
    goal3 = GameState(
        rooms=goal3_rooms, player_room="goal",
        inventory=["key", "gem"], score=2, max_score=2,
        flags={"unlocked_north_start"},
    )
    tasks.append(Task(
        task_id="zork_locked_door",
        train_examples=[(start3, goal3)],
        test_inputs=[start3],
        test_outputs=[goal3],
        difficulty=4.0,
    ))

    # Task 4: Simple movement only — go north twice
    world4 = _make_simple_world()
    start4 = GameState(rooms=world4, player_room="entrance", max_score=0)
    goal4 = GameState(rooms=_make_simple_world(), player_room="treasure_room", max_score=0)
    tasks.append(Task(
        task_id="zork_navigate_only",
        train_examples=[(start4, goal4)],
        test_inputs=[start4],
        test_outputs=[goal4],
        difficulty=1.0,
    ))

    return tasks
