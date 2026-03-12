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

    def __init__(self):
        self._dynamic_prims: dict[str, Primitive] = {}

    def register_primitive(self, primitive: Primitive) -> None:
        """Register a library-learned primitive so execute() can resolve it."""
        self._dynamic_prims[primitive.name] = primitive

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
        prim = _ZORK_PRIM_MAP.get(program.root) or self._dynamic_prims.get(program.root)
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
    - Room proximity (graph distance to goal room, with partial credit)
    - Inventory match (does the player have the right items?)
    - Score match (game score vs max possible)
    - Flags match (have required actions been performed?)
    """

    @staticmethod
    def _bfs_distance(rooms: dict[str, Room], src: str, dst: str) -> int:
        """Shortest path length between two rooms. Returns -1 if unreachable."""
        if src == dst:
            return 0
        visited = {src}
        frontier = [src]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for room_name in frontier:
                room = rooms.get(room_name)
                if room is None:
                    continue
                for neighbor in room.exits.values():
                    if neighbor == dst:
                        return dist
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return -1  # unreachable

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        if not isinstance(predicted, GameState) or not isinstance(expected, GameState):
            return 1.0

        errors = []

        # Room proximity: partial credit based on graph distance
        if predicted.player_room == expected.player_room:
            room_match = 1.0
        else:
            dist = self._bfs_distance(
                predicted.rooms, predicted.player_room, expected.player_room)
            if dist < 0:
                room_match = 0.0  # unreachable
            else:
                # Closer = higher match. dist=1 → 0.5, dist=2 → 0.33, etc.
                room_match = 1.0 / (1.0 + dist)
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


def _make_linear_world(n_rooms: int = 5) -> dict[str, Room]:
    """A linear chain of rooms: room_0 → room_1 → ... → room_{n-1}."""
    rooms = {}
    for i in range(n_rooms):
        exits = {}
        if i > 0:
            exits["south"] = f"room_{i-1}"
        if i < n_rooms - 1:
            exits["north"] = f"room_{i+1}"
        items = []
        if i == n_rooms - 1:
            items = ["prize"]
        rooms[f"room_{i}"] = Room(
            name=f"room_{i}",
            description=f"Room {i}.",
            exits=exits,
            items=items,
        )
    return rooms


def _make_branching_world() -> dict[str, Room]:
    """A world with a central hub and 3 branches, each with an item."""
    return {
        "hub": Room(
            name="hub",
            description="A crossroads.",
            exits={"north": "cave", "east": "garden", "west": "library"},
        ),
        "cave": Room(
            name="cave",
            description="A dark cave.",
            exits={"south": "hub"},
            items=["torch"],
        ),
        "garden": Room(
            name="garden",
            description="A peaceful garden.",
            exits={"west": "hub"},
            items=["flower"],
        ),
        "library": Room(
            name="library",
            description="Shelves of old books.",
            exits={"east": "hub"},
            items=["book"],
        ),
    }


def _make_multi_key_world() -> dict[str, Room]:
    """World requiring two keys to reach the final room."""
    return {
        "foyer": Room(
            name="foyer",
            description="Grand entrance.",
            exits={"north": "hall", "east": "closet"},
        ),
        "closet": Room(
            name="closet",
            description="A small closet.",
            exits={"west": "foyer"},
            items=["silver_key"],
        ),
        "hall": Room(
            name="hall",
            description="A grand hall.",
            exits={"south": "foyer", "north": "locked_north", "east": "study"},
            locked_exits={"north": "silver_key"},
        ),
        "study": Room(
            name="study",
            description="A quiet study.",
            exits={"west": "hall"},
            items=["gold_key"],
        ),
        "locked_north": Room(
            name="locked_north",
            description="Past the silver door.",
            exits={"south": "hall", "north": "vault"},
            locked_exits={"north": "gold_key"},
        ),
        "vault": Room(
            name="vault",
            description="The final vault!",
            exits={"south": "locked_north"},
            items=["diamond"],
        ),
    }


def get_sample_tasks() -> list[Task]:
    """Return sample Zork tasks for testing.

    20 tasks across 4 difficulty levels:
    - Level 1 (depth 1): single actions (move or take)
    - Level 2 (depth 2): two-step sequences
    - Level 3 (depth 3): three-step sequences
    - Level 4 (depth 4+): multi-step with locked doors
    """
    tasks = []

    # ---- Level 1: Single actions (difficulty 1.0) ----

    # T1: Go north once
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=0)
    goal = GameState(rooms=_make_simple_world(), player_room="hallway", max_score=0)
    tasks.append(Task(
        task_id="zork_go_north",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=1.0,
    ))

    # T2: Take lamp from entrance
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=1)
    gw = _make_simple_world()
    gw["entrance"].items = []
    goal = GameState(rooms=gw, player_room="entrance", inventory=["lamp"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_take_lamp",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=1.0,
    ))

    # T3: Go east from hallway to armory
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="hallway", max_score=0)
    goal = GameState(rooms=_make_simple_world(), player_room="armory", max_score=0)
    tasks.append(Task(
        task_id="zork_go_east",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=1.0,
    ))

    # T4: Take sword from armory
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="armory", max_score=1)
    gw = _make_simple_world()
    gw["armory"].items = []
    goal = GameState(rooms=gw, player_room="armory", inventory=["sword"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_take_sword",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=1.0,
    ))

    # T5: Go north in branching world
    w = _make_branching_world()
    start = GameState(rooms=w, player_room="hub", max_score=0)
    goal = GameState(rooms=_make_branching_world(), player_room="cave", max_score=0)
    tasks.append(Task(
        task_id="zork_hub_to_cave",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=1.0,
    ))

    # ---- Level 2: Two-step sequences (difficulty 2.0) ----

    # T6: Take lamp, go north
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=1)
    gw = _make_simple_world()
    gw["entrance"].items = []
    goal = GameState(rooms=gw, player_room="hallway", inventory=["lamp"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_take_and_move",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=2.0,
    ))

    # T7: Go north twice (entrance → hallway → treasure_room)
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=0)
    goal = GameState(rooms=_make_simple_world(), player_room="treasure_room", max_score=0)
    tasks.append(Task(
        task_id="zork_navigate_only",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=2.0,
    ))

    # T8: Go north, take torch from cave
    w = _make_branching_world()
    start = GameState(rooms=w, player_room="hub", max_score=1)
    gw = _make_branching_world()
    gw["cave"].items = []
    goal = GameState(rooms=gw, player_room="cave", inventory=["torch"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_hub_take_torch",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=2.0,
    ))

    # T9: Go east, take flower from garden
    w = _make_branching_world()
    start = GameState(rooms=w, player_room="hub", max_score=1)
    gw = _make_branching_world()
    gw["garden"].items = []
    goal = GameState(rooms=gw, player_room="garden", inventory=["flower"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_hub_take_flower",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=2.0,
    ))

    # T10: Go west, take book from library
    w = _make_branching_world()
    start = GameState(rooms=w, player_room="hub", max_score=1)
    gw = _make_branching_world()
    gw["library"].items = []
    goal = GameState(rooms=gw, player_room="library", inventory=["book"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_hub_take_book",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=2.0,
    ))

    # ---- Level 3: Three-step sequences (difficulty 3.0) ----

    # T11: Navigate to treasure room and take treasure (N, N, take)
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=1)
    gw = _make_simple_world()
    gw["treasure_room"].items = []
    goal = GameState(rooms=gw, player_room="treasure_room",
                     inventory=["treasure"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_navigate_take",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=3.0,
    ))

    # T12: Take lamp, go north, go east (to armory with lamp)
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=1)
    gw = _make_simple_world()
    gw["entrance"].items = []
    goal = GameState(rooms=gw, player_room="armory", inventory=["lamp"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_lamp_to_armory",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=3.0,
    ))

    # T13: Go north, take sword, go back south (armory round trip)
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="hallway", max_score=1)
    gw = _make_simple_world()
    gw["armory"].items = []
    goal = GameState(rooms=gw, player_room="hallway", inventory=["sword"], score=1, max_score=1)
    # Note: go east, take sword, go west
    tasks.append(Task(
        task_id="zork_sword_roundtrip",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=3.0,
    ))

    # T14: Linear world - go north 3 times (room_0 → room_3)
    w = _make_linear_world(5)
    start = GameState(rooms=w, player_room="room_0", max_score=0)
    goal = GameState(rooms=_make_linear_world(5), player_room="room_3", max_score=0)
    tasks.append(Task(
        task_id="zork_linear_3",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=3.0,
    ))

    # T15: Get key, unlock door (key_room → start → through locked door)
    w = _make_locked_world()
    start = GameState(rooms=w, player_room="start", max_score=1)
    gw = _make_locked_world()
    gw["key_room"].items = []
    del gw["start"].locked_exits["north"]
    goal = GameState(rooms=gw, player_room="locked_passage",
                     inventory=["key"], score=1, max_score=1,
                     flags={"unlocked_north_start"})
    tasks.append(Task(
        task_id="zork_get_key_unlock",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=3.0,
    ))

    # ---- Level 4: Multi-step with complex goals (difficulty 4.0+) ----

    # T16: Unlock door, reach goal, take gem (4+ steps)
    w = _make_locked_world()
    start = GameState(rooms=w, player_room="start", max_score=2)
    gw = _make_locked_world()
    gw["key_room"].items = []
    gw["goal"].items = []
    del gw["start"].locked_exits["north"]
    goal = GameState(rooms=gw, player_room="goal",
                     inventory=["key", "gem"], score=2, max_score=2,
                     flags={"unlocked_north_start"})
    tasks.append(Task(
        task_id="zork_locked_door",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=4.0,
    ))

    # T17: Linear world - go north 4 times to reach prize
    w = _make_linear_world(5)
    start = GameState(rooms=w, player_room="room_0", max_score=1)
    gw = _make_linear_world(5)
    gw["room_4"].items = []
    goal = GameState(rooms=gw, player_room="room_4",
                     inventory=["prize"], score=1, max_score=1)
    tasks.append(Task(
        task_id="zork_linear_take_prize",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=4.0,
    ))

    # T18: Multi-key world - get silver_key, unlock first door
    w = _make_multi_key_world()
    start = GameState(rooms=w, player_room="foyer", max_score=1)
    gw = _make_multi_key_world()
    gw["closet"].items = []
    del gw["hall"].locked_exits["north"]
    goal = GameState(rooms=gw, player_room="locked_north",
                     inventory=["silver_key"], score=1, max_score=1,
                     flags={"unlocked_north_hall"})
    tasks.append(Task(
        task_id="zork_silver_key",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=4.0,
    ))

    # T19: Multi-key world - get both keys, reach vault (6+ steps)
    w = _make_multi_key_world()
    start = GameState(rooms=w, player_room="foyer", max_score=3)
    gw = _make_multi_key_world()
    gw["closet"].items = []
    gw["study"].items = []
    gw["vault"].items = []
    del gw["hall"].locked_exits["north"]
    del gw["locked_north"].locked_exits["north"]
    goal = GameState(rooms=gw, player_room="vault",
                     inventory=["silver_key", "gold_key", "diamond"],
                     score=3, max_score=3,
                     flags={"unlocked_north_hall", "unlocked_north_locked_north"})
    tasks.append(Task(
        task_id="zork_vault_heist",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=5.0,
    ))

    # T20: Take lamp, navigate to treasure room, take treasure, go back
    w = _make_simple_world()
    start = GameState(rooms=w, player_room="entrance", max_score=2)
    gw = _make_simple_world()
    gw["entrance"].items = []
    gw["treasure_room"].items = []
    goal = GameState(rooms=gw, player_room="entrance",
                     inventory=["lamp", "treasure"], score=2, max_score=2)
    tasks.append(Task(
        task_id="zork_roundtrip_loot",
        train_examples=[(start, goal)],
        test_inputs=[start],
        test_outputs=[goal],
        difficulty=5.0,
    ))

    return tasks
