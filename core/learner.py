"""
The Universal Learning Loop.

This is the invariant core. It NEVER imports anything domain-specific.
It depends only on the 4 interfaces defined in interfaces.py.

The loop:
    WAKE:   observe → hypothesize → execute → score → store
    SLEEP:  analyze solutions → extract sub-programs → compress → add to library
    REPEAT: library grows → search space shrinks → harder problems become tractable
"""

from __future__ import annotations
import logging
import math
import multiprocessing
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from .interfaces import (
    Environment,
    Grammar,
    DriveSignal,
    Memory,
    Program,
    Task,
    ScoredProgram,
    LibraryEntry,
    Primitive,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Transition Matrix — DreamCoder-style generative prior
# =============================================================================

class TransitionMatrix:
    """
    Learns P(child_op | parent_op) from successful programs.

    This biases random program generation toward compositions
    that have been observed in solutions. The key DreamCoder insight:
    not all compositions are equally likely to be useful.
    """

    def __init__(self, smoothing: float = 0.1):
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._totals: dict[str, int] = defaultdict(int)
        self._smoothing = smoothing

    def observe_program(self, program: Program) -> None:
        """Record parent->child transitions from a program tree."""
        for child in program.children:
            self._counts[program.root][child.root] += 1
            self._totals[program.root] += 1
            self.observe_program(child)

    def probability(self, parent_op: str, child_op: str, n_primitives: int) -> float:
        """P(child_op | parent_op) with Laplace smoothing."""
        count = self._counts[parent_op][child_op]
        total = self._totals[parent_op]
        smooth = self._smoothing
        return (count + smooth) / (total + smooth * n_primitives)

    def weighted_choice(self, parent_op: str, primitives: list[Primitive],
                        rng: random.Random) -> Primitive:
        """Choose a child primitive biased by the transition matrix."""
        n = len(primitives)
        if not self._totals.get(parent_op):
            return rng.choice(primitives)

        weights = [
            self.probability(parent_op, p.name, n) for p in primitives
        ]
        total_w = sum(weights)
        if total_w <= 0:
            return rng.choice(primitives)

        r = rng.random() * total_w
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return primitives[i]
        return primitives[-1]

    @property
    def size(self) -> int:
        """Number of observed transitions."""
        return sum(self._totals.values())

    def __repr__(self):
        return f"TransitionMatrix({self.size} transitions, {len(self._counts)} parents)"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SearchConfig:
    """Knobs for the beam search. Domain-agnostic."""
    beam_width: int = 200         # candidates kept per generation
    mutations_per_candidate: int = 3
    crossover_fraction: float = 0.3
    max_generations: int = 100    # compute budget = beam_width × max_generations
    energy_alpha: float = 1.0     # weight on prediction error
    energy_beta: float = 0.001    # weight on complexity cost
    early_stop_energy: float = 0.0  # stop if energy <= this (perfect solve)
    solve_threshold: float = 1e-4   # prediction_error <= this counts as solved
    seed: Optional[int] = None


@dataclass
class SleepConfig:
    """Knobs for the sleep/consolidation phase."""
    min_occurrences: int = 2      # sub-tree must appear in >= N solutions
    min_size: int = 2             # sub-tree must have >= N nodes
    max_library_size: int = 500   # cap on total library entries
    usefulness_decay: float = 0.95  # decay old entries each sleep cycle


@dataclass
class CurriculumConfig:
    """Knobs for curriculum-ordered learning."""
    sort_by_difficulty: bool = True
    wake_sleep_rounds: int = 3
    workers: int = 0  # 0 = auto-detect (performance cores), 1 = sequential


@dataclass
class WakeResult:
    """What comes out of one wake phase."""
    task_id: str
    solved: bool
    best: Optional[ScoredProgram]
    generations_used: int
    evaluations: int = 0    # total program evaluations (deterministic compute measure)
    wall_time: float = 0.0


@dataclass
class SleepResult:
    """What comes out of one sleep phase."""
    new_entries: list[LibraryEntry]
    library_size_before: int
    library_size_after: int
    wall_time: float


@dataclass
class RoundResult:
    """What comes out of one full wake-sleep round."""
    round_number: int
    wake_results: list[WakeResult]
    sleep_result: SleepResult
    tasks_solved: int
    tasks_total: int
    solve_rate: float
    cumulative_library_size: int


# =============================================================================
# Module-level worker for multiprocessing (must be picklable)
# =============================================================================

def _worker_init():
    """Initializer for worker processes: ignore SIGINT.

    The parent process handles Ctrl-C and terminates the pool.
    Without this, workers trap SIGINT and the pool hangs.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


def _wake_worker(args: tuple) -> "WakeResult":
    """
    Solve a single task in a child process.

    Receives a snapshot of the learner state, reconstructs a mini-Learner,
    and returns the WakeResult. This function is at module level so it can
    be pickled by ProcessPoolExecutor.
    """
    task, env, grammar, drive, library, search_cfg, transition_matrix = args

    # Reconstruct a lightweight learner with the snapshot
    from .memory import InMemoryStore
    memory = InMemoryStore()
    for entry in library:
        memory.add_to_library(entry)

    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=search_cfg,
    )
    learner._transition_matrix = transition_matrix

    # Solve — but skip memory recording (main process handles that)
    return learner._wake_on_task_no_record(task)


# =============================================================================
# The Learner — the invariant core
# =============================================================================

class Learner:
    """
    The Universal Learner.

    Takes 4 pluggable interfaces. The main loop never changes.
    Everything domain-specific lives in the plugins.
    """

    def __init__(
        self,
        environment: Environment,
        grammar: Grammar,
        drive: DriveSignal,
        memory: Memory,
        search_config: SearchConfig | None = None,
        sleep_config: SleepConfig | None = None,
    ):
        self.env = environment
        self.grammar = grammar
        self.drive = drive
        self.memory = memory
        self.search_cfg = search_config or SearchConfig()
        self.sleep_cfg = sleep_config or SleepConfig()

        self._rng = random.Random(self.search_cfg.seed)
        self._transition_matrix = TransitionMatrix()

    # -------------------------------------------------------------------------
    # WAKE PHASE: observe → hypothesize → execute → score → store
    # -------------------------------------------------------------------------

    def wake_on_task(self, task: Task) -> WakeResult:
        """
        Attempt to solve a single task via beam search.

        1. Get base primitives + library primitives
        2. Generate initial random candidates
        3. For each generation:
           a. Evaluate all candidates (execute + score)
           b. Keep the beam_width best
           c. Mutate and crossover to produce next generation
        4. Store the best solution if it's good enough
        """
        t0 = time.time()
        cfg = self.search_cfg

        # Combine hand-coded primitives with learned library entries
        base_prims = self.grammar.base_primitives()
        library_prims = self.grammar.inject_library(self.memory.get_library())
        all_prims = base_prims + library_prims

        # Initialize beam with small random programs
        beam = self._init_beam(all_prims, cfg.beam_width)

        best_so_far: Optional[ScoredProgram] = None
        gens_used = 0
        n_evals = 0

        for gen in range(cfg.max_generations):
            gens_used = gen + 1

            # Evaluate every candidate on all training examples
            scored = []
            for prog in beam:
                sp = self._evaluate_program(prog, task)
                n_evals += 1
                scored.append(sp)

                # Track global best
                if best_so_far is None or sp.energy < best_so_far.energy:
                    best_so_far = sp

            # Early stopping on perfect solve
            if best_so_far and best_so_far.energy <= cfg.early_stop_energy:
                logger.info(f"  [wake] Task {task.task_id}: perfect solve at gen {gen}")
                break

            # Keep the best
            scored.sort(key=lambda s: s.energy)
            survivors = [s.program for s in scored[: cfg.beam_width]]

            # Produce next generation
            next_gen = list(survivors)  # elitism: survivors carry over

            # Mutations
            for prog in survivors:
                for _ in range(cfg.mutations_per_candidate):
                    mutant = self.grammar.mutate(prog, all_prims)
                    next_gen.append(mutant)

            # Crossovers
            n_cross = int(len(survivors) * cfg.crossover_fraction)
            for _ in range(n_cross):
                a = self._rng.choice(survivors)
                b = self._rng.choice(survivors)
                child = self.grammar.crossover(a, b)
                next_gen.append(child)

            beam = next_gen

        # Record the episode and store solution if solved
        solved = best_so_far is not None and best_so_far.prediction_error <= self.search_cfg.solve_threshold
        if best_so_far:
            best_so_far.task_id = task.task_id
            self.memory.record_episode(
                task.task_id,
                task.train_examples,
                best_so_far.program,
                best_so_far.energy,
            )
            if solved:
                self.memory.store_solution(task.task_id, best_so_far)
                # Credit library entries that were used
                self._credit_library_usage(best_so_far.program)

        wall = time.time() - t0
        logger.info(
            f"  [wake] Task {task.task_id}: solved={solved}, "
            f"energy={best_so_far.energy:.6f}, gens={gens_used}, "
            f"evals={n_evals}, time={wall:.1f}s"
        )
        return WakeResult(
            task_id=task.task_id,
            solved=solved,
            best=best_so_far,
            generations_used=gens_used,
            evaluations=n_evals,
            wall_time=wall,
        )

    def _wake_on_task_no_record(self, task: Task) -> WakeResult:
        """
        Same as wake_on_task but does NOT write to memory.

        Used by parallel workers — the main process merges results.
        """
        t0 = time.time()
        cfg = self.search_cfg

        base_prims = self.grammar.base_primitives()
        library_prims = self.grammar.inject_library(self.memory.get_library())
        all_prims = base_prims + library_prims

        beam = self._init_beam(all_prims, cfg.beam_width)
        best_so_far: Optional[ScoredProgram] = None
        gens_used = 0
        n_evals = 0

        for gen in range(cfg.max_generations):
            gens_used = gen + 1

            scored = []
            for prog in beam:
                sp = self._evaluate_program(prog, task)
                n_evals += 1
                scored.append(sp)
                if best_so_far is None or sp.energy < best_so_far.energy:
                    best_so_far = sp

            if best_so_far and best_so_far.energy <= cfg.early_stop_energy:
                break

            scored.sort(key=lambda s: s.energy)
            survivors = [s.program for s in scored[: cfg.beam_width]]

            next_gen = list(survivors)
            for prog in survivors:
                for _ in range(cfg.mutations_per_candidate):
                    next_gen.append(self.grammar.mutate(prog, all_prims))

            n_cross = int(len(survivors) * cfg.crossover_fraction)
            for _ in range(n_cross):
                a = self._rng.choice(survivors)
                b = self._rng.choice(survivors)
                next_gen.append(self.grammar.crossover(a, b))

            beam = next_gen

        solved = (best_so_far is not None and
                  best_so_far.prediction_error <= self.search_cfg.solve_threshold)
        if best_so_far:
            best_so_far.task_id = task.task_id

        wall = time.time() - t0
        logger.info(
            f"  [wake] Task {task.task_id}: solved={solved}, "
            f"energy={best_so_far.energy:.6f}, gens={gens_used}, "
            f"evals={n_evals}, time={wall:.1f}s"
        )
        return WakeResult(
            task_id=task.task_id,
            solved=solved,
            best=best_so_far,
            generations_used=gens_used,
            evaluations=n_evals,
            wall_time=wall,
        )

    # -------------------------------------------------------------------------
    # SLEEP PHASE: analyze → extract → compress → add to library
    # -------------------------------------------------------------------------

    def sleep(self) -> SleepResult:
        """
        Consolidation phase — the "dream" step.

        1. Collect all solved programs
        2. Build transition matrix P(child_op | parent_op) from solutions
        3. Extract sub-trees that recur across multiple solutions
        4. Score by compression value: tasks_used × log(size)
        5. Add the best as new named primitives to the library
        """
        t0 = time.time()
        cfg = self.sleep_cfg

        solutions = self.memory.get_solutions()
        lib_before = len(self.memory.get_library())

        # 1. Build transition matrix from ALL solved programs
        #    This is the DreamCoder insight: learn which compositions work
        for task_id, scored in solutions.items():
            self._transition_matrix.observe_program(scored.program)

        logger.info(
            f"  [sleep] Transition matrix: {self._transition_matrix.size} transitions "
            f"from {len(solutions)} solved programs"
        )

        # 2. Extract all sub-trees from all solutions
        subtree_counts: dict[str, list[tuple[Program, str]]] = {}
        for task_id, scored in solutions.items():
            for subtree in self._enumerate_subtrees(scored.program):
                key = repr(subtree)
                if key not in subtree_counts:
                    subtree_counts[key] = []
                subtree_counts[key].append((subtree, task_id))

        # 3. Filter: must appear in >= min_occurrences different tasks,
        #    must be >= min_size nodes (no trivial single-node entries)
        candidates = []
        for key, occurrences in subtree_counts.items():
            task_ids = list(set(tid for _, tid in occurrences))
            subtree = occurrences[0][0]
            if len(task_ids) >= cfg.min_occurrences and subtree.size >= cfg.min_size:
                # Usefulness = tasks_used_in × log(size+1)
                # Log scaling prevents huge subtrees from dominating
                usefulness = len(task_ids) * math.log(subtree.size + 1)
                candidates.append((subtree, task_ids, usefulness))

        # 4. Sort by usefulness, add top entries to library
        candidates.sort(key=lambda c: c[2], reverse=True)

        existing_reprs = {repr(e.program) for e in self.memory.get_library()}
        new_entries = []
        for subtree, task_ids, usefulness in candidates:
            if repr(subtree) in existing_reprs:
                continue
            if len(self.memory.get_library()) + len(new_entries) >= cfg.max_library_size:
                break

            entry_name = f"learned_{lib_before + len(new_entries)}"
            entry = LibraryEntry(
                name=entry_name,
                program=subtree,
                usefulness=usefulness,
                reuse_count=0,
                source_tasks=task_ids,
                domain="",  # domain assigned by caller if desired
            )
            new_entries.append(entry)
            existing_reprs.add(repr(subtree))

        # 5. Add to memory
        for entry in new_entries:
            self.memory.add_to_library(entry)

        # 6. Decay old entries
        for entry in self.memory.get_library():
            if entry not in new_entries:
                self.memory.update_usefulness(
                    entry.name,
                    entry.usefulness * (cfg.usefulness_decay - 1),  # negative delta
                )

        lib_after = len(self.memory.get_library())
        wall = time.time() - t0

        logger.info(
            f"  [sleep] Extracted {len(new_entries)} new abstractions. "
            f"Library: {lib_before} → {lib_after}. Time: {wall:.1f}s"
        )
        return SleepResult(
            new_entries=new_entries,
            library_size_before=lib_before,
            library_size_after=lib_after,
            wall_time=wall,
        )

    # -------------------------------------------------------------------------
    # FULL CURRICULUM: wake-sleep rounds over ordered tasks
    # -------------------------------------------------------------------------

    @staticmethod
    def performance_core_count() -> int:
        """
        Return the number of performance cores (not all logical cores).

        Uses max(1, total_cores - 2) to leave headroom for the OS and UI,
        preventing the machine from becoming unresponsive during long runs.
        """
        total = os.cpu_count() or 1
        if total <= 2:
            return 1
        return total - 2

    def run_curriculum(
        self,
        tasks: list[Task],
        config: CurriculumConfig | None = None,
        on_task_done: "Optional[callable]" = None,
    ) -> list[RoundResult]:
        """
        Run multiple wake-sleep rounds over a task set.

        This is the top-level entry point. The compounding property
        should be visible in the RoundResults: solve rate should
        increase across rounds as the library grows.

        Uses performance cores for the wake phase (tasks are independent).
        Set workers=1 in CurriculumConfig to disable parallelism.

        Args:
            on_task_done: Optional callback(round_num, task_index, total_tasks,
                          wake_result) called after each task completes.
                          Used for live progress streaming.
        """
        cfg = config or CurriculumConfig()

        if cfg.sort_by_difficulty:
            tasks = sorted(tasks, key=lambda t: t.difficulty)

        # Resolve worker count: 0 = performance cores (not all cores)
        if cfg.workers <= 0:
            cfg = CurriculumConfig(
                sort_by_difficulty=cfg.sort_by_difficulty,
                wake_sleep_rounds=cfg.wake_sleep_rounds,
                workers=self.performance_core_count(),
            )

        results = []
        for round_num in range(cfg.wake_sleep_rounds):
            logger.info(f"=== Round {round_num + 1}/{cfg.wake_sleep_rounds} ===")
            logger.info(f"    Library size: {len(self.memory.get_library())}")
            logger.info(f"    Workers: {cfg.workers}")

            # WAKE: attempt all tasks (parallel across CPU cores)
            wake_results = self._wake_parallel(
                tasks, cfg.workers, round_num + 1, on_task_done)

            # SLEEP: consolidate (sequential — mutates shared state)
            sleep_result = self.sleep()

            # Metrics
            solved = sum(1 for w in wake_results if w.solved)
            total = len(wake_results)
            rate = solved / total if total > 0 else 0.0

            rr = RoundResult(
                round_number=round_num + 1,
                wake_results=wake_results,
                sleep_result=sleep_result,
                tasks_solved=solved,
                tasks_total=total,
                solve_rate=rate,
                cumulative_library_size=len(self.memory.get_library()),
            )
            results.append(rr)

            logger.info(
                f"=== Round {round_num + 1} summary: "
                f"solved {solved}/{total} ({rate:.1%}), "
                f"library={rr.cumulative_library_size} ==="
            )

        return results

    def _wake_parallel(
        self,
        tasks: list[Task],
        workers: int,
        round_num: int = 1,
        on_task_done: "Optional[callable]" = None,
    ) -> list[WakeResult]:
        """
        Run wake_on_task across tasks using a process pool.

        Each worker gets a snapshot of the current state (library, config,
        transition matrix) and solves tasks independently. Results are merged
        back into the main process's memory.

        Falls back to sequential if workers=1 or if the task count is small.
        """
        total_tasks = len(tasks)

        if workers <= 1 or len(tasks) <= 2:
            # Sequential — simple path
            wake_results = []
            for i, task in enumerate(tasks):
                wr = self.wake_on_task(task)
                wake_results.append(wr)
                if on_task_done:
                    on_task_done(round_num, i + 1, total_tasks, wr)
            return wake_results

        # Snapshot state for workers (picklable data only)
        library_snapshot = self.memory.get_library()
        search_cfg = self.search_cfg
        sleep_cfg = self.sleep_cfg
        transition_matrix = self._transition_matrix

        # Build worker args: (task, env, grammar, drive, library, search_cfg, tm)
        # Each worker reconstructs a mini-Learner with the snapshot.
        worker_args = [
            (task, self.env, self.grammar, self.drive,
             library_snapshot, search_cfg, transition_matrix)
            for task in tasks
        ]

        wake_results: list[WakeResult] = [None] * len(tasks)  # type: ignore
        completed_count = 0
        try:
            # Use initializer to make workers ignore SIGINT — the parent
            # process handles it and terminates the pool. Without this,
            # Ctrl-C gets stuck waiting for in-flight workers to finish.
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
            ) as pool:
                futures = {
                    pool.submit(_wake_worker, args): i
                    for i, args in enumerate(worker_args)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    wr = future.result()
                    wake_results[idx] = wr
                    completed_count += 1
                    if on_task_done:
                        on_task_done(round_num, completed_count, total_tasks, wr)
        except (OSError, RuntimeError) as e:
            # Fallback to sequential if multiprocessing fails
            # (e.g., pickling issues, resource limits)
            logger.warning(f"Parallel wake failed ({e}), falling back to sequential")
            wake_results = []
            for i, task in enumerate(tasks):
                wr = self.wake_on_task(task)
                wake_results.append(wr)
                if on_task_done:
                    on_task_done(round_num, i + 1, total_tasks, wr)
            return wake_results

        # Merge results back into main memory
        for wr in wake_results:
            if wr.best:
                self.memory.record_episode(
                    wr.task_id, [], wr.best.program, wr.best.energy)
                if wr.solved:
                    self.memory.store_solution(wr.task_id, wr.best)
                    self._credit_library_usage(wr.best.program)

        return wake_results

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _init_beam(self, primitives: list[Primitive], n: int) -> list[Program]:
        """
        Generate n random programs of varying depth (1-3).

        Uses the transition matrix to bias toward known-good compositions
        when available. Falls back to uniform random when no prior exists.
        """
        beam = []
        use_prior = self._transition_matrix.size > 0

        for i in range(n):
            # Vary depth: 40% depth-1, 40% depth-2, 20% depth-3
            r = self._rng.random()
            max_depth = 1 if r < 0.4 else (2 if r < 0.8 else 3)
            prog = self._random_program(primitives, max_depth, use_prior)
            beam.append(prog)
        return beam

    def _random_program(self, primitives: list[Primitive], max_depth: int,
                        use_prior: bool, parent_op: str = "") -> Program:
        """Generate a random program tree up to max_depth."""
        if max_depth <= 1:
            # Leaf: pick a primitive (prefer arity-0 or arity-1 as leaf)
            leaf_prims = [p for p in primitives if p.arity <= 1]
            if not leaf_prims:
                leaf_prims = primitives
            if use_prior and parent_op:
                prim = self._transition_matrix.weighted_choice(
                    parent_op, leaf_prims, self._rng)
            else:
                prim = self._rng.choice(leaf_prims)
            return Program(root=prim.name)

        # Internal node: pick a primitive with arity > 0
        if use_prior and parent_op:
            prim = self._transition_matrix.weighted_choice(
                parent_op, primitives, self._rng)
        else:
            prim = self._rng.choice(primitives)

        if prim.arity == 0:
            return Program(root=prim.name)

        children = []
        for _ in range(prim.arity):
            child = self._random_program(
                primitives, max_depth - 1, use_prior, prim.name)
            children.append(child)
        return Program(root=prim.name, children=children)

    def _evaluate_program(self, program: Program, task: Task) -> ScoredProgram:
        """Evaluate a program on all training examples, return scored result."""
        total_error = 0.0
        n = len(task.train_examples)

        for inp, expected in task.train_examples:
            try:
                predicted = self.env.execute(program, inp)
                err = self.drive.prediction_error(predicted, expected)
            except Exception:
                err = 1e6  # penalty for programs that crash
            total_error += err

        avg_error = total_error / n if n > 0 else total_error
        energy, pred_err, comp_cost = self.drive.energy(
            program, None, None,
            alpha=self.search_cfg.energy_alpha,
            beta=self.search_cfg.energy_beta,
        )
        # Override with actual averaged error
        pred_err = avg_error
        comp_cost = self.drive.complexity_cost(program)
        energy = self.search_cfg.energy_alpha * pred_err + self.search_cfg.energy_beta * comp_cost

        return ScoredProgram(
            program=program,
            energy=energy,
            prediction_error=pred_err,
            complexity_cost=comp_cost,
        )

    def _enumerate_subtrees(self, program: Program) -> list[Program]:
        """Yield every sub-tree in a program (including the root)."""
        result = [program]
        for child in program.children:
            result.extend(self._enumerate_subtrees(child))
        return result

    def _credit_library_usage(self, program: Program) -> None:
        """If a solved program uses library entries, increment their reuse count."""
        library_names = {e.name for e in self.memory.get_library()}
        for subtree in self._enumerate_subtrees(program):
            if subtree.root in library_names:
                self.memory.update_usefulness(subtree.root, 1.0)
