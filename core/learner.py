"""
The Universal Learning Loop.

This is the invariant core. It NEVER imports anything domain-specific.
It depends only on the 4 interfaces defined in interfaces.py.

The loop:
    WAKE:   observe → hypothesize → execute → score → store
    SLEEP:  analyze solutions → extract sub-programs → compress → add to library
    REPEAT: library grows → search space shrinks → harder problems become tractable

STRIPPED TO CORE: Only exhaustive enumeration + basic sleep.
Beam search, structural strategies, near-miss refinement, etc.
will be added back when justified by specific tasks.
"""

from __future__ import annotations
import copy
import logging
import math
import os
import random
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .types import (
    Program,
    Task,
    ScoredProgram,
    LibraryEntry,
    Primitive,
)
from .interfaces import (
    Environment,
    Grammar,
    DriveSignal,
    Memory,
)
from .config import SearchConfig, SleepConfig, CurriculumConfig
from .results import ParetoEntry, WakeResult, SleepResult, RoundResult
from .transition_matrix import TransitionMatrix

logger = logging.getLogger(__name__)


# =============================================================================
# Wake context — shared mutable state for the wake phase pipeline
# =============================================================================

class _WakeContext:
    """Holds all mutable state shared across wake phases."""
    __slots__ = (
        "task", "all_prims", "cfg", "eval_budget", "record", "t0",
        "best_so_far", "n_evals", "total_deduped", "gens_used",
        "pareto", "enum_candidates", "beam_scored",
    )

    def __init__(self, task, all_prims, cfg, eval_budget, record):
        self.task: Task = task
        self.all_prims: list[Primitive] = all_prims
        self.cfg: SearchConfig = cfg
        self.eval_budget: int = eval_budget
        self.record: bool = record
        self.t0: float = time.time()

        self.best_so_far: Optional[ScoredProgram] = None
        self.n_evals: int = 0
        self.total_deduped: int = 0
        self.gens_used: int = 0
        self.pareto: dict[int, ParetoEntry] = {}
        self.enum_candidates: list[ScoredProgram] = []
        self.beam_scored: list[ScoredProgram] = []

    def budget_ok(self) -> bool:
        return self.eval_budget <= 0 or self.n_evals < self.eval_budget

    @property
    def solved(self) -> bool:
        return (self.best_so_far is not None
                and self.best_so_far.max_example_error <= self.cfg.solve_threshold)

    def update_best(self, sp: ScoredProgram) -> None:
        if self.best_so_far is None or sp.energy < self.best_so_far.energy:
            self.best_so_far = sp


# =============================================================================
# Module-level worker for multiprocessing (must be picklable)
# =============================================================================

def _worker_init():
    """Initializer for worker processes: ignore SIGINT."""
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


def _wake_worker(args: tuple) -> WakeResult:
    """Solve a single task in a child process."""
    task, env, grammar, drive, library, search_cfg, transition_matrix, task_seed = args

    from .memory import InMemoryStore
    memory = InMemoryStore()
    for entry in library:
        memory.add_to_library(entry)

    from dataclasses import replace as _dc_replace
    worker_cfg = _dc_replace(search_cfg, seed=task_seed)

    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=worker_cfg,
    )
    learner._transition_matrix = transition_matrix
    grammar._rng = random.Random(task_seed)

    result = learner._wake_on_task_no_record(task)
    return result


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
    # WAKE PHASE
    # -------------------------------------------------------------------------

    def wake_on_task(self, task: Task) -> WakeResult:
        """Attempt to solve a single task via exhaustive enumeration."""
        return self._wake_core(task, record=True)

    def _wake_on_task_no_record(self, task: Task) -> WakeResult:
        """Same as wake_on_task but does NOT write to memory."""
        return self._wake_core(task, record=False)

    def _wake_core(self, task: Task, record: bool) -> WakeResult:
        """Shared wake logic. Runs pipeline of search phases."""
        cfg = self.search_cfg
        self.grammar.prepare_for_task(task)

        base_prims = self.grammar.base_primitives()
        library_prims = self.grammar.inject_library(self.memory.get_library())
        all_prims = base_prims + library_prims

        for lp in library_prims:
            self.env.register_primitive(lp)

        base_cells = cfg.eval_budget_base_cells
        cells = self._avg_cells(task)
        if cfg.eval_budget > 0:
            eval_budget = max(cfg.eval_budget * base_cells // max(cells, 1), 500)
            eval_budget = min(eval_budget, cfg.eval_budget * 4)
        else:
            eval_budget = 0

        ctx = _WakeContext(task, all_prims, cfg, eval_budget, record)

        for phase_fn in self._wake_phases():
            solved_by = phase_fn(ctx)
            if solved_by is not None:
                return self._make_solved_result(ctx, solved_by)

        return self._make_unsolved_result(ctx)

    def _wake_phases(self):
        """Return the ordered list of wake phase methods.

        Each phase takes a _WakeContext and returns a phase name string
        if the task was solved, or None to continue to the next phase.
        """
        return [
            self._phase_exhaustive,
            self._phase_per_row_column,
            self._phase_object_decomposition,
            self._phase_for_each_object,
            self._phase_conditional_per_object,
            self._phase_cross_reference,
            self._phase_near_miss_refinement,
            self._phase_color_fix,
            self._phase_beam_search,
            self._phase_post_beam_color_fix,
        ]

    def _phase_exhaustive(self, ctx: _WakeContext) -> Optional[str]:
        """Phase 1: Exhaustive enumeration of all programs up to depth 3."""
        if ctx.cfg.exhaustive_depth < 1:
            return None
        t = time.time()
        candidates, n_evals = self._exhaustive_enumerate(
            ctx.all_prims, ctx.task, ctx.cfg.exhaustive_depth,
            eval_budget=ctx.eval_budget)
        ctx.n_evals += n_evals
        for sp in candidates:
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
        ctx.enum_candidates.extend(candidates)
        logger.debug(f"  [wake] Phase 1 enumeration: {time.time()-t:.2f}s, {ctx.n_evals} evals")
        return "enumeration" if ctx.solved else None

    def _phase_per_row_column(self, ctx: _WakeContext) -> Optional[str]:
        """Try per-row/per-column decomposition."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        if not hasattr(self.env, 'try_per_row_column_decomposition'):
            return None
        t = time.time()
        result = self.env.try_per_row_column_decomposition(
            ctx.task, ctx.all_prims) if ctx.budget_ok() else None
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Per-row/col: {time.time()-t:.2f}s")
        return "per-row/column" if ctx.solved else None

    def _phase_object_decomposition(self, ctx: _WakeContext) -> Optional[str]:
        """Try per-object transforms via connected components."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        t = time.time()
        result = self.env.try_object_decomposition(ctx.task, ctx.all_prims) if ctx.budget_ok() else None
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Object decomp: {time.time()-t:.2f}s")
        return "object decomposition" if ctx.solved else None

    def _phase_for_each_object(self, ctx: _WakeContext) -> Optional[str]:
        """Apply top-K enumeration candidates per-object."""
        if ctx.solved or not ctx.enum_candidates or not ctx.budget_ok():
            return None
        if not self.grammar.allow_structural_phases():
            return None
        t = time.time()
        result = self.env.try_for_each_object(ctx.task, ctx.enum_candidates, top_k=10)
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] For-each-object: {time.time()-t:.2f}s")
        return "for-each-object" if ctx.solved else None

    def _phase_conditional_per_object(self, ctx: _WakeContext) -> Optional[str]:
        """Try if(pred, A, B) per-object."""
        if ctx.solved or not ctx.enum_candidates or not ctx.budget_ok():
            return None
        if not self.grammar.allow_structural_phases():
            return None
        predicates = self.grammar.get_predicates()
        if not predicates:
            return None
        t = time.time()
        result = self.env.try_conditional_per_object(
            ctx.task, ctx.enum_candidates, predicates, top_k=8)
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Cond-per-object: {time.time()-t:.2f}s")
        return "conditional per-object" if ctx.solved else None

    def _phase_cross_reference(self, ctx: _WakeContext) -> Optional[str]:
        """Cross-reference: one grid part informs another."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        t = time.time()
        result = self.env.try_cross_reference(ctx.task, ctx.all_prims)
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Cross-reference: {time.time()-t:.2f}s")
        return "cross-reference" if ctx.solved else None

    def _phase_conditional_search(self, ctx: _WakeContext) -> Optional[str]:
        """Try if(predicate, A, B) programs."""
        if not self.grammar.allow_structural_phases():
            return None
        predicates = self.grammar.get_predicates()
        if not predicates or not ctx.enum_candidates or not ctx.budget_ok():
            return None
        if ctx.solved:
            return None
        t = time.time()
        result, n_evals = self._try_conditional_search(
            predicates, ctx.enum_candidates, ctx.all_prims, ctx.task)
        ctx.n_evals += n_evals
        if result is not None:
            self._update_pareto_front(ctx.pareto, result)
            ctx.update_best(result)
            ctx.enum_candidates.append(result)
        logger.debug(f"  [wake] Conditional search: {time.time()-t:.2f}s")
        return "conditional" if ctx.solved else None

    def _phase_near_miss_refinement(self, ctx: _WakeContext) -> Optional[str]:
        """Append/prepend primitives to near-miss programs."""
        if ctx.solved or not ctx.enum_candidates or not ctx.budget_ok():
            return None
        if ctx.cfg.near_miss_threshold <= 0:
            return None
        t = time.time()
        n_examples = max(len(ctx.task.train_examples), 1)
        threshold = min(ctx.cfg.near_miss_threshold, 2.0 / n_examples)
        refined, n_evals = self._near_miss_refine(
            ctx.enum_candidates, ctx.all_prims, ctx.task, threshold)
        ctx.n_evals += n_evals
        for sp in refined:
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
        ctx.enum_candidates.extend(refined)
        logger.debug(f"  [wake] Near-miss refine: {time.time()-t:.2f}s, {n_evals} evals")
        return "near-miss refinement" if ctx.solved else None

    def _phase_color_fix(self, ctx: _WakeContext) -> Optional[str]:
        """Learn color remapping from near-miss programs."""
        if ctx.solved or not ctx.enum_candidates:
            return None
        t = time.time()
        result = self._try_color_fix(ctx.enum_candidates, ctx.task)
        if result is not None:
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, result)
            ctx.update_best(result)
        logger.debug(f"  [wake] Phase 2 color fix: {time.time()-t:.2f}s")
        return "color fix" if ctx.solved else None

    def _phase_beam_search(self, ctx: _WakeContext) -> Optional[str]:
        """Beam search with mutation/crossover."""
        if ctx.solved:
            return None
        t = time.time()
        cfg = ctx.cfg
        best_enum_error = ctx.best_so_far.prediction_error if ctx.best_so_far else 1.0
        if not ctx.budget_ok():
            logger.debug(f"  [wake] Beam search: SKIPPED (budget exceeded)")
            return None

        if cfg.max_generations <= 1:
            effective_gens = cfg.max_generations
        elif best_enum_error > 0.3:
            effective_gens = max(5, cfg.max_generations // 4)
        elif best_enum_error > 0.15:
            effective_gens = max(10, cfg.max_generations // 2)
        else:
            effective_gens = cfg.max_generations

        seed_progs = [sp.program for sp in sorted(
            ctx.enum_candidates, key=lambda s: s.energy)[:cfg.beam_width // 2]]
        n_random = max(cfg.beam_width - len(seed_progs), cfg.beam_width // 2)
        beam = seed_progs + self._init_beam(ctx.all_prims, n_random)

        for gen in range(effective_gens):
            ctx.gens_used = gen + 1
            if not ctx.budget_ok():
                break

            scored = []
            for prog in beam:
                sp = self._evaluate_program(prog, ctx.task)
                ctx.n_evals += 1
                scored.append(sp)
                self._update_pareto_front(ctx.pareto, sp)
                ctx.update_best(sp)

            if ctx.best_so_far and ctx.best_so_far.energy <= cfg.early_stop_energy:
                logger.info(f"  [wake] Task {ctx.task.task_id}: perfect solve at gen {gen}")
                break

            if cfg.semantic_dedup:
                scored, n_removed = self._semantic_dedup(scored, ctx.task)
                ctx.total_deduped += n_removed

            scored.sort(key=lambda s: s.energy)
            survivors = [s.program for s in scored[:cfg.beam_width]]
            next_gen = list(survivors)

            tm = self._transition_matrix if self._transition_matrix.size > 0 else None
            for prog in survivors:
                for _ in range(cfg.mutations_per_candidate):
                    next_gen.append(self.grammar.mutate(prog, ctx.all_prims, transition_matrix=tm))

            n_cross = int(len(survivors) * cfg.crossover_fraction)
            for _ in range(n_cross):
                a = self._rng.choice(survivors)
                b = self._rng.choice(survivors)
                next_gen.append(self.grammar.crossover(a, b))

            beam = next_gen
            ctx.beam_scored = scored

        logger.debug(f"  [wake] Beam search: {time.time()-t:.2f}s, gens={ctx.gens_used}")
        return "beam search" if ctx.solved else None

    def _phase_post_beam_color_fix(self, ctx: _WakeContext) -> Optional[str]:
        """Color remapping on beam + enumeration results."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        if not ctx.best_so_far or ctx.best_so_far.prediction_error <= ctx.cfg.solve_threshold:
            return None
        t = time.time()
        all_candidates = list(ctx.enum_candidates)
        if ctx.beam_scored:
            all_candidates.extend(ctx.beam_scored)
        result = self._try_color_fix(all_candidates, ctx.task)
        if result is not None:
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, result)
            ctx.update_best(result)
        logger.debug(f"  [wake] Post-beam color fix: {time.time()-t:.2f}s")
        return "post-beam color fix" if ctx.solved else None

    # -------------------------------------------------------------------------
    # Wake result builders
    # -------------------------------------------------------------------------

    def _make_solved_result(self, ctx: _WakeContext, phase_name: str) -> WakeResult:
        """Build WakeResult for a training-solved task."""
        top_sp, te, ts, n_perf, s_rank, tss = self._evaluate_top_k_on_test(
            ctx.enum_candidates, ctx.task, top_k=10)
        if top_sp is not None and top_sp is not ctx.best_so_far:
            ctx.best_so_far = top_sp
        if ts is None and ctx.best_so_far is not None:
            te, ts, tss = self._evaluate_on_test(ctx.best_so_far, ctx.task)
        ctx.best_so_far = self._try_simplify(ctx.best_so_far, ctx.task)
        ctx.best_so_far.task_id = ctx.task.task_id
        self._record_solve(ctx)
        front = self._extract_pareto_front(ctx.pareto)
        wall = time.time() - ctx.t0
        logger.info(
            f"  [wake] Task {ctx.task.task_id}: SOLVED by {phase_name}, "
            f"energy={ctx.best_so_far.energy:.6f}, evals={ctx.n_evals}, "
            f"candidates={n_perf}, time={wall:.1f}s")
        train_preds, test_preds = self._compute_predictions(ctx.best_so_far, ctx.task)
        return WakeResult(
            task_id=ctx.task.task_id, train_solved=True, best=ctx.best_so_far,
            generations_used=ctx.gens_used, evaluations=ctx.n_evals, wall_time=wall,
            pareto_front=front, dedup_count=ctx.total_deduped,
            test_error=te, test_solved=ts, test_solve_score=tss,
            n_train_perfect=n_perf, solving_rank=s_rank,
            train_predictions=train_preds, test_predictions=test_preds)

    def _make_unsolved_result(self, ctx: _WakeContext) -> WakeResult:
        """Build WakeResult for an unsolved task."""
        if ctx.best_so_far:
            ctx.best_so_far = self._try_simplify(ctx.best_so_far, ctx.task)
            ctx.best_so_far.task_id = ctx.task.task_id
            if ctx.record:
                self.memory.record_episode(
                    ctx.task.task_id, ctx.task.train_examples,
                    ctx.best_so_far.program, ctx.best_so_far.energy)
                self.memory.store_best_attempt(ctx.task.task_id, ctx.best_so_far)
        front = self._extract_pareto_front(ctx.pareto)
        wall = time.time() - ctx.t0
        train_preds, test_preds = self._compute_predictions(ctx.best_so_far, ctx.task)
        logger.info(
            f"  [wake] Task {ctx.task.task_id}: train_solved=False, "
            f"energy={(ctx.best_so_far.energy if ctx.best_so_far else 0):.6f}, "
            f"evals={ctx.n_evals}, time={wall:.1f}s")
        return WakeResult(
            task_id=ctx.task.task_id, train_solved=False,
            best=ctx.best_so_far,
            generations_used=ctx.gens_used, evaluations=ctx.n_evals,
            wall_time=wall, pareto_front=front,
            dedup_count=ctx.total_deduped,
            train_predictions=train_preds, test_predictions=test_preds)

    def _try_simplify(self, sp: ScoredProgram, task: Task) -> ScoredProgram:
        """Simplify program by removing identity steps; re-score if changed."""
        simplified = self._simplify_program(sp.program, task)
        if simplified is sp.program:
            return sp
        return self._evaluate_program(simplified, task)

    def _record_solve(self, ctx: _WakeContext) -> None:
        """Record solution in memory if record=True."""
        if ctx.record and ctx.best_so_far:
            self.memory.record_episode(
                ctx.task.task_id, ctx.task.train_examples,
                ctx.best_so_far.program, ctx.best_so_far.energy)
            self.memory.store_solution(ctx.task.task_id, ctx.best_so_far)
            self._credit_library_usage(ctx.best_so_far.program)

    def _compute_predictions(
        self, best: Optional[ScoredProgram], task: Task
    ) -> tuple[Optional[list], Optional[list]]:
        """Compute predicted outputs for train and test inputs."""
        if best is None:
            return None, None
        train_preds = []
        for inp, _ in task.train_examples:
            try:
                pred = self.env.execute(best.program, inp)
                train_preds.append(pred)
            except Exception:
                train_preds.append(inp)
        test_preds = []
        if task.test_inputs:
            for inp in task.test_inputs:
                try:
                    pred = self.env.execute(best.program, inp)
                    test_preds.append(pred)
                except Exception:
                    test_preds.append(inp)
        return train_preds, test_preds or None

    def _evaluate_on_test(
        self, best: Optional[ScoredProgram], task: Task
    ) -> tuple[Optional[float], Optional[bool], Optional[float]]:
        """Evaluate the best program on held-out test examples."""
        if best is None or not task.test_inputs or not task.test_outputs:
            return None, None, None
        if len(task.test_inputs) != len(task.test_outputs):
            return None, None, None

        total_error = 0.0
        max_test_error = 0.0
        n = len(task.test_inputs)
        n_solved = 0
        threshold = self.search_cfg.solve_threshold
        for inp, expected in zip(task.test_inputs, task.test_outputs):
            try:
                predicted = self.env.execute(best.program, inp)
                err = self.drive.prediction_error(predicted, expected)
            except Exception:
                err = 1e6
            total_error += err
            max_test_error = max(max_test_error, err)
            if err <= threshold:
                n_solved += 1

        avg_error = total_error / n if n > 0 else total_error
        test_solved = max_test_error <= self.search_cfg.solve_threshold
        exponent = self.sleep_cfg.example_solve_exponent
        test_solve_score = (n_solved / n) ** exponent if n > 0 else 0.0
        return avg_error, test_solved, test_solve_score

    def _evaluate_top_k_on_test(
        self, candidates: list[ScoredProgram], task: Task, top_k: int = 3
    ) -> tuple[Optional[ScoredProgram], Optional[float], Optional[bool], int, Optional[int], Optional[float]]:
        """Try top-k training-perfect candidates on test, return best."""
        threshold = self.search_cfg.solve_threshold
        if not task.test_inputs or not task.test_outputs:
            return None, None, None, 0, None, None

        seen: set[str] = set()
        perfect: list[ScoredProgram] = []
        for sp in candidates:
            if sp.prediction_error <= threshold:
                key = repr(sp.program)
                if key not in seen:
                    seen.add(key)
                    perfect.append(sp)

        if not perfect:
            return None, None, None, 0, None, None

        n_train_perfect = len(perfect)
        perfect.sort(key=lambda sp: (sp.program.size, sp.energy))

        best_test_error = None
        best_test_sp = None
        best_test_tss = None

        for rank, sp in enumerate(perfect[:top_k]):
            test_error, test_solved, tss = self._evaluate_on_test(sp, task)
            if test_error is not None and (best_test_error is None or test_error < best_test_error):
                best_test_error = test_error
                best_test_sp = sp
                best_test_tss = tss
            if test_solved:
                return sp, test_error, True, n_train_perfect, rank, tss

        if best_test_sp is not None:
            return best_test_sp, best_test_error, False, n_train_perfect, None, best_test_tss

        return perfect[0], None, None, n_train_perfect, None, None

    # -------------------------------------------------------------------------
    # SLEEP PHASE
    # -------------------------------------------------------------------------

    def _unsolved_quality(self, scored: ScoredProgram, cfg: SleepConfig) -> float:
        """Quality weight for an unsolved program in sleep phase."""
        base_quality = math.exp(-scored.prediction_error) * cfg.unsolved_weight
        if scored.example_solve_score > 0:
            return max(base_quality, scored.example_solve_score * cfg.unsolved_weight)
        return base_quality

    def _credit_primitives(self, program: Program, quality: float) -> None:
        """Credit all primitives in a program tree with quality weight."""
        self.memory.update_primitive_score(program.root, quality)
        for child in program.children:
            self._credit_primitives(child, quality)

    def sleep(self) -> SleepResult:
        """Consolidation phase — the "dream" step."""
        t0 = time.time()
        cfg = self.sleep_cfg

        solutions = self.memory.get_solutions()
        unsolved = self.memory.get_best_attempts()
        lib_before = len(self.memory.get_library())

        # Build transition matrix
        for scored in solutions.values():
            self._transition_matrix.observe_program(scored.program)
        for scored in unsolved.values():
            self._transition_matrix.observe_program(scored.program)

        # Credit primitives
        for scored in solutions.values():
            self._credit_primitives(scored.program, 1.0)
        for scored in unsolved.values():
            quality = self._unsolved_quality(scored, cfg)
            self._credit_primitives(scored.program, quality)

        # Extract subtrees
        subtree_counts: dict[str, list[tuple[Program, str, float]]] = {}

        for task_id, scored in solutions.items():
            for subtree in self._enumerate_subtrees(scored.program):
                key = repr(subtree)
                if key not in subtree_counts:
                    subtree_counts[key] = []
                subtree_counts[key].append((subtree, task_id, 1.0))

        for task_id, scored in unsolved.items():
            quality = self._unsolved_quality(scored, cfg)
            for subtree in self._enumerate_subtrees(scored.program):
                key = repr(subtree)
                if key not in subtree_counts:
                    subtree_counts[key] = []
                subtree_counts[key].append((subtree, task_id, quality))

        # Filter and score
        candidates = []
        for key, occurrences in subtree_counts.items():
            task_ids = sorted(set(tid for _, tid, _ in occurrences))
            subtree = occurrences[0][0]
            if len(task_ids) >= cfg.min_occurrences and subtree.size >= cfg.min_size:
                total_quality = sum(w for _, _, w in occurrences)
                usefulness = total_quality * math.log(subtree.size + 1)
                candidates.append((subtree, task_ids, usefulness))

        candidates.sort(key=lambda c: c[2], reverse=True)

        existing_reprs = {repr(e.program) for e in self.memory.get_library()}
        new_entries = []
        for subtree, task_ids, usefulness in candidates:
            if repr(subtree) in existing_reprs:
                continue
            entry_name = f"learned_{lib_before + len(new_entries)}"
            entry = LibraryEntry(
                name=entry_name,
                program=subtree,
                usefulness=usefulness,
                reuse_count=0,
                source_tasks=task_ids,
                domain="",
            )
            new_entries.append(entry)
            existing_reprs.add(repr(subtree))

        # Add to memory
        accepted = []
        for entry in new_entries:
            if self.memory.add_to_library(entry):
                accepted.append(entry)

        # Decay old entries
        for entry in self.memory.get_library():
            if entry not in accepted:
                self.memory.update_usefulness(
                    entry.name,
                    entry.usefulness * (cfg.usefulness_decay - 1),
                )

        # Prune dead entries
        pruned = self.memory.prune_library(min_usefulness=0.01)

        lib_after = len(self.memory.get_library())
        wall = time.time() - t0

        logger.info(
            f"  [sleep] Extracted {len(accepted)} new abstractions. "
            f"Library: {lib_before} → {lib_after}. Time: {wall:.1f}s"
        )
        return SleepResult(
            new_entries=accepted,
            library_size_before=lib_before,
            library_size_after=lib_after,
            wall_time=wall,
        )

    # -------------------------------------------------------------------------
    # CURRICULUM
    # -------------------------------------------------------------------------

    @staticmethod
    def performance_core_count() -> int:
        """Return the number of performance cores (P-cores)."""
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                p_cores = int(result.stdout.strip())
                if p_cores > 0:
                    return p_cores
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired, OSError):
            pass
        total = os.cpu_count() or 1
        if total <= 2:
            return 1
        return total - 2

    def run_curriculum(
        self,
        tasks: list[Task],
        config: CurriculumConfig | None = None,
        on_task_done: "Optional[callable]" = None,
        on_round_done: "Optional[callable]" = None,
    ) -> list[RoundResult]:
        """Run multiple wake-sleep rounds over a task set."""
        cfg = config or CurriculumConfig()

        if cfg.sort_by_difficulty:
            tasks = sorted(tasks, key=lambda t: t.difficulty)
        else:
            tasks = list(tasks)
            random.Random(self.search_cfg.seed or 42).shuffle(tasks)

        if cfg.workers <= 0:
            cfg.workers = self.performance_core_count()

        results = []
        for round_num in range(cfg.wake_sleep_rounds):
            logger.info(f"=== Round {round_num + 1}/{cfg.wake_sleep_rounds} ===")
            logger.info(f"    Library size: {len(self.memory.get_library())}")
            logger.info(f"    Workers: {cfg.workers}")

            wake_results = self._wake_parallel(
                tasks, cfg.workers, round_num + 1, on_task_done)

            sleep_result = self.sleep()

            train_solved = sum(1 for w in wake_results if w.train_solved)
            total = len(wake_results)
            train_rate = train_solved / total if total > 0 else 0.0

            rr = RoundResult(
                round_number=round_num + 1,
                wake_results=wake_results,
                sleep_result=sleep_result,
                train_solved=train_solved,
                tasks_total=total,
                train_solve_rate=train_rate,
                cumulative_library_size=len(self.memory.get_library()),
            )
            results.append(rr)

            if on_round_done:
                on_round_done(round_num + 1, rr, self.memory)

            logger.info(
                f"=== Round {round_num + 1} summary: "
                f"solved {rr.solved}/{total} ({rr.solve_rate:.1%}), "
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
        """Run wake_on_task across tasks using a process pool."""
        total_tasks = len(tasks)
        base_seed = self.search_cfg.seed or 0

        if workers <= 1 or len(tasks) <= 2:
            wake_results = []
            for i, task in enumerate(tasks):
                wr = self.wake_on_task(task)
                wake_results.append(wr)
                if on_task_done:
                    on_task_done(round_num, i + 1, total_tasks, wr)
            return wake_results

        library_snapshot = self.memory.get_library()
        search_cfg = self.search_cfg
        transition_matrix = self._transition_matrix

        worker_args = []
        for i, task in enumerate(tasks):
            task_seed = hash((base_seed, round_num, i)) & 0x7FFFFFFF
            worker_args.append(
                (task, self.env, self.grammar, self.drive,
                 library_snapshot, search_cfg, transition_matrix, task_seed)
            )

        wake_results: list[WakeResult] = [None] * len(tasks)  # type: ignore
        completed_count = 0

        try:
            _mp_ctx = _mp.get_context("forkserver")
        except ValueError:
            _mp_ctx = _mp.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            mp_context=_mp_ctx,
        )
        try:
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
            pool.shutdown(wait=True)
        except KeyboardInterrupt:
            import signal as _sig
            for pid in pool._processes:
                try:
                    os.kill(pid, _sig.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        except (OSError, RuntimeError) as e:
            pool.shutdown(wait=False, cancel_futures=True)
            logger.warning(f"Parallel wake failed ({e}), falling back to sequential")
            wake_results = []
            for i, task in enumerate(tasks):
                wr = self.wake_on_task(task)
                wake_results.append(wr)
                if on_task_done:
                    on_task_done(round_num, i + 1, total_tasks, wr)
            return wake_results

        for wr in wake_results:
            if wr and wr.best:
                self.memory.record_episode(
                    wr.task_id, [], wr.best.program, wr.best.energy)
                if wr.train_solved:
                    self.memory.store_solution(wr.task_id, wr.best)
                    self._credit_library_usage(wr.best.program)
                else:
                    self.memory.store_best_attempt(wr.task_id, wr.best)

        return wake_results

    # -------------------------------------------------------------------------
    # Exhaustive enumeration
    # -------------------------------------------------------------------------

    def _exhaustive_enumerate(
        self,
        primitives: list[Primitive],
        task: Task,
        max_depth: int = 2,
        top_k: int = 15,
        eval_budget: int = 0,
    ) -> tuple[list[ScoredProgram], int]:
        """Enumerate ALL programs up to max_depth and evaluate them."""
        scored: list[ScoredProgram] = []
        n_evals = 0
        solve_thresh = self.search_cfg.solve_threshold
        pair_top_k = self.search_cfg.exhaustive_pair_top_k
        triple_top_k = self.search_cfg.exhaustive_triple_top_k

        def _budget_ok() -> bool:
            return eval_budget <= 0 or n_evals < eval_budget

        # --- Depth 1: all single transform primitives ---
        unary_prims = [p for p in primitives
                       if p.arity <= 1 and p.kind == "transform"]
        prim_by_name: dict[str, Primitive] = {p.name: p for p in unary_prims}
        depth1_solved = False
        noop_prims: set[str] = set()
        for prim in unary_prims:
            prog = Program(root=prim.name)
            sp = self._evaluate_program(prog, task)
            scored.append(sp)
            n_evals += 1
            if sp.prediction_error <= solve_thresh:
                depth1_solved = True
            if sp.prediction_error > solve_thresh:
                is_noop = True
                for inp, _ in task.train_examples:
                    out = self.env.execute(prog, inp)
                    if out != inp:
                        is_noop = False
                        break
                if is_noop:
                    noop_prims.add(prim.name)

        # --- Parameterized prims with perception children ---
        param_prims = [p for p in primitives if p.kind == "parameterized"]
        percep_prims = [p for p in primitives if p.kind == "perception"]
        if param_prims and percep_prims:
            for pprim in param_prims:
                if pprim.arity == 1:
                    for perc in percep_prims:
                        prog = Program(root=pprim.name,
                                       children=[Program(root=perc.name)])
                        sp = self._evaluate_program(prog, task)
                        scored.append(sp)
                        n_evals += 1
                        if sp.prediction_error <= solve_thresh:
                            depth1_solved = True
                elif pprim.arity == 2:
                    for p1 in percep_prims:
                        for p2 in percep_prims:
                            prog = Program(root=pprim.name,
                                           children=[Program(root=p1.name),
                                                     Program(root=p2.name)])
                            sp = self._evaluate_program(prog, task)
                            scored.append(sp)
                            n_evals += 1
                            if sp.prediction_error <= solve_thresh:
                                depth1_solved = True

        if depth1_solved:
            return scored, n_evals

        if max_depth < 2 or not _budget_ok():
            return scored, n_evals

        # --- Build pair pool ---
        prim_scores = self.memory.get_primitive_scores()
        depth1_ranked = sorted(scored, key=lambda s: s.prediction_error)
        essential_names = self.grammar.essential_pair_concepts()

        depth1_scores: dict[str, float] = {}
        for sp in depth1_ranked:
            if sp.program.root not in depth1_scores:
                depth1_scores[sp.program.root] = sp.prediction_error

        def _pool_sort_key(name: str) -> float:
            d1_err = depth1_scores.get(name, 1.0)
            roi = prim_scores.get(name, 0.0)
            return d1_err / (1.0 + roi)

        depth1_ranked = sorted(
            depth1_ranked,
            key=lambda s: _pool_sort_key(s.program.root))

        seen_names: set[str] = set()
        pair_pool: list[str] = []
        top_scorer_cap = pair_top_k * 3 // 5
        for sp in depth1_ranked:
            name = sp.program.root
            if name not in seen_names:
                pair_pool.append(name)
                seen_names.add(name)
            if len(pair_pool) >= top_scorer_cap:
                break

        remaining_essentials = [
            n for n in essential_names
            if n not in seen_names and n in prim_by_name
        ]
        remaining_essentials.sort(key=_pool_sort_key)
        for name in remaining_essentials:
            if len(pair_pool) >= pair_top_k:
                break
            pair_pool.append(name)
            seen_names.add(name)

        for sp in depth1_ranked:
            if len(pair_pool) >= pair_top_k:
                break
            name = sp.program.root
            if name not in seen_names:
                pair_pool.append(name)
                seen_names.add(name)

        # Smart pruning for inner steps
        INNER_STEP_THRESHOLD = 0.70
        inner_pool = [
            name for name in pair_pool
            if name not in noop_prims
            and depth1_scores.get(name, 1.0) <= INNER_STEP_THRESHOLD
        ]
        if len(inner_pool) < pair_top_k // 3:
            inner_pool = [n for n in pair_pool[:pair_top_k // 2]
                          if n not in noop_prims]

        # --- Depth 2: K × K' pairs ---
        for outer_name in pair_pool:
            if not _budget_ok():
                break
            if outer_name in noop_prims:
                continue
            for inner_name in inner_pool:
                if not _budget_ok():
                    break
                prog = Program(root=outer_name, children=[
                    Program(root=inner_name)])
                sp = self._evaluate_program(prog, task)
                scored.append(sp)
                n_evals += 2
                if sp.prediction_error <= solve_thresh:
                    return scored, n_evals

        # --- Depth 2.5: Binary composition ---
        binary_prims = [p for p in primitives if p.arity == 2]
        if binary_prims and _budget_ok():
            OVERLAY_TOP_K = 15
            overlay_pool = [n for n in pair_pool[:OVERLAY_TOP_K]
                           if n not in noop_prims]
            for bp in binary_prims:
                if not _budget_ok():
                    break
                for a_name in overlay_pool:
                    if not _budget_ok():
                        break
                    for b_name in overlay_pool:
                        if not _budget_ok():
                            break
                        if a_name == b_name:
                            continue
                        prog = Program(
                            root=bp.name,
                            children=[
                                Program(root=a_name),
                                Program(root=b_name),
                            ],
                        )
                        sp = self._evaluate_program(prog, task)
                        scored.append(sp)
                        n_evals += 2
                        if sp.prediction_error <= solve_thresh:
                            return scored, n_evals

        if max_depth < 3 or not _budget_ok():
            return scored, n_evals

        # --- Depth 3 ---
        DEPTH3_SKIP_THRESHOLD = 0.65
        depth2_best = min(
            (s.prediction_error for s in scored if s.program.children),
            default=1.0)
        if depth2_best > DEPTH3_SKIP_THRESHOLD:
            return scored, n_evals

        depth2_ranked = sorted(
            [s for s in scored if s.program.children],
            key=lambda s: s.prediction_error)
        triple_seen: set[str] = set()
        triple_pool: list[str] = []

        depth2_cap = triple_top_k // 3
        for sp in depth2_ranked:
            if len(triple_pool) >= depth2_cap:
                break
            for name in [sp.program.root] + [
                    c.root for c in (sp.program.children or [])]:
                if name not in triple_seen and name in prim_by_name:
                    triple_pool.append(name)
                    triple_seen.add(name)

        for name in essential_names:
            if len(triple_pool) >= triple_top_k * 2 // 3:
                break
            if name not in triple_seen and name in prim_by_name:
                triple_pool.append(name)
                triple_seen.add(name)

        for sp in depth1_ranked:
            name = sp.program.root
            if name not in triple_seen:
                triple_pool.append(name)
                triple_seen.add(name)
            if len(triple_pool) >= triple_top_k:
                break

        for a in triple_pool:
            if not _budget_ok():
                break
            if a == "identity":
                continue
            for b in triple_pool:
                if not _budget_ok():
                    break
                if b == "identity":
                    continue
                for c in triple_pool:
                    if not _budget_ok():
                        break
                    if c == "identity":
                        continue
                    if a == b == c:
                        continue
                    prog = Program(root=a, children=[
                        Program(root=b, children=[
                            Program(root=c)])])
                    sp = self._evaluate_program(prog, task)
                    scored.append(sp)
                    n_evals += 3
                    if sp.prediction_error <= solve_thresh:
                        return scored, n_evals

        return scored, n_evals

    # -------------------------------------------------------------------------
    # Near-miss refinement
    # -------------------------------------------------------------------------

    def _near_miss_refine(
        self,
        candidates: list[ScoredProgram],
        primitives: list[Primitive],
        task: Task,
        threshold: float = 0.20,
    ) -> tuple[list[ScoredProgram], int]:
        """For programs scoring close-but-not-perfect, try appending/prepending
        each unary primitive to fix them.

        Cost: O(near_misses × unary_prims × 2) plus node replacement and
        two-step refinement for close misses.
        """
        solve_thresh = self.search_cfg.solve_threshold
        near_misses = [
            sp for sp in candidates
            if solve_thresh < sp.prediction_error <= threshold
        ]
        if not near_misses:
            return [], 0

        near_misses.sort(key=lambda s: s.prediction_error)
        near_misses = near_misses[:5]

        all_unary = [p for p in primitives if p.arity <= 1]
        refined: list[ScoredProgram] = []
        n_evals = 0

        for nm in near_misses:
            for prim in all_unary:
                # Append: prim(near_miss_program)
                prog_append = Program(
                    root=prim.name,
                    children=[copy.deepcopy(nm.program)],
                )
                sp = self._evaluate_program(prog_append, task)
                refined.append(sp)
                n_evals += 1
                if sp.prediction_error <= solve_thresh:
                    return refined, n_evals

                # Prepend: insert prim as the innermost step
                prog_prepend = copy.deepcopy(nm.program)
                node = prog_prepend
                while node.children:
                    node = node.children[0]
                old_root = node.root
                node.root = prim.name
                node.children = [Program(root=old_root)]
                sp = self._evaluate_program(prog_prepend, task)
                refined.append(sp)
                n_evals += 1
                if sp.prediction_error <= solve_thresh:
                    return refined, n_evals

        # Node replacement for depth-1+ near-misses
        NODE_REPLACE_PRIMS = 60
        depth1_sorted = sorted(
            [sp for sp in candidates if sp.program.depth == 1],
            key=lambda s: s.prediction_error)
        replace_names = set()
        for sp in depth1_sorted:
            replace_names.add(sp.program.root)
            if len(replace_names) >= NODE_REPLACE_PRIMS:
                break
        replace_prims = [p for p in all_unary if p.name in replace_names]

        for nm in near_misses:
            if nm.program.depth < 1:
                continue
            nodes_to_replace: list[Program] = []

            def _collect(node: Program, parent: Optional[Program] = None):
                if parent is not None:
                    nodes_to_replace.append(node)
                for child in (node.children or []):
                    _collect(child, node)

            _collect(nm.program)
            for target_node in nodes_to_replace:
                original_root = target_node.root
                for prim in replace_prims:
                    if prim.name == original_root:
                        continue
                    target_node.root = prim.name
                    prog_replaced = copy.deepcopy(nm.program)
                    target_node.root = original_root
                    sp = self._evaluate_program(prog_replaced, task)
                    refined.append(sp)
                    n_evals += 1
                    if sp.prediction_error <= solve_thresh:
                        return refined, n_evals

        # Two-step refinement for close misses (error < 0.10)
        TWO_STEP_THRESHOLD = 0.10
        close_misses = [sp for sp in refined if sp.prediction_error < TWO_STEP_THRESHOLD]
        if not close_misses:
            close_misses = [nm for nm in near_misses if nm.prediction_error < TWO_STEP_THRESHOLD]
        if close_misses:
            close_misses.sort(key=lambda s: s.prediction_error)
            close_misses = close_misses[:5]
            for cm in close_misses:
                for prim in all_unary:
                    prog_outer = Program(
                        root=prim.name,
                        children=[copy.deepcopy(cm.program)],
                    )
                    sp = self._evaluate_program(prog_outer, task)
                    refined.append(sp)
                    n_evals += 1
                    if sp.prediction_error <= solve_thresh:
                        return refined, n_evals

        # Binary near-miss refinement
        binary_prims = [p for p in primitives if p.arity == 2 and p.kind == "transform"]
        if binary_prims:
            BINARY_TOP_K = 15
            top_depth1 = sorted(
                [sp for sp in candidates if sp.program.depth == 1],
                key=lambda s: s.prediction_error)[:BINARY_TOP_K]
            for nm in near_misses[:5]:
                for bp in binary_prims:
                    for other in top_depth1:
                        prog_a = Program(root=bp.name, children=[
                            copy.deepcopy(nm.program),
                            copy.deepcopy(other.program)])
                        sp = self._evaluate_program(prog_a, task)
                        refined.append(sp)
                        n_evals += 1
                        if sp.prediction_error <= solve_thresh:
                            return refined, n_evals
                        prog_b = Program(root=bp.name, children=[
                            copy.deepcopy(other.program),
                            copy.deepcopy(nm.program)])
                        sp = self._evaluate_program(prog_b, task)
                        refined.append(sp)
                        n_evals += 1
                        if sp.prediction_error <= solve_thresh:
                            return refined, n_evals

        return refined, n_evals

    # -------------------------------------------------------------------------
    # Conditional search
    # -------------------------------------------------------------------------

    def _try_conditional_search(
        self,
        predicates: list[tuple[str, callable]],
        candidates: list[ScoredProgram],
        primitives: list[Primitive],
        task: Task,
        top_k: int = 15,
    ) -> tuple[Optional[ScoredProgram], int]:
        """Search for conditional programs: if pred(input) then A else B."""
        n_evals = 0
        solve_thresh = self.search_cfg.solve_threshold

        # Build candidate pool from top depth-1 + depth-2 programs
        depth1 = [sp for sp in candidates if sp.program.depth == 1]
        depth1.sort(key=lambda s: s.prediction_error)
        top_prims_names = []
        seen = set()
        for sp in depth1:
            if sp.program.root not in seen:
                top_prims_names.append(sp.program.root)
                seen.add(sp.program.root)
            if len(top_prims_names) >= top_k:
                break

        prim_map = {p.name: p for p in primitives}
        top_prims = [prim_map[n] for n in top_prims_names if n in prim_map]

        # Add top depth-2 programs as branch candidates
        depth2 = [sp for sp in candidates
                   if sp.program.depth == 2 and sp.program.children]
        depth2.sort(key=lambda s: s.prediction_error)
        depth2_added = 0
        DEPTH2_BRANCH_K = 15
        for sp in depth2:
            prog_repr = repr(sp.program)
            if prog_repr in seen:
                continue
            seen.add(prog_repr)

            def _make_d2_fn(prog=sp.program, env=self.env):
                def fn(grid):
                    return env.execute(prog, grid)
                return fn
            d2_prim = Primitive(
                name=prog_repr, arity=1, fn=_make_d2_fn(), domain="arc")
            top_prims.append(d2_prim)
            depth2_added += 1
            if depth2_added >= DEPTH2_BRANCH_K:
                break

        if len(top_prims) < 2:
            return None, 0

        best_result: Optional[ScoredProgram] = None

        for pred_name, pred_fn in predicates:
            true_indices = []
            false_indices = []
            for idx, (inp, _) in enumerate(task.train_examples):
                try:
                    if pred_fn(inp):
                        true_indices.append(idx)
                    else:
                        false_indices.append(idx)
                except Exception:
                    false_indices.append(idx)

            if not true_indices or not false_indices:
                continue

            true_scores: list[tuple[float, Primitive]] = []
            false_scores: list[tuple[float, Primitive]] = []

            for prim in top_prims:
                true_err = 0.0
                for idx in true_indices:
                    inp, expected = task.train_examples[idx]
                    try:
                        out = prim.fn(inp)
                        true_err += self.drive.prediction_error(out, expected)
                    except Exception:
                        true_err += 1.0
                true_scores.append((true_err / len(true_indices), prim))

                false_err = 0.0
                for idx in false_indices:
                    inp, expected = task.train_examples[idx]
                    try:
                        out = prim.fn(inp)
                        false_err += self.drive.prediction_error(out, expected)
                    except Exception:
                        false_err += 1.0
                false_scores.append((false_err / len(false_indices), prim))

            true_scores.sort(key=lambda x: x[0])
            false_scores.sort(key=lambda x: x[0])

            best_true = [p for _, p in true_scores[:5]]
            best_false = [p for _, p in false_scores[:5]]

            for then_prim in best_true:
                for else_prim in best_false:
                    if then_prim.name == else_prim.name:
                        continue

                    def _make_cond(pf, tf, ef):
                        def cond_fn(grid):
                            try:
                                if pf(grid):
                                    return tf(grid)
                                else:
                                    return ef(grid)
                            except Exception:
                                return grid
                        return cond_fn

                    cond_fn = _make_cond(pred_fn, then_prim.fn, else_prim.fn)
                    cond_name = f"if_{pred_name}_{then_prim.name}_else_{else_prim.name}"

                    cond_prim = Primitive(
                        name=cond_name, arity=1, fn=cond_fn, domain="arc")
                    self.env.register_primitive(cond_prim)

                    prog = Program(root=cond_name)
                    sp = self._evaluate_program(prog, task)
                    n_evals += 1

                    if sp.prediction_error <= solve_thresh:
                        return sp, n_evals
                    if best_result is None or sp.energy < best_result.energy:
                        best_result = sp

        return best_result, n_evals

    # -------------------------------------------------------------------------
    # Color fix
    # -------------------------------------------------------------------------

    def _try_color_fix(
        self,
        candidates: list[ScoredProgram],
        task: Task,
        threshold: float = 0.30,
    ) -> Optional[ScoredProgram]:
        """Try to fix near-miss programs by learning a color correction."""
        solve_thresh = self.search_cfg.solve_threshold
        near_misses = [
            sp for sp in candidates
            if solve_thresh < sp.prediction_error <= threshold
        ]
        if not near_misses:
            return None

        near_misses.sort(key=lambda s: s.prediction_error)
        near_misses = near_misses[:20]

        best_fix: Optional[ScoredProgram] = None

        # Try correction(identity)
        identity_outputs = [inp for inp, _ in task.train_examples]
        identity_expected = [exp for _, exp in task.train_examples]
        correction = self.env.infer_output_correction(
            identity_outputs, identity_expected)
        if correction is not None:
            sp = self._evaluate_program(correction, task)
            if sp.prediction_error <= solve_thresh:
                return sp
            if best_fix is None or sp.energy < best_fix.energy:
                best_fix = sp

        for nm in near_misses:
            outputs = []
            expected = []
            ok = True
            for inp, exp in task.train_examples:
                try:
                    out = self.env.execute(nm.program, inp)
                    outputs.append(out)
                    expected.append(exp)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            correction = self.env.infer_output_correction(outputs, expected)
            if correction is None:
                continue

            base = nm.program
            if base.root == "identity" and not base.children:
                fixed_prog = correction
            else:
                fixed_prog = Program(
                    root=correction.root,
                    children=[copy.deepcopy(base)],
                    params=correction.params,
                )
            sp = self._evaluate_program(fixed_prog, task)

            if sp.prediction_error >= nm.prediction_error:
                continue

            if sp.prediction_error <= solve_thresh:
                return sp
            if best_fix is None or sp.energy < best_fix.energy:
                best_fix = sp

        return best_fix

    # -------------------------------------------------------------------------
    # Beam search helpers
    # -------------------------------------------------------------------------

    def _init_beam(self, primitives: list[Primitive], n: int) -> list[Program]:
        """Generate n random programs of varying depth (1-4)."""
        beam = []
        use_prior = self._transition_matrix.size > 0
        for i in range(n):
            r = self._rng.random()
            max_depth = 1 if r < 0.2 else (2 if r < 0.55 else (3 if r < 0.85 else 4))
            prog = self._random_program(primitives, max_depth, use_prior)
            beam.append(prog)
        return beam

    def _random_program(self, primitives: list[Primitive], max_depth: int,
                        use_prior: bool, parent_op: str = "") -> Program:
        """Generate a random program tree up to max_depth."""
        if max_depth <= 1:
            leaf_prims = [p for p in primitives if p.arity <= 1]
            if not leaf_prims:
                leaf_prims = primitives
            if use_prior and parent_op:
                prim = self._transition_matrix.weighted_choice(
                    parent_op, leaf_prims, self._rng)
            else:
                prim = self._rng.choice(leaf_prims)
            return Program(root=prim.name)

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

    def _semantic_hash(self, program: Program, task: Task) -> str:
        """Hash a program by its outputs on training inputs."""
        precision = self.search_cfg.dedup_precision
        outputs = []
        for inp, _ in task.train_examples:
            try:
                val = self.env.execute(program, inp)
                if isinstance(val, (int, float)):
                    outputs.append(round(float(val), precision))
                elif isinstance(val, list):
                    outputs.append(tuple(tuple(row) for row in val) if val and isinstance(val[0], list) else tuple(val))
                else:
                    outputs.append(val)
            except Exception:
                outputs.append(None)
        return str(outputs)

    def _semantic_dedup(self, scored: list[ScoredProgram],
                        task: Task) -> tuple[list[ScoredProgram], int]:
        """Remove semantically duplicate programs from the scored list."""
        seen: dict[str, ScoredProgram] = {}
        for sp in scored:
            key = self._semantic_hash(sp.program, task)
            if key not in seen or sp.energy < seen[key].energy:
                seen[key] = sp
        deduped = sorted(seen.values(), key=lambda s: s.energy)
        return deduped, len(scored) - len(deduped)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _avg_cells(task: Task) -> int:
        """Max cell count across training input grids."""
        grids = [inp for inp, _ in task.train_examples]
        if not grids:
            return 1
        sizes = []
        for g in grids:
            try:
                if g and len(g) > 0 and len(g[0]) > 0:
                    sizes.append(len(g) * len(g[0]))
            except TypeError:
                continue
        return max(sizes) if sizes else 1

    def _evaluate_program(self, program: Program, task: Task) -> ScoredProgram:
        """Evaluate a program on all training examples, return scored result."""
        total_error = 0.0
        max_error = 0.0
        n = len(task.train_examples)
        n_solved = 0
        threshold = self.search_cfg.solve_threshold

        for inp, expected in task.train_examples:
            try:
                predicted = self.env.execute(program, inp)
                err = self.drive.prediction_error(predicted, expected)
            except Exception:
                err = 1e6
            total_error += err
            max_error = max(max_error, err)
            if err <= threshold:
                n_solved += 1

        avg_error = total_error / n if n > 0 else total_error
        comp_cost = self.drive.complexity_cost(program)
        energy = self.search_cfg.energy_alpha * avg_error + self.search_cfg.energy_beta * comp_cost

        exponent = self.sleep_cfg.example_solve_exponent
        solve_score = (n_solved / n) ** exponent if n > 0 else 0.0

        return ScoredProgram(
            program=program,
            energy=energy,
            prediction_error=avg_error,
            complexity_cost=comp_cost,
            max_example_error=max_error,
            example_solve_score=solve_score,
        )

    def _simplify_program(self, prog: Program, task: Task) -> Program:
        """Remove identity steps from a program tree (bottom-up)."""
        if not prog.children:
            return prog

        new_children = [self._simplify_program(c, task) for c in prog.children]
        changed = any(nc is not oc for nc, oc in zip(new_children, prog.children))
        result = (Program(root=prog.root, children=new_children, params=prog.params)
                  if changed else prog)

        if len(result.children) == 1:
            child = result.children[0]
            if self._outputs_equal(result, child, task):
                return child
            parent_only = Program(root=result.root, params=result.params)
            if self._outputs_equal(result, parent_only, task):
                return parent_only

        if len(result.children) == 2:
            if self._outputs_equal(result, result.children[0], task):
                return result.children[0]
            if self._outputs_equal(result, result.children[1], task):
                return result.children[1]

        return result

    def _outputs_equal(self, prog_a: Program, prog_b: Program,
                       task: Task) -> bool:
        """Check if two programs produce identical output on all training examples."""
        for inp, _ in task.train_examples:
            try:
                out_a = self.env.execute(prog_a, inp)
                out_b = self.env.execute(prog_b, inp)
                if out_a != out_b:
                    return False
            except Exception:
                return False
        return True

    def _update_pareto_front(self, pareto: dict[int, ParetoEntry],
                             sp: ScoredProgram) -> None:
        c = sp.program.size
        if c not in pareto or sp.prediction_error < pareto[c].prediction_error:
            pareto[c] = ParetoEntry(
                complexity=c,
                prediction_error=sp.prediction_error,
                energy=sp.energy,
                program=sp.program,
            )

    def _extract_pareto_front(self, pareto: dict[int, ParetoEntry]) -> list[ParetoEntry]:
        entries = sorted(pareto.values(), key=lambda e: e.complexity)
        front = []
        best_error = float('inf')
        for entry in entries:
            if entry.prediction_error < best_error:
                front.append(entry)
                best_error = entry.prediction_error
        return front

    def _enumerate_subtrees(self, program: Program) -> list[Program]:
        """Return every sub-tree in a program (including the root)."""
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
