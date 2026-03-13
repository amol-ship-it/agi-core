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
    Decomposition,
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
# Module-level worker for multiprocessing (must be picklable)
# =============================================================================

def _worker_init():
    """Initializer for worker processes: ignore SIGINT.

    The parent process handles Ctrl-C and terminates the pool.
    Without this, workers trap SIGINT and the pool hangs.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


def _wake_worker(args: tuple) -> WakeResult:
    """
    Solve a single task in a child process.

    Receives a snapshot of the learner state, reconstructs a mini-Learner,
    and returns the WakeResult. This function is at module level so it can
    be pickled by ProcessPoolExecutor.

    Each worker gets a per-task seed derived from the base seed + task index,
    ensuring deterministic results regardless of worker scheduling order.
    """
    task, env, grammar, drive, library, search_cfg, transition_matrix, task_seed = args

    # Reconstruct a lightweight learner with a per-task seed
    from .memory import InMemoryStore
    memory = InMemoryStore()
    for entry in library:
        memory.add_to_library(entry)

    # Override seed with per-task seed for deterministic parallel execution
    worker_cfg = SearchConfig(
        beam_width=search_cfg.beam_width,
        mutations_per_candidate=search_cfg.mutations_per_candidate,
        crossover_fraction=search_cfg.crossover_fraction,
        max_generations=search_cfg.max_generations,
        energy_alpha=search_cfg.energy_alpha,
        energy_beta=search_cfg.energy_beta,
        early_stop_energy=search_cfg.early_stop_energy,
        solve_threshold=search_cfg.solve_threshold,
        seed=task_seed,
        semantic_dedup=search_cfg.semantic_dedup,
        dedup_precision=search_cfg.dedup_precision,
        exhaustive_depth=search_cfg.exhaustive_depth,
        exhaustive_pair_top_k=search_cfg.exhaustive_pair_top_k,
        exhaustive_triple_top_k=search_cfg.exhaustive_triple_top_k,
        near_miss_threshold=search_cfg.near_miss_threshold,
        eval_budget=search_cfg.eval_budget,
    )

    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=worker_cfg,
    )
    learner._transition_matrix = transition_matrix

    # Re-seed the grammar's RNG with the per-task seed for deterministic mutations
    grammar._rng = random.Random(task_seed)

    # Log task start (flushes to help identify memory-hungry tasks)
    import sys as _sys
    import resource as _resource
    rss_mb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"    [worker pid={os.getpid()}] STARTING {task.task_id} (RSS={rss_mb:.0f}MB)",
          flush=True)

    # Solve — but skip memory recording (main process handles that)
    result = learner._wake_on_task_no_record(task)

    rss_mb_after = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"    [worker pid={os.getpid()}] FINISHED {task.task_id} (RSS={rss_mb_after:.0f}MB)",
          flush=True)
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
    # WAKE PHASE: observe → hypothesize → execute → score → store
    # -------------------------------------------------------------------------

    def wake_on_task(self, task: Task) -> WakeResult:
        """
        Attempt to solve a single task via exhaustive enumeration + beam search.

        Phase 1: Exhaustive enumeration of ALL programs up to exhaustive_depth.
                 This is cheap (N + N×K for depth 2) and catches easy tasks.
        Phase 2: Beam search with mutation/crossover for harder tasks.
                 Seeds the beam with the best programs from enumeration.

        The key compounding insight: learned library entries are 0-arity
        primitives. A depth-1 program using a learned concept IS a depth-3+
        program in disguise. So exhaustive depth-1 search over a rich
        vocabulary reaches further than deep search over a small one.
        """
        return self._wake_core(task, record=True)

    def _wake_on_task_no_record(self, task: Task) -> WakeResult:
        """
        Same as wake_on_task but does NOT write to memory.

        Used by parallel workers — the main process merges results.
        """
        return self._wake_core(task, record=False)

    def _wake_core(self, task: Task, record: bool) -> WakeResult:
        """Shared wake logic. When record=True, writes solutions to memory."""
        t0 = time.time()
        cfg = self.search_cfg

        # Let the grammar cache task-specific data (e.g. training pairs)
        self.grammar.prepare_for_task(task)

        # Combine hand-coded primitives with learned library entries
        base_prims = self.grammar.base_primitives()
        library_prims = self.grammar.inject_library(self.memory.get_library())
        all_prims = base_prims + library_prims

        # Register library primitives with the environment so it can execute them.
        # Library entries have fn=Program; the environment needs to know how to
        # resolve these during execution.
        for lp in library_prims:
            self.env.register_primitive(lp)

        best_so_far: Optional[ScoredProgram] = None
        n_evals = 0
        total_deduped = 0
        pareto: dict[int, ParetoEntry] = {}
        enum_candidates: list[ScoredProgram] = []

        # Cell-normalized compute budget (deterministic, reproducible).
        # Budget is in "ops" = depth-weighted evaluations × cell factor.
        # A depth-2 eval costs 2 ops (applies 2 primitives), depth-3 costs 3.
        # This makes the budget a true proxy for compute, not just eval count.
        DEFAULT_CELLS = 800
        cells = self._avg_cells(task)
        if cfg.eval_budget > 0:
            # Scale inversely with grid size: larger grids get fewer evals.
            eval_budget = max(cfg.eval_budget * DEFAULT_CELLS // max(cells, 1), 500)
            eval_budget = min(eval_budget, cfg.eval_budget * 4)  # cap at 4x base
        else:
            eval_budget = 0  # unlimited

        def _budget_ok() -> bool:
            """Check whether we've exceeded the per-task eval budget."""
            return eval_budget <= 0 or n_evals < eval_budget

        def _record_solve(sp: Optional[ScoredProgram] = None):
            """Record solution in memory if record=True."""
            prog = sp or best_so_far
            if record and prog:
                self.memory.record_episode(
                    task.task_id, task.train_examples,
                    prog.program, prog.energy)
                self.memory.store_solution(task.task_id, prog)
                self._credit_library_usage(prog.program)

        def _make_solved_result(phase_name: str, gens: int = 0, deduped: int = 0) -> WakeResult:
            """Build WakeResult for a training-solved task using top-k test evaluation."""
            nonlocal best_so_far
            top_sp, te, ts, n_perf, s_rank = self._evaluate_top_k_on_test(
                enum_candidates, task, top_k=10)
            # Use the test-best candidate as the reported best
            if top_sp is not None and top_sp is not best_so_far:
                best_so_far = top_sp
            # If enum_candidates didn't yield a test result (e.g. solve came
            # from correction/identity fix), evaluate best_so_far directly.
            if ts is None and best_so_far is not None:
                te, ts = self._evaluate_on_test(best_so_far, task)
            best_so_far.task_id = task.task_id
            _record_solve(best_so_far)
            front = self._extract_pareto_front(pareto)
            wall = time.time() - t0
            logger.info(
                f"  [wake] Task {task.task_id}: SOLVED by {phase_name}, "
                f"energy={best_so_far.energy:.6f}, evals={n_evals}, "
                f"candidates={n_perf}, time={wall:.1f}s")
            train_preds, test_preds = self._compute_predictions(best_so_far, task)
            return WakeResult(
                task_id=task.task_id, train_solved=True, best=best_so_far,
                generations_used=gens, evaluations=n_evals, wall_time=wall,
                pareto_front=front, dedup_count=deduped,
                test_error=te, test_solved=ts,
                n_train_perfect=n_perf, solving_rank=s_rank,
                train_predictions=train_preds, test_predictions=test_preds)

        # --- Phase 1: Exhaustive enumeration ---
        t_phase = time.time()
        if cfg.exhaustive_depth >= 1:
            enum_candidates, n_enum_evals = self._exhaustive_enumerate(
                all_prims, task, cfg.exhaustive_depth,
                eval_budget=eval_budget)
            n_evals += n_enum_evals
            for sp in enum_candidates:
                self._update_pareto_front(pareto, sp)
                if best_so_far is None or sp.energy < best_so_far.energy:
                    best_so_far = sp

            # Early exit if enumeration found a perfect solve
            if best_so_far and best_so_far.prediction_error <= cfg.solve_threshold:
                return _make_solved_result("enumeration")

        logger.debug(f"  [wake] Phase 1 enumeration: {time.time()-t_phase:.2f}s, {n_evals} evals")

        # --- Phase 1.1: Object decomposition ---
        # Try applying the same transform to each object independently.
        # High-ROI for tasks where objects are transformed in-place.
        t_phase = time.time()
        decomp_result = self.env.try_object_decomposition(task, all_prims) if _budget_ok() else None
        if decomp_result is not None:
            name, fn = decomp_result
            prog = Program(root=name)
            sp = self._evaluate_program(prog, task)
            n_evals += 1
            self._update_pareto_front(pareto, sp)
            if best_so_far is None or sp.energy < best_so_far.energy:
                best_so_far = sp
            enum_candidates.append(sp)

            if best_so_far and best_so_far.prediction_error <= cfg.solve_threshold:
                return _make_solved_result("object decomposition")

        logger.debug(f"  [wake] Phase 1.1 object decomp: {time.time()-t_phase:.2f}s")

        # --- Phase 1.15: Grammar-based decomposition ---
        # The grammar's decompose() breaks inputs into parts; we try applying
        # each primitive to all parts and recompose. This is the generic
        # version of object decomposition — works for any domain that
        # implements decompose/recompose on its Grammar.
        t_phase = time.time()
        if _budget_ok() and (not best_so_far or best_so_far.prediction_error > cfg.solve_threshold):
            decomp_result = self._try_grammar_decomposition(all_prims, task)
            if decomp_result is not None:
                n_evals += 1
                self._update_pareto_front(pareto, decomp_result)
                if best_so_far is None or decomp_result.energy < best_so_far.energy:
                    best_so_far = decomp_result
                enum_candidates.append(decomp_result)

                if best_so_far and best_so_far.prediction_error <= cfg.solve_threshold:
                    return _make_solved_result("grammar decomposition")

        logger.debug(f"  [wake] Phase 1.15 grammar decomp: {time.time()-t_phase:.2f}s")

        # --- Phase 1.25: Conditional search ---
        # Try if(predicate, A, B) programs. For each predicate, partition
        # training inputs into true/false groups and find best primitives
        # per group. Cost: O(P × top_k²) where P = #predicates (~17).
        t_phase = time.time()
        predicates = self.grammar.get_predicates()
        if predicates and enum_candidates and _budget_ok():
            cond_result, n_cond_evals = self._try_conditional_search(
                predicates, enum_candidates, all_prims, task)
            n_evals += n_cond_evals
            if cond_result is not None:
                self._update_pareto_front(pareto, cond_result)
                if best_so_far is None or cond_result.energy < best_so_far.energy:
                    best_so_far = cond_result
                enum_candidates.append(cond_result)

                if best_so_far.prediction_error <= cfg.solve_threshold:
                    return _make_solved_result("conditional")

        logger.debug(f"  [wake] Phase 1.25 conditional: {time.time()-t_phase:.2f}s")

        # --- Phase 1.5: Near-miss refinement ---
        # Try appending/prepending primitives to near-miss programs.
        # High-ROI: catches "almost right" programs that need one more step.
        t_phase = time.time()
        if cfg.near_miss_threshold > 0 and enum_candidates and _budget_ok():
            refine_candidates, n_refine_evals = self._near_miss_refine(
                enum_candidates, all_prims, task, cfg.near_miss_threshold)
            n_evals += n_refine_evals
            for sp in refine_candidates:
                self._update_pareto_front(pareto, sp)
                if best_so_far is None or sp.energy < best_so_far.energy:
                    best_so_far = sp
            enum_candidates.extend(refine_candidates)

            # Check if refinement found a perfect solve
            if best_so_far and best_so_far.prediction_error <= cfg.solve_threshold:
                return _make_solved_result("near-miss refinement")

        logger.debug(f"  [wake] Phase 1.5 near-miss refine: {time.time()-t_phase:.2f}s")

        # --- Phase 1.6: Fixed-point iteration ---
        # For near-miss depth-1 programs, try applying them repeatedly until
        # stable. Many ARC tasks need iterated application (fill propagation,
        # pattern growth, etc). Cost: O(near_misses × max_iters).
        t_phase = time.time()
        if enum_candidates and _budget_ok() and (not best_so_far or best_so_far.prediction_error > cfg.solve_threshold):
            fp_result = self._try_fixed_point(enum_candidates, task)
            if fp_result is not None:
                n_evals += 1
                self._update_pareto_front(pareto, fp_result)
                if best_so_far is None or fp_result.energy < best_so_far.energy:
                    best_so_far = fp_result
                enum_candidates.append(fp_result)

                if best_so_far and best_so_far.prediction_error <= cfg.solve_threshold:
                    return _make_solved_result("fixed-point iteration")

        logger.debug(f"  [wake] Phase 1.6 fixed-point: {time.time()-t_phase:.2f}s")

        # --- Phase 1.75: Color fix ---
        # For near-miss programs, try learning a color remapping from
        # pixel-level mismatches. Many ARC tasks differ from target by a
        # consistent color substitution. Cost: O(near_misses × examples).
        t_phase = time.time()
        if enum_candidates:
            color_fix_result = self._try_color_fix(enum_candidates, task)
            if color_fix_result is not None:
                n_evals += 1
                self._update_pareto_front(pareto, color_fix_result)
                if best_so_far is None or color_fix_result.energy < best_so_far.energy:
                    best_so_far = color_fix_result

                if best_so_far.prediction_error <= cfg.solve_threshold:
                    return _make_solved_result("color fix")

        logger.debug(f"  [wake] Phase 1.75 color fix: {time.time()-t_phase:.2f}s")

        # --- Phase 1.76: Identity-seeded correction ---
        # For same-shape tasks, try learning the ENTIRE transformation as
        # neighborhood rules from input→output directly (identity base).
        # This catches tasks where no enumerated program is close but the
        # transformation is describable as local cellular automaton rules.
        t_phase = time.time()
        identity_result = self._try_identity_correction(task)
        if identity_result is not None:
            n_evals += 1
            enum_candidates.append(identity_result)
            self._update_pareto_front(pareto, identity_result)
            if best_so_far is None or identity_result.energy < best_so_far.energy:
                best_so_far = identity_result

            if best_so_far.prediction_error <= cfg.solve_threshold:
                return _make_solved_result("identity correction")

        logger.debug(f"  [wake] Phase 1.76 identity correction: {time.time()-t_phase:.2f}s")

        # --- Phase 2: Beam search (seeded with top enumeration results) ---
        # Adaptive: reduce beam effort when enumeration found nothing promising.
        # If best error > 0.3, beam search rarely recovers — cap at 25% gens.
        # If best error > 0.15, moderate reduction — cap at 50% gens.
        # Skipped entirely if eval budget is exceeded.
        t_phase = time.time()
        gens_used = 0
        scored = []  # beam search results (may be empty if skipped)
        best_enum_error = best_so_far.prediction_error if best_so_far else 1.0
        if not _budget_ok():
            logger.debug(f"  [wake] Phase 2 beam search: SKIPPED (budget exceeded, {n_evals} evals)")
        else:
            if cfg.max_generations <= 1:
                effective_gens = cfg.max_generations
            elif best_enum_error > 0.3:
                effective_gens = max(5, cfg.max_generations // 4)
            elif best_enum_error > 0.15:
                effective_gens = max(10, cfg.max_generations // 2)
            else:
                effective_gens = cfg.max_generations

            seed_progs = [sp.program for sp in sorted(
                enum_candidates, key=lambda s: s.energy)[:cfg.beam_width // 2]]
            n_random = max(cfg.beam_width - len(seed_progs), cfg.beam_width // 2)
            beam = seed_progs + self._init_beam(all_prims, n_random)

            for gen in range(effective_gens):
                gens_used = gen + 1

                # Budget check: stop beam search if eval budget exceeded
                if not _budget_ok():
                    logger.debug(f"  [wake] Beam budget exceeded at gen {gen}, {n_evals} evals")
                    break

                # Evaluate every candidate on all training examples
                scored = []
                for prog in beam:
                    sp = self._evaluate_program(prog, task)
                    n_evals += 1
                    scored.append(sp)
                    self._update_pareto_front(pareto, sp)

                    # Track global best
                    if best_so_far is None or sp.energy < best_so_far.energy:
                        best_so_far = sp

                # Early stopping on perfect solve
                if best_so_far and best_so_far.energy <= cfg.early_stop_energy:
                    logger.info(f"  [wake] Task {task.task_id}: perfect solve at gen {gen}")
                    break

                # Semantic dedup: remove programs with identical output vectors
                if cfg.semantic_dedup:
                    scored, n_removed = self._semantic_dedup(scored, task)
                    total_deduped += n_removed

                # Keep the best
                scored.sort(key=lambda s: s.energy)
                survivors = [s.program for s in scored[: cfg.beam_width]]

                # Produce next generation
                next_gen = list(survivors)  # elitism: survivors carry over

                # Mutations (biased by transition matrix when available)
                tm = self._transition_matrix if self._transition_matrix.size > 0 else None
                for prog in survivors:
                    for _ in range(cfg.mutations_per_candidate):
                        mutant = self.grammar.mutate(prog, all_prims, transition_matrix=tm)
                        next_gen.append(mutant)

                # Crossovers
                n_cross = int(len(survivors) * cfg.crossover_fraction)
                for _ in range(n_cross):
                    a = self._rng.choice(survivors)
                    b = self._rng.choice(survivors)
                    child = self.grammar.crossover(a, b)
                    next_gen.append(child)

                beam = next_gen

        logger.debug(f"  [wake] Phase 2 beam search: {time.time()-t_phase:.2f}s, gens={gens_used}")

        # --- Phase 3: Post-beam color fix ---
        t_phase = time.time()
        # Try color remapping on beam results. Skip Phase 3a near-miss
        # refinement — it's expensive and Phase 1.5 already covered enum
        # near-misses. Only run color fix on beam candidates.
        if best_so_far and best_so_far.prediction_error > cfg.solve_threshold:
            # Build pool from beam + enumeration for color fix only
            all_candidates = list(enum_candidates)
            if scored:
                all_candidates.extend(scored)

            # Phase 3: Color fix on all candidates
            color_fixed = self._try_color_fix(all_candidates, task)
            if color_fixed is not None:
                n_evals += 1
                self._update_pareto_front(pareto, color_fixed)
                if color_fixed.energy < best_so_far.energy:
                    best_so_far = color_fixed

        logger.debug(f"  [wake] Phase 3 post-beam: {time.time()-t_phase:.2f}s")

        # Record the episode and store solution if solved
        solved = best_so_far is not None and best_so_far.prediction_error <= self.search_cfg.solve_threshold

        # Include beam search results in the candidate pool for top-k
        if scored:
            enum_candidates.extend(scored)

        if solved:
            return _make_solved_result("beam search", gens=gens_used, deduped=total_deduped)

        # Not solved — record best effort
        if best_so_far:
            best_so_far.task_id = task.task_id
            if record:
                self.memory.record_episode(
                    task.task_id,
                    task.train_examples,
                    best_so_far.program,
                    best_so_far.energy,
                )

        front = self._extract_pareto_front(pareto)
        wall = time.time() - t0
        train_preds, test_preds = self._compute_predictions(best_so_far, task)
        logger.info(
            f"  [wake] Task {task.task_id}: train_solved={solved}, "
            f"energy={best_so_far.energy:.6f}, gens={gens_used}, "
            f"evals={n_evals}, deduped={total_deduped}, "
            f"pareto={len(front)}, time={wall:.1f}s"
        )
        return WakeResult(
            task_id=task.task_id,
            train_solved=solved,
            best=best_so_far,
            generations_used=gens_used,
            evaluations=n_evals,
            wall_time=wall,
            pareto_front=front,
            dedup_count=total_deduped,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

    def _compute_predictions(
        self, best: Optional[ScoredProgram], task: Task
    ) -> tuple[Optional[list], Optional[list]]:
        """Compute predicted outputs for train and test inputs.

        Returns (train_predictions, test_predictions). Stored in results JSON
        so the visualizer doesn't need to re-execute programs (which may use
        dynamically created primitives unavailable after the run).
        """
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
    ) -> tuple[Optional[float], Optional[bool]]:
        """Evaluate the best program on held-out test examples.

        Returns (avg_test_error, test_solved) or (None, None) if no test data.
        """
        if best is None or not task.test_inputs or not task.test_outputs:
            return None, None
        if len(task.test_inputs) != len(task.test_outputs):
            return None, None

        total_error = 0.0
        n = len(task.test_inputs)
        for inp, expected in zip(task.test_inputs, task.test_outputs):
            try:
                predicted = self.env.execute(best.program, inp)
                err = self.drive.prediction_error(predicted, expected)
            except Exception:
                err = 1e6
            total_error += err

        avg_error = total_error / n if n > 0 else total_error
        test_solved = avg_error <= self.search_cfg.solve_threshold
        return avg_error, test_solved

    def _loocv_score(self, sp: ScoredProgram, task: Task) -> float:
        """Leave-one-out cross-validation score for a training-perfect candidate.

        For each training example, holds it out, re-prepares the grammar
        with the remaining N-1 examples (which re-learns any parameterized
        primitives), and checks if the program still produces correct output
        for the held-out example.  This catches programs whose learned
        components (e.g., color remapping, structural role primitives)
        overfit the full training set.

        Returns fraction of folds passed (1.0 = fully validated).
        """
        n = len(task.train_examples)
        if n < 2:
            return 1.0

        threshold = self.search_cfg.solve_threshold
        passed = 0

        for i in range(n):
            loo_examples = task.train_examples[:i] + task.train_examples[i + 1:]
            loo_task = Task(
                task_id=task.task_id,
                train_examples=loo_examples,
                test_inputs=task.test_inputs,
                test_outputs=task.test_outputs,
            )
            # Re-prepare grammar with N-1 examples — re-learns parameterized
            # primitives, which may now produce different mappings
            self.grammar.prepare_for_task(loo_task)

            # Evaluate on the held-out example
            held_inp, held_exp = task.train_examples[i]
            try:
                pred = self.env.execute(sp.program, held_inp)
                err = self.drive.prediction_error(pred, held_exp)
                if err <= threshold:
                    passed += 1
            except Exception:
                pass

        # Restore grammar state for the full task
        self.grammar.prepare_for_task(task)

        return passed / n

    def _evaluate_top_k_on_test(
        self, candidates: list[ScoredProgram], task: Task, top_k: int = 3
    ) -> tuple[Optional[ScoredProgram], Optional[float], Optional[bool], int, Optional[int]]:
        """Try top-k training-perfect candidates on test, return best.

        Collects all candidates with prediction_error <= threshold, deduplicates
        by program representation, and ranks by LOOCV score + program size.
        LOOCV catches overfitting: candidates whose learned components don't
        generalize to held-out training examples are demoted (not eliminated).

        Returns: (best_program, test_error, test_solved, n_train_perfect, solving_rank)
        """
        threshold = self.search_cfg.solve_threshold
        if not task.test_inputs or not task.test_outputs:
            return None, None, None, 0, None

        # Collect training-perfect candidates, deduplicate by program repr
        seen: set[str] = set()
        perfect: list[ScoredProgram] = []
        for sp in candidates:
            if sp.prediction_error <= threshold:
                key = repr(sp.program)
                if key not in seen:
                    seen.add(key)
                    perfect.append(sp)

        if not perfect:
            return None, None, None, 0, None

        n_train_perfect = len(perfect)

        # LOOCV validation: score top candidates to detect overfitting.
        # Only run when we have multiple candidates (the payoff scenario)
        # and enough training examples for meaningful cross-validation.
        loocv_scores: dict[str, float] = {}
        n_to_validate = min(len(perfect), top_k)
        if len(task.train_examples) >= 2 and n_train_perfect > 1:
            # Pre-sort by size for LOOCV (validate simpler programs first)
            perfect.sort(key=lambda sp: (sp.program.size, sp.energy))
            for sp in perfect[:n_to_validate]:
                key = repr(sp.program)
                loocv_scores[key] = self._loocv_score(sp, task)

        # Ensemble agreement: when multiple training-perfect candidates exist,
        # compute test outputs for all and prefer candidates whose output
        # matches the consensus (majority vote). Zero additional search cost.
        agreement_scores: dict[str, int] = {}
        if n_train_perfect > 2 and task.test_inputs:
            test_output_strs: dict[str, str] = {}
            for sp in perfect[:min(len(perfect), top_k * 3)]:
                key = repr(sp.program)
                try:
                    outputs = []
                    for inp in task.test_inputs:
                        out = self.env.execute(sp.program, inp)
                        outputs.append(str(out))
                    test_output_strs[key] = "|".join(outputs)
                except Exception:
                    pass

            # Count how many candidates produce each output
            from collections import Counter
            output_counts = Counter(test_output_strs.values())
            for key, out_str in test_output_strs.items():
                agreement_scores[key] = output_counts[out_str]

        # Sort by: agreement (desc), LOOCV score (desc), program size (asc), energy (asc)
        # Ensemble agreement breaks ties: prefer the candidate most others agree with.
        # LOOCV-passing candidates are tried first; failures are demoted
        # but not eliminated (they still get a chance on test)
        perfect.sort(key=lambda sp: (
            -agreement_scores.get(repr(sp.program), 0),
            -loocv_scores.get(repr(sp.program), 1.0),
            sp.program.size,
            sp.energy,
        ))

        best_test_error = None
        best_test_sp = None
        solving_rank = None

        for rank, sp in enumerate(perfect[:top_k]):
            test_error, test_solved = self._evaluate_on_test(sp, task)
            if test_error is not None and (best_test_error is None or test_error < best_test_error):
                best_test_error = test_error
                best_test_sp = sp
            if test_solved:
                solving_rank = rank
                return sp, test_error, True, n_train_perfect, rank

        # No candidate passed test — return the one with best test error
        if best_test_sp is not None:
            return best_test_sp, best_test_error, False, n_train_perfect, None

        return perfect[0], None, None, n_train_perfect, None

    # -------------------------------------------------------------------------
    # SLEEP PHASE: analyze → extract → compress → add to library
    # -------------------------------------------------------------------------

    def sleep(self) -> SleepResult:
        """
        Consolidation phase — the "dream" step.

        1. Collect all solved programs
        2. Build transition matrix P(child_op | parent_op) from solutions
        3. Extract sub-trees that recur across multiple solutions
        4. Score by compression value with diversity bonus
        5. Add the best as new named primitives to the library
        6. Decay old entries and prune dead ones
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

        # Build a map of task_id → solution root op for diversity scoring
        task_roots = {tid: scored.program.root for tid, scored in solutions.items()}

        # 3. Filter: must appear in >= min_occurrences different tasks,
        #    must be >= min_size nodes (no trivial single-node entries)
        candidates = []
        for key, occurrences in subtree_counts.items():
            task_ids = sorted(set(tid for _, tid in occurrences))
            subtree = occurrences[0][0]
            if len(task_ids) >= cfg.min_occurrences and subtree.size >= cfg.min_size:
                # Diversity bonus: reward subtrees that appear across solutions
                # with different root operations (structurally diverse contexts).
                # A subtree used in rotate(crop(x)) AND fill(crop(x)) is more
                # general than one only in rotate(crop(x)) variants.
                unique_roots = len(set(task_roots.get(tid, "") for tid in task_ids))
                diversity_bonus = 1.0 + 0.5 * math.log(max(unique_roots, 1))

                # Usefulness = tasks_used × log(size+1) × diversity_bonus
                usefulness = len(task_ids) * math.log(subtree.size + 1) * diversity_bonus
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

        # 7. Prune dead entries: remove library entries that have decayed
        #    below threshold and were never reused. Prevents the library
        #    from filling with stale abstractions that crowd out better ones.
        pruned = self.memory.prune_library(min_usefulness=0.01)

        lib_after = len(self.memory.get_library())
        wall = time.time() - t0

        logger.info(
            f"  [sleep] Extracted {len(new_entries)} new abstractions, "
            f"pruned {pruned} dead entries. "
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
        else:
            # Seeded shuffle for unbiased progress metrics
            tasks = list(tasks)
            random.Random(self.search_cfg.seed or 42).shuffle(tasks)

        # Resolve worker count: 0 = performance cores (not all cores)
        if cfg.workers <= 0:
            cfg.workers = self.performance_core_count()

        results = []
        for round_num in range(cfg.wake_sleep_rounds):
            logger.info(f"=== Round {round_num + 1}/{cfg.wake_sleep_rounds} ===")
            logger.info(f"    Library size: {len(self.memory.get_library())}")

            if cfg.sequential_compounding:
                logger.info("    Mode: sequential compounding")
                wake_results = self._wake_sequential_compounding(
                    tasks, round_num + 1, on_task_done)
            else:
                logger.info(f"    Workers: {cfg.workers}")
                wake_results = self._wake_parallel(
                    tasks, cfg.workers, round_num + 1, on_task_done)

            # Adaptive compute reallocation: re-run near-misses with boosted budget
            if cfg.adaptive_realloc:
                wake_results = self._adaptive_realloc_pass(
                    tasks, wake_results, cfg, round_num + 1, on_task_done)

            # SLEEP: consolidate (sequential — mutates shared state)
            sleep_result = self.sleep()

            # Metrics
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

            logger.info(
                f"=== Round {round_num + 1} summary: "
                f"solved {rr.solved}/{total} ({rr.solve_rate:.1%}), "
                f"train_matched {train_solved}/{total} ({train_rate:.1%}), "
                f"library={rr.cumulative_library_size} ==="
            )

        return results

    def _adaptive_realloc_pass(
        self,
        tasks: list[Task],
        wake_results: list[WakeResult],
        cfg: CurriculumConfig,
        round_num: int,
        on_task_done: "Optional[callable]" = None,
    ) -> list[WakeResult]:
        """
        Re-run near-miss tasks with boosted search budget.

        Identifies tasks that are close to being solved (low error but not
        solved) and gives them more compute. This is a purely algorithmic
        improvement — no data leakage since it's based on training error only.

        Returns the updated wake_results list with improved results merged in.
        """
        near_miss_thresh = self.search_cfg.near_miss_threshold
        near_miss_indices = []
        for i, wr in enumerate(wake_results):
            err = wr.best.prediction_error if wr.best else 1.0
            if not wr.train_solved and 0 < err < near_miss_thresh:
                near_miss_indices.append(i)

        if not near_miss_indices:
            return wake_results

        logger.info(
            f"    Adaptive realloc: {len(near_miss_indices)} near-miss tasks "
            f"(err < {near_miss_thresh}), re-running with boosted budget")

        # Save original config and apply boost
        orig_cfg = self.search_cfg
        boosted = copy.copy(orig_cfg)
        boosted.eval_budget = int(orig_cfg.eval_budget * cfg.adaptive_realloc_budget_multiplier)
        boosted.exhaustive_pair_top_k = (
            orig_cfg.exhaustive_pair_top_k + cfg.adaptive_realloc_pair_top_k_boost)
        boosted.exhaustive_triple_top_k = (
            orig_cfg.exhaustive_triple_top_k + cfg.adaptive_realloc_triple_top_k_boost)
        self.search_cfg = boosted

        try:
            near_miss_tasks = [tasks[i] for i in near_miss_indices]
            boosted_results = self._wake_parallel(
                near_miss_tasks, cfg.workers, round_num)
        finally:
            self.search_cfg = orig_cfg

        # Merge: keep the better result for each near-miss task
        improved = 0
        wake_results = list(wake_results)  # copy to avoid mutating original
        for idx, boosted_wr in zip(near_miss_indices, boosted_results):
            orig_wr = wake_results[idx]
            orig_energy = orig_wr.best.energy if orig_wr.best else float('inf')
            boosted_energy = boosted_wr.best.energy if boosted_wr.best else float('inf')
            # Better = newly solved, or lower energy
            if (boosted_wr.train_solved and not orig_wr.train_solved) or \
               (boosted_energy < orig_energy):
                # Accumulate evaluations from both passes
                merged = WakeResult(
                    task_id=boosted_wr.task_id,
                    train_solved=boosted_wr.train_solved,
                    best=boosted_wr.best,
                    generations_used=orig_wr.generations_used + boosted_wr.generations_used,
                    evaluations=orig_wr.evaluations + boosted_wr.evaluations,
                    wall_time=orig_wr.wall_time + boosted_wr.wall_time,
                    pareto_front=boosted_wr.pareto_front,
                    dedup_count=orig_wr.dedup_count + boosted_wr.dedup_count,
                    test_error=boosted_wr.test_error,
                    test_solved=boosted_wr.test_solved,
                )
                wake_results[idx] = merged
                if boosted_wr.train_solved and not orig_wr.train_solved:
                    improved += 1

        logger.info(
            f"    Adaptive realloc done: {improved} tasks newly solved")

        return wake_results

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
        back into the main process's memory in deterministic (index) order.

        Falls back to sequential if workers=1 or if the task count is small.

        Ctrl-C handling: the pool is NOT used as a context manager. On
        KeyboardInterrupt, we call pool.shutdown(wait=False, cancel_futures=True)
        which kills workers immediately instead of blocking on __exit__.
        """
        total_tasks = len(tasks)
        base_seed = self.search_cfg.seed or 0

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
        transition_matrix = self._transition_matrix

        # Build worker args with per-task seeds for deterministic parallel execution.
        # Each task gets seed = hash(base_seed, round_num, task_index) so that:
        #   - Different tasks get different RNG streams
        #   - Same task always gets the same stream (deterministic)
        #   - Different rounds get different streams (library changes anyway)
        worker_args = []
        for i, task in enumerate(tasks):
            task_seed = hash((base_seed, round_num, i)) & 0x7FFFFFFF
            worker_args.append(
                (task, self.env, self.grammar, self.drive,
                 library_snapshot, search_cfg, transition_matrix, task_seed)
            )

        wake_results: list[WakeResult] = [None] * len(tasks)  # type: ignore
        completed_count = 0

        # Don't use ProcessPoolExecutor as context manager — its __exit__
        # calls shutdown(wait=True), which blocks on Ctrl-C even when workers
        # ignore SIGINT. Instead, manage manually and call shutdown(wait=False)
        # on interrupt for immediate cleanup.
        # Use "fork" context so workers inherit parent memory instead of
        # re-importing from disk. This makes the pipeline robust to file
        # edits during execution (workers use the code snapshot from when
        # the parent process started, not the current on-disk version).
        _fork_ctx = _mp.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            mp_context=_fork_ctx,
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
            # Kill workers immediately — don't wait for them to finish.
            # shutdown(wait=False, cancel_futures=True) only cancels pending
            # futures; running workers keep going as orphan processes.
            # Explicitly terminate all child processes for clean exit.
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
            # Fallback to sequential if multiprocessing fails
            logger.warning(f"Parallel wake failed ({e}), falling back to sequential")
            wake_results = []
            for i, task in enumerate(tasks):
                wr = self.wake_on_task(task)
                wake_results.append(wr)
                if on_task_done:
                    on_task_done(round_num, i + 1, total_tasks, wr)
            return wake_results

        # Merge results back into main memory in deterministic (index) order
        for wr in wake_results:
            if wr and wr.best:
                self.memory.record_episode(
                    wr.task_id, [], wr.best.program, wr.best.energy)
                if wr.train_solved:
                    self.memory.store_solution(wr.task_id, wr.best)
                    self._credit_library_usage(wr.best.program)

        return wake_results

    # -------------------------------------------------------------------------
    # Sequential compounding — within-run knowledge transfer
    # -------------------------------------------------------------------------

    def _wake_sequential_compounding(
        self,
        tasks: list[Task],
        round_num: int = 1,
        on_task_done: "Optional[callable]" = None,
    ) -> list[WakeResult]:
        """
        Process tasks one at a time with immediate concept promotion.

        After each solved task, extract subtrees and add promising ones
        to the library immediately. Next task sees the expanded vocabulary.
        This is where "one algorithm compounds" becomes measurable.
        """
        total_tasks = len(tasks)
        wake_results = []

        for i, task in enumerate(tasks):
            wr = self.wake_on_task(task)
            wake_results.append(wr)
            if on_task_done:
                on_task_done(round_num, i + 1, total_tasks, wr)

            # Immediate concept promotion on solve
            if wr.train_solved and wr.best:
                self._immediate_promote(wr.best, task.task_id)

        return wake_results

    def _immediate_promote(self, scored: ScoredProgram, task_id: str) -> None:
        """
        Immediately promote a solved program's subtrees to the library.

        Unlike full sleep (which waits for min_occurrences across multiple tasks),
        this promotes any non-trivial subtree from a single solve.
        """
        existing_reprs = {repr(e.program) for e in self.memory.get_library()}

        for subtree in self._enumerate_subtrees(scored.program):
            if subtree.size < 2:
                continue
            key = repr(subtree)
            if key in existing_reprs:
                continue
            if len(self.memory.get_library()) >= self.sleep_cfg.max_library_size:
                break

            entry = LibraryEntry(
                name=f"promoted_{len(self.memory.get_library())}",
                program=subtree,
                usefulness=math.log(subtree.size + 1),
                reuse_count=0,
                source_tasks=[task_id],
                domain="",
            )
            self.memory.add_to_library(entry)
            existing_reprs.add(key)

        logger.info(
            f"  [promote] Task {task_id}: library now has "
            f"{len(self.memory.get_library())} entries"
        )

    # -------------------------------------------------------------------------
    # Exhaustive enumeration — the bootstrap phase
    # -------------------------------------------------------------------------

    def _near_miss_refine(
        self,
        candidates: list[ScoredProgram],
        primitives: list[Primitive],
        task: Task,
        threshold: float = 0.20,
    ) -> tuple[list[ScoredProgram], int]:
        """
        Near-miss refinement: for programs scoring close-but-not-perfect,
        try appending or prepending each primitive to fix them.

        Adapted from agi-mvp-general's high-ROI refinement phase.
        Programs with prediction_error < threshold but > solve_threshold
        are "almost right" — often they just need one more step
        (e.g. a crop, color fix, or flip).

        Cost: O(near_misses × refine_prims × 2).
        Optimized: uses top-5 near-misses and top-50 primitives (by depth-1
        score) plus essential pair concepts, instead of all primitives.
        """
        solve_thresh = self.search_cfg.solve_threshold
        near_misses = [
            sp for sp in candidates
            if solve_thresh < sp.prediction_error <= threshold
        ]
        if not near_misses:
            return [], 0

        # Sort by quality (best near-misses first), limit to top-5
        near_misses.sort(key=lambda s: s.prediction_error)
        near_misses = near_misses[:5]

        # Use ALL unary primitives for near-miss refinement.
        # Cost: 5 near-misses × ~280 prims × 2 = ~2800 evals — still cheap.
        # Many fixes involve primitives ranked low individually (e.g. a specific
        # color fill, a rare symmetry op) but critical as the final correction.
        all_unary = [p for p in primitives if p.arity <= 1]
        unary_prims = all_unary
        refined: list[ScoredProgram] = []
        n_evals = 0

        for nm in near_misses:
            for prim in unary_prims:
                # Try appending: prim(near_miss_program)
                prog_append = Program(
                    root=prim.name,
                    children=[copy.deepcopy(nm.program)],
                )
                sp = self._evaluate_program(prog_append, task)
                refined.append(sp)
                n_evals += 1
                if sp.prediction_error <= solve_thresh:
                    return refined, n_evals

                # Try prepending: insert prim as the innermost step.
                # For f(g(x)), prepend h gives f(g(h(x))).
                prog_prepend = copy.deepcopy(nm.program)
                # Walk to the deepest leaf and wrap it: leaf → prim(leaf)
                node = prog_prepend
                while node.children:
                    node = node.children[0]
                # node is the deepest leaf — wrap: old_leaf → prim(old_leaf)
                old_root = node.root
                node.root = prim.name
                node.children = [Program(root=old_root)]
                sp = self._evaluate_program(prog_prepend, task)
                refined.append(sp)
                n_evals += 1
                if sp.prediction_error <= solve_thresh:
                    return refined, n_evals

        # --- Node replacement for depth-1+ near-misses ---
        # For programs with at least one composition (e.g. f(g(x))),
        # try replacing each internal node with a different primitive.
        # This catches "right structure, wrong step" cases.
        # Cost: O(near_misses × depth × n_prims) — limited to top-60 prims.
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
                continue  # nothing to replace in a single primitive
            # Collect all internal nodes
            nodes_to_replace: list[Program] = []

            def _collect(node: Program, parent: Optional[Program] = None):
                if parent is not None:  # skip root (that's append)
                    nodes_to_replace.append(node)
                for child in (node.children or []):
                    _collect(child, node)

            _collect(nm.program)
            for target_node in nodes_to_replace:
                original_root = target_node.root
                for prim in replace_prims:
                    if prim.name == original_root:
                        continue
                    # Replace in-place, eval, restore
                    target_node.root = prim.name
                    prog_replaced = copy.deepcopy(nm.program)
                    target_node.root = original_root  # restore
                    sp = self._evaluate_program(prog_replaced, task)
                    refined.append(sp)
                    n_evals += 1
                    if sp.prediction_error <= solve_thresh:
                        return refined, n_evals

        # --- Two-step near-miss refinement ---
        # For the closest near-misses (error < 0.10), try prim2(prim1(program)).
        # Strategy: collect top-10 single-step improvements per near-miss,
        # then apply a second step to each.
        # Cost: O(close_misses × top_improved × refine_prims) ≈ 5 × 10 × 50 = 2500
        TWO_STEP_THRESHOLD = 0.10
        close_misses = [sp for sp in refined if sp.prediction_error < TWO_STEP_THRESHOLD]
        if not close_misses:
            # Also check original near-misses that were already close
            close_misses = [nm for nm in near_misses if nm.prediction_error < TWO_STEP_THRESHOLD]
        if close_misses:
            close_misses.sort(key=lambda s: s.prediction_error)
            close_misses = close_misses[:5]
            for cm in close_misses:
                for prim in unary_prims:
                    prog_outer = Program(
                        root=prim.name,
                        children=[copy.deepcopy(cm.program)],
                    )
                    sp = self._evaluate_program(prog_outer, task)
                    refined.append(sp)
                    n_evals += 1
                    if sp.prediction_error <= solve_thresh:
                        return refined, n_evals

        return refined, n_evals

    def _try_conditional_search(
        self,
        predicates: list[tuple[str, callable]],
        candidates: list[ScoredProgram],
        primitives: list[Primitive],
        task: Task,
        top_k: int = 15,
    ) -> tuple[Optional[ScoredProgram], int]:
        """Search for conditional programs: if pred(input) then A else B.

        For each predicate:
          1. Partition training inputs into true/false groups
          2. Skip trivial predicates (all-true or all-false)
          3. Score top-K single primitives on each group independently
          4. Try best 5×5 combinations per group

        Cost: ~P × top_k (for per-group scoring) + P × 25 (for combos)
        where P = number of non-trivial predicates.
        """
        n_evals = 0
        solve_thresh = self.search_cfg.solve_threshold

        # Build candidate pool: depth-1 primitives + top depth-2 programs.
        # Depth-2 programs are wrapped as on-the-fly Primitive objects so
        # they can be used uniformly in the conditional scoring below.
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

        # Resolve to Primitive objects
        prim_map = {p.name: p for p in primitives}
        top_prims = [prim_map[n] for n in top_prims_names if n in prim_map]

        # Add top depth-2 programs as branch candidates
        depth2 = [sp for sp in candidates
                   if sp.program.depth == 2 and sp.program.children]
        depth2.sort(key=lambda s: s.prediction_error)
        depth2_added = 0
        DEPTH2_BRANCH_K = 8
        for sp in depth2:
            prog_repr = repr(sp.program)
            if prog_repr in seen:
                continue
            seen.add(prog_repr)
            # Wrap depth-2 program as a Primitive for uniform handling
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
            # Partition training examples by predicate
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

            # Skip trivial predicates (no branching)
            if not true_indices or not false_indices:
                continue

            # Score each candidate on each group
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

            # Sort by per-group error (lower is better)
            true_scores.sort(key=lambda x: x[0])
            false_scores.sort(key=lambda x: x[0])

            # Try best 5×5 combos
            best_true = [p for _, p in true_scores[:5]]
            best_false = [p for _, p in false_scores[:5]]

            for then_prim in best_true:
                for else_prim in best_false:
                    if then_prim.name == else_prim.name:
                        continue

                    # Create a conditional primitive on the fly
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

                    # Register as a primitive
                    cond_prim = Primitive(
                        name=cond_name, arity=1, fn=cond_fn, domain="arc")
                    prim_map[cond_name] = cond_prim
                    self.env.register_primitive(cond_prim)

                    prog = Program(root=cond_name)
                    sp = self._evaluate_program(prog, task)
                    n_evals += 1

                    if sp.prediction_error <= solve_thresh:
                        return sp, n_evals
                    if best_result is None or sp.energy < best_result.energy:
                        best_result = sp

        return best_result, n_evals

    def _try_color_fix(
        self,
        candidates: list[ScoredProgram],
        task: Task,
        threshold: float = 0.30,
    ) -> Optional[ScoredProgram]:
        """Try to fix near-miss programs by learning a correction.

        For each candidate with prediction error <= threshold, execute it on all
        training inputs, compare outputs to expected, and ask the environment
        to infer a correction (color remap, neighborhood patch, etc.).
        If found, compose the correction on top and evaluate the result.

        Returns the best color-fixed ScoredProgram, or None.
        """
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

        for nm in near_misses:
            # Execute program on all training inputs
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

            # Ask environment for a correction (with 5x5 fallback)
            correction = self.env.infer_output_correction(
                outputs, expected, try_5x5=True)
            if correction is None:
                continue

            # Compose: correction(original_program)
            # Skip identity base — correction(identity) = correction
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

            # Only accept if the correction actually reduces error
            if sp.prediction_error >= nm.prediction_error:
                continue

            if sp.prediction_error <= solve_thresh:
                return sp
            if best_fix is None or sp.energy < best_fix.energy:
                best_fix = sp

        return best_fix

    def _try_identity_correction(
        self,
        task: Task,
    ) -> Optional[ScoredProgram]:
        """Try learning the entire transformation as neighborhood correction
        on the identity function (input → output directly).

        For same-shape tasks, if the transformation can be described as local
        cellular automaton rules (3x3 or 5x5 neighborhoods), we don't need
        a base program at all — the correction IS the solution.

        Uses a higher rule cap (100) since identity-seeded corrections need
        to capture the full transformation, not just residual errors.
        Also tries 5x5+ neighborhoods for longer-range dependencies.

        LOOCV: When there are 3+ training examples, holds out each example
        and verifies the correction learned from the others still works.
        This prevents accepting corrections that memorize training patterns
        (e.g., large neighborhood patches) without generalizing.
        """
        # Check if all training examples are same-shape
        for inp, exp in task.train_examples:
            try:
                inp_arr = self._to_array(inp)
                exp_arr = self._to_array(exp)
                if inp_arr.shape != exp_arr.shape:
                    return None
            except (ValueError, TypeError):
                return None

        # Use inputs directly as "program outputs" (identity function)
        inputs = [inp for inp, _ in task.train_examples]
        expected = [exp for _, exp in task.train_examples]

        # Try correction with higher rule cap and 5x5+ fallback
        correction = self.env.infer_output_correction(
            inputs, expected, max_rules=100, try_5x5=True)
        if correction is None:
            return None

        # The correction IS the full program (no base program needed)
        sp = self._evaluate_program(correction, task)

        # Only accept if it actually solves on training
        if sp.prediction_error > self.search_cfg.solve_threshold:
            return None

        # LOOCV: hold out each training example and verify generalization.
        # Skip if only 2 examples (too few for meaningful holdout).
        if len(task.train_examples) >= 3:
            for i in range(len(task.train_examples)):
                loo_inputs = inputs[:i] + inputs[i+1:]
                loo_expected = expected[:i] + expected[i+1:]

                loo_correction = self.env.infer_output_correction(
                    loo_inputs, loo_expected, max_rules=100, try_5x5=True)
                if loo_correction is None:
                    return None

                # Test on held-out example
                try:
                    pred = self.env.execute(loo_correction, inputs[i])
                    err = self.drive.prediction_error(pred, expected[i])
                    if err > self.search_cfg.solve_threshold:
                        return None  # correction doesn't generalize
                except Exception:
                    return None

        return sp

    @staticmethod
    def _to_array(grid):
        """Convert a grid to a numpy array for shape comparison."""
        import numpy as np
        return np.array(grid, dtype=np.int32)

    def _try_fixed_point(
        self,
        candidates: list[ScoredProgram],
        task: Task,
        threshold: float = 0.20,
        max_iters: int = 20,
    ) -> Optional[ScoredProgram]:
        """Try applying near-miss programs repeatedly until stable (fixed point).

        Many ARC tasks require iterative application: fill propagation, pattern
        growth, color spreading. For each near-miss depth-1 program, we apply
        it repeatedly and check if the converged result matches the expected.
        """
        solve_thresh = self.search_cfg.solve_threshold
        near_misses = [
            sp for sp in candidates
            if solve_thresh < sp.prediction_error <= threshold
            and sp.program.depth == 1  # only single primitives for now
        ]
        if not near_misses:
            return None

        near_misses.sort(key=lambda s: s.prediction_error)
        near_misses = near_misses[:10]

        best_result: Optional[ScoredProgram] = None

        for nm in near_misses:
            prim_name = nm.program.root

            # Build a fixed-point version
            fp_name = f"iterate_{prim_name}"

            def _make_fp(name=prim_name, iters=max_iters):
                def fp_fn(grid):
                    current = grid
                    for _ in range(iters):
                        result = self.env.execute(Program(root=name), current)
                        if not isinstance(result, list) or not result:
                            return current
                        if result == current:
                            return current
                        current = result
                    return current
                return fp_fn

            fp_fn = _make_fp()
            fp_prim = Primitive(name=fp_name, arity=1, fn=fp_fn, domain="")
            self.env.register_primitive(fp_prim)

            prog = Program(root=fp_name)
            sp = self._evaluate_program(prog, task)

            if sp.prediction_error <= solve_thresh:
                return sp
            if sp.prediction_error < nm.prediction_error:
                # Fixed-point improved over single application
                if best_result is None or sp.energy < best_result.energy:
                    best_result = sp

        return best_result

    def _try_grammar_decomposition(
        self,
        primitives: list[Primitive],
        task: Task,
    ) -> Optional[ScoredProgram]:
        """Try solving a task via the grammar's decompose/recompose.

        For each decomposition strategy and each unary primitive:
        1. Decompose each training input into parts
        2. Apply the primitive to each part
        3. Recompose the transformed parts
        4. Check if the recomposed result matches the expected output

        This is the generic "map-over-parts" search that works for any
        domain implementing decompose/recompose on its Grammar.
        """
        solve_thresh = self.search_cfg.solve_threshold
        unary_prims = [p for p in primitives if p.arity <= 1]

        # Get decompositions from the first training input to determine strategies
        if not task.train_examples:
            return None
        first_inp = task.train_examples[0][0]
        strategies = self.grammar.decompose(first_inp, task)
        if not strategies:
            return None

        best_result: Optional[ScoredProgram] = None

        for strategy_template in strategies:
            for prim in unary_prims:
                all_match = True
                for inp, expected in task.train_examples:
                    decomps = self.grammar.decompose(inp, task)
                    # Find matching strategy
                    decomp = None
                    for d in decomps:
                        if d.strategy == strategy_template.strategy:
                            decomp = d
                            break
                    if decomp is None:
                        all_match = False
                        break

                    # Apply primitive to each part
                    transformed = []
                    for part in decomp.parts:
                        try:
                            result = prim.fn(part) if prim.arity == 1 else part
                            if not isinstance(result, list) or not result:
                                result = part
                            transformed.append(result)
                        except Exception:
                            transformed.append(part)

                    # Recompose
                    recomposed = self.grammar.recompose(decomp, transformed)
                    if recomposed != expected:
                        all_match = False
                        break

                if all_match:
                    # Register as a primitive and evaluate
                    decomp_name = f"decomp_{strategy_template.strategy}_{prim.name}"

                    def _make_decomp_fn(strategy_name, prim_fn, grammar, task_ref):
                        def fn(grid):
                            decomps = grammar.decompose(grid, task_ref)
                            for d in decomps:
                                if d.strategy == strategy_name:
                                    parts = []
                                    for part in d.parts:
                                        try:
                                            r = prim_fn(part)
                                            parts.append(r if isinstance(r, list) and r else part)
                                        except Exception:
                                            parts.append(part)
                                    return grammar.recompose(d, parts)
                            return grid
                        return fn

                    fn = _make_decomp_fn(
                        strategy_template.strategy, prim.fn, self.grammar, task)
                    decomp_prim = Primitive(
                        name=decomp_name, arity=1, fn=fn, domain="")
                    self.env.register_primitive(decomp_prim)

                    prog = Program(root=decomp_name)
                    sp = self._evaluate_program(prog, task)
                    if sp.prediction_error <= solve_thresh:
                        return sp
                    if best_result is None or sp.energy < best_result.energy:
                        best_result = sp

        return best_result

    def _exhaustive_enumerate(
        self,
        primitives: list[Primitive],
        task: Task,
        max_depth: int = 2,
        top_k: int = 15,
        eval_budget: int = 0,
    ) -> tuple[list[ScoredProgram], int]:
        """
        Enumerate ALL programs up to max_depth and evaluate them.

        Depth 1: try every single primitive (N programs).
        Depth 2: top-K singles + essential concepts → K² pair combos.
        Depth 3: top-K singles + essential concepts → K³ triple combos.

        Adapted from agi-mvp-general's try_all_pairs / try_all_triples:
        both steps in a pair (and all three in a triple) are drawn from
        the same pool of top-scoring + structurally essential concepts.
        This catches solutions where the first step scores low individually
        but is critical as a structural setup (e.g. crop, fill, compress).

        eval_budget: max depth-weighted ops (0 = unlimited). A depth-d
        evaluation costs d ops. Budget is checked between depth phases
        and within inner loops.

        Returns (scored_programs, num_evaluations).
        """
        scored: list[ScoredProgram] = []
        n_evals = 0
        solve_thresh = self.search_cfg.solve_threshold
        pair_top_k = self.search_cfg.exhaustive_pair_top_k
        triple_top_k = self.search_cfg.exhaustive_triple_top_k

        def _budget_ok() -> bool:
            return eval_budget <= 0 or n_evals < eval_budget

        # --- Depth 1: all single primitives (cost: 1 op each) ---
        # Always try ALL depth-1 prims (349 evals, essentially free) to collect
        # multiple training-perfect candidates. This combats overfitting:
        # if both erase_4 and erase_rare match training, top-k test evaluation
        # can select the one that generalizes to test.
        unary_prims = [p for p in primitives if p.arity <= 1]
        prim_by_name: dict[str, Primitive] = {p.name: p for p in unary_prims}
        depth1_solved = False
        for prim in unary_prims:
            prog = Program(root=prim.name)
            sp = self._evaluate_program(prog, task)
            scored.append(sp)
            n_evals += 1  # depth 1 = 1 op
            if sp.prediction_error <= solve_thresh:
                depth1_solved = True
        # If depth-1 found matches, return all candidates for top-k test
        # evaluation. Skip depth-2+ since depth-1 solutions are simpler
        # (Occam's razor) and we now have multiple candidates to choose from.
        if depth1_solved:
            return scored, n_evals

        if max_depth < 2 or not _budget_ok():
            return scored, n_evals

        # --- Build pair pool: top-K singles + essential concepts ---
        # Strategy: include top-scoring singles first, then add essentials
        # that aren't already included (sorted by their depth-1 score so
        # the most task-relevant essentials get priority).
        depth1_ranked = sorted(scored, key=lambda s: s.prediction_error)
        essential_names = self.grammar.essential_pair_concepts()

        # Build a lookup: name → depth-1 score
        depth1_scores: dict[str, float] = {}
        for sp in depth1_ranked:
            if sp.program.root not in depth1_scores:
                depth1_scores[sp.program.root] = sp.prediction_error

        # Phase 1: top-scoring singles (up to 60% of pool)
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

        # Phase 2: essentials not already included, sorted by depth-1
        # score (most task-relevant first)
        remaining_essentials = [
            n for n in essential_names
            if n not in seen_names and n in prim_by_name
        ]
        remaining_essentials.sort(
            key=lambda n: depth1_scores.get(n, 1.0))
        for name in remaining_essentials:
            if len(pair_pool) >= pair_top_k:
                break
            pair_pool.append(name)
            seen_names.add(name)

        # Phase 3: fill any remaining slots with more top-scorers
        for sp in depth1_ranked:
            if len(pair_pool) >= pair_top_k:
                break
            name = sp.program.root
            if name not in seen_names:
                pair_pool.append(name)
                seen_names.add(name)

        # --- Smart pruning for depth-2: filter inner steps by quality ---
        # Inner steps with very high error (>0.7) rarely help in composition.
        # Keep the full pair_pool for outer steps (they transform results)
        # but restrict inner steps to those that produce useful intermediate results.
        INNER_STEP_THRESHOLD = 0.70
        inner_pool = [
            name for name in pair_pool
            if depth1_scores.get(name, 1.0) <= INNER_STEP_THRESHOLD
        ]
        # Fallback: if too few pass threshold, use top half by score
        if len(inner_pool) < pair_top_k // 3:
            inner_pool = pair_pool[:pair_top_k // 2]

        # --- Depth 2: smart K × K' pairs (cost: 2 ops each) ---
        for outer_name in pair_pool:
            if not _budget_ok():
                break
            if outer_name == "identity":
                continue  # identity(x) = x, already tested at depth 1
            for inner_name in inner_pool:
                if not _budget_ok():
                    break
                if inner_name == "identity":
                    continue  # f(identity) = f, already tested at depth 1
                prog = Program(root=outer_name, children=[
                    Program(root=inner_name)])
                sp = self._evaluate_program(prog, task)
                scored.append(sp)
                n_evals += 2  # depth 2 = 2 primitive applications
                if sp.prediction_error <= solve_thresh:
                    return scored, n_evals

        # --- Depth 2.5: Overlay (binary) composition ---
        # Try overlay(prog_a, prog_b) for top-scoring depth-1 programs.
        # Many ARC tasks require combining two independent transforms
        # (e.g. background pattern + foreground objects).
        # Cost: O(K²) where K = overlay_top_k (~15), so ~225 evals.
        binary_prims = [p for p in primitives if p.arity == 2]
        if binary_prims and _budget_ok():
            OVERLAY_TOP_K = 15
            overlay_pool = pair_pool[:OVERLAY_TOP_K]
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
                        n_evals += 2  # 2 child evals + 1 combine
                        if sp.prediction_error <= solve_thresh:
                            return scored, n_evals

        if max_depth < 3 or not _budget_ok():
            return scored, n_evals

        # --- Adaptive depth skip: if depth-2 best is poor, skip depth-3 ---
        # Depth-3 rarely helps when depth-2 can't find anything close.
        DEPTH3_SKIP_THRESHOLD = 0.50
        depth2_best = min(
            (s.prediction_error for s in scored if s.program.children),
            default=1.0)
        if depth2_best > DEPTH3_SKIP_THRESHOLD:
            return scored, n_evals

        # --- Build triple pool: error-guided, from top depth-2 results ---
        # Use primitives that worked well in depth-2 compositions.
        depth2_ranked = sorted(
            [s for s in scored if s.program.children],
            key=lambda s: s.prediction_error)
        triple_seen: set[str] = set()
        triple_pool: list[str] = []

        # Phase 1: extract primitives from top depth-2 programs
        depth2_cap = triple_top_k // 3
        for sp in depth2_ranked:
            if len(triple_pool) >= depth2_cap:
                break
            for name in [sp.program.root] + [
                    c.root for c in (sp.program.children or [])]:
                if name not in triple_seen and name in prim_by_name:
                    triple_pool.append(name)
                    triple_seen.add(name)

        # Phase 2: essentials
        triple_essential_cap = triple_top_k * 2 // 3
        for name in essential_names:
            if len(triple_pool) >= triple_essential_cap:
                break
            if name not in triple_seen and name in prim_by_name:
                triple_pool.append(name)
                triple_seen.add(name)

        # Phase 3: top singles to fill remaining slots
        for sp in depth1_ranked:
            name = sp.program.root
            if name not in triple_seen:
                triple_pool.append(name)
                triple_seen.add(name)
            if len(triple_pool) >= triple_top_k:
                break

        # --- Depth 3: exhaustive K³ triples (cost: 3 ops each) ---
        for a in triple_pool:
            if not _budget_ok():
                break
            if a == "identity":
                continue  # identity(b(c)) = b(c), already tested at depth 2
            for b in triple_pool:
                if not _budget_ok():
                    break
                if b == "identity":
                    continue  # a(identity(c)) = a(c), already tested at depth 2
                for c in triple_pool:
                    if not _budget_ok():
                        break
                    if c == "identity":
                        continue  # a(b(identity)) = a(b), already tested at depth 2
                    # Skip degenerate a(a(a(x))) — already tested as single
                    if a == b == c:
                        continue
                    prog = Program(root=a, children=[
                        Program(root=b, children=[
                            Program(root=c)])])
                    sp = self._evaluate_program(prog, task)
                    scored.append(sp)
                    n_evals += 3  # depth 3 = 3 primitive applications
                    if sp.prediction_error <= solve_thresh:
                        return scored, n_evals

        return scored, n_evals

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _avg_cells(task: Task) -> int:
        """Max cell count across training input grids.

        Uses max (not average) because the most expensive input determines
        the per-evaluation cost — primitives must process every input.
        """
        grids = [inp for inp, _ in task.train_examples]
        if not grids:
            return 1
        sizes = []
        for g in grids:
            try:
                if g and len(g) > 0 and len(g[0]) > 0:
                    sizes.append(len(g) * len(g[0]))
            except TypeError:
                continue  # non-grid input (e.g. scalar)
        return max(sizes) if sizes else 1

    def _init_beam(self, primitives: list[Primitive], n: int) -> list[Program]:
        """
        Generate n random programs of varying depth (1-3).

        Uses the transition matrix to bias toward known-good compositions
        when available. Falls back to uniform random when no prior exists.
        """
        beam = []
        use_prior = self._transition_matrix.size > 0

        for i in range(n):
            # Vary depth: 20% depth-1, 35% depth-2, 30% depth-3, 15% depth-4
            r = self._rng.random()
            max_depth = 1 if r < 0.2 else (2 if r < 0.55 else (3 if r < 0.85 else 4))
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
        """Evaluate a program on all training examples, return scored result.

        Uses max-error blending to penalize programs that match some examples
        but fail on others.  A program that perfectly matches 2/3 examples
        but fails on the third (avg=0.33, max=1.0) scores worse than one
        that partially matches all three (avg=0.33, max=0.4).  This reduces
        overfitting when tasks have few training examples.
        """
        total_error = 0.0
        max_error = 0.0
        n = len(task.train_examples)

        for inp, expected in task.train_examples:
            try:
                predicted = self.env.execute(program, inp)
                err = self.drive.prediction_error(predicted, expected)
            except Exception:
                err = 1e6  # penalty for programs that crash
            total_error += err
            max_error = max(max_error, err)

        avg_error = total_error / n if n > 0 else total_error
        # Blend average and max: penalizes inconsistent programs
        effective_error = max(avg_error, max_error * 0.3)
        comp_cost = self.drive.complexity_cost(program)
        energy = self.search_cfg.energy_alpha * effective_error + self.search_cfg.energy_beta * comp_cost

        return ScoredProgram(
            program=program,
            energy=energy,
            prediction_error=avg_error,
            complexity_cost=comp_cost,
        )

    def _semantic_hash(self, program: Program, task: Task) -> str:
        """Hash a program by its outputs on training inputs.

        Two programs that produce identical output vectors are semantically
        equivalent (e.g. cos(π/2 + x²) and sin(x²)).  For numeric outputs,
        rounds to `dedup_precision` decimal places. For grid outputs (list
        of lists), uses tuple representation for exact comparison.
        """
        precision = self.search_cfg.dedup_precision
        outputs = []
        for inp, _ in task.train_examples:
            try:
                val = self.env.execute(program, inp)
                if isinstance(val, (int, float)):
                    outputs.append(round(float(val), precision))
                elif isinstance(val, list):
                    # Grid output: convert to nested tuples for hashing
                    outputs.append(tuple(tuple(row) for row in val) if val and isinstance(val[0], list) else tuple(val))
                else:
                    outputs.append(val)
            except Exception:
                outputs.append(None)
        return str(outputs)

    def _semantic_dedup(self, scored: list[ScoredProgram],
                        task: Task) -> tuple[list[ScoredProgram], int]:
        """Remove semantically duplicate programs from the scored list.

        Keeps the lowest-energy program for each unique output vector.
        Returns (deduplicated list, number removed).
        """
        seen: dict[str, ScoredProgram] = {}
        for sp in scored:
            key = self._semantic_hash(sp.program, task)
            if key not in seen or sp.energy < seen[key].energy:
                seen[key] = sp
        deduped = sorted(seen.values(), key=lambda s: s.energy)
        return deduped, len(scored) - len(deduped)

    def _update_pareto_front(self, pareto: dict[int, ParetoEntry],
                             sp: ScoredProgram) -> None:
        """Update the Pareto front with a scored program if it improves
        the best-known error at its complexity level."""
        c = sp.program.size
        if c not in pareto or sp.prediction_error < pareto[c].prediction_error:
            pareto[c] = ParetoEntry(
                complexity=c,
                prediction_error=sp.prediction_error,
                energy=sp.energy,
                program=sp.program,
            )

    def _extract_pareto_front(self, pareto: dict[int, ParetoEntry]) -> list[ParetoEntry]:
        """Return the true Pareto front: entries where no other entry has
        both lower complexity AND lower error."""
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
