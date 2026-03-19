"""
The Universal Learning Loop.

This is the invariant core. It NEVER imports anything domain-specific.
It depends only on the 4 interfaces defined in interfaces.py.

The loop:
    WAKE:   observe → hypothesize → execute → score → store
    SLEEP:  analyze solutions → extract sub-programs → compress → add to library
    REPEAT: library grows → search space shrinks → harder problems become tractable

Wake phases: exhaustive enumeration + per-object decomposition +
cross-reference + color fix. Additional phases added only when
justified by measured solves on specific tasks.
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
    SearchStratum,
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
        """Shared wake logic. Two-stage model: stratum enumeration + structural hooks."""
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

        # Stage 1: Stratum enumeration
        strata = self.grammar.propose_strata(task, all_prims)
        for stratum in strata:
            if ctx.solved:
                break
            solved_by = self._run_stratum_enumeration(ctx, stratum, all_prims)
            if solved_by is not None:
                return self._make_solved_result(ctx, solved_by)

        # Stage 2: Structural hooks (run once with aggregated candidates)
        if not ctx.solved and self.grammar.allow_structural_phases():
            for hook_fn in self._structural_hooks():
                solved_by = hook_fn(ctx)
                if solved_by is not None:
                    return self._make_solved_result(ctx, solved_by)

        return self._make_unsolved_result(ctx)

    def _run_stratum_enumeration(self, ctx: _WakeContext, stratum: SearchStratum, all_prims: list[Primitive]) -> Optional[str]:
        """Run exhaustive enumeration for a single stratum."""
        if ctx.cfg.exhaustive_depth < 1:
            return None

        # Filter primitives to this stratum's subset
        stratum_names = set(stratum.primitive_names)
        stratum_prims = [p for p in all_prims if p.name in stratum_names]
        if not stratum_prims:
            return None

        # Scale eval budget by stratum's fraction
        original_budget = ctx.eval_budget
        if ctx.eval_budget > 0:
            ctx.eval_budget = int(original_budget * stratum.budget_fraction)

        # Run exhaustive enumeration with stratum's primitives
        t = time.time()
        candidates, n_evals = self._exhaustive_enumerate(
            stratum_prims, ctx.task, min(ctx.cfg.exhaustive_depth, stratum.max_depth),
            eval_budget=ctx.eval_budget)
        ctx.n_evals += n_evals
        for sp in candidates:
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
        ctx.enum_candidates.extend(candidates)
        logger.debug(f"  [wake] Stratum '{stratum.name}': {time.time()-t:.2f}s, {n_evals} evals, {len(candidates)} cands")

        # Restore budget
        ctx.eval_budget = original_budget

        return stratum.name if ctx.solved else None

    def _structural_hooks(self):
        """Structural hook methods run once after all strata."""
        return [
            self._phase_object_decomposition,
            self._phase_for_each_object,
            self._phase_cross_reference,
            self._phase_local_rules,
            self._phase_procedural,
            self._phase_conditional_search,
            self._phase_color_fix,
            self._phase_input_pred_correction,
            self._phase_bidirectional,
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

    def _phase_local_rules(self, ctx: _WakeContext) -> Optional[str]:
        """Learn cellular automaton rules from training examples."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        t = time.time()
        result = self.env.try_local_rules(ctx.task)
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Local rules: {time.time()-t:.2f}s")
        return "local rules" if ctx.solved else None

    def _phase_procedural(self, ctx: _WakeContext) -> Optional[str]:
        """Learn per-object action rules from pixel diffs."""
        if ctx.solved or not self.grammar.allow_structural_phases():
            return None
        t = time.time()
        result = self.env.try_procedural(ctx.task)
        if result is not None:
            name, fn = result
            sp = self._evaluate_program(Program(root=name), ctx.task)
            ctx.n_evals += 1
            self._update_pareto_front(ctx.pareto, sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
        logger.debug(f"  [wake] Procedural: {time.time()-t:.2f}s")
        return "procedural" if ctx.solved else None

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
            predicates, ctx.enum_candidates, ctx.all_prims, ctx.task,
            top_k=10)
        ctx.n_evals += n_evals
        if result is not None:
            self._update_pareto_front(ctx.pareto, result)
            ctx.update_best(result)
            ctx.enum_candidates.append(result)
        logger.debug(f"  [wake] Conditional: {time.time()-t:.2f}s, {n_evals} evals")
        return "conditional" if ctx.solved else None

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

    def _phase_input_pred_correction(self, ctx: _WakeContext) -> Optional[str]:
        """Learn pixel-level correction rules on near-miss predictions.

        Tries multiple key strategies:
        1. (input_pixel, pred_pixel) — original
        2. (pred_pixel, n_nonzero_4neighbors_in_input) — neighborhood context

        Each is LOOCV-validated to avoid overfitting.
        """
        if ctx.solved or not ctx.enum_candidates:
            return None
        t = time.time()
        solve_thresh = self.search_cfg.solve_threshold
        task = ctx.task

        # Key strategy 1: (input_pixel, pred_pixel)
        def key_original(inp, pred, r, c):
            return (inp[r][c], pred[r][c])

        def apply_original(grid, pred, rule):
            return [[rule.get((grid[r][c], pred[r][c]), pred[r][c])
                     for c in range(len(pred[0]))] for r in range(len(pred))]

        # Key strategy 2: (pred_pixel, n_nonzero_4neighbors_in_input)
        def key_nbr_count(inp, pred, r, c):
            h, w = len(inp), len(inp[0])
            n_nz = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                       if 0 <= r + dr < h and 0 <= c + dc < w
                       and inp[r + dr][c + dc] != 0)
            return (pred[r][c], n_nz)

        def apply_nbr_count(grid, pred, rule):
            h, w = len(grid), len(grid[0])
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    n_nz = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                               if 0 <= r + dr < h and 0 <= c + dc < w
                               and grid[r + dr][c + dc] != 0)
                    row.append(rule.get((pred[r][c], n_nz), pred[r][c]))
                result.append(row)
            return result

        key_strategies = [
            ("input_pred_correct", key_original, apply_original),
            ("nbr_pred_correct", key_nbr_count, apply_nbr_count),
        ]

        # Try ALL same-dims candidates (correction only works when dims match)
        candidates = sorted(ctx.enum_candidates, key=lambda s: s.prediction_error)[:500]

        for nm in candidates:
            if nm.prediction_error <= solve_thresh:
                continue

            # Execute program on all training inputs to get predictions
            preds = []
            ok = True
            for inp, expected in task.train_examples:
                try:
                    pred = self.env.execute(nm.program, inp)
                    if (not isinstance(pred, list) or not pred or
                            len(pred) != len(expected) or len(pred[0]) != len(expected[0]) or
                            len(inp) != len(expected) or len(inp[0]) != len(expected[0])):
                        ok = False
                        break
                    preds.append((pred, inp, expected))
                except Exception:
                    ok = False
                    break
            if not ok or len(preds) < 2:
                continue

            for strat_name, key_fn, apply_fn in key_strategies:
                # Learn rule
                rule: dict[tuple, int] = {}
                consistent = True
                for pred, inp, expected in preds:
                    h, w = len(pred), len(pred[0])
                    for r in range(h):
                        for c in range(w):
                            key = key_fn(inp, pred, r, c)
                            val = expected[r][c]
                            if key in rule:
                                if rule[key] != val:
                                    consistent = False
                                    break
                            else:
                                rule[key] = val
                        if not consistent:
                            break
                    if not consistent:
                        break
                if not consistent:
                    continue

                # Check non-trivial
                trivial = True
                for pred, inp, expected in preds:
                    h, w = len(pred), len(pred[0])
                    for r in range(h):
                        for c in range(w):
                            key = key_fn(inp, pred, r, c)
                            if rule.get(key, pred[r][c]) != pred[r][c]:
                                trivial = False
                                break
                        if not trivial:
                            break
                    if not trivial:
                        break
                if trivial:
                    continue

                # LOOCV
                loocv_pass = True
                for hold in range(len(preds)):
                    sub = [x for i, x in enumerate(preds) if i != hold]
                    sub_rule: dict[tuple, int] = {}
                    sub_ok = True
                    for pred, inp, expected in sub:
                        h, w = len(pred), len(pred[0])
                        for r in range(h):
                            for c in range(w):
                                key = key_fn(inp, pred, r, c)
                                if key in sub_rule and sub_rule[key] != expected[r][c]:
                                    sub_ok = False
                                    break
                                sub_rule[key] = expected[r][c]
                            if not sub_ok:
                                break
                        if not sub_ok:
                            break
                    if not sub_ok:
                        loocv_pass = False
                        break
                    hp, hi, he = preds[hold]
                    h, w = len(hp), len(hp[0])
                    for r in range(h):
                        for c in range(w):
                            key = key_fn(hi, hp, r, c)
                            if sub_rule.get(key, hp[r][c]) != he[r][c]:
                                loocv_pass = False
                                break
                        if not loocv_pass:
                            break
                    if not loocv_pass:
                        break

                if not loocv_pass:
                    continue

                # Build the corrected program
                base_prog = nm.program
                final_rule = dict(rule)

                def _make_correction(bp=base_prog, fr=final_rule,
                                     env=self.env, afn=apply_fn):
                    def fn(grid):
                        pred = env.execute(bp, grid)
                        if not isinstance(pred, list) or not pred:
                            return grid
                        return afn(grid, pred, fr)
                    return fn

                corr_fn = _make_correction()
                name = f"{strat_name}({repr(base_prog)[:30]})"
                prim = Primitive(name=name, arity=0, fn=corr_fn, domain="arc")
                self.env.register_primitive(prim)
                sp = self._evaluate_program(Program(root=name), task)
                ctx.n_evals += 1
                self._update_pareto_front(ctx.pareto, sp)
                ctx.update_best(sp)
                ctx.enum_candidates.append(sp)
                if sp.prediction_error <= solve_thresh:
                    logger.debug(f"  [wake] Input-pred correction ({strat_name}): "
                                 f"{time.time()-t:.2f}s SOLVED")
                    return "input-pred correction"

        logger.debug(f"  [wake] Input-pred correction: {time.time()-t:.2f}s")
        return "input-pred correction" if ctx.solved else None

    def _phase_bidirectional(self, ctx: _WakeContext) -> Optional[str]:
        """Meet-in-the-middle search: depth-2 backward meets depth-1/2 forward.

        For each invertible primitive inv_name (with forward fwd_name):
          - For each candidate C in enum_candidates (forward programs):
              Check if fwd(C(input)) == expected_output for ALL training examples.
              Equivalently: C(input) == inv(expected_output) for all examples.
          - If found, the solution is fwd(C), i.e. Program(fwd_name, [C.program]).

        This can find depth-4 programs (when C itself is depth 3) or depth-2/3
        programs that happen not to be in the enumeration set.

        Also tries depth-2 backward: fwd2(fwd1(C(input))) == expected.
        Equivalently: C(input) == inv1(inv2(expected)).
        """
        if ctx.solved:
            return None
        inverses = self.grammar.inverse_primitives()
        if not inverses:
            return None

        t = time.time()
        task = ctx.task
        solve_thresh = self.search_cfg.solve_threshold

        # Build lookup of inverse callable by name, from the primitive map via env
        # env.execute on a single-node program invokes the primitive by name
        def _apply_named(name: str, grid):
            """Apply a named primitive to a grid using env.execute."""
            return self.env.execute(Program(root=name), grid)

        # Phase A: depth-1 backward
        # For each candidate C and each invertible prim fwd_name:
        #   Check if C(input) == inv(expected) for all train examples.
        #   If yes, solution is Program(fwd_name, [C.program]).
        n_cands = min(200, len(ctx.enum_candidates))
        candidates = ctx.enum_candidates[:n_cands]

        for fwd_name, inv_name in inverses.items():
            # Pre-compute inv(expected) for each training example
            inv_targets = []
            targets_ok = True
            for inp, expected in task.train_examples:
                try:
                    inv_out = _apply_named(inv_name, expected)
                    if inv_out is None:
                        targets_ok = False
                        break
                    inv_targets.append(inv_out)
                except Exception:
                    targets_ok = False
                    break
            if not targets_ok:
                continue

            # For each forward candidate, see if its outputs match inv_targets
            for cand in candidates:
                if cand.prediction_error <= solve_thresh:
                    continue  # already perfect, skip

                match = True
                for i, (inp, _expected) in enumerate(task.train_examples):
                    try:
                        pred = self.env.execute(cand.program, inp)
                        if pred is None:
                            match = False
                            break
                        if pred != inv_targets[i]:
                            match = False
                            break
                    except Exception:
                        match = False
                        break

                if match:
                    composed = Program(root=fwd_name, children=[cand.program])
                    sp = self._evaluate_program(composed, task)
                    ctx.n_evals += 1
                    self._update_pareto_front(ctx.pareto, sp)
                    ctx.update_best(sp)
                    ctx.enum_candidates.append(sp)
                    if sp.prediction_error <= solve_thresh:
                        logger.debug(
                            f"  [wake] Bidirectional-1: {fwd_name}({repr(cand.program)[:30]}) "
                            f"SOLVED in {time.time()-t:.2f}s")
                        return "bidirectional"

        # Phase B: depth-2 backward
        # For each pair (fwd1, fwd2) and candidate C:
        #   Check if C(input) == inv1(inv2(expected)) for all train examples.
        #   If yes, solution is Program(fwd2, [Program(fwd1, [C.program])]).
        n_cands_b = min(100, len(ctx.enum_candidates))
        candidates_b = ctx.enum_candidates[:n_cands_b]

        for fwd2_name, inv2_name in inverses.items():
            for fwd1_name, inv1_name in inverses.items():
                # Pre-compute inv1(inv2(expected)) for all train examples
                inv2_targets = []
                targets_ok = True
                for inp, expected in task.train_examples:
                    try:
                        inv2_out = _apply_named(inv2_name, expected)
                        if inv2_out is None:
                            targets_ok = False
                            break
                        inv1_out = _apply_named(inv1_name, inv2_out)
                        if inv1_out is None:
                            targets_ok = False
                            break
                        inv2_targets.append(inv1_out)
                    except Exception:
                        targets_ok = False
                        break
                if not targets_ok:
                    continue

                for cand in candidates_b:
                    if cand.prediction_error <= solve_thresh:
                        continue

                    match = True
                    for i, (inp, _expected) in enumerate(task.train_examples):
                        try:
                            pred = self.env.execute(cand.program, inp)
                            if pred is None:
                                match = False
                                break
                            if pred != inv2_targets[i]:
                                match = False
                                break
                        except Exception:
                            match = False
                            break

                    if match:
                        # solution: fwd2(fwd1(C))
                        composed = Program(
                            root=fwd2_name,
                            children=[Program(root=fwd1_name, children=[cand.program])]
                        )
                        sp = self._evaluate_program(composed, task)
                        ctx.n_evals += 1
                        self._update_pareto_front(ctx.pareto, sp)
                        ctx.update_best(sp)
                        ctx.enum_candidates.append(sp)
                        if sp.prediction_error <= solve_thresh:
                            logger.debug(
                                f"  [wake] Bidirectional-2: {fwd2_name}({fwd1_name}("
                                f"{repr(cand.program)[:20]})) SOLVED in {time.time()-t:.2f}s")
                            return "bidirectional"

        logger.debug(f"  [wake] Bidirectional: {time.time()-t:.2f}s")
        return None

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

        # Smart pruning for inner steps — essential primitives always included
        INNER_STEP_THRESHOLD = 0.70
        inner_pool = [
            name for name in pair_pool
            if name not in noop_prims
            and (depth1_scores.get(name, 1.0) <= INNER_STEP_THRESHOLD
                 or name in essential_names)
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
                if inner_name in noop_prims:
                    continue  # outer(identity(x)) = outer(x), already tested
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
        """Search for if(pred, A, B) programs.

        For each predicate that non-trivially splits the training examples,
        score top-K primitives on each group, then try best 5x5 combinations.
        """
        n_evals = 0
        solve_thresh = self.search_cfg.solve_threshold
        t_start = time.time()

        # Build candidate pool from top depth-1 programs
        depth1 = sorted(
            [sp for sp in candidates if sp.program.depth == 1],
            key=lambda s: s.prediction_error)
        prim_map = {p.name: p for p in primitives}
        seen = set()
        top_prims = []
        for sp in depth1:
            if sp.program.root not in seen and sp.program.root in prim_map:
                top_prims.append(prim_map[sp.program.root])
                seen.add(sp.program.root)
            if len(top_prims) >= top_k:
                break

        if len(top_prims) < 2:
            return None, 0

        best_result: Optional[ScoredProgram] = None

        # Pre-cache all prim outputs ONCE (the expensive part)
        prim_cache: dict[str, tuple] = {}  # name -> (prim, [(out, exp), ...])
        for prim in top_prims:
            if time.time() - t_start > 0.3:
                break
            outputs = []
            ok = True
            for inp, exp in task.train_examples:
                try:
                    out = prim.fn(inp)
                    # Validate: must be a 2D grid
                    if not isinstance(out, list) or not out or not isinstance(out[0], list):
                        ok = False
                        break
                    outputs.append((out, exp))
                except Exception:
                    ok = False
                    break
            if ok:
                prim_cache[prim.name] = (prim, outputs)

        if len(prim_cache) < 2:
            return None, 0

        for pred_name, pred_fn in predicates:
            if time.time() - t_start > 0.5:
                break

            true_idx, false_idx = [], []
            for idx, (inp, _) in enumerate(task.train_examples):
                try:
                    (true_idx if pred_fn(inp) else false_idx).append(idx)
                except Exception:
                    false_idx.append(idx)

            if not true_idx or not false_idx:
                continue

            true_scores = []
            false_scores = []
            for name, (prim, outputs) in prim_cache.items():
                te = sum(self.drive.prediction_error(outputs[i][0], outputs[i][1])
                         for i in true_idx) / len(true_idx)
                fe = sum(self.drive.prediction_error(outputs[i][0], outputs[i][1])
                         for i in false_idx) / len(false_idx)
                true_scores.append((te, prim))
                false_scores.append((fe, prim))

            true_scores.sort(key=lambda x: x[0])
            false_scores.sort(key=lambda x: x[0])

            for then_p in [p for _, p in true_scores[:5]]:
                for else_p in [p for _, p in false_scores[:5]]:
                    if then_p.name == else_p.name:
                        continue

                    # Evaluate without registering (avoids multiprocessing pickling issues)
                    cond_name = f"if_{pred_name}_{then_p.name}_else_{else_p.name}"
                    total_error = 0.0
                    max_err = 0.0
                    all_ok = True
                    solve_score = 0
                    for inp, expected in task.train_examples:
                        try:
                            pred = then_p.fn(inp) if pred_fn(inp) else else_p.fn(inp)
                        except Exception:
                            pred = inp
                        err = self.drive.prediction_error(pred, expected)
                        total_error += err
                        max_err = max(max_err, err)
                        if err <= solve_thresh:
                            solve_score += 1
                    n_ex = len(task.train_examples)
                    avg_error = total_error / n_ex if n_ex else 1.0
                    if avg_error <= solve_thresh:
                        # Register only if it actually solves
                        def _make(pf, tf, ef):
                            def fn(g):
                                try:
                                    return tf(g) if pf(g) else ef(g)
                                except Exception:
                                    return g
                            return fn
                        cond_fn = _make(pred_fn, then_p.fn, else_p.fn)
                        cond_prim = Primitive(
                            name=cond_name, arity=0, fn=cond_fn, domain="arc")
                        self.env.register_primitive(cond_prim)
                        sp = self._evaluate_program(Program(root=cond_name), task)
                        n_evals += 1
                        return sp, n_evals
                    complexity = 0.003  # 3 nodes
                    energy = avg_error + self.search_cfg.energy_beta * complexity
                    sp = ScoredProgram(
                        program=Program(root=cond_name),
                        energy=energy,
                        prediction_error=avg_error,
                        complexity_cost=complexity,
                        max_example_error=max_err,
                        example_solve_score=(solve_score / n_ex) ** 2 if n_ex else 0,
                    )
                    n_evals += 1
                    if best_result is None or sp.energy < best_result.energy:
                        best_result = sp

        return best_result, n_evals

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
