"""
Tests for the new features:
1. Exhaustive enumeration (depth 1-2 before beam search)
2. New ARC primitives (extract_largest, symmetry, repeat, border, etc.)
3. Sequential compounding (within-run knowledge transfer)
4. Culture persistence (save/load library with program reconstruction)
5. prepare_for_task (verified no-op — Decision 72)
"""

import json
import os
import tempfile
import unittest

from core import (
    Program, Task, ScoredProgram, LibraryEntry,
    Learner, InMemoryStore, SearchConfig, SleepConfig, CurriculumConfig,
)
from core.memory import _program_to_dict, _program_from_dict
from domains.arc import ARCEnv, ARCGrammar, ARCDrive, ARC_PRIMITIVES
from domains.arc.dataset import make_sample_tasks
from domains.arc.primitives import (
    extract_largest_object, extract_smallest_object,
    anti_diagonal_mirror, make_symmetric_h, make_symmetric_v,
    repeat_pattern_right, repeat_pattern_down,
    add_border, remove_border,
    sort_rows_by_color_count, sort_cols_by_color_count,
    unique_rows, unique_cols, recolor_by_size_rank,
    extend_lines_h, extend_lines_v,
    rotate_90_cw, mirror_horizontal,
)


# =============================================================================
# New ARC primitives tests
# =============================================================================

class TestNewPrimitives(unittest.TestCase):
    """Test the 16 new ARC spatial/object primitives."""

    def test_extract_largest_object(self):
        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 0, 0],
        ]
        result = extract_largest_object(grid)
        # Largest connected component is the 2x2 block of 1s
        self.assertEqual(result, [[1, 1], [1, 1]])

    def test_extract_largest_object_empty(self):
        result = extract_largest_object([[0, 0], [0, 0]])
        self.assertEqual(result, [[0]])

    def test_extract_smallest_object(self):
        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 0, 0],
        ]
        result = extract_smallest_object(grid)
        # Smallest is the single pixel of color 2
        self.assertEqual(result, [[2]])

    def test_extract_smallest_object_empty(self):
        result = extract_smallest_object([[0, 0], [0, 0]])
        self.assertEqual(result, [[0]])

    def test_anti_diagonal_mirror(self):
        grid = [[1, 2], [3, 4]]
        result = anti_diagonal_mirror(grid)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_make_symmetric_h(self):
        grid = [[1, 2, 3, 4]]
        result = make_symmetric_h(grid)
        # Left half mirrored to right: [1, 2, 2, 1]
        self.assertEqual(result, [[1, 2, 2, 1]])

    def test_make_symmetric_v(self):
        grid = [[1], [2], [3], [4]]
        result = make_symmetric_v(grid)
        self.assertEqual(result, [[1], [2], [2], [1]])

    def test_repeat_pattern_right(self):
        grid = [[1, 2], [3, 4]]
        result = repeat_pattern_right(grid)
        self.assertEqual(result, [[1, 2, 1, 2], [3, 4, 3, 4]])

    def test_repeat_pattern_down(self):
        grid = [[1, 2]]
        result = repeat_pattern_down(grid)
        self.assertEqual(result, [[1, 2], [1, 2]])

    def test_add_border(self):
        grid = [[1, 1], [1, 1]]
        result = add_border(grid)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)

    def test_remove_border(self):
        grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        result = remove_border(grid)
        self.assertEqual(result, [[5]])

    def test_remove_border_too_small(self):
        grid = [[1, 2]]
        result = remove_border(grid)
        self.assertEqual(result, grid)

    def test_sort_rows_by_color_count(self):
        grid = [[1, 2, 3], [0, 0, 1], [1, 0, 0]]
        result = sort_rows_by_color_count(grid)
        # Row with 1 nonzero first, then 1, then 3
        counts = [sum(1 for c in row if c != 0) for row in result]
        self.assertEqual(counts, sorted(counts))

    def test_sort_cols_by_color_count(self):
        grid = [[1, 0], [1, 0], [0, 1]]
        result = sort_cols_by_color_count(grid)
        self.assertIsInstance(result, list)

    def test_unique_rows(self):
        grid = [[1, 2], [3, 4], [1, 2]]
        result = unique_rows(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_unique_rows_empty(self):
        result = unique_rows([])
        self.assertEqual(result, [[0]])

    def test_unique_cols(self):
        grid = [[1, 2, 1], [3, 4, 3]]
        result = unique_cols(grid)
        self.assertEqual(len(result[0]), 2)

    def test_recolor_by_size_rank(self):
        grid = [[3, 3, 3], [5, 5, 0], [5, 0, 0]]
        result = recolor_by_size_rank(grid)
        # 3 appears 3 times (most) -> 1, 5 appears 3 times -> 2 (or tied)
        flat = [c for row in result for c in row if c != 0]
        self.assertTrue(all(1 <= c <= 9 for c in flat))

    def test_recolor_by_size_rank_all_zeros(self):
        result = recolor_by_size_rank([[0, 0], [0, 0]])
        self.assertEqual(result, [[0, 0], [0, 0]])

    def test_extend_lines_h(self):
        grid = [[0, 1, 0], [0, 0, 0]]
        result = extend_lines_h(grid)
        self.assertEqual(result[0], [1, 1, 1])
        self.assertEqual(result[1], [0, 0, 0])

    def test_extend_lines_v(self):
        grid = [[0, 0], [1, 0], [0, 0]]
        result = extend_lines_v(grid)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[2][0], 1)

    def test_all_new_primitives_return_valid_grids(self):
        """All new unary primitives should return list of lists."""
        grid = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
        new_prim_names = {
            'extract_largest', 'extract_smallest', 'anti_diag_mirror',
            'make_sym_h', 'make_sym_v', 'repeat_right', 'repeat_down',
            'add_border', 'remove_border', 'sort_rows', 'sort_cols',
            'unique_rows', 'unique_cols', 'recolor_by_rank',
            'extend_lines_h', 'extend_lines_v',
        }
        for prim in ARC_PRIMITIVES:
            if prim.name in new_prim_names and prim.arity == 1:
                result = prim.fn(grid)
                self.assertIsInstance(result, list, f"{prim.name} didn't return list")
                self.assertTrue(len(result) > 0, f"{prim.name} returned empty")
                self.assertIsInstance(result[0], list, f"{prim.name} not list of lists")

    def test_primitive_count_increased(self):
        """Should have 222+ primitives after agi-mvp-general port."""
        self.assertGreaterEqual(len(ARC_PRIMITIVES), 200)


# =============================================================================
# Exhaustive enumeration tests
# =============================================================================

class TestExhaustiveEnumeration(unittest.TestCase):
    """Test exhaustive enumeration in the learner."""

    def _make_arc_learner(self, **kwargs):
        defaults = dict(
            beam_width=10, max_generations=5,
            solve_threshold=0.001, seed=42, energy_beta=0.002,
            exhaustive_depth=2, exhaustive_pair_top_k=10,
        )
        defaults.update(kwargs)
        return Learner(
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(**defaults),
        )

    def test_exhaustive_enumerate_depth1(self):
        learner = self._make_arc_learner()
        # Use invert_crop task — NOT solvable at depth 1
        task = make_sample_tasks()[7]
        prims = learner.grammar.base_primitives()

        scored, n_evals = learner._exhaustive_enumerate(prims, task, max_depth=1)
        # Should have evaluated all unary prims (no early exit since unsolvable at depth 1)
        unary_count = sum(1 for p in prims if p.arity <= 1)
        self.assertEqual(n_evals, unary_count)
        self.assertEqual(len(scored), unary_count)

    def test_exhaustive_enumerate_depth1_early_exit(self):
        learner = self._make_arc_learner()
        task = make_sample_tasks()[0]  # rot90 task — solvable at depth 1
        prims = learner.grammar.base_primitives()

        scored, n_evals = learner._exhaustive_enumerate(prims, task, max_depth=1)
        # Should early exit after finding the solution
        self.assertLessEqual(n_evals, len([p for p in prims if p.arity <= 1]))
        # At least one should be solved
        best = min(scored, key=lambda s: s.prediction_error)
        self.assertLessEqual(best.prediction_error, 0.001)

    def test_exhaustive_enumerate_depth2(self):
        learner = self._make_arc_learner()
        task = make_sample_tasks()[6]  # rot_mirror — needs depth 2
        prims = learner.grammar.base_primitives()

        scored, n_evals = learner._exhaustive_enumerate(prims, task, max_depth=2)
        # Should have found a solution (early exit possible)
        best = min(scored, key=lambda s: s.prediction_error)
        self.assertLessEqual(best.prediction_error, 0.001)

    def test_exhaustive_solves_rot90(self):
        """Exhaustive should find rot90cw as a depth-1 program."""
        learner = self._make_arc_learner(exhaustive_depth=1)
        task = make_sample_tasks()[0]  # rot90 task
        result = learner.wake_on_task(task)
        self.assertTrue(result.solved)
        self.assertEqual(result.generations_used, 0)  # solved by enumeration

    def test_exhaustive_solves_composition(self):
        """Exhaustive depth-2 should find rot90+mirror as a 2-step program."""
        learner = self._make_arc_learner(exhaustive_depth=2, exhaustive_pair_top_k=40)
        task = make_sample_tasks()[6]  # rot_mirror task
        result = learner.wake_on_task(task)
        self.assertTrue(result.solved)
        self.assertEqual(result.generations_used, 0)

    def test_exhaustive_disabled(self):
        """With exhaustive_depth=0, enumeration is skipped.

        The task may still be solved by other phases (object decomposition,
        conditional search, etc.) before beam search runs — so we just verify
        that the result is valid and evaluations were performed.
        """
        learner = self._make_arc_learner(exhaustive_depth=0, beam_width=30, max_generations=20)
        task = make_sample_tasks()[0]
        result = learner.wake_on_task(task)
        self.assertGreater(result.evaluations, 0)

    def test_no_record_also_enumerates(self):
        """_wake_on_task_no_record should also use enumeration."""
        learner = self._make_arc_learner(exhaustive_depth=1)
        task = make_sample_tasks()[0]
        result = learner._wake_on_task_no_record(task)
        self.assertTrue(result.solved)
        self.assertEqual(result.generations_used, 0)


# =============================================================================
# Sequential compounding tests
# =============================================================================

class TestSequentialCompounding(unittest.TestCase):
    """Test within-run sequential compounding."""

    def test_sequential_compounding_runs(self):
        tasks = make_sample_tasks()[:3]
        learner = Learner(
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(
                beam_width=30, max_generations=10,
                solve_threshold=0.001, seed=42,
                exhaustive_depth=2, exhaustive_pair_top_k=10,
            ),
        )
        results = learner.run_curriculum(
            tasks,
            CurriculumConfig(wake_sleep_rounds=1, sequential_compounding=True),
        )
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].solved, 0)

    def test_immediate_promote(self):
        """_immediate_promote should add subtrees to the library."""
        learner = Learner(
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(beam_width=10, max_generations=5, seed=42),
        )
        # Create a depth-2 solved program
        prog = Program(root="mirror_h", children=[Program(root="rot90cw")])
        scored = ScoredProgram(
            program=prog, energy=0.0, prediction_error=0.0,
            complexity_cost=2.0, task_id="test_task",
        )
        learner._immediate_promote(scored, "test_task")
        lib = learner.memory.get_library()
        # Should have promoted the depth-2 subtree
        self.assertGreater(len(lib), 0)
        # The promoted entry should be the full program (size >= 2)
        self.assertTrue(any(e.program.size >= 2 for e in lib))

    def test_immediate_promote_skips_trivial(self):
        """Depth-1 programs shouldn't be promoted (size < 2)."""
        learner = Learner(
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(beam_width=10, max_generations=5, seed=42),
        )
        prog = Program(root="rot90cw")  # depth 1, size 1
        scored = ScoredProgram(
            program=prog, energy=0.0, prediction_error=0.0,
            complexity_cost=1.0, task_id="test_task",
        )
        learner._immediate_promote(scored, "test_task")
        self.assertEqual(len(learner.memory.get_library()), 0)

    def test_immediate_promote_respects_max_library(self):
        """Should stop adding when library hits max size."""
        learner = Learner(
            environment=ARCEnv(),
            grammar=ARCGrammar(seed=42),
            drive=ARCDrive(),
            memory=InMemoryStore(),
            search_config=SearchConfig(beam_width=10, max_generations=5, seed=42),
            sleep_config=SleepConfig(max_library_size=1),
        )
        prog = Program(root="mirror_h", children=[
            Program(root="rot90cw", children=[Program(root="transpose")])])
        scored = ScoredProgram(
            program=prog, energy=0.0, prediction_error=0.0,
            complexity_cost=3.0, task_id="test",
        )
        learner._immediate_promote(scored, "test")
        # Should cap at max_library_size=1
        self.assertLessEqual(len(learner.memory.get_library()), 1)


# =============================================================================
# Culture persistence tests
# =============================================================================

class TestCulturePersistence(unittest.TestCase):
    """Test save/load culture with proper program reconstruction."""

    def test_program_to_dict_leaf(self):
        prog = Program(root="rot90cw")
        d = _program_to_dict(prog)
        self.assertEqual(d, {"root": "rot90cw"})

    def test_program_to_dict_tree(self):
        prog = Program(root="mirror_h", children=[Program(root="rot90cw")])
        d = _program_to_dict(prog)
        self.assertEqual(d["root"], "mirror_h")
        self.assertEqual(len(d["children"]), 1)
        self.assertEqual(d["children"][0]["root"], "rot90cw")

    def test_program_roundtrip(self):
        prog = Program(root="overlay", children=[
            Program(root="mirror_h", children=[Program(root="rot90cw")]),
            Program(root="transpose"),
        ])
        d = _program_to_dict(prog)
        restored = _program_from_dict(d)
        self.assertEqual(repr(restored), repr(prog))

    def test_save_load_culture(self):
        memory = InMemoryStore()
        # Add a library entry
        prog = Program(root="mirror_h", children=[Program(root="rot90cw")])
        entry = LibraryEntry(
            name="learned_0", program=prog, usefulness=5.0,
            reuse_count=3, source_tasks=["t1", "t2"], domain="arc",
        )
        memory.add_to_library(entry)

        # Add a solution
        sp = ScoredProgram(
            program=Program(root="crop_nonzero"), energy=0.001,
            prediction_error=0.0, complexity_cost=1.0, task_id="t1",
        )
        memory.store_solution("t1", sp)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            memory.save_culture(path)

            # Load into fresh memory
            memory2 = InMemoryStore()
            memory2.load_culture(path)

            # Verify library
            lib = memory2.get_library()
            self.assertEqual(len(lib), 1)
            self.assertEqual(lib[0].name, "learned_0")
            self.assertEqual(repr(lib[0].program), repr(prog))
            self.assertAlmostEqual(lib[0].usefulness, 5.0)
            self.assertEqual(lib[0].reuse_count, 3)
            self.assertEqual(lib[0].source_tasks, ["t1", "t2"])

            # Verify solutions
            sols = memory2.get_solutions()
            self.assertIn("t1", sols)
            self.assertEqual(sols["t1"].program.root, "crop_nonzero")
        finally:
            os.unlink(path)

    def test_save_culture_format(self):
        memory = InMemoryStore()
        prog = Program(root="test")
        memory.add_to_library(LibraryEntry(
            name="lib_0", program=prog, usefulness=1.0))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            memory.save_culture(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["version"], "1.0")
            self.assertEqual(data["library_count"], 1)
            # Program should be a dict, not repr string
            self.assertIsInstance(data["library"][0]["program"], dict)
        finally:
            os.unlink(path)


# =============================================================================
class TestPrepareForTaskNoOp(unittest.TestCase):
    """Verify prepare_for_task doesn't add unexpected task-scoped primitives."""

    def test_prepare_adds_no_task_prims(self):
        """prepare_for_task should not generate task_* primitives for
        simple tasks that don't match parameterized patterns."""
        grammar = ARCGrammar(seed=42)
        task = Task(
            task_id="test",
            train_examples=[
                ([[0, 1], [1, 0]], [[5, 1], [1, 5]]),
            ],
            test_inputs=[[[0]]],
        )
        grammar.prepare_for_task(task)
        task_prims = [p for p in grammar.base_primitives()
                      if p.name.startswith("task_")]
        self.assertEqual(task_prims, [],
            f"Expected no task_* primitives, got: {[p.name for p in task_prims]}"
        )


# =============================================================================
# Config tests
# =============================================================================

class TestNewConfigFields(unittest.TestCase):
    """Test new configuration fields."""

    def test_search_config_exhaustive_defaults(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.exhaustive_depth, 3)
        self.assertEqual(cfg.exhaustive_pair_top_k, 40)
        self.assertEqual(cfg.exhaustive_triple_top_k, 15)

    def test_search_config_eval_budget_default(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.eval_budget, 0)  # unlimited by default

    def test_curriculum_sequential_default(self):
        cfg = CurriculumConfig()
        self.assertFalse(cfg.sequential_compounding)

    def test_avg_cells(self):
        """Test cell count calculation for budget normalization."""
        task = Task(
            task_id="test",
            train_examples=[
                ([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]),  # 2×3=6 cells
                ([[1, 2], [3, 4], [5, 6], [7, 8]], [[1], [2]]),       # 4×2=8 cells
            ],
            test_inputs=[],
        )
        max_cells = Learner._avg_cells(task)
        self.assertEqual(max_cells, 8)  # max(6, 8) = 8

    def test_avg_cells_empty(self):
        """Empty task returns 1 (safe default)."""
        task = Task(task_id="test", train_examples=[], test_inputs=[])
        self.assertEqual(Learner._avg_cells(task), 1)


if __name__ == "__main__":
    unittest.main()
