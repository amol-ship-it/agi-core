"""
Tests for the ARC-AGI domain plugin.

Verifies:
1. The drive signal scores correctly
2. ARC task loading from JSON
3. Numpy conversion utilities

NOTE: Primitive-specific tests removed during zero-strip. They will be
re-added as each primitive is justified and added back.
"""

import json
import math
import os
import tempfile
import unittest

from core import Program, Task
from domains.arc import ARCEnv, ARCGrammar, ARCDrive, to_np, from_np
from domains.arc.drive import MAX_LOG_ERROR, DIM_MISMATCH_CAP
from domains.arc.dataset import load_arc_task, load_arc_dataset


class TestARCDrive(unittest.TestCase):
    """Test the ARC drive signal."""

    def test_perfect_match(self):
        drive = ARCDrive()
        grid = [[1, 2], [3, 4]]
        error = drive.prediction_error(grid, grid)
        self.assertAlmostEqual(error, 0.0)

    def test_total_mismatch(self):
        drive = ARCDrive()
        pred = [[1, 1], [1, 1]]
        exp = [[2, 2], [2, 2]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 1.0)
        self.assertLess(error, MAX_LOG_ERROR)

    def test_shape_mismatch_penalty(self):
        drive = ARCDrive()
        pred = [[1, 2, 3]]
        exp = [[1, 2]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)

    def test_none_inputs(self):
        drive = ARCDrive()
        self.assertAlmostEqual(drive.prediction_error(None, [[1]]), MAX_LOG_ERROR)
        self.assertAlmostEqual(drive.prediction_error([[1]], None), MAX_LOG_ERROR)

    def test_invalid_type(self):
        drive = ARCDrive()
        error = drive.prediction_error("not a grid", [[1]])
        self.assertAlmostEqual(error, MAX_LOG_ERROR)

    def test_log_scale_partial(self):
        """Partial match gives intermediate -log error."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 0.0)
        self.assertLess(error, MAX_LOG_ERROR)

    def test_log_scale_near_perfect(self):
        """-log(similarity) is small for near-perfect matches."""
        drive = ARCDrive()
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        error_partial = drive.prediction_error(pred, exp)
        pred_bad = [[5, 6], [7, 8]]
        error_bad = drive.prediction_error(pred_bad, exp)
        self.assertLess(error_partial, error_bad)
        self.assertGreater(error_bad / max(error_partial, 1e-10), 2.0)

    def test_complexity_cost(self):
        drive = ARCDrive()
        prog = Program(root="f", children=[Program(root="x")])
        self.assertAlmostEqual(drive.complexity_cost(prog), 2.0)


class TestARCTaskLoading(unittest.TestCase):
    """Test loading ARC tasks from JSON files."""

    def test_load_arc_task(self):
        task_data = {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]},
            ],
            "test": [
                {"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]},
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(task_data, f)
            path = f.name

        try:
            task = load_arc_task(path)
            self.assertEqual(len(task.train_examples), 2)
            self.assertEqual(len(task.test_inputs), 1)
            self.assertEqual(len(task.test_outputs), 1)
            self.assertGreater(task.difficulty, 0.0)
            self.assertIn("path", task.metadata)
        finally:
            os.unlink(path)

    def test_load_arc_dataset(self):
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"task{i}.json"), "w") as f:
                    json.dump(task_data, f)
            tasks = load_arc_dataset(tmpdir)
            self.assertEqual(len(tasks), 3)

    def test_load_arc_dataset_max_tasks(self):
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                with open(os.path.join(tmpdir, f"task{i}.json"), "w") as f:
                    json.dump(task_data, f)
            tasks = load_arc_dataset(tmpdir, max_tasks=2)
            self.assertEqual(len(tasks), 2)


class TestNumpyConversion(unittest.TestCase):
    """Test to_np / from_np roundtrip."""

    def test_roundtrip(self):
        grid = [[1, 2], [3, 4]]
        arr = to_np(grid)
        self.assertEqual(arr.shape, (2, 2))
        back = from_np(arr)
        self.assertEqual(back, grid)


class TestDimMismatchCap(unittest.TestCase):
    """Test that dimension-mismatched programs are capped in similarity."""

    def test_dim_mismatch_cap_applied(self):
        drive = ARCDrive()
        pred = [[1, 2, 3], [4, 5, 6]]
        exp = [[1, 2], [3, 4], [5, 6]]
        error = drive.prediction_error(pred, exp)
        self.assertGreater(error, 1.0)

    def test_dim_match_not_affected(self):
        drive = ARCDrive()
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2], [3, 4]]
        error = drive.prediction_error(pred, exp)
        self.assertAlmostEqual(error, 0.0)

    def test_partial_dim_gradient(self):
        drive = ARCDrive()
        exp = [[1, 2, 3], [4, 5, 6]]
        pred_partial = [[1, 2], [4, 5]]
        pred_full_mismatch = [[1, 2, 3]]
        err_partial = drive.prediction_error(pred_partial, exp)
        err_full = drive.prediction_error(pred_full_mismatch, exp)
        self.assertGreater(err_partial, 1.0)
        self.assertGreater(err_full, 1.0)


class TestVocabulary(unittest.TestCase):
    """Test that the system loads primitives correctly."""

    def test_base_primitives_loaded(self):
        grammar = ARCGrammar()
        prims = grammar.base_primitives()
        self.assertGreater(len(prims), 0)

    def test_env_execute_unknown_primitive(self):
        """Unknown primitive should return input unchanged."""
        env = ARCEnv()
        from core import Program
        prog = Program(root="nonexistent_prim")
        grid = [[1, 2], [3, 4]]
        result = env.execute(prog, grid)
        self.assertEqual(result, grid)


class TestProposeStrata(unittest.TestCase):
    """Test ARCGrammar.propose_strata()."""

    def _make_grammar_and_prims(self):
        grammar = ARCGrammar()
        prims = grammar.base_primitives()
        return grammar, prims

    def test_arc_grammar_propose_strata_always_includes_core(self):
        """Every task must get the exhaustive_core stratum."""
        grammar, prims = self._make_grammar_and_prims()
        # Simple same-dim task
        task = Task(
            task_id="t1",
            train_examples=[
                ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            ],
            test_inputs=[],
        )
        strata = grammar.propose_strata(task, prims)
        names = [s.name for s in strata]
        self.assertIn("exhaustive_core", names)
        # Core should have all primitives
        core = [s for s in strata if s.name == "exhaustive_core"][0]
        self.assertEqual(len(core.primitive_names), len(prims))

    def test_arc_grammar_propose_strata_triggers_local_rules(self):
        """Same-dim task with low diff should trigger local_rules."""
        grammar, prims = self._make_grammar_and_prims()
        # Same dims, low pixel diff (1/4 = 0.25 < 0.5)
        task = Task(
            task_id="t2",
            train_examples=[
                ([[1, 1], [1, 1]], [[1, 1], [1, 2]]),
            ],
            test_inputs=[],
        )
        strata = grammar.propose_strata(task, prims)
        names = [s.name for s in strata]
        self.assertIn("local_rules", names)
        # Check metadata
        local = [s for s in strata if s.name == "local_rules"][0]
        self.assertTrue(local.metadata.get("run_local_rules"))

    def test_arc_grammar_propose_strata_budget_sums_to_one(self):
        """Budget fractions across all strata should sum to ~1.0."""
        grammar, prims = self._make_grammar_and_prims()
        # A task that triggers several strata
        task = Task(
            task_id="t3",
            train_examples=[
                ([[1, 0, 2], [0, 0, 0], [3, 0, 4]],
                 [[1, 0, 2], [0, 0, 0], [3, 0, 5]]),
            ],
            test_inputs=[],
        )
        strata = grammar.propose_strata(task, prims)
        total = sum(s.budget_fraction for s in strata)
        self.assertAlmostEqual(total, 1.0, places=2)


class TestColormapLOOCV(unittest.TestCase):
    """Test that LOOCV rejects overfit colormaps."""

    def test_colormap_rejects_overfit_with_loocv(self):
        """A colormap that fits each example via unique keys should be rejected.

        Setup: 2 training examples for _try_half_colormap. Each example
        introduces a unique (both_nz, a_val, b_val) key not seen in the other.
        Training on ALL examples gives a consistent colormap that predicts
        both correctly. But hold out example 1 and train on just example 0 —
        the learned rule has no entry for example 1's key, so the prediction
        defaults to 0 ≠ expected → LOOCV rejects the candidate.
        """
        env = ARCEnv()

        # Each example uses a 2-row×1-col input (vertical split: top/bottom)
        # and a 1-row×1-col output.
        # key = (int(top != 0 and bot != 0), top, bot) → output
        #
        # Ex0: top=[[1]], bot=[[2]], out=[[3]]  → key (1,1,2)→3
        # Ex1: top=[[4]], bot=[[5]], out=[[6]]  → key (1,4,5)→6
        #
        # Both examples are independently consistent.
        # LOOCV: hold ex1, train on ex0 → map only has (1,1,2)→3.
        #        Apply to ex1 input [[4],[5]]: key (1,4,5) not in map → 0 ≠ 6
        #        → LOOCV fails → candidate rejected.
        task = Task(
            task_id="loocv_colormap_overfit",
            train_examples=[
                ([[1], [2]], [[3]]),   # ex0: unique key (1,1,2)→3
                ([[4], [5]], [[6]]),   # ex1: unique key (1,4,5)→6
            ],
            test_inputs=[],
        )

        result = env._try_half_colormap(task)
        self.assertIsNone(
            result,
            "LOOCV should reject a colormap where each example has unique "
            "keys not generalizable from the other examples."
        )

    def test_colormap_accepts_generalizable_mapping(self):
        """A colormap with consistent keys across examples should be accepted.

        Setup: 2 training examples for _try_half_colormap. Both examples
        share the same key structure: (1,1,2)→3 appears in both.
        LOOCV: hold out either example, the other still provides (1,1,2)→3,
        so the prediction for the held-out example is correct → LOOCV passes.
        """
        env = ARCEnv()

        # Ex0: top=[[1]], bot=[[2]], out=[[3]]  → key (1,1,2)→3
        # Ex1: top=[[1]], bot=[[2]], out=[[3]]  → key (1,1,2)→3 (same key/value)
        task = Task(
            task_id="loocv_colormap_generalizable",
            train_examples=[
                ([[1], [2]], [[3]]),   # ex0: key (1,1,2)→3
                ([[1], [2]], [[3]]),   # ex1: same key (1,1,2)→3
            ],
            test_inputs=[],
        )

        result = env._try_half_colormap(task)
        self.assertIsNotNone(
            result,
            "LOOCV should accept a colormap where all examples share "
            "consistent keys that generalize across held-out examples."
        )


class TestProceduralLOOCV(unittest.TestCase):
    """Test that LOOCV rejects overfit procedural rules."""

    def test_procedural_rejects_overfit_with_loocv(self):
        """A procedural rule that fits via unique per-example color keys is rejected.

        Setup: 2 training examples. Each has a single colored object that
        extends a ray to the right. Each example uses a different object color
        so the rule maps color→action uniquely per example.

        Training on ALL examples: rule = {1: extend_ray_right, 2: extend_ray_right}
        — consistent, predicts all correct.

        LOOCV: hold out ex1, train on ex0 → rule = {1: extend_ray_right}.
        Apply to ex1 (color=2 object): color=2 not in rule → nothing filled →
        prediction = input ≠ expected output → LOOCV rejects.
        """
        env = ARCEnv()

        # Ex0: object color=1 at (0,0) in 1×4 grid extends right
        # input  = [[1, 0, 0, 0]]
        # output = [[1, 1, 1, 1]]
        #
        # Ex1: object color=2 at (0,0) in 1×4 grid extends right
        # input  = [[2, 0, 0, 0]]
        # output = [[2, 2, 2, 2]]
        task = Task(
            task_id="loocv_procedural_overfit",
            train_examples=[
                ([[1, 0, 0, 0]], [[1, 1, 1, 1]]),   # ex0: color-1 ray
                ([[2, 0, 0, 0]], [[2, 2, 2, 2]]),   # ex1: color-2 ray
            ],
            test_inputs=[],
        )

        result = env.try_procedural(task)
        self.assertIsNone(
            result,
            "LOOCV should reject a procedural rule where each example "
            "introduces a unique color key not generalizable from others."
        )

    def test_procedural_accepts_generalizable_rule(self):
        """A procedural rule consistent across examples should be accepted.

        Setup: 2 training examples, both with color=1 objects that extend
        rays to the right (but in different grid positions/sizes). The rule
        maps color=1 → extend_ray(right), consistent across both.

        LOOCV: hold out either example, the other still provides the same
        color=1 rule → prediction is correct → LOOCV passes.
        """
        env = ARCEnv()

        # Ex0: color=1 at (0,0) extends right in 1×3 grid
        # Ex1: color=1 at (0,0) extends right in 1×4 grid
        # Both have color=1 → extend_ray_right with fill_color=1
        task = Task(
            task_id="loocv_procedural_generalizable",
            train_examples=[
                ([[1, 0, 0]], [[1, 1, 1]]),        # ex0: 1×3 ray
                ([[1, 0, 0, 0]], [[1, 1, 1, 1]]),  # ex1: 1×4 ray
            ],
            test_inputs=[],
        )

        result = env.try_procedural(task)
        self.assertIsNotNone(
            result,
            "LOOCV should accept a procedural rule that generalizes across "
            "examples (same color key → same action in all examples)."
        )


if __name__ == "__main__":
    unittest.main()
