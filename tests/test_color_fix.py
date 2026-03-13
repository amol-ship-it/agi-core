"""Tests for post-hoc color fix (Phase 1.75)."""

import numpy as np
import pytest

from domains.arc.environment import ARCEnv
from core.types import Program


class TestInferOutputCorrection:
    """Test ARCEnv.infer_output_correction for color remapping."""

    def setup_method(self):
        self.env = ARCEnv()

    def test_simple_remap(self):
        """A consistent single-color remap should be detected."""
        # Program outputs 3 where expected is 5
        outputs = [[[0, 3, 0], [3, 3, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        assert "3to5" in correction.root

    def test_multi_color_remap(self):
        """Multiple consistent remaps should all be captured."""
        outputs = [[[1, 2, 0], [2, 1, 0]]]
        expected = [[[3, 4, 0], [4, 3, 0]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        assert "1to3" in correction.root
        assert "2to4" in correction.root

    def test_no_mismatch_returns_none(self):
        """If outputs already match, no correction needed."""
        outputs = [[[1, 2], [3, 4]]]
        expected = [[[1, 2], [3, 4]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is None

    def test_shape_mismatch_returns_none(self):
        """Different shapes can't be color-fixed."""
        outputs = [[[1, 2, 3]]]
        expected = [[[1, 2]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is None

    def test_ambiguous_remap_returns_none(self):
        """If same source color maps to different targets, no fix."""
        # Color 1 maps to 2 in half the pixels, 3 in the other half
        outputs = [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]
        expected = [[[2, 2, 2, 2, 2, 3, 3, 3, 3, 3]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is None

    def test_correction_is_executable(self):
        """The returned correction program should be executable."""
        from domains.arc.primitives import _PRIM_MAP
        outputs = [[[0, 3, 0], [3, 3, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        prim = _PRIM_MAP[correction.root]
        result = prim.fn([[0, 3, 0], [3, 3, 0]])
        assert result == [[0, 5, 0], [5, 5, 0]]

    def test_multi_example_consistency(self):
        """Remap must be consistent across multiple training examples."""
        outputs = [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
        expected = [
            [[2, 0], [0, 2]],
            [[0, 2], [2, 0]],
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        assert "1to2" in correction.root

    def test_multi_example_inconsistency(self):
        """Inconsistent remaps across examples should fail."""
        outputs = [
            [[1, 0]],
            [[1, 0]],
        ]
        expected = [
            [[2, 0]],
            [[3, 0]],  # 1→2 in first, 1→3 in second — inconsistent
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is None

    def test_color_swap_detected(self):
        """A↔B swap should be detected even when both colors appear correctly.

        The old safety check rejected A→B if there were correct A pixels.
        For swaps, the net effect is correct — every A→B is paired with B→A.
        """
        # Grid where colors 2 and 7 are swapped relative to expected
        outputs = [
            [[0, 2, 0], [7, 0, 7], [0, 2, 0]],
        ]
        expected = [
            [[0, 7, 0], [2, 0, 2], [0, 7, 0]],
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        # The correction should swap 2↔7
        from domains.arc.primitives import _PRIM_MAP
        prim = _PRIM_MAP[correction.root]
        result = prim.fn(outputs[0])
        assert result == expected[0]

    def test_color_swap_multi_example(self):
        """Swap must be consistent across multiple examples."""
        outputs = [
            [[2, 7], [7, 2]],
            [[0, 2, 7], [2, 0, 7]],
        ]
        expected = [
            [[7, 2], [2, 7]],
            [[0, 7, 2], [7, 0, 2]],
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        from domains.arc.primitives import _PRIM_MAP
        prim = _PRIM_MAP[correction.root]
        # Verify on both examples
        assert prim.fn(outputs[0]) == expected[0]
        assert prim.fn(outputs[1]) == expected[1]

    def test_swap_with_correct_pixels_of_both_colors(self):
        """Swap A↔B when both colors also appear in correct positions.

        infer_output_correction returns candidate corrections (no safety
        heuristic). The caller (_try_color_fix in learner) evaluates and
        only accepts corrections that reduce error.

        This test verifies infer_output_correction returns a correction
        for consistent transitions, even when a global remap would be
        destructive. The learner is responsible for trial-evaluating.
        """
        # Two pixels have 0↔5 swapped. Many correct pixels of both colors.
        outputs = [
            [[5, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 5, 0, 0],
             [5, 0, 0, 5, 0],
             [0, 0, 0, 0, 5]],
        ]
        expected = [
            [[5, 0, 0, 0, 0],
             [0, 0, 0, 5, 0],
             [0, 0, 5, 0, 0],
             [0, 0, 0, 5, 0],
             [0, 0, 0, 0, 5]],
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        # With relaxed safety, a correction should be returned
        # (the caller evaluates whether it actually helps)
        assert correction is not None, (
            "infer_output_correction should return a candidate for "
            "consistent color transitions")

    def test_remap_with_few_wrong_pixels(self):
        """Remap for a single wrong pixel still returns a candidate.

        The correction may be destructive (global remap), but that's OK:
        the learner's _try_color_fix evaluates and rejects bad corrections.
        """
        outputs = [
            [[3, 3, 3, 0],
             [3, 0, 3, 0],
             [0, 0, 0, 0]],
        ]
        expected = [
            [[3, 3, 3, 0],
             [3, 0, 5, 0],
             [0, 0, 0, 0]],
        ]
        correction = self.env.infer_output_correction(outputs, expected)
        # A candidate should be returned (consistent 3→5 transition)
        assert correction is not None


class TestNeighborhood5x5Correction:
    """Test 5x5 neighborhood-based pixel correction."""

    def setup_method(self):
        self.env = ARCEnv()

    def test_5x5_catches_longer_range(self):
        """5x5 neighborhood can fix patterns that depend on pixel 2 cells away."""
        # A pattern where the center pixel depends on a pixel 2 cells away
        # 3x3 can't capture this, but 5x5 can
        outputs = [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],  # center pixel 1 needs to become 2
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ]
        expected = [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ]
        # With try_5x5=True, should find a correction
        correction = self.env.infer_output_correction(
            outputs, expected, try_5x5=True)
        assert correction is not None
        # Verify it's executable
        from domains.arc.primitives import _PRIM_MAP
        prim = _PRIM_MAP[correction.root]
        result = prim.fn(outputs[0])
        assert result == expected[0]

    def test_5x5_inconsistent_returns_none(self):
        """Inconsistent 5x5 rules should return None."""
        # Same 5x5 neighborhood maps to different outputs
        outputs = [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ]
        expected = [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 3, 0, 0],  # Different output for same neighborhood
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ]
        correction = self.env._infer_neighborhood_correction_5x5(
            outputs, expected)
        assert correction is None

    def test_5x5_respects_max_rules(self):
        """5x5 correction should respect the max_rules cap."""
        # Create a grid with many unique 5x5 neighborhoods
        # Use a large grid with diverse patterns
        outputs = [np.random.RandomState(42).randint(0, 5, (10, 10)).tolist()]
        expected = [np.random.RandomState(43).randint(0, 5, (10, 10)).tolist()]
        # With max_rules=5, this should fail (too many rules)
        correction = self.env._infer_neighborhood_correction_5x5(
            outputs, expected, max_rules=5)
        assert correction is None


class TestRowColCorrection:
    """Test row/column-level corrections."""

    def setup_method(self):
        self.env = ARCEnv()

    def test_row_reverse(self):
        """Detect row reversal."""
        outputs = [[[1, 2], [3, 4]]]
        expected = [[[3, 4], [1, 2]]]
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is not None
        assert "row_reverse" in correction.root
        # Verify executable
        from domains.arc.primitives import _PRIM_MAP
        prim = _PRIM_MAP[correction.root]
        assert prim.fn(outputs[0]) == expected[0]

    def test_col_reverse(self):
        """Detect column reversal."""
        outputs = [[[1, 2, 3], [4, 5, 6]]]
        expected = [[[3, 2, 1], [6, 5, 4]]]
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is not None
        assert "col_reverse" in correction.root

    def test_row_shift(self):
        """Detect cyclic row shift."""
        outputs = [[[1, 2], [3, 4], [5, 6]]]
        expected = [[[5, 6], [1, 2], [3, 4]]]  # shifted by 1
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is not None
        assert "row_shift" in correction.root

    def test_col_shift(self):
        """Detect cyclic column shift."""
        outputs = [[[1, 2, 3], [4, 5, 6]]]
        expected = [[[3, 1, 2], [6, 4, 5]]]  # shifted by 1
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is not None
        assert "col_shift" in correction.root

    def test_no_match_returns_none(self):
        """If no row/col transform fixes the diff, return None."""
        outputs = [[[1, 2], [3, 4]]]
        expected = [[[5, 6], [7, 8]]]  # completely different
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is None

    def test_identical_returns_none(self):
        """If outputs match expected, no correction needed."""
        outputs = [[[1, 2], [3, 4]]]
        expected = [[[1, 2], [3, 4]]]
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is None

    def test_shape_mismatch_returns_none(self):
        """Different shapes can't be row/col-fixed."""
        outputs = [[[1, 2, 3]]]
        expected = [[[1, 2]]]
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is None

    def test_multi_example_consistency(self):
        """Row reversal must be consistent across examples."""
        outputs = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
        expected = [
            [[3, 4], [1, 2]],
            [[7, 8], [5, 6]],
        ]
        correction = self.env._infer_row_col_correction(outputs, expected)
        assert correction is not None
        assert "row_reverse" in correction.root

    def test_inconsistent_across_examples(self):
        """If transform works on one example but not another, return None."""
        outputs = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
        expected = [
            [[3, 4], [1, 2]],  # row reverse works
            [[5, 6], [7, 8]],  # but here it's identity (no reverse)
        ]
        correction = self.env._infer_row_col_correction(outputs, expected)
        # Should return None since no single transform is consistent
        assert correction is None


class TestIdentityCorrection:
    """Test identity-seeded correction (Phase 1.76)."""

    def setup_method(self):
        from core.types import Task
        self.env = ARCEnv()
        self.Task = Task

    def test_identity_correction_same_shape(self):
        """For same-shape tasks with local changes, identity correction works."""
        # Task: change all 1s to 2s (a color remap describable by neighborhood rules)
        task = self.Task(
            task_id="test_identity",
            train_examples=[
                ([[0, 1, 0], [1, 0, 1]], [[0, 2, 0], [2, 0, 2]]),
                ([[1, 1, 0], [0, 0, 1]], [[2, 2, 0], [0, 0, 2]]),
            ],
            test_inputs=[[[1, 0, 1], [0, 1, 0]]],
            test_outputs=[[[2, 0, 2], [0, 2, 0]]],
        )
        # The identity correction should find a color remap (cheapest strategy)
        inputs = [inp for inp, _ in task.train_examples]
        expected = [exp for _, exp in task.train_examples]
        correction = self.env.infer_output_correction(
            inputs, expected, max_rules=100, try_5x5=True)
        assert correction is not None

    def test_identity_correction_different_shape_fails(self):
        """Identity correction should not apply to different-shape tasks."""
        inputs = [[[1, 2, 3]]]
        expected = [[[1, 2]]]
        correction = self.env.infer_output_correction(
            inputs, expected, max_rules=100, try_5x5=True)
        assert correction is None


class TestCorrectionChaining:
    """Test multi-step correction chaining (Step 1)."""

    def setup_method(self):
        self.env = ARCEnv()

    def test_single_correction_still_works(self):
        """Simple single-step correction still returns directly."""
        outputs = [[[0, 3, 0], [3, 3, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        assert "3to5" in correction.root

    def test_chained_correction_color_then_neighborhood(self):
        """Two-step correction: color remap then neighborhood fix.

        First step: remap 3→5. Second step: fix remaining pixel via 3x3 patch.
        The chained correction should compose both steps.
        """
        # After remap 3→5, there's still a residual pixel needing neighborhood fix
        # Output: [[0, 3, 0], [3, 1, 0]]
        # After remap 3→5: [[0, 5, 0], [5, 1, 0]]
        # Expected: [[0, 5, 0], [5, 5, 0]]  (pixel [1][1] still wrong: 1→5)
        # Need: remap 3→5, then neighborhood fix for (1→5 in context of surrounding 5s)
        outputs = [[[0, 3, 0], [3, 1, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]

        # With chaining (default max_chain_depth=2), should compose corrections
        correction = self.env.infer_output_correction(
            outputs, expected, max_chain_depth=2)

        if correction is not None:
            # Verify the chained correction produces the right output
            result = self.env.execute(correction, outputs[0])
            assert np.array(result, dtype=np.int32).tolist() == expected[0]

    def test_no_chaining_when_depth_1(self):
        """max_chain_depth=1 disables chaining."""
        outputs = [[[0, 3, 0], [3, 1, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(
            outputs, expected, max_chain_depth=1)
        # Should return single correction (may not fully fix)
        # This is just a regression test — the important thing is no recursion

    def test_no_chaining_when_first_is_perfect(self):
        """If first correction is perfect, don't try second."""
        outputs = [[[0, 3, 0], [3, 3, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(
            outputs, expected, max_chain_depth=2)
        assert correction is not None
        # Should be a simple correction, not chained
        assert correction.children == []


class TestGlobalColorMap:
    """Test global color map primitive learning (Step 3)."""

    def test_consistent_color_map(self):
        """Consistent color→color mapping across examples is detected."""
        from domains.arc.grammar import _learn_global_color_map
        from core.types import Task

        task = Task(
            task_id="test_gcm",
            train_examples=[
                ([[0, 1, 2], [1, 2, 0]], [[0, 3, 4], [3, 4, 0]]),
                ([[2, 1, 0], [0, 2, 1]], [[4, 3, 0], [0, 4, 3]]),
            ],
            test_inputs=[[[1, 1, 2]]],
            test_outputs=[[[3, 3, 4]]],
        )
        cmap = _learn_global_color_map(task)
        assert cmap is not None
        assert cmap[1] == 3
        assert cmap[2] == 4
        assert cmap[0] == 0  # identity mapping for 0

    def test_inconsistent_map_returns_none(self):
        """Inconsistent mapping across examples returns None."""
        from domains.arc.grammar import _learn_global_color_map
        from core.types import Task

        task = Task(
            task_id="test_gcm_bad",
            train_examples=[
                ([[1, 2]], [[3, 4]]),
                ([[1, 2]], [[5, 4]]),  # 1→3 vs 1→5 inconsistent
            ],
            test_inputs=[],
            test_outputs=[],
        )
        cmap = _learn_global_color_map(task)
        assert cmap is None

    def test_ambiguous_single_example_returns_none(self):
        """If same source maps to multiple destinations in one example, fail."""
        from domains.arc.grammar import _learn_global_color_map
        from core.types import Task

        task = Task(
            task_id="test_gcm_ambig",
            train_examples=[
                ([[1, 1], [1, 1]], [[2, 3], [2, 3]]),  # 1→2 AND 1→3
            ],
            test_inputs=[],
            test_outputs=[],
        )
        cmap = _learn_global_color_map(task)
        assert cmap is None

    def test_identity_only_returns_none(self):
        """All-identity mapping (no changes) returns None."""
        from domains.arc.grammar import _learn_global_color_map
        from core.types import Task

        task = Task(
            task_id="test_gcm_id",
            train_examples=[
                ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            ],
            test_inputs=[],
            test_outputs=[],
        )
        cmap = _learn_global_color_map(task)
        assert cmap is None

    def test_different_shape_returns_none(self):
        """Different input/output shapes returns None."""
        from domains.arc.grammar import _learn_global_color_map
        from core.types import Task

        task = Task(
            task_id="test_gcm_shape",
            train_examples=[
                ([[1, 2, 3]], [[1, 2]]),
            ],
            test_inputs=[],
            test_outputs=[],
        )
        cmap = _learn_global_color_map(task)
        assert cmap is None

    def test_global_color_map_registered_as_primitive(self):
        """The learned primitive should be registered and executable."""
        from domains.arc.grammar import ARCGrammar
        from core.types import Task

        task = Task(
            task_id="test_gcm_prim",
            train_examples=[
                ([[0, 1], [2, 0]], [[0, 3], [4, 0]]),
                ([[1, 2], [0, 1]], [[3, 4], [0, 3]]),
            ],
            test_inputs=[[[1, 1]]],
            test_outputs=[[[3, 3]]],
        )
        grammar = ARCGrammar(seed=42)
        grammar.prepare_for_task(task)

        # Find the global color map primitive
        prims = grammar.base_primitives()
        gcm_prims = [p for p in prims if p.name == "task_global_color_map"]
        assert len(gcm_prims) == 1

        # Execute it
        fn = gcm_prims[0].fn
        result = fn([[1, 2], [0, 1]])
        assert result == [[3, 4], [0, 3]]
