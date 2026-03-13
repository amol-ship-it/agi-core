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


class TestCorrectionSimplified:
    """Test that correction is single-step and uses no 5x5+ neighborhoods."""

    def setup_method(self):
        self.env = ARCEnv()

    def test_single_correction_still_works(self):
        """Simple single-step correction still returns directly."""
        outputs = [[[0, 3, 0], [3, 3, 0]]]
        expected = [[[0, 5, 0], [5, 5, 0]]]
        correction = self.env.infer_output_correction(outputs, expected)
        assert correction is not None
        assert "3to5" in correction.root
        # Should be a simple correction, not chained
        assert correction.children == []

    def test_shape_mismatch_returns_none(self):
        """Different shapes can't be corrected."""
        inputs = [[[1, 2, 3]]]
        expected = [[[1, 2]]]
        correction = self.env.infer_output_correction(inputs, expected)
        assert correction is None

    def test_3x3_respects_max_rules_cap(self):
        """3x3 neighborhood correction respects the default max_rules=10 cap."""
        # Create a grid with many unique 3x3 neighborhoods — should exceed cap
        outputs = [np.random.RandomState(42).randint(0, 5, (10, 10)).tolist()]
        expected = [np.random.RandomState(43).randint(0, 5, (10, 10)).tolist()]
        # Default max_rules=10, so this should fail (too many rules)
        correction = self.env.infer_output_correction(outputs, expected)
        # Should be None — too many neighborhood rules for the cap
        assert correction is None


