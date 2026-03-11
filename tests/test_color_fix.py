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
