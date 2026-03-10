"""
ARC-AGI Drive Signal: pixel accuracy + program complexity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core import DriveSignal, Program
from .primitives import to_np


class ARCDrive(DriveSignal):
    """
    ARC drive signal: pixel edit distance + program complexity.

    prediction_error = fraction of cells that differ between predicted and expected.
    This is pixel_accuracy inverted: 0.0 = perfect match, 1.0 = nothing matches.
    """

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        """Pixel edit distance: fraction of cells that differ."""
        if predicted is None or expected is None:
            return 1.0

        try:
            pred = to_np(predicted)
            exp = to_np(expected)
        except (ValueError, TypeError):
            return 1.0

        # If shapes differ, that's a significant error
        if pred.shape != exp.shape:
            # Partial credit: penalize by shape mismatch
            shape_penalty = 0.3
            # Try to compare overlapping region
            min_r = min(pred.shape[0], exp.shape[0])
            min_c = min(pred.shape[1], exp.shape[1])
            if min_r == 0 or min_c == 0:
                return 1.0
            overlap_pred = pred[:min_r, :min_c]
            overlap_exp = exp[:min_r, :min_c]
            pixel_err = float(np.sum(overlap_pred != overlap_exp)) / (exp.shape[0] * exp.shape[1])
            return min(1.0, pixel_err + shape_penalty)

        total_cells = exp.shape[0] * exp.shape[1]
        if total_cells == 0:
            return 0.0
        return float(np.sum(pred != exp)) / total_cells

    def complexity_cost(self, program: Program) -> float:
        """Program size as complexity measure."""
        return float(program.size)

