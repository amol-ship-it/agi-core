"""
ARC-AGI Drive Signal: structural similarity scoring + program complexity.

Adapted from agi-mvp-general's partial-credit scorer.
Uses weighted composite of pixel accuracy, dimension match, color overlap,
and nonzero density similarity to create a smooth fitness landscape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core import DriveSignal, Program
from .primitives import to_np


class ARCDrive(DriveSignal):
    """
    ARC drive signal: structural similarity + program complexity.

    prediction_error is a weighted composite:
      0.60 × pixel_accuracy (inverted: 0 = perfect)
      0.15 × dimension_match (1 if same shape, 0 otherwise)
      0.15 × color_overlap (Jaccard on non-bg color palettes)
      0.10 × nonzero_similarity (ratio of nonzero pixel counts)

    This creates a smoother fitness landscape than binary pixel match,
    enabling beam search and evolution to make incremental progress.
    """

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        """Structural similarity distance: 0.0 = perfect match, 1.0 = nothing matches."""
        if predicted is None or expected is None:
            return 1.0

        try:
            pred = to_np(predicted)
            exp = to_np(expected)
        except (ValueError, TypeError):
            return 1.0

        pred_h, pred_w = pred.shape
        exp_h, exp_w = exp.shape

        if pred_h == 0 or pred_w == 0 or exp_h == 0 or exp_w == 0:
            return 1.0

        # --- Component 1: Pixel accuracy (weight 0.60) ---
        if pred.shape == exp.shape:
            total = exp_h * exp_w
            pixel_acc = float(np.sum(pred == exp)) / total  # 1.0 = perfect
            dim_match = 1.0
        else:
            # Shape mismatch: compute partial pixel accuracy on overlap
            min_r = min(pred_h, exp_h)
            min_c = min(pred_w, exp_w)
            overlap_match = float(np.sum(pred[:min_r, :min_c] == exp[:min_r, :min_c]))
            total = exp_h * exp_w
            pixel_acc = overlap_match / total
            # Partial dimension credit
            dim_match = 0.0
            if pred_h == exp_h:
                dim_match += 0.5
            if pred_w == exp_w:
                dim_match += 0.5

        # --- Component 2: Color palette overlap (weight 0.15) ---
        # Jaccard similarity on non-background colors present
        pred_colors = set()
        exp_colors = set()
        for c in range(1, 10):
            if np.any(pred == c):
                pred_colors.add(c)
            if np.any(exp == c):
                exp_colors.add(c)

        if pred_colors or exp_colors:
            inter = len(pred_colors & exp_colors)
            union = len(pred_colors | exp_colors)
            color_overlap = inter / union if union > 0 else 1.0
        else:
            color_overlap = 1.0  # both all-background

        # --- Component 3: Nonzero pixel density similarity (weight 0.10) ---
        pred_nz = int(np.count_nonzero(pred))
        exp_nz = int(np.count_nonzero(exp))
        max_nz = max(pred_nz, exp_nz, 1)
        nz_sim = 1.0 - abs(pred_nz - exp_nz) / max_nz

        # Weighted composite (higher = better match)
        similarity = (0.60 * pixel_acc
                      + 0.15 * dim_match
                      + 0.15 * color_overlap
                      + 0.10 * nz_sim)

        # Invert: prediction_error = 0 means perfect, 1 means worst
        return max(0.0, 1.0 - similarity)

    def complexity_cost(self, program: Program) -> float:
        """Program size as complexity measure."""
        return float(program.size)
