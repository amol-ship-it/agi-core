"""
ARC-AGI Drive Signal: structural similarity scoring + program complexity.

Adapted from agi-mvp-general's partial-credit scorer.
Uses weighted composite of pixel accuracy, dimension match, color overlap,
and nonzero density similarity to create a smooth fitness landscape.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core import DriveSignal, Program
from .primitives import to_np


MAX_LOG_ERROR = 20.0  # cap for zero similarity (-log transform)
DIM_MISMATCH_CAP = 0.35  # max similarity when dimensions differ (sweep: 0.25/0.35/0.45)


class ARCDrive(DriveSignal):
    """
    ARC drive signal: -log(structural_similarity) + program complexity.

    prediction_error uses a 3-level hierarchy:
      **Example**: -log(structural_similarity) per (input, output) pair
      **Split**: Mean of example scores → prediction_error (0 iff all perfect)
      **Task**: Train split during search; test split for validation

    The structural similarity is a weighted composite:
      0.60 × pixel_accuracy
      0.15 × dimension_match (1 if same shape, 0 otherwise)
      0.15 × color_overlap (Jaccard on non-bg color palettes)
      0.10 × nonzero_similarity (ratio of nonzero pixel counts)

    The -log transform makes exact matches exponentially more rewarded
    while keeping continuous gradients for search. For small errors,
    -log(1-x) ≈ x, so existing thresholds remain valid.
    """

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        """Structural similarity distance: 0.0 = perfect match, MAX_LOG_ERROR = worst."""
        if predicted is None or expected is None:
            return MAX_LOG_ERROR

        try:
            pred = to_np(predicted)
            exp = to_np(expected)
        except (ValueError, TypeError):
            return MAX_LOG_ERROR

        pred_h, pred_w = pred.shape
        exp_h, exp_w = exp.shape

        if pred_h == 0 or pred_w == 0 or exp_h == 0 or exp_w == 0:
            return MAX_LOG_ERROR

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
        # Use np.unique once per array instead of scanning 9 times with np.any.
        # This is ~10x faster: O(n) unique vs O(9n) scans.
        pred_colors = set(pred.flat) - {0}
        exp_colors = set(exp.flat) - {0}

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

        # Cap similarity when dimensions differ — wrong-dim programs must never
        # score better than moderate right-dim programs.  This prevents them
        # from entering near-miss refinement and wasting eval budget.
        if dim_match < 1.0:
            similarity = min(similarity, DIM_MISMATCH_CAP)

        # -log transform: 0 = perfect, higher = worse
        if similarity <= 0.0:
            return MAX_LOG_ERROR
        return min(-math.log(similarity), MAX_LOG_ERROR)

    def complexity_cost(self, program: Program) -> float:
        """Program size as complexity measure."""
        return float(program.size)
