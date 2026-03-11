"""Tests for conditional search (Phase 1.25)."""

import pytest

from domains.arc.primitives import (
    ARC_PREDICATES,
    _pred_is_symmetric_h,
    _pred_is_symmetric_v,
    _pred_is_square,
    _pred_has_single_color,
    _pred_is_tall,
    _pred_is_wide,
    _pred_has_many_colors,
    _pred_is_small,
    _pred_is_large,
    _pred_has_two_colors,
    _pred_has_frame,
    _pred_has_diag_sym,
    _pred_is_odd_dims,
    _pred_has_h_stripe,
    _pred_has_v_stripe,
)
from domains.arc.grammar import ARCGrammar


class TestPredicates:
    """Test that ARC predicates work correctly."""

    def test_is_symmetric_h(self):
        assert _pred_is_symmetric_h([[1, 2, 1], [3, 4, 3]])
        assert not _pred_is_symmetric_h([[1, 2, 3]])

    def test_is_symmetric_v(self):
        assert _pred_is_symmetric_v([[1, 2], [3, 4], [1, 2]])
        assert not _pred_is_symmetric_v([[1, 2], [3, 4]])

    def test_is_square(self):
        assert _pred_is_square([[1, 2], [3, 4]])
        assert not _pred_is_square([[1, 2, 3], [4, 5, 6]])

    def test_is_tall(self):
        assert _pred_is_tall([[1], [2], [3]])
        assert not _pred_is_tall([[1, 2, 3]])

    def test_is_wide(self):
        assert _pred_is_wide([[1, 2, 3]])
        assert not _pred_is_wide([[1], [2], [3]])

    def test_has_single_color(self):
        assert _pred_has_single_color([[0, 1, 0], [1, 0, 1]])
        assert not _pred_has_single_color([[1, 2, 0], [0, 1, 2]])

    def test_has_many_colors(self):
        assert _pred_has_many_colors([[1, 2, 3, 4]])
        assert not _pred_has_many_colors([[1, 2, 0]])

    def test_has_two_colors(self):
        assert _pred_has_two_colors([[1, 2, 0], [2, 1, 0]])
        assert not _pred_has_two_colors([[1, 0, 0]])

    def test_is_small(self):
        assert _pred_is_small([[1, 2], [3, 4]])  # 4 cells < 50
        assert not _pred_is_small([[0]*10 for _ in range(6)])  # 60 cells

    def test_is_large(self):
        assert _pred_is_large([[0]*15 for _ in range(15)])  # 225 cells
        assert not _pred_is_large([[1, 2], [3, 4]])

    def test_has_frame(self):
        grid = [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ]
        assert _pred_has_frame(grid)
        assert not _pred_has_frame([[1, 1], [1, 1]])

    def test_has_diag_sym(self):
        grid = [
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ]
        assert _pred_has_diag_sym(grid)
        assert not _pred_has_diag_sym([[1, 2], [3, 4]])

    def test_is_odd_dims(self):
        assert _pred_is_odd_dims([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert not _pred_is_odd_dims([[1, 2], [3, 4]])

    def test_has_h_stripe(self):
        assert _pred_has_h_stripe([[1, 1, 1], [2, 3, 4]])
        assert not _pred_has_h_stripe([[1, 2, 3]])

    def test_has_v_stripe(self):
        assert _pred_has_v_stripe([[1, 2], [1, 3], [1, 4]])
        assert not _pred_has_v_stripe([[1, 2], [3, 4]])


class TestARCGrammarPredicates:
    """Test that ARCGrammar returns predicates."""

    def test_get_predicates_returns_list(self):
        grammar = ARCGrammar()
        preds = grammar.get_predicates()
        assert isinstance(preds, list)
        assert len(preds) == 17  # 17 predicates

    def test_predicates_are_callable(self):
        grammar = ARCGrammar()
        for name, fn in grammar.get_predicates():
            assert isinstance(name, str)
            assert callable(fn)

    def test_predicates_match_module_level(self):
        grammar = ARCGrammar()
        preds = grammar.get_predicates()
        assert len(preds) == len(ARC_PREDICATES)
