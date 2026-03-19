"""Tests for guided depth-4/5 search."""
import pytest
from core.config import SearchConfig


class TestGuidedSearchConfig:
    def test_default_guided_fields(self):
        cfg = SearchConfig()
        assert cfg.guided_depth4_top_k == 20
        assert cfg.guided_depth5_top_k == 10
        assert cfg.guided_nearmiss_top_k == 5
        assert cfg.guided_budget_fraction == 0.30

    def test_custom_guided_fields(self):
        cfg = SearchConfig(guided_depth4_top_k=15, guided_depth5_top_k=8)
        assert cfg.guided_depth4_top_k == 15
        assert cfg.guided_depth5_top_k == 8
