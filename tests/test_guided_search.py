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


from core.learner import _WakeContext
from core.types import Task, Program, ScoredProgram


class TestWakeContextDepth1Scores:
    def test_depth1_scores_initialized_empty(self):
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        assert ctx.depth1_scores == {}

    def test_depth1_scores_mutable(self):
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores["rotate_90"] = 0.15
        ctx.depth1_scores["mirror_h"] = 0.02
        assert len(ctx.depth1_scores) == 2
        assert ctx.depth1_scores["mirror_h"] == 0.02
