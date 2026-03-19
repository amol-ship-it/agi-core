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


from core.learner import Learner
from core.interfaces import Environment, Grammar, DriveSignal, Memory


def _make_learner():
    """Create a minimal Learner with mock interfaces."""
    from unittest.mock import MagicMock
    env = MagicMock(spec=Environment)
    grammar = MagicMock(spec=Grammar)
    drive = MagicMock(spec=DriveSignal)
    memory = MagicMock(spec=Memory)
    memory.get_library.return_value = []
    return Learner(env, grammar, drive, memory)


class TestSelectGuidedPool:
    def test_basic_pool_selection(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores = {
            "mirror_h": 0.02,
            "rotate_90": 0.15,
            "fill_enclosed": 0.30,
            "gravity_down": 0.50,
            "transpose": 0.70,
            "identity": 0.90,
        }
        ctx.enum_candidates = [
            ScoredProgram(
                program=Program(root="outline", children=[Program(root="erode")]),
                energy=0.1, prediction_error=0.05, complexity_cost=0.002,
            ),
        ]

        pool = learner._select_guided_pool(ctx, top_k=4)
        assert len(pool) == 4
        assert pool[0] == "mirror_h"
        assert pool[1] == "rotate_90"
        assert "outline" in pool or "erode" in pool or "fill_enclosed" in pool

    def test_pool_respects_top_k(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores = {f"prim_{i}": i * 0.01 for i in range(50)}
        ctx.enum_candidates = []
        pool = learner._select_guided_pool(ctx, top_k=10)
        assert len(pool) == 10

    def test_pool_empty_scores(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=1000,
            record=False,
        )
        ctx.depth1_scores = {}
        ctx.enum_candidates = []
        pool = learner._select_guided_pool(ctx, top_k=10)
        assert pool == []


class TestGuidedDeepSearch:
    def test_guided_search_returns_none_when_no_budget(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(eval_budget=100, guided_budget_fraction=0.30),
            eval_budget=100,
            record=False,
        )
        ctx.n_evals = 100
        ctx.depth1_scores = {"prim_a": 0.1}
        result = learner._guided_deep_search(ctx)
        assert result is None

    def test_guided_search_returns_none_when_already_solved(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=10000,
            record=False,
        )
        ctx.depth1_scores = {"prim_a": 0.1}
        ctx.best_so_far = ScoredProgram(
            program=Program(root="prim_a"),
            energy=0.001, prediction_error=0.0, complexity_cost=0.001,
            max_example_error=0.0,
        )
        result = learner._guided_deep_search(ctx)
        assert result is None

    def test_guided_search_returns_none_when_no_depth1_scores(self):
        learner = _make_learner()
        ctx = _WakeContext(
            task=Task(task_id="t1", train_examples=[], test_inputs=[]),
            all_prims=[],
            cfg=SearchConfig(),
            eval_budget=10000,
            record=False,
        )
        ctx.depth1_scores = {}
        result = learner._guided_deep_search(ctx)
        assert result is None
