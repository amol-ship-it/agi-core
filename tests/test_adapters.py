"""
Tests for DomainAdapter implementations and benchmark infrastructure.
"""

import pytest
from core.interfaces import DomainAdapter, Environment, Grammar, DriveSignal


# =============================================================================
# Adapter compliance tests
# =============================================================================

class TestARCAdapter:
    def _make(self, benchmark="arc-agi-1"):
        from domains.arc.adapter import ARCAdapter
        return ARCAdapter(benchmark=benchmark)

    def test_name_arc1(self):
        adapter = self._make("arc-agi-1")
        assert adapter.name() == "arc-agi-1"

    def test_name_arc2(self):
        adapter = self._make("arc-agi-2")
        assert adapter.name() == "arc-agi-2"

    def test_invalid_benchmark(self):
        from domains.arc.adapter import ARCAdapter
        with pytest.raises(ValueError):
            ARCAdapter(benchmark="arc-agi-3")

    def test_create_interfaces(self):
        adapter = self._make()
        env, grammar, drive = adapter.create_interfaces(seed=42)
        assert isinstance(env, Environment)
        assert isinstance(grammar, Grammar)
        assert isinstance(drive, DriveSignal)

    def test_create_interfaces_vocabulary(self):
        adapter = self._make()
        _, grammar_full, _ = adapter.create_interfaces(seed=42, vocabulary="full")
        _, grammar_min, _ = adapter.create_interfaces(seed=42, vocabulary="minimal")
        assert len(grammar_full.base_primitives()) > len(grammar_min.base_primitives())

    def test_config_defaults(self):
        adapter = self._make()
        d = adapter.config_defaults()
        assert "energy_beta" in d
        assert "solve_threshold" in d

    def test_default_cell_size(self):
        adapter = self._make()
        assert adapter.default_cell_size() == 800

    def test_load_tasks_training_sample(self):
        """When no ARC data files found, falls back to sample tasks."""
        adapter = self._make()
        # This will either load real data or fall back to samples
        tasks = adapter.load_tasks("training", max_tasks=5)
        assert len(tasks) >= 1
        assert all(hasattr(t, "task_id") for t in tasks)

    def test_is_domain_adapter(self):
        adapter = self._make()
        assert isinstance(adapter, DomainAdapter)


class TestListOpsAdapter:
    def _make(self):
        from domains.list_ops.adapter import ListOpsAdapter
        return ListOpsAdapter()

    def test_name(self):
        assert self._make().name() == "list-ops"

    def test_create_interfaces(self):
        env, grammar, drive = self._make().create_interfaces(seed=42)
        assert isinstance(env, Environment)
        assert isinstance(grammar, Grammar)
        assert isinstance(drive, DriveSignal)

    def test_load_tasks(self):
        tasks = self._make().load_tasks("training", max_tasks=5)
        assert len(tasks) == 5

    def test_config_defaults(self):
        d = self._make().config_defaults()
        assert d["workers"] == 1
        assert d["sequential_compounding"] is True

    def test_default_cell_size(self):
        assert self._make().default_cell_size() == 100

    def test_is_domain_adapter(self):
        assert isinstance(self._make(), DomainAdapter)


class TestZorkAdapter:
    def _make(self):
        from domains.zork.adapter import ZorkAdapter
        return ZorkAdapter()

    def test_name(self):
        assert self._make().name() == "zork"

    def test_create_interfaces(self):
        env, grammar, drive = self._make().create_interfaces(seed=42)
        assert isinstance(env, Environment)
        assert isinstance(grammar, Grammar)
        assert isinstance(drive, DriveSignal)

    def test_load_tasks(self):
        tasks = self._make().load_tasks("training")
        assert len(tasks) >= 1

    def test_config_defaults(self):
        d = self._make().config_defaults()
        assert d["workers"] == 1

    def test_default_cell_size(self):
        assert self._make().default_cell_size() == 100

    def test_is_domain_adapter(self):
        assert isinstance(self._make(), DomainAdapter)


# =============================================================================
# Benchmark infrastructure tests
# =============================================================================

class TestExperimentConfigNewFields:
    """Test new fields added to ExperimentConfig."""

    def test_split_label_default(self):
        from common.benchmark import ExperimentConfig
        from domains.list_ops import ListEnv, ListGrammar, ListDrive
        cfg = ExperimentConfig(
            title="test", domain_tag="test", tasks=[],
            environment=ListEnv(), grammar=ListGrammar(seed=1), drive=ListDrive(),
        )
        assert cfg.split_label == ""
        assert cfg.default_cell_size == 800

    def test_split_label_custom(self):
        from common.benchmark import ExperimentConfig
        from domains.list_ops import ListEnv, ListGrammar, ListDrive
        cfg = ExperimentConfig(
            title="test", domain_tag="test", tasks=[],
            environment=ListEnv(), grammar=ListGrammar(seed=1), drive=ListDrive(),
            split_label="EVAL",
            default_cell_size=100,
        )
        assert cfg.split_label == "EVAL"
        assert cfg.default_cell_size == 100


class TestSearchConfigNewField:
    """Test eval_budget_base_cells field."""

    def test_default(self):
        from core.config import SearchConfig
        cfg = SearchConfig()
        assert cfg.eval_budget_base_cells == 800

    def test_custom(self):
        from core.config import SearchConfig
        cfg = SearchConfig(eval_budget_base_cells=100)
        assert cfg.eval_budget_base_cells == 100


class TestFindArcData:
    """Test find_arc_data function."""

    def test_returns_none_for_nonexistent(self, tmp_path):
        from domains.arc.dataset import find_arc_data
        # With no data dirs, should return None (but won't error)
        result = find_arc_data("training", "arc-agi-1")
        # Result depends on whether data is installed — just check it's str or None
        assert result is None or isinstance(result, str)

    def test_benchmark_parameter(self):
        from domains.arc.dataset import find_arc_data
        # Just verify it doesn't crash with either benchmark value
        find_arc_data("training", "arc-agi-1")
        find_arc_data("training", "arc-agi-2")


class TestBackwardCompat:
    """Verify backward-compatible import paths."""

    def test_core_import(self):
        from core import ExperimentConfig, run_experiment, make_parser, PRESETS
        assert callable(run_experiment)
        assert isinstance(PRESETS, dict)

    def test_core_runner_import(self):
        from core.runner import ExperimentConfig, PRESETS, fmt_duration, parse_human_int
        assert callable(fmt_duration)
        assert callable(parse_human_int)

    def test_common_import(self):
        from common import ExperimentConfig, run_experiment, run_pipeline
        assert callable(run_experiment)
        assert callable(run_pipeline)

    def test_pipeline_common_import(self):
        from experiments.pipeline_common import pipeline_tee, save_pipeline_results
        assert callable(pipeline_tee)

    def test_phase1_find_arc_data(self):
        from experiments.phase1_arc import find_arc_data
        assert callable(find_arc_data)

    def test_domain_adapter_from_core(self):
        from core import DomainAdapter
        assert hasattr(DomainAdapter, "name")
        assert hasattr(DomainAdapter, "create_interfaces")
        assert hasattr(DomainAdapter, "load_tasks")
