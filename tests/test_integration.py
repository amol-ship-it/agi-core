"""
Integration / smoke tests for the full pipeline.

These tests run the actual benchmark infrastructure end-to-end on
small task sets to catch regressions in CLI wiring, result serialization,
progress tracking, and pipeline orchestration.
"""

import json
import os
import tempfile

import pytest

# run_experiment enforces PYTHONHASHSEED=0 by relaunching the process.
# Set it before importing so the check passes without a relaunch.
os.environ["PYTHONHASHSEED"] = "0"

from common.benchmark import (
    ExperimentConfig,
    run_experiment,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_arc_config(max_tasks=3, **overrides):
    """Build a minimal ExperimentConfig for ARC with sample tasks."""
    from domains.arc.adapter import ARCAdapter
    adapter = ARCAdapter(benchmark="arc-agi-1")
    env, grammar, drive = adapter.create_interfaces(seed=42)
    tasks = adapter.load_tasks("training", max_tasks=max_tasks)
    defaults = adapter.config_defaults()
    return ExperimentConfig(
        title="Integration Test",
        domain_tag="integration_test",
        tasks=tasks,
        environment=env,
        grammar=grammar,
        drive=drive,
        rounds=1,
        beam_width=1,
        max_generations=1,
        workers=1,
        seed=42,
        compute_cap=500_000,
        energy_beta=defaults.get("energy_beta", 0.001),
        solve_threshold=defaults.get("solve_threshold", 0.001),
        exhaustive_depth=2,
        exhaustive_pair_top_k=10,
        exhaustive_triple_top_k=5,
        runs_dir=tempfile.mkdtemp(),
        no_log=True,
        mode="quick",
        default_cell_size=adapter.default_cell_size(),
        **overrides,
    )


def _make_list_config(max_tasks=3, **overrides):
    """Build a minimal ExperimentConfig for list_ops with sample tasks."""
    from domains.list_ops.adapter import ListOpsAdapter
    adapter = ListOpsAdapter()
    env, grammar, drive = adapter.create_interfaces(seed=42)
    tasks = adapter.load_tasks("training", max_tasks=max_tasks)
    defaults = adapter.config_defaults()
    return ExperimentConfig(
        title="Integration Test List",
        domain_tag="integration_test_list",
        tasks=tasks,
        environment=env,
        grammar=grammar,
        drive=drive,
        rounds=1,
        beam_width=1,
        max_generations=1,
        workers=1,
        seed=42,
        compute_cap=500_000,
        energy_beta=defaults.get("energy_beta", 0.001),
        solve_threshold=defaults.get("solve_threshold", 0.001),
        exhaustive_depth=2,
        exhaustive_pair_top_k=10,
        exhaustive_triple_top_k=5,
        runs_dir=tempfile.mkdtemp(),
        no_log=True,
        mode="quick",
        default_cell_size=adapter.default_cell_size(),
        **overrides,
    )


# =============================================================================
# Smoke tests — run_experiment end-to-end
# =============================================================================

class TestRunExperimentARC:
    """End-to-end smoke tests for the ARC domain via run_experiment."""

    def test_run_produces_result(self):
        cfg = _make_arc_config(max_tasks=3)
        result = run_experiment(cfg)
        assert result is not None
        assert hasattr(result, "results_path")

    def test_result_json_valid(self):
        cfg = _make_arc_config(max_tasks=3)
        result = run_experiment(cfg)
        # The results JSON should exist and be valid
        assert os.path.exists(result.results_path)
        with open(result.results_path) as f:
            data = json.load(f)
        assert "meta" in data
        assert "summary" in data
        assert "tasks" in data

    def test_result_summary_fields(self):
        cfg = _make_arc_config(max_tasks=3)
        result = run_experiment(cfg)
        with open(result.results_path) as f:
            data = json.load(f)
        summary = data["summary"]
        assert "n_tasks" in summary
        assert "last_round_solved" in summary
        assert "last_round_train_solved" in summary
        assert summary["n_tasks"] == 3

    def test_result_tasks_populated(self):
        cfg = _make_arc_config(max_tasks=3)
        result = run_experiment(cfg)
        with open(result.results_path) as f:
            data = json.load(f)
        assert len(data["tasks"]) == 3
        # Each task should have required fields
        for task_id, task_data in data["tasks"].items():
            assert "solved" in task_data
            assert "train_solved" in task_data
            assert "energy" in task_data or "prediction_error" in task_data

    def test_jsonl_output_exists(self):
        cfg = _make_arc_config(max_tasks=3)
        result = run_experiment(cfg)
        # Find the .jsonl file in the same directory
        results_dir = os.path.dirname(result.results_path)
        jsonl_files = [f for f in os.listdir(results_dir) if f.endswith(".jsonl")]
        assert len(jsonl_files) >= 1
        # Each line should be valid JSON
        jsonl_path = os.path.join(results_dir, jsonl_files[0])
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)  # should not raise


class TestRunExperimentListOps:
    """End-to-end smoke test for the list_ops domain."""

    def test_run_produces_result(self):
        cfg = _make_list_config(max_tasks=3)
        result = run_experiment(cfg)
        assert result is not None
        with open(result.results_path) as f:
            data = json.load(f)
        assert data["summary"]["n_tasks"] == 3


class TestRunExperimentZork:
    """End-to-end smoke test for the Zork domain."""

    def test_run_produces_result(self):
        from domains.zork.adapter import ZorkAdapter
        adapter = ZorkAdapter()
        env, grammar, drive = adapter.create_interfaces(seed=42)
        tasks = adapter.load_tasks("training", max_tasks=3)
        defaults = adapter.config_defaults()
        cfg = ExperimentConfig(
            title="Integration Test Zork",
            domain_tag="integration_test_zork",
            tasks=tasks,
            environment=env,
            grammar=grammar,
            drive=drive,
            rounds=1,
            beam_width=1,
            max_generations=1,
            workers=1,
            seed=42,
            compute_cap=500_000,
            energy_beta=defaults.get("energy_beta", 0.001),
            solve_threshold=defaults.get("solve_threshold", 0.001),
            exhaustive_depth=2,
            exhaustive_pair_top_k=10,
            exhaustive_triple_top_k=5,
            runs_dir=tempfile.mkdtemp(),
            no_log=True,
            mode="quick",
            default_cell_size=adapter.default_cell_size(),
        )
        result = run_experiment(cfg)
        assert result is not None
        with open(result.results_path) as f:
            data = json.load(f)
        assert data["summary"]["n_tasks"] == 3


# =============================================================================
# CLI entry point smoke test
# =============================================================================

class TestUnifiedCLI:
    """Test that the unified CLI module loads correctly."""

    def test_domain_adapters_loadable(self):
        """All registered domain adapters can be lazy-loaded."""
        from common.__main__ import DOMAIN_ADAPTERS, _load_adapter
        for domain_name in DOMAIN_ADAPTERS:
            adapter = _load_adapter(domain_name)
            assert adapter.name() == domain_name
