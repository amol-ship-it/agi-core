"""
List Operations Domain Adapter.
"""

from __future__ import annotations

from typing import Optional

from core import Task
from core.interfaces import DomainAdapter, Environment, Grammar, DriveSignal
from . import ListEnv, ListGrammar, ListDrive, get_sample_tasks


class ListOpsAdapter(DomainAdapter):
    """Adapter for the list operations domain."""

    def name(self) -> str:
        return "list-ops"

    def create_interfaces(self, seed: int = 42, **kwargs) -> tuple[Environment, Grammar, DriveSignal]:
        return ListEnv(), ListGrammar(seed=seed), ListDrive()

    def load_tasks(self, split: str, data_dir: Optional[str] = None,
                   max_tasks: int = 0) -> list[Task]:
        """List ops uses generated tasks (no external data files)."""
        tasks = get_sample_tasks(seed=42)
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        return tasks

    def config_defaults(self) -> dict:
        return {
            "workers": 1,
            "compute_cap": 0,
            "exhaustive_depth": 2,
            "exhaustive_pair_top_k": 22,
            "sequential_compounding": True,
        }

    def default_cell_size(self) -> int:
        return 100  # lists are small
