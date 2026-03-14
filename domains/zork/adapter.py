"""
Zork Domain Adapter.
"""

from __future__ import annotations

from typing import Optional

from core import Task
from core.interfaces import DomainAdapter, Environment, Grammar, DriveSignal
from . import ZorkEnv, ZorkGrammar, ZorkDrive, get_sample_tasks


class ZorkAdapter(DomainAdapter):
    """Adapter for the Zork text adventure domain."""

    def name(self) -> str:
        return "zork"

    def create_interfaces(self, seed: int = 42, **kwargs) -> tuple[Environment, Grammar, DriveSignal]:
        return ZorkEnv(), ZorkGrammar(seed=seed), ZorkDrive()

    def load_tasks(self, split: str, data_dir: Optional[str] = None,
                   max_tasks: int = 0) -> list[Task]:
        """Zork uses generated tasks (no external data files)."""
        tasks = get_sample_tasks()
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        return tasks

    def config_defaults(self) -> dict:
        return {
            "workers": 1,
            "compute_cap": 0,
            "exhaustive_depth": 2,
            "exhaustive_pair_top_k": 30,
        }

    def default_cell_size(self) -> int:
        return 100  # action sequences are small
