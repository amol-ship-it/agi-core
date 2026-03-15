"""
ARC-AGI Domain Adapter.

Provides ARCAdapter for both ARC-AGI-1 and ARC-AGI-2 benchmarks.
Consolidates: task loading, data path search, viz generation, config defaults.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from core import Task
from core.interfaces import DomainAdapter, Environment, Grammar, DriveSignal
from .environment import ARCEnv
from .grammar import ARCGrammar
from .drive import ARCDrive
from .dataset import load_arc_dataset, make_sample_tasks, find_arc_data


class ARCAdapter(DomainAdapter):
    """Adapter for ARC-AGI-1 and ARC-AGI-2 benchmarks.

    Args:
        benchmark: 'arc-agi-1' or 'arc-agi-2'
    """

    def __init__(self, benchmark: str = "arc-agi-1"):
        if benchmark not in ("arc-agi-1", "arc-agi-2"):
            raise ValueError(f"Unknown benchmark: {benchmark!r}")
        self.benchmark = benchmark

    def name(self) -> str:
        return self.benchmark

    def create_interfaces(self, seed: int = 42, **kwargs) -> tuple[Environment, Grammar, DriveSignal]:
        return ARCEnv(), ARCGrammar(seed=seed, vocabulary="atomic"), ARCDrive()

    def load_tasks(self, split: str, data_dir: Optional[str] = None,
                   max_tasks: int = 0) -> list[Task]:
        """Load ARC tasks for the given split.

        For arc-agi-2 training: tries ARC-AGI-2 first, falls back to ARC-AGI-1.
        For training with no data found: uses built-in sample tasks.
        """
        if self.benchmark == "arc-agi-2":
            return self._load_arc2_tasks(split, data_dir, max_tasks)
        return self._load_arc1_tasks(split, data_dir, max_tasks)

    def _load_arc1_tasks(self, split, data_dir, max_tasks):
        data_dir = data_dir or find_arc_data(split, "arc-agi-1")
        if data_dir:
            print(f"  Loading ARC-AGI {split} tasks from {data_dir}...")
            tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
            print(f"  Loaded {len(tasks)} tasks")
        elif split == "training":
            print("  ARC dataset not found. Using built-in sample tasks.")
            print("    (git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI)")
            tasks = make_sample_tasks()
            if max_tasks > 0:
                tasks = tasks[:max_tasks]
            print(f"  Created {len(tasks)} sample ARC tasks")
        else:
            print(f"  ERROR: {split.capitalize()} data not found. Searched:")
            from .dataset import ARC1_DATA_SEARCH_PATHS
            for p in ARC1_DATA_SEARCH_PATHS:
                print(f"    {p.format(split=split)}")
            sys.exit(1)
        if not tasks:
            print("  ERROR: No tasks loaded.")
            sys.exit(1)
        return tasks

    def _load_arc2_tasks(self, split, data_dir, max_tasks):
        if split == "training":
            return self._load_arc2_train_tasks(data_dir, max_tasks)

        # Evaluation: must be ARC-AGI-2 data
        data_dir = data_dir or find_arc_data(split, "arc-agi-2")
        if data_dir:
            print(f"  Loading ARC-AGI-2 {split} tasks from {data_dir}...")
            tasks = load_arc_dataset(data_dir, max_tasks=max_tasks)
            print(f"  Loaded {len(tasks)} tasks")
        else:
            print(f"  ERROR: ARC-AGI-2 {split} data not found. Searched:")
            from .dataset import ARC2_DATA_SEARCH_PATHS
            for p in ARC2_DATA_SEARCH_PATHS:
                print(f"    {p.format(split=split)}")
            print("\n  To get the data:")
            print("    git clone https://github.com/arcprize/arc-agi.git data/ARC-AGI-2")
            sys.exit(1)
        if not tasks:
            print("  ERROR: No tasks loaded.")
            sys.exit(1)
        return tasks

    def _load_arc2_train_tasks(self, data_dir, max_tasks):
        """Load training tasks — prefer ARC-AGI-2 training, fall back to ARC-AGI-1."""
        arc2_dir = data_dir or find_arc_data("training", "arc-agi-2")
        if arc2_dir:
            print(f"  Loading ARC-AGI-2 training tasks from {arc2_dir}...")
            tasks = load_arc_dataset(arc2_dir, max_tasks=max_tasks)
            print(f"  Loaded {len(tasks)} tasks")
            return tasks

        arc1_dir = find_arc_data("training", "arc-agi-1")
        if arc1_dir:
            print(f"  ARC-AGI-2 training not found, falling back to ARC-AGI-1...")
            print(f"  Loading ARC-AGI-1 training tasks from {arc1_dir}...")
            tasks = load_arc_dataset(arc1_dir, max_tasks=max_tasks)
            print(f"  Loaded {len(tasks)} tasks")
            return tasks

        print("  ARC training data not found. Using built-in sample tasks.")
        tasks = make_sample_tasks()
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        print(f"  Created {len(tasks)} sample ARC tasks")
        return tasks

    def config_defaults(self) -> dict:
        return {
            "energy_beta": 0.002,
            "solve_threshold": 0.001,
            "mutations_per_candidate": 2,
            "crossover_fraction": 0.3,
        }

    def default_cell_size(self) -> int:
        return 800  # median ARC grid size

    def post_run_hooks(self, result) -> list[str]:
        """Generate HTML visualization from results."""
        if not result:
            return []
        results_path = result.results_path
        if not results_path or not os.path.exists(results_path):
            return []
        try:
            from experiments.visualize_results import generate_html
            output_base = os.path.splitext(results_path)[0]
            return generate_html(results_path, output_base)
        except Exception as e:
            import traceback
            print(f"  (visualization failed for {results_path}: {e})")
            traceback.print_exc()
            return []
