"""
ARC-AGI Domain Plugin.

Implements the 4 interfaces for the ARC-AGI grid transformation domain.
Grid representation: list[list[int]] where each int is a color 0-9.
    0 = black (background), 1-9 = colors.

Module layout:
    primitives.py   - All Grid→Grid transform functions + primitive registry
    objects.py      - Connected component detection + object-level ops
    environment.py  - ARCEnv (execute programs on grids)
    grammar.py      - ARCGrammar (composition, mutation, crossover)
    drive.py        - ARCDrive (scoring: pixel accuracy + complexity)
    dataset.py      - Task loading from ARC-AGI JSON files + sample tasks
"""

from .primitives import ARC_PRIMITIVES, Grid, to_np, from_np
from .environment import ARCEnv
from .grammar import ARCGrammar
from .drive import ARCDrive
from .dataset import load_arc_task, load_arc_dataset, make_sample_tasks, find_arc_data

__all__ = [
    "ARCEnv", "ARCGrammar", "ARCDrive",
    "ARC_PRIMITIVES",
    "load_arc_task", "load_arc_dataset", "make_sample_tasks", "find_arc_data",
    "Grid", "to_np", "from_np",
]
