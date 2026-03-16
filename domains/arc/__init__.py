"""
ARC-AGI Domain Plugin.

Implements the 4 interfaces for the ARC-AGI grid transformation domain.
Grid representation: list[list[int]] where each int is a color 0-9.
    0 = black (background), 1-9 = colors.

Module layout:
    transformation_primitives.py - Atomic Grid→Grid transforms + parameterized factories
    perception_primitives.py     - Atomic Grid→Value perception
    primitives.py                - Registry (_PRIM_MAP) + utilities (to_np, from_np)
    objects.py                   - Connected component detection
    analysis.py                  - Deterministic I/O analysis for phase ordering
    environment.py               - ARCEnv (execute programs on grids)
    grammar.py                   - ARCGrammar (composition, mutation, crossover)
    drive.py                     - ARCDrive (scoring: pixel accuracy + complexity)
    dataset.py                   - Task loading from ARC-AGI JSON files + sample tasks
"""

from .primitives import Grid, to_np, from_np
from .environment import ARCEnv
from .grammar import ARCGrammar
from .drive import ARCDrive
from .dataset import load_arc_task, load_arc_dataset, make_sample_tasks, find_arc_data

__all__ = [
    "ARCEnv", "ARCGrammar", "ARCDrive",
    "load_arc_task", "load_arc_dataset", "make_sample_tasks", "find_arc_data",
    "Grid", "to_np", "from_np",
]
