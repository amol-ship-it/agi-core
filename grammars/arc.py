"""
Backward-compatible shim: re-exports from domains.arc.

The actual implementation has moved to domains/arc/ with proper module split:
    domains/arc/primitives.py   - Grid→Grid transforms + registry
    domains/arc/objects.py      - Connected component detection
    domains/arc/environment.py  - ARCEnv
    domains/arc/grammar.py      - ARCGrammar
    domains/arc/drive.py        - ARCDrive
    domains/arc/dataset.py      - Task loading + sample tasks
"""

# Re-export everything from the new location
from domains.arc.primitives import *  # noqa: F401,F403
from domains.arc.primitives import (  # noqa: F401 — explicit private exports for tests
    _find_connected_components,
    _make_keep_color,
    _make_replace_color,
    _PRIM_MAP,
    _build_arc_primitives,
)
from domains.arc.environment import ARCEnv  # noqa: F401
from domains.arc.grammar import ARCGrammar  # noqa: F401
from domains.arc.drive import ARCDrive  # noqa: F401
from domains.arc.dataset import (  # noqa: F401
    load_arc_task, load_arc_dataset, make_sample_tasks,
)
