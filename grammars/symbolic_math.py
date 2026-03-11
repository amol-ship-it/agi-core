"""
Backward-compatible shim: re-exports from domains.symbolic_math.
"""
from domains.symbolic_math import *  # noqa: F401,F403
from domains.symbolic_math import (  # noqa: F401
    _safe_div, _safe_log, _safe_sqrt, _safe_exp,
    _PRIM_MAP, _eval_tree_raw, _collect_const_nodes,
    optimize_constants,
)
