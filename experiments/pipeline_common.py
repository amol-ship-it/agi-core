"""
Backward-compatibility shim: re-exports from common.benchmark.

All pipeline utilities now live in common/benchmark.py.
"""

from common.benchmark import (  # noqa: F401
    pipeline_tee,
    save_pipeline_results,
    print_pipeline_summary,
)
