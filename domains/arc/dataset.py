"""
ARC-AGI dataset loading, sample tasks, and data directory auto-detection.
"""

from __future__ import annotations

import json
import os
import sys

from core import Task


def load_arc_task(path: str) -> Task:
    """
    Load a single ARC-AGI task from a JSON file.

    ARC JSON format:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test":  [{"input": [[...]], "output": [[...]]}, ...]
    }
    """
    with open(path) as f:
        data = json.load(f)

    task_id = os.path.splitext(os.path.basename(path))[0]

    train_examples = [(ex["input"], ex["output"]) for ex in data["train"]]
    test_inputs = [ex["input"] for ex in data["test"]]
    test_outputs = [ex.get("output", []) for ex in data["test"]]

    # Estimate difficulty: more training examples + larger grids = harder
    avg_size = sum(
        len(inp) * len(inp[0]) for inp, _ in train_examples
    ) / max(1, len(train_examples))
    difficulty = avg_size / 9.0  # normalize roughly

    return Task(
        task_id=task_id,
        train_examples=train_examples,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        difficulty=difficulty,
        metadata={"path": path},
    )


def load_arc_dataset(directory: str, max_tasks: int = 0) -> list[Task]:
    """Load all ARC-AGI tasks from a directory of JSON files."""
    tasks = []
    json_files = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
    if max_tasks > 0:
        json_files = json_files[:max_tasks]

    for fname in json_files:
        path = os.path.join(directory, fname)
        try:
            task = load_arc_task(path)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: skipping {fname}: {e}")

    return tasks


# =============================================================================
# Built-in sample tasks for testing without ARC data files
# =============================================================================

# Inline helper functions for sample task generation (no dependency on primitives)

def _rotate_90_cw(grid):
    if not grid:
        return grid
    return [list(row) for row in zip(*grid[::-1])]

def _mirror_horizontal(grid):
    if not grid:
        return grid
    return [row[::-1] for row in grid]

def _mirror_vertical(grid):
    if not grid:
        return grid
    return list(reversed([row[:] for row in grid]))

def _trim_rows(grid):
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    top = 0
    while top < h and all(c == 0 for c in grid[top]):
        top += 1
    bot = h - 1
    while bot >= top and all(c == 0 for c in grid[bot]):
        bot -= 1
    if top > bot:
        return grid
    return [row[:] for row in grid[top:bot + 1]]

def _trim_cols(grid):
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    left = 0
    while left < w and all(grid[r][left] == 0 for r in range(h)):
        left += 1
    right = w - 1
    while right >= left and all(grid[r][right] == 0 for r in range(h)):
        right -= 1
    if left > right:
        return grid
    return [row[left:right + 1] for row in grid]

def _invert_colors(grid):
    return [[9 - c for c in row] for row in grid]

def _gravity_down(grid):
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        non_zero = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            result[h - len(non_zero) + i][c] = val
    return result


def make_sample_tasks() -> list[Task]:
    """
    Create a small set of hand-crafted ARC-like tasks for testing.
    These test basic geometric and color transforms.
    """
    tasks = []

    # Task 1: Rotate 90 CW (easy)
    grid1_in = [[1, 0], [0, 2]]
    grid1_out = _rotate_90_cw(grid1_in)
    tasks.append(Task(
        task_id="sample_rot90",
        train_examples=[
            (grid1_in, grid1_out),
            ([[3, 0, 0], [0, 4, 0], [0, 0, 5]], _rotate_90_cw([[3, 0, 0], [0, 4, 0], [0, 0, 5]])),
        ],
        test_inputs=[[[1, 2], [3, 4]]],
        test_outputs=[_rotate_90_cw([[1, 2], [3, 4]])],
        difficulty=1.0,
    ))

    # Task 2: Mirror horizontal (easy)
    grid2_in = [[1, 2, 3], [4, 5, 6]]
    grid2_out = _mirror_horizontal(grid2_in)
    tasks.append(Task(
        task_id="sample_mirror_h",
        train_examples=[
            (grid2_in, grid2_out),
            ([[7, 0, 8]], _mirror_horizontal([[7, 0, 8]])),
        ],
        test_inputs=[[[1, 0], [0, 2]]],
        test_outputs=[_mirror_horizontal([[1, 0], [0, 2]])],
        difficulty=1.0,
    ))

    # Task 3: Crop to content (medium — needs depth-2: trim_cols(trim_rows(x)))
    grid3_in = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
    grid3_out = [[1, 2], [3, 4]]
    tasks.append(Task(
        task_id="sample_crop",
        train_examples=[
            (grid3_in, grid3_out),
            ([[0, 0, 0], [0, 5, 0], [0, 0, 0]], [[5]]),
        ],
        test_inputs=[[[0, 0, 0], [0, 7, 8], [0, 0, 0]]],
        test_outputs=[[[7, 8]]],
        difficulty=2.0,
    ))

    # Task 4: Mirror vertical (easy)
    grid4_in = [[1, 2], [3, 4], [5, 6]]
    grid4_out = _mirror_vertical(grid4_in)
    tasks.append(Task(
        task_id="sample_mirror_v",
        train_examples=[
            (grid4_in, grid4_out),
            ([[9, 8], [7, 6]], _mirror_vertical([[9, 8], [7, 6]])),
        ],
        test_inputs=[[[1, 0, 2], [0, 3, 0]]],
        test_outputs=[_mirror_vertical([[1, 0, 2], [0, 3, 0]])],
        difficulty=1.0,
    ))

    # Task 5: Gravity down (medium)
    grid5_in = [[1, 0, 2], [0, 3, 0], [0, 0, 0]]
    grid5_out = _gravity_down(grid5_in)
    tasks.append(Task(
        task_id="sample_gravity_down",
        train_examples=[
            (grid5_in, grid5_out),
            ([[0, 4, 0], [5, 0, 6], [0, 0, 0]], _gravity_down([[0, 4, 0], [5, 0, 6], [0, 0, 0]])),
        ],
        test_inputs=[[[7, 0, 0], [0, 0, 8], [0, 9, 0]]],
        test_outputs=[_gravity_down([[7, 0, 0], [0, 0, 8], [0, 9, 0]])],
        difficulty=2.5,
    ))

    # Task 6: Fill enclosed regions (harder)
    grid6_in = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    grid6_out = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    tasks.append(Task(
        task_id="sample_fill_enclosed",
        train_examples=[
            (grid6_in, grid6_out),
            ([[2, 2, 2, 2], [2, 0, 0, 2], [2, 2, 2, 2]], [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]),
        ],
        test_inputs=[[[3, 3, 3], [3, 0, 3], [3, 0, 3], [3, 3, 3]]],
        test_outputs=[[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]],
        difficulty=3.0,
    ))

    # Task 7: Compose rot90 + mirror_h (harder — needs depth-2 program)
    def rot_then_mirror(g):
        return _mirror_horizontal(_rotate_90_cw(g))
    tasks.append(Task(
        task_id="sample_rot_mirror",
        train_examples=[
            ([[1, 0], [0, 2]], rot_then_mirror([[1, 0], [0, 2]])),
            ([[3, 4, 5], [6, 7, 8]], rot_then_mirror([[3, 4, 5], [6, 7, 8]])),
        ],
        test_inputs=[[[1, 2, 3]]],
        test_outputs=[rot_then_mirror([[1, 2, 3]])],
        difficulty=4.0,
    ))

    # Task 8: Invert + trim (harder — needs depth-2)
    def invert_trim(g):
        return _trim_cols(_invert_colors(g))
    tasks.append(Task(
        task_id="sample_invert_trim",
        train_examples=[
            ([[0, 5, 0], [0, 3, 0]], invert_trim([[0, 5, 0], [0, 3, 0]])),
            ([[0, 0, 7, 0], [0, 0, 2, 0]], invert_trim([[0, 0, 7, 0], [0, 0, 2, 0]])),
        ],
        test_inputs=[[[0, 1, 0], [0, 4, 0], [0, 6, 0]]],
        test_outputs=[invert_trim([[0, 1, 0], [0, 4, 0], [0, 6, 0]])],
        difficulty=4.5,
    ))

    return tasks


# =============================================================================
# Data directory auto-detection
# =============================================================================

ARC1_DATA_SEARCH_PATHS = [
    "data/ARC-AGI/data/{split}",
    "../ARC-AGI/data/{split}",
    os.path.expanduser("~/ARC-AGI/data/{split}"),
    "data/arc-agi/data/{split}",
]

ARC2_DATA_SEARCH_PATHS = [
    "data/ARC-AGI-2/data/{split}",
    "../ARC-AGI-2/data/{split}",
    os.path.expanduser("~/ARC-AGI-2/data/{split}"),
    "data/arc-agi-2/data/{split}",
]


def find_arc_data(split: str = "training", benchmark: str = "arc-agi-1") -> str | None:
    """Auto-detect ARC data directory."""
    paths = ARC2_DATA_SEARCH_PATHS if benchmark == "arc-agi-2" else ARC1_DATA_SEARCH_PATHS
    for pattern in paths:
        path = pattern.format(split=split)
        if os.path.isdir(path):
            return path
    return None
