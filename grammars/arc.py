"""
Domain plugin: ARC-AGI Grid Transformations.

Implements all 4 interfaces for the ARC-AGI domain.
Imports ONLY from core — no domain cross-contamination.

Grid representation: list[list[int]] where each int is a color 0-9.
    0 = black (background), 1-9 = colors.

Primitives adapted from vibhor-77/agi-mvp-general.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import Counter
from typing import Any, Optional

import numpy as np

from core import (
    Environment,
    Grammar,
    DriveSignal,
    Primitive,
    Program,
    Task,
    Observation,
    LibraryEntry,
)

# Type alias for grids
Grid = list[list[int]]


# =============================================================================
# Grid utilities
# =============================================================================

def to_np(grid: Grid) -> np.ndarray:
    """Convert list-of-lists grid to numpy array."""
    return np.array(grid, dtype=np.int32)


def from_np(arr: np.ndarray) -> Grid:
    """Convert numpy array back to list-of-lists."""
    return arr.tolist()


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) of a grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid[0] else 0)


def valid_grid(grid: Grid) -> bool:
    """Check if a grid is valid (non-empty, rectangular, values 0-9)."""
    if not grid or not grid[0]:
        return False
    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            return False
        for v in row:
            if not (0 <= v <= 9):
                return False
    return True


def empty_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    """Create an empty grid filled with a single color."""
    return [[fill] * cols for _ in range(rows)]


# =============================================================================
# ARC Grid Primitives — the atomic building blocks
# =============================================================================
# Each primitive takes a Grid and returns a Grid.
# Adapted from agi-mvp-general/arc_agent/primitives.py

def identity(grid: Grid) -> Grid:
    """No-op: return the grid unchanged."""
    return [row[:] for row in grid]


# --- Geometric transforms ---

def rotate_90_cw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=-1))


def rotate_90_ccw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees counter-clockwise."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=1))


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    arr = to_np(grid)
    return from_np(np.rot90(arr, k=2))


def mirror_horizontal(grid: Grid) -> Grid:
    """Flip grid left-right."""
    return [row[::-1] for row in grid]


def mirror_vertical(grid: Grid) -> Grid:
    """Flip grid top-bottom."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Transpose grid (swap rows and columns)."""
    arr = to_np(grid)
    return from_np(arr.T)


# --- Color transforms ---

def invert_colors(grid: Grid) -> Grid:
    """Invert all non-zero colors: c -> 10 - c (0 stays 0)."""
    return [[0 if c == 0 else 10 - c for c in row] for row in grid]


def replace_bg_with_most_common(grid: Grid) -> Grid:
    """Replace background (0) with the most common non-zero color."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return [row[:] for row in grid]
    mc = Counter(flat).most_common(1)[0][0]
    return [[mc if c == 0 else c for c in row] for row in grid]


def keep_color(grid: Grid, color: int) -> Grid:
    """Keep only pixels of the specified color, zero everything else."""
    return [[c if c == color else 0 for c in row] for row in grid]


def remove_color(grid: Grid, color: int) -> Grid:
    """Remove all pixels of the specified color (set to 0)."""
    return [[0 if c == color else c for c in row] for row in grid]


def most_common_color(grid: Grid) -> int:
    """Return the most common non-zero color in the grid."""
    flat = [c for row in grid for c in row if c != 0]
    if not flat:
        return 0
    return Counter(flat).most_common(1)[0][0]


def fill_color(grid: Grid, color: int) -> Grid:
    """Fill entire grid with a single color."""
    rows, cols = grid_shape(grid)
    return empty_grid(rows, cols, color)


# --- Color-parameterized primitives (curried for colors 1-9) ---

def _make_replace_color(from_c: int, to_c: int):
    """Create a primitive that replaces from_c with to_c."""
    def fn(grid: Grid) -> Grid:
        return [[to_c if c == from_c else c for c in row] for row in grid]
    fn.__name__ = f"replace_{from_c}_with_{to_c}"
    return fn


def _make_keep_color(color: int):
    """Create a primitive that keeps only the given color."""
    def fn(grid: Grid) -> Grid:
        return [[c if c == color else 0 for c in row] for row in grid]
    fn.__name__ = f"keep_color_{color}"
    return fn


# --- Spatial / cropping ---

def crop_to_nonzero(grid: Grid) -> Grid:
    """Crop to the bounding box of all non-zero pixels."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        return [[0]]
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return from_np(arr[r_min:r_max + 1, c_min:c_max + 1])


def crop_to_color(grid: Grid, color: int) -> Grid:
    """Crop to the bounding box of a specific color."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr == color)
    if nonzero.size == 0:
        return [[0]]
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return from_np(arr[r_min:r_max + 1, c_min:c_max + 1])


def get_top_half(grid: Grid) -> Grid:
    """Return the top half of the grid."""
    h = len(grid)
    return [row[:] for row in grid[:h // 2]]


def get_bottom_half(grid: Grid) -> Grid:
    """Return the bottom half of the grid."""
    h = len(grid)
    return [row[:] for row in grid[h // 2:]]


def get_left_half(grid: Grid) -> Grid:
    """Return the left half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[:w // 2] for row in grid]


def get_right_half(grid: Grid) -> Grid:
    """Return the right half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[w // 2:] for row in grid]


# --- Tiling / scaling ---

def tile_2x2(grid: Grid) -> Grid:
    """Tile the grid 2x2."""
    arr = to_np(grid)
    return from_np(np.tile(arr, (2, 2)))


def tile_3x3(grid: Grid) -> Grid:
    """Tile the grid 3x3."""
    arr = to_np(grid)
    return from_np(np.tile(arr, (3, 3)))


def scale_2x(grid: Grid) -> Grid:
    """Scale each pixel to a 2x2 block."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1))


def scale_3x(grid: Grid) -> Grid:
    """Scale each pixel to a 3x3 block."""
    arr = to_np(grid)
    return from_np(np.repeat(np.repeat(arr, 3, axis=0), 3, axis=1))


# --- Gravity ---

def gravity_down(grid: Grid) -> Grid:
    """Drop all non-zero pixels to the bottom of each column."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for c in range(cols):
        col = arr[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[rows - len(nonzero):, c] = nonzero
    return from_np(result)


def gravity_up(grid: Grid) -> Grid:
    """Push all non-zero pixels to the top of each column."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for c in range(cols):
        col = arr[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[:len(nonzero), c] = nonzero
    return from_np(result)


def gravity_left(grid: Grid) -> Grid:
    """Push all non-zero pixels to the left of each row."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for r in range(rows):
        row = arr[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, :len(nonzero)] = nonzero
    return from_np(result)


def gravity_right(grid: Grid) -> Grid:
    """Push all non-zero pixels to the right of each row."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    for r in range(rows):
        row = arr[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, cols - len(nonzero):] = nonzero
    return from_np(result)


# --- Pattern detection / filling ---

def outline(grid: Grid) -> Grid:
    """Draw outline around non-zero regions (mark boundary pixels)."""
    arr = to_np(grid)
    nonzero = arr != 0
    # A pixel is boundary if it's non-zero AND at least one 4-neighbor is zero or OOB
    padded = np.pad(arr, 1, mode='constant', constant_values=0)
    neighbor_zero = (
        (padded[:-2, 1:-1] == 0) |  # top
        (padded[2:, 1:-1] == 0) |    # bottom
        (padded[1:-1, :-2] == 0) |   # left
        (padded[1:-1, 2:] == 0)      # right
    )
    result = np.where(nonzero & neighbor_zero, arr, 0)
    return from_np(result)


def fill_enclosed(grid: Grid) -> Grid:
    """Fill enclosed background regions with the surrounding color."""
    arr = to_np(grid).copy()
    rows, cols = arr.shape
    # Flood fill from borders to find non-enclosed background
    visited = np.zeros((rows, cols), dtype=bool)
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and arr[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Fill enclosed zeros: use most common non-zero neighbor color
    enclosed = (arr == 0) & ~visited
    if not np.any(enclosed):
        return from_np(arr)

    # For enclosed cells, find the most common adjacent non-zero color
    mc = most_common_color(grid)
    padded = np.pad(arr, 1, mode='constant', constant_values=0)
    enc_rows, enc_cols = np.where(enclosed)
    for r, c in zip(enc_rows, enc_cols):
        # Check 4-neighbors in padded array (offset by 1)
        neighbors = []
        for nr, nc in [(r, c+1), (r+2, c+1), (r+1, c), (r+1, c+2)]:
            v = padded[nr, nc]
            if v != 0:
                neighbors.append(v)
        arr[r, c] = Counter(neighbors).most_common(1)[0][0] if neighbors else mc
    return from_np(arr)


def denoise_3x3(grid: Grid) -> Grid:
    """Replace each pixel with the majority color in its 3x3 neighborhood."""
    arr = to_np(grid)
    rows, cols = arr.shape
    result = arr.copy()
    # For each color 0-9, count occurrences in each cell's 3x3 neighborhood
    # using convolution-like approach with numpy
    padded = np.pad(arr, 1, mode='edge')
    counts = np.zeros((rows, cols, 10), dtype=np.int32)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            shifted = padded[1 + dr:rows + 1 + dr, 1 + dc:cols + 1 + dc]
            for color in range(10):
                counts[:, :, color] += (shifted == color).astype(np.int32)
    result = np.argmax(counts, axis=2).astype(np.int32)
    return from_np(result)


# --- Logical combinations of halves ---

def xor_halves_v(grid: Grid) -> Grid:
    """XOR top and bottom halves (keep pixel if only in one half)."""
    h = len(grid)
    half = h // 2
    top = to_np(grid[:half])
    bot = to_np(grid[half:half + half])
    if top.shape != bot.shape:
        return grid
    result = np.where((top != 0) ^ (bot != 0), np.maximum(top, bot), 0)
    return from_np(result)


def or_halves_v(grid: Grid) -> Grid:
    """OR top and bottom halves (overlay non-zero pixels)."""
    h = len(grid)
    half = h // 2
    top = to_np(grid[:half])
    bot = to_np(grid[half:half + half])
    if top.shape != bot.shape:
        return grid
    result = np.where(top != 0, top, bot)
    return from_np(result)


def xor_halves_h(grid: Grid) -> Grid:
    """XOR left and right halves."""
    arr = to_np(grid)
    w = arr.shape[1]
    half = w // 2
    left = arr[:, :half]
    right = arr[:, half:half + half]
    if left.shape != right.shape:
        return grid
    result = np.where((left != 0) ^ (right != 0), np.maximum(left, right), 0)
    return from_np(result)


def or_halves_h(grid: Grid) -> Grid:
    """OR left and right halves."""
    arr = to_np(grid)
    w = arr.shape[1]
    half = w // 2
    left = arr[:, :half]
    right = arr[:, half:half + half]
    if left.shape != right.shape:
        return grid
    result = np.where(left != 0, left, right)
    return from_np(result)


# --- Object detection helpers ---

def count_colors(grid: Grid) -> int:
    """Count unique non-zero colors in the grid."""
    return len(set(c for row in grid for c in row if c != 0))


def find_bounding_box(grid: Grid) -> tuple[int, int, int, int]:
    """Return (r_min, c_min, r_max, c_max) of non-zero pixels."""
    arr = to_np(grid)
    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        return (0, 0, 0, 0)
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    return (int(r_min), int(c_min), int(r_max), int(c_max))


# --- Composable binary operations (for composing two grids) ---

def overlay(base: Grid, top: Grid) -> Grid:
    """Overlay top grid onto base (non-zero pixels from top win)."""
    base_arr = to_np(base)
    top_arr = to_np(top)
    # Pad to same size if needed
    max_r = max(base_arr.shape[0], top_arr.shape[0])
    max_c = max(base_arr.shape[1], top_arr.shape[1])
    result = np.zeros((max_r, max_c), dtype=np.int32)
    result[:base_arr.shape[0], :base_arr.shape[1]] = base_arr
    tr, tc = top_arr.shape
    mask = top_arr != 0
    result[:tr, :tc][mask] = top_arr[mask]
    return from_np(result)


# =============================================================================
# Build the ARC primitive registry
# =============================================================================

def _build_arc_primitives() -> list[Primitive]:
    """Build the complete list of ARC primitives."""
    prims = []

    # Arity-1 primitives (Grid -> Grid): the core set
    unary_ops = [
        ("identity",        identity),
        ("rot90cw",         rotate_90_cw),
        ("rot90ccw",        rotate_90_ccw),
        ("rot180",          rotate_180),
        ("mirror_h",        mirror_horizontal),
        ("mirror_v",        mirror_vertical),
        ("transpose",       transpose),
        ("invert_colors",   invert_colors),
        ("crop_nonzero",    crop_to_nonzero),
        ("top_half",        get_top_half),
        ("bottom_half",     get_bottom_half),
        ("left_half",       get_left_half),
        ("right_half",      get_right_half),
        ("tile_2x2",        tile_2x2),
        ("tile_3x3",        tile_3x3),
        ("scale_2x",        scale_2x),
        ("scale_3x",        scale_3x),
        ("gravity_down",    gravity_down),
        ("gravity_up",      gravity_up),
        ("gravity_left",    gravity_left),
        ("gravity_right",   gravity_right),
        ("outline",         outline),
        ("fill_enclosed",   fill_enclosed),
        ("denoise_3x3",     denoise_3x3),
        ("xor_halves_v",    xor_halves_v),
        ("or_halves_v",     or_halves_v),
        ("xor_halves_h",    xor_halves_h),
        ("or_halves_h",     or_halves_h),
        ("replace_bg_mc",   replace_bg_with_most_common),
    ]

    for name, fn in unary_ops:
        prims.append(Primitive(name=name, arity=1, fn=fn, domain="arc"))

    # Color-specific keep/remove primitives for colors 1-9
    for color in range(1, 10):
        prims.append(Primitive(
            name=f"keep_c{color}",
            arity=1,
            fn=_make_keep_color(color),
            domain="arc",
        ))

    # Color replacement pairs (most common ARC transforms)
    # Only include a manageable subset: replace each color with bg and vice versa
    for from_c in range(1, 10):
        prims.append(Primitive(
            name=f"recolor_{from_c}_to_0",
            arity=1,
            fn=_make_replace_color(from_c, 0),
            domain="arc",
        ))

    # Arity-2 primitives (compose two transforms): overlay is the main one
    prims.append(Primitive(name="overlay", arity=2, fn=overlay, domain="arc"))

    return prims


ARC_PRIMITIVES = _build_arc_primitives()
_PRIM_MAP = {p.name: p for p in ARC_PRIMITIVES}


# =============================================================================
# Environment: execute grid transformation programs
# =============================================================================

class ARCEnv(Environment):
    """
    ARC-AGI environment.

    Programs are trees of grid transformations.
    Execute means: apply the transformation pipeline to an input grid.
    """

    def __init__(self):
        self._current_task: Optional[Task] = None

    def load_task(self, task: Task) -> Observation:
        self._current_task = task
        return Observation(
            data=[inp for inp, _ in task.train_examples],
            metadata={"task_id": task.task_id},
        )

    def execute(self, program: Program, input_data: Any) -> Any:
        """Execute the program tree on an input grid."""
        return self._eval_tree(program, input_data)

    def reset(self):
        self._current_task = None

    def _eval_tree(self, node: Program, grid: Grid) -> Grid:
        """Recursively evaluate a program tree on a grid."""
        prim = _PRIM_MAP.get(node.root)
        if prim is None:
            # Unknown primitive (possibly a learned library entry)
            # Return grid unchanged to avoid crashes
            return grid

        try:
            if prim.arity == 0:
                # Nullary: return the input grid (identity-like)
                return grid
            elif prim.arity == 1:
                # Unary: apply to the result of the single child
                if node.children:
                    child_grid = self._eval_tree(node.children[0], grid)
                else:
                    child_grid = grid
                result = prim.fn(child_grid)
                if not isinstance(result, list) or not result:
                    return grid
                return result
            elif prim.arity == 2:
                # Binary: apply to results of both children
                left = self._eval_tree(node.children[0], grid) if len(node.children) > 0 else grid
                right = self._eval_tree(node.children[1], grid) if len(node.children) > 1 else grid
                result = prim.fn(left, right)
                if not isinstance(result, list) or not result:
                    return grid
                return result
        except Exception:
            return grid

        return grid


# =============================================================================
# Grammar: composition rules for ARC grid transforms
# =============================================================================

class ARCGrammar(Grammar):
    """
    Grammar for composing ARC grid transformation programs.

    Programs are trees where:
    - Leaves are unary primitives applied directly to the input grid
    - Internal nodes compose the outputs of their children
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def base_primitives(self) -> list[Primitive]:
        return list(ARC_PRIMITIVES)

    def compose(self, outer: Primitive, inner_programs: list[Program]) -> Program:
        return Program(root=outer.name, children=inner_programs)

    def mutate(self, program: Program, primitives: list[Primitive]) -> Program:
        """Randomly modify one node in the tree."""
        prog = copy.deepcopy(program)
        nodes = self._collect_nodes(prog)
        if not nodes:
            return prog

        target = self._rng.choice(nodes)
        prim = _PRIM_MAP.get(target.root)
        current_arity = prim.arity if prim else 1

        # Pick a replacement with the same arity
        same_arity = [p for p in primitives if p.arity == current_arity]
        if same_arity:
            new_prim = self._rng.choice(same_arity)
            target.root = new_prim.name

        return prog

    def crossover(self, a: Program, b: Program) -> Program:
        """Replace a random subtree in a with a random subtree from b."""
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)

        a_nodes = self._collect_nodes(a_copy)
        b_nodes = self._collect_nodes(b_copy)

        if not a_nodes or not b_nodes:
            return a_copy

        target = self._rng.choice(a_nodes)
        donor = self._rng.choice(b_nodes)

        target.root = donor.root
        target.children = donor.children
        target.params = donor.params

        return a_copy

    def _collect_nodes(self, program: Program) -> list[Program]:
        result = [program]
        for child in program.children:
            result.extend(self._collect_nodes(child))
        return result


# =============================================================================
# Drive Signal: pixel accuracy + complexity
# =============================================================================

class ARCDrive(DriveSignal):
    """
    ARC drive signal: pixel edit distance + program complexity.

    prediction_error = fraction of cells that differ between predicted and expected.
    This is pixel_accuracy inverted: 0.0 = perfect match, 1.0 = nothing matches.
    """

    def prediction_error(self, predicted: Any, expected: Any) -> float:
        """Pixel edit distance: fraction of cells that differ."""
        if predicted is None or expected is None:
            return 1.0

        try:
            pred = to_np(predicted)
            exp = to_np(expected)
        except (ValueError, TypeError):
            return 1.0

        # If shapes differ, that's a significant error
        if pred.shape != exp.shape:
            # Partial credit: penalize by shape mismatch
            shape_penalty = 0.3
            # Try to compare overlapping region
            min_r = min(pred.shape[0], exp.shape[0])
            min_c = min(pred.shape[1], exp.shape[1])
            if min_r == 0 or min_c == 0:
                return 1.0
            overlap_pred = pred[:min_r, :min_c]
            overlap_exp = exp[:min_r, :min_c]
            pixel_err = float(np.sum(overlap_pred != overlap_exp)) / (exp.shape[0] * exp.shape[1])
            return min(1.0, pixel_err + shape_penalty)

        total_cells = exp.shape[0] * exp.shape[1]
        if total_cells == 0:
            return 0.0
        return float(np.sum(pred != exp)) / total_cells

    def complexity_cost(self, program: Program) -> float:
        """Program size as complexity measure."""
        return float(program.size)


# =============================================================================
# Task loader: ARC-AGI JSON format
# =============================================================================

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

def make_sample_tasks() -> list[Task]:
    """
    Create a small set of hand-crafted ARC-like tasks for testing.
    These test basic geometric and color transforms.
    """
    tasks = []

    # Task 1: Rotate 90 CW (easy)
    grid1_in = [[1, 0], [0, 2]]
    grid1_out = rotate_90_cw(grid1_in)
    tasks.append(Task(
        task_id="sample_rot90",
        train_examples=[
            (grid1_in, grid1_out),
            ([[3, 0, 0], [0, 4, 0], [0, 0, 5]], rotate_90_cw([[3, 0, 0], [0, 4, 0], [0, 0, 5]])),
        ],
        test_inputs=[[[1, 2], [3, 4]]],
        test_outputs=[rotate_90_cw([[1, 2], [3, 4]])],
        difficulty=1.0,
    ))

    # Task 2: Mirror horizontal (easy)
    grid2_in = [[1, 2, 3], [4, 5, 6]]
    grid2_out = mirror_horizontal(grid2_in)
    tasks.append(Task(
        task_id="sample_mirror_h",
        train_examples=[
            (grid2_in, grid2_out),
            ([[7, 0, 8]], mirror_horizontal([[7, 0, 8]])),
        ],
        test_inputs=[[[1, 0], [0, 2]]],
        test_outputs=[mirror_horizontal([[1, 0], [0, 2]])],
        difficulty=1.0,
    ))

    # Task 3: Crop to non-zero (medium)
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
    grid4_out = mirror_vertical(grid4_in)
    tasks.append(Task(
        task_id="sample_mirror_v",
        train_examples=[
            (grid4_in, grid4_out),
            ([[9, 8], [7, 6]], mirror_vertical([[9, 8], [7, 6]])),
        ],
        test_inputs=[[[1, 0, 2], [0, 3, 0]]],
        test_outputs=[mirror_vertical([[1, 0, 2], [0, 3, 0]])],
        difficulty=1.0,
    ))

    # Task 5: Gravity down (medium)
    grid5_in = [[1, 0, 2], [0, 3, 0], [0, 0, 0]]
    grid5_out = gravity_down(grid5_in)
    tasks.append(Task(
        task_id="sample_gravity_down",
        train_examples=[
            (grid5_in, grid5_out),
            ([[0, 4, 0], [5, 0, 6], [0, 0, 0]], gravity_down([[0, 4, 0], [5, 0, 6], [0, 0, 0]])),
        ],
        test_inputs=[[[7, 0, 0], [0, 0, 8], [0, 9, 0]]],
        test_outputs=[gravity_down([[7, 0, 0], [0, 0, 8], [0, 9, 0]])],
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
        return mirror_horizontal(rotate_90_cw(g))
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

    # Task 8: Invert + crop (harder — needs depth-2)
    def invert_crop(g):
        return crop_to_nonzero(invert_colors(g))
    tasks.append(Task(
        task_id="sample_invert_crop",
        train_examples=[
            ([[0, 0, 0], [0, 5, 0], [0, 0, 0]], invert_crop([[0, 0, 0], [0, 5, 0], [0, 0, 0]])),
            ([[0, 3, 0], [0, 0, 0]], invert_crop([[0, 3, 0], [0, 0, 0]])),
        ],
        test_inputs=[[[0, 0], [0, 2], [0, 0]]],
        test_outputs=[invert_crop([[0, 0], [0, 2], [0, 0]])],
        difficulty=4.5,
    ))

    return tasks


# =============================================================================
# Quick demo: run the loop on sample ARC tasks
# =============================================================================

if __name__ == "__main__":
    import logging
    import time
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core import Learner, InMemoryStore, SearchConfig, CurriculumConfig
    from core import WakeResult, extract_metrics, print_compounding_table

    print("=" * 70)
    print("  ARC-AGI SAMPLE TASKS DEMO")
    print("  The core algorithm searches for grid transformation programs")
    print("=" * 70)

    tasks = make_sample_tasks()

    # Describe the tasks for the user
    task_descriptions = {
        "rotate_90": "Rotate grid 90° clockwise",
        "rotate_180": "Rotate grid 180°",
        "mirror_h": "Mirror grid horizontally",
        "mirror_v": "Mirror grid vertically",
        "double": "Scale grid 2x (each cell becomes 2×2)",
        "crop_nonzero": "Crop to bounding box of non-zero cells",
        "gravity_down": "Drop non-zero cells to bottom (gravity)",
        "fill_enclosed": "Fill enclosed zero regions with color",
    }

    print(f"\n  Tasks ({len(tasks)} sample grid puzzles, no dataset needed):")
    for t in tasks:
        desc = task_descriptions.get(t.task_id, "")
        grid_in = t.train_examples[0][0] if t.train_examples else []
        h, w = len(grid_in), len(grid_in[0]) if grid_in else 0
        print(f"    {t.task_id:<16s} {h}x{w}  {desc:<45s} (difficulty={t.difficulty})")
    print()

    # Wire up the 4 plugins
    env = ARCEnv()
    grammar = ARCGrammar(seed=42)
    drive = ARCDrive()
    memory = InMemoryStore()

    learner = Learner(
        environment=env,
        grammar=grammar,
        drive=drive,
        memory=memory,
        search_config=SearchConfig(
            beam_width=150,
            max_generations=60,
            mutations_per_candidate=2,
            crossover_fraction=0.3,
            energy_beta=0.002,
            solve_threshold=0.001,  # pixel error < 0.1% counts as solved
            seed=42,
        ),
    )

    print(f"  Primitives:  {len(grammar.base_primitives())} "
          f"(rotate, flip, crop, fill, recolor, scale, ...)")
    print(f"  Beam width:  150")
    print(f"  Generations: 60 per task")
    print(f"  Workers:     {Learner.performance_core_count()}")
    print()

    # Live progress callback
    t0 = time.time()

    def on_task_done(round_num, task_index, total_tasks, wr: WakeResult):
        elapsed = time.time() - t0
        icon = "✓" if wr.solved else "✗"
        energy_str = f"{wr.best.energy:.4f}" if wr.best else "N/A"
        prog_str = repr(wr.best.program) if wr.best else ""
        desc = task_descriptions.get(wr.task_id, "")
        print(f"  {icon} R{round_num} [{task_index:>2}/{total_tasks}] "
              f"{wr.task_id:<16s} E={energy_str:<8s} "
              f"gens={wr.generations_used:<4d} "
              f"{wr.wall_time:.1f}s  [{elapsed:.0f}s elapsed]")
        if wr.solved:
            print(f"       program: {prog_str}")

    # Run the curriculum
    results = learner.run_curriculum(
        tasks,
        CurriculumConfig(
            sort_by_difficulty=True,
            wake_sleep_rounds=3,
        ),
        on_task_done=on_task_done,
    )

    # Print the compounding curve
    print("\n" + "=" * 70)
    print("  COMPOUNDING CURVE — does solve rate increase across rounds?")
    print("=" * 70)
    metrics = extract_metrics(results)
    print_compounding_table(metrics)

    total_time = time.time() - t0
    total_solved = sum(1 for rr in results for wr in rr.wake_results if wr.solved)
    total_tasks = sum(len(rr.wake_results) for rr in results)
    print(f"\n  Total time: {total_time:.1f}s  "
          f"Solved: {total_solved}/{total_tasks}")

    # Save library
    os.makedirs("library", exist_ok=True)
    memory.save("library/arc_library.json")
    print(f"\n  Library: {len(memory.get_library())} learned abstractions")
    for entry in memory.get_library():
        print(f"    {entry.name}: {entry.program} "
              f"(useful={entry.usefulness:.1f}, reused={entry.reuse_count}x)")
    print()
    print("  For the full ARC benchmark, run:")
    print("    python -m experiments.phase1_arc")
    print()
