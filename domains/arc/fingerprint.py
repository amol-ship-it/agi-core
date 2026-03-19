"""
Fast structural fingerprinting of ARC tasks.

Analyzes training examples to compute features that guide stratum selection.
Self-contained (no imports from objects.py) for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.types import Task


@dataclass
class TaskFingerprint:
    """Structural features of an ARC task, computed from training examples."""
    dim_change: str = "same"          # "same", "shrink", "grow", "variable"
    has_separators: bool = False      # grid divided by uniform rows/cols
    n_sections: int = 1              # how many sections separators create
    symmetry: set = field(default_factory=set)  # {"h","v","diag","rot"}
    symmetry_broken: bool = False    # input has symmetry, output fixes it
    n_objects: int = 0               # connected component count (non-zero, 4-conn)
    object_size_var: float = 0.0     # variance of object sizes
    has_periodic: bool = False       # repeating tile detected
    has_holes: bool = False          # enclosed zero regions exist
    n_colors_in: int = 0            # distinct colors in inputs
    n_colors_out: int = 0           # distinct colors in outputs
    colors_added: int = 0           # colors in output but not input
    colors_removed: int = 0         # colors in input but not output
    pixel_diff_ratio: float = 0.0   # fraction of changed pixels (same-dim)
    output_is_subgrid: bool = False # output appears inside input
    is_recoloring: bool = False     # same non-zero structure, different colors


def fingerprint_task(task: Task) -> TaskFingerprint:
    """Compute a TaskFingerprint from a task's training examples.

    Fast: < 1ms per typical task. Features are majority-voted across examples.
    """
    fp = TaskFingerprint()
    examples = task.train_examples
    if not examples:
        return fp

    # --- dim_change ---
    fp.dim_change = _compute_dim_change(examples)

    # --- colors ---
    all_colors_in: set[int] = set()
    all_colors_out: set[int] = set()
    for inp, out in examples:
        all_colors_in |= _grid_colors(inp)
        all_colors_out |= _grid_colors(out)
    fp.n_colors_in = len(all_colors_in)
    fp.n_colors_out = len(all_colors_out)
    fp.colors_added = len(all_colors_out - all_colors_in)
    fp.colors_removed = len(all_colors_in - all_colors_out)

    # --- pixel_diff_ratio (same-dim only) ---
    fp.pixel_diff_ratio = _compute_pixel_diff(examples)

    # --- objects (from inputs) ---
    obj_counts = []
    all_sizes: list[int] = []
    for inp, _ in examples:
        comps = _find_components(inp)
        obj_counts.append(len(comps))
        all_sizes.extend(c_size for _, c_size in comps)
    fp.n_objects = round(sum(obj_counts) / len(obj_counts)) if obj_counts else 0
    if len(all_sizes) >= 2:
        mean_s = sum(all_sizes) / len(all_sizes)
        fp.object_size_var = sum((s - mean_s) ** 2 for s in all_sizes) / len(all_sizes)
    elif len(all_sizes) == 1:
        fp.object_size_var = 0.0

    # --- has_holes (majority vote) ---
    hole_votes = [_grid_has_holes(inp) for inp, _ in examples]
    fp.has_holes = sum(hole_votes) > len(hole_votes) / 2

    # --- separators ---
    sep_votes = []
    section_counts = []
    for inp, _ in examples:
        has_sep, n_sec = _detect_separators(inp)
        sep_votes.append(has_sep)
        section_counts.append(n_sec)
    fp.has_separators = sum(sep_votes) > len(sep_votes) / 2
    fp.n_sections = round(sum(section_counts) / len(section_counts)) if section_counts else 1

    # --- symmetry ---
    fp.symmetry = _detect_symmetry_set(examples)

    # --- symmetry_broken ---
    fp.symmetry_broken = _detect_symmetry_broken(examples)

    # --- is_recoloring ---
    recol_votes = [_is_recoloring_pair(inp, out) for inp, out in examples]
    fp.is_recoloring = all(recol_votes) if recol_votes else False

    # --- has_periodic ---
    periodic_votes = [_detect_periodic(inp) for inp, _ in examples]
    fp.has_periodic = sum(periodic_votes) > len(periodic_votes) / 2

    # --- output_is_subgrid ---
    subgrid_votes = [_output_is_subgrid(inp, out) for inp, out in examples]
    fp.output_is_subgrid = sum(subgrid_votes) > len(subgrid_votes) / 2

    return fp


# ---- Internal helpers (self-contained, no external imports) ----

def _grid_colors(grid: list[list[int]]) -> set[int]:
    """Return set of all colors in grid."""
    return {c for row in grid for c in row}


def _compute_dim_change(examples: list[tuple]) -> str:
    """Determine if outputs are same/shrink/grow/variable relative to inputs."""
    changes = set()
    for inp, out in examples:
        h_in, w_in = len(inp), len(inp[0]) if inp else 0
        h_out, w_out = len(out), len(out[0]) if out else 0
        in_sz = h_in * w_in
        out_sz = h_out * w_out
        if h_in == h_out and w_in == w_out:
            changes.add("same")
        elif out_sz < in_sz:
            changes.add("shrink")
        else:
            changes.add("grow")
    if len(changes) == 1:
        return changes.pop()
    return "variable"


def _compute_pixel_diff(examples: list[tuple]) -> float:
    """Average fraction of pixels that differ (same-dim only). -1 if dims differ."""
    ratios = []
    for inp, out in examples:
        h_in, w_in = len(inp), len(inp[0]) if inp else 0
        h_out, w_out = len(out), len(out[0]) if out else 0
        if h_in != h_out or w_in != w_out:
            return -1.0
        total = h_in * w_in
        if total == 0:
            continue
        diff = sum(1 for r in range(h_in) for c in range(w_in) if inp[r][c] != out[r][c])
        ratios.append(diff / total)
    return sum(ratios) / len(ratios) if ratios else 0.0


def _find_components(grid: list[list[int]]) -> list[tuple[set, int]]:
    """Find connected components (4-connectivity, non-zero).

    Returns list of (pixel_set, size) tuples.
    """
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                pixels: set[tuple[int, int]] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr][cc] or grid[cr][cc] == 0:
                        continue
                    # Only same-color connectivity
                    if grid[cr][cc] != grid[r][c]:
                        continue
                    visited[cr][cc] = True
                    pixels.add((cr, cc))
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
                components.append((pixels, len(pixels)))
    return components


def _grid_has_holes(grid: list[list[int]]) -> bool:
    """Check if grid has enclosed zero regions (holes).

    Flood fill from border zeros; any unfilled zero is a hole.
    """
    if not grid or not grid[0]:
        return False
    h, w = len(grid), len(grid[0])
    if h <= 2 or w <= 2:
        return False

    visited = [[False] * w for _ in range(h)]
    # Seed from border zeros
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                if not visited[r][c]:
                    visited[r][c] = True
                    stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                stack.append((nr, nc))

    # Any unvisited zero is a hole
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not visited[r][c]:
                return True
    return False


def _detect_separators(grid: list[list[int]]) -> tuple[bool, int]:
    """Detect uniform rows/cols that act as separators.

    Returns (has_separators, n_sections).
    """
    if not grid or not grid[0]:
        return False, 1
    h, w = len(grid), len(grid[0])

    # Check for uniform rows (all same color, different from adjacent rows)
    sep_rows: list[int] = []
    for r in range(h):
        row = grid[r]
        if len(set(row)) == 1:
            # Check it's a separator (not just a normal content row)
            color = row[0]
            is_sep = False
            if r > 0 and any(grid[r-1][c] != color for c in range(w)):
                is_sep = True
            if r < h - 1 and any(grid[r+1][c] != color for c in range(w)):
                is_sep = True
            if is_sep:
                sep_rows.append(r)

    # Check for uniform cols
    sep_cols: list[int] = []
    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]
        if len(set(col_vals)) == 1:
            color = col_vals[0]
            is_sep = False
            if c > 0 and any(grid[r][c-1] != color for r in range(h)):
                is_sep = True
            if c < w - 1 and any(grid[r][c+1] != color for r in range(h)):
                is_sep = True
            if is_sep:
                sep_cols.append(c)

    has_sep = len(sep_rows) > 0 or len(sep_cols) > 0
    # Sections = (row_sections) * (col_sections)
    row_sections = len(sep_rows) + 1
    col_sections = len(sep_cols) + 1
    n_sections = max(row_sections, col_sections)
    if sep_rows and sep_cols:
        n_sections = row_sections * col_sections

    return has_sep, n_sections


def _detect_symmetry_set(examples: list[tuple]) -> set:
    """Detect symmetry flags in input grids."""
    flags: set[str] = set()
    for inp, _ in examples:
        if not inp or not inp[0]:
            continue
        h, w = len(inp), len(inp[0])

        # Horizontal (left-right mirror)
        if all(inp[r][c] == inp[r][w - 1 - c] for r in range(h) for c in range(w // 2)):
            flags.add("h")

        # Vertical (top-bottom mirror)
        if all(inp[r][c] == inp[h - 1 - r][c] for r in range(h // 2) for c in range(w)):
            flags.add("v")

        # Diagonal (transpose symmetry, only for square)
        if h == w and all(inp[r][c] == inp[c][r] for r in range(h) for c in range(w)):
            flags.add("diag")

        # 180-degree rotation
        if all(inp[r][c] == inp[h - 1 - r][w - 1 - c] for r in range(h) for c in range(w)):
            flags.add("rot")

    return flags


def _detect_symmetry_broken(examples: list[tuple]) -> bool:
    """Check if input has near-symmetry that output fixes."""
    for inp, out in examples:
        if not inp or not inp[0]:
            continue
        h, w = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0]) if out else 0
        if h != oh or w != ow:
            continue

        # Check if output has better horizontal symmetry than input
        in_h_diff = sum(1 for r in range(h) for c in range(w // 2) if inp[r][c] != inp[r][w-1-c])
        out_h_diff = sum(1 for r in range(h) for c in range(w // 2) if out[r][c] != out[r][w-1-c])
        if in_h_diff > 0 and out_h_diff == 0:
            return True

        # Check vertical
        in_v_diff = sum(1 for r in range(h // 2) for c in range(w) if inp[r][c] != inp[h-1-r][c])
        out_v_diff = sum(1 for r in range(h // 2) for c in range(w) if out[r][c] != out[h-1-r][c])
        if in_v_diff > 0 and out_v_diff == 0:
            return True

    return False


def _is_recoloring_pair(inp: list[list[int]], out: list[list[int]]) -> bool:
    """Check if output has same non-zero structure as input but different colors."""
    if not inp or not out or not inp[0] or not out[0]:
        return False
    h_in, w_in = len(inp), len(inp[0])
    h_out, w_out = len(out), len(out[0])
    if h_in != h_out or w_in != w_out:
        return False

    # Check that non-zero positions are identical
    has_diff = False
    for r in range(h_in):
        for c in range(w_in):
            in_nz = inp[r][c] != 0
            out_nz = out[r][c] != 0
            if in_nz != out_nz:
                return False
            if in_nz and inp[r][c] != out[r][c]:
                has_diff = True
    return has_diff


def _detect_periodic(grid: list[list[int]]) -> bool:
    """Detect if grid has a repeating tile pattern."""
    if not grid or not grid[0]:
        return False
    h, w = len(grid), len(grid[0])
    # Try small tile sizes
    for th in range(1, h // 2 + 1):
        if h % th != 0:
            continue
        for tw in range(1, w // 2 + 1):
            if w % tw != 0:
                continue
            # Check if tile (0:th, 0:tw) repeats
            is_periodic = True
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != grid[r % th][c % tw]:
                        is_periodic = False
                        break
                if not is_periodic:
                    break
            if is_periodic:
                return True
    return False


def _output_is_subgrid(inp: list[list[int]], out: list[list[int]]) -> bool:
    """Check if the output grid appears as a subgrid inside the input."""
    if not inp or not out or not inp[0] or not out[0]:
        return False
    h_in, w_in = len(inp), len(inp[0])
    h_out, w_out = len(out), len(out[0])
    if h_out > h_in or w_out > w_in:
        return False
    if h_out == h_in and w_out == w_in:
        return False  # same size, not a subgrid extraction

    for r in range(h_in - h_out + 1):
        for c in range(w_in - w_out + 1):
            match = True
            for dr in range(h_out):
                for dc in range(w_out):
                    if inp[r + dr][c + dc] != out[dr][dc]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return True
    return False
