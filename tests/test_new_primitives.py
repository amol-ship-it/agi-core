"""Tests for new primitives added in the vocabulary expansion."""

import unittest

from domains.arc.transformation_primitives import (
    extract_largest_object, extract_smallest_object, hollow_objects,
    complete_horizontal_symmetry, complete_vertical_symmetry,
    fill_diagonal_symmetry, remove_border, denoise_majority,
    keep_unique_rows, keep_unique_cols, extract_repeating_pattern,
    xor_grids, subtract_grids,
)
from domains.arc.perception_primitives import (
    has_horizontal_symmetry, has_vertical_symmetry,
    nonzero_pixel_count, unique_color_count, most_common_nonzero,
    largest_object_size, grid_density,
)
from domains.arc.analysis import analyze_task, TaskSignature


# Sample grids for testing
GRID_3x3 = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]

GRID_WITH_TWO_OBJECTS = [
    [1, 1, 0, 0, 2],
    [1, 0, 0, 0, 2],
    [0, 0, 0, 0, 2],
]

SYMMETRIC_H = [
    [1, 0, 1],
    [2, 0, 2],
    [1, 0, 1],
]

BORDERED = [
    [5, 5, 5, 5],
    [5, 1, 2, 5],
    [5, 3, 4, 5],
    [5, 5, 5, 5],
]

TILED = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 2, 1, 2],
    [3, 4, 3, 4],
]


class TestExtractObject(unittest.TestCase):
    def test_largest_object(self):
        result = extract_largest_object(GRID_WITH_TWO_OBJECTS)
        # Object 1 has 3 pixels, object 2 has 3 pixels — largest by first found
        assert isinstance(result, list)
        assert len(result) > 0

    def test_smallest_object(self):
        grid = [
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = extract_smallest_object(grid)
        # Object 2 is smaller (1 pixel)
        assert len(result) == 1 or (len(result[0]) == 1)

    def test_empty_grid(self):
        result = extract_largest_object([])
        assert result == []

    def test_single_pixel(self):
        grid = [[0, 0], [0, 5]]
        result = extract_largest_object(grid)
        assert result == [[5]]


class TestHollowObjects(unittest.TestCase):
    def test_hollow_filled_square(self):
        # 3x3 filled square — center pixel is interior
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        result = hollow_objects(grid)
        # Center pixel [2][2] has all non-bg neighbors → interior → removed
        assert result[2][2] == 0
        # Edge pixels remain
        assert result[1][1] == 1
        assert result[1][2] == 1

    def test_single_pixel_unchanged(self):
        grid = [[0, 0], [0, 5]]
        result = hollow_objects(grid)
        assert result[1][1] == 5  # no interior pixels


class TestSymmetryCompletion(unittest.TestCase):
    def test_horizontal_completion(self):
        # Partial h-symmetry: left side has content, right side is bg
        grid = [
            [1, 2, 0],
            [3, 0, 0],
        ]
        result = complete_horizontal_symmetry(grid)
        assert result[0][2] == 1  # mirror of [0][0]
        assert result[1][2] == 3  # mirror of [1][0]

    def test_vertical_completion(self):
        grid = [
            [1, 2],
            [0, 0],
        ]
        result = complete_vertical_symmetry(grid)
        assert result[1][0] == 1  # mirror of [0][0]
        assert result[1][1] == 2  # mirror of [0][1]

    def test_already_symmetric(self):
        result = complete_horizontal_symmetry(SYMMETRIC_H)
        assert result == SYMMETRIC_H


class TestRemoveBorder(unittest.TestCase):
    def test_remove_border(self):
        result = remove_border(BORDERED)
        assert result == [[1, 2], [3, 4]]

    def test_too_small(self):
        result = remove_border([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]


class TestDenoiseMajority(unittest.TestCase):
    def test_denoise(self):
        grid = [
            [1, 1, 1],
            [1, 5, 1],  # center pixel is noise
            [1, 1, 1],
        ]
        result = denoise_majority(grid)
        assert result[1][1] == 1  # majority of neighbors is 1


class TestUniqueRowsCols(unittest.TestCase):
    def test_unique_rows(self):
        grid = [[1, 2], [3, 4], [1, 2], [3, 4]]
        result = keep_unique_rows(grid)
        assert result == [[1, 2], [3, 4]]

    def test_unique_cols(self):
        grid = [[1, 2, 1], [3, 4, 3]]
        result = keep_unique_cols(grid)
        assert result == [[1, 2], [3, 4]]


class TestExtractRepeatingPattern(unittest.TestCase):
    def test_tiled_pattern(self):
        result = extract_repeating_pattern(TILED)
        assert result == [[1, 2], [3, 4]]

    def test_no_pattern(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = extract_repeating_pattern(grid)
        assert result == grid


class TestBinaryOps(unittest.TestCase):
    def test_xor(self):
        g1 = [[1, 0], [0, 2]]
        g2 = [[0, 3], [4, 0]]
        result = xor_grids(g1, g2)
        assert result == [[1, 3], [4, 2]]

    def test_xor_overlap(self):
        g1 = [[1, 0], [0, 2]]
        g2 = [[1, 3], [0, 0]]
        result = xor_grids(g1, g2)
        assert result[0][0] == 0  # both non-zero → 0
        assert result[0][1] == 3  # only g2

    def test_subtract(self):
        g1 = [[1, 2], [3, 4]]
        g2 = [[0, 1], [1, 0]]
        result = subtract_grids(g1, g2)
        assert result == [[1, 0], [0, 4]]


# =============================================================================
# Perception primitive tests
# =============================================================================

class TestNewPerceptions(unittest.TestCase):
    def test_horizontal_symmetry(self):
        assert has_horizontal_symmetry(SYMMETRIC_H) == 1
        assert has_horizontal_symmetry([[1, 2, 3]]) == 0

    def test_vertical_symmetry(self):
        grid = [[1, 2], [1, 2]]
        assert has_vertical_symmetry(grid) == 1
        assert has_vertical_symmetry([[1, 2], [3, 4]]) == 0

    def test_nonzero_count(self):
        assert nonzero_pixel_count(GRID_3x3) == 5

    def test_unique_color_count(self):
        assert unique_color_count(GRID_3x3) == 1  # only color 1
        assert unique_color_count(GRID_WITH_TWO_OBJECTS) == 2  # colors 1, 2

    def test_most_common_nonzero(self):
        assert most_common_nonzero(GRID_WITH_TWO_OBJECTS) == 1  # or 2, both appear 3x
        assert most_common_nonzero(GRID_3x3) == 1

    def test_largest_object_size(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        assert largest_object_size(grid) == 8  # the ring of 1s

    def test_grid_density(self):
        d = grid_density(GRID_3x3)
        assert d > 0 and d < 100

    def test_empty(self):
        assert nonzero_pixel_count([]) == 0
        assert has_horizontal_symmetry([]) == 0


# =============================================================================
# Analysis module tests
# =============================================================================

class TestAnalysis(unittest.TestCase):
    def test_same_dimensions(self):
        examples = [
            ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
        ]
        sig = analyze_task(examples)
        assert sig.dim_relation == "same"

    def test_scaled_dimensions(self):
        inp = [[1, 2], [3, 4]]
        out = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        sig = analyze_task([(inp, out)])
        assert sig.dim_relation == "scaled"
        assert sig.scale_factor == (2, 2)
        assert sig.prioritize_scale_tile is True

    def test_different_dimensions(self):
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[1, 2], [3, 4]]  # 3x3 → 2x2: not integer scale
        sig = analyze_task([(inp, out)])
        assert sig.dim_relation == "different"
        assert sig.output_smaller is True
        assert sig.skip_gravity is True

    def test_color_preserved(self):
        examples = [
            ([[1, 2], [3, 0]], [[2, 1], [0, 3]]),
        ]
        sig = analyze_task(examples)
        assert sig.color_relation == "preserved"

    def test_color_subset(self):
        examples = [
            ([[1, 2, 3], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]),
        ]
        sig = analyze_task(examples)
        assert sig.color_relation == "subset"

    def test_symmetry_detection(self):
        symmetric_grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        examples = [
            ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], symmetric_grid),
        ]
        sig = analyze_task(examples)
        assert sig.has_symmetry in ("h", "v", "both")

    def test_empty_examples(self):
        sig = analyze_task([])
        assert sig.dim_relation == "same"

    def test_object_count_same(self):
        examples = [
            ([[1, 0, 2], [0, 0, 0]], [[2, 0, 1], [0, 0, 0]]),
        ]
        sig = analyze_task(examples)
        assert sig.object_count_change == "same"

    def test_recommended_phases(self):
        inp = [[1, 2], [3, 4]]
        out = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        sig = analyze_task([(inp, out)])
        assert "prioritize_scale_tile" in sig.recommended_phases


if __name__ == "__main__":
    unittest.main()
