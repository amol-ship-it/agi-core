"""
Tests for atomic primitives (transformation + perception + parameterized).

Verifies:
1. Each truly atomic transform produces valid output
2. Perception primitives return correct values
3. Parameterized factories produce working transforms
4. Grammar integration with vocabulary='atomic'
5. Environment execution of atomic programs
6. crop_to_content is compositional (trim_rows + trim_cols)
"""

import pytest
import unittest

from core import Program, Task, Primitive
from core.types import ScoredProgram

from domains.arc.transformation_primitives import (
    # Atomic transforms
    rotate_90_cw, rotate_90_ccw, rotate_180,
    mirror_horizontal, mirror_vertical, transpose,
    trim_rows, trim_cols,
    crop_half_top, crop_half_bottom,
    crop_half_left, crop_half_right,
    pad_border, binarize, invert_colors,
    dilate, erode, gravity_down, fill_enclosed, overlay,
    # Parameterized factories
    _scale_factory, _tile_factory, _downscale_factory,
    _swap_colors_factory, _keep_color_factory, _erase_color_factory,
    # Builders
    build_atomic_primitives, build_parameterized_primitives,
    ATOMIC_ESSENTIAL_PAIR_CONCEPTS,
)
from domains.arc.perception_primitives import (
    background_color, dominant_color, rarest_color, accent_color,
    n_colors, n_foreground_colors, grid_height, grid_width,
    grid_min_dim, n_objects,
    build_perception_primitives,
)
from domains.arc.grammar import ARCGrammar
from domains.arc.environment import ARCEnv


# =============================================================================
# Test grids
# =============================================================================

SMALL_GRID = [[1, 2], [3, 4]]
ALL_ZERO = [[0, 0], [0, 0]]
EMPTY_GRID = [[0]]
CENTER_PIXEL = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
TWO_OBJECTS = [
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
]
ENCLOSED = [
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
]


# =============================================================================
# 1. Geometric transforms
# =============================================================================

class TestGeometricPrimitives:
    def test_rotate_90_cw(self):
        assert rotate_90_cw(SMALL_GRID) == [[3, 1], [4, 2]]

    def test_rotate_90_ccw(self):
        assert rotate_90_ccw(SMALL_GRID) == [[2, 4], [1, 3]]

    def test_rotate_180(self):
        assert rotate_180(SMALL_GRID) == [[4, 3], [2, 1]]

    def test_mirror_horizontal(self):
        assert mirror_horizontal(SMALL_GRID) == [[2, 1], [4, 3]]

    def test_mirror_vertical(self):
        assert mirror_vertical(SMALL_GRID) == [[3, 4], [1, 2]]

    def test_transpose(self):
        assert transpose(SMALL_GRID) == [[1, 3], [2, 4]]

    def test_geometric_on_1x1(self):
        for fn in [rotate_90_cw, rotate_90_ccw, rotate_180,
                   mirror_horizontal, mirror_vertical, transpose]:
            result = fn(EMPTY_GRID)
            assert isinstance(result, list) and len(result) >= 1


# =============================================================================
# 2. Spatial transforms
# =============================================================================

class TestSpatialPrimitives:
    def test_trim_rows(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert trim_rows(grid) == [[0, 1, 0]]

    def test_trim_cols(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert trim_cols(grid) == [[0], [1], [0]]

    def test_trim_rows_no_zeros(self):
        grid = [[1, 2], [3, 4]]
        assert trim_rows(grid) == [[1, 2], [3, 4]]

    def test_trim_cols_no_zeros(self):
        grid = [[1, 2], [3, 4]]
        assert trim_cols(grid) == [[1, 2], [3, 4]]

    def test_trim_rows_all_zero(self):
        assert trim_rows(ALL_ZERO) == ALL_ZERO

    def test_trim_cols_all_zero(self):
        assert trim_cols(ALL_ZERO) == ALL_ZERO

    def test_crop_to_content_is_compositional(self):
        """crop_to_content = trim_cols(trim_rows(x)) = trim_rows(trim_cols(x))"""
        grid = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
        assert trim_cols(trim_rows(grid)) == [[1, 2], [3, 4]]
        assert trim_rows(trim_cols(grid)) == [[1, 2], [3, 4]]

    def test_crop_to_content_single_pixel(self):
        grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        assert trim_cols(trim_rows(grid)) == [[5]]

    def test_crop_half_top(self):
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert crop_half_top(grid) == [[1, 2], [3, 4]]

    def test_crop_half_bottom(self):
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert crop_half_bottom(grid) == [[5, 6], [7, 8]]

    def test_crop_half_left(self):
        grid = [[1, 2, 3, 4]]
        assert crop_half_left(grid) == [[1, 2]]

    def test_crop_half_right(self):
        grid = [[1, 2, 3, 4]]
        assert crop_half_right(grid) == [[3, 4]]

    def test_pad_border(self):
        assert pad_border([[1]]) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


# =============================================================================
# 3. Color transforms
# =============================================================================

class TestColorPrimitives:
    def test_binarize(self):
        assert binarize([[0, 3, 5], [7, 0, 2]]) == [[0, 1, 1], [1, 0, 1]]

    def test_invert_colors(self):
        result = invert_colors([[0, 5], [9, 3]])
        assert result == [[9, 4], [0, 6]]


# =============================================================================
# 4. Morphological transforms
# =============================================================================

class TestMorphologicalPrimitives:
    def test_dilate_center_pixel(self):
        result = dilate(CENTER_PIXEL)
        assert result[1][1] != 0
        assert result[0][1] != 0
        assert result[2][1] != 0
        assert result[0][0] == 0

    def test_erode_center_pixel(self):
        assert erode(CENTER_PIXEL)[1][1] == 0

    def test_erode_solid_block(self):
        solid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = erode(solid)
        assert result[1][1] == 1
        assert result[0][0] == 0

    def test_dilate_empty(self):
        assert dilate(ALL_ZERO) == ALL_ZERO

    def test_erode_empty(self):
        assert erode(ALL_ZERO) == ALL_ZERO

    def test_erode_then_dilate(self):
        """Morphological opening = erode then dilate."""
        solid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        opened = dilate(erode(solid))
        assert isinstance(opened, list)


# =============================================================================
# 5. Physics and fill
# =============================================================================

class TestPhysicsAndFill:
    def test_gravity_down(self):
        grid = [[1, 0], [0, 2], [0, 0]]
        result = gravity_down(grid)
        assert result[2][0] == 1
        assert result[2][1] == 2

    def test_fill_enclosed(self):
        result = fill_enclosed(ENCLOSED)
        assert result[1][1] != 0
        assert result[1][2] != 0

    def test_fill_enclosed_no_enclosed(self):
        grid = [[1, 0], [0, 1]]
        result = fill_enclosed(grid)
        assert result == grid


# =============================================================================
# 6. Overlay
# =============================================================================

class TestOverlay:
    def test_overlay(self):
        base = [[0, 1], [0, 0]]
        top = [[2, 0], [0, 3]]
        result = overlay(base, top)
        assert result[0][0] == 2
        assert result[0][1] == 1
        assert result[1][1] == 3


# =============================================================================
# 7. Parameterized factories
# =============================================================================

class TestParameterizedFactories:
    def test_scale_factory(self):
        assert _scale_factory(2)([[1]]) == [[1, 1], [1, 1]]

    def test_scale_factory_3x(self):
        result = _scale_factory(3)([[1]])
        assert len(result) == 3 and len(result[0]) == 3

    def test_tile_factory(self):
        assert _tile_factory(2)([[1]]) == [[1, 1], [1, 1]]

    def test_downscale_factory(self):
        assert _downscale_factory(2)([[1, 1], [1, 1]]) == [[1]]

    def test_scale_guards(self):
        assert _scale_factory(100)([[1]]) == [[1]]

    def test_swap_colors(self):
        result = _swap_colors_factory(1, 2)([[1, 2], [3, 0]])
        assert result == [[2, 1], [3, 0]]

    def test_keep_color(self):
        result = _keep_color_factory(1)([[1, 2], [3, 1]])
        assert result == [[1, 0], [0, 1]]

    def test_erase_color(self):
        result = _erase_color_factory(2)([[1, 2], [2, 3]])
        assert result == [[1, 0], [0, 3]]


# =============================================================================
# 8. Perception primitives
# =============================================================================

class TestPerceptionPrimitives:
    def test_background_color(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert background_color(grid) == 0

    def test_dominant_color(self):
        grid = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
        assert dominant_color(grid) == 1

    def test_rarest_color(self):
        grid = [[1, 1, 2], [1, 1, 0], [0, 0, 0]]
        assert rarest_color(grid) == 2

    def test_n_colors(self):
        assert n_colors([[0, 1], [2, 3]]) == 4

    def test_n_foreground_colors(self):
        assert n_foreground_colors([[0, 1], [2, 0]]) == 2

    def test_grid_height(self):
        assert grid_height([[1, 2], [3, 4], [5, 6]]) == 3

    def test_grid_width(self):
        assert grid_width([[1, 2, 3]]) == 3

    def test_grid_min_dim(self):
        assert grid_min_dim([[1, 2, 3], [4, 5, 6]]) == 2

    def test_n_objects(self):
        assert n_objects(TWO_OBJECTS) == 2

    def test_n_objects_empty(self):
        assert n_objects(ALL_ZERO) == 0


# =============================================================================
# 9. Builder tests
# =============================================================================

class TestBuilders:
    def test_build_atomic_count(self):
        prims = build_atomic_primitives()
        # 6 geometric + 7 spatial + 2 color + 2 morphological
        # + 1 physics + 1 fill + 1 label_components + 2 binary = 22
        assert len(prims) == 22

    def test_all_have_names(self):
        for p in build_atomic_primitives():
            assert p.name
            assert p.domain == "arc"

    def test_all_unary_produce_valid_output(self):
        for p in build_atomic_primitives():
            if p.arity == 1:
                result = p.fn(SMALL_GRID)
                assert isinstance(result, list), f"{p.name} didn't return a list"
                assert len(result) > 0, f"{p.name} returned empty list"

    def test_binary_prims(self):
        prims = build_atomic_primitives()
        binary = [p for p in prims if p.arity == 2]
        assert len(binary) == 2
        names = {p.name for p in binary}
        assert "overlay" in names
        assert "mask_by" in names

    def test_parameterized_prims(self):
        prims = build_parameterized_primitives()
        names = {p.name for p in prims}
        assert "swap_colors" in names
        assert "scale" in names
        assert "tile" in names
        for p in prims:
            assert p.kind == "parameterized"

    def test_perception_prims(self):
        prims = build_perception_primitives()
        names = {p.name for p in prims}
        assert "background_color" in names
        assert "grid_height" in names
        assert "n_objects" in names
        for p in prims:
            assert p.kind == "perception"

    def test_no_compound_prims(self):
        """Verify no compound operations in atomic set."""
        prims = build_atomic_primitives()
        names = {p.name for p in prims}
        compound_names = [
            "crop_to_content",  # compositional: trim_rows + trim_cols
            "extract_largest_object", "extract_smallest_object",
            "keep_largest_component", "keep_smallest_component",
        ]
        for name in compound_names:
            assert name not in names, f"compound prim {name} should not be in atomic set"

    def test_essential_pair_concepts(self):
        assert "trim_rows" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        assert "trim_cols" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        assert "dilate" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        assert "mirror_vertical" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS


# =============================================================================
# 10. Grammar integration
# =============================================================================

class TestGrammarIntegration(unittest.TestCase):
    def test_grammar_atomic_base_primitives(self):
        g = ARCGrammar(seed=42, vocabulary="atomic")
        prims = g.base_primitives()
        names = {p.name for p in prims}
        # All 6 geometric should be present
        assert "rotate_90_clockwise" in names
        assert "rotate_90_counterclockwise" in names
        assert "rotate_180" in names
        assert "mirror_horizontal" in names
        assert "mirror_vertical" in names
        assert "transpose" in names
        # trim_rows/trim_cols replace crop_to_content
        assert "trim_rows" in names
        assert "trim_cols" in names
        assert "crop_to_content" not in names
        # parameterized and perception
        assert "swap_colors" in names
        assert "background_color" in names

    def test_grammar_blocks_structural(self):
        g = ARCGrammar(seed=42, vocabulary="atomic")
        assert g.allow_structural_phases() is False


# =============================================================================
# 11. Environment execution
# =============================================================================

class TestEnvironmentExecution(unittest.TestCase):
    def test_env_executes_atomic(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        prog = Program(root="dilate")
        result = env.execute(prog, CENTER_PIXEL)
        assert isinstance(result, list)
        assert result[0][1] != 0  # dilated

    def test_env_executes_depth2(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        prog = Program(root="erode", children=[Program(root="dilate")])
        result = env.execute(prog, CENTER_PIXEL)
        assert isinstance(result, list)

    def test_env_executes_parameterized(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        prog = Program(root="swap_colors", children=[
            Program(root="background_color"),
            Program(root="dominant_color"),
        ])
        grid = [[0, 0, 1], [0, 1, 0], [0, 0, 0]]
        result = env.execute(prog, grid)
        assert isinstance(result, list)
        # background (0) and dominant (1) swapped
        assert result[0][2] == 0
        assert result[0][0] == 1

    def test_env_crop_to_content_composed(self):
        """crop_to_content = trim_cols(trim_rows(x)) executed via program tree."""
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        prog = Program(root="trim_cols", children=[Program(root="trim_rows")])
        grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        result = env.execute(prog, grid)
        assert result == [[5]]


if __name__ == "__main__":
    unittest.main()
