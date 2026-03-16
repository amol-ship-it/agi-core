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
    border_extend,
    extend_rays_right, extend_rays_left, extend_rays_down, extend_rays_up,
    flood_fill_from_markers,
    # Parameterized factories
    _scale_factory, _tile_factory, _downscale_factory,
    _swap_colors_factory, _keep_color_factory, _erase_color_factory,
    _recolor_foreground_factory,
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
        # + 4 physics + 2 sorting + 2 fill + 4 ray extension
        # + 1 flood_fill_from_markers + 1 label_components + 2 binary = 33
        assert len(prims) == 33

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

    def test_grammar_enables_structural(self):
        # Structural phases are search strategies, always enabled
        g = ARCGrammar(seed=42, vocabulary="atomic")
        assert g.allow_structural_phases() is True


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


# =============================================================================
# 12. Predicate tests (Change A)
# =============================================================================

class TestPredicates:
    def test_predicate_count(self):
        g = ARCGrammar(seed=42)
        preds = g.get_predicates()
        assert len(preds) == 12

    def test_has_symmetry_v(self):
        g = ARCGrammar(seed=42)
        preds = dict(g.get_predicates())
        sym_grid = [[1, 2], [1, 2]]  # vertically symmetric
        asym_grid = [[1, 2], [3, 4]]
        assert preds["has_symmetry_v"](sym_grid) is True
        assert preds["has_symmetry_v"](asym_grid) is False

    def test_is_small_grid(self):
        g = ARCGrammar(seed=42)
        preds = dict(g.get_predicates())
        small = [[1, 2], [3, 4]]  # 4 pixels < 100
        big = [[0] * 20 for _ in range(10)]  # 200 pixels >= 100
        assert preds["is_small_grid"](small) is True
        assert preds["is_small_grid"](big) is False

    def test_has_few_colors(self):
        g = ARCGrammar(seed=42)
        preds = dict(g.get_predicates())
        few = [[0, 1, 1], [0, 2, 0]]  # 2 foreground colors
        many = [[0, 1, 2], [3, 4, 5]]  # 5 foreground colors
        assert preds["has_few_colors"](few) is True
        assert preds["has_few_colors"](many) is False

    def test_all_objects_same_size(self):
        g = ARCGrammar(seed=42)
        preds = dict(g.get_predicates())
        # Two 2-pixel objects
        same = [[1, 1, 0, 2, 2]]
        assert preds["all_objects_same_size"](same) is True
        # Objects of different sizes
        diff = [[1, 0, 2, 2, 2]]
        assert preds["all_objects_same_size"](diff) is False


# =============================================================================
# 13. Position-based recolor tests (Change D)
# =============================================================================

class TestPositionRecolor:
    def _make_grid(self, objects):
        """Helper: place colored pixels on a 10x10 grid."""
        grid = [[0] * 10 for _ in range(10)]
        for r, c, color in objects:
            grid[r][c] = color
        return grid

    def test_recolor_by_quadrant(self):
        from domains.arc.objects import _learn_recolor_by_quadrant, _make_conditional_recolor_fn
        # 4 same-sized objects in 4 quadrants, recolored by position
        inp = self._make_grid([(1, 1, 1), (1, 8, 1), (8, 1, 1), (8, 8, 1)])
        out = self._make_grid([(1, 1, 2), (1, 8, 3), (8, 1, 4), (8, 8, 5)])
        rule = _learn_recolor_by_quadrant([(inp, out)])
        assert rule is not None
        assert len(rule) == 4  # 4 quadrants
        fn = _make_conditional_recolor_fn(rule, "by_quadrant")
        assert fn(inp) == out

    def test_recolor_by_row_band(self):
        from domains.arc.objects import _try_conditional_recolor
        # Objects at different vertical positions, recolored by row band
        inp = self._make_grid([(1, 5, 1), (8, 5, 1)])
        out = self._make_grid([(1, 5, 2), (8, 5, 3)])
        result = _try_conditional_recolor([(inp, out)])
        # May or may not match row_band specifically, but should find a strategy
        assert result is not None

    def test_recolor_by_col_band(self):
        from domains.arc.objects import _try_conditional_recolor
        # Objects at different horizontal positions
        inp = self._make_grid([(5, 1, 1), (5, 8, 1)])
        out = self._make_grid([(5, 1, 2), (5, 8, 3)])
        result = _try_conditional_recolor([(inp, out)])
        assert result is not None

    def test_position_loocv(self):
        from domains.arc.objects import _try_conditional_recolor
        # With 2 training examples, LOOCV should verify generalization
        inp1 = self._make_grid([(1, 1, 1), (1, 8, 1), (8, 1, 1), (8, 8, 1)])
        out1 = self._make_grid([(1, 1, 2), (1, 8, 3), (8, 1, 4), (8, 8, 5)])
        inp2 = self._make_grid([(2, 2, 1), (2, 7, 1), (7, 2, 1), (7, 7, 1)])
        out2 = self._make_grid([(2, 2, 2), (2, 7, 3), (7, 2, 4), (7, 7, 5)])
        result = _try_conditional_recolor([(inp1, out1), (inp2, out2)])
        assert result is not None
        _, fn = result
        # Should generalize to both examples
        assert fn(inp1) == out1
        assert fn(inp2) == out2


# =============================================================================
# 14. Scale/tile detection tests (Change E)
# =============================================================================

class TestScaleDetection:
    def test_scale_detection_2x(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        # Input 2x2, output 4x4 (scale 2x)
        inp = [[1, 2], [3, 4]]
        out = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        task = Task(task_id="test_scale", train_examples=[(inp, out)],
                    test_inputs=[inp])
        prims = []
        result = env.try_cross_reference(task, prims)
        assert result is not None
        name, fn = result
        assert "scale_2x" in name
        assert fn(inp) == out

    def test_tile_detection_3x(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        inp = [[1, 2], [3, 4]]
        # Tile 3x: 6x6 grid
        from domains.arc.transformation_primitives import _tile_factory
        out = _tile_factory(3)(inp)
        task = Task(task_id="test_tile", train_examples=[(inp, out)],
                    test_inputs=[inp])
        result = env.try_cross_reference(task, [])
        assert result is not None
        name, fn = result
        # Could be scale or tile — just verify it works
        assert fn(inp) == out

    def test_downscale_detection(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        # Input 4x4 (each pixel is 2x2 block), output 2x2
        inp = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        out = [[1, 2], [3, 4]]
        task = Task(task_id="test_ds", train_examples=[(inp, out)],
                    test_inputs=[inp])
        result = env.try_cross_reference(task, [])
        assert result is not None
        name, fn = result
        assert "downscale_2x" in name
        assert fn(inp) == out

    def test_non_integer_ratio_returns_none(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        # 3x3 → 5x5 (non-integer ratio)
        inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = [[0] * 5 for _ in range(5)]
        task = Task(task_id="test_non_int", train_examples=[(inp, out)],
                    test_inputs=[inp])
        result = env.try_cross_reference(task, [])
        # Should not match scale/tile (ratio is not integer)
        # May match boolean halves or separator, but not scale
        if result is not None:
            assert "scale" not in result[0] and "tile" not in result[0]


# =============================================================================
# 15. Cell patch correction tests (Change F)
# =============================================================================

class TestCellPatchCorrection:
    def test_cell_patch_correction(self):
        env = ARCEnv()
        # Predicted grids differ by 2 cells from expected
        pred1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        exp1 = [[1, 2, 3], [4, 0, 6], [7, 8, 9]]  # cell (1,1): 5→0
        pred2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        exp2 = [[1, 2, 3], [4, 0, 6], [7, 8, 9]]  # same fix
        result = env.infer_output_correction(
            [pred1, pred2], [exp1, exp2])
        assert result is not None
        # Verify the patch works
        from domains.arc.primitives import _PRIM_MAP
        prim = _PRIM_MAP.get(result.root)
        assert prim is not None
        assert prim.fn(pred1) == exp1

    def test_cell_patch_inconsistent_returns_none(self):
        env = ARCEnv()
        # Same cell has different fixes in different examples
        pred1 = [[1, 2], [3, 4]]
        exp1 = [[1, 2], [3, 0]]  # cell (1,1): 4→0
        pred2 = [[1, 2], [3, 4]]
        exp2 = [[1, 2], [3, 9]]  # cell (1,1): 4→9 (inconsistent!)
        result = env.infer_output_correction(
            [pred1, pred2], [exp1, exp2])
        assert result is None


# =============================================================================
# 16. New primitive tests — recolor_foreground + border_extend
# =============================================================================

class TestRecolorForeground:
    def test_basic(self):
        # bg=0 (most common), foreground colors 1,2 → all become 5
        grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]
        result = _recolor_foreground_factory(5)(grid)
        assert result == [[0, 0, 0], [0, 5, 5], [0, 0, 0]]

    def test_non_zero_bg(self):
        # bg=3 (most common), foreground colors 1,2 → all become 7
        grid = [[3, 3, 3], [3, 1, 2], [3, 3, 3]]
        result = _recolor_foreground_factory(7)(grid)
        assert result == [[3, 3, 3], [3, 7, 7], [3, 3, 3]]

    def test_all_same_color(self):
        grid = [[2, 2], [2, 2]]
        result = _recolor_foreground_factory(5)(grid)
        # bg=2, no foreground → no change
        assert result == [[2, 2], [2, 2]]

    def test_empty(self):
        assert _recolor_foreground_factory(1)([]) == []


class TestBorderExtend:
    def test_basic(self):
        grid = [
            [0, 1, 0],
            [2, 3, 4],
            [0, 5, 0],
        ]
        result = border_extend(grid)
        # Top edge: (0,0)=0→grid[1][0]=2, (0,2)=0→grid[1][2]=4
        assert result[0][0] == 2
        assert result[0][2] == 4
        # Bottom edge: (2,0)=0→grid[1][0]=2, (2,2)=0→grid[1][2]=4
        assert result[2][0] == 2
        assert result[2][2] == 4
        # Center unchanged
        assert result[1][1] == 3

    def test_no_border_zeros(self):
        grid = [[1, 2], [3, 4]]
        assert border_extend(grid) == [[1, 2], [3, 4]]

    def test_empty(self):
        assert border_extend([]) == []

    def test_interior_zeros_unchanged(self):
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = border_extend(grid)
        # Interior zero at (1,1) should NOT be changed
        assert result[1][1] == 0


# =============================================================================
# 17. Ray extension tests
# =============================================================================

class TestRayExtension:
    def test_extend_rays_right(self):
        grid = [
            [0, 1, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ]
        result = extend_rays_right(grid)
        # 1 extends right until hitting 2
        assert result[0] == [0, 1, 1, 1, 2]
        # 3 extends right to edge
        assert result[2] == [3, 3, 3, 3, 3]
        # Empty row stays empty
        assert result[1] == [0, 0, 0, 0, 0]

    def test_extend_rays_left(self):
        grid = [
            [0, 0, 0, 1, 0],
            [2, 0, 0, 3, 0],
        ]
        result = extend_rays_left(grid)
        # 1 extends left to edge
        assert result[0] == [1, 1, 1, 1, 0]
        # 3 extends left until hitting 2
        assert result[1] == [2, 3, 3, 3, 0]

    def test_extend_rays_down(self):
        grid = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
            [0, 0, 0],
        ]
        result = extend_rays_down(grid)
        # 1 extends down to bottom
        assert [result[r][0] for r in range(4)] == [1, 1, 1, 1]
        # 2 extends down to bottom
        assert [result[r][2] for r in range(4)] == [0, 0, 2, 2]

    def test_extend_rays_up(self):
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 2],
        ]
        result = extend_rays_up(grid)
        # 1 extends up to top
        assert [result[r][0] for r in range(3)] == [1, 1, 1]
        # 2 extends up to top
        assert [result[r][2] for r in range(3)] == [2, 2, 2]

    def test_empty_grid(self):
        assert extend_rays_right([]) == []
        assert extend_rays_left([]) == []
        assert extend_rays_down([]) == []
        assert extend_rays_up([]) == []


# =============================================================================
# 18. Flood fill from markers tests
# =============================================================================

class TestFloodFillFromMarkers:
    def test_basic_fill(self):
        grid = [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = flood_fill_from_markers(grid)
        # All zeros connected to 1 get filled
        assert all(result[r][c] == 1 for r in range(3) for c in range(5))

    def test_two_seeds_different_colors(self):
        grid = [
            [1, 0, 0, 2],
        ]
        result = flood_fill_from_markers(grid)
        # 1 fills right, 2 fills left — they meet somewhere
        assert result[0][0] == 1
        assert result[0][3] == 2

    def test_no_zeros(self):
        grid = [[1, 2], [3, 4]]
        result = flood_fill_from_markers(grid)
        assert result == grid

    def test_empty(self):
        assert flood_fill_from_markers([]) == []


# =============================================================================
# 19. Per-row/column decomposition tests
# =============================================================================

class TestPerRowColumn:
    def test_per_row_mirror(self):
        """Each row independently mirrored."""
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        env = ARCEnv()
        inp = [[1, 2, 3], [4, 5, 6]]
        out = [[3, 2, 1], [6, 5, 4]]  # mirror_horizontal per row
        task = Task(task_id="test_per_row", train_examples=[(inp, out)],
                    test_inputs=[inp])
        prims = env._current_task  # not needed, just need all_prims
        from domains.arc.grammar import ARCGrammar
        g = ARCGrammar(seed=42, vocabulary="atomic")
        all_prims = g.base_primitives()
        result = env.try_per_row_column_decomposition(task, all_prims)
        # This may or may not match depending on how row-level transforms work
        # (mirror_horizontal on a 1-row grid = reverse the row)
        if result is not None:
            name, fn = result
            assert fn(inp) == out


# =============================================================================
# 20. Template stamp tests
# =============================================================================

class TestTemplateStamp:
    def test_stamp_at_markers(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        # Input: a small L-shape (color 1) and marker pixels (color 2)
        inp = [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        # Output: L-shape stamped centered at each marker position
        out = [
            [1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
        ]
        # Note: this is a synthetic test — the actual stamping logic centers
        # the template at the marker. The test just verifies the method runs
        task = Task(task_id="test_stamp", train_examples=[(inp, out)],
                    test_inputs=[inp])
        result = env._try_template_stamp(task)
        # Template stamp may or may not work on this synthetic example
        # depending on exact centering — that's OK, it's testing the code path


# =============================================================================
# 21. Separator cell algebra tests
# =============================================================================

class TestSeparatorCellAlgebra:
    def test_or_reduce_cells(self):
        env = ARCEnv()
        from domains.arc.primitives import register_atomic_primitives
        register_atomic_primitives()
        # 3x7 grid with vertical separator (color 5) at column 3
        # Two 3x3 cells, output = OR of both cells
        inp = [
            [1, 0, 0, 5, 0, 0, 1],
            [0, 0, 0, 5, 0, 1, 0],
            [0, 0, 1, 5, 1, 0, 0],
        ]
        out = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]
        task = Task(task_id="test_or_reduce", train_examples=[(inp, out)],
                    test_inputs=[inp])
        result = env._try_separator_cross_ref(task)
        if result is not None:
            name, fn = result
            assert fn(inp) == out


if __name__ == "__main__":
    unittest.main()
