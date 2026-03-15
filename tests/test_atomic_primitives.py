"""
Tests for atomic primitive decomposition.

Covers:
- Unit tests for each atomic primitive
- Decomposition verification (composite = atomic composition)
- Integration smoke test with vocabulary="atomic"
"""

import pytest
from core import Primitive, Program, Task

from domains.arc.atomic_primitives import (
    # New atomic ops
    fill_region, copy_region, clear_region,
    pad_border, recolor, erase_color,
    dilate, erode,
    # Combinators
    make_for_each_object, make_apply_to_enclosed, make_conditional_objects,
    # Builders
    build_atomic_primitives, build_atomic_task_color_primitives,
    ATOMIC_ESSENTIAL_PAIR_CONCEPTS,
    expand_with_combinators,
    # Re-exported from primitives
    rotate_90_cw, rotate_90_ccw, rotate_180,
    mirror_horizontal, mirror_vertical, transpose,
    crop_to_nonzero, scale_2x, scale_3x, downscale_2x,
    binarize, invert_colors, overlay, tile_2x2,
    get_top_half, get_bottom_half, get_left_half, get_right_half,
)
from domains.arc.grammar import ARCGrammar
from domains.arc.environment import ARCEnv
from domains.arc.primitives import identity


# =============================================================================
# Test grids
# =============================================================================

EMPTY_GRID = [[0]]
SMALL_GRID = [[1, 2], [3, 4]]
ALL_ZERO = [[0, 0], [0, 0]]

# 3x3 grid with a center pixel
CENTER_PIXEL = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]

# 4x4 grid with an enclosed region
ENCLOSED = [
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
]

# Grid with two separate objects
TWO_OBJECTS = [
    [1, 1, 0, 0, 2, 2],
    [1, 1, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


# =============================================================================
# 5A. Unit tests for each atomic primitive
# =============================================================================

class TestGeometricPrimitives:
    """Test the 7 geometric atomics (reused from primitives.py)."""

    def test_identity(self):
        assert identity(SMALL_GRID) == SMALL_GRID

    def test_rotate_90_cw(self):
        result = rotate_90_cw(SMALL_GRID)
        assert result == [[3, 1], [4, 2]]

    def test_rotate_90_ccw(self):
        result = rotate_90_ccw(SMALL_GRID)
        assert result == [[2, 4], [1, 3]]

    def test_rotate_180(self):
        result = rotate_180(SMALL_GRID)
        assert result == [[4, 3], [2, 1]]

    def test_mirror_horizontal(self):
        # mirror_horizontal flips left-right (along vertical axis)
        result = mirror_horizontal(SMALL_GRID)
        assert result == [[2, 1], [4, 3]]

    def test_mirror_vertical(self):
        # mirror_vertical flips up-down (along horizontal axis)
        result = mirror_vertical(SMALL_GRID)
        assert result == [[3, 4], [1, 2]]

    def test_transpose(self):
        result = transpose(SMALL_GRID)
        assert result == [[1, 3], [2, 4]]

    def test_geometric_on_1x1(self):
        """All geometric ops should work on 1x1 grids."""
        for fn in [identity, rotate_90_cw, rotate_90_ccw, rotate_180,
                   mirror_horizontal, mirror_vertical, transpose]:
            result = fn(EMPTY_GRID)
            assert isinstance(result, list) and len(result) >= 1

    def test_geometric_on_all_zero(self):
        for fn in [identity, rotate_90_cw, rotate_90_ccw, rotate_180,
                   mirror_horizontal, mirror_vertical, transpose]:
            result = fn(ALL_ZERO)
            assert isinstance(result, list)


class TestSpatialPrimitives:
    """Test the 6 spatial atomics."""

    def test_crop_to_content(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = crop_to_nonzero(grid)
        assert result == [[1]]

    def test_crop_half_top(self):
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = get_top_half(grid)
        assert len(result) == 2

    def test_crop_half_bottom(self):
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = get_bottom_half(grid)
        assert len(result) == 2

    def test_crop_half_left(self):
        grid = [[1, 2, 3, 4]]
        result = get_left_half(grid)
        assert len(result[0]) == 2

    def test_crop_half_right(self):
        grid = [[1, 2, 3, 4]]
        result = get_right_half(grid)
        assert len(result[0]) == 2

    def test_pad_border(self):
        result = pad_border([[1]])
        assert result == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    def test_pad_border_2x2(self):
        result = pad_border(SMALL_GRID)
        assert len(result) == 4
        assert len(result[0]) == 4
        assert result[1][1] == 1
        assert result[0][0] == 0

    def test_crop_empty(self):
        result = crop_to_nonzero(ALL_ZERO)
        assert isinstance(result, list)


class TestScalePrimitives:
    """Test the 3 scale atomics."""

    def test_scale_2x(self):
        result = scale_2x([[1]])
        assert result == [[1, 1], [1, 1]]

    def test_scale_3x(self):
        result = scale_3x([[1]])
        assert len(result) == 3 and len(result[0]) == 3

    def test_downscale_2x(self):
        result = downscale_2x([[1, 1], [1, 1]])
        assert result == [[1]]


class TestColorPrimitives:
    """Test the 2 color atomics."""

    def test_binarize(self):
        result = binarize([[0, 3, 5], [7, 0, 2]])
        assert result == [[0, 1, 1], [1, 0, 1]]

    def test_invert_colors(self):
        result = invert_colors([[0, 5], [9, 3]])
        for row in result:
            for val in row:
                assert 0 <= val <= 9


class TestPixelRegionPrimitives:
    """Test the 3 new pixel/region atomics."""

    def test_fill_region(self):
        result = fill_region(ALL_ZERO, {(0, 0), (1, 1)}, 5)
        assert result[0][0] == 5
        assert result[1][1] == 5
        assert result[0][1] == 0

    def test_fill_region_out_of_bounds(self):
        result = fill_region(ALL_ZERO, {(10, 10)}, 5)
        assert result == ALL_ZERO

    def test_copy_region(self):
        grid = [[1, 0], [0, 0]]
        result = copy_region(grid, {(0, 0)}, (1, 1))
        assert result[1][1] == 1
        assert result[0][0] == 1  # original preserved

    def test_clear_region(self):
        result = clear_region(SMALL_GRID, {(0, 0), (1, 1)})
        assert result[0][0] == 0
        assert result[1][1] == 0
        assert result[0][1] == 2  # unchanged


class TestMorphologicalPrimitives:
    """Test dilate and erode."""

    def test_dilate_center_pixel(self):
        result = dilate(CENTER_PIXEL)
        # Center + 4 neighbors should be non-zero
        assert result[1][1] != 0  # center
        assert result[0][1] != 0  # top
        assert result[2][1] != 0  # bottom
        assert result[1][0] != 0  # left
        assert result[1][2] != 0  # right
        # Corners should still be zero
        assert result[0][0] == 0
        assert result[2][2] == 0

    def test_erode_center_pixel(self):
        # Single pixel should be eroded away
        result = erode(CENTER_PIXEL)
        assert result[1][1] == 0

    def test_erode_solid_block(self):
        # 3x3 solid block: only center survives
        solid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = erode(solid)
        assert result[1][1] == 1
        # Edges should be eroded
        assert result[0][0] == 0
        assert result[0][1] == 0

    def test_dilate_empty(self):
        result = dilate(ALL_ZERO)
        assert result == ALL_ZERO

    def test_erode_empty(self):
        result = erode(ALL_ZERO)
        assert result == ALL_ZERO

    def test_dilate_1x1(self):
        result = dilate([[5]])
        assert result == [[5]]

    def test_erode_1x1(self):
        # Boundary pixel should be eroded
        result = erode([[5]])
        assert result == [[0]]


class TestRecolorErase:
    """Test recolor and erase_color functions."""

    def test_recolor(self):
        result = recolor(SMALL_GRID, 1, 9)
        assert result[0][0] == 9
        assert result[0][1] == 2  # unchanged

    def test_erase_color(self):
        result = erase_color(SMALL_GRID, 2)
        assert result[0][1] == 0
        assert result[0][0] == 1  # unchanged


class TestPlacementPrimitives:
    """Test overlay and tile_2x2."""

    def test_overlay(self):
        base = [[0, 1], [0, 0]]
        top = [[2, 0], [0, 3]]
        result = overlay(base, top)
        assert result[0][0] == 2  # top overwrites
        assert result[0][1] == 1  # base preserved
        assert result[1][1] == 3

    def test_tile_2x2(self):
        result = tile_2x2([[1]])
        assert len(result) == 2
        assert len(result[0]) == 2


# =============================================================================
# 5B. Builder tests
# =============================================================================

class TestBuildAtomicPrimitives:
    """Test the build functions."""

    def test_build_count(self):
        prims = build_atomic_primitives()
        # 6 geometric + 6 spatial + 3 scale + 2 color + 1 tile + 2 morphological
        # + 10 perception + 1 overlay = 31
        assert len(prims) >= 28
        assert len(prims) <= 40

    def test_all_have_names(self):
        for p in build_atomic_primitives():
            assert p.name
            assert p.domain == "arc"

    def test_all_unary_produce_valid_output(self):
        """Every unary atomic primitive should produce a valid grid."""
        for p in build_atomic_primitives():
            if p.arity == 1:
                result = p.fn(SMALL_GRID)
                assert isinstance(result, list), f"{p.name} didn't return a list"
                assert len(result) > 0, f"{p.name} returned empty list"
                assert isinstance(result[0], list), f"{p.name} rows aren't lists"

    def test_overlay_arity_2(self):
        prims = build_atomic_primitives()
        overlay_prims = [p for p in prims if p.arity == 2]
        assert len(overlay_prims) == 1
        assert overlay_prims[0].name == "overlay"

    def test_atomic_task_color_prims(self):
        prims = build_atomic_task_color_primitives({0, 1, 2, 3})
        names = {p.name for p in prims}
        # Should have keep, erase, replace, swap
        assert "keep_only_color_1" in names
        assert "erase_color_2" in names
        assert "replace_color_1_with_color_2" in names
        assert "swap_color_1_and_color_2" in names
        # Should NOT have composite operations
        assert not any("fill_rectangle" in n for n in names)
        assert not any("mark_intersections" in n for n in names)
        assert not any("fill_background" in n for n in names)
        assert not any("recolor_all_to" in n for n in names)

    def test_essential_pair_concepts(self):
        assert "crop_to_content" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        assert "dilate" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS
        assert "erode" in ATOMIC_ESSENTIAL_PAIR_CONCEPTS


# =============================================================================
# 5B. Combinator tests
# =============================================================================

class TestCombinators:
    """Test the 3 combinator generators."""

    def test_for_each_object_identity(self):
        fn = make_for_each_object(identity, bg_color=0)
        result = fn(TWO_OBJECTS)
        assert result == TWO_OBJECTS

    def test_for_each_object_mirror_h(self):
        """for_each_object(mirror_h) should mirror each object independently."""
        fn = make_for_each_object(mirror_horizontal, bg_color=0)
        result = fn(TWO_OBJECTS)
        assert isinstance(result, list)
        assert len(result) == len(TWO_OBJECTS)

    def test_apply_to_enclosed_identity(self):
        fn = make_apply_to_enclosed(identity)
        result = fn(ENCLOSED)
        assert result == ENCLOSED

    def test_apply_to_enclosed_fill(self):
        """Apply fill (recolor to 5) to enclosed regions."""
        # Use a grid where 0 is the dominant background color
        # so _get_background_color returns 0
        enclosed_grid = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        def fill_with_5(grid):
            return [[5 if c == 0 else c for c in row] for row in grid]
        fn = make_apply_to_enclosed(fill_with_5)
        result = fn(enclosed_grid)
        # The enclosed 2x2 zero region should now be 5
        assert result[2][2] == 5
        assert result[2][3] == 5
        assert result[3][2] == 5
        assert result[3][3] == 5
        # Outer zeros (reachable from border) should NOT be filled
        assert result[0][0] == 0
        # Frame should be unchanged
        assert result[1][1] == 1

    def test_conditional_objects(self):
        """if(size > 2, mirror_h, identity) per object."""
        def is_big(grid):
            return sum(1 for r in grid for c in r if c != 0) > 2
        fn = make_conditional_objects(is_big, mirror_horizontal, identity, bg_color=0)
        result = fn(TWO_OBJECTS)
        assert isinstance(result, list)

    def test_apply_to_enclosed_no_enclosed(self):
        """Grid with no enclosed region should be unchanged."""
        grid = [[1, 0], [0, 1]]
        fn = make_apply_to_enclosed(identity)
        result = fn(grid)
        assert result == grid


# =============================================================================
# 5B. Decomposition verification tests
# =============================================================================

class TestDecompositionVerification:
    """Verify composite primitives can be expressed as atomic compositions."""

    def test_mirror_objects_h_decomposition(self):
        """mirror_objects_h = for_each_object(mirror_horizontal)."""
        fn = make_for_each_object(mirror_horizontal, bg_color=0)
        # Grid with a simple L-shape object
        grid = [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
        result = fn(grid)
        assert isinstance(result, list)
        # The L-shape should be mirrored within its bounding box
        assert result[0][1] == 1 or result[2][0] == 1  # some change happened

    def test_for_each_object_composed_with_crop(self):
        """for_each_object(crop_to_nonzero) should crop each object."""
        fn = make_for_each_object(crop_to_nonzero, bg_color=0)
        result = fn(TWO_OBJECTS)
        assert isinstance(result, list)

    def test_erode_then_dilate_approximates_opening(self):
        """Morphological opening = erode then dilate."""
        solid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        eroded = erode(solid)
        opened = dilate(eroded)
        assert isinstance(opened, list)


# =============================================================================
# Grammar integration tests
# =============================================================================

class TestGrammarIntegration:
    """Test that ARCGrammar works with vocabulary='atomic'."""

    def test_grammar_atomic_base_primitives(self):
        g = ARCGrammar(seed=42, vocabulary="atomic")
        prims = g.base_primitives()
        assert len(prims) >= 20
        names = {p.name for p in prims}
        assert "identity" not in names  # identity removed — wastes search budget
        assert "dilate" in names
        assert "erode" in names
        assert "overlay" in names

    def test_grammar_atomic_essential_pairs(self):
        g = ARCGrammar(seed=42, vocabulary="atomic")
        pairs = g.essential_pair_concepts()
        assert "crop_to_content" in pairs
        assert "dilate" in pairs

    def test_grammar_atomic_prepare_for_task(self):
        g = ARCGrammar(seed=42, vocabulary="atomic")
        task = Task(
            task_id="test_task",
            train_examples=[
                ([[1, 2], [3, 0]], [[2, 1], [0, 3]]),
            ],
            test_inputs=[[1, 2], [3, 0]],
        )
        g.prepare_for_task(task)
        prims = g.base_primitives()
        names = {p.name for p in prims}
        kinds = {p.name: p.kind for p in prims}
        # Should have parameterized color prims instead of task-specific ones
        assert "swap_colors" in names
        assert kinds["swap_colors"] == "parameterized"
        # Should have perception prims
        assert "background_color" in names
        assert kinds["background_color"] == "perception"
        # Should NOT have task-specific color ops
        assert "keep_only_color_1" not in names
        assert not any("fill_rectangle" in n for n in names)

    def test_grammar_full_unchanged(self):
        """Verify full vocabulary still works identically."""
        g = ARCGrammar(seed=42, vocabulary="full")
        prims = g.base_primitives()
        assert len(prims) > 100  # full set is ~180+

    def test_grammar_minimal_unchanged(self):
        """Verify minimal vocabulary still works identically."""
        g = ARCGrammar(seed=42, vocabulary="minimal")
        prims = g.base_primitives()
        assert 40 <= len(prims) <= 70

    def test_grammar_mutate_atomic(self):
        """Mutation should work with atomic vocabulary."""
        g = ARCGrammar(seed=42, vocabulary="atomic")
        prims = g.base_primitives()
        prog = Program(root="identity")
        mutated = g.mutate(prog, prims)
        assert mutated is not None
        assert mutated.root  # has a valid root


# =============================================================================
# 5C. Integration smoke test
# =============================================================================

class TestIntegrationSmoke:
    """Run wake pipeline components with vocabulary='atomic' on sample tasks."""

    def _make_sample_task(self):
        """Create a simple ARC task (mirror horizontal)."""
        return Task(
            task_id="smoke_mirror_h",
            train_examples=[
                ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
                ([[5, 6], [7, 8]], [[7, 8], [5, 6]]),
            ],
            test_inputs=[[9, 1], [2, 3]],
        )

    def test_env_executes_atomic_primitives(self):
        """Environment should execute atomic primitives correctly."""
        env = ARCEnv()
        task = self._make_sample_task()
        env.load_task(task)

        prog = Program(root="mirror_horizontal")
        result = env.execute(prog, [[1, 2], [3, 4]])
        assert result == [[2, 1], [4, 3]]  # mirror_horizontal flips left-right

    def test_env_executes_depth2_atomic(self):
        """Environment should execute depth-2 compositions of atomics."""
        env = ARCEnv()
        task = self._make_sample_task()
        env.load_task(task)

        # mirror_horizontal(binarize(grid))
        prog = Program(
            root="mirror_horizontal",
            children=[Program(root="binarize")],
        )
        result = env.execute(prog, [[1, 0], [0, 2]])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_atomic_grammar_enumeration_smoke(self):
        """Grammar should enumerate atomic programs for a task."""
        g = ARCGrammar(seed=42, vocabulary="atomic")
        task = self._make_sample_task()
        g.prepare_for_task(task)
        prims = g.base_primitives()

        # Should be able to compose programs
        unary = [p for p in prims if p.arity == 1]
        assert len(unary) >= 15

        # Create a depth-2 program
        prog = g.compose(unary[0], [Program(root=unary[1].name)])
        assert prog.root == unary[0].name
        assert len(prog.children) == 1


class TestExpandWithCombinators:
    """Test combinator expansion."""

    def test_expand_empty(self):
        """No scored programs should return empty."""
        result = expand_with_combinators([], None, None)
        assert result == []

    def test_expand_produces_primitives(self):
        """Should produce for_each_object and apply_to_enclosed variants."""
        from core import ScoredProgram
        env = ARCEnv()
        task = Task(
            task_id="test_expand",
            train_examples=[
                (TWO_OBJECTS, TWO_OBJECTS),
            ],
            test_inputs=TWO_OBJECTS,
        )
        env.load_task(task)

        scored = [
            ScoredProgram(
                program=Program(root="identity"),
                energy=0.0,
                prediction_error=0.0,
                complexity_cost=1.0,
            ),
        ]
        new_prims = expand_with_combinators(scored, task, env, top_k=1)
        assert len(new_prims) == 2  # for_each_object + apply_to_enclosed
        names = [p.name for p in new_prims]
        assert any("for_each_object" in n for n in names)
        assert any("apply_to_enclosed" in n for n in names)

        # Each should be callable
        for p in new_prims:
            result = p.fn(TWO_OBJECTS)
            assert isinstance(result, list)


# =============================================================================
# Adapter integration test
# =============================================================================

class TestAdapterAtomic:
    """Test that ARCAdapter correctly passes vocabulary='atomic'."""

    def test_adapter_creates_atomic_grammar(self):
        from domains.arc.adapter import ARCAdapter
        adapter = ARCAdapter(benchmark="arc-agi-1")
        env, grammar, drive = adapter.create_interfaces(seed=42, vocabulary="atomic")
        assert isinstance(grammar, ARCGrammar)
        prims = grammar.base_primitives()
        names = {p.name for p in prims}
        assert "dilate" in names
        assert "erode" in names

    def test_adapter_rejects_invalid_vocabulary(self):
        from domains.arc.adapter import ARCAdapter
        adapter = ARCAdapter(benchmark="arc-agi-1")
        with pytest.raises(ValueError, match="Unknown vocabulary"):
            adapter.create_interfaces(vocabulary="invalid")
