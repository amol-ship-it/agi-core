"""Tests for object decomposition (per-object transforms)."""

import pytest
from domains.arc.objects import (
    _find_connected_components,
    find_foreground_shapes,
    place_subgrid,
    apply_transform_per_object,
    _match_objects_by_position,
    _shape_signature,
    _compactness,
    _has_hole,
)


class TestConnectedComponents:
    def test_single_object(self):
        grid = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
        comps = _find_connected_components(grid)
        assert len(comps) == 1
        assert comps[0]["color"] == 1
        assert comps[0]["size"] == 3

    def test_two_objects_different_colors(self):
        grid = [[1, 0, 2], [1, 0, 2], [0, 0, 0]]
        comps = _find_connected_components(grid)
        assert len(comps) == 2

    def test_two_objects_same_color_disconnected(self):
        grid = [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
        comps = _find_connected_components(grid)
        assert len(comps) == 2

    def test_empty_grid(self):
        grid = [[0, 0], [0, 0]]
        assert _find_connected_components(grid) == []


class TestFindForegroundShapes:
    def test_returns_subgrid(self):
        grid = [[0, 0, 0], [0, 1, 1], [0, 0, 0]]
        shapes = find_foreground_shapes(grid)
        assert len(shapes) == 1
        assert shapes[0]["subgrid"] == [[1, 1]]
        assert shapes[0]["position"] == (1, 1)
        assert shapes[0]["size"] == 2

    def test_multiple_objects(self):
        grid = [[1, 0, 2], [0, 0, 0], [3, 0, 0]]
        shapes = find_foreground_shapes(grid)
        assert len(shapes) == 3


class TestPlaceSubgrid:
    def test_basic_placement(self):
        canvas = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        subgrid = [[1, 1]]
        result = place_subgrid(canvas, subgrid, (1, 1))
        assert result == [[0, 0, 0], [0, 1, 1], [0, 0, 0]]

    def test_transparent(self):
        canvas = [[5, 5], [5, 5]]
        subgrid = [[1, 0], [0, 1]]
        result = place_subgrid(canvas, subgrid, (0, 0), transparent_color=0)
        assert result == [[1, 5], [5, 1]]


class TestApplyTransformPerObject:
    def test_identity_transform(self):
        grid = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_transform_per_object(grid, lambda g: g, bg_color=0)
        assert result == grid

    def test_mirror_per_object(self):
        # Object is [[1, 1], [1, 0]]. Mirror H → [[1, 1], [0, 1]]
        grid = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        def mirror_h(g):
            return [row[::-1] for row in g]
        result = apply_transform_per_object(grid, mirror_h, bg_color=0)
        assert result == [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]


class TestShapeHelpers:
    def test_shape_signature_invariant(self):
        shape1 = {"subgrid": [[1, 1], [1, 0]], "bbox": (0, 0, 1, 1),
                  "color": 1, "size": 3, "position": (0, 0)}
        shape2 = {"subgrid": [[2, 2], [2, 0]], "bbox": (5, 5, 6, 6),
                  "color": 2, "size": 3, "position": (5, 5)}
        assert _shape_signature(shape1) == _shape_signature(shape2)

    def test_compactness_rectangle(self):
        shape = {"bbox": (0, 0, 1, 1), "size": 4}
        assert _compactness(shape) == 1.0

    def test_compactness_l_shape(self):
        shape = {"bbox": (0, 0, 1, 1), "size": 3}
        assert _compactness(shape) == 0.75

    def test_has_hole_ring(self):
        shape = {"subgrid": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                 "bbox": (0, 0, 2, 2), "size": 8, "position": (0, 0)}
        assert _has_hole(shape)

    def test_no_hole_solid(self):
        shape = {"subgrid": [[1, 1], [1, 1]],
                 "bbox": (0, 0, 1, 1), "size": 4, "position": (0, 0)}
        assert not _has_hole(shape)


class TestMatchObjects:
    def test_simple_match(self):
        inp = [[1, 0, 2], [0, 0, 0]]
        out = [[3, 0, 4], [0, 0, 0]]
        matches = _match_objects_by_position(inp, out)
        assert matches is not None
        assert len(matches) == 2
