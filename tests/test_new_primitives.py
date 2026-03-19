"""Tests for new primitives added in stratified search expansion."""
import pytest


# =============================================================================
# Task 10: Inpainting primitives
# =============================================================================

def test_inpaint_by_neighbors():
    from domains.arc.transformation_primitives import inpaint_by_neighbors
    grid = [[2, 2, 2], [2, 0, 2], [2, 2, 2]]
    result = inpaint_by_neighbors(grid)
    assert result[1][1] == 2

def test_inpaint_by_neighbors_no_holes():
    from domains.arc.transformation_primitives import inpaint_by_neighbors
    grid = [[1, 2], [3, 4]]
    result = inpaint_by_neighbors(grid)
    assert result == grid

def test_symmetry_complete_horizontal():
    from domains.arc.transformation_primitives import symmetry_complete
    # Perfect horizontal symmetry: each row reads the same L→R and R→L.
    # Half the right side zeroed out so it needs inpainting.
    # Row 0: [1, 2, 2, 1] — mirror pairs (0,3)=1,1 match; (1,2)=2,2 match → score=1.0
    # We zero out one of the matched positions to create a hole:
    grid = [[1, 2, 0, 1], [3, 4, 4, 3]]
    result = symmetry_complete(grid)
    # [0][2] should be filled to 2 (mirror of [0][1])
    assert result[0][2] == 2

def test_symmetry_complete_no_change_when_low_score():
    from domains.arc.transformation_primitives import symmetry_complete
    # Grid with no symmetry and no zeros — should return unchanged
    grid = [[1, 2], [3, 4]]
    result = symmetry_complete(grid)
    assert result == grid

def test_fill_by_row_col_pattern():
    from domains.arc.transformation_primitives import fill_by_row_col_pattern
    grid = [[1, 1, 0], [0, 0, 0], [2, 0, 0]]
    result = fill_by_row_col_pattern(grid)
    # Row 0 dominant color is 1 → zero at [0][2] should become 1
    assert result[0][2] == 1

def test_inpaint_diagonal():
    from domains.arc.transformation_primitives import inpaint_diagonal
    # Main diagonal: all 3s → zero should be filled
    grid = [[3, 0, 0], [0, 0, 0], [0, 0, 3]]
    result = inpaint_diagonal(grid)
    assert result[1][1] == 3

def test_inpaint_from_template():
    from domains.arc.transformation_primitives import inpaint_from_template
    # 2x2 pattern [[1,2],[3,4]] repeated; one instance has a zero
    grid = [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 0],
        [3, 4, 3, 4],
    ]
    result = inpaint_from_template(grid)
    assert result[2][3] == 2


# =============================================================================
# Task 10: Denoising primitives
# =============================================================================

def test_remove_isolated():
    from domains.arc.transformation_primitives import remove_isolated
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    result = remove_isolated(grid)
    assert result[1][1] == 0

def test_remove_isolated_keeps_connected():
    from domains.arc.transformation_primitives import remove_isolated
    grid = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
    result = remove_isolated(grid)
    assert result[0][0] == 1
    assert result[0][1] == 1

def test_majority_filter_3x3():
    from domains.arc.transformation_primitives import majority_filter_3x3
    grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    result = majority_filter_3x3(grid)
    assert result[1][1] == 1

def test_morphological_close():
    from domains.arc.transformation_primitives import morphological_close
    # Two 3x3 blocks of 1s separated by a 1-pixel gap in the center row.
    # After dilate the gap fills; after erode the center cell of row 2 survives
    # because it is fully surrounded by dilated 1s.
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    result = morphological_close(grid)
    # The center gap at [2][3] should be bridged after closing
    assert result[2][3] == 1


# =============================================================================
# Task 10: Grid structure primitives
# =============================================================================

def test_remove_border():
    from domains.arc.transformation_primitives import remove_border
    grid = [[5, 5, 5], [5, 1, 5], [5, 5, 5]]
    result = remove_border(grid)
    assert result == [[1]]

def test_remove_border_too_small():
    from domains.arc.transformation_primitives import remove_border
    grid = [[1, 2], [3, 4]]
    result = remove_border(grid)
    assert result == grid


# =============================================================================
# Task 10: Cardinal extension primitives
# =============================================================================

def test_extend_up():
    from domains.arc.transformation_primitives import extend_up
    grid = [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    result = extend_up(grid)
    assert result[0][1] == 1
    assert result[1][1] == 1
    assert result[2][1] == 1

def test_extend_left():
    from domains.arc.transformation_primitives import extend_left
    grid = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    result = extend_left(grid)
    assert result[1][0] == 1
    assert result[1][1] == 1

def test_extend_right():
    from domains.arc.transformation_primitives import extend_right
    grid = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    result = extend_right(grid)
    assert result[1][1] == 1
    assert result[1][2] == 1


# =============================================================================
# Task 11: Object relationship primitives
# =============================================================================

def test_n_objects_perception():
    from domains.arc.perception_primitives import n_objects
    grid = [[1, 0, 2], [0, 0, 0], [3, 0, 0]]
    assert n_objects(grid) == 3

def test_n_objects_empty():
    from domains.arc.perception_primitives import n_objects
    grid = [[0, 0], [0, 0]]
    assert n_objects(grid) == 0

def test_n_objects_connected():
    from domains.arc.perception_primitives import n_objects
    grid = [[1, 1], [1, 0]]
    assert n_objects(grid) == 1

def test_draw_line_between_objects():
    from domains.arc.transformation_primitives import draw_line_between_objects
    grid = [[1, 0, 0, 1], [0, 0, 0, 0]]
    result = draw_line_between_objects(grid)
    assert result[0][1] == 1
    assert result[0][2] == 1

def test_draw_line_between_objects_vertical():
    from domains.arc.transformation_primitives import draw_line_between_objects
    grid = [[1, 0], [0, 0], [1, 0]]
    result = draw_line_between_objects(grid)
    assert result[1][0] == 1

def test_color_by_object_rank():
    from domains.arc.transformation_primitives import color_by_object_rank
    grid = [[1, 1, 0, 2, 0, 3, 3, 3]]
    result = color_by_object_rank(grid)
    assert result[0][5] == 1  # largest (3 cells)
    assert result[0][0] == 2  # medium (2 cells)
    assert result[0][3] == 3  # smallest (1 cell)

def test_color_by_object_rank_empty():
    from domains.arc.transformation_primitives import color_by_object_rank
    grid = [[0, 0], [0, 0]]
    result = color_by_object_rank(grid)
    assert result == grid
