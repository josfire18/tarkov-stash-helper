"""Segmentation: multi-cell items, identical adjacent stacks, rotation, empty cells, clipped rows."""
import cv2
import pytest

from identify.grid import detect_grid
from identify.segment import segment_panel, _otsu_1d
from synth import render_panel


def _segment(img):
    g = detect_grid(img)
    assert g.panels, 'grid not found'
    return g.panels[0], segment_panel(img, g.panels[0])


def _shapes(items):
    return {(i.col, i.row, i.w, i.h) for i in items if not i.empty}


ITEMS = [(0, 0, 3, 2), (3, 0, 1, 1), (4, 0, 1, 1), (5, 0, 2, 1), (3, 1, 1, 3), (4, 1, 4, 1),
         (0, 2, 1, 1), (1, 2, 2, 1), (4, 2, 2, 2)]


@pytest.mark.parametrize('pitch', [47.25, 63.0, 94.5])
def test_multi_cell_items_become_single_footprints(pitch):
    img, _, expected, empties = render_panel(pitch, 8, 5, ITEMS)
    panel, items = _segment(img)
    want = {cell for cell, _ in expected}
    assert _shapes(items) == want
    for it in items:
        if it.empty:
            assert (it.col, it.row) in empties


def test_rect_matches_geometry():
    img, _, expected, _ = render_panel(63.0, 8, 5, ITEMS)
    _, items = _segment(img)
    by_cell = {(i.col, i.row, i.w, i.h): i.rect for i in items}
    for cell, rect in expected:
        got = by_cell[cell]
        assert all(abs(a - b) <= 1 for a, b in zip(got, rect)), (cell, got, rect)


def test_identical_adjacent_stacks_stay_separate():
    # five identical 1x1 stacks in a row (the legacy engine merged identical neighbours)
    from synth import make_icon
    ic = make_icon(7, 1, 1)
    icons = {(c, 0): ic for c in range(5)}
    img, _, expected, _ = render_panel(63.0, 6, 2, [(c, 0, 1, 1) for c in range(5)], icons=icons)
    _, items = _segment(img)
    singles = [i for i in items if not i.empty and i.row == 0]
    assert len(singles) == 5 and all(i.w == i.h == 1 for i in singles)


def test_rotated_footprints_are_just_the_enclosed_shape():
    # a 2x1 and the same item rotated (1x2) side by side
    img, _, expected, _ = render_panel(63.0, 5, 3, [(0, 0, 2, 1), (3, 0, 1, 2)])
    _, items = _segment(img)
    assert {(0, 0, 2, 1), (3, 0, 1, 2)} <= _shapes(items)


def test_empty_cells_are_flagged():
    img, _, _, empties = render_panel(63.0, 6, 3, [(0, 0, 2, 2)])
    _, items = _segment(img)
    flagged = {(i.col, i.row) for i in items if i.empty}
    assert flagged == empties
    assert len(flagged) == 6 * 3 - 4 + 0 or len(flagged) == 14


def test_clipped_rows_are_marked_and_segmented():
    img, _, expected, _ = render_panel(63.0, 6, 5, [(0, 0, 1, 2), (1, 0, 2, 1), (3, 3, 1, 2)],
                                       top_partial=0.4, bottom_partial=0.3)
    panel, items = _segment(img)
    assert panel.clip_top and panel.clip_bottom
    assert all(i.clipped_top for i in items if i.row == 0)
    assert all(i.clipped_bottom for i in items if i.row + i.h == panel.n_rows)


def test_jpeg_does_not_merge_neighbours():
    img, _, expected, _ = render_panel(63.0, 8, 5, ITEMS)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _, items = _segment(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    want = {cell for cell, _ in expected}
    got = _shapes(items)
    assert want <= got                       # every real item is recovered intact
    # anything extra may only be (merged) empty cells, never a real item's cells
    real_cells = {(c + dc, r + dr) for c, r, w, h in want for dc in range(w) for dr in range(h)}
    for c, r, w, h in got - want:
        assert not ({(c + dc, r + dr) for dc in range(w) for dr in range(h)} & real_cells)


def test_otsu_splits_a_bimodal_distribution():
    import numpy as np
    vals = np.array([0.0, 0.05, 0.1, 0.1, 0.3, 0.95, 1.0, 1.0, 1.0, 0.98])
    t, j = _otsu_1d(vals, 0.5, 0.9)
    assert 0.5 <= t <= 0.9 and j > 0.8
