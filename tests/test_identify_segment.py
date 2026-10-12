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


# ---------------------------------------------------------------------------
# art that looks like a border line (large containers / backpacks / cloth)
# ---------------------------------------------------------------------------

import numpy as np

from synth import LINE, make_icon


def _paint(img, x0, y0, x1, y1, bgr):
    img[y0:y1, x0:x1] = bgr


def test_grey_band_inside_a_big_item_is_not_a_border():
    # a container lid: a thick band in exactly the border colour runs along the row boundary
    # of a 3x3 item (the Lucky Scav Junk box split into 3x1 + 3x2 because of it)
    img, _, _, _ = render_panel(63.0, 6, 5, [(1, 1, 3, 3)])
    panel, items = _segment(img)
    y = panel.ys[2]
    _paint(img, panel.xs[1] + 3, y - 4, panel.xs[4] - 2, y + 5, LINE)
    _, items = _segment(img)
    assert (1, 1, 3, 3) in _shapes(items)


def test_diagonal_hatch_in_border_colour_is_not_a_border():
    # the hatched background of an unexamined item: diagonal stripes of the border colour
    img, _, _, _ = render_panel(63.0, 6, 5, [(1, 1, 3, 3)])
    panel, _ = _segment(img)
    x0, x1, y0, y1 = panel.xs[1] + 2, panel.xs[4] - 1, panel.ys[1] + 2, panel.ys[4] - 1
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (xx + yy) % 4 < 2:
                img[yy, xx] = LINE
    _, items = _segment(img)
    assert (1, 1, 3, 3) in _shapes(items)


def test_partial_grey_cloth_across_a_cell_boundary_does_not_split_a_2x1():
    # Ripstop fabric: a grey patch in the border colour covering almost all of the shared edge
    img, _, _, _ = render_panel(63.0, 6, 3, [(1, 1, 2, 1)])
    panel, _ = _segment(img)
    x = panel.xs[2]
    _paint(img, x - 20, panel.ys[1] + 3, x + 20, panel.ys[2] - 2, LINE)
    _, items = _segment(img)
    assert (1, 1, 2, 1) in _shapes(items)


def test_green_cloth_edge_next_to_flat_margin_is_not_a_border():
    # a backpack icon: a flat dark margin column (looks like an empty slot) then green camo
    # starting exactly at the cell boundary (the step-against-empty test used to call that a line)
    img, _, _, _ = render_panel(63.0, 6, 5, [(1, 1, 3, 3)], icons={(1, 1): np.zeros((190, 190, 4), np.uint8)})
    panel, _ = _segment(img)
    x = panel.xs[2]
    camo = np.zeros((panel.ys[4] - panel.ys[1] - 3, panel.xs[4] - x - 1, 3), np.uint8)
    camo[:] = (66, 92, 76)
    camo[::3, ::2] = (58, 80, 64)
    img[panel.ys[1] + 2:panel.ys[4] - 1, x + 1:panel.xs[4]] = camo
    _, items = _segment(img)
    assert (1, 1, 3, 3) in _shapes(items)


def test_real_border_next_to_an_empty_slot_still_counts():
    # the other side of the step test: an item beside empty slots stays separate from them
    img, _, expected, _ = render_panel(63.0, 6, 3, [(0, 0, 2, 2), (2, 0, 1, 1)])
    _, items = _segment(img)
    assert {(0, 0, 2, 2), (2, 0, 1, 1)} <= _shapes(items)


def test_empty_rule_gradient_slot_vs_dark_item():
    """A smooth-gradient backdrop (std 8-11, like a tall scav rig slot) is empty; a dark item of the same
    std whose variation is structure, not a gradient, is not; a dimmed stash only empties on the flat backdrop."""
    import numpy as np
    from identify.segment import cell_is_empty
    h, w = 171, 87
    grad = np.tile(np.linspace(24, 55, h)[:, None], (1, w)).astype(np.float32)        # std ~ 11 over the cell
    img = np.repeat(np.clip(grad, 0, 255).astype(np.uint8)[:, :, None], 3, 2)
    img = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode='edge')
    assert img[4:-4, 4:-4].std() > 8 and cell_is_empty(img, (0, 0, w + 8, h + 8))
    rng = np.random.RandomState(1)
    art = np.full((h + 8, w + 8), 25, np.float32)
    art[30:60, 20:70] = 55                                                            # a dark blob: structure
    art += rng.normal(0, 1.0, art.shape)
    img2 = np.repeat(np.clip(art, 0, 255).astype(np.uint8)[:, :, None], 3, 2)
    assert 8 < img2[4:-4, 4:-4].std() < 11 and not cell_is_empty(img2, (0, 0, w + 8, h + 8))
    flat = np.full((90, 90, 3), 22, np.uint8)
    flat[::2] += 3
    assert cell_is_empty(flat, (0, 0, 90, 90), dim=True)
    dimmed_item = np.full((90, 90, 3), 25, np.uint8)
    dimmed_item[10:80, 10:40] = 36                                                    # dimmed art, std ~7
    assert cell_is_empty(dimmed_item, (0, 0, 90, 90)) and not cell_is_empty(dimmed_item, (0, 0, 90, 90), dim=True)
