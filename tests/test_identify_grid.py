"""Grid detection on synthetic stashes: scales, partial rows, stretch, chrome, two panels."""
import numpy as np
import pytest

from identify.grid import detect_grid, expected_pitch, line_mask, ridge_masks
from synth import render_panel

ITEMS = [(0, 0, 2, 2), (3, 0, 1, 3), (4, 1, 3, 1), (0, 3, 1, 1), (2, 3, 1, 1), (5, 3, 2, 2)]


@pytest.mark.parametrize('pitch', [31.5, 47.25, 63.0, 94.5, 126.0])
def test_pitch_and_dimensions_at_several_ui_scales(pitch):
    img, rect, _, _ = render_panel(pitch, 8, 6, ITEMS)
    g = detect_grid(img)
    assert len(g.panels) == 1
    p = g.panels[0]
    assert abs(p.pitch_x - pitch) / pitch < 0.01
    assert abs(p.pitch_y - pitch) / pitch < 0.01
    assert (p.n_cols, p.n_rows) == (8, 6)
    assert not p.clip_top and not p.clip_bottom
    assert abs(p.x0 - rect[0]) <= 1 and abs(p.y0 - rect[1]) <= 1


@pytest.mark.parametrize('pitch', [47.25, 63.0, 94.5])
def test_partial_top_row_is_found(pitch):
    # the first row is 40 % hidden by the viewport: it must still be a row (the legacy detector lost it)
    img, rect, _, _ = render_panel(pitch, 8, 7, ITEMS, top_partial=0.4)
    p = detect_grid(img).panels[0]
    assert p.n_rows == 7
    assert p.clip_top and not p.clip_bottom
    assert abs(p.y0 - rect[1]) <= 2
    # the cut row is shorter than a full row
    assert p.ys[1] - p.ys[0] < 0.75 * pitch


@pytest.mark.parametrize('pitch', [63.0, 94.5])
def test_partial_bottom_row_is_found(pitch):
    img, _, _, _ = render_panel(pitch, 8, 7, ITEMS, bottom_partial=0.35)
    p = detect_grid(img).panels[0]
    assert p.n_rows == 7 and p.clip_bottom and not p.clip_top


def test_both_edges_partial():
    img, _, _, _ = render_panel(63.0, 8, 8, ITEMS, top_partial=0.5, bottom_partial=0.5)
    p = detect_grid(img).panels[0]
    assert p.n_rows == 8 and p.clip_top and p.clip_bottom


def test_stretched_capture_has_independent_axes():
    img, _, _, _ = render_panel(84.0, 8, 6, ITEMS, pitch_y=63.0)      # 4:3 stretched to 16:9
    g = detect_grid(img)
    p = g.panels[0]
    assert abs(p.pitch_x - 84.0) < 1.0 and abs(p.pitch_y - 63.0) < 1.0
    assert any('anisotropic' in w for w in g.warnings)


def test_ui_chrome_does_not_add_rows():
    img, _, _, _ = render_panel(63.0, 8, 6, ITEMS, chrome=True, origin=(40, 70))
    p = detect_grid(img).panels[0]
    assert p.n_rows == 6


def test_two_panels_side_by_side_share_one_pitch():
    a, _, _, _ = render_panel(63.0, 6, 5, ITEMS[:4], origin=(20, 20), size=(900, 420))
    b, _, _, _ = render_panel(63.0, 3, 3, [(0, 0, 2, 2)], origin=(520, 70), size=(900, 420))
    # composite: panel b drawn on top of the right part of a
    img = a.copy()
    img[:, 480:] = b[:, 480:]
    g = detect_grid(img)
    assert len(g.panels) == 2
    assert abs(g.panels[0].pitch_x - g.panels[1].pitch_x) < 0.5
    assert g.panels[1].x0 > g.panels[0].x1 - 5


def test_screen_scale_hint_cross_check():
    assert expected_pitch(1920, 1080) == pytest.approx(63.0)
    assert expected_pitch(2560, 1440) == pytest.approx(84.0)
    assert expected_pitch(3840, 2160) == pytest.approx(126.0)
    assert expected_pitch(3440, 1440) == pytest.approx(84.0)      # ultrawide scales by height
    img, _, _, _ = render_panel(63.0, 8, 6, ITEMS)
    g = detect_grid(img, pitch_hint=expected_pitch(2560, 1440))   # wrong hint: warn, keep measuring
    assert abs(g.panels[0].pitch_x - 63.0) < 1.0
    assert any('differs from screen-implied' in w for w in g.warnings)


def test_jpeg_compressed_stash_is_still_a_grid():
    import cv2
    img, _, _, _ = render_panel(63.0, 8, 6, ITEMS)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    p = detect_grid(cv2.imdecode(buf, cv2.IMREAD_COLOR)).panels[0]
    assert (p.n_cols, p.n_rows) == (8, 6)


def test_line_mask_accepts_the_top_gradient_brightened_line():
    px = np.array([[[73, 81, 84]]], np.uint8)[:, :, ::-1]            # nominal BGR (84, 81, 73) reversed back
    nominal = np.array([[(84, 81, 73)]], np.uint8).reshape(1, 1, 3)
    bright = np.array([[(107, 94, 84)]], np.uint8).reshape(1, 1, 3)   # measured under the UI gradient
    art = np.array([[(30, 140, 60)]], np.uint8).reshape(1, 1, 3)
    assert line_mask(nominal)[0, 0] and line_mask(bright)[0, 0] and not line_mask(art)[0, 0]


def test_ridge_masks_see_a_blurred_line():
    import cv2
    img = np.full((60, 60, 3), 30, np.uint8)
    img[:, 30] = (90, 88, 84)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    _, vertical = ridge_masks(img, contrast=7)
    assert vertical[:, 29:32].any(axis=1).mean() > 0.9


def test_sub_grids_a_few_px_apart_are_both_found():
    # a backpack drawn as a 4-column grid, a 12 px gap and a 2-column grid whose lattice is offset
    # (the Terraframe pockets).  The two are one line mesh (the top/bottom borders run across
    # the gap); the lattice fit alone keeps only the bigger half.
    left, _, _, _ = render_panel(63.0, 4, 5, [(0, 0, 2, 2), (2, 1, 1, 3)], origin=(20, 20), chrome=False,
                                 size=(520, 400))
    right, _, _, _ = render_panel(63.0, 2, 5, [(0, 0, 2, 3), (0, 3, 1, 1)], origin=(20 + 4 * 63 + 12, 20),
                                  chrome=False, size=(520, 400))
    img = left.copy()
    x0 = 20 + 4 * 63 + 12 - 2
    img[:, x0:] = right[:, x0:]
    y_top, y_bot = 20, 20 + 5 * 63
    for y in (y_top, y_bot):
        img[y, 20 + 4 * 63:20 + 4 * 63 + 13] = (84, 81, 73)
    g = detect_grid(img)
    spans = sorted((p.x0, p.x1, p.n_cols, p.n_rows) for p in g.panels)
    assert len(g.panels) == 2, spans
    assert [s[2] for s in spans] == [4, 2] and all(s[3] == 5 for s in spans)
    assert abs(spans[1][0] - (20 + 4 * 63 + 12)) <= 1
