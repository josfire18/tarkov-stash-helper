"""
test_fir.py — offline, no-network, no-Tesseract tests for the Found-in-Raid
(FiR) checkmark detector (A3):
  * app.py's detect_fir three-valued corner-window classifier
  * app.py's get_protected_ids fir_only return shape

Run with: python -m pytest test_fir.py -q
"""

import numpy as np

import app as tsh_app


# ---------------------------------------------------------------------------
# detect_fir
# ---------------------------------------------------------------------------

# A simple 1-cell-pitch grid: cell (0,0)'s footprint right edge sits at x=63,
# top edge at y=0, so detect_fir's corner window is img[0:18, 45:63] (c=18,
# since round(18 * 63 / 63) == 18).
GRID = {'cell_w': 63, 'cell_h': 63, 'origin_x': 0, 'origin_y': 0}
BG = (40, 40, 42)          # dark, low-saturation stash background
BRIGHT = (200, 200, 200)   # bright, zero-saturation — what a white ✓ looks like


def _blank_frame(h=90, w=90):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = BG
    return frame


def test_detect_fir_true_for_bright_cluster_away_from_left_edge():
    frame = _blank_frame()
    # Window is img[0:18, 45:63]; paint a cluster inside it, away from column
    # 45 (the window's left edge) so it isn't mistaken for label overflow.
    frame[2:6, 50:55] = BRIGHT   # 4*5 = 20 px, within [FIR_MIN_PX, 0.5*18*18]
    assert tsh_app.detect_fir(frame, GRID, 0, 0, 1, 1) is True


def test_detect_fir_false_for_empty_window():
    frame = _blank_frame()   # all background, no bright pixels anywhere
    assert tsh_app.detect_fir(frame, GRID, 0, 0, 1, 1) is False


def test_detect_fir_none_when_bright_touches_left_edge():
    frame = _blank_frame()
    # Window's left edge is image column 45 (x2 - c == 63 - 18 == 45).
    # A bright pixel there reads as an item-name label overflowing the
    # corner, not a checkmark — must be indeterminate, not True/False.
    frame[5, 45] = BRIGHT
    frame[5, 51] = BRIGHT  # plus a small cluster elsewhere, so this isn't
                           # simply the n==0 path either
    assert tsh_app.detect_fir(frame, GRID, 0, 0, 1, 1) is None


def test_detect_fir_none_for_oversized_bright_area():
    frame = _blank_frame()
    # Fill the window almost entirely bright EXCEPT column 45 (the window's
    # left edge), so this exercises the "too much bright area" branch
    # specifically rather than tripping the left-edge guard first.
    frame[0:18, 46:63] = BRIGHT   # 18 rows * 17 cols = 306 px > 0.5*18*18=162
    assert tsh_app.detect_fir(frame, GRID, 0, 0, 1, 1) is None


def test_detect_fir_none_for_window_off_image():
    frame = _blank_frame()
    # A grid whose footprint's right edge is nowhere near the actual image —
    # the clipped window degenerates to empty/near-empty.
    far_grid = {'cell_w': 63, 'cell_h': 63, 'origin_x': 1000, 'origin_y': 0}
    assert tsh_app.detect_fir(frame, far_grid, 0, 0, 1, 1) is None


def test_detect_fir_none_for_low_count_below_min_px():
    frame = _blank_frame()
    # A handful of bright pixels below FIR_MIN_PX (6) — too little signal to
    # call it either way, away from the window's left edge.
    frame[2:4, 50:52] = BRIGHT   # 2*2 = 4 px < FIR_MIN_PX
    assert tsh_app.detect_fir(frame, GRID, 0, 0, 1, 1) is None


# ---------------------------------------------------------------------------
# get_protected_ids — fir_only shape
# ---------------------------------------------------------------------------

def test_get_protected_ids_fir_only_shape(monkeypatch):
    # Offline: no cached task data, so the aggregate (task/hideout) pass is
    # skipped entirely and only the keep-list pass runs.
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: None)

    keep_list = {
        'categories': [
            {'id': 'kappa', 'label': 'Kappa (Collector)', 'items': [
                {'id': 'kappa_item', 'name': 'Kappa Test Item',
                 'aliases': [], 'acquired': False},
            ]},
            {'id': 'tasks', 'label': 'Task Items (manual)', 'items': [
                {'id': 'task_item', 'name': 'Task Test Item',
                 'aliases': [], 'acquired': False},
            ]},
        ]
    }
    price_idx = {
        'kappa test item': {'id': 'tid_kappa', 'name': 'Kappa Test Item'},
        'task test item':  {'id': 'tid_task',  'name': 'Task Test Item'},
    }

    protected = tsh_app.get_protected_ids(keep_list, price_idx)

    assert protected['tid_kappa'] == {'reason': 'On keep list', 'fir_only': True}
    assert protected['tid_task']  == {'reason': 'On keep list', 'fir_only': False}


def test_get_protected_ids_skips_acquired_entries(monkeypatch):
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: None)

    keep_list = {
        'categories': [
            {'id': 'kappa', 'label': 'Kappa (Collector)', 'items': [
                {'id': 'kappa_done', 'name': 'Already Acquired',
                 'aliases': [], 'acquired': True},
            ]},
        ]
    }
    price_idx = {
        'already acquired': {'id': 'tid_done', 'name': 'Already Acquired'},
    }

    protected = tsh_app.get_protected_ids(keep_list, price_idx)
    assert 'tid_done' not in protected
