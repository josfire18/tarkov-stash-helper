"""
test_ux.py — offline, no-network tests for v0.3.0's region/scan UX (E):
  * app.py's _crop_calibration pixel-crop math
  * app.py's _calibration_crop_if_fresh TTL gate

Run with: python -m pytest test_ux.py -q
"""

import time

import numpy as np

import app as tsh_app


# ---------------------------------------------------------------------------
# _crop_calibration
# ---------------------------------------------------------------------------

def _numbered_frame(h=100, w=200):
    """A frame where every pixel encodes its own (row, col) — makes crop
    correctness trivially verifiable by reading the corner/edge values back."""
    frame = np.zeros((h, w, 3), dtype=np.uint16)
    rows = np.arange(h).reshape(h, 1)
    cols = np.arange(w).reshape(1, w)
    frame[:, :, 0] = rows
    frame[:, :, 1] = cols
    return frame.astype(np.uint8)  # BGR-shaped uint8 is fine for indexing checks


def test_crop_calibration_crops_expected_sub_array():
    frame = _numbered_frame(h=100, w=200)
    region = {'x': 20, 'y': 10, 'w': 50, 'h': 30}
    cropped = tsh_app._crop_calibration(frame, region)
    assert cropped is not None
    assert cropped.shape[:2] == (30, 50)
    # Top-left of the crop must equal frame[y, x]
    assert np.array_equal(cropped[0, 0], frame[10, 20])
    # Bottom-right corner of the crop (inclusive) must equal frame[y+h-1, x+w-1]
    assert np.array_equal(cropped[-1, -1], frame[39, 69])


def test_crop_calibration_none_frame_returns_none():
    assert tsh_app._crop_calibration(None, {'x': 0, 'y': 0, 'w': 10, 'h': 10}) is None


def test_crop_calibration_none_region_returns_none():
    frame = _numbered_frame()
    assert tsh_app._crop_calibration(frame, None) is None
    assert tsh_app._crop_calibration(frame, {}) is None


def test_crop_calibration_out_of_bounds_region_returns_none():
    frame = _numbered_frame(h=100, w=200)
    # Entirely past the frame's edge -> empty crop -> None
    region = {'x': 500, 'y': 500, 'w': 50, 'h': 50}
    assert tsh_app._crop_calibration(frame, region) is None


def test_crop_calibration_clamps_partially_out_of_bounds_region():
    frame = _numbered_frame(h=100, w=200)
    # Region hangs off the right/bottom edge — should clamp, not raise or wrap.
    region = {'x': 180, 'y': 90, 'w': 50, 'h': 50}
    cropped = tsh_app._crop_calibration(frame, region)
    assert cropped is not None
    assert cropped.shape[:2] == (10, 20)   # clamped to (100-90, 200-180)


# ---------------------------------------------------------------------------
# _calibration_crop_if_fresh — TTL gate
# ---------------------------------------------------------------------------

def test_calibration_crop_fresh_frame_is_used(monkeypatch):
    frame = _numbered_frame(h=50, w=50)
    tsh_app._last_calibration['bgr'] = frame
    tsh_app._last_calibration['ts'] = time.time()   # just captured
    settings = {'region': {'x': 5, 'y': 5, 'w': 10, 'h': 10}}
    try:
        result = tsh_app._calibration_crop_if_fresh(settings)
        assert result is not None
        assert result.shape[:2] == (10, 10)
    finally:
        tsh_app._last_calibration['bgr'] = None
        tsh_app._last_calibration['ts'] = 0


def test_calibration_crop_stale_frame_is_rejected():
    frame = _numbered_frame(h=50, w=50)
    tsh_app._last_calibration['bgr'] = frame
    # Older than CALIBRATION_TTL — must be treated as stale, not used.
    tsh_app._last_calibration['ts'] = time.time() - (tsh_app.CALIBRATION_TTL + 5)
    settings = {'region': {'x': 5, 'y': 5, 'w': 10, 'h': 10}}
    try:
        assert tsh_app._calibration_crop_if_fresh(settings) is None
    finally:
        tsh_app._last_calibration['bgr'] = None
        tsh_app._last_calibration['ts'] = 0


def test_calibration_crop_no_region_returns_none_even_if_fresh():
    frame = _numbered_frame(h=50, w=50)
    tsh_app._last_calibration['bgr'] = frame
    tsh_app._last_calibration['ts'] = time.time()
    try:
        assert tsh_app._calibration_crop_if_fresh({'region': None}) is None
        assert tsh_app._calibration_crop_if_fresh({}) is None
    finally:
        tsh_app._last_calibration['bgr'] = None
        tsh_app._last_calibration['ts'] = 0


# ---------------------------------------------------------------------------
# capture_for_scan — falls back to live grab (via capture_stash_image) and
# records a warning when the calibration frame is stale/missing.
# ---------------------------------------------------------------------------

def test_capture_for_scan_falls_back_and_warns_when_calibration_missing(monkeypatch):
    tsh_app._last_calibration['bgr'] = None
    tsh_app._last_calibration['ts'] = 0

    called = {}
    def fake_capture(settings, require_region=False):
        called['require_region'] = require_region
        return 'PIL_IMG', 'BGR_IMG'

    monkeypatch.setattr(tsh_app, 'capture_stash_image', fake_capture)
    warnings = []
    result = tsh_app.capture_for_scan({'region': None}, from_calibration=True,
                                       require_region=True, warnings=warnings)
    assert result == ('PIL_IMG', 'BGR_IMG')
    assert called['require_region'] is True
    assert len(warnings) == 1
    assert 'stale' in warnings[0] or 'unavailable' in warnings[0]


def test_capture_for_scan_uses_calibration_frame_without_warning():
    frame = _numbered_frame(h=50, w=50)
    tsh_app._last_calibration['bgr'] = frame
    tsh_app._last_calibration['ts'] = time.time()
    settings = {'region': {'x': 0, 'y': 0, 'w': 20, 'h': 20}}
    warnings = []
    try:
        img, img_bgr = tsh_app.capture_for_scan(settings, from_calibration=True,
                                                 require_region=True, warnings=warnings)
        assert img_bgr.shape[:2] == (20, 20)
        assert warnings == []
    finally:
        tsh_app._last_calibration['bgr'] = None
        tsh_app._last_calibration['ts'] = 0
