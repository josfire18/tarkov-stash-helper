"""
Found-in-Raid check mark detection (three-valued, ported from ``app.detect_fir``).

Semantics are unchanged and load-bearing: ``True`` = confidently FiR, ``False`` =
confidently not FiR, ``None`` = indeterminate - callers must never treat ``None``
as "not FiR".

What changed is *where and how* the mark is read.  The legacy code thresholds the
top-right corner of the footprint, which in the current client is where the item's
name is printed (``data/eval/fir_tiles`` shows nothing but label glyphs), so it can
neither find the mark nor tell "no mark" from "label overflow".  In the current UI
the FiR mark is a small tick-in-a-circle at the footprint's **bottom right**, and it
moves up ~14 px when a stack count / durability text occupies the very corner.  So
the detector looks for the circle with normalised cross-correlation against a
13x10 grey template (cut from ``stash1.png``) over the right-hand edge strip of the
tile, after the tile has been rescaled to 64 px/slot.
"""
from __future__ import annotations

import cv2
import numpy as np

_T = np.array([
    [25, 20, 30, 57, 115, 135, 138, 101, 53, 25, 52, 65, 27],
    [20, 29, 87, 154, 109, 79, 76, 126, 145, 87, 140, 155, 20],
    [26, 59, 164, 66, 31, 21, 27, 22, 107, 165, 164, 76, 27],
    [19, 136, 104, 44, 63, 31, 20, 46, 131, 168, 111, 27, 20],
    [26, 151, 66, 109, 169, 73, 48, 135, 161, 133, 135, 20, 26],
    [20, 158, 64, 56, 138, 167, 144, 161, 67, 101, 136, 27, 20],
    [27, 126, 124, 25, 48, 134, 160, 73, 34, 143, 107, 20, 26],
    [20, 67, 157, 84, 30, 45, 64, 39, 100, 160, 43, 27, 20],
    [26, 20, 94, 152, 129, 85, 93, 132, 156, 69, 26, 20, 26],
    [20, 26, 20, 67, 116, 143, 135, 114, 50, 26, 20, 26, 20],
], np.float32)

FIR_YES = 0.62
FIR_NO = 0.38
SLOT_HI = 64


def detect_fir(tile_hi_bgr: np.ndarray | None) -> bool | None:
    """Three-valued FiR read of a tile that was rescaled to 64 px/slot.

    ``None`` when the tile is missing / too small / clipped, or when the
    correlation is in the grey zone between the two thresholds."""
    if tile_hi_bgr is None:
        return None
    h, w = tile_hi_bgr.shape[:2]
    if h < 28 or w < 40:
        return None
    g = cv2.cvtColor(tile_hi_bgr, cv2.COLOR_BGR2GRAY)
    # right-hand strip, bottom ~40 px (mark sits at the corner or one row above)
    strip = g[max(0, h - 40):h, max(0, w - 26):w]
    if strip.shape[0] < _T.shape[0] or strip.shape[1] < _T.shape[1]:
        return None
    # the pattern is bright-on-dark: flat strips (no bright pixels) are decisively "no"
    if strip.max() < 110:
        return False
    res = cv2.matchTemplate(strip.astype(np.float32), _T, cv2.TM_CCOEFF_NORMED)
    best = float(res.max())
    if best >= FIR_YES:
        return True
    if best <= FIR_NO:
        return False
    return None
