"""
Pinned / locked items: the game draws faint diagonal stripes ("/", 45 degrees) over the whole cell
background of an item the player pinned or locked (locked ones also get a padlock icon).  Nobody
wants advice on those, so the scan skips them before any identification work.

The test is a matched filter per cell: the high-passed, clipped cell is correlated with a 45-degree
sinusoid of the stripe period (a fixed share of the cell pitch, ~0.16).  Stripes give a strong,
one-sided response; an icon's edges respond weakly and about as much to the mirrored ("\\")
direction, which is the control.  One striped cell is enough (big cases cover their middle cells
with an opaque picture).  Measured on a 1440p-class stash screenshot: striped cells score
0.11-0.42, unmarked cells at most 0.08.
"""
from __future__ import annotations

import cv2
import numpy as np

PERIOD_SHARE = (0.135, 0.185)   # stripe period along x (or y) as a share of the cell pitch
PERIOD_STEPS = 9
CLIP = 8.0                      # grey levels: stripes are ~+-7, icon edges are clipped to this
MIN_SCORE = 0.09                # normalised response of the stripe direction
MIN_RATIO = 3.0                 # ... and that many times the mirrored direction's
INSET = 4 / 69                  # share of the pitch trimmed off each cell side (border line, glow)


def _cell_score(gray: np.ndarray, pitch: float) -> tuple[float, float]:
    """(stripe response, mirrored response) of one cell, both normalised by the cell's rms."""
    best = (0.0, 0.0)
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    anti, diag = xs + ys, xs - ys
    for share in np.linspace(*PERIOD_SHARE, PERIOD_STEPS):
        period = share * pitch
        hp = np.clip(gray - cv2.GaussianBlur(gray, (0, 0), period / 2), -CLIP, CLIP)
        rms = float(np.sqrt((hp * hp).mean())) + 1e-6
        k = -2j * np.pi / period
        a = abs((hp * np.exp(k * anti)).mean()) / rms
        if a > best[0]:
            best = (a, abs((hp * np.exp(k * diag)).mean()) / rms)
    return best


def is_pinned(img_bgr: np.ndarray, rect, w_cells: int, h_cells: int, pitch_x: float, pitch_y: float) -> bool:
    """True when the item at ``rect`` (x, y, width, height) carries the pinned / locked stripes."""
    x, y, rw, rh = (int(v) for v in rect)
    if rw <= 0 or rh <= 0 or w_cells <= 0 or h_cells <= 0:
        return False
    pitch = (pitch_x + pitch_y) / 2
    cw, ch = rw / w_cells, rh / h_cells
    inset = max(2, int(round(INSET * pitch)))
    H, W = img_bgr.shape[:2]
    for r in range(h_cells):
        for c in range(w_cells):
            x0, y0 = int(x + c * cw) + inset, int(y + r * ch) + inset
            x1, y1 = min(W, int(x + (c + 1) * cw) - inset), min(H, int(y + (r + 1) * ch) - inset)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            cell = cv2.cvtColor(img_bgr[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            a, d = _cell_score(cell, pitch)
            if a >= MIN_SCORE and a >= MIN_RATIO * d:
                return True
    return False
