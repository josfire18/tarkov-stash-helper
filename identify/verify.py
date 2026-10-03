"""
Difference-focused pair verification for label twins.

When the printed label reads equally well for two items (colour variants of one helmet, loose
rounds of different calibres that share "FMJ", a gun and its part) the whole-tile residual and
the DINO cosine separate them by a hair.  This answers the narrower question "which of these
two pictures is this tile?" by looking only where the two pictures disagree:

    D      = pixels where composite(A) and composite(B) differ by more than DIFF_MIN levels
             (label band, bottom band and border excluded - the game draws text there)
    err_X  = mean |tile - composite(X)| over D, after the best +-SHIFT px alignment of X

Both composites use the tile's own background colour and the comparison runs at the catalog's
63 px/slot.  ``ratio = (err_a + 1) / (err_b + 1)``: 1.0 means no evidence, < 1 favours A.
Measured on the labelled screenshots (docs/accuracy.md): reliable between *variants of one
family*, NOT as a general verifier (a tarkov.dev picture often differs from the game's render).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SLOT
from .masks import BOTTOM_PX, LABEL_PX

DIFF_MIN = 28.0       # mean |A-B| over channels for a pixel to count as "where they differ"
MIN_PIXELS = 40       # fewer differing pixels than this = the two pictures are the same
SHIFT = 2             # alignment search (px at 63 px/slot)


@dataclass
class PairResult:
    err_a: float
    err_b: float
    n_diff: int

    @property
    def ratio(self) -> float:
        if self.n_diff < MIN_PIXELS:
            return 1.0
        return (self.err_a + 1.0) / (self.err_b + 1.0)


def _composite(bgra: np.ndarray, bg: np.ndarray, rot: int) -> np.ndarray:
    if rot:
        bgra = np.rot90(bgra, rot)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return bgra[:, :, :3].astype(np.float32) * a + (1.0 - a) * bg[None, None, :]


def _shifted_err(tile: np.ndarray, comp: np.ndarray, m: np.ndarray) -> float:
    h, w = m.shape
    s = SHIFT
    tm = tile[s:h - s, s:w - s]
    mm = m[s:h - s, s:w - s]
    if not mm.any():
        return 0.0
    best = 1e9
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            c = comp[s + dy:h - s + dy, s + dx:w - s + dx]
            best = min(best, float(np.abs(tm - c).mean(axis=2)[mm].mean()))
    return best


def tile_hi(img_bgr: np.ndarray, rect: tuple, W: int, H: int) -> np.ndarray | None:
    """The tile at the catalog's full resolution ((63W+1) x (63H+1)), float32."""
    x, y, w, h = rect
    if x < 0 or y < 0 or x + w > img_bgr.shape[1] or y + h > img_bgr.shape[0] or w < 8 or h < 8:
        return None
    return cv2.resize(img_bgr[y:y + h, x:x + w], (SLOT * W + 1, SLOT * H + 1),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def pair(tile: np.ndarray, icon_a: np.ndarray, rot_a: int, icon_b: np.ndarray, rot_b: int,
         bg) -> PairResult | None:
    """Compare a full-resolution tile with two BGRA icons (see module docstring)."""
    if icon_a is None or icon_b is None or tile is None:
        return None
    bg = np.asarray(bg, np.float32)
    ca, cb = _composite(icon_a, bg, rot_a), _composite(icon_b, bg, rot_b)
    if ca.shape != tile.shape or cb.shape != tile.shape:
        return None
    h, w = tile.shape[:2]
    m = np.abs(ca - cb).mean(axis=2) > DIFF_MIN
    top, bot = int(round(LABEL_PX)), int(round(BOTTOM_PX))
    m[:top + 1] = False
    m[h - bot - 1:] = False
    m[:, :3] = False
    m[:, w - 3:] = False
    n = int(m.sum())
    if n < MIN_PIXELS:
        return PairResult(0.0, 0.0, n)
    return PairResult(_shifted_err(tile, ca, m), _shifted_err(tile, cb, m), n)
