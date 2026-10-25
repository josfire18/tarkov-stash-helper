"""
Items already named on a normal stash view, recognised again on the trader screens.

The trader sell tab (and the flea offer windows) draw the stash differently: every item the trader
does not buy is dimmed, and the ones it does buy carry a price tag over the printed name.  Neither
the catalogue match nor the label read works well there, so most of those items used to land in
"Check" (and a dimmed armour plate could come back as a drum magazine).  The same items were named
confidently a moment earlier on the stash or character screen.

So every confidently named, uncut tile of a normal view leaves a small picture signature here: the
middle of the tile (the name / price-tag band at the top and the count / durability band at the
bottom are cut off), grey, high-passed and normalised, which makes it blind to the dimming.  On a
dimming screen an item the engine could not name confidently is compared with the signatures of the
same footprint; a near-identical picture of one single item names it.

Measured on a 1440p Therapist sell tab against the character screen seen just before it: the same
item scores 0.93-0.99, the closest different item (two Boreas keycards that differ only in a
printed letter pair) 0.875.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

RES = 24                 # signature pixels per cell
HP_SIGMA = 2.0           # high-pass (signature pixels): drops the dimming and the backdrop shading
TOP_BAND = 0.30          # share of a cell cut off the top (name label / price tag)
BOTTOM_BAND = 0.22       # ... and off the bottom (stack count, durability, FiR tick)
INSET = 4 / 84           # share of the pitch trimmed off the left / right (border line)
MIN_SCORE = 0.93         # correlation with the best signature
MIN_MARGIN = 0.05        # ... and this much above the best signature of another item
MAX_ENTRIES = 4000


def signature(img_bgr: np.ndarray, rect, w_cells: int, h_cells: int, pitch_x: float, pitch_y: float):
    """Unit-norm picture signature of the item at ``rect`` (x, y, width, height), or None."""
    x, y, rw, rh = (int(round(v)) for v in rect)
    if w_cells <= 0 or h_cells <= 0 or rw <= 0 or rh <= 0:
        return None
    top, bot = int(TOP_BAND * pitch_y), int(BOTTOM_BAND * pitch_y)
    ins = max(2, int(round(INSET * pitch_x)))
    H, W = img_bgr.shape[:2]
    x0, x1, y0, y1 = x + ins, x + rw - ins, y + top, y + rh - bot
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img_bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    out_h = int(round((h_cells - TOP_BAND - BOTTOM_BAND) * RES))
    g = cv2.resize(g.astype(np.float32), (w_cells * RES, out_h), interpolation=cv2.INTER_AREA)
    g = g - cv2.GaussianBlur(g, (0, 0), HP_SIGMA)
    g -= g.mean()
    n = float(np.linalg.norm(g))
    if n < 1e-3:
        return None
    return (g / n).ravel()


class SeenItems:
    """Thread-safe, bounded memory of ``signature -> identification`` per footprint."""

    def __init__(self, maxsize: int = MAX_ENTRIES):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()       # (w, h, item_id, rounded sig hash) -> (sig, payload)
        self._lock = threading.Lock()

    def add(self, sig, w_cells: int, h_cells: int, payload: dict) -> None:
        if sig is None or not payload.get('item_id'):
            return
        key = (w_cells, h_cells, payload['item_id'], hash(np.round(sig * 50).astype(np.int8).tobytes()))
        with self._lock:
            self._d[key] = (sig, payload)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def lookup(self, sig, w_cells: int, h_cells: int):
        """``(payload, score)`` of the one item whose stored picture matches ``sig``, else None."""
        if sig is None:
            return None
        with self._lock:
            rows = [(s, p) for (w, h, _, _), (s, p) in self._d.items()
                    if w == w_cells and h == h_cells and s.shape == sig.shape]
        if not rows:
            return None
        scores = np.stack([s for s, _ in rows]) @ sig
        order = np.argsort(-scores)
        best = int(order[0])
        top = float(scores[best])
        if top < MIN_SCORE:
            return None
        best_id = rows[best][1]['item_id']
        other = next((float(scores[i]) for i in order[1:] if rows[int(i)][1]['item_id'] != best_id), -1.0)
        if top - other < MIN_MARGIN:
            return None
        return rows[best][1], top

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)
