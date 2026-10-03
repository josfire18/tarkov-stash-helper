"""
Per-tile result cache for near-realtime re-scans.

A live viewer re-scans a view that is mostly unchanged (the next poll, a scroll, an opened
container).  Every item tile is keyed by a hash of its exact pixels (frame included, so the stack
count, FiR tick and label are part of the key) plus the pitch; a hit reuses the identification
and only the tile's position is refreshed.  Any pixel change - a different count, a hover
highlight, a partial redraw - is a miss and the tile is identified from scratch.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np


def tile_key(img_bgr: np.ndarray, rect: tuple, pitch_x: float, pitch_y: float, extra: str = '') -> str | None:
    x, y, w, h = rect
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_bgr.shape[1] or y + h > img_bgr.shape[0]:
        return None
    crop = np.ascontiguousarray(img_bgr[y:y + h, x:x + w])
    hs = hashlib.blake2b(crop.tobytes(), digest_size=16)
    hs.update(f'{w}x{h}|{pitch_x:.1f}|{pitch_y:.1f}|{extra}'.encode())   # pitch estimates jitter in the 3rd decimal
    return hs.hexdigest()


class TileCache:
    """Thread-safe LRU of ``key -> payload`` (the payload is whatever the engine stores)."""

    def __init__(self, maxsize: int = 20000):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key is None:
            return None
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key, value) -> None:
        if key is None:
            return
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._d)
