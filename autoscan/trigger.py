"""
When to scan: stability + "same view as last scan" + debounce, as a pure state machine.

Inputs are plain observations (no clock, no capture, no Windows), so the whole policy is
unit-testable.  One call to :meth:`Trigger.observe` per poll returns what the service should do.

Policy
------
* inventory on screen but not yet *stable* (two consecutive polls whose thumbnails are
  near-identical)                                  -> ``settling`` (scrolling lands here, so a
  scroll produces exactly one scan, at the resting position)
* stable, and the view differs from the last scan's -> ``scan``
* stable and the same view as the last scan         -> ``scanned`` (no work)
* "the same view as the last scan" means the dHash agrees AND at most ``CHANGE_FRAC`` of the
  thumbnail pixels differ from the scanned one: one moved item / opened container / scroll step
  is a different view even though most of the screen is unchanged.  Such a view is scanned as
  soon as it is stable (two polls, ~0.5 s) - the identify pipeline re-reads only the tiles whose
  pixels changed (identify.tilecache), so a re-scan is cheap
* a scan is never started within ``min_gap_s`` (0.5 s) of the previous one - only a guard against
  thrash on animated backdrops; the old 3 s gap made the live view lag behind the player
* a failed scan (state ``failed``) is not retried until the view changes or ``retry_s`` has passed
* inventory seen without the main-menu bar (the in-raid inventory) -> ``raid_inventory``: never
  scanned unless ``allow_raid`` is set (Settings > Live > "Work in raid")
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

THUMB = (160, 90)
STABLE_DIFF = 16            # per-pixel level change that counts as "changed"
STABLE_FRAC = 0.004         # <= 0.4 % of thumbnail pixels changed between polls -> stable
HASH_BITS = (32, 32)        # dHash grid
SAME_VIEW_BITS = 0.07       # <= 7 % of the hash bits differ -> same view as the last scan
CHANGE_FRAC = 0.0015        # ... and <= 0.15 % of the thumbnail pixels (~22 of 14400: one cell) differ


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Small grayscale copy for stability / hashing (decimate first: cheap on a 4 Mpx frame)."""
    h, w = frame.shape[:2]
    s = max(1, h // 360)
    g = cv2.cvtColor(np.ascontiguousarray(frame[::s, ::s]), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA)


def is_stable(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Two consecutive thumbnails are near-identical (cursor blink / jitter allowed)."""
    if a is None or b is None or a.shape != b.shape:
        return False
    changed = np.count_nonzero(cv2.absdiff(a, b) > STABLE_DIFF)
    return changed <= STABLE_FRAC * a.size


def view_hash(thumb: np.ndarray) -> np.ndarray:
    """Difference hash (bool vector) of a thumbnail: robust to brightness, exact enough to tell
    one scroll position / stash tab from another."""
    g = cv2.resize(thumb, (HASH_BITS[0] + 1, HASH_BITS[1]), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).ravel()


def changed_fraction(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Share of thumbnail pixels that differ clearly between two thumbnails (1.0 = unknown)."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return np.count_nonzero(cv2.absdiff(a, b) > STABLE_DIFF) / a.size


def same_view(h1: np.ndarray | None, h2: np.ndarray | None) -> bool:
    if h1 is None or h2 is None or h1.shape != h2.shape:
        return False
    return np.count_nonzero(h1 != h2) <= SAME_VIEW_BITS * h1.size


@dataclass
class Observation:
    inventory: bool
    menu_chrome: bool = True
    thumb: np.ndarray | None = None
    blank: bool = False


@dataclass
class Decision:
    state: str                  # waiting | raid_inventory | settling | scan | scanned | failed | blank
    scan: bool = False


class Trigger:
    def __init__(self, min_gap_s: float = 0.5, retry_s: float = 30.0, allow_raid: bool = False):
        self.min_gap_s = min_gap_s
        self.retry_s = retry_s
        self.allow_raid = allow_raid
        self._prev_thumb: np.ndarray | None = None
        self._last_hash: np.ndarray | None = None      # view of the last scan attempt
        self._last_thumb: np.ndarray | None = None
        self._last_ts = -1e9
        self._last_ok = True

    def reset_view(self):
        """Forget the last scanned view (e.g. a manual rescan wants the next frame scanned)."""
        self._last_hash = None
        self._last_thumb = None

    def scan_started(self, thumb: np.ndarray, now: float):
        self._last_hash, self._last_ts = view_hash(thumb), now
        self._last_thumb = thumb
        self._last_ok = True

    def scan_failed(self, now: float):
        self._last_ok = False
        self._last_ts = now

    def observe(self, obs: Observation, now: float) -> Decision:
        prev, self._prev_thumb = self._prev_thumb, obs.thumb
        if obs.blank:
            return Decision('blank')
        if not obs.inventory:
            return Decision('waiting')
        if not obs.menu_chrome and not self.allow_raid:
            return Decision('raid_inventory')
        if not is_stable(prev, obs.thumb):
            return Decision('settling')
        if (same_view(self._last_hash, view_hash(obs.thumb))
                and changed_fraction(self._last_thumb, obs.thumb) <= CHANGE_FRAC):
            if self._last_ok:
                return Decision('scanned')
            if now - self._last_ts < self.retry_s:
                return Decision('failed')
        if now - self._last_ts < self.min_gap_s:
            return Decision('settling')
        return Decision('scan', scan=True)
