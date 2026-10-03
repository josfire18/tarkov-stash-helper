"""Names for the game's cached icons, learned from confirmed scans.

EFT's local icon cache holds pixel-exact renders of every item the game has
shown, but under recycled file numbers with no item id, so the catalog can only
guess their names by comparing them to tarkov.dev's pictures (and a recycled
number makes an old guess wrong).  When a scanned tile is a near-literal pixel
match to a cached icon *and* its printed label reads an exact, unambiguous short
name, that pairing is certain; it is recorded here keyed by the icon's pixel
content, so it survives file-number recycling and overrides the guess on every
later scan.  The store is a small JSON file next to the other per-user data.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import cv2

_lock = threading.Lock()
_hash_cache: dict = {}


def content_key(path: str) -> str | None:
    """Stable key for an icon file: SHA-1 of its decoded pixels (not its bytes or
    name, which change when the game re-encodes or renumbers the cache)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    ck = (path, st.st_mtime_ns, st.st_size)
    if ck in _hash_cache:
        return _hash_cache[ck]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    key = None if img is None else hashlib.sha1(img.tobytes() + bytes(str(img.shape), 'ascii')).hexdigest()
    _hash_cache[ck] = key
    return key


class LearnedNames:
    """``{content_key: {"id", "name", "seen", "at"}}`` persisted as JSON."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        try:
            with open(path, encoding='utf-8') as fh:
                self.data = json.load(fh)
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str | None, min_seen: int = 1) -> str | None:
        e = self.data.get(key) if key else None
        return e['id'] if e and e.get('seen', 1) >= min_seen else None

    def learn(self, key: str | None, item_id: str, name: str) -> int:
        """Record one confirmation of ``key -> item_id``; returns how many times this
        exact pairing has now been confirmed (a different item restarts the count)."""
        if not key or not item_id:
            return 0
        with _lock:
            e = self.data.get(key)
            if e and e.get('certain'):
                return e['seen'] if e['id'] == item_id else 0      # certain bindings are final
            if e and e['id'] == item_id:
                e['seen'] = e.get('seen', 1) + 1
            else:
                e = self.data[key] = {'id': item_id, 'name': name, 'seen': 1, 'at': int(time.time())}
            self._save()
            return e['seen']

    def bind_certain(self, key: str | None, item_id: str, name: str, confirmations: int = 2) -> None:
        """Permanent binding from a *certain* identification (an exact game-font label match on a
        tile that exactly matches this render): usable at once, and never overwritten by an
        ordinary read."""
        if not key or not item_id:
            return
        with _lock:
            e = self.data.get(key)
            if e and e.get('certain') and e['id'] != item_id:
                return                               # two certain sources disagree: keep the first
            self.data[key] = {'id': item_id, 'name': name, 'seen': max(confirmations, (e or {}).get('seen', 0)),
                              'certain': True, 'at': int(time.time())}
            self._save()

    def _save(self) -> None:
        tmp = self.path + '.tmp'
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
