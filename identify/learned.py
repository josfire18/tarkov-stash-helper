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

    def get(self, key: str | None) -> str | None:
        e = self.data.get(key) if key else None
        return e['id'] if e else None

    def learn(self, key: str | None, item_id: str, name: str) -> bool:
        """Record ``key -> item_id``; returns True if this changed anything."""
        if not key or not item_id:
            return False
        with _lock:
            e = self.data.get(key)
            if e and e['id'] == item_id:
                e['seen'] = e.get('seen', 1) + 1
                changed = False
            else:
                self.data[key] = {'id': item_id, 'name': name, 'seen': 1, 'at': int(time.time())}
                changed = True
            self._save()
        return changed

    def _save(self) -> None:
        tmp = self.path + '.tmp'
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
