"""
Stack-count / durability reader (bottom-right digits, e.g. ``80``, ``3/3``, ``296/400``).

Tesseract is poor on this pixel font (it reads the slashed zero as 8: "80" -> "88";
measured 10/25 correct on labelled counts of stash1).  The font is fixed, so a tiny glyph
classifier does better: threshold the bottom-right band, split it into connected glyphs,
normalise each to 8x11 and match against prototypes (``assets/digits.npz``) by cosine
similarity.  The reader returns ``None`` unless every glyph is recognised confidently, so a
"count" is never a guess.

Prototypes are learned from labelled tiles with :func:`train` (``python -m identify.digits
<image> <counts.json>``): glyph crops are assigned by position to the characters of the
labelled string.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'digits.npz')
GW, GH = 8, 11
ACCEPT = 0.72
CHARS = '0123456789/'

_protos: dict | None = None


def _load() -> dict:
    global _protos
    if _protos is None:
        _protos = {}
        if os.path.exists(ASSET):
            z = np.load(ASSET, allow_pickle=False)
            for ch, v in zip(z['chars'], z['vecs']):
                _protos.setdefault(str(ch), []).append(v)
    return _protos


def glyphs(hi_tile: np.ndarray) -> list[np.ndarray]:
    """Glyph images (binary, GH x GW float) of the count band of a 64 px/slot tile,
    left to right; [] when no plausible text line is found."""
    h, w = hi_tile.shape[:2]
    if h < 30 or w < 40:
        return []
    band = hi_tile[h - 15:h - 1, max(0, w - 56):w - 1]
    mx = band.max(axis=2)
    if int(mx.max()) - int(mx.min()) < 60:
        return []
    _, m = cv2.threshold(mx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if 6 <= ch <= 13 and 1 <= cw <= 10 and area >= 5:
            comps.append((x, y, cw, ch, i))
    if not comps:
        return []
    comps.sort()
    # one text line: glyph baselines agree
    base = np.median([y + ch for _, y, _, ch, _ in comps])
    comps = [c for c in comps if abs((c[1] + c[3]) - base) <= 2]
    out = []
    for x, y, cw, ch, i in comps:
        g = (lab[y:y + ch, x:x + cw] == i).astype(np.float32)
        out.append(cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA))
    return out


def _classify(g: np.ndarray) -> tuple[str, float]:
    best, bs = '', -1.0
    v = g.reshape(-1) - g.mean()
    nv = np.linalg.norm(v) + 1e-6
    for ch, vecs in _load().items():
        for p in vecs:
            pv = p - p.mean()
            s = float(v @ pv / (nv * (np.linalg.norm(pv) + 1e-6)))
            if s > bs:
                best, bs = ch, s
    return best, bs


def read_count(hi_tile: np.ndarray | None) -> tuple[int | None, str]:
    """(count, raw text).  ``count`` is the first number ('296/400' -> 296) or None."""
    if hi_tile is None or not _load():
        return None, ''
    gl = glyphs(hi_tile)
    if not gl or len(gl) > 8:
        return None, ''
    chars = []
    for g in gl:
        ch, s = _classify(g)
        if s < ACCEPT:
            return None, ''
        chars.append(ch)
    text = ''.join(chars)
    digits = ''
    for ch in text:
        if ch == '/':
            break
        digits += ch
    if not digits or text.startswith('/'):
        return None, text
    return int(digits), text


def train(hi_tiles: list, labels: list[str], out: str = ASSET) -> dict:
    """Learn glyph prototypes: ``labels[i]`` is the exact text printed on ``hi_tiles[i]``."""
    chars, vecs, skipped = [], [], 0
    for tile, text in zip(hi_tiles, labels):
        gl = glyphs(tile)
        if len(gl) != len(text):
            skipped += 1
            continue
        for ch, g in zip(text, gl):
            chars.append(ch)
            vecs.append(g.reshape(-1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, chars=np.array(chars), vecs=np.array(vecs, np.float32).reshape(-1, GW * GH))
    global _protos
    _protos = None
    return {'glyphs': len(chars), 'skipped': skipped, 'classes': sorted(set(chars))}
