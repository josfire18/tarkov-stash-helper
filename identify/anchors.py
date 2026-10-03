"""
Anchor 1: certainty from a *pixel-exact* match against the game's own render.

EFT's local icon cache (``%LOCALAPPDATA%\\Temp\\Battlestate Games\\EscapeFromTarkov\\Icon Cache\\live``)
holds the textures the game itself rendered for every item it has shown.  The stash tile draws
exactly that texture (at 63 px/slot 1:1, at other UI scales resampled by the GPU), so a tile and
its own render agree to within compression noise over every opaque icon pixel, while the best
render of any *other* item is off by a wide margin (measured in ``docs/accuracy.md``).

The comparison (:func:`exact_residual`):

* only pixels where the render is fully opaque count (no dependence on the rarity tint or the
  cell background), minus the overlay bands (label strip on top, count / FiR / calibre at the
  bottom) and the frame line;
* the render is resampled to the measured pitch with the GPU's own filter (bilinear) and
  aligned by a sub-pixel shift search (the screen phase of a panel is not an integer);
* the score is the mean absolute level difference over those pixels (0-255 scale).

A tile is *exact* with a render when that score is below :data:`EXACT_MAX` (per pitch band) and
there is enough opaque area to say so.  It is CERTAIN when the exact render's item is known and
no render of a different item is also exact (see :meth:`CacheIndex.certify`).

Render identities come from (strongest first) the learned-binding store (a render that was seen
under an exact, closed-set label match - Anchor 2 - see :mod:`identify.learned`) and the catalog's
picture association to tarkov.dev icons (``catalog.meta['cache_assoc']``).
"""
from __future__ import annotations

import glob
import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SLOT
from .masks import BOTTOM_PX, LABEL_PX

THUMB = 16                     # px/slot of the coarse shortlist stacks
SHORTLIST = 10                 # renders checked exactly per tile
OPAQUE = 250                   # render alpha (0-255) that counts as fully opaque
MIN_PIXELS = 250               # opaque pixels needed (at 63 px/slot) to call a match exact
EXACT_MAX = {63: 3.0, 0: 4.5}  # mean abs level error of an exact match: native pitch / resampled
WRONG_MIN_RATIO = 2.5          # a different item's render must be this x worse than the exact one
# A render's identity from the catalog's picture association (render vs tarkov.dev icon) is only
# trusted when the tarkov.dev icon is (nearly) the same pixels and nothing else comes close:
# tarkov.dev art can differ from the game's render while another item's art happens to look like
# it (the Secure Flash drive's render associates to "Flash drive with Mr. Kerman's hash codes" at
# distance 4.2 vs 15.6).  Everything else is named only by Anchor 2 (learned bindings).
ASSOC_TRUST_D = 1.5
ASSOC_TRUST_RATIO = 3.0


@dataclass
class Render:
    fname: str
    path: str
    W: int
    H: int
    key: str
    item_id: str = ''
    id_src: str = ''           # 'learned' | 'assoc' | ''
    hint_id: str = ''          # untrusted picture association (never certifies)


def _rot(img: np.ndarray, rot: int) -> np.ndarray:
    return img if rot == 0 else np.ascontiguousarray(np.rot90(img, rot))


def footprint(img: np.ndarray) -> tuple[int, int] | None:
    h, w = img.shape[:2]
    W, H = max(1, int(round((w - 1) / SLOT))), max(1, int(round((h - 1) / SLOT)))
    if abs(w - (SLOT * W + 1)) > 3 or abs(h - (SLOT * H + 1)) > 3:
        return None
    return W, H


def content_key_of(img: np.ndarray) -> str:
    """Same key as :func:`identify.learned.content_key` (SHA-1 of the decoded pixels)."""
    return hashlib.sha1(img.tobytes() + bytes(str(img.shape), 'ascii')).hexdigest()


def band_mask(h: int, w: int, pitch: float, H: int) -> np.ndarray:
    """bool [h, w]: pixels of a tile outside the overlay bands and the frame line."""
    m = np.ones((h, w), bool)
    top = int(np.ceil((LABEL_PX + 1) * pitch / SLOT)) + 1
    bot = int(np.ceil((BOTTOM_PX + 1) * pitch / SLOT)) + 1
    b = max(2, int(np.ceil(2 * pitch / SLOT)))
    m[:top] = False
    m[max(0, h - bot):] = False
    m[:, :b] = False
    m[:, w - b:] = False
    return m


def warp_render(bgra: np.ndarray, out_w: int, out_h: int, scale_x: float, scale_y: float,
                dx: float = 0.0, dy: float = 0.0, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resample a render to the screen: pixel centres map as ``x_d + .5 = s (x_s + .5) + dx``."""
    if scale_x == 1.0 and scale_y == 1.0 and dx == int(dx) and dy == int(dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(bgra, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    M = np.float32([[scale_x, 0, 0.5 * scale_x - 0.5 + dx], [0, scale_y, 0.5 * scale_y - 0.5 + dy]])
    return cv2.warpAffine(bgra, M, (out_w, out_h), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _score(tile: np.ndarray, rend: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    m = mask & (rend[:, :, 3] >= OPAQUE)
    n = int(m.sum())
    if n == 0:
        return 999.0, 0
    d = np.abs(tile[m].astype(np.int16) - rend[:, :, :3][m].astype(np.int16))
    return float(d.mean()), n


def exact_residual(tile: np.ndarray, bgra: np.ndarray, pitch_x: float, pitch_y: float, H: int,
                   shifts: float = 2.0, interp: int = cv2.INTER_LINEAR, phase=None):
    """Best aligned residual of a tile crop (BGR, frame included) against a render (BGRA at
    63 px/slot, already in the tile's orientation).  Returns ``(score, n_px, (dx, dy))``.

    ``phase`` = a known (dx, dy) to try first (one per panel); the search then stays near it."""
    h, w = tile.shape[:2]
    sx = (w - 1) / max(1, bgra.shape[1] - 1) if abs(pitch_x - SLOT) > 0.5 else 1.0
    sy = (h - 1) / max(1, bgra.shape[0] - 1) if abs(pitch_y - SLOT) > 0.5 else 1.0
    mask = band_mask(h, w, pitch_y, H)
    best = (999.0, 0, (0.0, 0.0))
    native = sx == 1.0 and sy == 1.0
    if native:
        grid = [(dx, dy) for dy in range(-int(shifts), int(shifts) + 1) for dx in range(-int(shifts), int(shifts) + 1)]
        if phase is not None:
            grid = [phase] + grid
        for dx, dy in grid:
            s, n = _score(tile, warp_render(bgra, w, h, 1.0, 1.0, dx, dy), mask)
            if s < best[0]:
                best = (s, n, (dx, dy))
        return best
    # resampled: coarse 0.5 px grid, then 0.125 px refinement around the best
    if phase is None:
        steps = np.arange(-shifts, shifts + 1e-6, 0.5)
        grid = [(dx, dy) for dy in steps for dx in steps]
    else:
        grid = [(phase[0] + a, phase[1] + b) for a in (-0.5, 0, 0.5) for b in (-0.5, 0, 0.5)]
    for dx, dy in grid:
        s, n = _score(tile, warp_render(bgra, w, h, sx, sy, dx, dy, interp), mask)
        if s < best[0]:
            best = (s, n, (dx, dy))
    for step in (0.25, 0.125):
        cx, cy = best[2]
        for a in (-step, 0, step):
            for b in (-step, 0, step):
                if a == 0 and b == 0:
                    continue
                s, n = _score(tile, warp_render(bgra, w, h, sx, sy, cx + a, cy + b, interp), mask)
                if s < best[0]:
                    best = (s, n, (cx + a, cy + b))
    return best


def exact_threshold(pitch: float) -> float:
    return EXACT_MAX[63] if abs(pitch - SLOT) <= 0.5 else EXACT_MAX[0]


@dataclass
class AnchorResult:
    item_id: str                     # '' when the exact render's item is unknown
    certain: bool
    score: float                     # residual of the best render
    runner: float                    # best residual of a render of a different (or unknown) item
    render: Render | None = None
    rot: int = 0
    phase: tuple = (0.0, 0.0)
    n_px: int = 0
    note: str = ''
    ranked: list = field(default_factory=list)   # [(score, Render, rot)] best first


class CacheIndex:
    """Every icon-cache render, grouped by footprint, with a coarse thumbnail stack for
    shortlisting and an LRU of full-resolution images for the exact check."""

    def __init__(self, cache_dir: str | None, assoc: dict | None = None, learned=None, canon: dict | None = None,
                 log=print):
        self.dir = cache_dir
        self.renders: list[Render] = []
        self.by_fp: dict[tuple[int, int], list[int]] = {}
        self._thumb: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._full: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        files = sorted(glob.glob(os.path.join(cache_dir, '*.png')))
        with ThreadPoolExecutor(max_workers=8) as ex:
            imgs = list(ex.map(lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED), files))
        seen: dict[str, int] = {}
        thumbs: dict[tuple[int, int], list] = {}
        for p, img in zip(files, imgs):
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            fp = footprint(img)
            if fp is None:
                continue
            if img.shape[:2] != (SLOT * fp[1] + 1, SLOT * fp[0] + 1):
                continue                              # odd size: not a stash render
            key = content_key_of(img)
            if key in seen:
                continue                              # same pixels under another file number
            fname = os.path.basename(p)
            r = Render(fname, p, fp[0], fp[1], key)
            canon = canon or {}
            if learned is not None:
                iid = learned.get(key)
                if iid:
                    r.item_id, r.id_src = canon.get(iid, iid), 'learned'
            a = (assoc or {}).get(fname)
            if a and a[0]:
                r.hint_id = canon.get(a[0], a[0])
                if not r.item_id and a[1] <= ASSOC_TRUST_D and a[2] >= ASSOC_TRUST_RATIO * max(a[1], 0.3):
                    r.item_id, r.id_src = r.hint_id, 'assoc'
            seen[key] = len(self.renders)
            self.by_fp.setdefault(fp, []).append(len(self.renders))
            thumbs.setdefault(fp, []).append(_thumb(img, fp[0], fp[1]))
            self.renders.append(r)
        for fp, lst in thumbs.items():
            self._thumb[fp] = (np.stack([t[0] for t in lst]), np.stack([t[1] for t in lst]))
        self.key_index = {r.key: i for i, r in enumerate(self.renders)}
        self._orig = [(r.item_id, r.id_src) for r in self.renders]

    def reset_bindings(self) -> None:
        """Back to the identities the index was built with (drops in-memory bindings)."""
        for r, (iid, src) in zip(self.renders, self._orig):
            r.item_id, r.id_src = iid, src

    def __len__(self) -> int:
        return len(self.renders)

    def bind(self, key: str, item_id: str) -> None:
        """Name a render (learned binding, certain)."""
        i = self.key_index.get(key)
        if i is not None:
            self.renders[i].item_id, self.renders[i].id_src = item_id, 'learned'

    def full(self, i: int) -> np.ndarray | None:
        with self._lock:
            img = self._full.get(i)
            if img is not None:
                self._full.move_to_end(i)
                return img
        img = cv2.imread(self.renders[i].path, cv2.IMREAD_UNCHANGED)
        if img is None or content_key_of(img) != self.renders[i].key:
            return None                               # the game recycled the file number
        with self._lock:
            self._full[i] = img
            if len(self._full) > 1500:
                self._full.popitem(last=False)
        return img

    # ------------------------------------------------------------------
    def shortlist(self, tile: np.ndarray, W: int, H: int, pitch_y: float, k: int = SHORTLIST):
        """[(coarse score, render index, rot)] for a tile of footprint W x H (both rotations
        of an H x W render included), best first."""
        th = cv2.resize(tile, (THUMB * W, THUMB * H), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = band_mask(THUMB * H, THUMB * W, THUMB, H)
        out = []
        for fp, rots in (((W, H), (0,)), ((H, W), (-1, 1) if W != H else ())):
            st = self._thumb.get(fp)
            if st is None or not rots:
                continue
            rgb, al = st
            idx = self.by_fp[fp]
            for rot in rots:
                R = rgb if rot == 0 else np.rot90(rgb, rot, axes=(1, 2))
                A = al if rot == 0 else np.rot90(al, rot, axes=(1, 2))
                wgt = A & m[None]
                n = wgt.sum(axis=(1, 2))
                d = (np.abs(R - th[None]).mean(axis=3) * wgt).sum(axis=(1, 2)) / np.maximum(n, 1)
                d[n < 8] = 999.0
                for j in np.argsort(d)[:k]:
                    out.append((float(d[j]), idx[int(j)], rot))
        out.sort(key=lambda t: t[0])
        return out[:k]

    def match(self, tile: np.ndarray, W: int, H: int, pitch_x: float, pitch_y: float,
              phase=None, k: int = SHORTLIST) -> AnchorResult | None:
        """Exact check of the shortlisted renders.  None when no render shortlists."""
        sl = self.shortlist(tile, W, H, pitch_y, k)
        if not sl:
            return None
        ranked = []
        for _, i, rot in sl:
            img = self.full(i)
            if img is None:
                continue
            s, n, ph = exact_residual(tile, _rot(img, rot), pitch_x, pitch_y, H, phase=phase)
            ranked.append((s, n, ph, i, rot))
        if not ranked:
            return None
        ranked.sort(key=lambda t: t[0])
        s0, n0, ph0, i0, rot0 = ranked[0]
        r0 = self.renders[i0]
        runner = next((s for s, _, _, i, _ in ranked[1:]
                       if self.renders[i].item_id != r0.item_id or not r0.item_id), 999.0)
        res = AnchorResult(r0.item_id, False, s0, runner, r0, rot0, ph0, n0,
                           ranked=[(s, self.renders[i], rot) for s, _, _, i, rot in ranked])
        thr = exact_threshold(pitch_y)
        min_px = MIN_PIXELS * (pitch_x * pitch_y) / (SLOT * SLOT)
        if s0 > thr:
            res.note = 'no exact render'
        elif n0 < min_px:
            res.note = 'too little opaque area'
        elif not r0.item_id:
            res.note = 'exact render of an unnamed item'
        elif runner <= max(thr, WRONG_MIN_RATIO * s0):
            res.note = 'another item renders the same'
        else:
            res.certain = True
        return res


def _thumb(img: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    a = img[:, :, 3].astype(np.float32) / 255.0
    rgb = cv2.resize(img[:, :, :3].astype(np.float32), (THUMB * W, THUMB * H), interpolation=cv2.INTER_AREA)
    al = cv2.resize(a, (THUMB * W, THUMB * H), interpolation=cv2.INTER_AREA)
    return rgb, al >= 0.97
