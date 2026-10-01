"""
Stage 1: masked pixel residual against same-footprint catalog templates.

A tile is the pixels of one item footprint, rescaled to 32 px/slot (the catalog's
stage-1 resolution - the whole-footprint resize also absorbs fractional UI scales
and stretched captures, each axis independently).  Each template is stored
premultiplied, so it can be composited over *the tile's own background colour*:

    pred   = rgb*alpha + (1 - alpha) * bg
    resid  = w_pos * relu(tile - pred) + w_neg * relu(pred - tile)
    score  = sum(resid) / n_pixels                      (mean abs level difference)

``bg`` is the median of strips just inside the tile border, so rarity tints, the
hatched "special" backgrounds and the UI gradient need no per-rarity template
stack.  ``w_pos`` is small inside the label / bottom bands (see :mod:`identify.masks`)
because the game's overlays only ever *add* bright pixels there.

The scan is coarse-to-fine: 16 px/slot over every candidate, then 32 px/slot over the
best ``REFINE`` of them.  Tiles clipped by the viewport are compared with the visible
rows of each template of a plausible full height.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .catalog import Catalog, SizeStack
from .config import STAGE1_SLOT
from .masks import band_rows, residual_weights

S = STAGE1_SLOT
REFINE = 160          # candidates kept after the coarse pass


@dataclass
class Tile:
    """A normalised item tile (stage-1 resolution)."""
    img: np.ndarray                 # uint8 [h, 32W, 3]  (h = 32H, or fewer when clipped)
    W: int
    H: int                          # full footprint height in slots (hypothesis)
    clip: str = ''                  # '' | 'top' | 'bottom'
    vis_rows: int | None = None
    bg: np.ndarray | None = None    # float32 [3]


@dataclass
class Cand:
    row: int                        # global catalog row
    score: float                    # mean abs residual (lower is better)
    rotated: bool = False
    W: int = 0
    H: int = 0
    rot: int = 0                    # quarter turns of the source icon (-1 cw, +1 ccw)


def normalize_tile(img_bgr: np.ndarray, rect: tuple, W: int, pitch_x: float, pitch_y: float,
                   clip_top: bool = False, clip_bottom: bool = False) -> Tile | None:
    """Crop ``rect`` (x, y, w, h incl. border) and rescale to the stage-1 grid."""
    x, y, w, h = rect
    H_img, W_img = img_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W_img, x + w), min(H_img, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = img_bgr[y0:y1, x0:x1]
    tw = S * W
    clip = 'top' if clip_top and not clip_bottom else 'bottom' if clip_bottom else ''
    if clip:
        vis = max(4, int(round((y1 - y0 - 1) / pitch_y * S)))
        t = cv2.resize(crop, (tw, vis), interpolation=cv2.INTER_AREA)
        H = max(1, int(np.ceil((vis - 2) / S)))
        tile = Tile(t, W, H, clip, vis)
    else:
        H = max(1, int(round((y1 - y0 - 1) / pitch_y)))
        t = cv2.resize(crop, (tw, S * H), interpolation=cv2.INTER_AREA)
        tile = Tile(t, W, H)
    tile.bg = estimate_bg(tile)
    return tile


def estimate_bg(tile: Tile) -> np.ndarray:
    """Background colour from strips just inside the left/right border, outside
    the label/bottom bands (median => robust to art touching the strip)."""
    h, w = tile.img.shape[:2]
    top, bot = band_rows(S, max(1, h // S))
    r0 = min(top + 1, h // 3)
    r1 = max(r0 + 2, h - bot - 1) if tile.clip != 'bottom' else h - 1
    if tile.clip == 'top':
        r0 = 1
    strips = np.concatenate([tile.img[r0:r1, 2:5].reshape(-1, 3), tile.img[r0:r1, w - 5:w - 2].reshape(-1, 3)])
    if strips.size == 0:
        strips = tile.img.reshape(-1, 3)
    return np.median(strips, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_coarse_cache: dict[int, tuple] = {}


def _pool2(a: np.ndarray) -> np.ndarray:
    n, h, w = a.shape[:3]
    h2, w2 = h // 2, w // 2
    a = a[:, :h2 * 2, :w2 * 2].astype(np.float32)
    if a.ndim == 4:
        return a.reshape(n, h2, 2, w2, 2, -1).mean(axis=(2, 4))
    return a.reshape(n, h2, 2, w2, 2).mean(axis=(2, 4))


def _coarse(stack: SizeStack) -> tuple[np.ndarray, np.ndarray]:
    k = id(stack)
    c = _coarse_cache.get(k)
    if c is None or c[0] is not stack:
        c = (stack, _pool2(stack.prem), _pool2(stack.alpha))
        if len(_coarse_cache) > 64:
            _coarse_cache.clear()
        _coarse_cache[k] = c
    return c[1], c[2]


BG_MIN_PIXELS = 60     # transparent pixels needed before a template's own bg estimate is trusted
BG_ALPHA_MAX = 12      # alpha (0-255) below which a template pixel counts as "background"


def _residual(prem: np.ndarray, alpha: np.ndarray, tile: np.ndarray, bg0: np.ndarray,
              wp: np.ndarray, wn: np.ndarray, norm: float) -> np.ndarray:
    """Mean weighted abs residual of each template (axis 0) against the tile.

    The background is estimated *per template*: mean tile colour over the pixels where
    that template is transparent (a global strip estimate fails when the art touches
    the border strips - the 'Pile of meds' icon fills the whole tile).  Templates with
    almost no transparent area fall back to the strip estimate ``bg0``."""
    a = alpha[..., None] * (1.0 / 255.0)
    m = ((alpha < BG_ALPHA_MAX) & (wp[None] > 0.5)).astype(np.float32)
    cnt = m.sum(axis=(1, 2))
    s_t = (m[..., None] * tile[None]).sum(axis=(1, 2))
    bgc = np.where((cnt >= BG_MIN_PIXELS)[:, None], s_t / np.maximum(cnt, 1)[:, None], bg0[None])

    def one(bg):
        pred = prem + (1.0 - a) * bg[:, None, None, :]
        d = tile[None] - pred
        r = np.maximum(d, 0) * wp[None, ..., None] + np.maximum(-d, 0) * wn[None, ..., None]
        return r.sum(axis=(1, 2, 3)) / (3.0 * norm)

    return np.minimum(one(np.broadcast_to(bg0, bgc.shape)), one(bgc))


# --------------------------------------------------------------------------
# optional torch backend (same maths, GPU resident template stacks)
# --------------------------------------------------------------------------

_torch_state: dict = {'dev': None, 'torch': None, 'stacks': {}}


def enable_torch(device: str | None = None) -> bool:
    """Run stage 1 on the GPU if torch + CUDA are importable (``device='cpu'`` or a
    missing torch leaves the numpy path in place).  Returns True when active."""
    try:
        import torch
    except Exception:
        _torch_state.update(dev=None, torch=None)
        return False
    dev = device or ('cuda' if torch.cuda.is_available() else None)
    if dev != 'cuda':
        _torch_state.update(dev=None, torch=None)
        return False
    _torch_state.update(dev=dev, torch=torch, stacks={})
    return True


def torch_active() -> bool:
    return _torch_state['dev'] is not None


def _gpu_stack(stack: SizeStack):
    k = id(stack)
    e = _torch_state['stacks'].get(k)
    if e is None or e[0] is not stack:
        t = _torch_state['torch']
        dev = _torch_state['dev']
        e = (stack, t.from_numpy(stack.prem).to(dev), t.from_numpy(stack.alpha).to(dev))
        _torch_state['stacks'][k] = e
    return e[1], e[2]


def _residual_torch(stack: SizeStack, tile: Tile, wp: np.ndarray, wn: np.ndarray, norm: float) -> np.ndarray:
    t = _torch_state['torch']
    dev = _torch_state['dev']
    prem, alpha = _gpu_stack(stack)
    if tile.clip == 'bottom':
        prem, alpha = prem[:, :tile.vis_rows], alpha[:, :tile.vis_rows]
    elif tile.clip == 'top':
        prem, alpha = prem[:, -tile.vis_rows:], alpha[:, -tile.vis_rows:]
    timg = t.from_numpy(tile.img.astype(np.float32)).to(dev)
    bg = t.from_numpy(tile.bg.astype(np.float32)).to(dev)
    wpt = t.from_numpy(wp).to(dev)[None, ..., None]
    wnt = t.from_numpy(wn).to(dev)[None, ..., None]
    out = []
    for i in range(0, prem.shape[0], 1024):            # bound peak memory
        al = alpha[i:i + 1024]
        a = al.float().unsqueeze(-1) * (1.0 / 255.0)
        m = ((al < BG_ALPHA_MAX) & (wpt[..., 0] > 0.5)).float()
        cnt = m.sum(dim=(1, 2))
        s_t = (m.unsqueeze(-1) * timg[None]).sum(dim=(1, 2))
        bgc = t.where((cnt >= BG_MIN_PIXELS)[:, None], s_t / cnt.clamp(min=1)[:, None], bg[None])
        pr = prem[i:i + 1024].float()
        wsum = lambda p_: (t.relu(timg[None] - p_) * wpt + t.relu(p_ - timg[None]) * wnt).sum(dim=(1, 2, 3))
        r = t.minimum(wsum(pr + (1.0 - a) * bg), wsum(pr + (1.0 - a) * bgc[:, None, None, :])) / (3.0 * norm)
        out.append(r)
    return t.cat(out).cpu().numpy()


def _crop_stack(prem: np.ndarray, alpha: np.ndarray, tile: Tile):
    """Visible rows of a template stack for a clipped tile."""
    if not tile.clip:
        return prem, alpha
    v = tile.vis_rows
    if tile.clip == 'bottom':
        return prem[:, :v], alpha[:, :v]
    return prem[:, -v:], alpha[:, -v:]


def score_stack(stack: SizeStack, tile: Tile) -> np.ndarray:
    """Residual of every template in ``stack`` against ``tile`` (coarse then fine)."""
    prem, alpha = _crop_stack(stack.prem, stack.alpha, tile)
    n = prem.shape[0]
    h, w = tile.img.shape[:2]
    if prem.shape[1] != h:                       # visible-row mismatch (very short clip)
        return np.full(n, 1e3, np.float32)
    wp, wn = residual_weights(tile.W, stack.H if not tile.clip else tile.H, S,
                              tile.vis_rows, tile.clip)
    norm = float((wn > 0).sum())
    if torch_active():
        return _residual_torch(stack, tile, wp, wn, norm)
    timg = tile.img.astype(np.float32)
    if n > REFINE:
        cp, ca = _coarse(stack)
        if tile.clip:
            v2 = tile.vis_rows // 2
            cp, ca = (cp[:, :v2], ca[:, :v2]) if tile.clip == 'bottom' else (cp[:, -v2:], ca[:, -v2:])
        th, tw = cp.shape[1:3]
        t2 = cv2.resize(timg, (tw, th), interpolation=cv2.INTER_AREA)
        wp2 = cv2.resize(wp, (tw, th), interpolation=cv2.INTER_AREA)
        wn2 = cv2.resize(wn, (tw, th), interpolation=cv2.INTER_AREA)
        coarse = _residual(cp, ca, t2, tile.bg, wp2, wn2, float((wn2 > 0.5).sum()))
        keep = np.argpartition(coarse, REFINE)[:REFINE]
        res = np.full(n, 1e3, np.float32)
        res[keep] = _residual(prem[keep].astype(np.float32), alpha[keep].astype(np.float32),
                              timg, tile.bg, wp, wn, norm)
        return res
    return _residual(prem.astype(np.float32), alpha.astype(np.float32), timg, tile.bg, wp, wn, norm)


def stage1(cat: Catalog, tile: Tile, top_k: int | None = 10, rotations: bool = True) -> list[Cand]:
    """Distinct items ranked by residual (one entry per item id; anonymous ``build``
    templates stay separate).  ``top_k=None`` returns every item of the footprint."""
    heights = [tile.H] if not tile.clip else [tile.H + k for k in range(0, 4)]
    best: dict[str, Cand] = {}
    for Hh in heights:
        for st in cat.stacks_for(tile.W, Hh, rotations=rotations and tile.W != Hh):
            if tile.clip and st.rotated is False and st.H != Hh:
                continue
            sub_tile = tile if not tile.clip else Tile(tile.img, tile.W, Hh, tile.clip, tile.vis_rows, tile.bg)
            res = score_stack(st, sub_tile)
            order = np.argsort(res)[:(max(top_k * 3, 30) if top_k else len(res))]
            for j in order:
                row = int(st.idx[j])
                iid = str(cat.ids[row]) or f'#{row}'
                sc = float(res[j])
                c = best.get(iid)
                if c is None or sc < c.score:
                    best[iid] = Cand(row, sc, st.rotated, st.W, Hh, st.rot)
    out = sorted(best.values(), key=lambda c: c.score)
    return out[:top_k] if top_k else out
