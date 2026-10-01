"""
Stage 2: DINOv2-small re-ranking of the stage-1 shortlist.

Pixel residuals are excellent when the template and the tile are the same render,
but they treat a one-pixel misregistration like a different item and cannot tell
which of several near-identical silhouettes is the right *picture*.  A small
self-supervised ViT embeds the overall appearance: the assessment measured
44/45 top-1 on labelled meds with DINOv2-small against 43/45 for plain masked MAD
(NCC was the weak point).

Design notes
------------
* Lazy: ``torch``/``transformers`` are only imported when stage 2 is first used, and
  every call degrades gracefully (``available() == False``) so a build without torch
  just runs stages 1 + 3.
* The label and bottom bands are painted with the tile's background colour before
  embedding *both* sides, so printed names / counts do not leak into similarity.
* Catalog embeddings (icon composited over its rarity tint, same band blanking)
  are cached on disk (``EMB_PATH``, keyed by catalog row + catalog build time) and
  filled lazily; rotated candidates are embedded on the fly.
* GPU if available (fp16), CPU otherwise.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .config import EMB_PATH, SLOT
from .masks import LABEL_PX, BOTTOM_PX

MODEL_ID = 'facebook/dinov2-small'
RES = 224

TINT_BGR = {   # EFT rarity backgrounds (BGR), measured in the legacy engine (see app.EFT_BG_TINTS)
    'black': (20, 19, 19), 'grey': (30, 29, 28), 'default': (54, 54, 53), 'blue': (45, 39, 29),
    'violet': (41, 29, 38), 'yellow': (33, 48, 47), 'green': (24, 34, 27), 'orange': (24, 30, 37),
    'red': (29, 32, 49),
}

_state: dict = {'tried': False, 'ok': False, 'model': None, 'dev': None, 'err': ''}


def available(device: str | None = None) -> bool:
    """Try to load the model once; ``False`` if torch/transformers/weights are missing."""
    if _state['tried']:
        return _state['ok']
    _state['tried'] = True
    try:
        import torch
        from transformers import AutoModel
        dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            model = AutoModel.from_pretrained(MODEL_ID)
        model = model.to(dev).eval()
        if dev == 'cuda':
            model = model.half()
        _state.update(ok=True, model=model, dev=dev, torch=torch)
    except Exception as e:                       # pragma: no cover - environment dependent
        _state['err'] = f'{type(e).__name__}: {e}'
        _state['ok'] = False
    return _state['ok']


def device() -> str | None:
    return _state['dev']


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def blank_bands(img_bgr: np.ndarray, bg, slot_px: float, clip: str = '') -> np.ndarray:
    """Paint the label (top) and overlay (bottom) bands with the background colour."""
    out = img_bgr.copy()
    h = out.shape[0]
    top = int(round(LABEL_PX * slot_px / SLOT))
    bot = int(round(BOTTOM_PX * slot_px / SLOT))
    bg = np.asarray(bg, np.uint8)
    if clip != 'top':
        out[:top] = bg
    if clip != 'bottom':
        out[h - bot:] = bg
    return out


def embed(imgs_bgr: list, batch: int = 64) -> np.ndarray:
    """L2-normalised [n, 768] float32 embeddings (CLS + mean patch token)."""
    torch = _state['torch']
    model, dev = _state['model'], _state['dev']
    out = []
    for i in range(0, len(imgs_bgr), batch):
        x = np.stack([cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (RES, RES),
                                 interpolation=cv2.INTER_CUBIC) for b in imgs_bgr[i:i + batch]])
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        if dev == 'cuda':
            t = t.half()
        with torch.no_grad():
            h = model(pixel_values=t).last_hidden_state
            e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()
            e = torch.nn.functional.normalize(e, dim=1)
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def composite_icon(bgra: np.ndarray, bg, rotate: int = 0) -> np.ndarray:
    """Icon over ``bg`` (BGR), optionally rotated (+1 = ccw, -1 = cw quarter turns)."""
    if rotate:
        bgra = np.rot90(bgra, rotate)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
    return np.clip(rgb * a + (1 - a) * np.asarray(bg, np.float32), 0, 255).astype(np.uint8)


class EmbeddingStore:
    """Lazy, disk-cached embeddings of catalog icons (rotation 0, rarity tint bg)."""

    def __init__(self, catalog, path: str = EMB_PATH):
        self.cat = catalog
        self.path = path
        self.stamp = float(catalog.meta.get('built', 0.0))
        self.vecs: dict[int, np.ndarray] = {}
        self.dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                z = np.load(self.path, allow_pickle=False)
                if abs(float(z['stamp']) - self.stamp) < 1e-3:
                    rows, vecs = z['rows'], z['vecs']
                    self.vecs = {int(r): v for r, v in zip(rows, vecs.astype(np.float32))}
        except Exception:
            self.vecs = {}

    def save(self) -> None:
        if not self.dirty or not self.vecs:
            return
        rows = np.array(sorted(self.vecs), np.int32)
        vecs = np.stack([self.vecs[int(r)] for r in rows]).astype(np.float16)
        tmp = self.path + '.tmp.npz'
        np.savez_compressed(tmp, rows=rows, vecs=vecs, stamp=np.array(self.stamp))
        os.replace(tmp, self.path)
        self.dirty = False

    def _icon_image(self, row: int) -> np.ndarray | None:
        full = self.cat.full_icon(row)
        if full is None:
            return None
        W, H = int(self.cat.tw[row]), int(self.cat.th[row])
        bg = TINT_BGR.get(str(self.cat.tint[row]), TINT_BGR['default'])
        comp = composite_icon(full, bg)
        return blank_bands(comp, bg, comp.shape[1] / W)

    def get(self, rows: list[int]) -> np.ndarray:
        """Embeddings for catalog rows (computed + cached on demand)."""
        need = [r for r in rows if r not in self.vecs]
        if need:
            imgs, keep = [], []
            for r in need:
                im = self._icon_image(r)
                if im is not None:
                    imgs.append(im)
                    keep.append(r)
            if imgs:
                for r, v in zip(keep, embed(imgs)):
                    self.vecs[r] = v
                self.dirty = True
        return np.stack([self.vecs.get(r, np.zeros(768, np.float32)) for r in rows])

    def precompute(self, log=print) -> None:
        """Embed every catalog row (about half a minute on a GPU)."""
        t = time.time()
        rows = [r for r in range(len(self.cat)) if r not in self.vecs]
        for i in range(0, len(rows), 512):
            self.get(rows[i:i + 512])
        self.save()
        log(f'[dino] embedded {len(rows)} icons in {time.time() - t:.1f}s')
