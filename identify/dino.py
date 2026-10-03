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

TINT_BGR = {   # EFT rarity backgrounds (BGR), measured by diffing tarkov.dev grid images against base images
    'black': (20, 19, 19), 'grey': (30, 29, 28), 'default': (54, 54, 53), 'blue': (45, 39, 29),
    'violet': (41, 29, 38), 'yellow': (33, 48, 47), 'green': (24, 34, 27), 'orange': (24, 30, 37),
    'red': (29, 32, 49),
}

_state: dict = {'tried': False, 'ok': False, 'model': None, 'dev': None, 'err': '', 'backend': None}

# int8 ONNX export of the same model (scripts/export_dino_onnx.py + dynamic quantisation): the
# packaged exe has no torch, so stage 2 runs on onnxruntime (CPU) from this file instead.
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'dino_v2s_int8.onnx')
ONNX_SHA256 = 'c9b318a97ebd394bb6412a1f8658c29557a509227d1c23608018a5bec1150be2'


def _load_torch(device):
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
    _state.update(ok=True, model=model, dev=dev, torch=torch, backend='torch')


def _load_onnx():
    import hashlib
    import onnxruntime as ort
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(ONNX_PATH)
    with open(ONNX_PATH, 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != ONNX_SHA256:
        raise ValueError(f'{os.path.basename(ONNX_PATH)} checksum mismatch')
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, so, providers=['CPUExecutionProvider'])
    _state.update(ok=True, model=sess, dev='cpu', backend='onnx')


def available(device: str | None = None, backend: str = 'auto') -> bool:
    """Load the model once; ``False`` if no backend works.  ``backend``: 'torch' (torch +
    transformers, GPU when present), 'onnx' (onnxruntime + the bundled int8 export) or 'auto'
    (torch if importable, else onnx).  The first call decides for the whole process."""
    if _state['tried']:
        return _state['ok']
    _state['tried'] = True
    errs = []
    order = {'torch': ('torch',), 'onnx': ('onnx',)}.get(backend, ('torch', 'onnx'))
    for b in order:
        try:
            _load_torch(device) if b == 'torch' else _load_onnx()
            return True
        except Exception as e:                   # pragma: no cover - environment dependent
            errs.append(f'{b}: {type(e).__name__}: {e}')
    _state['err'] = '; '.join(errs)
    _state['ok'] = False
    return False


def backend() -> str | None:
    return _state['backend']


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
    model, dev = _state['model'], _state['dev']
    out = []
    for i in range(0, len(imgs_bgr), batch):
        x = np.stack([cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (RES, RES),
                                 interpolation=cv2.INTER_CUBIC) for b in imgs_bgr[i:i + batch]])
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        if _state['backend'] == 'onnx':
            out.append(model.run(None, {'pixel_values': x})[0].astype(np.float32))
            continue
        torch = _state['torch']
        t = torch.from_numpy(x).to(dev)
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

    def __init__(self, catalog, path: str | None = None):
        self.cat = catalog
        # embeddings of the two backends differ slightly (int8 weights): never mix them
        self.path = path or (EMB_PATH if _state['backend'] != 'onnx' else EMB_PATH[:-4] + '_onnx.npz')
        self.stamp = float(catalog.meta.get('built', 0.0))
        self.vecs: dict[int, np.ndarray] = {}
        self.rvecs: dict[tuple, np.ndarray] = {}     # (row, rot) of rotated icons (persisted)
        self.mem: dict[tuple, np.ndarray] = {}       # anything else (viewport-clipped crops), this run only
        self.dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                z = np.load(self.path, allow_pickle=False)
                if abs(float(z['stamp']) - self.stamp) < 1e-3:
                    rows, vecs = z['rows'], z['vecs']
                    self.vecs = {int(r): v for r, v in zip(rows, vecs.astype(np.float32))}
                    if 'rrows' in z.files:
                        self.rvecs = {(int(r), int(o)): v for r, o, v in
                                      zip(z['rrows'], z['rrots'], z['rvecs'].astype(np.float32))}
        except Exception:
            self.vecs, self.rvecs = {}, {}

    def save(self) -> None:
        if not self.dirty or not self.vecs:
            return
        rows = np.array(sorted(self.vecs), np.int32)
        vecs = np.stack([self.vecs[int(r)] for r in rows]).astype(np.float16)
        rk = sorted(self.rvecs)
        extra = {}
        if rk:
            extra = dict(rrows=np.array([k[0] for k in rk], np.int32), rrots=np.array([k[1] for k in rk], np.int8),
                         rvecs=np.stack([self.rvecs[k] for k in rk]).astype(np.float16))
        tmp = self.path + '.tmp.npz'
        np.savez_compressed(tmp, rows=rows, vecs=vecs, stamp=np.array(self.stamp), **extra)
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

    def get_keyed(self, keys: list, make) -> dict:
        """Embeddings of derived icon images: ``keys`` are ``(row, rot)`` (rotated icon, persisted)
        or longer tuples (clipped crops, kept for this run); ``make(key)`` renders the image."""
        out, need, imgs = {}, [], []
        for k in keys:
            src = self.rvecs if len(k) == 2 else self.mem
            v = src.get(k)
            if v is not None:
                out[k] = v
            elif k not in need:
                im = make(k)
                if im is not None:
                    need.append(k)
                    imgs.append(im)
        if imgs:
            if len(self.mem) > 20000:
                self.mem.clear()
            for k, v in zip(need, embed(imgs)):
                (self.rvecs if len(k) == 2 else self.mem)[k] = v
                out[k] = v
                if len(k) == 2:
                    self.dirty = True
        return out

    def precompute(self, log=print) -> None:
        """Embed every catalog row (about half a minute on a GPU)."""
        t = time.time()
        rows = [r for r in range(len(self.cat)) if r not in self.vecs]
        for i in range(0, len(rows), 512):
            self.get(rows[i:i + 512])
        self.save()
        log(f'[dino] embedded {len(rows)} icons in {time.time() - t:.1f}s')


if __name__ == '__main__':             # python -m identify.dino  -> embed the whole catalog once
    from .catalog import load_catalog
    if not available():
        raise SystemExit(f"DINO unavailable: {_state['err']}")
    EmbeddingStore(load_catalog()).precompute()
