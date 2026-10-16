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
  are cached on disk (``EMB_PATH``, keyed by icon file + tint + model id) and
  filled lazily; rotated candidates are embedded on the fly.
* GPU if available (fp16), CPU otherwise.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import threading
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


@contextlib.contextmanager
def low_impact():
    """Run background work (the start-up warm-up) without hurting a game running next to it: this
    thread drops to below-normal priority and torch's CPU pool shrinks to a quarter of the cores
    (restored afterwards).  Never fatal."""
    old_threads = None
    try:
        k = ctypes.WinDLL('kernel32')
        k.GetCurrentThread.restype = ctypes.c_void_p
        k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k.SetThreadPriority(k.GetCurrentThread(), -1)      # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:
        pass
    try:
        if _state.get('torch') is not None:
            t = _state['torch']
            old_threads = t.get_num_threads()
            t.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    except Exception:
        old_threads = None
    try:
        yield
    finally:
        if old_threads is not None:
            try:
                _state['torch'].set_num_threads(old_threads)
            except Exception:
                pass


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


EMB_VERSION = 2            # bump when the embedded picture changes (tint table, band blanking, resolution)
_CHUNK = 192               # icons decoded + embedded per step (bounds memory: never thousands of renders at once)


class EmbeddingStore:
    """Lazy, disk-cached embeddings of catalog icons (rotation 0, rarity tint bg).

    The cache is keyed by the *icon itself* (source file name + size + mtime, rarity tint,
    footprint) plus the model id, not by catalog row / build time: rebuilding the catalog (new
    prices, a new game cache) keeps every embedding whose icon file is unchanged, and only new or
    changed icons are embedded again.  Vectors are held and written as float16 (about 1.5 KB each)."""

    def __init__(self, catalog, path: str = EMB_PATH):
        self.cat = catalog
        self.path = path
        self.lock = threading.RLock()
        self.by_key: dict[str, np.ndarray] = {}        # icon key -> float16 [768]
        self._row_keys: dict[int, str | None] = {}
        self.dirty = False
        self.loaded = 0
        self._load()

    # -- keys ---------------------------------------------------------------
    def _key(self, row: int) -> str | None:
        k = self._row_keys.get(row, '')
        if k != '':
            return k
        try:
            path = self.cat.meta.get('paths', {}).get(int(row))
            st = os.stat(path) if path else None
        except OSError:
            st = None
        if st is None:
            k = None
        else:
            k = (f'{os.path.basename(path)}|{st.st_size}|{st.st_mtime_ns // 1000}|{self.cat.tint[row]}'
                 f'|{int(self.cat.tw[row])}x{int(self.cat.th[row])}')
        self._row_keys[row] = k
        return k

    def has(self, row: int) -> bool:
        k = self._key(row)
        return k is not None and k in self.by_key

    # -- disk ---------------------------------------------------------------
    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                z = np.load(self.path, allow_pickle=False)
                if 'keys' in z.files and int(z['version']) == EMB_VERSION and str(z['model']) == MODEL_ID:
                    keys, vecs = z['keys'], z['vecs']
                    self.by_key = {str(k): v for k, v in zip(keys, vecs.astype(np.float16))}
                    self.loaded = len(self.by_key)
        except Exception:
            self.by_key = {}

    def save(self) -> None:
        with self.lock:
            if not self.dirty or not self.by_key:
                return
            keys = np.array(sorted(self.by_key))
            vecs = np.stack([self.by_key[k] for k in keys]).astype(np.float16)
            self.dirty = False
        tmp = self.path + '.tmp.npz'
        try:
            np.savez_compressed(tmp, keys=keys, vecs=vecs, version=np.array(EMB_VERSION),
                                model=np.array(MODEL_ID))
            os.replace(tmp, self.path)
        except OSError:
            self.dirty = True

    def _icon_image(self, row: int) -> np.ndarray | None:
        full = self.cat.full_icon(row, keep=False)
        if full is None:
            return None
        W, H = int(self.cat.tw[row]), int(self.cat.th[row])
        bg = TINT_BGR.get(str(self.cat.tint[row]), TINT_BGR['default'])
        comp = composite_icon(full, bg)
        return blank_bands(comp, bg, comp.shape[1] / W)

    def missing(self, rows) -> list[int]:
        with self.lock:
            return [r for r in rows if self._key(r) is not None and self._key(r) not in self.by_key]

    def get(self, rows: list[int]) -> np.ndarray:
        """float32 [n, 768] embeddings for catalog rows (computed + cached on demand; rows with
        no readable icon come back as zeros)."""
        need = self.missing(rows)
        if need:
            self._embed_rows(need)
        zero = np.zeros(768, np.float32)
        with self.lock:
            out = [(self.by_key.get(self._key(r)) if self._key(r) is not None else None) for r in rows]
        return np.stack([zero if v is None else v.astype(np.float32) for v in out])

    def _embed_rows(self, need: list[int], persist_every: int = 1536) -> None:
        since_save = 0
        for i in range(0, len(need), _CHUNK):
            imgs, keys = [], []
            for r in need[i:i + _CHUNK]:
                im = self._icon_image(r)
                if im is not None:
                    imgs.append(im)
                    keys.append(self._key(r))
            if not imgs:
                continue
            vecs = embed(imgs).astype(np.float16)
            del imgs
            with self.lock:
                for k, v in zip(keys, vecs):
                    self.by_key[k] = v
                self.dirty = True
            since_save += len(keys)
            if since_save >= persist_every:        # a long cold build survives being interrupted
                self.save()
                since_save = 0
        self.save()

    def precompute(self, log=print, rows=None) -> int:
        """Embed every catalog row that has no cached vector yet; returns how many were new."""
        t = time.time()
        todo = self.missing(range(len(self.cat)) if rows is None else rows)
        if todo:
            self._embed_rows(todo)
        log(f'[dino] embedded {len(todo)} icons in {time.time() - t:.1f}s ({self.loaded} cached)')
        return len(todo)


if __name__ == '__main__':             # python -m identify.dino  -> embed the whole catalog once
    from .catalog import load_catalog
    if not available():
        raise SystemExit(f"DINO unavailable: {_state['err']}")
    EmbeddingStore(load_catalog()).precompute()
