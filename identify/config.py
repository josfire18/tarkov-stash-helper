"""
Shared constants and settings for the v2 identification engine.

Everything here is derived from how EFT draws the stash (measured on real
1080p screenshots, see ``data/eval/stash1.png``):

* One 1x1 slot is 63 px at 1080p; an item of W x H slots occupies a
  ``(63*W+1) x (63*H+1)`` pixel tile (the +1 is the shared border line).  The
  icons in EFT's local Icon Cache and tarkov.dev's base images use exactly this
  geometry, which is why they can be compared pixel-for-pixel after the tile is
  rescaled to 63 px/slot.
* Every item is framed by a 1 px line, colour (84, 81, 73) BGR / (73, 81, 84)
  RGB.  Lines that would run *inside* a multi-cell item are not drawn - that is
  the signal :mod:`identify.segment` uses to recover item footprints.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

SLOT = 63                      # px per slot at the catalog's reference scale
LINE_BGR = (84, 81, 73)        # item/cell border colour (BGR)
LINE_TOL = 14                  # per-channel tolerance when masking line pixels

MIN_PITCH = 20.0               # smallest supported px/slot (very small windows)
MAX_PITCH = 220.0              # largest supported px/slot (4K + big UI scale)

# data/ lives next to the .exe when frozen (PyInstaller extracts code to a temp dir)
ROOT = (os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data')
TMPL_SRC_DIR = os.path.join(DATA_DIR, 'tmpl_src')
PRICES_PATH = os.path.join(DATA_DIR, 'prices_cache.json')
CATALOG_PATH = os.path.join(DATA_DIR, 'identify_catalog_v2.npz')
EMB_PATH = os.path.join(DATA_DIR, 'identify_dino_v2.npz')
LEARNED_PATH = os.path.join(DATA_DIR, 'learned_icons.json')   # cache icon -> item, from confirmed scans

CATALOG_SCHEMA = 6             # bump when the .npz layout / semantics change
STAGE1_SLOT = 32               # px/slot of the stage-1 (MAD) template stacks


@dataclass
class EngineSettings:
    """Run-time knobs for :func:`identify.pipeline.scan`."""
    use_dino: bool = True            # stage 2 (torch + transformers, or onnxruntime + the bundled export)
    dino_backend: str = 'auto'       # 'auto' | 'torch' | 'onnx' (see identify.dino.available)
    use_ocr: bool = True             # stage 3 (needs pytesseract + Tesseract)
    accelerate: bool = True          # run stage 1 on the GPU when torch+CUDA are importable
    top_k: int = 16                  # candidates handed from stage 1 to stage 2
    uncertain_below: float = 0.80    # calibrated confidence below this => uncertain
    pitch_hint: float | None = None  # expected px/slot when the UI scale is known
    device: str | None = None        # 'cuda' | 'cpu' | None (auto)
    tesseract_cmd: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: dict | None) -> 'EngineSettings':
        s = (settings or {}).get('identify_v2') or {}
        known = {k: v for k, v in s.items() if k in cls.__dataclass_fields__}
        return cls(**known)
