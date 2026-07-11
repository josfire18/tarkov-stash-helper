"""
Icon classifier for Tarkov Stash Helper.

Trains a small CNN on tarkov.dev item icons so the stash scanner can
identify every item by visual appearance rather than OCR.

Pipeline:
  build_or_update()  ← called automatically when icon DB is built
      ↓  downloads icons (already done by icon_db), creates Dataset,
         trains IconNet, saves weights + metadata
  classify_cell()    ← called per cell during sell_scan
      ↓  resizes crop to MODEL_SIZE, runs single forward pass, returns
         (item_id, confidence) for the highest-scoring class in the
         size-appropriate subset.

Auto-update: the metadata file stores the set of item IDs the model was
trained on.  build_or_update() compares that set against the current
price-cache items; if new items are found it retrains from scratch so
nothing is missed.
"""

import os
import json
import time
import random
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA        = os.path.join(BASE, 'data')
ICONS_DIR   = os.path.join(DATA, 'icons')
MODEL_PATH  = os.path.join(DATA, 'icon_model.pt')
META_PATH   = os.path.join(DATA, 'icon_model_meta.json')

MODEL_SIZE  = 96      # pixels — input to CNN (square, aspect-preserved with padding)
N_AUG       = 8       # augmented samples per icon per epoch (kept low for CPU speed)
EPOCHS      = 15
BATCH       = 256     # large batch keeps the GPU fed (RTX has plenty of VRAM)
LR          = 3e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP     = DEVICE == 'cuda'   # mixed-precision on GPU only

BG_COLOR    = (38, 42, 44)   # stash cell background (BGR)

# Shared training-state dict (read by the status endpoint)
train_state = {
    'running':   False,
    'epoch':     0,
    'epochs':    EPOCHS,
    'loss':      None,
    'acc':       None,
    'done':      False,
    'new_items': 0,
    'error':     None,
    'ts_start':  0,
}
_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class IconNet(nn.Module):
    """
    Lightweight CNN — ~1.2M params.
    96×96 input → 4 stages of (Conv→BN→ReLU + ResBlock + MaxPool) → GAP → Linear.
    Inference on CPU: ~2 ms per image.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        def stage(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                _ResBlock(cout),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            stage(3,   32),   # 96 → 48
            stage(32,  64),   # 48 → 24
            stage(64,  128),  # 24 → 12
            stage(128, 256),  # 12 →  6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Preprocessing — shared between training and inference
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_NORMALIZE = T.Normalize(_MEAN, _STD)


def composite_on_dark(icon_bgra, bg=BG_COLOR):
    """Alpha-composite a BGRA icon on a dark stash background → uint8 BGR."""
    if icon_bgra.ndim == 2:
        return cv2.cvtColor(icon_bgra, cv2.COLOR_GRAY2BGR)
    if icon_bgra.shape[2] == 3:
        return icon_bgra
    rgb   = icon_bgra[:, :, :3].astype(np.float32)
    alpha = icon_bgra[:, :, 3:4].astype(np.float32) / 255.0
    bg_arr = np.full_like(rgb, bg, dtype=np.float32)
    return (rgb * alpha + bg_arr * (1 - alpha)).astype(np.uint8)


def pad_to_square(img_bgr):
    """Pad shorter side with the stash background colour so the icon is centred."""
    h, w = img_bgr.shape[:2]
    if h == w:
        return img_bgr
    size = max(h, w)
    out = np.full((size, size, 3), BG_COLOR, dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img_bgr
    return out


def preprocess_icon(path):
    """Load icon PNG → square BGR uint8 ready for augmentation / inference."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    composited = composite_on_dark(raw)
    return pad_to_square(composited)


def to_tensor(img_bgr):
    """BGR uint8 → normalised float32 tensor (C, H, W)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _NORMALIZE(t)


def cell_to_tensor(crop_bgr):
    """
    Preprocess a stash-cell crop for inference.
    Pads to square, resizes to MODEL_SIZE, normalises.
    """
    sq = pad_to_square(crop_bgr)
    sq = cv2.resize(sq, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
    return to_tensor(sq)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(img_bgr, size=MODEL_SIZE):
    """
    Fast numpy-only augmentation pipeline (~0.3 ms/image on CPU).
    Returns a normalised float32 tensor (C, H, W).
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1. Random translation (±6 px)
    dx, dy = random.randint(-6, 6), random.randint(-6, 6)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=BG_COLOR)

    # 2. Random scale (0.88–1.12)
    sc = random.uniform(0.88, 1.12)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((max(h, nh), max(w, nw), 3), BG_COLOR, dtype=np.uint8)
    oy, ox = (canvas.shape[0] - nh) // 2, (canvas.shape[1] - nw) // 2
    canvas[oy:oy+nh, ox:ox+nw] = img
    img = canvas

    # 3. Cell border (present in every real cell)
    bc = random.randint(45, 85)
    t  = random.randint(1, 3)
    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (bc, bc, bc), t)

    # 4. Highlight border (30 % chance — gold or green)
    if random.random() < 0.3:
        col = random.choice([(30, 140, 180), (40, 160, 40)])
        cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), col,
                      random.randint(1, 3))

    # 5. Brightness / contrast jitter (numpy, no PIL)
    f = img.astype(np.float32)
    f *= random.uniform(0.70, 1.30)           # brightness
    f  = (f - 127.5) * random.uniform(0.80, 1.20) + 127.5  # contrast
    img = np.clip(f, 0, 255).astype(np.uint8)

    # 6. Gaussian noise
    if random.random() < 0.5:
        noise = np.random.randint(-12, 12, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 7. Pad to square → resize to MODEL_SIZE
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return to_tensor(img)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IconDataset(Dataset):
    """
    Loads each icon once.  __getitem__ applies a fresh random augmentation
    each call, so every epoch sees different variants without storing N_AUG
    copies in memory.  Effective dataset size = len(icons) × N_AUG.
    """
    def __init__(self, entries, n_aug=N_AUG):
        self.n_aug = n_aug
        self.imgs, self.classes = [], []
        for e in entries:
            img = preprocess_icon(e['path'])
            if img is None:
                continue
            self.imgs.append(img)
            self.classes.append(e['class_id'])

    def __len__(self):
        return len(self.imgs) * self.n_aug

    def __getitem__(self, idx):
        i = idx % len(self.imgs)
        return augment(self.imgs[i]), self.classes[i]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_training(items_meta, progress_cb=None):
    """
    items_meta: list of {'id', 'name', 'W', 'H'}  (only items with downloaded icons)

    Trains IconNet from scratch and saves to MODEL_PATH + META_PATH.
    """
    # Build class list — sorted for determinism
    items_meta = sorted(items_meta, key=lambda x: x['id'])
    id_to_cls  = {it['id']: i for i, it in enumerate(items_meta)}
    cls_to_id  = {i: it['id'] for i, it in enumerate(items_meta)}
    n_classes  = len(items_meta)

    # Build per-size index so inference can restrict classes to valid sizes
    size_to_cls = {}
    for it in items_meta:
        key = (it['W'], it['H'])
        size_to_cls.setdefault(key, []).append(id_to_cls[it['id']])

    # Dataset
    entries = [
        {'class_id': id_to_cls[it['id']],
         'path':     os.path.join(ICONS_DIR, f"{it['id']}.png")}
        for it in items_meta
        if os.path.exists(os.path.join(ICONS_DIR, f"{it['id']}.png"))
    ]
    if not entries:
        raise RuntimeError("No icon images found — run Build Icon DB first.")

    dataset = IconDataset(entries, n_aug=N_AUG)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — no icons loaded. Run Build Icon DB first.")
    # num_workers=0 avoids Windows spawn issues inside background threads
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE == 'cuda'))

    # Model, optimiser, scheduler
    model     = IconNet(n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(xb)
                loss   = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        epoch_loss = total_loss / total
        epoch_acc  = correct / total
        with _state_lock:
            train_state.update({'epoch': epoch, 'loss': round(epoch_loss, 4),
                                'acc': round(epoch_acc * 100, 1)})
        if progress_cb:
            progress_cb(epoch, EPOCHS, epoch_loss, epoch_acc)
        print(f"[classifier] epoch {epoch}/{EPOCHS}  "
              f"loss={epoch_loss:.4f}  acc={epoch_acc*100:.1f}%")

    # Save weights + metadata
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        'num_classes': n_classes,
        'id_to_cls':   id_to_cls,
        'cls_to_id':   cls_to_id,
        'size_to_cls': {f'{w}x{h}': cls_list
                        for (w, h), cls_list in size_to_cls.items()},
        'item_ids':    [it['id'] for it in items_meta],
        'trained_at':  time.time(),
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"[classifier] saved {n_classes}-class model → {MODEL_PATH}")
    return model, meta


def build_or_update(price_cache, force=False, progress_cb=None):
    """
    Check if the model needs (re)training and do so if required.

    Triggers training when:
      - No model exists
      - force=True
      - New item IDs appear in the price cache that weren't in the last training run

    Returns (needs_training: bool, new_item_count: int).
    """
    items = price_cache.get('items', [])
    current_ids = {it['id'] for it in items if
                   os.path.exists(os.path.join(ICONS_DIR, f"{it['id']}.png"))}

    trained_ids = set()
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding='utf-8') as f:
            meta = json.load(f)
        trained_ids = set(meta.get('item_ids', []))

    new_ids = current_ids - trained_ids
    needs   = force or not os.path.exists(MODEL_PATH) or bool(new_ids)

    if not needs:
        return False, 0

    items_meta = [
        {'id': it['id'], 'name': it['name'],
         'W': it.get('width') or 1, 'H': it.get('height') or 1}
        for it in items
        if it['id'] in current_ids
    ]

    with _state_lock:
        train_state.update({
            'running': True, 'epoch': 0, 'epochs': EPOCHS,
            'loss': None, 'acc': None, 'done': False,
            'new_items': len(new_ids), 'error': None,
            'ts_start': time.time(),
        })

    try:
        _run_training(items_meta, progress_cb=progress_cb)
        with _state_lock:
            train_state.update({'running': False, 'done': True})
    except Exception as e:
        with _state_lock:
            train_state.update({'running': False, 'done': False, 'error': str(e)})
        raise

    return True, len(new_ids)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_model_cache = {'model': None, 'meta': None}
_model_lock  = threading.Lock()


def load_model():
    """Load (or return cached) trained model + metadata. Returns (model, meta) or (None, None)."""
    with _model_lock:
        if _model_cache['model'] is not None:
            return _model_cache['model'], _model_cache['meta']
        if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
            return None, None
        with open(META_PATH, encoding='utf-8') as f:
            meta = json.load(f)
        model = IconNet(meta['num_classes'])
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        _model_cache['model'] = model
        _model_cache['meta']  = meta
        return model, meta


def invalidate_model_cache():
    """Call after retraining so the next inference loads fresh weights."""
    with _model_lock:
        _model_cache['model'] = None
        _model_cache['meta']  = None


def classify_cell(crop_bgr, W, H, model, meta, min_conf=0.40):
    """
    Run a single-cell crop through the classifier.

    Only considers classes whose (W, H) matches the crop's slot dimensions,
    so a 1×1 cell won't accidentally match a 2×3 Lion.

    Returns (item_id, confidence) or (None, 0.0) if below threshold.
    """
    size_key   = f'{W}x{H}'
    valid_cls  = meta['size_to_cls'].get(size_key)
    if not valid_cls:
        return None, 0.0

    tensor = cell_to_tensor(crop_bgr).unsqueeze(0)   # (1, C, H, W)
    with torch.no_grad():
        logits = model(tensor)[0]                      # (num_classes,)

    # Mask to valid size classes, softmax over that subset
    mask        = torch.full_like(logits, float('-inf'))
    valid_t     = torch.tensor(valid_cls, dtype=torch.long)
    mask[valid_t] = logits[valid_t]
    probs       = torch.softmax(mask, dim=0)
    best_cls    = int(probs.argmax())
    confidence  = float(probs[best_cls])

    if confidence < min_conf:
        return None, 0.0

    item_id = meta['cls_to_id'].get(str(best_cls)) or meta['cls_to_id'].get(best_cls)
    return item_id, confidence
