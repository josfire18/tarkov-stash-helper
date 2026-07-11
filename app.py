import os
import sys
import shutil
import json
import base64
import uuid
import threading
import time
from io import BytesIO

import math
import requests as http_requests

from flask import Flask, jsonify, request, render_template
import mss
from PIL import Image, ImageDraw
import pytesseract
from rapidfuzz import process as rfuzz
from pynput import keyboard
import cv2
import numpy as np

FROZEN = getattr(sys, 'frozen', False)

# When run from source, data/ lives next to app.py. When packaged with
# PyInstaller (--onefile), __file__ resolves inside the temp extraction dir,
# so data/ must instead live next to the .exe or user settings/caches would
# vanish every launch.
if FROZEN:
    BASE = os.path.dirname(sys.executable)
    # templates/ (and any other bundled read-only assets) still ship inside
    # the PyInstaller bundle.
    BUNDLE = getattr(sys, '_MEIPASS', BASE)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLE = BASE

app = Flask(__name__, template_folder=os.path.join(BUNDLE, 'templates'))

DATA = os.path.join(BASE, 'data')
SETTINGS_PATH  = os.path.join(DATA, 'settings.json')
KEEPLIST_PATH  = os.path.join(DATA, 'keep_list.json')
PRICES_PATH    = os.path.join(DATA, 'prices_cache.json')
KAPPA_WIKI_PATH  = os.path.join(DATA, 'kappa_wiki.json')   # cached Collector item names from the wiki
TASKS_CACHE_PATH = os.path.join(DATA, 'tasks_cache.json')  # cached tasks + hideout requirements (tarkov.dev)
PROGRESS_PATH    = os.path.join(DATA, 'progress.json')     # user task/hideout completion + have-counts
ICONS_DIR      = os.path.join(DATA, 'icons')          # legacy 64×64 iconLink thumbnails (UI only)
TMPL_SRC_DIR   = os.path.join(DATA, 'tmpl_src')        # transparent per-slot base images (BGRA PNG)
os.makedirs(DATA, exist_ok=True)
os.makedirs(ICONS_DIR, exist_ok=True)
os.makedirs(TMPL_SRC_DIR, exist_ok=True)

# Tesseract-OCR is an external (non-pip) dependency the user must install
# separately. pytesseract only finds it automatically if it's on PATH; fall
# back to the default Windows install location so a fresh setup doesn't
# require manually editing PATH.
if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
    _default_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(_default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _default_tesseract

OCR_THRESHOLD        = 65    # fuzzy match score cutoff
OCR_SCALE            = 2     # scale factor applied to image before OCR (larger = better accuracy)
PRICE_CACHE_TTL      = 1800  # seconds (30 min)
KAPPA_WIKI_TTL       = 86400 # seconds (24 h) — Collector list changes rarely
TASKS_CACHE_TTL      = 86400 # seconds (24 h) — task/hideout requirements change per patch
FLEA_MIN_PROFIT      = 10000 # recommend flea only if net > trader by this much
TARKOV_API           = 'https://api.tarkov.dev/graphql'
# The raw wiki page (escapefromtarkov.fandom.com/wiki/Collector) sits behind a
# Cloudflare bot challenge and returns "Just a moment..." to plain HTTP clients.
# The MediaWiki API serves the identical rendered HTML unchallenged — always
# fetch through it, never the page URL.
COLLECTOR_WIKI_API   = ('https://escapefromtarkov.fandom.com/api.php'
                        '?action=parse&page=Collector&prop=text&format=json&formatversion=2')
ICON_MATCH_THRESHOLD = 0.68  # cv2.TM_CCOEFF_NORMED score cutoff for icon matching
CANONICAL_PER_SLOT   = 64    # px per 1×1 slot in the canonical-size template
ICON_MATCH_MIN_SCORE = 0.40  # NCC threshold to accept an icon match
LABEL_BLANK_PX       = 13    # top rows of each cell to overwrite with bg colour (removes item-name label)
CORNER_BLANK_PX      = 18    # top-right (FiR ✓) and bottom-right (stack count) corner blanking
NCC_MARGIN_MIN       = 0.04  # require top-1 NCC to beat top-2 by this much (rejects ambiguous matches)
ICON_DB_PATH         = os.path.join(DATA, 'icon_db.npz')
DB_VERSION           = 5     # bump whenever the vector format changes; forces a rebuild
_STASH_BG_BGR        = (38, 42, 44)  # Tarkov stash cell background colour (BGR) — used for blanked (masked-out) regions
GRID_PITCH_1080P     = 63    # px per slot in-game @ 1080p (NOT 64 — verified from the icon cache geometry)

# EFT rarity background tints (BGR), calibrated by diffing tarkov.dev grid-image
# (background baked) against base-image (transparent alpha) over 6 items/colour.
# Templates are alpha-composited onto these so anti-aliased icon edges match the
# real tinted stash cell.  Keyed by the tarkov.dev `backgroundColor` field.
EFT_BG_TINTS = {
    'black':   (20, 19, 19),
    'grey':    (30, 29, 28),
    'default': (54, 54, 53),
    'blue':    (45, 39, 29),
    'violet':  (41, 29, 38),
    'yellow':  (33, 48, 47),
    'green':   (24, 34, 27),
    'orange':  (24, 30, 37),
    'red':     (29, 32, 49),
}
DEFAULT_TINT = (54, 54, 53)

# The scanner only cares about items / parts / components — NOT ammo, full
# weapons, weapon presets, or storage containers.  Templates for these tarkov.dev
# `types` are never built, so they can't be matched or become distractors.
# (Weapon *parts* — barrels, stocks, scopes, grips, suppressors — are type
# 'mods' and are kept.)
EXCLUDED_TYPES = {'ammo', 'ammoBox', 'gun', 'preset', 'container'}


def is_target_item(item):
    """True if the item is a match target (not ammo / gun / preset / container)."""
    return not (set(item.get('types') or ()) & EXCLUDED_TYPES)

PRICE_QUERY = '''{
  items {
    id name shortName basePrice avg24hPrice low24hPrice iconLink width height
    backgroundColor gridImageLink baseImageLink types
    sellFor { vendor { name } priceRUB }
  }
}'''

TASKS_QUERY = '''{
  tasks {
    id name minPlayerLevel kappaRequired
    trader { name }
    objectives {
      id type
      ... on TaskObjectiveItem { count foundInRaid item { id name shortName } }
    }
  }
  hideoutStations {
    id name
    levels { id level itemRequirements { count item { id name shortName } } }
  }
}'''


class ScanError(Exception):
    """User-actionable scan failure (no region set, capture failed, ...)."""


# Shared state for hotkey-triggered scans
_last_scan = {'image': None, 'detections': [], 'ts': 0,
              'error': None, 'warnings': [], 'checklist_matches': []}
_scan_lock = threading.Lock()

# Live progress of the currently running scan, polled by the frontend.
_scan_state = {'running': False, 'phase': None, 'done': 0, 'total': 0, 'ts': 0}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_json(path, default_fn):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    data = default_fn()
    save_json(path, data)
    return data

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

def fetch_prices():
    """Fetch all item prices from tarkov.dev and write to cache."""
    r = http_requests.post(TARKOV_API, json={'query': PRICE_QUERY}, timeout=30)
    items = r.json()['data']['items']
    cache = {'timestamp': time.time(), 'items': items}
    save_json(PRICES_PATH, cache)
    return cache

def get_prices():
    """Return cached prices, refreshing if stale."""
    if os.path.exists(PRICES_PATH):
        cache = load_json(PRICES_PATH, lambda: None)
        if cache and time.time() - cache.get('timestamp', 0) < PRICE_CACHE_TTL:
            return cache
    return fetch_prices()

def build_price_index(cache):
    """Build name/shortname lookup from cache."""
    idx = {}
    for item in cache.get('items', []):
        idx[item['name'].lower()] = item
        idx[item['shortName'].lower()] = item
    return idx


# ---------------------------------------------------------------------------
# Shared scan helpers
# ---------------------------------------------------------------------------

_tesseract_ok = None

def tesseract_available():
    """True if the Tesseract binary is installed and runnable (cached)."""
    global _tesseract_ok
    if _tesseract_ok is None:
        try:
            pytesseract.get_tesseract_version()
            _tesseract_ok = True
        except Exception:
            _tesseract_ok = False
    return _tesseract_ok


def capture_stash_image(settings, require_region=False):
    """
    Grab the configured monitor/region and return (img_pil, img_bgr).
    Raises ScanError when require_region is set but no region is configured.
    """
    region = settings.get('region')
    if require_region and not region:
        raise ScanError('No stash region set. Click "📐 Set Stash Region" first.')
    with mss.mss() as sct:
        monitors = sct.monitors
        if len(monitors) < 2:
            raise ScanError('No monitor available for capture.')
        monitor_idx = settings.get('monitor', 0)
        monitor = monitors[monitor_idx + 1] if monitor_idx + 1 < len(monitors) else monitors[1]
        if region:
            capture_region = {
                'left':   monitor['left'] + region['x'],
                'top':    monitor['top']  + region['y'],
                'width':  region['w'],
                'height': region['h'],
            }
        else:
            capture_region = monitor
        raw = sct.grab(capture_region)
        img = Image.frombytes('RGB', raw.size, raw.bgra, 'raw', 'BGRX')
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img, img_bgr


def map_keep_entries_to_ids(keep_list, price_idx):
    """
    Map every keep-list entry to its tarkov.dev item, conservatively.
    Returns ({tdev_id: entry}, [unmapped_names]).

    Order of trust: exact full-name match > persisted tdev_id > unambiguous
    alias hits (≥4 chars, all agreeing on one item) > tight fuzzy fallback.
    The old first-alias-wins logic let generic aliases like "Beer" or "Water"
    claim the wrong item entirely.
    """
    id_ok = set()
    for it_key, it in price_idx.items():
        id_ok.add(it['id'])

    mapped, unmapped = {}, []
    for cat in keep_list['categories']:
        for entry in cat['items']:
            data = price_idx.get(entry['name'].lower())
            if data is None and entry.get('tdev_id') in id_ok:
                mapped[entry['tdev_id']] = entry
                continue
            if data is None:
                hits = {}
                for alias in entry.get('aliases', []):
                    if len(alias) < 4:
                        continue
                    hit = price_idx.get(alias.lower())
                    if hit:
                        hits[hit['id']] = hit
                if len(hits) == 1:
                    data = next(iter(hits.values()))
            if data is None:
                names = [k for k in price_idx]
                r = rfuzz.extractOne(entry['name'].lower(), names, score_cutoff=93)
                if r:
                    data = price_idx[r[0]]
            if data is not None:
                mapped[data['id']] = entry
            else:
                unmapped.append(entry['name'])
    return mapped, unmapped

# ---------------------------------------------------------------------------
# Icon download + template matching helpers
# ---------------------------------------------------------------------------

def icon_cache_path(item_id):
    return os.path.join(ICONS_DIR, f'{item_id}.png')

_icon_session = None
def _get_icon_session():
    global _icon_session
    if _icon_session is None:
        s = http_requests.Session()
        adapter = http_requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
        s.mount('https://', adapter)
        s.mount('http://',  adapter)
        _icon_session = s
    return _icon_session

def download_icon(item_id, icon_url):
    """Download icon PNG from tarkov.dev and cache it. Returns local path or None."""
    path = icon_cache_path(item_id)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        r = _get_icon_session().get(icon_url, timeout=10)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
            return path
    except Exception:
        pass
    return None

# match_icon() (whole-screenshot cv2.matchTemplate against 64×64 iconLink
# thumbnails) was retired: it stretched aspect-wrong thumbnails and drifted
# across the grid.  Keep-list highlighting now runs through the same masked-NCC
# icon DB as the sell scanner (see /api/screenshot).


# ---------------------------------------------------------------------------
# Icon matching via canonical-resolution normalized cross-correlation (NCC).
#
# Strategy:
#   1. At DB build time: composite every item icon on a dark stash-like bg,
#      resize it to a CANONICAL size (CANONICAL_PER_SLOT × W, CANONICAL_PER_SLOT × H),
#      convert to grayscale, and store the flattened uint8 vector.
#   2. At scan time: crop each candidate cell block, resize to the same
#      canonical size, flatten, and compute NCC against every template of
#      that (W,H) size via a single batched dot product.
#
# Because every cell is resized independently, sub-pixel grid drift across
# the screenshot does NOT accumulate — each cell is normalized to the same
# reference frame as the database icons.
# ---------------------------------------------------------------------------

_UINT64 = np.uint64   # kept for backwards compat, unused by NCC

def _composite_on_dark(icon_bgra, bg_rgb=(38, 42, 44)):
    """Composite a BGRA icon onto a solid background colour (BGR tuple)."""
    if icon_bgra.ndim == 2:
        return cv2.cvtColor(icon_bgra, cv2.COLOR_GRAY2BGR)
    if icon_bgra.shape[2] == 3:
        return icon_bgra
    rgb = icon_bgra[:, :, :3].astype(np.float32)
    alpha = (icon_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    bg = np.full_like(rgb, bg_rgb, dtype=np.float32)
    out = rgb * alpha + bg * (1.0 - alpha)
    return out.astype(np.uint8)


def tint_for(item):
    """Return the EFT stash-cell background tint (BGR) for an item's rarity colour."""
    return EFT_BG_TINTS.get(item.get('backgroundColor') or 'default', DEFAULT_TINT)


def base_image_path(item_id):
    return os.path.join(TMPL_SRC_DIR, f'{item_id}.png')


def download_base_image(item_id, url):
    """
    Download a tarkov.dev base-image (transparent, per-slot resolution) and
    cache it as a BGRA PNG.  Returns local path or None.

    The base-image is the correct-geometry, real-alpha source that replaces the
    fatal 64×64 iconLink thumbnail for template matching.
    """
    path = base_image_path(item_id)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not url:
        return None
    try:
        r = _get_icon_session().get(url, timeout=12)
        if r.status_code != 200:
            return None
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)   # decodes webp → BGRA
        if img is None or img.size == 0:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            cv2.imwrite(path, img)
        else:
            # No alpha (rare) — store as-is; mask falls back to saturation heuristic
            cv2.imwrite(path, img)
        return path
    except Exception:
        return None

def _apply_label_blank(img_bgr):
    """
    Overwrite Tarkov UI overlay regions with the stash background colour:
      - top LABEL_BLANK_PX rows  → erases the white item-name label
      - top-right corner         → erases the Found-in-Raid ✓
      - bottom-right corner      → erases the white stack-count digits

    Game crops: removes UI artefacts the game overlays on the cell.
    Icon templates: those regions are already transparent-composited to bg,
                    so this is effectively a no-op (kept for symmetry).

    Overwriting (not cropping) preserves image dimensions so the downstream
    resize step has no geometry distortion.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    n = min(LABEL_BLANK_PX, h // 4)
    if n > 0:
        out[:n, :] = _STASH_BG_BGR
    c = min(CORNER_BLANK_PX, h // 3, w // 3)
    if c > 0:
        # top-right corner (FiR check mark)
        out[:c, w - c:] = _STASH_BG_BGR
        # bottom-right corner (stack-count digits)
        out[h - c:, w - c:] = _STASH_BG_BGR
    return out


def _canonical_bgr_flat(img_bgr, W, H):
    """
    Resize a BGR image to the canonical grid size (W*SLOT × H*SLOT) and return a
    flat float32 array of shape (W*SLOT * H*SLOT * 3,).

    Full BGR colour gives the NCC 3× the signal of grayscale and lets per-item
    colour signatures drive matching.  The [0,210] clip suppresses residual
    white UI artefacts that have no equivalent in the clean templates.

    NOTE: near-identical items that share a silhouette and differ only by a
    small coloured region + printed label (the whole stimulant-injector family)
    are NOT reliably separable here — the shared shape dominates NCC.  Chroma
    amplification was tried and made it worse (it turned the neutral body into
    matching noise / universal high correlation).  Exact separation of those
    needs the game's index.json hash → item mapping (see icon_cache notes).
    """
    tw = W * CANONICAL_PER_SLOT
    th = H * CANONICAL_PER_SLOT
    resized = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    resized = np.clip(resized, 0, 210)   # suppress residual white UI artefacts
    return resized.astype(np.float32).reshape(-1)


def _build_foreground_mask(raw, w, h):
    """
    Build a per-pixel foreground mask at canonical resolution, BGR-replicated
    flat (size W*SLOT * H*SLOT * 3).  Used to weight NCC so background pixels
    don't dominate matches on icons with mostly-transparent assets (Meds,
    screws, ornaments, etc.).

    Source preference:
      1. PNG alpha channel (alpha > 32) — most accurate.
      2. Saturation fallback for non-alpha sources: |max(BGR) - min(BGR)| > 12
         excludes near-uniform dark/grey areas.
    """
    tw = w * CANONICAL_PER_SLOT
    th = h * CANONICAL_PER_SLOT
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        mask2d = (alpha > 32).astype(np.uint8)
    else:
        bgr = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        bgr = bgr[:, :, :3].astype(np.int16)
        sat = bgr.max(axis=2) - bgr.min(axis=2)
        mask2d = (sat > 12).astype(np.uint8)
    mask2d = cv2.resize(mask2d, (tw, th), interpolation=cv2.INTER_NEAREST)
    # Zero out regions that get blanked by _apply_label_blank so they don't
    # contribute to NCC — keeps the mask consistent with the BGR templates.
    n = min(LABEL_BLANK_PX, th // 4)
    if n > 0:
        mask2d[:n, :] = 0
    c = min(CORNER_BLANK_PX, th // 3, tw // 3)
    if c > 0:
        mask2d[:c, tw - c:] = 0
        mask2d[th - c:, tw - c:] = 0
    # Replicate per-pixel mask across BGR channels and flatten
    mask3 = np.repeat(mask2d[:, :, None], 3, axis=2)
    return mask3.reshape(-1).astype(np.uint8)


def _template_from_bgra(src, W, H, tint):
    """
    From a BGRA (or BGR) source image at any resolution, produce a
    (canonical BGR vector, foreground mask) pair for footprint W×H, or None
    if the icon has no usable foreground.

    `src` is composited onto the item's rarity `tint` so anti-aliased edges
    match the real tinted stash cell.  The alpha channel drives the foreground
    mask; masked-out (background) pixels never contribute to NCC.
    """
    composited = _composite_on_dark(src, tint)
    composited = _apply_label_blank(composited)   # consistent with crop-side blanking
    canon = _canonical_bgr_flat(composited, W, H)   # clip + chroma-weighted Lab applied inside
    mask = _build_foreground_mask(src, W, H)
    if mask.sum() < 16:
        return None
    return canon, mask


def _load_template_source(item):
    """
    Return a BGRA (or BGR) source image for an item's template, preferring the
    correct-geometry tarkov.dev base-image (transparent, per-slot resolution).
    Falls back to the grid-image (background baked, no alpha) if base is absent.
    Returns (src_ndarray, has_alpha) or (None, False).
    """
    path = download_base_image(item['id'], item.get('baseImageLink'))
    if path:
        src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if src is not None and src.size and src.ndim == 3 and src.shape[2] == 4:
            return src, True
    # Fallback: grid image (baked background — weaker foreground mask)
    gpath = download_base_image(item['id'] + '-g', item.get('gridImageLink'))
    if gpath:
        src = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)
        if src is not None and src.size:
            return src, (src.ndim == 3 and src.shape[2] == 4)
    return None, False


def _rotations_for(src, W, H):
    """
    Yield (footprint_wh, rotated_flag, rotated_src) variants for a template.

    Always yields the native orientation.  For non-square items, also yields
    both 90° rotations (game rotates the icon pixels when placed rotated) into
    the transposed (H,W) footprint bucket.  Both CW and CCW are indexed because
    the in-game rotation direction is orientation-dependent; the id-aware margin
    check keeps the two same-item variants from rejecting each other.
    """
    yield (W, H), False, src
    if W != H:
        yield (H, W), True, cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        yield (H, W), True, cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _build_one_template(item):
    """
    Worker: download the base image and produce API-source template record(s).

    Returns a list of records
        (footprint_wh, item_id, name, source, rotated, canon_vec, mask)
    — one native plus (for non-square items) two rotated variants — or [].
    """
    W = item.get('width') or 1
    H = item.get('height') or 1
    tint = tint_for(item)
    src, _has_alpha = _load_template_source(item)
    if src is None:
        return []
    records = []
    for (fw, fh), rotated, rsrc in _rotations_for(src, W, H):
        made = _template_from_bgra(rsrc, fw, fh, tint)
        if made is None:
            continue
        canon, mask = made
        records.append(((fw, fh), item['id'], item['name'], 'api', rotated, canon, mask))
    return records


def _finalize_db(by_size_raw):
    """
    Convert {(w,h): {'ids': [...], 'names': [...], 'tmpls': [uint8 vec, ...],
                     'masks': [uint8 vec, ...]}} into runtime form with
    precomputed masked-NCC vectors.

    Each template is masked-mean-centred and unit-normed using ITS OWN mask,
    so at match time:
        scores = (tmpls_unit @ cell) / sqrt(masks @ cell² - (masks @ cell)² / mask_counts)
    correctly computes NCC restricted to each template's foreground region.
    """
    out = {}
    for (w, h), b in by_size_raw.items():
        if not b['tmpls']:
            continue
        tmpls = np.vstack(b['tmpls']).astype(np.float32)    # (N, D)
        masks = np.vstack(b['masks']).astype(np.float32)     # (N, D), 0/1
        mask_counts = masks.sum(axis=1)                      # (N,)
        # Per-template masked mean = Σ(mask * tmpl) / Σ(mask)
        masked_means = (masks * tmpls).sum(axis=1) / np.maximum(mask_counts, 1.0)  # (N,)
        # Mean-centre INSIDE the mask, zero OUTSIDE — multiply-by-mask handles both
        centered = (tmpls - masked_means[:, None]) * masks
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        tmpls_unit = (centered / norms).astype(np.float32)
        n = len(b['ids'])
        out[(w, h)] = {
            'ids':         b['ids'],
            'names':       b['names'],
            'sources':     b.get('sources', ['api'] * n),
            'rotated':     b.get('rotated', [False] * n),
            'tmpls_unit':  tmpls_unit,
            'masks':       masks.astype(np.float32),
            'mask_counts': mask_counts.astype(np.float32),
        }
    return out


def _append_record(by_size_raw, wh, item_id, name, source, rotated, vec, mask):
    b = by_size_raw.setdefault(wh, {'ids': [], 'names': [], 'sources': [],
                                    'rotated': [], 'tmpls': [], 'masks': []})
    b['ids'].append(item_id)
    b['names'].append(name)
    b['sources'].append(source)
    b['rotated'].append(bool(rotated))
    b['tmpls'].append(vec)
    b['masks'].append(mask)


def build_icon_db(price_cache, progress_cb=None, workers=24, use_cache=True):
    """
    Build a size-bucketed database of BGR colour templates ready for masked NCC.

    Two template sources are merged:
      1. The game's local icon cache (pixel-perfect, includes modded builds) —
         associated to item IDs by icon_cache.build_cache_templates().  Preferred.
      2. tarkov.dev base-images composited onto the rarity tint — full catalog
         coverage, including items this account has never rendered.

    Non-square items also get 90°-rotated variants so rotated placements match.
    Persists to ICON_DB_PATH (npz).  Returns runtime dict.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Only build templates for match targets — dropping ammo/guns/presets/cases
    # removes ~1k distractors (esp. near-identical ammo) and can't be matched.
    all_items = price_cache.get('items', [])
    items = [it for it in all_items if is_target_item(it)]
    print(f"[icon_db] target items: {len(items)} of {len(all_items)} "
          f"(excluded {len(all_items) - len(items)} ammo/gun/preset/container)")
    total = len(items)
    done = 0
    by_size_raw = {}

    # --- Source 1: game icon cache (visually associated) ---------------------
    cache_ids = set()
    if use_cache:
        try:
            import icon_cache
            cache_recs = icon_cache.build_cache_templates(
                items, _template_from_bgra, tint_for, _load_template_source)
            for (wh, item_id, name, rotated, vec, mask) in cache_recs:
                _append_record(by_size_raw, wh, item_id, name, 'cache', rotated, vec, mask)
                cache_ids.add(item_id)
            print(f"[icon_db] cache templates: {len(cache_recs)} "
                  f"({len(cache_ids)} distinct items)")
        except Exception as e:
            print(f"[icon_db] icon-cache pass skipped: {e}")

    # --- Source 2: tarkov.dev base-images (full catalog) ---------------------
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_build_one_template, it) for it in items]
        for fut in as_completed(futures):
            done += 1
            if progress_cb and done % 25 == 0:
                progress_cb(done, total)
            try:
                records = fut.result()
            except Exception:
                continue
            for (wh, item_id, name, source, rotated, vec, mask) in records:
                _append_record(by_size_raw, wh, item_id, name, source, rotated, vec, mask)
    if progress_cb:
        progress_cb(done, total)

    save_icon_db(by_size_raw)
    return _finalize_db(by_size_raw)


def save_icon_db(by_size_raw):
    """Save raw uint8 BGR templates + foreground masks + metadata to .npz."""
    meta = {'_version': DB_VERSION}   # version stamp — mismatches trigger rebuild
    saves = {}
    for (w, h), b in by_size_raw.items():
        if not b['tmpls']:
            continue
        key = f'{w}x{h}'
        n = len(b['ids'])
        saves[f'T_{key}'] = np.vstack(b['tmpls']).astype(np.uint8)
        saves[f'M_{key}'] = np.vstack(b['masks']).astype(np.uint8)
        meta[key] = {
            'ids':     b['ids'],
            'names':   b['names'],
            'sources': b.get('sources', ['api'] * n),
            'rotated': [bool(x) for x in b.get('rotated', [False] * n)],
        }
    meta_bytes = json.dumps(meta).encode('utf-8')
    saves['META'] = np.frombuffer(meta_bytes, dtype=np.uint8)
    np.savez_compressed(ICON_DB_PATH, **saves)


def load_icon_db():
    """Load the persisted icon DB, or None if missing/invalid/outdated."""
    if not os.path.exists(ICON_DB_PATH):
        return None
    try:
        z = np.load(ICON_DB_PATH, allow_pickle=False)
        meta = json.loads(bytes(z['META']).decode('utf-8'))
        db_ver = meta.get('_version', 1)
        if db_ver != DB_VERSION:
            print(f"[icon_db] version mismatch (file={db_ver}, code={DB_VERSION}) — "
                  "rebuild required (click Build Icon DB)")
            return None
        by_size_raw = {}
        for key, info in meta.items():
            if key.startswith('_'):      # skip internal fields like _version
                continue
            w, h = (int(x) for x in key.split('x'))
            arr = z[f'T_{key}']   # (N, D) uint8
            marr = z[f'M_{key}']  # (N, D) uint8
            n = len(info['ids'])
            by_size_raw[(w, h)] = {
                'ids':     info['ids'],
                'names':   info['names'],
                'sources': info.get('sources', ['api'] * n),
                'rotated': info.get('rotated', [False] * n),
                'tmpls':   [arr[i] for i in range(arr.shape[0])],
                'masks':   [marr[i] for i in range(marr.shape[0])],
            }
        return _finalize_db(by_size_raw)
    except Exception as e:
        global _icon_db_error
        _icon_db_error = f'Icon DB load failed: {e}'
        print(f"[icon_db] load failed: {e}")
        return None


# Global in-memory DB (populated on first use)
_icon_db = None
_icon_db_error = None
_icon_db_lock = threading.Lock()

def get_icon_db():
    """Return the runtime icon DB, loading from disk if needed."""
    global _icon_db
    if _icon_db is None:
        with _icon_db_lock:
            if _icon_db is None:
                _icon_db = load_icon_db()
    return _icon_db


# ---------------------------------------------------------------------------
# Grid-cell identification using the canonical-template DB
# ---------------------------------------------------------------------------

def _cell_block(img_bgr, col, row, W, H, grid, pad=0):
    """Pixel rect of a W×H cell block starting at (col, row), optionally padded."""
    x = grid['origin_x'] + col * grid['cell_w'] - pad
    y = grid['origin_y'] + row * grid['cell_h'] - pad
    w = W * grid['cell_w'] + 2 * pad
    h = H * grid['cell_h'] + 2 * pad
    sh, sw = img_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(sw, x + w), min(sh, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


def _cell_is_empty(img_bgr, col, row, grid):
    """
    Empty stash cells are pure background — very uniform dark pixels.
    Tarkov background ≈ (38,42,44) BGR → gray ≈ 41.
    An item will raise either the std (texture/shape) OR have pixels
    meaningfully above the background level, even if the item is dark.
    """
    crop = _cell_block(img_bgr, col, row, 1, 1, grid)
    if crop is None or crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 8 and gray.shape[1] > 8:
        gray = gray[4:-4, 4:-4]
    # Pixels meaningfully above the dark stash background (~41 gray)
    above_bg = float((gray > 58).mean())
    # Empty = essentially uniform AND almost nothing above background
    return float(gray.std()) < 5.0 and above_bg < 0.04


def _canonical_cell_vec(img_bgr, col, row, W, H, grid):
    """
    Extract the (W×H) cell block from the screenshot and return a flat
    float32 BGR vector at canonical resolution, ready for masked NCC.

    Preprocessing:
      1. _apply_label_blank — overwrites top rows + right corners with bg
         colour to remove Tarkov's label, FiR ✓, and stack-count overlays.
      2. Clip [0, 210] — suppresses residual white UI artefacts that have
         no equivalent in the clean icon templates.

    Per-template masked-mean-centring + unit-norming is done at match time
    inside `identify_items_by_icon`, since each template uses its own mask.
    """
    crop = _cell_block(img_bgr, col, row, W, H, grid)
    if crop is None or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    crop = _apply_label_blank(crop)
    canon = _canonical_bgr_flat(crop, W, H)   # clip + chroma-weighted Lab applied inside
    return canon


OCR_LABEL_BASE   = 78   # rapidfuzz WRatio floor to consider an OCR name candidate at all
OCR_AGREE_CUTOFF = 78   # OCR agrees with NCC → accept (mutual confirmation)
OCR_OVER_CUTOFF  = 84   # OCR overrides a *different* NCC identity → needs this
OCR_SHORT_OVER   = 99   # short OCR tokens (≤3 chars) overriding NCC need near-exact (blocks 'Li'→Splint)


def _ocr_cell_label(img_bgr, col, row, W, grid):
    """
    OCR the item name the game prints across the top of a cell (single line,
    left-aligned).  This is the signal NCC lacks: it separates same-shape items
    like L1 / SJ12 / eTG-c that share one injector silhouette.
    Returns cleaned text (leading UI junk stripped) or ''.
    """
    x = grid['origin_x'] + col * grid['cell_w']
    y = grid['origin_y'] + row * grid['cell_h']
    w = W * grid['cell_w']
    lh = max(12, int(round(grid['cell_h'] * 0.30)))
    sh, sw = img_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(sw, x + w), min(sh, y + lh)
    if x2 - x1 < 8 or y2 - y1 < 6:
        return ''
    if not tesseract_available():
        return ''
    strip = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    strip = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    try:
        txt = pytesseract.image_to_string(strip, config='--psm 7')
    except Exception:
        return ''
    txt = txt.strip().replace('\n', ' ')
    # Strip leading non-alphanumeric OCR junk (FiR tick / tint speckle) but keep
    # inner punctuation like CALOK-B, AHF1-M, #FireKlean.
    while txt and not (txt[0].isalnum() or txt[0] == '#'):
        txt = txt[1:]
    return txt.strip()


def build_label_matcher(price_cache):
    """
    Build a fuzzy matcher from OCR'd cell text → item over the whole target
    catalogue (not ammo/gun/preset/container).  Identity is decided by the
    printed name; the footprint is then taken from the item's own catalogue size
    (see identify_items_by_icon) rather than NCC's guessed size — so a 1×2 MGT
    can't be mislabelled a 2×1 and over-claim its neighbour.

    Restricting candidates to the detected footprint (plus 1×1, the common
    fallback when NCC over-sizes) keeps short tokens like 'L1' from colliding
    with different-size items and avoids same-name cross-size ambiguity
    (CAT tourniquet vs Cat figurine).

    Returns matcher(text, fw, fh) -> (item_id, score, name, native_w, native_h) | None.
    """
    from rapidfuzz import process, fuzz
    by_size = {}   # (w,h) -> (choices, ids, names)
    for it in price_cache.get('items', []):
        if not is_target_item(it):
            continue
        wh = (it.get('width') or 1, it.get('height') or 1)
        b = by_size.setdefault(wh, ([], [], []))
        for s in {it.get('shortName') or '', it.get('name') or ''}:
            if s:
                b[0].append(s.lower())
                b[1].append(it['id'])
                b[2].append(it['name'])

    def matcher(text, fw, fh):
        if not text or len(text) < 2:
            return None
        best = None
        for wh in {(fw, fh), (fh, fw)}:
            b = by_size.get(wh)
            if not b:
                continue
            r = process.extractOne(text.lower(), b[0], scorer=fuzz.WRatio,
                                   score_cutoff=OCR_LABEL_BASE)
            if r and (best is None or r[1] > best[1]):
                _, score, idx = r
                best = (b[1][idx], score, b[2][idx], wh[0], wh[1])
        return best

    return matcher


def _min_score_for_size(W, H):
    """Adaptive NCC threshold: looser for tiny sparse icons, tighter for big rich ones."""
    area = W * H
    if area == 1:
        return 0.35
    if area <= 2:
        return 0.40
    return 0.50


OCC_FRAC          = 0.72   # a footprint is a size candidate only if this fraction of its cells are occupied
CACHE_SRC_BONUS   = 0.02   # adjust-score bias favouring exact game-cache templates over API templates


def _build_occupancy(img_bgr, grid, n_rows, n_cols):
    """Boolean [n_rows, n_cols] occupancy map (True = item present in cell)."""
    occ = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            occ[r, c] = not _cell_is_empty(img_bgr, c, r, grid)
    return occ


def _footprint_fits(occ, claimed, col, row, W, H, n_cols, n_rows):
    """
    Return the occupied fraction of the W×H block at (col,row) if it is fully
    unclaimed and in-bounds, else -1.  Used to gate which sizes are even tested,
    so a 2×1 template can't claim a footprint whose second cell is empty.
    """
    if col + W > n_cols or row + H > n_rows:
        return -1.0
    occ_cells = 0
    for dc in range(W):
        for dr in range(H):
            if claimed[row + dr, col + dc]:
                return -1.0
            if occ[row + dr, col + dc]:
                occ_cells += 1
    return occ_cells / float(W * H)


def _best_with_margin(scores, ids):
    """
    Return (best_index, best_score, id_aware_margin).

    The margin compares the top score to the best score belonging to a
    *different* item ID, so near-duplicate rotations/presets of the same item
    never reject each other.
    """
    i = int(np.argmax(scores))
    top1 = float(scores[i])
    top1_id = ids[i]
    top2 = 0.0
    for j in np.argsort(scores)[::-1]:
        if ids[int(j)] != top1_id:
            top2 = float(scores[int(j)])
            break
    return i, top1, top1 - top2


def validate_grid(grid, ui_scale=1.0, tol=0.12):
    """
    True if the detected cell pitch is consistent with EFT's 63px/slot @1080p
    (scaled by ui_scale).  Rejects autocorrelation locks onto 2× the true pitch
    or on UI chrome.
    """
    if not grid:
        return False
    expected = GRID_PITCH_1080P * ui_scale
    lo, hi = expected * (1 - tol) - 1, expected * (1 + tol) + 1
    return lo <= grid['cell_w'] <= hi and lo <= grid['cell_h'] <= hi


def resolve_grid(img_bgr, settings, persist_fn=None):
    """
    Detect the stash grid, validate it against the expected 63px pitch, and fall
    back to the last-good persisted grid (settings['grid']) or the flat cell_size
    when detection is noisy.  On a fresh valid detection, persist it via
    persist_fn(grid) so one bad frame can't derail future scans.
    """
    ui_scale = settings.get('ui_scale', 1.0)
    grid = detect_stash_grid(img_bgr)
    if validate_grid(grid, ui_scale):
        if persist_fn:
            persist_fn(grid)
        return grid, 'detected'
    saved = settings.get('grid')
    if validate_grid(saved, ui_scale):
        return dict(saved), 'persisted'
    cs = settings.get('cell_size', 64)
    return {'cell_w': cs, 'cell_h': cs, 'origin_x': 0, 'origin_y': 0}, 'fallback'


def _masked_ncc_scores(cell_vec, bucket):
    """
    Per-template masked NCC against all templates in a size bucket.

    For each template t with mask m_t (per-pixel 0/1, BGR-replicated):
        cm_t        = (m_t · cell)  / Σ(m_t)                       # cell mean inside m_t
        num_t       = (m_t * (cell - cm_t)) · tmpl_unit_t          # numerator
                    = tmpl_unit_t · cell        (since tmpl_unit·m_t = 0 by construction)
        denom_t²    = Σ m_t * (cell - cm_t)²
                    = (m_t · cell²) - (m_t · cell)² / Σ(m_t)
        score_t     = num_t / denom_t

    Returns (N,) float32 NCC scores.
    """
    masks       = bucket['masks']         # (N, D) float32
    mask_counts = bucket['mask_counts']   # (N,)   float32
    tmpls_unit  = bucket['tmpls_unit']    # (N, D) float32

    cell_sq    = cell_vec * cell_vec
    sums_c     = masks @ cell_vec               # (N,)
    sums_c2    = masks @ cell_sq                # (N,)
    var_c      = sums_c2 - (sums_c * sums_c) / np.maximum(mask_counts, 1.0)
    denom      = np.sqrt(np.maximum(var_c, 1e-6))
    num        = tmpls_unit @ cell_vec          # (N,)
    return num / denom


def identify_items_by_icon(img_bgr, grid, icon_db, min_score=ICON_MATCH_MIN_SCORE,
                           label_matcher=None, progress_cb=None):
    """
    Footprint-first identification, with optional OCR-label fusion.

    1. Build a per-cell occupancy map from the image.
    2. Walk occupied, unclaimed cells row-major.  For each anchor, only test
       (W,H) sizes whose footprint is actually occupied (measured, not guessed),
       so a wrong multi-cell size can't over-claim empty neighbours.
    3. Masked NCC against every template of that size (rotations + presets
       included).  The winning (item, size) needs to clear the size-adaptive
       threshold AND beat the best *different-item* score by >NCC_MARGIN_MIN.
    4. Ties within NCC_MARGIN prefer exact game-cache templates over API ones.
    5. If `label_matcher` is given, OCR the game's printed name across the top of
       the footprint.  A confident name match is AUTHORITATIVE for identity
       (NCC keeps the footprint/rotation) — this is what separates same-shape
       items (L1 vs SJ12 vs eTG-c) that NCC alone cannot.

    Detections carry `rotated` and `source` (ncc / cache / api / ocr).
    """
    if not icon_db:
        return []

    sh, sw = img_bgr.shape[:2]
    cw, ch = grid['cell_w'], grid['cell_h']
    ox, oy = grid['origin_x'], grid['origin_y']
    n_cols = max(0, (sw - ox) // cw)
    n_rows = max(0, (sh - oy) // ch)
    if n_cols == 0 or n_rows == 0:
        return []

    occ = _build_occupancy(img_bgr, grid, n_rows, n_cols)
    claimed = np.zeros((n_rows, n_cols), dtype=bool)

    # Test larger footprints first so a correct multi-cell item claims its cells
    # before any 1×1 sub-region can.
    sizes = sorted(icon_db.keys(), key=lambda wh: -wh[0] * wh[1])

    detections = []
    n_ambig = 0
    occ_total = int(occ.sum())
    # Row-major cumulative count of occupied cells — monotonic progress even
    # when multi-cell claims let the walk skip ahead.
    occ_cum = np.cumsum(occ.reshape(-1))

    for row in range(n_rows):
        for col in range(n_cols):
            if claimed[row, col] or not occ[row, col]:
                continue
            if progress_cb:
                progress_cb(int(occ_cum[row * n_cols + col]), occ_total)

            best = None   # (adj, sc, margin, i, W, H)
            for (W, H) in sizes:
                frac = _footprint_fits(occ, claimed, col, row, W, H, n_cols, n_rows)
                if frac < OCC_FRAC and not (W == 1 and H == 1):
                    continue
                if frac < 0:
                    continue

                vec = _canonical_cell_vec(img_bgr, col, row, W, H, grid)
                if vec is None:
                    continue
                bucket = icon_db[(W, H)]
                scores = _masked_ncc_scores(vec, bucket)
                i, sc, margin = _best_with_margin(scores, bucket['ids'])

                # Size preference + exact-cache-source bias.
                adj = sc + (W * H - 1) * 0.012
                if bucket['sources'][i] == 'cache':
                    adj += CACHE_SRC_BONUS
                if best is None or adj > best[0]:
                    best = (adj, sc, margin, i, W, H, bucket)

            # Footprint / NCC identity (may be None if no vector could be built).
            if best is not None:
                _, sc, margin, i, W, H, bucket = best
                ncc_id, ncc_name = bucket['ids'][i], bucket['names'][i]
                rotated, ncc_src = bool(bucket['rotated'][i]), bucket['sources'][i]
            else:
                W = H = 1; sc = margin = 0.0
                ncc_id = ncc_name = None; rotated = False; ncc_src = 'ncc'

            # OCR-label fusion — the printed name is authoritative for identity,
            # but tiered so a short OCR token can't partial-match its way over a
            # different NCC identity (e.g. 'Li' → 'Splint').
            ocr_hit = None
            if label_matcher is not None:
                text = _ocr_cell_label(img_bgr, col, row, W, grid)
                cand = label_matcher(text, W, H)
                if cand is not None:
                    o_id, o_sc, o_name, o_w, o_h = cand
                    agrees = ncc_id is not None and o_id == ncc_id
                    if agrees:
                        need = OCR_AGREE_CUTOFF
                    elif len(text) <= 3:
                        need = OCR_SHORT_OVER
                    else:
                        need = OCR_OVER_CUTOFF
                    if o_sc >= need:
                        ocr_hit = cand

            if ocr_hit is not None:
                item_id, ocr_sc, name, _o_w, _o_h = ocr_hit
                detections.append({
                    'col': col, 'row': row, 'W': W, 'H': H,
                    'item_id': item_id, 'name': name,
                    'rotated': rotated,
                    'source':  'ocr' if (ncc_id is None or item_id != ncc_id) else 'ncc+ocr',
                    'score':   round(float(ocr_sc), 1),
                })
                claimed[row:row + H, col:col + W] = True
                continue

            # Over-sizing rescue: NCC's area bias can pick W>1 for a 1×1 item,
            # which garbles the OCR crop.  If nothing matched, retry OCR at 1×1.
            if (label_matcher is not None and (W > 1 or H > 1)
                    and _footprint_fits(occ, claimed, col, row, 1, 1, n_cols, n_rows) >= 0):
                text1 = _ocr_cell_label(img_bgr, col, row, 1, grid)
                cand1 = label_matcher(text1, 1, 1)
                if cand1 is not None:
                    o_id, o_sc, o_name, _w, _h = cand1
                    need = OCR_SHORT_OVER if len(text1) <= 3 else OCR_OVER_CUTOFF
                    if o_sc >= need:
                        detections.append({
                            'col': col, 'row': row, 'W': 1, 'H': 1,
                            'item_id': o_id, 'name': o_name,
                            'rotated': False, 'source': 'ocr',
                            'score': round(float(o_sc), 1),
                        })
                        claimed[row, col] = True
                        continue

            if best is None:
                continue
            size_min = _min_score_for_size(W, H)
            if sc >= size_min and margin >= NCC_MARGIN_MIN:
                detections.append({
                    'col': col, 'row': row, 'W': W, 'H': H,
                    'item_id': ncc_id, 'name': ncc_name,
                    'rotated': rotated,
                    'source':  ncc_src,
                    'score':   round(sc * 100, 1),
                })
                claimed[row:row + H, col:col + W] = True
            elif sc >= size_min and margin < NCC_MARGIN_MIN:
                n_ambig += 1
                print(f"[ncc] AMBIG ({col:2d},{row:2d}) {W}×{H}  "
                      f"sc={sc:.3f} m={margin:.3f}  '{ncc_name}'")

    if detections:
        scores = [d['score'] for d in detections]
        print(f"[ncc] matched {len(detections)} items ({n_ambig} ambiguous) | "
              f"scores {min(scores):.1f}–{max(scores):.1f}%  avg {sum(scores)/len(scores):.1f}%")
    return detections



def prefetch_keep_list_icons(keep_list, price_index, scale=2.0):
    """Download icons for all keep-list items using tarkov.dev iconLink."""
    downloaded = 0
    for cat in keep_list['categories']:
        for item in cat['items']:
            name_lower = item['name'].lower()
            item_data = price_index.get(name_lower)
            if not item_data:
                # Try aliases
                for alias in item.get('aliases', []):
                    item_data = price_index.get(alias.lower())
                    if item_data:
                        break
            if item_data and item_data.get('iconLink'):
                path = download_icon(item_data['id'], item_data['iconLink'])
                if path:
                    downloaded += 1
    return downloaded


def best_trader_price(item_data):
    """Return (trader_name, priceRUB) for the highest trader sell offer."""
    best = (None, 0)
    for sf in item_data.get('sellFor', []):
        vendor = sf.get('vendor', {}).get('name', '')
        if vendor.lower() == 'flea market':
            continue
        p = sf.get('priceRUB', 0) or 0
        if p > best[1]:
            best = (vendor, p)
    return best

def calc_flea_fee(base_price, listing_price):
    """Tarkov flea market listing fee formula."""
    if not base_price or not listing_price:
        return 0
    q0, q = base_price, listing_price
    fee = q0 * 0.03 * (4 ** math.log10(q0 / q)) + q * 0.03 * (4 ** math.log10(q / q0))
    return round(fee)

def price_420(target):
    """Floor target to nearest price ending in 420. Falls back to target-1."""
    if target <= 420:
        return max(1, target - 1)
    rem = target % 1000
    p = (target - rem + 420) if rem >= 420 else (target - rem - 580)
    return p if p >= 420 else max(1, target - 1)

def sell_recommendation(item_data):
    """
    Returns dict with trader, flea, and recommendation.
    Flea is recommended only if net-after-fee exceeds trader by FLEA_MIN_PROFIT.
    """
    trader_name, trader_price = best_trader_price(item_data)
    base_price   = item_data.get('basePrice') or 0
    low24h       = item_data.get('low24hPrice') or 0
    avg24h       = item_data.get('avg24hPrice') or 0
    flea_ref     = low24h or avg24h  # prefer lowest current listing

    rec = {
        'trader_name':  trader_name,
        'trader_price': trader_price,
        'flea_list':    None,
        'flea_net':     None,
        'recommend':    'trader',
        'reason':       '',
    }

    if not flea_ref or not trader_price:
        rec['reason'] = 'No flea data' if not flea_ref else 'No trader data'
        return rec

    flea_list = price_420(flea_ref)
    flea_fee  = calc_flea_fee(base_price, flea_list)
    flea_net  = flea_list - flea_fee
    rec['flea_list'] = flea_list
    rec['flea_net']  = flea_net

    if flea_net - trader_price >= FLEA_MIN_PROFIT:
        rec['recommend'] = 'flea'
        rec['reason']    = f'+{(flea_net - trader_price):,} over trader after fees'
    else:
        rec['reason'] = f'Flea net {flea_net:,} not {FLEA_MIN_PROFIT//1000}k+ above trader'

    return rec


# ---------------------------------------------------------------------------
# Default data
# ---------------------------------------------------------------------------

# F9 conflicts with EFT's own binds and popular overlay tools, so the default
# is a modifier combo nothing else claims.
DEFAULT_HOTKEY = '<ctrl>+<shift>+s'

def default_settings():
    return {
        'region': None, 'monitor': 0, 'hotkey': DEFAULT_HOTKEY, 'prestige': 3,
        'scan_countdown': 3,   # seconds before the manual Scan button captures (0 = instant)
        'cell_size': 64,    # fallback: pixels per slot when grid auto-detect fails
        'icon_scale': 2.0,  # scale applied to tarkov.dev icons for template matching
        'ui_scale': 1.0,    # EFT UI scale — expected grid pitch = 63 * ui_scale px/slot
        'icon_cache_path': None,  # override for the EFT icon-cache folder (auto-discovered if null)
        'grid': None,       # last-good detected grid, persisted so a noisy frame can't derail a scan
    }

def default_keep_list():
    return {
        'categories': [
            {
                'id': 'kappa',
                'label': 'Kappa (Collector)',
                'items': [
                    # --- Still needed ---
                    {'id': 'tea',         'name': '42 Signature Blend English Tea', 'aliases': ['42 Sig', 'English Tea'],             'acquired': False},
                    {'id': 'axe',         'name': 'Antique axe',                    'aliases': ['Antique axe'],                       'acquired': False},
                    {'id': 'armband',     'name': 'Armband (Evasion)',               'aliases': ['Armband', 'Evasion'],                'acquired': False},
                    {'id': 'bear_buddy',  'name': 'BEAR Buddy plush toy',            'aliases': ['BEAR Buddy'],                        'acquired': False},
                    {'id': 'drd',         'name': 'DRD body armor',                  'aliases': ['DRD'],                               'acquired': False},
                    {'id': 'phone',       'name': 'Golden 1GPhone smartphone',        'aliases': ['1GPhone', 'Golden phone'],           'acquired': False},
                    {'id': 'loot_lord',   'name': 'Loot Lord plushie',               'aliases': ['Loot Lord'],                         'acquired': False},
                    {'id': 'wz_wallet',   'name': 'WZ Wallet',                        'aliases': ['WZ Wallet'],                         'acquired': False},
                    {'id': 'dumbbell',    'name': 'Mazoni golden dumbbell',           'aliases': ['Mazoni', 'Dumbbell'],                'acquired': False},
                    {'id': 'splint',      'name': 'Tigzresq splint',                  'aliases': ['Tigzresq', 'Splint'],                'acquired': False},
                    # --- Acquired ---
                    {'id': 'firesteel',   'name': 'Old firesteel',                    'aliases': ['Firesteel'],                         'acquired': True},
                    {'id': 'book',        'name': 'Battered antique book',            'aliases': ['Book', 'Battered book'],             'acquired': True},
                    {'id': 'fireklean',   'name': '#FireKlean gun lube',              'aliases': ['FireKlea', 'FireKlean'],             'acquired': True},
                    {'id': 'rooster',     'name': 'Golden rooster figurine',          'aliases': ['Rooster'],                           'acquired': True},
                    {'id': 'badge',       'name': 'Silver Badge',                     'aliases': ['Badge'],                             'acquired': True},
                    {'id': 'beard_oil',   'name': "Deadlyslob's beard oil",           'aliases': ['BeardOil', 'Beard Oil'],             'acquired': True},
                    {'id': 'mayo',        'name': 'Jar of DevilDog mayo',             'aliases': ['Mayo', 'DevilDog'],                  'acquired': True},
                    {'id': 'sprats',      'name': 'Can of sprats',                    'aliases': ['Sprats'],                            'acquired': True},
                    {'id': 'mustache',    'name': 'Fake mustache',                    'aliases': ['Mustache'],                          'acquired': True},
                    {'id': 'kotton',      'name': 'Kotton beanie',                    'aliases': ['Kotton'],                            'acquired': True},
                    {'id': 'raven',       'name': 'Raven figurine',                   'aliases': ['Raven'],                             'acquired': True},
                    {'id': 'pestily',     'name': 'Pestily plague mask',              'aliases': ['Pestily'],                           'acquired': True},
                    {'id': 'shroud',      'name': 'Shroud half-mask',                 'aliases': ['Shroud'],                            'acquired': True},
                    {'id': 'drlupo',      'name': "Can of Dr. Lupo's coffee beans",   'aliases': ["DrLupo's", 'Dr Lupo'],               'acquired': True},
                    {'id': 'veritas',     'name': 'Veritas guitar pick',              'aliases': ['Veritas'],                           'acquired': True},
                    {'id': 'ratcola',     'name': 'Can of RatCola soda',              'aliases': ['RatCola'],                           'acquired': True},
                    {'id': 'smoke',       'name': 'Smoke balaclava',                  'aliases': ['Smoke'],                             'acquired': True},
                    {'id': 'lvndmark',    'name': "LVNDMARK's rat poison",            'aliases': ['LVNDMARK', 'Polson', 'Rat poison'],  'acquired': True},
                    {'id': 'forklift',    'name': 'Missam forklift key',              'aliases': ['Missam'],                            'acquired': True},
                    {'id': 'vhs',         'name': 'Video cassette (Cyborg Killer)',   'aliases': ['VHS', 'Cyborg Killer'],              'acquired': True},
                    {'id': 'bakeezy',     'name': 'BakeEzy cook book',                'aliases': ['BakeEzy'],                           'acquired': True},
                    {'id': 'johnb',       'name': 'JohnB Liquid DNB glasses',         'aliases': ['JohnB'],                             'acquired': True},
                    {'id': 'baddie',      'name': "Baddie's red beard",               'aliases': ['Baddie'],                            'acquired': True},
                    {'id': 'gingy',       'name': 'Gingy keychain',                   'aliases': ['Gingy'],                             'acquired': True},
                    {'id': 'egg',         'name': 'Golden egg',                        'aliases': ['Egg'],                               'acquired': True},
                    {'id': 'pass_',       'name': 'Press pass (NoiceGuy)',             'aliases': ['Pass', 'NoiceGuy'],                  'acquired': True},
                    {'id': 'axel',        'name': 'Axel parrot figurine',              'aliases': ['Axel'],                              'acquired': True},
                    {'id': 'glorious',    'name': 'Glorious E armored mask',           'aliases': ['Glorious'],                          'acquired': True},
                    {'id': 'inseq',       'name': 'Inseq gas pipe wrench',             'aliases': ['Inseq'],                             'acquired': True},
                    {'id': 'viibiin',     'name': 'Viibiin sneaker',                   'aliases': ['Viiblin', 'Viibiin'],                'acquired': True},
                    {'id': 'tamatthi',    'name': 'Tamatthi kunai knife replica',      'aliases': ['Tamatthi'],                          'acquired': True},
                    {'id': 'nut_sack',    'name': 'Nut Sack balaclava',                'aliases': ['Nut Sack', 'NutSack'],               'acquired': True},
                    {'id': 'domontovich', 'name': 'Domontovich ushanka hat',           'aliases': ['Domontovich'],                       'acquired': True},
                ]
            },
            {
                'id': 'tasks',
                'label': 'Task Items (manual)',
                'items': [
                    {'id': 'fleece',       'name': 'Fleece fabric',                'aliases': ['Fleece'],           'count': 10, 'task': 'Ragman - Textile Part 2',    'acquired': False},
                    {'id': 'cordura',      'name': 'Cordura polyamide fabric',     'aliases': ['Cordura'],          'count': 10, 'task': 'Ragman - Textile Part 2',    'acquired': False},
                    {'id': 'kektape',      'name': 'KEKTAPE duct tape',            'aliases': ['KEK', 'KEKTAPE'],   'count': 5,  'task': 'Ragman - Textile Part 2',    'acquired': False},
                    {'id': 'bear_tag',     'name': 'Dogtag BEAR',                  'aliases': ['BEAR'],             'count': 20, 'task': 'Peacekeeper - Trophies',     'acquired': False},
                    {'id': 'usec_tag',     'name': 'Dogtag USEC',                  'aliases': ['USEC'],             'count': 20, 'task': 'Peacekeeper - Trophies',     'acquired': False},
                    {'id': 'bear_tag6',    'name': 'Dogtag BEAR (Punisher 6)',     'aliases': [],                   'count': 7,  'task': 'Prapor - Punisher Part 6',   'acquired': False},
                    {'id': 'usec_tag6',    'name': 'Dogtag USEC (Punisher 6)',     'aliases': [],                   'count': 7,  'task': 'Prapor - Punisher Part 6',   'acquired': False},
                    {'id': 'cult_knife',   'name': 'Cultist knife',                'aliases': ['Cultist knife'],    'count': 12, 'task': 'Skier - Night Sweep',        'acquired': False},
                    {'id': 'epsilon',      'name': 'Secure container Epsilon',     'aliases': ['Epsilon'],          'count': 1,  'task': 'Fence - The Choice',         'acquired': False},
                    {'id': 'vodka',        'name': 'Bottle of Tarkovskaya vodka',  'aliases': ['Vodka'],            'count': 10, 'task': 'Ragman - Booze',             'acquired': False},
                    {'id': 'whiskey',      'name': 'Bottle of Dan Jackiel whiskey','aliases': ['Whiskey'],          'count': 10, 'task': 'Ragman - Booze',             'acquired': False},
                    {'id': 'beer',         'name': 'Bottle of Pevko Light beer',   'aliases': ['Pevko', 'Beer'],    'count': 20, 'task': 'Ragman - Booze',             'acquired': False},
                    {'id': 'water',        'name': 'Canister purified water',      'aliases': ['Water'],            'count': 3,  'task': 'Ragman - Booze',             'acquired': False},
                    {'id': 'fig_bear',     'name': 'BEAR operative figurine',      'aliases': [],                   'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_usec',     'name': 'USEC operative figurine',      'aliases': [],                   'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_killa',    'name': 'Killa figurine',               'aliases': ['Killa'],            'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_reshala',  'name': 'Reshala figurine',             'aliases': ['Reshala'],          'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_ryzhy',    'name': 'Ryzhy figurine',               'aliases': ['Ryzhy'],            'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_scav',     'name': 'Scav figurine',                'aliases': ['Scav'],             'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_tagilla',  'name': 'Tagilla figurine',             'aliases': ['Tagilla'],          'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_cultist',  'name': 'Cultist figurine',             'aliases': ['Cultist'],          'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_den',      'name': 'Den figurine',                 'aliases': ['Den'],              'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                    {'id': 'fig_mutkevich','name': 'Politician Mutkevich figurine','aliases': ['Mutkevich'],        'count': 1,  'task': 'Ragman - New Beginning',     'acquired': False},
                ]
            }
        ]
    }


# ---------------------------------------------------------------------------
# Kappa (Collector) list — synced from the wiki
# ---------------------------------------------------------------------------

from html.parser import HTMLParser

class _WikiTableParser(HTMLParser):
    """Collect each top-level <table> as a list of rows of stripped cell text.
    Nested tables are parsed but discarded so their text can't pollute cells."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._stack = []   # one dict per open <table>

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._stack.append({'rows': [], 'row': None, 'cell': None})
        elif self._stack:
            t = self._stack[-1]
            if tag == 'tr':
                t['row'] = []
                t['cell'] = None
            elif tag in ('td', 'th') and t['row'] is not None:
                t['cell'] = []
            elif tag == 'br' and t['cell'] is not None:
                t['cell'].append(' ')

    def handle_endtag(self, tag):
        if not self._stack:
            return
        t = self._stack[-1]
        if tag == 'table':
            done = self._stack.pop()
            if done['cell'] is not None and done['row'] is not None:
                done['row'].append(' '.join(''.join(done['cell']).split()))
            if done['row']:
                done['rows'].append(done['row'])
            if not self._stack:          # nested tables are dropped
                self.tables.append(done['rows'])
        elif tag == 'tr' and t['row'] is not None:
            if t['cell'] is not None:
                t['row'].append(' '.join(''.join(t['cell']).split()))
                t['cell'] = None
            t['rows'].append(t['row'])
            t['row'] = None
        elif tag in ('td', 'th') and t['cell'] is not None:
            t['row'].append(' '.join(''.join(t['cell']).split()))
            t['cell'] = None

    def handle_data(self, data):
        if self._stack and self._stack[-1]['cell'] is not None:
            self._stack[-1]['cell'].append(data)


def _extract_kappa_names(html):
    """Pull the "Item name" column out of the Collector page's item table.
    Locates the column by header text (not position) and right-aligns data
    rows against the header row, since the wiki uses rowspan/colspan cells."""
    p = _WikiTableParser()
    p.feed(html)
    for table in p.tables:
        header_i = col = None
        for i, row in enumerate(table):
            low = [c.strip().lower() for c in row]
            if 'item name' in low:
                header_i, col = i, low.index('item name')
                break
        if header_i is None:
            continue
        header_len = len(table[header_i])
        names = []
        for row in table[header_i + 1:]:
            if len(row) < 2:
                continue
            j = col + (len(row) - header_len)   # right-edge alignment
            if 0 <= j < len(row):
                name = row[j].strip()
                if name:
                    names.append(name)
        if len(names) >= 20:
            return names
    raise ValueError('Collector item table not found (or suspiciously small) — '
                     'wiki layout may have changed')


def fetch_kappa_names(force=False):
    """Return the current Collector item names from the wiki, cached 24 h."""
    if not force and os.path.exists(KAPPA_WIKI_PATH):
        cache = load_json(KAPPA_WIKI_PATH, lambda: None)
        if cache and time.time() - cache.get('timestamp', 0) < KAPPA_WIKI_TTL:
            return cache['names']
    r = http_requests.get(COLLECTOR_WIKI_API, timeout=30, headers={
        'User-Agent': 'TarkovStashHelper/1.0 (github.com/josfire18/tarkov-stash-helper)',
    })
    r.raise_for_status()
    names = _extract_kappa_names(r.json()['parse']['text'])
    save_json(KAPPA_WIKI_PATH, {'timestamp': time.time(), 'names': names})
    return names


def merge_kappa_into_keep_list(keep_list, names, price_idx=None):
    """
    Merge fresh wiki names into the kappa category, preserving user state.
      - name already present (exact or fuzzy ≥90) → keep the entry (acquired,
        aliases, id survive); fuzzy hits are renamed to the wiki spelling with
        the old name kept as an alias.
      - new wiki name → appended unchecked.
      - entry no longer on the wiki (and not user-added) → flagged stale, never
        deleted.
    Idempotent. Returns {'added', 'stale', 'total'}.
    """
    cat = next((c for c in keep_list['categories'] if c['id'] == 'kappa'), None)
    if cat is None:
        cat = {'id': 'kappa', 'label': 'Kappa (Collector)', 'items': []}
        keep_list['categories'].insert(0, cat)

    by_name = {it['name'].casefold(): it for it in cat['items']}
    claimed = set()   # entry ids already matched to a wiki name
    added = []

    for n in names:
        entry = by_name.get(n.casefold())
        if entry is None:
            # Fuzzy rescue for wiki renames ('Press pass (NoiceGuy)' →
            # 'Press pass (issued for NoiceGuy)') so check-state survives.
            # Token-set + punctuation stripping scores real renames at 100
            # while unrelated kappa items stay under ~45.
            from rapidfuzz import fuzz as _fuzz, utils as _futils
            cands = {it['name']: it for it in cat['items']
                     if it['id'] not in claimed and it.get('source') != 'custom'}
            if cands:
                hit = rfuzz.extractOne(n, list(cands.keys()),
                                       scorer=_fuzz.token_set_ratio,
                                       processor=_futils.default_process,
                                       score_cutoff=95)
                if hit:
                    entry = cands[hit[0]]
                    old = entry['name']
                    if old.casefold() != n.casefold():
                        entry['name'] = n
                        if old not in entry.get('aliases', []):
                            entry.setdefault('aliases', []).append(old)
        if entry is not None:
            claimed.add(entry['id'])
            entry['source'] = 'wiki'
            entry.pop('stale', None)
        else:
            new = {'id': str(uuid.uuid4())[:8], 'name': n, 'aliases': [],
                   'acquired': False, 'source': 'wiki'}
            cat['items'].append(new)
            claimed.add(new['id'])
            added.append(n)

    stale = []
    for it in cat['items']:
        if it['id'] in claimed or it.get('source') == 'custom':
            continue
        it['stale'] = True
        stale.append(it['name'])

    # Persist tarkov.dev ids so scans can map entries without alias guessing.
    if price_idx:
        keys = list(price_idx.keys())
        for it in cat['items']:
            if it.get('tdev_id'):
                continue
            data = price_idx.get(it['name'].lower())
            if data is None:
                hit = rfuzz.extractOne(it['name'].lower(), keys, score_cutoff=93)
                if hit:
                    data = price_idx[hit[0]]
            if data:
                it['tdev_id'] = data['id']

    return {'added': added, 'stale': stale, 'total': len(cat['items'])}


def kappa_sync(force=False):
    """Fetch + merge + save. Returns the merge summary. Raises on failure
    (keep_list.json is never touched when the fetch/parse fails)."""
    names = fetch_kappa_names(force=force)
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    price_idx = None
    try:
        price_idx = build_price_index(get_prices())
    except Exception as e:
        print(f"[kappa] price index unavailable during sync: {e}")
    summary = merge_kappa_into_keep_list(keep_list, names, price_idx)
    save_json(KEEPLIST_PATH, keep_list)
    print(f"[kappa] synced {len(names)} wiki items — "
          f"{len(summary['added'])} added, {len(summary['stale'])} stale")
    return summary


# ---------------------------------------------------------------------------
# Task & hideout requirements — from tarkov.dev
# ---------------------------------------------------------------------------

def default_progress():
    return {'completed_tasks': [], 'completed_hideout': [], 'have': {}}


def fetch_tasks():
    """Fetch task + hideout item requirements from tarkov.dev, cache 24 h."""
    r = http_requests.post(TARKOV_API, json={'query': TASKS_QUERY}, timeout=60)
    payload = r.json()
    if payload.get('errors'):
        raise RuntimeError(f"tarkov.dev tasks query failed: {payload['errors']}")
    data = payload.get('data') or {}
    cache = {
        'timestamp':       time.time(),
        'tasks':           data.get('tasks') or [],
        'hideoutStations': data.get('hideoutStations') or [],
    }
    save_json(TASKS_CACHE_PATH, cache)
    return cache


def get_tasks(allow_fetch=True):
    """Cached task data, refreshed when stale. With allow_fetch=False, returns
    whatever cache exists (or None) without touching the network — used inside
    scans so a scan can never block on tarkov.dev."""
    if os.path.exists(TASKS_CACHE_PATH):
        cache = load_json(TASKS_CACHE_PATH, lambda: None)
        if cache and (not allow_fetch
                      or time.time() - cache.get('timestamp', 0) < TASKS_CACHE_TTL):
            return cache
    return fetch_tasks() if allow_fetch else None


# Money hand-ins (e.g. "Compensation for Damage" wants 1M roubles) aren't
# stash items worth tracking — they'd bury real requirements in the aggregate.
CURRENCY_NAMES = {'roubles', 'dollars', 'euros'}

def compute_tasks_view(cache, progress):
    """
    Server-side merged view of what the player still needs.
    Only 'giveItem' objectives count as hand-ins; objectives with a null item
    are skipped (tarkov.dev is migrating TaskObjectiveItem.item → items).
    """
    done_tasks  = set(progress.get('completed_tasks', []))
    done_levels = set(progress.get('completed_hideout', []))
    have        = progress.get('have', {})

    agg = {}
    def _acc(item, count, fir, src_type, src_name, active):
        rec = agg.setdefault(item['id'], {
            'item_id': item['id'], 'name': item.get('name') or '?',
            'shortName': item.get('shortName') or '',
            'total_needed': 0, 'fir_needed': 0, 'sources': [],
        })
        if active:
            rec['total_needed'] += count
            if fir:
                rec['fir_needed'] += count
            rec['sources'].append({'type': src_type, 'name': src_name,
                                   'count': count, 'fir': bool(fir)})

    tasks_out = []
    for t in cache.get('tasks', []):
        items = []
        for o in (t.get('objectives') or []):
            if o.get('type') != 'giveItem':
                continue
            it, cnt = o.get('item'), o.get('count') or 0
            if not it or not it.get('id') or cnt <= 0:
                continue
            if (it.get('name') or '').lower() in CURRENCY_NAMES:
                continue
            fir = bool(o.get('foundInRaid'))
            items.append({'item_id': it['id'], 'name': it.get('name') or '?',
                          'count': cnt, 'fir': fir})
            _acc(it, cnt, fir, 'task',
                 f"{(t.get('trader') or {}).get('name', '?')} — {t['name']}",
                 active=t['id'] not in done_tasks)
        if not items:
            continue   # only hand-in tasks are interesting here
        tasks_out.append({
            'id': t['id'], 'name': t['name'],
            'trader': (t.get('trader') or {}).get('name', '?'),
            'minPlayerLevel': t.get('minPlayerLevel') or 0,
            'kappaRequired': bool(t.get('kappaRequired')),
            'completed': t['id'] in done_tasks,
            'items': items,
        })

    stations_out = []
    for s in cache.get('hideoutStations', []):
        levels = []
        for lv in (s.get('levels') or []):
            items = []
            for req in (lv.get('itemRequirements') or []):
                it, cnt = req.get('item'), req.get('count') or 0
                if not it or not it.get('id') or cnt <= 0:
                    continue
                if (it.get('name') or '').lower() in CURRENCY_NAMES:
                    continue
                items.append({'item_id': it['id'], 'name': it.get('name') or '?',
                              'count': cnt})
                _acc(it, cnt, False, 'hideout', f"{s['name']} L{lv['level']}",
                     active=lv['id'] not in done_levels)
            levels.append({'id': lv['id'], 'level': lv['level'],
                           'completed': lv['id'] in done_levels, 'items': items})
        if levels:
            stations_out.append({'id': s['id'], 'name': s['name'], 'levels': levels})

    aggregate = []
    for rec in agg.values():
        if rec['total_needed'] <= 0:
            continue
        rec['have'] = int(have.get(rec['item_id'], 0))
        aggregate.append(rec)
    aggregate.sort(key=lambda r: -(r['total_needed'] - min(r['have'], r['total_needed'])))

    return {
        'aggregate': aggregate,
        'tasks':     tasks_out,
        'stations':  stations_out,
        'cache_age_minutes': round((time.time() - cache.get('timestamp', 0)) / 60, 1),
    }


def get_protected_ids(keep_list, price_idx):
    """
    {tarkov.dev item id: reason} for everything the player should NOT sell:
    unacquired keep-list entries + task/hideout items still short of their
    required count.  Task data is cache-only here — a scan never waits on
    the network for it.
    """
    protected = {}
    mapped, _unmapped = map_keep_entries_to_ids(keep_list, price_idx)
    for tid, entry in mapped.items():
        if not entry.get('acquired'):
            protected[tid] = 'On keep list'
    try:
        cache = get_tasks(allow_fetch=False)
        if cache:
            progress = load_json(PROGRESS_PATH, default_progress)
            view = compute_tasks_view(cache, progress)
            for rec in view['aggregate']:
                if rec['have'] >= rec['total_needed']:
                    continue
                srcs = rec['sources']
                first = srcs[0]['name'] if srcs else 'tasks'
                more = f' +{len(srcs) - 1} more' if len(srcs) > 1 else ''
                protected.setdefault(
                    rec['item_id'],
                    f"Needed: {first} (×{rec['total_needed']}){more}")
    except Exception as e:
        print(f"[tasks] protected-id pass skipped: {e}")
    return protected


# ---------------------------------------------------------------------------
# Grid detection helpers
# ---------------------------------------------------------------------------

def _dominant_period(sig, lo=35, hi=200):
    """
    Find the dominant repeating period in `sig` (1-D numpy array) using FFT
    autocorrelation.  Returns an int period in [lo, hi] or None.
    """
    sig = np.asarray(sig, dtype=float)
    sig -= sig.mean()
    n = len(sig)
    if n < hi * 2:
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    S = np.fft.rfft(sig, n=nfft)
    acorr = np.fft.irfft(S * np.conj(S), n=nfft)[:n].real
    acorr[0] = 0
    window = acorr[lo:hi + 1]
    if window.max() <= 0:
        return None
    return int(lo + int(np.argmax(window)))


def _grid_phase(sig, period):
    """
    Given a signal and a known period, find the offset (0..period-1) where
    the repeating grid lines fall — i.e. the origin coordinate mod period.
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    # Pad to a multiple of period then fold and sum
    r = n % period
    padded = np.pad(sig, (0, period - r)) if r else sig
    folded = padded.reshape(-1, period).sum(axis=0)
    return int(np.argmax(folded))


def detect_stash_grid(img_bgr, lo=35, hi=200):
    """
    Auto-detect the Tarkov stash grid parameters from the screenshot.
    Uses the repeating edge pattern (cell borders) via autocorrelation.

    Returns dict {cell_w, cell_h, origin_x, origin_y} or None on failure.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Project each axis — peaks mark the grid lines
    h_proj = np.sum(np.abs(gy), axis=1)   # rows → horizontal line positions
    v_proj = np.sum(np.abs(gx), axis=0)   # cols → vertical line positions

    cell_h = _dominant_period(h_proj, lo, hi)
    cell_w = _dominant_period(v_proj, lo, hi)
    if not cell_h or not cell_w:
        return None

    origin_y = _grid_phase(h_proj, cell_h)
    origin_x = _grid_phase(v_proj, cell_w)

    return {
        'cell_w': int(cell_w), 'cell_h': int(cell_h),
        'origin_x': int(origin_x), 'origin_y': int(origin_y),
    }


def snap_to_cell(px, py, grid):
    """Return the (col, row) grid cell index nearest to pixel (px, py)."""
    cx = max(0, round((px - grid['origin_x']) / grid['cell_w']))
    cy = max(0, round((py - grid['origin_y']) / grid['cell_h']))
    return cx, cy


def grid_rect(cx, cy, slots_w, slots_h, grid, pad=2):
    """Pixel bounding rect for a grid region starting at (cx,cy), spanning slots."""
    ox, oy = grid['origin_x'], grid['origin_y']
    cw, ch = grid['cell_w'], grid['cell_h']
    return (ox + cx * cw - pad,
            oy + cy * ch - pad,
            ox + (cx + slots_w) * cw + pad,
            oy + (cy + slots_h) * ch + pad)


def fallback_rect(x, y, w, h, slots_w, slots_h, cell_size, pad=2):
    """
    When grid detection fails, snap to a cell_size grid and extend by slot count.
    Tarkov item labels appear at the TOP of the icon cell, so we expand downward.
    """
    snapped_x = round(x / cell_size) * cell_size
    snapped_y = round(y / cell_size) * cell_size
    return (snapped_x - pad,
            snapped_y - pad,
            snapped_x + slots_w * cell_size + pad,
            snapped_y + slots_h * cell_size + pad)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

_badge_font_cache = {}
def _badge_font(size=11):
    if size not in _badge_font_cache:
        from PIL import ImageFont
        for path in ['C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf',
                     'C:/Windows/Fonts/calibrib.ttf', 'C:/Windows/Fonts/segoeui.ttf']:
            try:
                _badge_font_cache[size] = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
        else:
            _badge_font_cache[size] = ImageFont.load_default()
    return _badge_font_cache[size]


def draw_badge(draw, x, y, label, bg=(30, 160, 30, 230)):
    """Draw a small numbered badge at pixel position (x, y)."""
    font = _badge_font(11)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 3
    draw.rectangle([x, y, x + tw + pad * 2 + 1, y + th + pad * 2],
                   fill=bg, outline=(255, 255, 255, 160), width=1)
    draw.text((x + pad, y + pad), label, fill=(255, 255, 255, 255), font=font)


def preprocess_for_ocr(img: Image.Image):
    """
    Scale the image up OCR_SCALE× and convert to grayscale.
    Larger text is dramatically more accurate for Tesseract.
    Returns (upscaled_pil_image, scale_factor).
    """
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    up = cv2.resize(gray, (w * OCR_SCALE, h * OCR_SCALE), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(up), float(OCR_SCALE)


def group_ocr_by_lines(ocr_data, scale=1.0):
    """
    Aggregate Tesseract word-level results into line-level phrases.
    Coordinates are divided by `scale` to map back to original image space.
    Returns list of {'text', 'x', 'y', 'w', 'h', 'conf'}.
    """
    from collections import defaultdict
    lines = defaultdict(list)
    for i, text in enumerate(ocr_data['text']):
        text = text.strip()
        if not text or ocr_data['level'][i] != 5:   # word level only
            continue
        conf = int(ocr_data['conf'][i])
        if conf < 15:
            continue
        key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
        lines[key].append({
            'text': text,
            'x': ocr_data['left'][i],   'y': ocr_data['top'][i],
            'w': ocr_data['width'][i],  'h': ocr_data['height'][i],
            'conf': conf,
        })

    result = []
    for words in lines.values():
        if not words:
            continue
        full_text = ' '.join(w['text'] for w in words)
        x1 = min(w['x'] for w in words)
        y1 = min(w['y'] for w in words)
        x2 = max(w['x'] + w['w'] for w in words)
        y2 = max(w['y'] + w['h'] for w in words)
        result.append({
            'text': full_text,
            'x': int(x1 / scale), 'y': int(y1 / scale),
            'w': int((x2 - x1) / scale), 'h': int((y2 - y1) / scale),
            'conf': sum(w['conf'] for w in words) / len(words),
        })
    return result


def build_alias_map(keep_list):
    """Returns list of (alias_lower, item) for fuzzy matching."""
    entries = []
    for cat in keep_list['categories']:
        for item in cat['items']:
            entries.append((item['name'].lower(), item))
            for alias in item.get('aliases', []):
                entries.append((alias.lower(), item))
    return entries


def ocr_stash(img: Image.Image, alias_map, grid=None, cell_size=64):
    """
    Run OCR on stash image.  When `grid` is provided (from detect_stash_grid),
    highlights snap precisely to cell boundaries with correct multi-slot sizing.
    Returns (annotated_image, detections).
    """
    if not tesseract_available():
        return img, []
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                     config='--psm 11 --oem 3')
    draw = ImageDraw.Draw(img, 'RGBA')
    detections = []
    seen_positions = set()

    for i, text in enumerate(data['text']):
        text = text.strip()
        if not text or len(text) < 3:
            continue
        if int(data['conf'][i]) < 20:
            continue

        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        pos_key = (x // 10, y // 10)
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        aliases = [a for a, _ in alias_map]
        match = rfuzz.extractOne(text.lower(), aliases, score_cutoff=OCR_THRESHOLD)
        if not match:
            continue

        matched_alias, score, idx = match
        item = alias_map[idx][1]

        color  = (80, 160, 80, 120) if item['acquired'] else (180, 50, 50, 140)
        border = (80, 200, 80, 255) if item['acquired'] else (220, 60, 60, 255)

        # Text appears at the top-left of the item cell; snap to grid
        if grid:
            cx, cy = snap_to_cell(x, y, grid)
            rx1, ry1, rx2, ry2 = grid_rect(cx, cy, 1, 1, grid)
        else:
            rx1, ry1, rx2, ry2 = fallback_rect(x, y, w, h, 1, 1, cell_size)

        draw.rectangle([rx1, ry1, rx2, ry2], fill=color, outline=border, width=2)
        detections.append({
            'text': text, 'matched': item['name'], 'score': score,
            'acquired': item['acquired'], 'x': rx1, 'y': ry1,
            'w': rx2 - rx1, 'h': ry2 - ry1,
        })

    return img, detections


# ---------------------------------------------------------------------------
# Keep-list scan — one pipeline shared by the hotkey and the Scan button
# ---------------------------------------------------------------------------

def _persist_grid(g):
    s = load_json(SETTINGS_PATH, default_settings)
    s['grid'] = g
    save_json(SETTINGS_PATH, s)


def run_keep_scan():
    """
    Capture → grid → OCR pass → icon-DB pass, annotated for the keep list.
    Returns {image, detections, grid_src, warnings, checklist_matches}.
    Raises ScanError/Exception — callers decide how to surface it.
    """
    _scan_state.update({'running': True, 'phase': 'capture',
                        'done': 0, 'total': 0, 'ts': time.time()})
    try:
        settings   = load_json(SETTINGS_PATH, default_settings)
        keep_list  = load_json(KEEPLIST_PATH, default_keep_list)
        alias_map  = build_alias_map(keep_list)
        cell_size  = settings.get('cell_size', 64)
        warnings   = []

        if not tesseract_available():
            warnings.append('Tesseract OCR not installed — name reading disabled '
                            '(winget install UB-Mannheim.TesseractOCR)')

        img, img_bgr = capture_stash_image(settings)

        _scan_state['phase'] = 'grid'
        grid, grid_src = resolve_grid(img_bgr, settings, persist_fn=_persist_grid)
        print(f"Grid[{grid_src}]: cell={grid['cell_w']}×{grid['cell_h']} "
              f"origin=({grid['origin_x']},{grid['origin_y']})")

        # OCR pass — grid-snapped highlights for keep-list aliases
        img, detections = ocr_stash(img, alias_map, grid=grid, cell_size=cell_size)

        # Icon-DB pass (finds keep-list items OCR missed)
        _scan_state['phase'] = 'match'
        found_entries = {}   # entry_id -> entry, for the confirm-to-check bar
        for d in detections:
            for alias, entry in alias_map:
                if entry['name'] == d['matched']:
                    found_entries[entry['id']] = entry
                    break

        prices    = get_prices()
        price_idx = build_price_index(prices)
        keepid_to_entry, unmapped = map_keep_entries_to_ids(keep_list, price_idx)
        if unmapped:
            warnings.append(f"{len(unmapped)} keep-list item(s) not in the item "
                            f"catalog: {', '.join(unmapped[:3])}"
                            + ('…' if len(unmapped) > 3 else ''))

        icon_db = get_icon_db()
        if not icon_db:
            warnings.append('Icon DB not built — icon matching skipped '
                            '(build it from the Sell Advisor page)')
        if icon_db and keepid_to_entry:
            draw = ImageDraw.Draw(img, 'RGBA')
            label_matcher = build_label_matcher(prices)
            def _cb(done, total):
                _scan_state['done'], _scan_state['total'] = done, total
            ocr_positions = {(d['x'] // 20, d['y'] // 20) for d in detections}
            for d in identify_items_by_icon(img_bgr, grid, icon_db,
                                            label_matcher=label_matcher,
                                            progress_cb=_cb):
                entry = keepid_to_entry.get(d['item_id'])
                if not entry:
                    continue
                x = grid['origin_x'] + d['col'] * grid['cell_w']
                y = grid['origin_y'] + d['row'] * grid['cell_h']
                w = d['W'] * grid['cell_w']
                h = d['H'] * grid['cell_h']
                pos_key = (x // 20, y // 20)
                if pos_key in ocr_positions:
                    continue
                ocr_positions.add(pos_key)
                found_entries[entry['id']] = entry
                color  = (80, 160, 80, 130) if entry['acquired'] else (180, 50, 50, 150)
                border = (80, 200, 80, 255) if entry['acquired'] else (220, 60, 60, 255)
                draw.rectangle([x, y, x + w, y + h], fill=color, outline=border, width=2)
                detections.append({
                    'text': f'[icon] {entry["name"]}', 'matched': entry['name'],
                    'score': d['score'], 'acquired': entry['acquired'],
                    'x': x, 'y': y, 'w': w, 'h': h,
                })

        checklist_matches = [
            {'entry_id': e['id'], 'name': e['name']}
            for e in found_entries.values() if not e.get('acquired')
        ]

        buf = BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        return {'image': encoded, 'detections': detections, 'grid_src': grid_src,
                'warnings': warnings, 'checklist_matches': checklist_matches}
    finally:
        _scan_state.update({'running': False, 'phase': None, 'ts': time.time()})


def do_scan():
    """Hotkey-triggered scan: same pipeline as the Scan button, results (or the
    error) parked in _last_scan for the frontend poller."""
    try:
        result = run_keep_scan()
        with _scan_lock:
            _last_scan.update(result, error=None, ts=time.time())
    except Exception as e:
        import traceback
        print(f"[hotkey scan] ERROR:\n{traceback.format_exc()}")
        with _scan_lock:
            _last_scan.update({'image': None, 'detections': [], 'warnings': [],
                               'checklist_matches': [], 'error': str(e),
                               'ts': time.time()})


# ---------------------------------------------------------------------------
# Global hotkey
# ---------------------------------------------------------------------------

class HotkeyManager:
    """Owns the pynput GlobalHotKeys listener; supports live rebinding."""
    def __init__(self):
        self._listener = None
        self._lock = threading.Lock()
        self.current = None

    @staticmethod
    def validate(hotkey_str):
        """Raises ValueError if pynput can't parse the combo."""
        keyboard.HotKey.parse(hotkey_str)

    def _on_activate(self):
        print(f"Hotkey {self.current} triggered — scanning...")
        threading.Thread(target=do_scan, daemon=True).start()

    def register(self, hotkey_str):
        self.validate(hotkey_str)
        with self._lock:
            if self._listener is not None:
                try:
                    self._listener.stop()
                except Exception:
                    pass
                self._listener = None
            listener = keyboard.GlobalHotKeys({hotkey_str: self._on_activate})
            listener.daemon = True
            listener.start()
            self._listener = listener
            self.current = hotkey_str
        print(f"Hotkey listener active — press {hotkey_str} in-game to scan")


hotkey_manager = HotkeyManager()


def start_hotkey_listener():
    settings = load_json(SETTINGS_PATH, default_settings)
    hotkey_str = settings.get('hotkey', DEFAULT_HOTKEY)
    # Migrate the old F9 default: EFT and common overlays grab function keys,
    # so plain F9 frequently never reaches this app.
    if hotkey_str == '<f9>':
        hotkey_str = DEFAULT_HOTKEY
        settings['hotkey'] = hotkey_str
        save_json(SETTINGS_PATH, settings)
        print(f"Hotkey migrated F9 → {hotkey_str} (F9 conflicts with EFT itself)")
    try:
        hotkey_manager.register(hotkey_str)
    except Exception as e:
        print(f"Hotkey listener failed: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/last-scan', methods=['GET'])
def last_scan():
    since = float(request.args.get('since', 0))
    with _scan_lock:
        if _last_scan['ts'] > since and (_last_scan['image'] or _last_scan.get('error')):
            return jsonify({'ready': True, **_last_scan})
    return jsonify({'ready': False})

@app.route('/api/scan-status', methods=['GET'])
def scan_status():
    return jsonify(dict(_scan_state))

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'tesseract':      tesseract_available(),
        'tesseract_cmd':  pytesseract.pytesseract.tesseract_cmd,
        'icon_db_ready':  get_icon_db() is not None,
        'icon_db_error':  _index_build_state.get('error') or _icon_db_error,
        'prices_cached':  os.path.exists(PRICES_PATH),
        'hotkey':         hotkey_manager.current,
    })

@app.route('/api/hotkey', methods=['POST'])
def set_hotkey():
    hotkey_str = (request.json or {}).get('hotkey', '').strip()
    try:
        hotkey_manager.register(hotkey_str)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Invalid hotkey: {e}'}), 400
    settings = load_json(SETTINGS_PATH, default_settings)
    settings['hotkey'] = hotkey_str
    save_json(SETTINGS_PATH, settings)
    return jsonify({'ok': True, 'hotkey': hotkey_str})

@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    try:
        return jsonify(run_keep_scan())
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[screenshot] ERROR:\n{tb}")
        return jsonify({'error': str(e), 'image': None, 'detections': [],
                        'warnings': [], 'checklist_matches': []})

@app.route('/api/keep-list', methods=['GET'])
def get_keep_list():
    return jsonify(load_json(KEEPLIST_PATH, default_keep_list))

@app.route('/api/keep-list/toggle', methods=['POST'])
def toggle_item():
    data = request.json
    item_id = data.get('id')
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        for item in cat['items']:
            if item['id'] == item_id:
                item['acquired'] = not item['acquired']
                save_json(KEEPLIST_PATH, keep_list)
                return jsonify({'ok': True, 'acquired': item['acquired']})
    return jsonify({'ok': False, 'error': 'Item not found'}), 404

@app.route('/api/keep-list/add', methods=['POST'])
def add_item():
    data = request.json
    cat_id = data.get('category', 'kappa')
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        if cat['id'] == cat_id:
            new_item = {
                'id': str(uuid.uuid4())[:8],
                'name': data['name'],
                'aliases': data.get('aliases', []),
                'acquired': False,
                'source': 'custom',   # wiki sync must never flag user items stale
            }
            if 'task' in data:
                new_item['task'] = data['task']
            if 'count' in data:
                new_item['count'] = data['count']
            cat['items'].append(new_item)
            save_json(KEEPLIST_PATH, keep_list)
            return jsonify({'ok': True, 'item': new_item})
    return jsonify({'ok': False, 'error': 'Category not found'}), 404

@app.route('/api/keep-list/remove/<item_id>', methods=['DELETE'])
def remove_item(item_id):
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        cat['items'] = [i for i in cat['items'] if i['id'] != item_id]
    save_json(KEEPLIST_PATH, keep_list)
    return jsonify({'ok': True})

@app.route('/api/keep-list/acquire', methods=['POST'])
def acquire_items():
    """Batch-mark entries acquired — the scan confirm bar's one-click action."""
    ids = set((request.json or {}).get('ids') or [])
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    updated = 0
    for cat in keep_list['categories']:
        for item in cat['items']:
            if item['id'] in ids and not item['acquired']:
                item['acquired'] = True
                updated += 1
    if updated:
        save_json(KEEPLIST_PATH, keep_list)
    return jsonify({'ok': True, 'updated': updated})

@app.route('/api/kappa/refresh', methods=['POST'])
def kappa_refresh():
    """Force a wiki sync. keep_list.json is untouched when fetch/parse fails."""
    try:
        summary = kappa_sync(force=True)
        return jsonify({'ok': True, **summary})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 502

@app.route('/api/calibration-screenshot', methods=['GET'])
def calibration_screenshot():
    """Capture full screen for region selection — delay lets user tab to game first."""
    delay = int(request.args.get('delay', 0))
    if delay:
        time.sleep(delay)
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        raw = sct.grab(monitor)
        img = Image.frombytes('RGB', raw.size, raw.bgra, 'raw', 'BGRX')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=75)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': encoded, 'width': img.width, 'height': img.height})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(load_json(SETTINGS_PATH, default_settings))

@app.route('/api/settings', methods=['POST'])
def save_settings():
    settings = request.json
    save_json(SETTINGS_PATH, settings)
    return jsonify({'ok': True})

@app.route('/api/reset', methods=['POST'])
def reset_keep_list():
    save_json(KEEPLIST_PATH, default_keep_list())
    # Land on current wiki data when online; the hardcoded defaults are only
    # the offline fallback.
    try:
        kappa_sync()
    except Exception as e:
        print(f"[kappa] post-reset sync skipped: {e}")
    return jsonify({'ok': True})

# ---------------------------------------------------------------------------
# Tasks & hideout page
# ---------------------------------------------------------------------------

@app.route('/tasks')
def tasks_page():
    return render_template('tasks.html')

@app.route('/api/tasks', methods=['GET'])
def api_tasks():
    try:
        cache = get_tasks()
    except Exception as e:
        return jsonify({'error': str(e)}), 502
    progress = load_json(PROGRESS_PATH, default_progress)
    return jsonify(compute_tasks_view(cache, progress))

@app.route('/api/tasks/refresh', methods=['POST'])
def api_tasks_refresh():
    try:
        cache = fetch_tasks()
        return jsonify({'ok': True, 'tasks': len(cache['tasks']),
                        'stations': len(cache['hideoutStations'])})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 502

@app.route('/api/tasks/complete', methods=['POST'])
def api_tasks_complete():
    data = request.json or {}
    typ, tid, done = data.get('type'), data.get('id'), bool(data.get('done'))
    if typ not in ('task', 'hideout') or not tid:
        return jsonify({'ok': False, 'error': 'type must be task|hideout, id required'}), 400
    progress = load_json(PROGRESS_PATH, default_progress)
    key = 'completed_tasks' if typ == 'task' else 'completed_hideout'
    ids = set(progress.get(key, []))
    (ids.add if done else ids.discard)(tid)
    progress[key] = sorted(ids)
    save_json(PROGRESS_PATH, progress)
    return jsonify({'ok': True})

@app.route('/api/tasks/have', methods=['POST'])
def api_tasks_have():
    data = request.json or {}
    item_id = data.get('item_id')
    if not item_id:
        return jsonify({'ok': False, 'error': 'item_id required'}), 400
    try:
        delta = int(data.get('delta') or 0)
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'delta must be an integer'}), 400
    progress = load_json(PROGRESS_PATH, default_progress)
    have = progress.setdefault('have', {})
    have[item_id] = max(0, int(have.get(item_id, 0)) + delta)
    save_json(PROGRESS_PATH, progress)
    return jsonify({'ok': True, 'have': have[item_id]})


# ---------------------------------------------------------------------------
# Sell page
# ---------------------------------------------------------------------------

@app.route('/sell')
def sell_page():
    return render_template('sell.html')

@app.route('/api/prices/status', methods=['GET'])
def prices_status():
    if not os.path.exists(PRICES_PATH):
        return jsonify({'cached': False, 'age_minutes': None, 'count': 0})
    cache = load_json(PRICES_PATH, lambda: {})
    age = (time.time() - cache.get('timestamp', 0)) / 60
    return jsonify({'cached': True, 'age_minutes': round(age, 1), 'count': len(cache.get('items', []))})

@app.route('/api/prices/refresh', methods=['POST'])
def prices_refresh():
    try:
        cache = fetch_prices()
        return jsonify({'ok': True, 'count': len(cache['items'])})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/debug/grid', methods=['POST'])
def debug_grid():
    """Capture the configured region and return detected grid parameters."""
    settings = load_json(SETTINGS_PATH, default_settings)
    img, img_bgr = capture_stash_image(settings)
    grid = detect_stash_grid(img_bgr)
    return jsonify({'grid': grid, 'img_size': [img.width, img.height]})

@app.route('/api/icons/prefetch', methods=['POST'])
def icons_prefetch():
    """Pre-download icons for all keep-list items in background."""
    def _run():
        keep_list = load_json(KEEPLIST_PATH, default_keep_list)
        prices    = get_prices()
        price_idx = build_price_index(prices)
        settings  = load_json(SETTINGS_PATH, default_settings)
        scale     = settings.get('icon_scale', 2.0)
        n = prefetch_keep_list_icons(keep_list, price_idx, scale)
        print(f"Icon prefetch complete: {n} icons downloaded/verified")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Icon prefetch started in background'})

_index_build_state = {'running': False, 'done': 0, 'total': 0, 'ts': 0, 'error': None}

@app.route('/api/icons/build-index', methods=['POST'])
def icons_build_index():
    """Download all icons and build the NCC icon index."""
    if _index_build_state['running']:
        return jsonify({'ok': False, 'message': 'Build already in progress'})

    def _run():
        global _icon_db
        _index_build_state.update({'running': True, 'done': 0, 'total': 0,
                                   'ts': time.time(), 'error': None})
        try:
            prices = get_prices()
            _index_build_state['total'] = len(prices.get('items', []))
            def cb(done, total):
                _index_build_state['done']  = done
                _index_build_state['total'] = total
            db = build_icon_db(prices, progress_cb=cb)
            with _icon_db_lock:
                _icon_db = db
            _index_build_state['done'] = _index_build_state['total']
            print(f"[icon_db] built: {sum(len(b['ids']) for b in db.values())} items")
        except Exception as e:
            _index_build_state['error'] = str(e)
            print(f"[icon_db] build failed: {e}")
            return
        finally:
            _index_build_state['running'] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Icon DB build started'})


@app.route('/api/icons/index-status', methods=['GET'])
def icons_index_status():
    db = get_icon_db()
    summary = None
    if db:
        summary = {
            'sizes':       len(db),
            'total_items': sum(len(b['ids']) for b in db.values()),
        }
    return jsonify({'build': _index_build_state, 'index': summary})


@app.route('/api/icons/matcher-status', methods=['GET'])
def icons_matcher_status():
    """Unified status endpoint for the sell page — covers icon DB build + readiness."""
    db = get_icon_db()
    item_count = sum(len(b['ids']) for b in db.values()) if db else 0
    return jsonify({
        'running':    _index_build_state['running'],
        'done_count': _index_build_state.get('done', 0),
        'total':      _index_build_state.get('total', 0),
        'error':      _index_build_state.get('error') or _icon_db_error,
        'ready':      db is not None,
        'item_count': item_count,
    })


@app.route('/api/sell-scan', methods=['POST'])
def sell_scan():
  try:
    return _sell_scan_inner()
  except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"[sell_scan] ERROR:\n{tb}")
    return jsonify({'error': str(e), 'traceback': tb, 'image': None, 'results': [], 'grid': None})

def _sell_scan_inner():
    settings = load_json(SETTINGS_PATH, default_settings)

    # --- Icon DB required ----------------------------------------------------
    icon_db = get_icon_db()
    if not icon_db:
        return jsonify({
            'image': None, 'results': [], 'grid': None,
            'error': 'Icon index not built yet. Click "Build Icon DB" first.',
        })

    _scan_state.update({'running': True, 'phase': 'capture',
                        'done': 0, 'total': 0, 'ts': time.time()})
    try:
        # --- Screenshot (region required for sell scans) ----------------------
        try:
            img, img_bgr = capture_stash_image(settings, require_region=True)
        except ScanError as e:
            return jsonify({'image': None, 'results': [], 'grid': None,
                            'error': str(e)})

        # --- Grid detection (constrained to the 63px pitch, persisted) -------
        _scan_state['phase'] = 'grid'
        grid, grid_src = resolve_grid(img_bgr, settings, persist_fn=_persist_grid)
        print(f"Grid[{grid_src}]: cell={grid['cell_w']}×{grid['cell_h']} "
              f"origin=({grid['origin_x']},{grid['origin_y']})")

        prices     = get_prices()
        price_idx  = build_price_index(prices)
        id_to_item = {it['id']: it for it in prices.get('items', [])}

        # Everything the player shouldn't sell: unacquired keep-list entries +
        # task/hideout items still short of their required count.
        keep_list = load_json(KEEPLIST_PATH, default_keep_list)
        protected = get_protected_ids(keep_list, price_idx)

        warnings = []
        if not tesseract_available():
            warnings.append('Tesseract OCR not installed — name reading disabled '
                            '(winget install UB-Mannheim.TesseractOCR)')

        # --- NCC + OCR-label identification (uses existing icon DB) ----------
        _scan_state['phase'] = 'match'
        def _cb(done, total):
            _scan_state['done'], _scan_state['total'] = done, total
        label_matcher = build_label_matcher(prices)
        raw_detections = identify_items_by_icon(img_bgr, grid, icon_db,
                                                label_matcher=label_matcher,
                                                progress_cb=_cb)
        print(f"[sell_scan] matches: {len(raw_detections)}")

        # --- Number items in row-major order, KEEP items highlighted cyan ----
        draw    = ImageDraw.Draw(img, 'RGBA')
        results = []
        num     = 1

        keep_results = []   # KEEP items — appended after numbered items
        for d in sorted(raw_detections, key=lambda r: (r['row'], r['col'])):
            bx = grid['origin_x'] + d['col'] * grid['cell_w'] + 2
            by = grid['origin_y'] + d['row'] * grid['cell_h'] + 2

            item_data = id_to_item.get(d['item_id'])
            if not item_data:
                continue

            if d['item_id'] in protected:
                # Cyan = KEEP: unmistakably different from flea-green and
                # trader-gold so "do not sell" reads at a glance.
                fx1 = grid['origin_x'] + d['col'] * grid['cell_w']
                fy1 = grid['origin_y'] + d['row'] * grid['cell_h']
                fx2 = fx1 + d['W'] * grid['cell_w']
                fy2 = fy1 + d['H'] * grid['cell_h']
                draw.rectangle([fx1, fy1, fx2, fy2],
                               fill=(0, 180, 220, 60), outline=(0, 220, 255, 255), width=3)
                draw_badge(draw, bx, by, 'KEEP', bg=(0, 140, 180, 230))
                keep_results.append({
                    'num':          'K',
                    'matched_name': item_data['name'],
                    'score':        d['score'],
                    'col':          d['col'], 'row': d['row'],
                    'W':            d['W'],   'H':   d['H'],
                    'rotated':      d.get('rotated', False),
                    'x': bx, 'y': by,
                    'recommend':    'keep',
                    'trader_name':  None, 'trader_price': None,
                    'flea_list':    None, 'flea_net':     None,
                    'reason':       protected[d['item_id']],
                })
                continue

            rec = sell_recommendation(item_data)
            bg  = (30, 150, 30, 230) if rec['recommend'] == 'flea' else (180, 120, 20, 230)
            draw_badge(draw, bx, by, str(num), bg=bg)

            results.append({
                'num':          num,
                'matched_name': item_data['name'],
                'score':        d['score'],
                'col':          d['col'], 'row': d['row'],
                'W':            d['W'],   'H':   d['H'],
                'rotated':      d.get('rotated', False),
                'x': bx, 'y': by,
                **rec,
            })
            num += 1

        results.extend(keep_results)   # KEEP items always at the end

        buf = BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        return jsonify({
            'image':    encoded,
            'results':  results,
            'grid':     grid,
            'warnings': warnings,
        })
    finally:
        _scan_state.update({'running': False, 'phase': None, 'ts': time.time()})


HOST = '127.0.0.1'
PORT = 8877
URL = f'http://{HOST}:{PORT}'


def run_server():
    """The Flask app + all /api routes are unchanged — this just serves them
    on localhost instead of the old port-80/custom-hostname setup. The window
    below is the only thing that changed; nothing about scanning, OCR, or the
    icon DB was touched."""
    from waitress import serve
    serve(app, host=HOST, port=PORT, _quiet=True)


def _startup_maintenance():
    """One-shot background housekeeping: purge retired CNN model files from
    user machines and freshen the kappa list from the wiki when stale."""
    for fn in ('icon_model.pt', 'icon_model_meta.json', 'icon_index.json'):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[cleanup] removed retired file {fn}")
            except Exception as e:
                print(f"[cleanup] could not remove {fn}: {e}")
    try:
        cache = load_json(KAPPA_WIKI_PATH, lambda: None) if os.path.exists(KAPPA_WIKI_PATH) else None
        if not cache or time.time() - cache.get('timestamp', 0) > KAPPA_WIKI_TTL:
            kappa_sync()
    except Exception as e:
        print(f"[kappa] startup sync skipped (offline?): {e}")


def _run_app():
    """Hosts the local Flask UI in a native window (pywebview) instead of a
    browser tab, so there's no URL for the user to see or navigate to — it
    just looks like a normal desktop app. Closing the window minimizes to
    the tray; Quit from the tray menu actually exits."""
    import webview
    import pystray
    from icon_asset import load_tray_image

    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=_startup_maintenance, daemon=True).start()
    start_hotkey_listener()

    window = webview.create_window(
        'Tarkov Stash Helper', URL,
        width=1180, height=860, min_size=(900, 640),
    )

    quitting = threading.Event()

    def on_closing():
        if quitting.is_set():
            return True  # allow the real close (Quit was chosen from the tray)
        window.hide()
        return False  # veto the close — minimize to tray instead

    window.events.closing += on_closing

    def on_open(icon, item):
        window.show()

    def on_quit(icon, item):
        quitting.set()
        icon.stop()
        window.destroy()
        os._exit(0)

    tray_icon = pystray.Icon(
        'TarkovStashHelper',
        load_tray_image(),
        'Tarkov Stash Helper',
        menu=pystray.Menu(
            pystray.MenuItem('Open Stash Helper', on_open, default=True),
            pystray.MenuItem('Quit', on_quit),
        ),
    )
    threading.Thread(target=tray_icon.run, daemon=True).start()

    webview.start()  # blocks; owns the main thread


if __name__ == '__main__':
    # Packaged windowed builds (PyInstaller --windowed) have no console and
    # sys.stdout is None, so print() would raise — route output to a log file.
    if FROZEN and sys.stdout is None:
        log_path = os.path.join(DATA, 'app.log')
        log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        print(f"Tarkov Stash Helper starting (internal: {URL})")
    _run_app()
