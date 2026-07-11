"""
EFT icon-cache reader + visual association for the Tarkov Stash Helper.

Escape From Tarkov writes pixel-perfect icon renders — the exact pixels the game
draws in the stash, including the account's own modded weapon builds — to

    %LOCALAPPDATA%\\Temp\\Battlestate Games\\EscapeFromTarkov\\Icon Cache\\live\\

as `N.png` files (BGRA, real alpha) at per-slot resolution `(63·W+1)×(63·H+1)`
plus an `index.json` (`{bsgItemHash: fileNumber}`).

We do NOT port BSG's item-hash algorithm (fragile, needs the game's items.json).
Instead we map each cache PNG to a tarkov.dev item ID **visually**: a pooled-vector
shortlist over references of the same slot size, verified with masked NCC.  The
association is persisted to `data/cache_map.json` and refreshed incrementally
(only new/changed cache files, by mtime) on each rebuild.

Confident matches become the preferred runtime templates (exact game pixels).
Modded weapon builds that only partially match their base weapon are kept as
`preset` templates tagged with the best-guess base-item ID.  Anything below the
floor threshold is recorded as `unknown` and contributes no template.
"""

import os
import json
import glob
import time
import threading

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SETTINGS_PATH  = os.path.join(DATA, 'settings.json')
CACHE_MAP_PATH = os.path.join(DATA, 'cache_map.json')

GRID_PITCH = 63              # px per slot @ 1080p (icon cache + tarkov.dev grid geometry)
POOL       = 16              # pooled-descriptor side length for the shortlist
SHORTLIST  = 12              # NCC-verify this many top pooled candidates
STRONG     = 0.86            # >= this NCC → confident item match (preset=False)
FLOOR      = 0.55            # >= this (but < STRONG) → preset variant of best guess
NEUTRAL_TINT = (38, 42, 44)  # association happens tint-agnostic (masks drop the bg)


# ---------------------------------------------------------------------------
# Cache-folder discovery
# ---------------------------------------------------------------------------

def default_cache_dir():
    la = os.environ.get('LOCALAPPDATA')
    if not la:
        return None
    return os.path.join(la, 'Temp', 'Battlestate Games', 'EscapeFromTarkov',
                        'Icon Cache', 'live')


def find_cache_dir():
    """Return the icon-cache folder — settings override first, then the default.
    Returns None if nothing usable exists (graceful fallback)."""
    override = None
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                override = json.load(f).get('icon_cache_path')
    except Exception:
        pass
    for cand in (override, default_cache_dir()):
        if cand and os.path.isdir(cand):
            return cand
    return None


def slot_size(w_px, h_px):
    """Pixel dims → (W, H) slot footprint using the 63px pitch."""
    return (max(1, int(round((w_px - 1) / GRID_PITCH))),
            max(1, int(round((h_px - 1) / GRID_PITCH))))


# ---------------------------------------------------------------------------
# Descriptors + scoring
# ---------------------------------------------------------------------------

def _composite_gray(bgra, tint=NEUTRAL_TINT):
    """BGRA → grayscale composited on `tint` (float32)."""
    if bgra.ndim == 2:
        return bgra.astype(np.float32)
    if bgra.shape[2] == 3:
        return cv2.cvtColor(bgra, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rgb = bgra[:, :, :3].astype(np.float32)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, tint, dtype=np.float32)
    comp = (rgb * a + bg * (1.0 - a)).astype(np.uint8)
    return cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _pooled(bgra):
    """Fixed-size pooled grayscale descriptor (mean-centred, unit-norm) for the
    shortlist.  Size-agnostic so it can rank references cheaply."""
    g = _composite_gray(bgra)
    g = cv2.resize(g, (POOL, POOL), interpolation=cv2.INTER_AREA).reshape(-1)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n) if n > 1e-6 else g


def _alpha_mask(bgra, W, H, per_slot=64):
    """Foreground mask (alpha>32) at canonical resolution, flat float32."""
    tw, th = W * per_slot, H * per_slot
    if bgra.ndim == 3 and bgra.shape[2] == 4:
        m = (bgra[:, :, 3] > 32).astype(np.uint8)
    else:
        bgr = bgra if bgra.ndim == 3 else cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)
        s = bgr[:, :, :3].astype(np.int16)
        m = ((s.max(2) - s.min(2)) > 12).astype(np.uint8)
    m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    return m.reshape(-1).astype(np.float32)


def _canon_gray(bgra, W, H, per_slot=64):
    """Canonical grayscale vector composited on the neutral bg, flat float32."""
    g = _composite_gray(bgra)
    return cv2.resize(g, (W * per_slot, H * per_slot),
                      interpolation=cv2.INTER_AREA).reshape(-1)


def _masked_corr(a, ma, b, mb):
    """Symmetric masked NCC between two canonical vectors over the shared
    foreground region."""
    m = ma * mb
    cnt = m.sum()
    if cnt < 32:
        return -1.0
    am = (a - (a * m).sum() / cnt) * m
    bm = (b - (b * m).sum() / cnt) * m
    na = np.linalg.norm(am)
    nb = np.linalg.norm(bm)
    if na < 1e-6 or nb < 1e-6:
        return -1.0
    return float((am * bm).sum() / (na * nb))


# ---------------------------------------------------------------------------
# Reference index (tarkov.dev base-images, grouped by slot size)
# ---------------------------------------------------------------------------

def _build_reference_index(items, load_src_fn, workers=24):
    """
    Download/decode every item's base-image once and group by slot size.

    Returns {(W,H): {'ids':[...], 'names':[...], 'pooled':(N,POOL²) float32,
                     'srcs':[bgra,...]}}.
    `srcs` are kept in memory for masked-NCC verification of shortlisted refs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(it):
        src, _ = load_src_fn(it)
        if src is None:
            return None
        W = it.get('width') or 1
        H = it.get('height') or 1
        return (W, H), it['id'], it['name'], _pooled(src), src

    index = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                print(f"[icon_cache] reference index… {done}/{len(items)}")
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is None:
                continue
            wh, iid, name, pooled, src = res
            b = index.setdefault(wh, {'ids': [], 'names': [], 'pooled': [], 'srcs': []})
            b['ids'].append(iid)
            b['names'].append(name)
            b['pooled'].append(pooled)
            b['srcs'].append(src)
    for wh, b in index.items():
        b['pooled'] = np.vstack(b['pooled']).astype(np.float32) if b['pooled'] \
            else np.zeros((0, POOL * POOL), np.float32)
    return index


def _associate_one(cache_bgra, W, H, ref_bucket):
    """
    Match a single cache icon against references of its slot size.
    Returns (item_id, name, score) for the best reference, or (None, None, best).
    """
    if ref_bucket is None or not ref_bucket['ids']:
        return None, None, -1.0
    q = _pooled(cache_bgra)
    sims = ref_bucket['pooled'] @ q                      # cosine (both unit-norm)
    k = min(SHORTLIST, sims.shape[0])
    cand = np.argpartition(sims, -k)[-k:]
    cq = _canon_gray(cache_bgra, W, H)
    mq = _alpha_mask(cache_bgra, W, H)
    best_i, best_sc = -1, -1.0
    for i in cand:
        ref = ref_bucket['srcs'][int(i)]
        rc = _canon_gray(ref, W, H)
        rm = _alpha_mask(ref, W, H)
        sc = _masked_corr(cq, mq, rc, rm)
        if sc > best_sc:
            best_sc, best_i = sc, int(i)
    if best_i < 0:
        return None, None, -1.0
    return ref_bucket['ids'][best_i], ref_bucket['names'][best_i], best_sc


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_cache_map():
    if os.path.exists(CACHE_MAP_PATH):
        try:
            with open(CACHE_MAP_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache_map(cmap):
    tmp = CACHE_MAP_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cmap, f)
    os.replace(tmp, CACHE_MAP_PATH)


# ---------------------------------------------------------------------------
# Public: incremental association
# ---------------------------------------------------------------------------

_assoc_lock = threading.Lock()


def associate_cache(items, load_src_fn, force=False):
    """
    Incrementally associate every icon-cache PNG to an item ID (visually).

    Only new/changed files (by mtime, unless force) are re-associated; existing
    entries are reused.  Persists and returns the cache-map:
        {filename: {item_id, name, score, mtime, preset, unknown, W, H}}
    """
    cache_dir = find_cache_dir()
    if not cache_dir:
        print("[icon_cache] cache folder not found — skipping (fallback to API templates)")
        return {}

    files = glob.glob(os.path.join(cache_dir, '*.png'))
    cmap = load_cache_map()

    todo = []
    for fp in files:
        fn = os.path.basename(fp)
        mt = os.path.getmtime(fp)
        prev = cmap.get(fn)
        if not force and prev and abs(prev.get('mtime', 0) - mt) < 1.0:
            continue
        todo.append((fp, fn, mt))

    if not todo:
        print(f"[icon_cache] {len(files)} cache icons, all associations current")
        return cmap

    print(f"[icon_cache] associating {len(todo)} new/changed of {len(files)} cache icons…")
    ref_index = _build_reference_index(items, load_src_fn)

    n_strong = n_preset = n_unknown = 0
    for i, (fp, fn, mt) in enumerate(todo):
        if i and i % 250 == 0:
            print(f"[icon_cache] associate… {i}/{len(todo)}")
        bgra = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.size == 0:
            continue
        h_px, w_px = bgra.shape[:2]
        W, H = slot_size(w_px, h_px)
        iid, name, score = _associate_one(bgra, W, H, ref_index.get((W, H)))
        entry = {'mtime': mt, 'W': W, 'H': H, 'score': round(float(score), 4)}
        if iid is not None and score >= STRONG:
            entry.update({'item_id': iid, 'name': name, 'preset': False, 'unknown': False})
            n_strong += 1
        elif iid is not None and score >= FLOOR:
            entry.update({'item_id': iid, 'name': name, 'preset': True, 'unknown': False})
            n_preset += 1
        else:
            entry.update({'item_id': iid, 'name': name, 'preset': False, 'unknown': True})
            n_unknown += 1
        cmap[fn] = entry

    # Drop entries whose cache files were deleted by the game
    live = {os.path.basename(f) for f in files}
    for fn in [k for k in cmap if k not in live]:
        del cmap[fn]

    save_cache_map(cmap)
    print(f"[icon_cache] associated: {n_strong} strong, {n_preset} preset, "
          f"{n_unknown} unknown")
    return cmap


# ---------------------------------------------------------------------------
# Public: cache → runtime templates
# ---------------------------------------------------------------------------

def build_cache_templates(items, template_fn, tint_fn, load_src_fn, force=False):
    """
    Produce runtime template records from the game's icon cache.

    template_fn(src_bgra, W, H, tint) -> (canon_vec, mask) | None
    tint_fn(item)                     -> BGR tint tuple
    load_src_fn(item)                 -> (base_bgra, has_alpha)

    Returns list of (footprint_wh, item_id, name, rotated, canon, mask), one
    native plus (for non-square icons) two 90°-rotated variants per associated
    cache icon.  `unknown` associations are skipped.
    """
    with _assoc_lock:
        cmap = associate_cache(items, load_src_fn, force=force)
    if not cmap:
        return []

    id_to_item = {it['id']: it for it in items}
    cache_dir = find_cache_dir()
    records = []
    for fn, entry in cmap.items():
        if entry.get('unknown') or not entry.get('item_id'):
            continue
        item = id_to_item.get(entry['item_id'])
        if item is None:
            continue
        fp = os.path.join(cache_dir, fn)
        bgra = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.size == 0:
            continue
        W, H = entry.get('W', 1), entry.get('H', 1)
        tint = tint_fn(item)
        name = item['name'] + (' (build)' if entry.get('preset') else '')
        variants = [((W, H), False, bgra)]
        if W != H:
            variants.append(((H, W), True, cv2.rotate(bgra, cv2.ROTATE_90_CLOCKWISE)))
            variants.append(((H, W), True, cv2.rotate(bgra, cv2.ROTATE_90_COUNTERCLOCKWISE)))
        for (fw, fh), rotated, src in variants:
            made = template_fn(src, fw, fh, tint)
            if made is None:
                continue
            canon, mask = made
            records.append(((fw, fh), item['id'], name, rotated, canon, mask))
    return records
