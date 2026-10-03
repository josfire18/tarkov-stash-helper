"""
Template catalog of *every* item (no category exclusion).

Sources
-------
``data/tmpl_src/<item_id>.png``   tarkov.dev base images (BGRA, ~63 px/slot).  One per
                                  item - ammo, guns, presets and containers included
                                  (the legacy engine threw those away; that alone made
                                  ~1/3 of a stash unidentifiable).
EFT Icon Cache ``<n>.png``        pixel-exact renders by the game of every icon the
                                  account ever displayed, *including the player's own
                                  weapon builds*.  The file numbers are recycled by the
                                  game, so ``cache_map.json`` goes stale.  We therefore
                                  re-associate on every catalog build: a cache icon whose
                                  composite is (near) identical to exactly one tarkov.dev
                                  icon of the same footprint becomes an extra ``cache``
                                  template of that item; one that matches nothing is a
                                  weapon *build*, stored as an anonymous ``build`` template
                                  (its identity is later decided from the printed label).

Representation
--------------
For stage-1 matching icons are stored *premultiplied* (``rgb * alpha`` and ``alpha``)
at 32 px/slot, grouped per footprint ``(W, H)``.  At match time the stack is
composited over the background colour *measured from the tile itself*, which covers
every rarity tint, hatched "unexamined" backgrounds and the top-of-panel gradient
without needing one composite per rarity.  The rarity tint table is still stored per
item and used as a soft prior.  Rotations (90 deg) are derived on demand from the
(H, W) stack and cached.

Persistence: one versioned ``.npz`` (``CATALOG_SCHEMA``) carrying a signature of its
sources; :func:`load_catalog` rebuilds when the signature changes.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import (CATALOG_PATH, CATALOG_SCHEMA, PRICES_PATH, SLOT, STAGE1_SLOT,
                     TMPL_SRC_DIR)

ASSOC_MAX = 8.0       # max composite distance for a cache icon to be assigned to an item
ASSOC_RATIO = 1.4     # ... and it must beat the best *different* item by this factor

CATEGORY_ORDER = [
    ('ammo', ('ammo', 'ammoBox')),
    ('weapon', ('gun', 'preset')),
    ('container', ('container',)),
    ('keys', ('keys',)),
    ('meds', ('meds', 'injectors')),
    ('mod', ('mods', 'suppressor', 'pistolGrip')),
    ('gear', ('armor', 'rig', 'backpack', 'helmet', 'armorPlate', 'glasses', 'headphones', 'wearable')),
    ('provisions', ('provisions',)),
    ('barter', ('barter',)),
    ('grenade', ('grenade',)),
]


def category_of(types) -> str:
    t = set(types or ())
    for name, members in CATEGORY_ORDER:
        if t & set(members):
            return name
    return 'other'


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class SizeStack:
    """All templates of one footprint at stage-1 resolution (premultiplied)."""
    W: int
    H: int
    idx: np.ndarray            # [N] global catalog row of each template
    prem: np.ndarray           # [N, h, w, 3] uint8  (rgb * alpha)
    alpha: np.ndarray          # [N, h, w]    uint8
    rotated: bool = False      # stack was produced by rotating the (H, W) stack
    rot: int = 0               # quarter turns applied: -1 = clockwise, +1 = counter-clockwise


@dataclass
class Catalog:
    ids: np.ndarray            # [M] item id ('' for anonymous builds)
    names: np.ndarray
    shorts: np.ndarray
    tint: np.ndarray           # backgroundColor name
    cats: np.ndarray           # category
    src: np.ndarray            # 'api' | 'cache' | 'build'
    preset: np.ndarray         # bool: tarkov.dev 'preset' (default weapon build) item
    tw: np.ndarray             # footprint width (slots)
    th: np.ndarray             # footprint height
    stacks: dict = field(default_factory=dict)       # (W,H) -> SizeStack
    meta: dict = field(default_factory=dict)
    _rot_cache: dict = field(default_factory=dict)
    _full_cache: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    # -- candidate lookup -------------------------------------------------
    def stacks_for(self, W: int, H: int, rotations: bool = True) -> list[SizeStack]:
        """Stacks whose templates can appear as a W x H tile: the (W, H) stack
        itself plus 90-degree rotations of the (H, W) stack (EFT lets the player
        rotate items; the icon is rotated with them)."""
        out = []
        s = self.stacks.get((W, H))
        if s is not None:
            out.append(s)
        if rotations and (H, W) in self.stacks:
            out.extend(self._rotated(H, W))
        return out

    def _rotated(self, W0: int, H0: int) -> list[SizeStack]:
        """cw and ccw rotations of the (W0, H0) stack -> (H0, W0) footprints."""
        key = (W0, H0)
        if key not in self._rot_cache:
            s = self.stacks[key]
            cw = SizeStack(H0, W0, s.idx, np.ascontiguousarray(np.rot90(s.prem, -1, axes=(1, 2))),
                           np.ascontiguousarray(np.rot90(s.alpha, -1, axes=(1, 2))), rotated=True, rot=-1)
            ccw = SizeStack(H0, W0, s.idx, np.ascontiguousarray(np.rot90(s.prem, 1, axes=(1, 2))),
                            np.ascontiguousarray(np.rot90(s.alpha, 1, axes=(1, 2))), rotated=True, rot=1)
            self._rot_cache[key] = [cw, ccw]
        return self._rot_cache[key]

    # -- full resolution --------------------------------------------------
    def full_icon(self, row: int) -> np.ndarray | None:
        """Full-resolution BGRA icon of a catalog row (re-read from its source)."""
        if row in self._full_cache:
            return self._full_cache[row]
        path = self.meta.get('paths', {}).get(int(row))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            W, H = int(self.tw[row]), int(self.th[row])
            if img.shape[:2] != (SLOT * H + 1, SLOT * W + 1):
                img = cv2.resize(img, (SLOT * W + 1, SLOT * H + 1), interpolation=cv2.INTER_AREA)
        else:
            img = None
        if len(self._full_cache) > 4000:
            self._full_cache.clear()
        self._full_cache[row] = img
        return img


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def footprint_of_image(img: np.ndarray) -> tuple[int, int]:
    """Footprint (W, H) implied by an icon's pixel size ``(63W+1) x (63H+1)``
    (some tarkov.dev images use 64 px/slot; rounding absorbs both)."""
    h, w = img.shape[:2]
    return max(1, int(round((w - 1) / SLOT))), max(1, int(round((h - 1) / SLOT)))


def to_stage1(bgra: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied (rgb*alpha, alpha) at ``STAGE1_SLOT`` px/slot from a BGRA icon."""
    s = STAGE1_SLOT
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    prem = bgra[:, :, :3].astype(np.float32) * a
    prem = cv2.resize(prem, (s * W, s * H), interpolation=cv2.INTER_AREA)
    al = cv2.resize(a[:, :, 0], (s * W, s * H), interpolation=cv2.INTER_AREA)
    return (np.clip(prem + 0.5, 0, 255).astype(np.uint8),
            np.clip(al * 255 + 0.5, 0, 255).astype(np.uint8))


def default_cache_dir() -> str | None:
    la = os.environ.get('LOCALAPPDATA')
    if not la:
        return None
    d = os.path.join(la, 'Temp', 'Battlestate Games', 'EscapeFromTarkov', 'Icon Cache', 'live')
    return d if os.path.isdir(d) else None


def _dir_signature(path: str | None, pattern: str = '*.png') -> str:
    if not path or not os.path.isdir(path):
        return 'none'
    n = 0
    newest = 0.0
    total = 0
    with os.scandir(path) as it:
        for e in it:
            if e.name.lower().endswith('.png'):
                st = e.stat()
                n += 1
                total += st.st_size
                newest = max(newest, st.st_mtime)
    return f'{n}:{total}:{int(newest)}'


CACHE_SLACK = 25      # the game keeps adding icons while it runs: rebuild only after this many new files


def _cache_count(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return 0
    with os.scandir(path) as it:
        return sum(1 for e in it if e.name.lower().endswith('.png'))


def _items_signature(prices_path: str) -> str:
    """Hash of the item fields the catalog is built from (identity, names, size, tint, kind).
    Prices are deliberately not part of it: they change every refresh (hourly) and must not
    trigger a catalog rebuild."""
    try:
        with open(prices_path, encoding='utf-8') as f:
            items = json.load(f)['items']
        rows = sorted((it['id'], it.get('name') or '', it.get('shortName') or '',
                       it.get('backgroundColor') or '', it.get('width'), it.get('height'),
                       sorted(it.get('types') or ())) for it in items)
        return hashlib.sha1(json.dumps(rows).encode()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError):
        return 'none'


def source_signature(prices_path: str, tmpl_dir: str) -> str:
    """Signature of the *mandatory* sources (schema, prices, tarkov.dev images).  The icon cache
    is tracked separately by file count (see :func:`load_catalog`)."""
    ps = _items_signature(prices_path)
    raw = f'{CATALOG_SCHEMA}|{STAGE1_SLOT}|{ps}|{_dir_signature(tmpl_dir)}'
    return hashlib.sha1(raw.encode()).hexdigest()


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _read(path: str):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def build_catalog(prices_path: str = PRICES_PATH, tmpl_dir: str = TMPL_SRC_DIR,
                  cache_dir: str | None = None, with_cache: bool = True,
                  log=print) -> Catalog:
    """Build the catalog from sources (see module docstring)."""
    t0 = time.time()
    with open(prices_path, encoding='utf-8') as f:
        items = json.load(f)['items']

    rows: list[dict] = []
    paths: dict[int, str] = {}
    stage: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}

    # ---- tarkov.dev base images -------------------------------------------
    todo = [(it, os.path.join(tmpl_dir, it['id'] + '.png')) for it in items]
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(lambda t: _read(t[1]) if os.path.exists(t[1]) else None, todo))
    n_missing = 0
    for (it, path), img in zip(todo, imgs):
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            n_missing += 1
            continue
        W, H = footprint_of_image(img)
        if img.shape[:2] != (SLOT * H + 1, SLOT * W + 1):
            img = cv2.resize(img, (SLOT * W + 1, SLOT * H + 1), interpolation=cv2.INTER_AREA)
        prem, al = to_stage1(img, W, H)
        r = len(rows)
        rows.append(dict(id=it['id'], name=it.get('name') or '', short=it.get('shortName') or '',
                         tint=it.get('backgroundColor') or 'default',
                         cat=category_of(it.get('types')), src='api', W=W, H=H,
                         preset='preset' in (it.get('types') or ())))
        stage[r] = (prem, al, W, H)
        paths[r] = path
    n_api = len(rows)

    # ---- EFT icon cache: re-associate every build ---------------------------
    n_cache = n_build = n_amb = 0
    cdir = cache_dir if cache_dir is not None else default_cache_dir()
    if with_cache and cdir:
        by_size: dict[tuple[int, int], list[int]] = {}
        for r in range(n_api):
            by_size.setdefault((rows[r]['W'], rows[r]['H']), []).append(r)
        comp = {k: _composites([stage[r] for r in v]) for k, v in by_size.items()}
        files = sorted(glob.glob(os.path.join(cdir, '*.png')))
        with ThreadPoolExecutor(max_workers=8) as ex:
            cimgs = list(ex.map(_read, files))
        for fpath, img in zip(files, cimgs):
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            W, H = footprint_of_image(img)
            if abs(img.shape[1] - (SLOT * W + 1)) > 3 or abs(img.shape[0] - (SLOT * H + 1)) > 3:
                continue                       # odd-sized special render (not a slot grid)
            if img.shape[:2] != (SLOT * H + 1, SLOT * W + 1):
                img = cv2.resize(img, (SLOT * W + 1, SLOT * H + 1), interpolation=cv2.INTER_AREA)
            prem, al = to_stage1(img, W, H)
            ids = by_size.get((W, H), [])
            best_r, best_d, second_d = None, 1e9, 1e9
            if ids:
                d = _composite_dist(prem, al, comp[(W, H)])
                order = np.argsort(d)
                best_r, best_d = ids[int(order[0])], float(d[order[0]])
                # second-best among *different items*
                for j in order[1:6]:
                    if rows[ids[int(j)]]['id'] != rows[best_r]['id']:
                        second_d = float(d[j])
                        break
            # A cache icon is the game's own render of the item, so it matches its
            # tarkov.dev icon to within a few levels (measured 2-5 on stash1); it is a
            # *clear* match when it beats the best different item by 1.4x.
            clear = best_r is not None and best_d <= ASSOC_MAX and second_d >= ASSOC_RATIO * best_d
            if clear:
                if best_d > 0.6:               # genuinely different render => worth keeping
                    r = len(rows)
                    base = rows[best_r]
                    rows.append(dict(id=base['id'], name=base['name'], short=base['short'],
                                     tint=base['tint'], cat=base['cat'], src='cache', W=W, H=H,
                                     preset=base['preset']))
                    stage[r] = (prem, al, W, H)
                    paths[r] = fpath
                    n_cache += 1
            elif best_r is not None and best_d <= ASSOC_MAX:
                n_amb += 1                     # twins (e.g. colour variants): the api templates cover them
            elif al.mean() > 8:
                r = len(rows)
                rows.append(dict(id='', name=f'build #{os.path.basename(fpath)[:-4]}', short='',
                                 tint='default', cat='build', src='build', W=W, H=H, preset=False))
                stage[r] = (prem, al, W, H)
                paths[r] = fpath
                n_build += 1

    # ---- stacks ---------------------------------------------------------------
    by_size_rows: dict[tuple[int, int], list[int]] = {}
    for r, row in enumerate(rows):
        by_size_rows.setdefault((row['W'], row['H']), []).append(r)
    stacks = {}
    for (W, H), rs in by_size_rows.items():
        stacks[(W, H)] = SizeStack(W, H, np.asarray(rs, np.int32),
                                   np.stack([stage[r][0] for r in rs]),
                                   np.stack([stage[r][1] for r in rs]))
    cat = Catalog(
        ids=np.array([r['id'] for r in rows]), names=np.array([r['name'] for r in rows]),
        shorts=np.array([r['short'] for r in rows]), tint=np.array([r['tint'] for r in rows]),
        cats=np.array([r['cat'] for r in rows]), src=np.array([r['src'] for r in rows]),
        preset=np.array([r['preset'] for r in rows], bool),
        tw=np.array([r['W'] for r in rows], np.int16), th=np.array([r['H'] for r in rows], np.int16),
        stacks=stacks,
        meta={'paths': paths, 'built': time.time(), 'n_api': n_api, 'n_cache': n_cache,
              'n_build': n_build, 'n_ambiguous_cache': n_amb, 'n_missing_src': n_missing})
    log(f'[catalog] {n_api} api + {n_cache} cache + {n_build} build templates '
        f'({n_missing} items without image, {n_amb} ambiguous cache icons) in {time.time() - t0:.1f}s')
    return cat


def _composites(entries) -> np.ndarray:
    """Composite stage-1 templates over mid grey -> float32 [N, h, w, 3] (for association)."""
    g = 40.0
    prem = np.stack([e[0] for e in entries]).astype(np.float32)
    al = np.stack([e[1] for e in entries]).astype(np.float32)[..., None] / 255.0
    return prem + (1 - al) * g


def _composite_dist(prem: np.ndarray, al: np.ndarray, comp: np.ndarray) -> np.ndarray:
    a = al.astype(np.float32)[..., None] / 255.0
    c = prem.astype(np.float32) + (1 - a) * 40.0
    # inset to ignore the border line / outermost pixels
    return np.abs(comp[:, 1:-1, 1:-1] - c[None, 1:-1, 1:-1]).mean(axis=(1, 2, 3))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_catalog(cat: Catalog, path: str = CATALOG_PATH, signature: str = '', cache_count: int = 0) -> None:
    arrs = {
        'schema': np.array(CATALOG_SCHEMA), 'signature': np.array(signature), 'cache_count': np.array(cache_count),
        'ids': cat.ids, 'names': cat.names, 'shorts': cat.shorts, 'tint': cat.tint,
        'cats': cat.cats, 'src': cat.src, 'preset': cat.preset, 'tw': cat.tw, 'th': cat.th,
        'meta_json': np.array(json.dumps({k: v for k, v in cat.meta.items() if k != 'paths'})),
        'paths_json': np.array(json.dumps({str(k): v for k, v in cat.meta.get('paths', {}).items()})),
    }
    for (W, H), s in cat.stacks.items():
        arrs[f'idx_{W}x{H}'] = s.idx
        arrs[f'prem_{W}x{H}'] = s.prem
        arrs[f'alpha_{W}x{H}'] = s.alpha
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def _load_npz(path: str):
    z = np.load(path, allow_pickle=False)
    if int(z['schema']) != CATALOG_SCHEMA:
        return None, '', 0
    stacks = {}
    for k in z.files:
        if k.startswith('idx_'):
            W, H = (int(v) for v in k[4:].split('x'))
            stacks[(W, H)] = SizeStack(W, H, z[k], z[f'prem_{W}x{H}'], z[f'alpha_{W}x{H}'])
    meta = json.loads(str(z['meta_json']))
    meta['paths'] = {int(k): v for k, v in json.loads(str(z['paths_json'])).items()}
    cat = Catalog(ids=z['ids'], names=z['names'], shorts=z['shorts'], tint=z['tint'], cats=z['cats'],
                  src=z['src'], preset=z['preset'], tw=z['tw'], th=z['th'], stacks=stacks, meta=meta)
    return cat, str(z['signature']), int(z['cache_count'])


def load_catalog(path: str = CATALOG_PATH, prices_path: str = PRICES_PATH,
                 tmpl_dir: str = TMPL_SRC_DIR, cache_dir: str | None = None,
                 force_rebuild: bool = False, cache_slack: int = CACHE_SLACK, log=print) -> Catalog:
    """Load the persisted catalog, rebuilding it when the schema, the prices or the tarkov.dev
    images changed, or when the EFT icon cache grew/shrank by ``cache_slack`` files or more
    (the running game adds icons constantly; a 35 s rebuild per new icon would be absurd)."""
    cdir = cache_dir if cache_dir is not None else default_cache_dir()
    sig = source_signature(prices_path, tmpl_dir)
    n_cache = _cache_count(cdir)
    if not force_rebuild and os.path.exists(path):
        try:
            cat, old, old_n = _load_npz(path)
            if (cat is not None and old == sig
                    and abs(n_cache - old_n) < max(1, cache_slack)):
                return cat
        except Exception as e:                       # corrupt / partial file => rebuild
            log(f'[catalog] cannot load {path}: {e}')
    cat = build_catalog(prices_path, tmpl_dir, cdir, log=log)
    save_catalog(cat, path, sig, n_cache)
    return cat
