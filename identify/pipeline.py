"""
End-to-end identification: ``scan(image) -> list[Detection]``.

    grid.detect_grid      panels + lattice (all rows/cols, clipped edge rows)
    segment.segment_panel item footprints from the drawn borders
    match.stage1          masked residual vs same-footprint catalog templates (all items)
    dino                  stage 2: appearance re-rank of the shortlist (optional)
    ocr                   stage 3: printed short name -> authority for ammo / weapons,
                          tie-break elsewhere (optional)
    fir                   Found-in-Raid tick (three-valued)

Fusion
------
Every candidate gets one score (higher is better)::

    S = -W_RES * residual  +  W_DINO * cosine  +  W_OCR * g(ocr_agreement)

with ``g`` mapping the fuzzy OCR score to [-1, 1] (0 when nothing was readable).
Ammo calibres share one silhouette, so for ammo the OCR term is dominant (the legacy
engine had to throw ammo away for exactly that reason).  Built weapons never match a
base icon, so for weapons the identity comes from the printed short name
(``_weapon_by_label``) while the *footprint* comes from the borders.

The reported confidence is a logistic function of four evidence features (margin to the
runner-up, residual of the winner, OCR agreement, DINO agreement) with coefficients
fitted on the labelled stash (``CALIB``); anything below ``uncertain_below`` is flagged
``uncertain`` instead of being guessed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from . import dino as dino_mod
from . import digits as digits_mod
from . import ocr as ocr_mod
from .catalog import Catalog, load_catalog
from .config import EngineSettings, SLOT
from .fir import detect_fir
from .grid import GridResult, detect_grid
from .match import Cand, Tile, normalize_tile, stage1
from .match import enable_torch as match_enable_torch, disable_torch as match_disable_torch
from .segment import segment_panel

# fusion weights (tuned on data/eval/stash1.truth.full.json, see test_scan.py --score)
W_RES = 1.0
W_DINO = 18.0
W_OCR = 5.0
OCR_POOL_DELTA = 7.0       # OCR may consider items whose residual is within this of the best
RES_PLAUSIBLE = 8.5        # mean level error above which a visual match needs the label's support
OCR_AUTH = 90.0            # fuzzy score at/above which a label is considered a real read
# logistic calibration of the reported confidence: p = sigmoid(b + sum(w * feature)); fitted by
# `python -m identify.calibrate` on degraded variants of the labelled stash (see that module)
CALIB = {'b': 1.753, 'margin': 0.372, 'res': -0.311, 'ocr': 3.225, 'dino': 0.947}


@dataclass
class Detection:
    panel: int
    col: int
    row: int
    w: int
    h: int
    rotated: bool
    item_id: str
    name: str
    confidence: float
    uncertain: bool
    evidence: dict = field(default_factory=dict)
    count: int | None = None
    fir: bool | None = None
    rect: tuple = (0, 0, 0, 0)          # pixel rect (x, y, w, h) of the visible footprint
    clipped: bool = False
    category: str = ''
    short: str = ''

    def to_legacy(self) -> dict:
        """The record shape the legacy engine produced (what app.py consumes)."""
        x, y, w, h = self.rect
        return {'col': self.col, 'row': self.row, 'W': self.w, 'H': self.h,
                'item_id': self.item_id, 'name': self.name, 'rotated': self.rotated,
                'source': 'v2', 'score': round(self.confidence * 100, 1), 'fir': self.fir,
                'panel': self.panel, 'px': x, 'py': y, 'pw': w, 'ph': h,
                'uncertain': self.uncertain, 'count': self.count}

    def as_dict(self) -> dict:
        d = asdict(self)
        d['rect'] = list(self.rect)
        return d


@dataclass
class ScanResult:
    detections: list
    grid: GridResult
    items: list = field(default_factory=list)       # every segmented footprint (incl. empty)
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _g(ocr_score: float | None) -> float:
    if ocr_score is None:
        return 0.0
    return max(-1.0, min(1.0, (ocr_score - 60.0) / 35.0))


class Engine:
    """Holds the heavy, reusable state (catalog, embeddings, settings)."""

    def __init__(self, settings: EngineSettings | None = None, catalog: Catalog | None = None):
        self.s = settings or EngineSettings()
        self.cat = catalog if catalog is not None else load_catalog()
        self.store: dino_mod.EmbeddingStore | None = None
        ocr_mod.set_tesseract_cmd(self.s.tesseract_cmd)
        self._api = None
        self._api_fold: list = []
        self._id_row: dict = {}
        self.preset_base: dict = {}
        self._hits_cache: dict = {}
        if self.s.accelerate:
            match_enable_torch(self.s.device)
        else:
            match_disable_torch()
        self._api_rows()

    # ------------------------------------------------------------------
    def _use_dino(self) -> bool:
        return bool(self.s.use_dino and dino_mod.available(self.s.device))

    def _use_ocr(self) -> bool:
        return bool(self.s.use_ocr and ocr_mod.tesseract_available())

    # ------------------------------------------------------------------
    def scan(self, img_bgr: np.ndarray) -> ScanResult:
        T: dict[str, float] = {}
        t0 = time.perf_counter()
        warnings: list[str] = []
        grid = detect_grid(img_bgr, self.s.pitch_hint)
        warnings += grid.warnings
        T['grid'] = time.perf_counter() - t0

        t = time.perf_counter()
        items = []
        for pi, panel in enumerate(grid.panels):
            for it in segment_panel(img_bgr, panel, pi):
                items.append((it, panel))
        T['segment'] = time.perf_counter() - t

        # ---- tiles + OCR (labels) -----------------------------------------------------
        # OCR runs first: the printed names tell stage 1 which templates must be scored
        # exactly (the numpy path prunes with a coarse pass), and OCR needs no stage-1 result.
        t = time.perf_counter()
        work = []
        for it, panel in items:
            if it.empty:
                continue
            tile = normalize_tile(img_bgr, it.rect, it.w, panel.pitch_x, panel.pitch_y,
                                  it.clipped_top, it.clipped_bottom)
            if tile is not None:
                work.append({'it': it, 'panel': panel, 'tile': tile, 'ocr_all': []})
        if self._use_ocr():
            flat, owner = [], []
            for i, w in enumerate(work):
                if w['it'].clipped_top:
                    continue
                st = ocr_mod.label_strip(img_bgr, w['it'].rect, w['panel'].pitch_x, w['panel'].pitch_y)
                if st is None:
                    continue
                for v in ocr_mod.variants_of(st):
                    flat.append(v)
                    owner.append(i)
            texts = ocr_mod.read_strips(flat)
            for i, txt in zip(owner, texts):
                if txt and txt not in work[i]['ocr_all']:
                    work[i]['ocr_all'].append(txt)
        T['ocr'] = time.perf_counter() - t

        # ---- stage 1 ---------------------------------------------------------------
        t = time.perf_counter()
        for w in work:
            force = set()
            for txt in w['ocr_all']:
                force.update(self._label_hits(txt, w['tile'].W))
            w['cands'] = stage1(self.cat, w['tile'], top_k=None, rotations=True, force_ids=force)
        T['stage1'] = time.perf_counter() - t

        # ---- candidate pools + OCR agreement ------------------------------------------
        t = time.perf_counter()
        for w in work:
            self._prepare_pool(w)
        T['pool'] = time.perf_counter() - t

        # ---- DINO ------------------------------------------------------------------
        t = time.perf_counter()
        if self._use_dino():
            self._dino_pass(img_bgr, work)
        T['dino'] = time.perf_counter() - t

        # ---- fusion + FIR ------------------------------------------------------------
        t = time.perf_counter()
        dets = []
        for w in work:
            d = self._decide(img_bgr, w)
            if d is not None:
                dets.append(d)
        dets.sort(key=lambda d: (d.panel, d.row, d.col))
        T['fuse'] = time.perf_counter() - t
        T['total'] = time.perf_counter() - t0
        if self.store is not None:
            self.store.save()
        return ScanResult(dets, grid, [it for it, _ in items], T, warnings)

    # ------------------------------------------------------------------
    def _dino_pass(self, img_bgr, work) -> None:
        """Stage 2: cosine similarity between the tile and each shortlisted template.

        Both sides have their label / bottom bands painted with the background colour.
        Un-clipped, un-rotated candidates use the disk-cached catalog embeddings; rotated
        ones and everything for viewport-clipped tiles are embedded on the fly."""
        if self.store is None:
            self.store = dino_mod.EmbeddingStore(self.cat)
        q_imgs, q_idx = [], []
        for i, w in enumerate(work):
            hi = _hi_tile(img_bgr, w['it'], w['panel'], w['tile'])
            w['hi'] = hi
            if hi is None:
                continue
            w['top'] = w['dset']
            bg = w['tile'].bg.astype(np.uint8)
            q_imgs.append(dino_mod.blank_bands(hi, bg, SLOT, w['tile'].clip))
            q_idx.append(i)
        if not q_imgs:
            return
        qe = dino_mod.embed(q_imgs)
        direct_rows, fly_imgs, fly_keys = [], [], []
        for i in q_idx:
            tile = work[i]['tile']
            for c in work[i]['top']:
                if c.rot == 0 and not tile.clip:
                    direct_rows.append(c.row)
                    continue
                full = self.cat.full_icon(c.row)
                if full is None:
                    continue
                bg = TINT.get(str(self.cat.tint[c.row]), TINT['default'])
                comp = dino_mod.composite_icon(full, bg, c.rot)
                if tile.clip:
                    vis = int(round(tile.vis_rows / 32.0 * 64.0))
                    comp = comp[:vis] if tile.clip == 'bottom' else comp[-vis:]
                fly_imgs.append(dino_mod.blank_bands(comp, bg, SLOT, tile.clip))
                fly_keys.append((i, c.row, c.rot))
        direct_rows = sorted(set(direct_rows))
        demb = dict(zip(direct_rows, self.store.get(direct_rows))) if direct_rows else {}
        femb = dict(zip(fly_keys, dino_mod.embed(fly_imgs))) if fly_imgs else {}
        for qi, i in enumerate(q_idx):
            tile = work[i]['tile']
            sims = {}
            for c in work[i]['top']:
                v = demb.get(c.row) if (c.rot == 0 and not tile.clip) else femb.get((i, c.row, c.rot))
                if v is not None:
                    sims[(c.row, c.rot)] = float(qe[qi] @ v)
            work[i]['sims'] = sims

    # ------------------------------------------------------------------
    def _prepare_pool(self, w) -> None:
        """Candidate pool, per-candidate OCR agreement and the DINO shortlist.

        The pool is the best residuals *plus* every item of this footprint whose
        printed short name matches the label: a modded item (mount + sight) draws far
        from its base icon, so only the label can bring it into the race."""
        cat = self.cat
        cands = w['cands']
        best_res = cands[0].score if cands else 0.0
        K = self.s.top_k
        top_rows = {c.row for c in cands[:K]}
        by_id = {(str(cat.ids[c.row]) or f'#{c.row}'): c for c in cands}
        it_w = w['tile'].W
        texts = w.get('ocr_all') or []
        oid: dict[str, float] = {}
        best_text, best_top = '', -1.0
        for txt in texts:
            hits = self._label_hits(txt, it_w)
            top_here = max(hits.values(), default=0.0)
            for iid, sc in hits.items():
                if sc > oid.get(iid, -1.0):
                    oid[iid] = sc
            if top_here > best_top:
                best_text, best_top = txt, top_here
        pool = [c for i, c in enumerate(cands)
                if (i < K or c.score <= best_res + OCR_POOL_DELTA)][:400]
        have = {c.row for c in pool}
        for iid, sc in sorted(oid.items(), key=lambda kv: -kv[1])[:12]:
            c = by_id.get(iid)
            if c is not None and sc >= 60 and c.row not in have:
                pool.append(c)
                have.add(c.row)
        oscores = {}
        if texts:
            for c in pool:
                oscores[c.row] = oid.get(str(cat.ids[c.row]), 25.0)
        w['ocr_best'] = best_text
        w['pool'] = pool
        w['oscores'] = oscores
        # DINO shortlist: best residuals + best OCR agreement
        extra = sorted((c for c in pool if oscores.get(c.row, 0) >= 60 and c.row not in top_rows),
                       key=lambda c: -oscores[c.row])[:6]
        w['dset'] = list(cands[:K]) + extra

    def _label_hits(self, text: str, width: int = 1) -> dict:
        """item id -> fuzzy label score for the best matches of ``text`` among all
        catalog items (cached per text; presets excluded - they map to their base gun)."""
        h = self._hits_cache.get((text, width))
        if h is None:
            rows = self._api_rows()
            idxs = ocr_mod.prefilter(text, self._api_fold, limit=80, folded=True)
            h = {}
            for i in idxs:
                r = int(rows[i])
                sc = ocr_mod.fuzzy_score(text, str(self.cat.names[r]), str(self.cat.shorts[r]), width)
                if sc >= 40:
                    iid = str(self.cat.ids[r])
                    if sc > h.get(iid, -1.0):
                        h[iid] = sc
            if len(self._hits_cache) > 5000:
                self._hits_cache.clear()
            self._hits_cache[(text, width)] = h
        return h

    # ------------------------------------------------------------------
    def _decide(self, img_bgr, w) -> Detection | None:
        cat = self.cat
        it, panel, tile, cands = w['it'], w['panel'], w['tile'], w['cands']
        if not cands:
            return None
        text = w.get('ocr_best', '') or ''
        sims = w.get('sims') or {}
        have_dino = bool(sims)
        best_res = cands[0].score
        pool = w['pool']
        ocr_scores = w['oscores']
        sim_floor = (min(sims.values()) - 0.05) if sims else 0.0

        scored = []
        for c in pool:
            s_dino = sims.get((c.row, c.rot))
            S = -W_RES * c.score
            if have_dino:
                S += W_DINO * (s_dino if s_dino is not None else sim_floor)
            o = ocr_scores.get(c.row)
            if text:
                S += W_OCR * _g(o)
            scored.append((S, c, s_dino, o))
        scored.sort(key=lambda t: -t[0])
        top = scored[0]
        chosen_S, c, s_dino, o = top

        # A build template (anonymous cache render of a modded item) or a poor visual match
        # only says what the footprint looks like: the identity comes from the printed label.
        auth = None
        if cat.src[c.row] == 'build' or (str(cat.cats[c.row]) == 'weapon' and c.score > 3.0) \
                or best_res > 9.0:
            auth = self._label_authority(w.get('ocr_all') or [], tile,
                                         prefer_weapon=str(cat.cats[c.row]) in ('weapon', 'build'))
        elif text and (o is None or o < 60):
            # label conflict: the picture says one thing, a clean unambiguous printed name
            # says another (e.g. a 2x1 suppressor whose label names a 1x1 flash hider)
            auth = self._label_authority(w.get('ocr_all') or [], tile, prefer_weapon=False)
            if auth is not None and str(cat.ids[auth[0]]) == str(cat.ids[c.row]):
                auth = None
        item_id = str(cat.ids[c.row])
        name = str(cat.names[c.row])
        evidence = {'residual': round(c.score, 3), 'stage1_best': round(best_res, 3),
                    'dino': None if s_dino is None else round(s_dino, 4),
                    'ocr_text': text, 'ocr': None if o is None else round(o, 1),
                    'source': str(cat.src[c.row])}
        chosen_row = c.row
        twins = 1
        if auth is not None:
            row, sc, twins = auth
            item_id, name = str(cat.ids[row]), str(cat.names[row])
            evidence.update(ocr_authority=True, ocr=round(sc, 1), source='ocr-authority', twins=twins)
            o = sc
            chosen_row = row
        elif item_id == '':
            evidence['note'] = 'modded item / build with unreadable label'
        if item_id in self.preset_base:          # a default-build preset icon -> its base gun
            base = self.preset_base[item_id]
            evidence['preset'] = name
            item_id, name, chosen_row = str(cat.ids[base]), str(cat.names[base]), base

        margin = 5.0
        for S2, c2, _, _ in scored[1:]:
            if str(cat.ids[c2.row]) != item_id:
                margin = chosen_S - S2
                break
        evidence['margin'] = round(margin, 3)
        alts, seen = [], {item_id}
        for S2, c2, d2, o2 in scored[1:]:
            iid2 = str(cat.ids[c2.row])
            if iid2 not in seen:
                seen.add(iid2)
                alts.append({'name': str(cat.names[c2.row]), 'S': round(S2, 2), 'res': round(c2.score, 2),
                             'dino': None if d2 is None else round(d2, 3),
                             'ocr': None if o2 is None else round(o2, 1)})
            if len(alts) == 3:
                break
        evidence['alternatives'] = alts
        evidence['catalog_row'] = int(chosen_row)
        evidence['rot'] = int(c.rot)

        feats = {'margin': min(margin, 6.0), 'res': min(c.score, 20.0),
                 'ocr': _g(o) if (text and o is not None) else 0.0,
                 'dino': (s_dino - 0.5) if s_dino is not None else 0.0}
        z = CALIB['b'] + sum(CALIB[k] * v for k, v in feats.items())
        if item_id == '':
            z -= 4.0
        if twins > 1:
            z -= 1.5 * math.log(twins)
        conf = _sigmoid(z)
        # absolute plausibility: a large residual is only acceptable when the printed name
        # vouches for the item.  Without it the tile is something the catalog does not
        # contain (new item, modded build, unknown icon) - flag it instead of guessing.
        label_ok = bool(text) and o is not None and o >= 85
        if c.score > RES_PLAUSIBLE and not label_ok:
            conf = min(conf, 0.35)
            evidence['note'] = 'no close visual match and no readable label'
        evidence['features'] = {k: round(v, 3) for k, v in feats.items()}

        x, y, wpx, hpx = it.rect
        clipped = bool(it.clipped_top or it.clipped_bottom)
        hi = w.get('hi')
        fir = None
        if not clipped:
            if hi is None:
                hi = _hi_tile(img_bgr, it, panel, tile)
            fir = detect_fir(hi)
        count, count_text = digits_mod.read_count(hi) if not clipped else (None, '')
        if count_text:
            evidence['count_text'] = count_text
        H_det = c.H if it.clipped_bottom else it.h
        return Detection(
            panel=it.panel, col=it.col, row=it.row, w=it.w, h=H_det, rotated=bool(c.rotated),
            item_id=item_id, name=name, confidence=float(conf),
            uncertain=bool(conf < self.s.uncertain_below or item_id == ''),
            evidence=evidence, count=count, fir=fir, rect=(x, y, wpx, hpx), clipped=clipped,
            category=str(cat.cats[chosen_row]), short=str(cat.shorts[chosen_row]))

    # ------------------------------------------------------------------
    def _api_rows(self) -> np.ndarray:
        if self._api is None:
            c = self.cat
            self._api = np.where((c.src == 'api') & (~c.preset))[0]
            self._api_fold = [ocr_mod.fold(str(c.shorts[r]) or str(c.names[r])) for r in self._api]
            self._id_row = {str(c.ids[r]): int(r) for r in self._api}
            # preset -> base gun: longest gun name that prefixes the preset's name
            guns = [(str(c.names[r]), int(r)) for r in self._api if str(c.cats[r]) == 'weapon']
            guns.sort(key=lambda t: -len(t[0]))
            self.preset_base = {}
            for r in np.where(c.preset & (c.src == 'api'))[0]:
                nm = str(c.names[r])
                for gname, grow in guns:
                    if nm.startswith(gname):
                        self.preset_base[str(c.ids[r])] = grow
                        break
        return self._api

    def _label_authority(self, texts: list, tile: Tile, prefer_weapon: bool):
        """Identity from the printed short name over *all* base items (any footprint no
        larger than the tile: a modded item draws bigger than its base; a viewport-clipped
        tile may hide up to 3 rows).  Needs a confident read (>= ``OCR_AUTH``) that beats
        any *different* name by 5 points.  Equal-score twins (same short name) are resolved
        by preferring base weapons when the picture looks like a weapon, else footprint
        closeness, and are counted so the confidence can be lowered.
        Returns ``(row, score, n_twins)`` or None."""
        self._api_rows()
        cat = self.cat
        max_area = tile.W * (tile.H + (3 if tile.clip else 0))
        best = None
        for txt in texts:
            hits = self._label_hits(txt, tile.W)
            sc = [(s_, self._id_row[i]) for i, s_ in hits.items() if i in self._id_row]
            sc = [(s_, r) for s_, r in sc
                  if int(cat.tw[r]) * int(cat.th[r]) <= max_area or str(cat.cats[r]) == 'weapon']
            if not sc:
                continue
            sc.sort(key=lambda t: -t[0])
            if sc[0][0] >= OCR_AUTH and (best is None or sc[0][0] > best[0][0][0]):
                best = (sc, txt)
        if best is None:
            return None
        sc = best[0]
        top = sc[0][0]
        rivals = {ocr_mod.canon_nospace(str(cat.shorts[r])) for s_, r in sc if s_ >= top - 5.0}
        if len(rivals) > 1:
            return None
        twins = [r for s_, r in sc if s_ >= top - 0.01]
        area = tile.W * tile.H

        def key(r):
            weap = str(cat.cats[r]) == 'weapon'
            return (0 if (prefer_weapon and weap) else 1, abs(int(cat.tw[r]) * int(cat.th[r]) - area), r)
        twins.sort(key=key)
        return twins[0], float(top), len(twins)


TINT = dino_mod.TINT_BGR


def _hi_tile(img_bgr, it, panel, tile=None):
    """Tile at 64 px/slot (FIR + DINO input).  Viewport-clipped tiles keep only their
    visible rows; None when the footprint leaves the image."""
    x, y, w, h = it.rect
    if x < 0 or y < 0 or x + w > img_bgr.shape[1] or y + h > img_bgr.shape[0]:
        return None
    if it.clipped_top or it.clipped_bottom:
        vis = int(round((h - 1) / panel.pitch_y * 64))
        if vis < 12:
            return None
        return cv2.resize(img_bgr[y:y + h, x:x + w], (64 * it.w, vis), interpolation=cv2.INTER_AREA)
    return cv2.resize(img_bgr[y:y + h, x:x + w], (64 * it.w, 64 * it.h), interpolation=cv2.INTER_AREA)


_default: Engine | None = None


def get_engine(settings: EngineSettings | None = None) -> Engine:
    global _default
    if _default is None or (settings is not None and _default.s != settings):
        _default = Engine(settings)
    return _default


def scan(image: np.ndarray, settings: EngineSettings | None = None) -> list[Detection]:
    """Identify every item in a BGR screenshot.  Convenience wrapper around
    :meth:`Engine.scan` using a shared engine."""
    return get_engine(settings).scan(image).detections
