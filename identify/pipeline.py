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
import threading
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from . import dino as dino_mod
from . import digits as digits_mod
from . import ocr as ocr_mod
from .catalog import Catalog, load_catalog
from .config import EngineSettings, LEARNED_PATH, SLOT
from .learned import LearnedNames, content_key
from .fir import detect_fir
from .grid import GridResult, detect_grid
from .match import Cand, Tile, normalize_tile, stage1
from .match import enable_torch as match_enable_torch, disable_torch as match_disable_torch
from .segment import segment_panel
from . import anchors as anchors_mod
from . import fontlabel as fontlabel_mod
from . import tilecache
from .catalog import default_cache_dir

# fusion weights (tuned on data/eval/stash1.truth.full.json, see test_scan.py --score)
W_RES = 1.0
W_DINO = 18.0
W_OCR = 5.0
OCR_POOL_DELTA = 7.0       # OCR may consider items whose residual is within this of the best
RES_PLAUSIBLE = 8.5        # mean level error above which a visual match needs the label's support
OCR_AUTH = 90.0            # fuzzy score at/above which a label is considered a real read
LABEL_PARTIAL_CAP = 94.0   # best score a non-exact label match can get (exact = 100)
LABEL_RIVAL_GAP = 5.0      # a label this much better than the chosen item's gets a say
LITERAL_RES = 5.0          # a stage-1 residual this low with LITERAL_DINO is a near pixel-exact match
LITERAL_DINO = 0.85
ANCHOR_SHORTLIST = 6       # icon-cache renders checked exactly per tile (Anchor 1)
TIE_MARGIN = 0.1           # fused-score gap below which two differently named items are indistinguishable
TIE_CONF_CAP = 0.5         # ...and the answer is a coin flip, so it may not be reported as certain
LEARN_MAX_RES = 4.0        # stage-1 residual of a near-literal match to a cached icon
NAMED_DINO_SLACK = 0.03    # a named candidate this close in DINO to a nameless winner may name it
LEARN_DISTINCT = 2.0       # learn only if every different look-alike is at least this x worse
LEARN_CONFIRMATIONS = 2    # independent confirmations before a learned name is used
TWIN_GAP_OK = 2.0          # fused-score lead that settles same-label twins by picture (see docs/accuracy.md)
TWIN_GAP_OK_RES = 6.0      # ...when the picture score is the residual alone (no DINO: colour variants differ by little)
LABEL_VERIFIED_CONF = 0.97  # exact, unambiguous label read of an item that fits the footprint
WEAPON_SURE_CONF = 0.95    # certainly a gun (guns are skipped, the exact gun does not matter)
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

    def to_record(self) -> dict:
        """The flat dict shape app.py's scan routes consume."""
        x, y, w, h = self.rect
        return {'col': self.col, 'row': self.row, 'W': self.w, 'H': self.h,
                'item_id': self.item_id, 'name': self.name, 'rotated': self.rotated,
                'source': 'v2', 'score': round(self.confidence * 100, 1), 'fir': self.fir,
                'panel': self.panel, 'px': x, 'py': y, 'pw': w, 'ph': h,
                'uncertain': self.uncertain, 'count': self.count, 'category': self.category}

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


def literal_beats_label(residual: float, dino: float | None, label_score: float) -> bool:
    """True when a tile is a near pixel-exact, DINO-confirmed match to a template and the only
    contrary evidence is a partial label read: a letter or two misread ("Egg" -> "Mass") must not
    overrule the picture.  An exact (100) read still can, since look-alike rounds sit within a
    hair of each other in pixels."""
    return (label_score < 100.0 and residual <= LITERAL_RES
            and dino is not None and dino >= LITERAL_DINO)


def is_tie(gap: float | None) -> bool:
    """Two differently named best candidates whose fused scores are equal within noise."""
    return gap is not None and gap < TIE_MARGIN


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
        self._lock = threading.Lock()      # scans are serialised (hotkey + button can race)
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
        self.learned = LearnedNames(self.s.extra.get('learned_path', LEARNED_PATH))
        self._row_key: dict = {}
        self._apply_learned()
        self.tile_cache = tilecache.TileCache()
        self.anchor_idx = None
        self.labeler = None
        self._fp_rows: dict = {}
        self._vis_cache: dict = {}
        self._phases: dict = {}
        self._init_anchors()

    # ------------------------------------------------------------------
    def _init_anchors(self) -> None:
        """Anchor 1 (icon-cache renders) and Anchor 2 (the game's label font).  Either may be
        unavailable (no game cache / font): it is then simply off."""
        ex = self.s.extra
        if not ex.get('anchors', True):
            return
        canon = {k: str(self.cat.ids[v]) for k, v in self.preset_base.items()}
        self._canon = canon
        try:
            cdir = ex.get('cache_dir') or default_cache_dir()
            if cdir:
                self.anchor_idx = anchors_mod.CacheIndex(cdir, self.cat.meta.get('cache_assoc') or {},
                                                         _CertainBindings(self.learned), canon)
        except Exception as e:                     # pragma: no cover - never fatal
            print(f'[anchors] icon-cache index unavailable: {e}')
        try:
            self.labeler = fontlabel_mod.LabelRenderer.create(log=lambda *a: None)
        except Exception as e:                     # pragma: no cover
            print(f'[anchors] label font unavailable: {e}')

    def reset_anchors(self) -> None:
        """Forget certain bindings made in memory and the tile cache (eval: clean state)."""
        self.tile_cache.clear()
        if self.anchor_idx is not None:
            self.anchor_idx.reset_bindings()

    def _fp_candidates(self, W: int, H: int) -> list:
        """api rows (no presets) whose item can occupy a W x H tile."""
        k = (W, H)
        if k not in self._fp_rows:
            t = Tile(np.zeros((1, 1, 3), np.uint8), W, H)
            self._fp_rows[k] = [int(r) for r in self._api_rows() if self._footprint_ok(int(r), t, False)]
        return self._fp_rows[k]

    def _visible(self, short: str, w: int, pitch: float) -> str:
        k = (short, w, round(pitch, 2))
        v = self._vis_cache.get(k)
        if v is None:
            v = self.labeler.visible_text(short, w, pitch)
            if len(self._vis_cache) > 200000:
                self._vis_cache.clear()
            self._vis_cache[k] = v
        return v

    def _certify(self, img_bgr, w, det) -> None:
        """Anchors 1 + 2 (docs/accuracy.md).  A tile is CERTAIN when its pixels exactly match a
        game render of a known item, or its label exactly matches the game-font rendering of one
        item's short name among every item that fits the footprint (and the two never disagree).
        A label-certified tile binds its exact render to the item permanently."""
        it, panel, tile = w['it'], w['panel'], w['tile']
        ev = det.evidence
        if it.clipped_top or it.clipped_bottom:
            return
        x, y, wpx, hpx = it.rect
        if x < 0 or y < 0 or x + wpx > img_bgr.shape[1] or y + hpx > img_bgr.shape[0]:
            return
        W, H = int(it.w), int(it.h)
        px, py = float(panel.pitch_x), float(panel.pitch_y)
        crop = img_bgr[y:y + hpx, x:x + wpx]
        a1 = None
        if self.anchor_idx is not None and len(self.anchor_idx):
            ph = self._phases.get(id(panel))
            a1 = self.anchor_idx.match(crop, W, H, px, py, phase=ph, k=ANCHOR_SHORTLIST)
            if a1 is not None and ph is None and a1.score <= anchors_mod.exact_threshold(py):
                self._phases[id(panel)] = a1.phase
        id1 = a1.item_id if (a1 is not None and a1.certain) else ''
        exact1 = a1 is not None and a1.score <= anchors_mod.exact_threshold(py) and \
            a1.n_px >= anchors_mod.MIN_PIXELS * px * py / (SLOT * SLOT)
        if a1 is not None:
            ev['anchor_render'] = {'score': round(a1.score, 2), 'runner': round(a1.runner, 2),
                                   'id': a1.item_id, 'hint': a1.render.hint_id if a1.render else '',
                                   'note': a1.note}
        id2, twins = '', []
        if self.labeler is not None:
            st = fontlabel_mod.strip_of(img_bgr, it.rect, py)
            if st is not None:
                fits, id2, twins = self._label_anchor(st, w, det, a1, W, H, py)
                if fits:
                    ev['anchor_label'] = {'text': fits[0].text, 'total': round(fits[0].total, 2),
                                          'runner': round(fits[1].total, 2) if len(fits) > 1 else None,
                                          'runner_text': fits[1].text if len(fits) > 1 else None,
                                          'certain': bool(id2), 'twins': len(twins)}
        if id2 == '' and len(twins) > 1 and exact1 and a1.item_id in twins:
            id2 = a1.item_id                       # label twins ("BP" ammo) settled by the exact render
        if id1 and id2 and id1 != id2:
            det.uncertain = True
            det.confidence = min(det.confidence, 0.5)
            ev['note'] = 'anchors disagree (exact render vs exact label)'
            print(f'[anchors] contradiction at {it.rect}: render {id1} vs label {id2}')
            return
        cid = id2 or id1
        if not cid and len(twins) > 1:
            # the label was matched exactly, and the game prints that text for several items
            # ("MP5": receiver / magazine / gun, "D3CRX": two colours) that no exact render
            # separates: whatever the picture prefers is a guess between them
            guns = {str(self.cat.cats[self._id_row[t_]]) == 'weapon' for t_ in twins if t_ in self._id_row}
            if not (det.category == 'weapon' and guns == {True}):     # all guns: "a gun" is enough
                det.uncertain = True
                det.confidence = min(det.confidence, 0.5)
                ev['note'] = f'printed label is shared by {len(twins)} items; picture alone decides'
            return
        if not cid:
            if self.s.extra.get('strict_anchors'):
                det.uncertain = True
                det.confidence = min(det.confidence, 0.79)
            return
        if id2 and exact1 and not id1 and a1.render is not None:
            self._bind(a1.render, id2)              # C: next time the render alone is certain
        r = self._id_row.get(cid)
        if r is None:
            return
        c = self.cat
        if det.item_id != cid:
            ev['pipeline_choice'] = det.name
        det.item_id, det.name = cid, str(c.names[r])
        det.category, det.short = str(c.cats[r]), str(c.shorts[r])
        det.confidence = 0.995
        det.uncertain = False
        ev['certified'] = 'render+label' if (id1 and id2) else ('label' if id2 else 'render')

    def _label_anchor(self, st, w, det, a1, W, H, pitch):
        """Closed-set label match.  Returns ``(fits, certified item id or '', twin ids)``."""
        from rapidfuzz import process as rf_process, distance as rf_distance
        c = self.cat
        rows = self._fp_candidates(W, H)
        sw = st.shape[1]
        by_vis: dict = {}
        for r in rows:
            s_ = str(c.shorts[r])
            if s_:
                by_vis.setdefault(self._visible(s_, sw, pitch), set()).add(str(c.ids[r]))
        texts = []
        seen_ids = set()
        for c2 in w.get('pool') or []:
            iid = self._canon.get(str(c.ids[c2.row]), str(c.ids[c2.row]))
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
            if len(seen_ids) >= 12:
                break
        for txt in w.get('ocr_all') or []:
            seen_ids.update(i for i, sc in self._label_hits(txt, W).items() if sc >= 60)
        for iid in (det.item_id, a1.item_id if a1 else '', a1.render.hint_id if (a1 and a1.render) else ''):
            if iid:
                seen_ids.add(iid)
        for iid in seen_ids:
            r = self._id_row.get(iid)
            if r is not None and r in self._fp_set(W, H):
                texts.append(str(c.shorts[r]))
        # adversarial neighbours: the printed names closest to the leading guesses
        keys = list(by_vis)
        for q in [det.short] + list(w.get('ocr_all') or [])[:2]:
            if q:
                texts += [k for k, _, _ in rf_process.extract(q, keys, scorer=rf_distance.Levenshtein.distance,
                                                              limit=6)]
        texts = [t_ for t_ in dict.fromkeys(texts) if t_]
        if not texts:
            return [], '', []
        fits = self.labeler.match(st, texts, pitch)
        if not fontlabel_mod.certain(fits):
            return fits, '', []
        # the game prints the winner's text for every one of these items
        twins = sorted(by_vis.get(fits[0].text, set()))
        if len(twins) == 1:
            return fits, twins[0], twins
        return fits, '', twins

    def _fp_set(self, W, H) -> set:
        k = ('set', W, H)
        if k not in self._fp_rows:
            self._fp_rows[k] = set(self._fp_candidates(W, H))
        return self._fp_rows[k]

    def _bind(self, render, item_id: str) -> None:
        r = self._id_row.get(item_id)
        name = str(self.cat.names[r]) if r is not None else item_id
        self.learned.bind_certain(render.key, item_id, name)
        self.anchor_idx.bind(render.key, item_id)

    # ------------------------------------------------------------------
    def _icon_key(self, row: int) -> str | None:
        """Content key of a cached-icon row (None for tarkov.dev rows)."""
        if row not in self._row_key:
            path = self.cat.meta.get('paths', {}).get(int(row))
            self._row_key[row] = content_key(path) if path and self.cat.src[row] != 'api' else None
        return self._row_key[row]

    def _name_row(self, row: int, item_id: str) -> bool:
        """Point catalog row ``row`` at the api item ``item_id`` (in memory)."""
        a = self._id_row.get(item_id)
        if a is None:
            return False
        c = self.cat
        for arr in (c.ids, c.names, c.shorts, c.cats, c.tint):
            arr[row] = arr[a]
        c.src[row] = 'cache'
        return True

    def _apply_learned(self) -> None:
        """Name every cached icon the store has a confirmed item for, overriding
        both anonymous builds and stale guessed associations."""
        if not self.learned.data:
            return
        c = self.cat
        for r in np.where(c.src != 'api')[0]:
            iid = self.learned.get(self._icon_key(int(r)), min_seen=LEARN_CONFIRMATIONS)
            if iid and str(c.ids[r]) != iid:
                self._name_row(int(r), iid)

    def _learn(self, row: int, item_id: str, residual: float, label_score: float, twins: int) -> None:
        """Record a cached icon's item when the evidence is literal: the tile is a
        near pixel-exact match to that icon AND its label is an exact, unique read."""
        if (self.cat.src[row] == 'api' or residual > LEARN_MAX_RES
                or label_score < 100.0 or twins != 1 or self._read_is_ambiguous(item_id)):
            return
        # the item must be able to take this icon's footprint (a 1x1 sight never draws 1x2) and
        # its label must not be one glyph away from another item's ("MPX FS" / "MPX F5")
        a = self._id_row[item_id]
        c = self.cat
        same = (int(c.tw[a]), int(c.th[a])) in ((int(c.tw[row]), int(c.th[row])), (int(c.th[row]), int(c.tw[row])))
        if not same and str(c.cats[a]) not in ('weapon', 'mod'):
            return
        f = self._fold_of.get(a)
        if not f or sum(1 for v in self._fold_of.values() if v == f) > 1:
            return
        key = self._icon_key(row)
        name = str(self.cat.names[self._id_row[item_id]]) if item_id in self._id_row else item_id
        # one read is not proof (OCR can drop a letter and land on another real name),
        # so a pairing takes effect only once it has been confirmed independently
        if key and self.learned.learn(key, item_id, name) >= LEARN_CONFIRMATIONS:
            self._name_row(row, item_id)

    def _read_is_ambiguous(self, item_id: str) -> bool:
        """True when this item's short name sits inside another item's short name, so a
        read of it could be that other name missing letters ("Diary" in "SDiary")."""
        r = self._id_row.get(item_id)
        if r is None:
            return True
        mine = ocr_mod.canon_nospace(str(self.cat.shorts[r]))
        if not mine:
            return True
        if not hasattr(self, '_short_set'):
            self._short_set = {ocr_mod.canon_nospace(str(self.cat.shorts[a])) for a in self._api}
        return any(mine != s and mine in s for s in self._short_set)

    # ------------------------------------------------------------------
    def _use_dino(self) -> bool:
        return bool(self.s.use_dino and dino_mod.available(self.s.device))

    def _use_ocr(self) -> bool:
        return bool(self.s.use_ocr and ocr_mod.tesseract_available())

    # ------------------------------------------------------------------
    def scan(self, img_bgr: np.ndarray) -> ScanResult:
        with self._lock:
            return self._scan(img_bgr)

    def _scan(self, img_bgr: np.ndarray) -> ScanResult:
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
        cached = []
        for it, panel in items:
            if it.empty:
                continue
            key = tilecache.tile_key(img_bgr, it.rect, panel.pitch_x, panel.pitch_y,
                                     f'{it.w}x{it.h}|{int(it.clipped_top)}{int(it.clipped_bottom)}')
            hit = self.tile_cache.get(key)
            if hit is not None:
                cached.append(_from_cache(hit, it))
                continue
            tile = normalize_tile(img_bgr, it.rect, it.w, panel.pitch_x, panel.pitch_y,
                                  it.clipped_top, it.clipped_bottom)
            if tile is not None:
                work.append({'it': it, 'panel': panel, 'tile': tile, 'ocr_all': [], 'key': key})
        T['cache_hits'] = len(cached)
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
        dets = list(cached)
        for w in work:
            d = self._decide(img_bgr, w)
            if d is not None:
                dets.append(d)
                w['det'] = d
        T['fuse'] = time.perf_counter() - t

        # ---- anchors: exact game render / exact game-font label --------------------------
        self._phases = {}
        t = time.perf_counter()
        for w in work:
            if w.get('det') is not None:
                self._certify(img_bgr, w, w['det'])
                self.tile_cache.put(w['key'], _to_cache(w['det']))
        T['anchors'] = time.perf_counter() - t
        dets.sort(key=lambda d: (d.panel, d.row, d.col))
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
        self._associate_builds()
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
            work[i]['qe'] = qe[qi]

    # ------------------------------------------------------------------
    BUILD_SIM, BUILD_MARGIN = 0.90, 0.08

    def _associate_builds(self) -> None:
        """Give anonymous cache renders their item once, by picture.

        The cache re-association at catalog build time only links a render that is nearly
        pixel-identical to one tarkov.dev icon.  Many barter / loot items are rendered by the
        game differently from the web icon and stay anonymous 'build' templates, which can then
        only be named from an OCR read (and are flagged uncertain when that fails).  Here a
        non-weapon build is assigned to the api item of the same footprint whose DINO embedding
        is very close *and* clearly closer than any differently-named item.  Weapon builds
        (modded guns) are never assigned: their identity is the label's job.  In-memory only."""
        if getattr(self, '_builds_done', False) or self.store is None:
            return
        self._builds_done = True
        c = self.cat
        builds = [int(r) for r in np.where(c.src == 'build')[0]]
        api = np.where((c.src == 'api') & (~c.preset) & (c.cats != 'weapon'))[0]
        if not builds or not len(api):
            return
        Eb = self.store.get(builds)
        Ea = self.store.get([int(r) for r in api])
        for i, r in enumerate(builds):
            same = [j for j, a in enumerate(api)
                    if (c.tw[a], c.th[a]) == (c.tw[r], c.th[r]) or (c.tw[a], c.th[a]) == (c.th[r], c.tw[r])]
            if len(same) < 2:
                continue
            sims = Ea[same] @ Eb[i]
            order = np.argsort(-sims)
            best = same[int(order[0])]
            bname = str(c.names[api[best]])
            rival = max((float(sims[k]) for k in order if str(c.names[api[same[int(k)]]]) != bname), default=0.0)
            if sims[order[0]] >= self.BUILD_SIM and sims[order[0]] - rival >= self.BUILD_MARGIN:
                a = api[best]
                for arr in (c.ids, c.names, c.shorts, c.cats):
                    arr[r] = arr[a]
                c.src[r] = 'cache'

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

    def _best_label(self, w, tile) -> float:
        """Best label score any item gets from this tile's OCR reads."""
        return max((max(self._label_hits(t, tile.W).values(), default=0.0)
                    for t in (w.get('ocr_all') or [])), default=0.0)

    def _label_hits(self, text: str, width: int = 1) -> dict:
        """item id -> fuzzy label score for the best matches of ``text`` among all
        catalog items (cached per text; presets excluded - they map to their base gun)."""
        h = self._hits_cache.get((text, width))
        if h is None:
            rows = self._api_rows()
            idxs = ocr_mod.prefilter(text, self._api_fold, limit=80, folded=True)
            h = {}
            exact = ocr_mod.canon_nospace(text)
            for i in idxs:
                r = int(rows[i])
                sc = ocr_mod.fuzzy_score(text, str(self.cat.names[r]), str(self.cat.shorts[r]), width)
                # Only a read that IS the printed short name may score 100: a partial or
                # truncated match ("Scdr." inside "F scdr.") must not tie the exact one.
                if exact and ocr_mod.canon_nospace(str(self.cat.shorts[r])) == exact:
                    sc = 100.0
                else:
                    sc = min(sc, LABEL_PARTIAL_CAP)
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
        edge = _at_viewport_edge(it, panel, tile)

        def gun_probe() -> bool:
            # guns print their calibre bottom-left ("20ga", "5.45x39"); parts never do
            if it.clipped_bottom or not self._use_ocr():
                return False
            st = ocr_mod.bottom_strip(img_bgr, it.rect, panel.pitch_x, panel.pitch_y, left=True)
            return st is not None and ocr_mod.looks_like_caliber(ocr_mod.read_strips([st])[0])
        if cat.src[c.row] == 'build' or (str(cat.cats[c.row]) == 'weapon' and c.score > 3.0) \
                or best_res > 9.0:
            auth = self._label_authority(w.get('ocr_all') or [], tile,
                                         prefer_weapon=str(cat.cats[c.row]) in ('weapon', 'build'),
                                         res_by_row=_best_res_by_id(cat, cands), at_edge=edge,
                                         twin_rank=self._twin_ranker(w), is_gun=gun_probe)
        elif text and (o is None or o < 60 or o < self._best_label(w, tile) - LABEL_RIVAL_GAP):
            # label conflict: the picture says one thing, a clean unambiguous printed name
            # says another (e.g. a 2x1 suppressor whose label names a 1x1 flash hider)
            auth = self._label_authority(w.get('ocr_all') or [], tile, prefer_weapon=False,
                                         res_by_row=_best_res_by_id(cat, cands), at_edge=edge,
                                         twin_rank=self._twin_ranker(w), is_gun=gun_probe)
            if auth is not None and (str(cat.ids[auth[0]]) == str(cat.ids[c.row])
                                     or literal_beats_label(c.score, s_dino, auth[1])):
                auth = None
        if auth is None and text and o is not None and o >= OCR_AUTH and str(cat.cats[c.row]) != 'weapon':
            # the picture chose a part whose printed name is also a gun's ("TOZ-106" stock / gun):
            # a gun prints its calibre bottom-left, a part does not
            rd = self._id_row
            guns = {rd[i] for t_ in (w.get('ocr_all') or []) for i, s_ in self._label_hits(t_, tile.W).items()
                    if i in rd and str(cat.cats[rd[i]]) == 'weapon' and s_ >= o - 0.01
                    and self._footprint_ok(rd[i], tile, edge)}
            me = rd.get(str(cat.ids[c.row]))
            close = False
            if guns and me is not None:
                # only when the picture cannot tell the part from the gun (incl. its presets)
                rk = self._twin_ranker(w)([me] + sorted(guns))
                g_best = max((rk[g] for g in guns if g in rk), default=None)
                close = g_best is not None and me in rk and rk[me] - g_best < TWIN_GAP_OK
            if close and gun_probe():
                auth = self._label_authority(w.get('ocr_all') or [], tile, prefer_weapon=True,
                                             res_by_row=_best_res_by_id(cat, cands), at_edge=edge,
                                             twin_rank=self._twin_ranker(w))
        item_id = str(cat.ids[c.row])
        name = str(cat.names[c.row])
        evidence = {'residual': round(c.score, 3), 'stage1_best': round(best_res, 3),
                    'dino': None if s_dino is None else round(s_dino, 4),
                    'ocr_text': text, 'ocr': None if o is None else round(o, 1),
                    'source': str(cat.src[c.row]),
                    'visual': f'{cat.names[c.row]} [{cat.src[c.row]} #{c.row}]'}
        chosen_row = c.row
        twins = 1
        twin_gap = None
        if auth is not None:
            row, sc, twins, twin_gap = auth
            item_id, name = str(cat.ids[row]), str(cat.names[row])
            evidence.update(ocr_authority=True, ocr=round(sc, 1), source='ocr-authority', twins=twins,
                            twin_gap=None if twin_gap is None else round(twin_gap, 2))
            o = sc
            chosen_row = row
        elif item_id == '':
            # A nameless cached render won on pixels but the label was unreadable: take the
            # best *named* candidate that looks as much like the tile (same DINO, within noise).
            named = [(S2, c2, d2) for S2, c2, d2, _ in scored
                     if str(cat.ids[c2.row]) and d2 is not None and s_dino is not None
                     and d2 >= s_dino - NAMED_DINO_SLACK]
            if named:
                _, c2, _ = named[0]
                item_id, name, chosen_row = str(cat.ids[c2.row]), str(cat.names[c2.row]), c2.row
                evidence['note'] = 'nameless cached icon; named by closest look-alike'
            else:
                evidence['note'] = 'modded item / build with unreadable label'
        # Literal evidence teaches the cache: the tile is a near pixel-exact match to a cached
        # icon and the label reads one item exactly -> that icon IS that item, from now on.
        if item_id and (o or 0) >= 100.0:
            # ...and it must be distinctive: look-alikes (loose rounds of different calibres)
            # sit within a hair of each other, and learning one would teach the wrong name.
            best = cands[0]
            rival = next((c2.score for c2 in cands[1:]
                          if str(cat.ids[c2.row]) != str(cat.ids[best.row])
                          or cat.src[c2.row] == 'build'), 99.0)
            if rival >= LEARN_DISTINCT * max(best.score, 0.5):
                self._learn(int(best.row), item_id, float(best.score), float(o), twins)
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
        tie_gap = None
        if auth is None and item_id and not twins > 1:
            tie_gap = min((chosen_S - S2 for S2, c2, _, _ in scored[1:] if str(cat.names[c2.row]) != name),
                          default=None)
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
        pool_rows, seen_n = [], set()
        for S2, c2, _, _ in scored:
            nm = str(cat.names[c2.row])
            if nm not in seen_n:
                seen_n.add(nm)
                pool_rows.append((int(c2.row), int(c2.rot), nm, round(S2, 2)))
            if len(pool_rows) == 6:
                break
        evidence['pool_rows'] = pool_rows
        evidence['catalog_row'] = int(chosen_row)
        evidence['rot'] = int(c.rot)

        feats = {'margin': min(margin, 6.0), 'res': min(c.score, 20.0),
                 'ocr': _g(o) if (text and o is not None) else 0.0,
                 'dino': (s_dino - 0.5) if s_dino is not None else 0.0}
        z = CALIB['b'] + sum(CALIB[k] * v for k, v in feats.items())
        if item_id == '':
            z -= 4.0
        twin_ok = twin_gap is not None and twin_gap >= (TWIN_GAP_OK if have_dino else TWIN_GAP_OK_RES)
        if twins > 1 and not twin_ok:
            z -= 1.5 * math.log(twins)
        conf = _sigmoid(z)
        if twins > 1 and not twin_ok:
            # same-label twins the picture did not separate: the choice is a coin flip
            conf = min(conf, TIE_CONF_CAP)
            evidence['note'] = 'label twins not separated by the picture'
        if auth is not None and auth[1] >= 100.0 and (twins == 1 or twin_ok):
            # label-verified: the printed short name was read exactly, no differently named
            # item (not even one that differs by an ambiguous glyph) fits this footprint, and
            # among same-label twins the picture decided clearly
            conf = max(conf, LABEL_VERIFIED_CONF)
            evidence['verified'] = 'label'
        if str(cat.cats[chosen_row]) == 'weapon' and self._weapon_sure(
                evidence.get('pool_rows'), auth if (twins == 1 or twin_ok) else None):
            # guns are skipped by the sell list: being sure it IS a gun is all that matters
            conf = max(conf, WEAPON_SURE_CONF)
            evidence['verified'] = 'weapon'
        if is_tie(tie_gap):
            conf = min(conf, TIE_CONF_CAP)
            evidence['note'] = 'indistinguishable look-alike (same picture, same label read)'
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
            self.gun_presets = {}       # base gun row -> preset item ids
            self.gun_sizes = {grow: {(int(c.tw[grow]), int(c.th[grow]))} for _, grow in guns}
            for r in np.where(c.preset & (c.src == 'api'))[0]:
                nm = str(c.names[r])
                for gname, grow in guns:
                    if nm.startswith(gname):
                        self.preset_base[str(c.ids[r])] = grow
                        self.gun_presets.setdefault(grow, []).append(str(c.ids[r]))
                        self.gun_sizes[grow].add((int(c.tw[r]), int(c.th[r])))
                        break
            self._fold_of = {int(r): ocr_mod.fold_glyph(str(c.shorts[r])) for r in self._api}
        return self._api

    def _weapon_like(self, row: int) -> bool:
        c = self.cat
        return bool(str(c.cats[row]) == 'weapon' or c.preset[row] or c.src[row] == 'build')

    def _weapon_sure(self, pool_rows, auth) -> bool:
        """The tile is certainly *a gun* (which one does not matter: guns are skipped).  True when
        the printed name confidently names a gun, or every one of the best three differently
        named pictures is a gun / preset / weapon build."""
        if auth is not None and auth[1] >= OCR_AUTH and str(self.cat.cats[auth[0]]) == 'weapon':
            return True
        top = (pool_rows or [])[:3]
        return len(top) >= 2 and all(self._weapon_like(r) for r, *_ in top)

    def _footprint_ok(self, row: int, tile: Tile, at_edge: bool) -> bool:
        """Can item ``row`` occupy this tile?  An item's footprint is fixed (either orientation)
        unless it takes attachments: weapons (any of their preset sizes, or bigger when built)
        and weapon mods (a handguard with rails) only ever grow.  At the viewport edge a tile can
        be cut, so anything goes there."""
        if at_edge:
            return True
        cat = self.cat
        W, H = tile.W, tile.H
        sizes = self.gun_sizes.get(row) or {(int(cat.tw[row]), int(cat.th[row]))}
        if any((w_, h_) in ((W, H), (H, W)) for w_, h_ in sizes):
            return True
        if str(cat.cats[row]) in ('weapon', 'mod') and 'magazine' not in str(cat.names[row]).lower():
            return any(w_ * h_ <= W * H for w_, h_ in sizes)      # magazines take nothing: fixed size
        return False

    def _twin_ranker(self, w):
        """Picture score of label twins: ``-W_RES * residual + W_DINO * cosine`` of each twin's
        best template, the same fusion that ranks the pool.  Twins share a short name (BP, BS
        ...) and, for ammo of one box size, often a near-identical silhouette; the residual
        alone is brightness-sensitive and ranks the wrong calibre's pack first on a dim
        capture, DINO sees the pack's art.  Returns ``f(rows) -> {row: score}`` (rows without a
        visual candidate are absent; DINO is left out unless every twin has an embedding)."""
        cat, cands = self.cat, w['cands']
        qe = w.get('qe')

        def rank(rows):
            best = {}
            for c in cands:
                iid = str(cat.ids[c.row])
                if iid and iid not in best:
                    best[iid] = c
            have = {}
            for r in rows:
                # a gun is also every default build (preset) of it: built guns draw like those
                ids = [str(cat.ids[r])] + list(getattr(self, 'gun_presets', {}).get(r, ()))
                cs = [best[i] for i in ids if i in best]
                if cs:
                    have[r] = cs
            sims = w.get('sims') or {}
            use_dino = qe is not None and self.store is not None and not w['tile'].clip

            def fused(c):
                s = -W_RES * c.score
                if not use_dino:
                    return s, True
                v = sims.get((c.row, c.rot))
                if v is None and c.rot == 0:
                    v = float(qe @ self.store.get([c.row])[0])
                return (s + W_DINO * v, True) if v is not None else (s, False)
            vals = {r: [fused(c) for c in cs] for r, cs in have.items()}
            if use_dino and all(ok for v in vals.values() for _, ok in v) and len(have) == len(rows):
                return {r: max(s for s, _ in v) for r, v in vals.items()}
            return {r: max(-W_RES * c.score for c in cs) for r, cs in have.items()}
        return rank

    def _label_authority(self, texts: list, tile: Tile, prefer_weapon: bool, res_by_row: dict | None = None,
                         at_edge: bool = False, twin_rank=None, is_gun=None):
        """Identity from the printed short name over *all* base items whose footprint can be
        this tile (:meth:`_footprint_ok`).  All OCR variants of the label are pooled: needs a
        confident read (>= ``OCR_AUTH``) and no *different* name within 5 points, except names
        that only differ by glyphs the label font makes ambiguous (``MPX F5`` / ``MPX FS``: same
        :func:`ocr.fold`), which are label twins like equal short names.  Twins are resolved by
        preferring base weapons when the picture looks like a weapon, then exact footprint, then
        the picture (:meth:`_twin_ranker`).
        Returns ``(row, score, n_finalists, picture_gap)`` or None; ``picture_gap`` is the
        fused-score lead of the chosen twin over the best differently named finalist (None when
        there was only one finalist or no picture score)."""
        self._api_rows()
        cat = self.cat
        allsc: dict[int, float] = {}
        for txt in texts:
            for i, s_ in self._label_hits(txt, tile.W).items():
                r = self._id_row.get(i)
                if r is not None and s_ > allsc.get(r, -1.0):
                    allsc[r] = s_
        sc = [(s_, r) for r, s_ in allsc.items() if self._footprint_ok(r, tile, False)]
        if at_edge and not any(s_ >= 100.0 for s_, _ in sc):
            # the viewport may cut an item: an exact read of a bigger item is possible there
            sc += [(s_, r) for r, s_ in allsc.items() if s_ >= 100.0 and (s_, r) not in sc]
        if not sc:
            return None
        sc.sort(key=lambda t: -t[0])
        top = sc[0][0]
        if top < OCR_AUTH:
            return None
        f0 = self._fold_of.get(sc[0][1], '')
        rivals = {self._fold_of.get(r, '') for s_, r in sc if s_ >= top - 5.0}
        if len(rivals) > 1:
            return None
        twins = [r for s_, r in sc if s_ >= top - 0.01 or (f0 and self._fold_of.get(r) == f0)]
        W, H = tile.W, tile.H
        def coarse(r):
            weap = str(cat.cats[r]) == 'weapon'
            sizes = self.gun_sizes.get(r) or {(int(cat.tw[r]), int(cat.th[r]))}
            exact = any((a, b) in ((W, H), (H, W)) for a, b in sizes)
            return (0 if (prefer_weapon and weap) else 1, 0 if exact else 1)
        c0 = min(coarse(r) for r in twins)
        finalists = [r for r in twins if coarse(r) == c0]
        vis = twin_rank(finalists) if (twin_rank and len(finalists) > 1) else {}
        kinds = {str(cat.cats[r]) == 'weapon' for r in finalists}
        probed, n_before = False, len(finalists)
        if not prefer_weapon and len(kinds) == 2 and is_gun is not None:
            ranked = sorted(vis.values(), reverse=True)
            if (len(ranked) < 2 or ranked[0] - ranked[1] < TWIN_GAP_OK) and is_gun():
                # "TOZ-106" names the gun and its stock and the picture cannot tell: a calibre
                # printed bottom-left says gun (parts print none)
                prefer_weapon = True
                probed, n_before = True, len(finalists)
                c0 = min(coarse(r) for r in twins)
                finalists = [r for r in twins if coarse(r) == c0]
                vis = twin_rank(finalists) if (twin_rank and len(finalists) > 1) else {}

        def key(r):
            if vis:
                rr = -round(vis.get(r, -99.0), 1)
            else:
                rr = round(res_by_row.get(r, 99.0), 1) if res_by_row else 0.0
            return coarse(r) + (rr, r)
        finalists.sort(key=key)
        gap = None
        if len(finalists) > 1 and vis and finalists[0] in vis:
            nm0 = str(cat.names[finalists[0]])
            others = [vis.get(r, -99.0) for r in finalists[1:] if str(cat.names[r]) != nm0]
            if others:
                gap = vis[finalists[0]] - max(others)
        if probed:
            # the calibre read only breaks the tie: art can look like text, so never "verified"
            return finalists[0], float(top), max(2, n_before), None
        return finalists[0], float(top), len(finalists), gap


class _CertainBindings:
    """Read-only view of the learned store: only bindings made by a certain anchor."""

    def __init__(self, learned: LearnedNames):
        self.learned = learned

    def get(self, key):
        e = self.learned.data.get(key) if key else None
        return e['id'] if e and e.get('certain') else None


_CACHE_FIELDS = ('item_id', 'name', 'confidence', 'uncertain', 'evidence', 'count', 'fir', 'category',
                 'short', 'rotated', 'w', 'h')


def _to_cache(d: 'Detection') -> dict:
    return {k: getattr(d, k) for k in _CACHE_FIELDS}


def _from_cache(c: dict, it) -> 'Detection':
    x, y, wpx, hpx = it.rect
    ev = dict(c['evidence'])
    ev['cached'] = True
    return Detection(panel=it.panel, col=it.col, row=it.row, w=c['w'], h=c['h'], rotated=c['rotated'],
                     item_id=c['item_id'], name=c['name'], confidence=c['confidence'], uncertain=c['uncertain'],
                     evidence=ev, count=c['count'], fir=c['fir'], rect=(x, y, wpx, hpx),
                     clipped=bool(it.clipped_top or it.clipped_bottom), category=c['category'], short=c['short'])


def _at_viewport_edge(it, panel, tile) -> bool:
    """True when an item tile touches the top or bottom of its visible panel, where the
    stash viewport can cut an item in half (or clip detection already said so)."""
    if getattr(tile, 'clip', '') or getattr(it, 'clipped_top', False) or getattr(it, 'clipped_bottom', False):
        return True
    x, y, w, h = it.rect
    tol = max(3.0, 0.35 * h / max(int(it.h), 1))
    return abs((y + h) - panel.y1) <= tol or abs(y - panel.y0) <= tol


def _best_res_by_id(cat, cands) -> dict:
    """catalog 'api' row -> best stage-1 residual of any template of the same item (cache /
    build templates are mapped back to their item), for resolving label twins by picture."""
    ids = {}
    for c in cands:
        iid = str(cat.ids[c.row])
        if iid and (iid not in ids or c.score < ids[iid]):
            ids[iid] = c.score
    return _ResByRow(cat, ids)


class _ResByRow(dict):
    def __init__(self, cat, ids):
        super().__init__()
        self._cat, self._ids = cat, ids

    def get(self, row, default=None):
        return self._ids.get(str(self._cat.ids[row]), default)

    def __bool__(self):
        return True


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
