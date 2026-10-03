"""
Scene layer: which grids on the screen are the stash, the player's own gear, loot, an open
container window, or the "Available" slot item-picker (which must never be scanned).

``detect_grid`` only sees line lattices.  Neighbouring grids that share the lattice phase come
out as ONE panel (gear column + item picker + stash = a 21-column panel on a real 1440p frame),
so the roles have to come from *landmarks*, found cheaply on the whole frame:

1. floating windows   a dark rectangle with a 1 px border, a title bar and a RED close button
                      (dark red rounded rectangle, white X).  The button is found by colour and
                      shape, the border/title bar traced from it, the title read with Tesseract
                      ("Available" = the slot picker).  Windows occlude whatever is beneath.
2. stash toolbar      a column of small bright filter icons at a fixed x just left of the stash
                      grid.  Everything right of it is the (10 column wide) stash in the lobby.
3. slot headers       UPPERCASE bars above the equipment slots (TACTICAL RIG, POCKETS, BACKPACK,
                      POUCH, SPECIAL SLOTS, ...).  Bright text blobs are found by colour/size and
                      read in ONE Tesseract call; a grid right of / under a header takes its role.

Everything is relative to the frame size / UI pitch (EFT scales its UI with the height), so any
resolution works.  The raid rules are deliberately simple (see ``_raid_role``): own gear on the
left, loot on the right, anything unsure is 'unknown' with a low confidence.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SLOT
from .grid import GridResult, Panel, line_mask

ROLES = ('stash', 'own_rig', 'own_pockets', 'own_backpack', 'own_pouch', 'own_special', 'own_other',
         'loot', 'container_window', 'picker', 'unknown')

# slot header text -> role.  kind 'right': the grid sits right of the slot box (2 cells wide)
# under the header; kind 'under': the slot boxes are drawn directly under the header.
HEADERS = {
    'EARPIECE': ('own_other', 'right'), 'HEADWEAR': ('own_other', 'right'),
    'FACE COVER': ('own_other', 'right'), 'ARMBAND': ('own_other', 'right'),
    'BODY ARMOR': ('own_other', 'right'), 'EYEWEAR': ('own_other', 'right'),
    'DOGTAG': ('own_other', 'right'), 'ON SLING': ('own_other', 'right'),
    'ON BACK': ('own_other', 'right'), 'HOLSTER': ('own_other', 'right'),
    'SHEATH': ('own_other', 'right'),
    'TACTICAL RIG': ('own_rig', 'right'), 'BACKPACK': ('own_backpack', 'right'),
    'POUCH': ('own_pouch', 'right'),
    'POCKETS': ('own_pockets', 'under'), 'SPECIAL SLOTS': ('own_special', 'under'),
}
_SIDE = {'stash': 'stash', 'loot': 'loot', 'container_window': 'stash',
         'picker': 'other', 'unknown': 'other'}


@dataclass
class Region:
    bbox: tuple                 # (x, y, w, h)
    role: str
    side: str = 'other'         # 'own' | 'loot' | 'stash' | 'other'
    title: str | None = None
    confidence: float = 0.0
    title_h: int = 0            # windows: height of the title band in px

    def as_dict(self) -> dict:
        x, y, w, h = (int(v) for v in self.bbox)
        return {'bbox': [x, y, w, h], 'role': self.role, 'side': self.side,
                'title': self.title, 'confidence': round(float(self.confidence), 2)}


@dataclass
class SceneInfo:
    scene: str = 'unknown'      # lobby_gear|lobby_stash|raid_inventory|raid_loot|trader|none|unknown
    in_raid: bool | None = None
    regions: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    toolbar: tuple | None = None      # (x0, x1) of the stash filter toolbar
    headers: list = field(default_factory=list)   # [(name, (x, y, w, h))]
    zones: list = field(default_factory=list)    # coarse header-anchored search zones (internal)
    pitch: float = float(SLOT)
    frame_wh: tuple = (0, 0)


# --------------------------------------------------------------------------
# floating windows
# --------------------------------------------------------------------------

def find_close_buttons(frame: np.ndarray) -> list:
    """Red close buttons ``[(x, y, w, h)]``: saturated dark red rounded rectangle (~1.5 x as wide
    as tall, 0.012-0.03 of the frame height) with a white glyph in the middle."""
    H, W = frame.shape[:2]
    sc = 2 if H >= 900 else 1
    small = frame[::sc, ::sc].astype(np.int16)
    B, G, R = small[..., 0], small[..., 1], small[..., 2]
    m = ((R >= 40) & (R > 2 * G) & (R > 2 * B) & (G < 60)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for x, y, w, h, a in st[1:]:
        x, y, w, h = x * sc, y * sc, w * sc, h * sc
        if not (0.011 * H <= h <= 0.032 * H and 0.8 <= w / h <= 2.1):
            continue
        if a * sc * sc < 0.7 * w * h:
            continue
        crop = frame[y:y + h, x:x + w]
        white = crop.min(axis=2) >= 150
        frac = white.mean()
        if not (0.03 <= frac <= 0.4):
            continue
        ys, xs = np.nonzero(white)
        if abs(xs.mean() / w - 0.5) > 0.2 or abs(ys.mean() / h - 0.5) > 0.2:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    return out


def _col_frac(frame, x, y0, y1, ref, tol, below=True):
    col = frame[y0:y1, x].astype(np.int16)
    return float((np.abs(col - ref).max(axis=1) <= tol).mean())


def trace_window(frame: np.ndarray, btn: tuple):
    """Window rectangle ``(x, y, w, h)`` for a close button, or None.

    Right edge: scan right of the button while the title-bar colour persists, the brightest
    column after that is the border.  Left edge: nearest column to the left that is a border
    coloured vertical line over the whole title band.  Top / bottom: the contiguous border
    coloured run along that left column."""
    H, W = frame.shape[:2]
    bx, by, bw, bh = btn
    cy = by + bh // 2
    x = bx + bw + 1
    if x + 3 >= W:
        return None
    bg = frame[cy, x].astype(np.int16)
    y0, y1 = by, by + bh
    # --- right edge
    xe = None
    for xx in range(x, min(W - 1, x + int(0.03 * W))):
        if _col_frac(frame, xx, y0, y1, bg, 10) < 0.7:
            xe = xx
            break
    if xe is None:
        return None
    best, bx_ = -1.0, xe
    for xx in range(xe, min(W - 1, xe + 7)):
        g = float(frame[y0:y1, xx].astype(np.float32).mean())
        if g > best:
            best, bx_ = g, xx
    if best < 60:
        return None
    bcol = np.median(frame[y0:y1, bx_].astype(np.int16), axis=0)
    xr = bx_
    while xr + 1 < W and _col_frac(frame, xr + 1, y0, y1, bcol, 30) >= 0.8:
        xr += 1
    # --- left edge
    xl = None
    for xx in range(bx - 3 * bw, max(0, bx - int(0.6 * W)), -1):
        if _col_frac(frame, xx, y0, y1, bcol, 30) >= 0.85:
            xl = xx
            break
    if xl is None:
        return None
    while xl - 1 >= 0 and _col_frac(frame, xl - 1, y0, y1, bcol, 30) >= 0.85:
        xl -= 1
    # --- top / bottom along the left border column
    def run(step):
        yy, gap = cy, 0
        last = cy
        while 0 <= yy + step < H:
            yy += step
            if np.abs(frame[yy, xl].astype(np.int16) - bcol).max() <= 16:
                last, gap = yy, 0
            else:
                gap += 1
                if gap > 3:
                    break
        return last
    top, bot = run(-1), run(1)
    w, h = xr - xl + 1, bot - top + 1
    if w < 4 * bw or h < 2 * bh:
        return None
    return int(xl), int(top), int(w), int(h)


def _has_grid_lines(frame, rect, title_h) -> bool:
    """True when the window interior shows drawn cell borders (a container) rather than text."""
    x, y, w, h = rect
    sub = frame[y + title_h:y + h - 2, x + 2:x + w - 2]
    if sub.size == 0:
        return False
    lm = line_mask(sub)
    if lm.shape[0] < 20 or lm.shape[1] < 20:
        return False
    k = max(12, int(0.03 * frame.shape[0]))
    hr = cv2.morphologyEx(lm.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vr = cv2.morphologyEx(lm.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return int(hr.any(axis=1).sum()) >= 2 and int(vr.any(axis=0).sum()) >= 2


# --------------------------------------------------------------------------
# stash toolbar
# --------------------------------------------------------------------------

def find_toolbar(frame: np.ndarray):
    """(x0, x1) of the stash filter toolbar (a column of small bright icons), or None."""
    H, W = frame.shape[:2]
    xa, xb = int(0.55 * W), int(0.75 * W)
    ya, yb = int(0.1 * H), int(0.55 * H)
    g = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
    bright = g > 150
    cs = bright.sum(axis=0).astype(np.float32)
    cs = np.convolve(cs, np.ones(3) / 3, mode='same')
    # a column of stacked icons: bright in many rows, but only in a narrow band
    on = cs >= 0.05 * (yb - ya)
    runs = []
    x = 0
    while x < len(on):
        if on[x]:
            x2 = x
            while x2 + 1 < len(on) and on[x2 + 1]:
                x2 += 1
            if 0.006 * W <= x2 - x + 1 <= 0.016 * W:
                runs.append((float(cs[x:x2 + 1].sum()), x, x2))
            x = x2 + 1
        else:
            x += 1
    for _, rx0, rx1 in sorted(runs, reverse=True):
        # the band must contain several separate icons (vertical blobs), not one bright bar
        band = bright[:, rx0:rx1 + 1].any(axis=1).astype(np.uint8)
        nb, _, st, _ = cv2.connectedComponentsWithStats(band.reshape(-1, 1), connectivity=4)
        icons = [s_ for s_ in st[1:] if 0.008 * H <= s_[3] <= 0.04 * H]
        if len(icons) >= 4:
            return int(xa + rx0), int(xa + rx1)
    return None


# --------------------------------------------------------------------------
# slot headers + OCR
# --------------------------------------------------------------------------

_OCR_CACHE: dict = {}


def _ocr_strips(strips: list, psm: int = 6) -> list:
    """Read N small BGR strips with ONE Tesseract call (stitched on a white canvas).  Strips are
    binarised first and their text is cached by that bitmap: slot headers are identical on every
    scan, so after the first one only new strips are read."""
    from . import ocr as ocr_mod
    if not strips or ocr_mod.pytesseract is None or not ocr_mod.tesseract_available():
        return [''] * len(strips)
    tiles, keys = [], []
    for s in strips:
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if s.ndim == 3 else s
        sc = 2.4 if g.shape[0] < 40 else 1.4
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if g.mean() < 127:                    # light text on dark -> make it dark on light
            g = 255 - g
        tiles.append(g)
        keys.append((g.shape, hash(g.tobytes())))
    out = [_OCR_CACHE.get(k) for k in keys]
    todo = [i for i, o in enumerate(out) if o is None]
    if todo:
        gap = 30
        wmax = max(tiles[i].shape[1] for i in todo)
        canvas = np.full((sum(tiles[i].shape[0] + gap for i in todo) + gap, wmax + 2 * gap), 255, np.uint8)
        y, ys = gap, {}
        for i in todo:
            t = tiles[i]
            canvas[y:y + t.shape[0], gap:gap + t.shape[1]] = t
            ys[i] = (y, y + t.shape[0])
            y += t.shape[0] + gap
        try:
            d = ocr_mod.pytesseract.image_to_data(canvas, config=f'--psm {psm}',
                                                  output_type=ocr_mod.pytesseract.Output.DICT)
        except Exception:
            return [o or '' for o in out]
        words = {i: [] for i in todo}
        for j, t in enumerate(d['text']):
            t = (t or '').strip()
            if not t:
                continue
            cy = d['top'][j] + d['height'][j] / 2
            for i, (a, b) in ys.items():
                if a - gap / 2 <= cy <= b + gap / 2:
                    words[i].append((d['left'][j], t))
                    break
        for i in todo:
            out[i] = ' '.join(t for _, t in sorted(words[i]))
            if len(_OCR_CACHE) > 4000:
                _OCR_CACHE.clear()
            _OCR_CACHE[keys[i]] = out[i]
    return [o or '' for o in out]


def _grid_extent(frame: np.ndarray, zone: Region, windows: list, pitch: float):
    """Tight bbox of the drawn cell/slot borders in a header zone, or None when there are none
    (no such equipment, or hidden).  Border runs are grouped (9 px gaps between rig slots are
    bridged) and only groups whose top sits under the header (or touches a window, which may hide
    the real top) count."""
    H, W = frame.shape[:2]
    zx, zy, zw, zh = (int(v) for v in zone.bbox)
    zx, zy = max(0, zx), max(0, zy)
    zw, zh = min(zw, W - zx), min(zh, H - zy)
    if zw < pitch or zh < pitch:
        return None
    lm = line_mask(frame[zy:zy + zh, zx:zx + zw])
    for w in windows:
        wx, wy, ww, wh = w.bbox
        lm[max(0, wy - 2 - zy):max(0, wy + wh + 3 - zy), max(0, wx - 2 - zx):max(0, wx + ww + 3 - zx)] = False
    m = lm.astype(np.uint8)
    k = max(8, int(0.5 * pitch))
    hr = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vr = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    both = cv2.dilate(hr | vr, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.22 * pitch), int(0.22 * pitch))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(both, connectivity=8)
    top_ok = (zone.bbox[1] + 0.2 * pitch) - zy          # header bottom in crop coordinates
    boxes = []
    for x, y, w, h, a in st[1:]:
        if w < 0.7 * pitch or h < 0.7 * pitch:
            continue
        touches = any(abs((zy + y) - (wy + wh)) < 0.3 * pitch and wx - pitch < zx + x < wx + ww + pitch
                      for (wx, wy, ww, wh) in (w_.bbox for w_ in windows))
        if y <= top_ok + 1.0 * pitch or touches:
            boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    pad = int(0.11 * pitch)                              # undo the dilation
    return tuple(int(v) for v in (zx + x0 + pad, zy + y0 + pad, x1 - x0 - 2 * pad, y1 - y0 - 2 * pad))


def _match_header(text: str):
    t = re.sub(r'[^A-Z ]', '', text.upper()).strip()
    if len(t) < 4:
        return None
    best, score = None, 0.0
    for name in HEADERS:
        r = difflib.SequenceMatcher(None, t, name).ratio()
        if r > score:
            best, score = name, r
    return (best, score) if score >= 0.8 else None


def find_header_candidates(frame: np.ndarray, x_max: int) -> list:
    """Bright uppercase-text blobs (x, y, w, h) that may be slot headers."""
    H, W = frame.shape[:2]
    y0, y1 = int(0.07 * H), int(0.93 * H)
    sub = frame[y0:y1, :x_max]
    m = (sub.min(axis=2) >= 190).astype(np.uint8)
    k = max(9, int(round(0.0125 * H)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    c = [(int(x), int(y + y0), int(w), int(h)) for x, y, w, h, a in st[1:]
         if 0.0055 * H <= h <= 0.017 * H and 0.028 * H <= w <= 0.16 * H]
    c.sort(key=lambda b: -b[2] * b[3])
    return c[:40]


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze_scene(frame_bgr: np.ndarray, *, in_raid=None) -> SceneInfo:
    H, W = frame_bgr.shape[:2]
    pitch = SLOT * min(W / 1920.0, H / 1080.0)
    info = SceneInfo(in_raid=in_raid, pitch=pitch, frame_wh=(W, H))

    # ---- windows --------------------------------------------------------
    wins = []                       # [rect, title_h]
    for btn in find_close_buttons(frame_bgr):
        r = trace_window(frame_bgr, btn)
        if r is not None and not any(abs(r[0] - w[0][0]) < 4 and abs(r[1] - w[0][1]) < 4 for w in wins):
            wins.append([r, btn[1] + btn[3] - r[1] + 3])

    # ---- toolbar (stash) --------------------------------------------------
    tb = find_toolbar(frame_bgr)
    info.toolbar = tb
    lobby = (not in_raid) if in_raid is not None else (tb is not None)
    info.in_raid = (not lobby) if in_raid is None else in_raid

    # ---- headers + titles in ONE ocr call ----------------------------------
    x_max = (tb[0] - 4) if (tb and lobby) else int(0.97 * W)
    cands = find_header_candidates(frame_bgr, x_max)
    cands = [c for c in cands if not any(_inside((c[0] + c[2] / 2, c[1] + c[3] / 2), w[0]) for w in wins)]
    strips = []
    for (x, y, w, h) in cands:
        strips.append(frame_bgr[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3])
    tstrips = []
    for rect, th in wins:
        x, y, w, h = rect
        tw = int(min(0.4 * w, 7.5 * th))
        tstrips.append(frame_bgr[y + 2:y + th - 3, x + 3:x + 3 + tw])
    texts = _ocr_strips(strips + tstrips)
    htexts, ttexts = texts[:len(strips)], texts[len(strips):]

    for c, t in zip(cands, htexts):
        mh = _match_header(t)
        if mh:
            info.headers.append((mh[0], c))
    # a header name appears once per side; keep the best (largest) per (name, side)
    # ---- window regions -----------------------------------------------------
    for (rect, th), title in zip(wins, ttexts):
        title = _clean_title(title)
        low = re.sub(r'[^a-z]', '', title.lower())
        if difflib.SequenceMatcher(None, low[:9], 'available').ratio() >= 0.75 or low.startswith('availab'):
            role, conf = 'picker', 0.95
        elif _has_grid_lines(frame_bgr, rect, th):
            role, conf = 'container_window', 0.85
        else:
            role, conf = 'unknown', 0.6
        side = _SIDE.get(role, 'other')
        if info.in_raid and role == 'container_window':
            side = 'loot' if (rect[0] + rect[2] / 2) >= 0.55 * W else 'other'
        info.windows.append(Region(rect, role, side, title or None, conf, th))
    info.regions.extend(info.windows)

    # ---- stash region ---------------------------------------------------------
    if tb and lobby:
        sx = tb[1] + int(0.003 * W)
        info.regions.append(Region((sx, int(0.0715 * H), int(10 * pitch + 0.006 * W), int(0.811 * H)),
                                   'stash', 'stash', None, 0.95))

    # ---- header zones -----------------------------------------------------------
    for name, (x, y, w, h) in info.headers:
        role, kind = HEADERS[name]
        hx0 = x - 5
        ybot = y + h + 5
        side = 'own'
        if info.in_raid and hx0 >= 0.55 * W:
            role, side = 'loot', 'loot'
        zx = hx0 + (1.6 * pitch if kind == 'right' else -0.3 * pitch)
        zy1 = (ybot + 1.5 * pitch) if kind == 'under' else int(0.883 * H)
        for n2, (x2, y2, w2, h2) in info.headers:      # next header below (same column) bounds it
            if kind == 'right' and abs(x2 - x) < 0.5 * pitch and y2 > y + 3:
                zy1 = min(zy1, y2 - 2)
        zx1 = (tb[0] - 2) if (tb and lobby) else int(0.97 * W)
        if kind == 'under':
            for n2, (x2, y2, w2, h2) in info.headers:  # next header to the right on the same row
                if abs(y2 - y) < 0.3 * pitch and x2 > x + pitch:
                    zx1 = min(zx1, x2 - 8)
        info.zones.append(Region((int(zx), int(ybot - 0.2 * pitch), int(zx1 - zx), int(zy1 - ybot + 0.2 * pitch)),
                                 role, side, name, 0.8))
    for z in info.zones:
        if z.role in ('own_other',):
            continue
        bb = _grid_extent(frame_bgr, z, info.windows, pitch)
        if bb is not None:
            info.regions.append(Region(bb, z.role, z.side, z.title, 0.85))

    # ---- scene class ---------------------------------------------------------------
    nh = len(info.headers)
    if info.in_raid:
        loot_h = any(r.side == 'loot' for r in info.regions if r.title in HEADERS)
        if nh >= 2:
            info.scene = 'raid_loot' if loot_h else 'raid_inventory'
        else:
            info.scene = 'raid_loot' if info.windows else 'unknown'
    elif tb is not None:
        info.scene = 'lobby_gear' if nh >= 3 else 'lobby_stash'
    elif nh >= 3 or info.windows:
        info.scene = 'unknown'
    else:
        info.scene = 'none'
    return info


def _clean_title(t: str) -> str:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", t or '')
    while len(toks) > 1 and len(toks[0]) == 1:      # icon next to the title read as a letter
        toks.pop(0)
    out = ' '.join(toks)
    letters = [ch for ch in out if ch.isalpha()]
    if 2 <= len(letters) <= 6 and sum(ch.isupper() for ch in letters) >= 2:
        out = out.upper()                            # spaced caps ("S I C C") come back mixed-case
    return out


def _inside(pt, rect) -> bool:
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


# --------------------------------------------------------------------------
# splitting detect_grid panels
# --------------------------------------------------------------------------

def _raid_role(scene: SceneInfo, cx: float, W: int):
    """In-raid grids without a header: right of 0.6 W = loot (low confidence), else unknown.
    A grid is never confidently called loot."""
    return ('loot', 'loot', 0.5) if cx >= 0.6 * W else ('unknown', 'other', 0.3)


def _cell_edges(lm: np.ndarray, panel: Panel, c: int, r: int) -> int:
    """Number of the 4 edges of cell (c, r) that carry a drawn border line."""
    xs, ys = panel.xs, panel.ys
    x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
    n = 0
    for (a0, a1, b0, b1, horiz) in ((x0 + 2, x1 - 1, y0, y0, True), (x0 + 2, x1 - 1, y1, y1, True),
                                     (y0 + 2, y1 - 1, x0, x0, False), (y0 + 2, y1 - 1, x1, x1, False)):
        if a1 <= a0:
            continue
        if horiz:
            seg = lm[max(0, b0 - 1):b0 + 2, max(0, a0):a1]
            ok = seg.any(axis=0).mean() >= 0.6 if seg.size else False
        else:
            seg = lm[max(0, a0):a1, max(0, b0 - 1):b0 + 2]
            ok = seg.any(axis=1).mean() >= 0.6 if seg.size else False
        n += bool(ok)
    return n


def _rects_from_mask(mask: np.ndarray) -> list:
    """Cover a boolean cell mask with rectangles (identical consecutive row runs merge)."""
    nr, nc = mask.shape
    open_: dict = {}
    out = []
    for r in range(nr + 1):
        runs = set()
        if r < nr:
            c = 0
            while c < nc:
                if mask[r, c]:
                    c2 = c
                    while c2 + 1 < nc and mask[r, c2 + 1]:
                        c2 += 1
                    runs.add((c, c2))
                    c = c2 + 1
                else:
                    c += 1
        for key in list(open_):
            if key not in runs:
                out.append((key[0], key[1], open_.pop(key), r - 1))
        for key in runs:
            open_.setdefault(key, r)
    return out


def split_panels(grid_result: GridResult, scene: SceneInfo, frame_bgr: np.ndarray | None = None) -> list:
    """Cut ``detect_grid`` panels at region boundaries and window edges.

    Every cell of every panel gets a role: inside a window -> that window's role (the picker and
    non-container windows are dropped), else right of the stash toolbar -> stash, else the
    nearest slot header's role, else own_other (lobby) / loot-or-unknown (raid).  A cell that
    touches any window it does not belong to is occluded and dropped; cells of own grids need
    drawn border lines (the lattice of a merged panel also spans empty background).  Returns
    ``[(Panel, Region)]`` ordered like the source panels."""
    W, H = scene.frame_wh
    pitch = scene.pitch
    tb = scene.toolbar
    lobby = not scene.in_raid
    sx = (tb[1] + int(0.003 * W)) if (tb and lobby) else None
    wins = [w for w in scene.windows]
    heads = []
    for name, (x, y, w, h) in scene.headers:
        role, kind = HEADERS[name]
        heads.append({'name': name, 'role': role, 'kind': kind, 'x0': x - 5, 'ybot': y + h + 5, 'y': y, 'x': x})
    out = []
    lm_cache = {}
    for panel in grid_result.panels:
        nc, nr = panel.n_cols, panel.n_rows
        labels = np.empty((nr, nc), object)           # key per cell or None
        for r in range(nr):
            for c in range(nc):
                x0, x1, y0, y1 = panel.xs[c], panel.xs[c + 1], panel.ys[r], panel.ys[r + 1]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                labels[r, c] = _cell_key(panel, c, r, cx, cy, wins, sx, lobby, heads, scene, pitch, W, H)
        # occlusion mask for the clip flags
        occl = np.zeros((nr, nc), bool)
        for r in range(nr):
            for c in range(nc):
                occl[r, c] = _touches_window(panel, c, r, wins, None)
        for key in {k for k in labels.ravel() if k is not None}:
            m = np.array([[k == key for k in row] for row in labels])
            role, side, conf, title, wi = key
            if role not in ('stash', 'container_window') and not role.startswith(('own', 'loot', 'unknown')):
                continue
            if role not in ('stash', 'container_window') and frame_bgr is not None:
                # drawn borders needed (lattice spans background); closing fills big-item interiors
                lm = lm_cache.get(id(panel))
                if lm is None:
                    lm = lm_cache[id(panel)] = line_mask(frame_bgr[max(0, panel.y0 - 2):panel.y1 + 2,
                                                                  max(0, panel.x0 - 2):panel.x1 + 2])
                lm_off = (max(0, panel.x0 - 2), max(0, panel.y0 - 2))
                shifted = _Shift(lm, lm_off)
                ev = np.zeros((nr, nc), bool)
                for r, c in zip(*np.nonzero(m)):
                    ev[r, c] = _cell_edges(shifted, panel, c, r) >= 2
                ev = cv2.morphologyEx(ev.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
                m = m & ev
            for (c0, c1, r0, r1) in _rects_from_mask(m):
                xs = panel.xs[c0:c1 + 2]
                ys = panel.ys[r0:r1 + 2]
                ct = (panel.clip_top and r0 == 0) or (r0 > 0 and bool(occl[r0 - 1, c0:c1 + 1].any()))
                cb = (panel.clip_bottom and r1 == nr - 1) or (r1 < nr - 1 and bool(occl[r1 + 1, c0:c1 + 1].any()))
                p = Panel(x0=int(xs[0]), y0=int(ys[0]), x1=int(xs[-1]) + 1, y1=int(ys[-1]) + 1,
                          ox=float(xs[0]), oy=float(ys[0]), pitch_x=panel.pitch_x, pitch_y=panel.pitch_y,
                          n_cols=c1 - c0 + 1, n_rows=r1 - r0 + 1, xs=list(xs), ys=list(ys),
                          clip_top=bool(ct), clip_bottom=bool(cb), strength=panel.strength)
                reg = Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), role, side, title, conf)
                out.append((p, reg))
    if frame_bgr is not None:
        out = _own_gear_pieces(out, grid_result, scene, frame_bgr)
    out.sort(key=lambda t: (t[0].y0, t[0].x0))
    return out


def _own_gear_pieces(out: list, grid_result: GridResult, scene: SceneInfo, frame_bgr: np.ndarray) -> list:
    """Replace / complete the pieces of the player's own gear regions with the cell groups of
    :mod:`identify.owngrid` (rig slots with gaps, pockets, a faint-lined backpack): ``detect_grid``
    needs one lattice in the stash border colour and misses them.  Own pieces ``detect_grid`` found
    are dropped where an own-grid frame covers them; the stash path is untouched."""
    from .owngrid import OWN_ROLES, region_panels
    pitch = scene.pitch
    mp = grid_result.pitch_x
    if mp and 0.9 * pitch <= mp <= 1.1 * pitch:          # EFT draws every cell at the stash pitch
        pitch = float(mp)
    wins = [w.bbox for w in scene.windows]
    mine = []
    for reg in scene.regions:
        if reg.role not in OWN_ROLES:
            continue
        for p in region_panels(frame_bgr, reg.bbox, pitch, wins):
            if any(_overlap_frac(p, q) > 0.5 for q, _ in mine):
                continue
            p.own = True
            mine.append((p, Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), reg.role, reg.side, reg.title, 0.85)))
    keep = [(p, r) for p, r in out
            if not (r.role in OWN_ROLES and any(_overlap_frac(p, q) > 0.3 for q, _ in mine))]
    for p, r in keep:
        if r.role in OWN_ROLES:
            p.own = True
    return keep + mine


def _overlap_frac(a: Panel, b: Panel) -> float:
    ix = min(a.x1, b.x1) - max(a.x0, b.x0)
    iy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / float(min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0)))


class _Shift:
    """Indexable view of a line mask cropped at ``off`` that answers in frame coordinates."""
    def __init__(self, lm, off):
        self.lm, self.off = lm, off
        self.shape = lm.shape

    def __getitem__(self, idx):
        ys, xs = idx
        ox, oy = self.off
        ys = slice(max(0, ys.start - oy), max(0, ys.stop - oy))
        xs = slice(max(0, xs.start - ox), max(0, xs.stop - ox))
        return self.lm[ys, xs]


def _touches_window(panel, c, r, wins, skip):
    x0, x1, y0, y1 = panel.xs[c], panel.xs[c + 1], panel.ys[r], panel.ys[r + 1]
    for i, w in enumerate(wins):
        if i == skip:
            continue
        wx, wy, ww, wh = w.bbox
        ix = min(x1, wx + ww + 2) - max(x0, wx - 2)
        iy = min(y1, wy + wh + 2) - max(y0, wy - 2)
        if ix > 3 and iy > 3:
            return True
    return False


def _cell_key(panel, c, r, cx, cy, wins, sx, lobby, heads, scene, pitch, W, H):
    """(role, side, confidence, title, window_index) for one cell, or None when it must not be read."""
    x0, x1, y0, y1 = panel.xs[c], panel.xs[c + 1], panel.ys[r], panel.ys[r + 1]
    for i, w in enumerate(wins):                   # windows first: they sit on top of everything
        wx, wy, ww, wh = w.bbox
        if wx <= cx <= wx + ww and wy <= cy <= wy + wh:
            if w.role == 'container_window':
                title_h = 0
                # interior only: below the title band, fully inside the border, clear of other windows
                if y0 >= wy + w.title_h - 2 and x0 >= wx and x1 <= wx + ww + 2 and y1 <= wy + wh + 2 \
                        and not _touches_window(panel, c, r, wins, i):
                    return (w.role, w.side, w.confidence, w.title, i)
            return None
    if _touches_window(panel, c, r, wins, None):   # partly covered by a window: occluded
        return None
    if lobby and sx is not None and cx >= sx and 0.0715 * H <= cy <= 0.883 * H:
        return ('stash', 'stash', 0.95, None, -1)
    # header relation
    best = None
    for h in heads:
        if h['kind'] == 'right':
            ok = cx >= h['x0'] + 1.6 * pitch and cy >= h['ybot'] - 0.2 * pitch and cx <= h['x0'] + 11 * pitch
        else:
            ok = abs(cx - (h['x0'] + 2.5 * pitch)) <= 3.2 * pitch and cx >= h['x0'] - 0.3 * pitch \
                and h['ybot'] - 0.2 * pitch <= cy <= h['ybot'] + 1.5 * pitch
            if ok:                                  # a sibling header to the right owns its own slots
                for h2 in heads:
                    if h2 is not h and abs(h2['y'] - h['y']) < 0.3 * pitch and h2['x'] > h['x'] + pitch \
                            and cx >= h2['x0'] - 0.3 * pitch:
                        ok = False
        if ok and (best is None or h['ybot'] > best['ybot']):
            best = h
    if best is not None:
        role = best['role']
        side = 'own'
        if scene.in_raid and best['x0'] >= 0.55 * W:
            role, side = 'loot', 'loot'
        return (role, side, 0.8, best['name'], -1)
    if lobby:
        return ('own_other', 'own', 0.6, None, -1)
    role, side, conf = _raid_role(scene, cx, W)
    return (role, side, conf, None, -1)


# --------------------------------------------------------------------------
# debugging aid
# --------------------------------------------------------------------------

ROLE_BGR = {'stash': (80, 200, 60), 'own_rig': (255, 160, 40), 'own_pockets': (255, 200, 80),
            'own_backpack': (255, 120, 20), 'own_pouch': (255, 90, 90), 'own_special': (255, 180, 140),
            'own_other': (200, 160, 120), 'loot': (40, 140, 255), 'container_window': (0, 230, 255),
            'picker': (60, 60, 255), 'unknown': (180, 180, 180)}


def annotate(frame_bgr: np.ndarray, scene: SceneInfo, pieces: list | None = None) -> np.ndarray:
    """Copy of the frame with regions (and the cells that will be scanned) outlined by role colour."""
    out = frame_bgr.copy()
    lw = max(2, int(round(frame_bgr.shape[0] / 540)))
    if pieces:
        for p, r in pieces:
            col = ROLE_BGR.get(r.role, (255, 255, 255))
            ov = out.copy()
            cv2.rectangle(ov, (p.x0, p.y0), (p.x1, p.y1), col, -1)
            out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
            for x in p.xs:
                cv2.line(out, (int(x), p.y0), (int(x), p.y1), col, 1)
            for y in p.ys:
                cv2.line(out, (p.x0, int(y)), (p.x1, int(y)), col, 1)
    for r in scene.regions:
        x, y, w, h = (int(v) for v in r.bbox)
        col = ROLE_BGR.get(r.role, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, lw)
        cv2.putText(out, f"{r.role} {r.title or ''}".strip(), (x + 4, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out
