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
         'loot', 'container_window', 'picker', 'incoming', 'trader_goods', 'offer_window', 'handover',
         'unknown')

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
RAID_LOOT_X = 0.63         # in a raid the player's grids end left of this fraction of the width, loot starts right of it
_SIDE = {'stash': 'stash', 'loot': 'loot', 'container_window': 'stash', 'incoming': 'stash',
         'picker': 'other', 'trader_goods': 'other', 'offer_window': 'other', 'handover': 'other',
         'unknown': 'other'}
# screens that draw the stash items the deal cannot use dimmed (trader sell tab, flea offer)
DIMMING_SCENES = ("trader", "flea_offer", "flea_handover", "showcase")
# roles whose items are the player's own to sell: the sell advice covers exactly these (app.py)
SCAV_GRID_ROLES = ('own_rig', 'own_pockets', 'own_backpack', 'own_pouch', 'own_special')
ADVISED_ROLES = ('stash', 'container_window', 'incoming')
# roles whose cells are cut out of the detected panels and read
SCANNED_ROLES = ADVISED_ROLES


@dataclass
class Region:
    bbox: tuple                 # (x, y, w, h)
    role: str
    side: str = 'other'         # 'own' | 'loot' | 'stash' | 'other'
    title: str | None = None
    confidence: float = 0.0
    title_h: int = 0            # windows: height of the title band in px
    hidden: tuple = ()          # loot grids: (col, row) cells still hidden by the search (never read)

    def as_dict(self) -> dict:
        x, y, w, h = (int(v) for v in self.bbox)
        d = {'bbox': [x, y, w, h], 'role': self.role, 'side': self.side,
             'title': self.title, 'confidence': round(float(self.confidence), 2)}
        if self.hidden:
            d['hidden'] = [[int(c), int(r)] for c, r in self.hidden]
        return d


@dataclass
class SceneInfo:
    scene: str = 'unknown'      # lobby_gear|lobby_stash|flea_offer|flea_handover|trader|showcase|
                                # raid_inventory|raid_loot|none|unknown
    in_raid: bool | None = None
    regions: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    toolbar: tuple | None = None      # (x0, x1) of the stash filter toolbar
    stash_bbox: tuple | None = None   # (x, y, w, h) of the stash grid (from the toolbar + border lines)
    side_bbox: tuple | None = None    # (x, y, w, h) of the SHOWCASE grid left of the stash (trader / receive)
    headers: list = field(default_factory=list)   # [(name, (x, y, w, h))]
    zones: list = field(default_factory=list)    # coarse header-anchored search zones (internal)
    pitch: float = float(SLOT)
    frame_wh: tuple = (0, 0)
    loot: object = None               # in a raid: identify.lootgrid.LootInfo of the right-hand LOOT column
    searching: bool = False           # the LOOT header shows the red SEARCHING cancel button


# --------------------------------------------------------------------------
# floating windows
# --------------------------------------------------------------------------

def find_close_buttons(frame: np.ndarray, relaxed: bool = False) -> list:
    """Red close buttons ``[(x, y, w, h)]``: saturated dark red rounded rectangle (~1.5 x as wide
    as tall, 0.012-0.03 of the frame height) with a white glyph in the middle.

    ``relaxed`` also returns red rectangles with no (or a half hidden) glyph and a narrower shape:
    a tooltip may cover most of the button.  Such candidates are only trusted after
    :func:`trace_window` finds a window behind them."""
    H, W = frame.shape[:2]
    sc = 2 if H >= 900 else 1
    small = frame[::sc, ::sc].astype(np.int16)
    B, G, R = small[..., 0], small[..., 1], small[..., 2]
    m = ((R >= 40) & (R > 2 * G) & (R > 2 * B) & (G < 60)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for x, y, w, h, a in st[1:]:
        x, y, w, h = x * sc, y * sc, w * sc, h * sc
        if not (0.011 * H <= h <= 0.032 * H and (0.4 if relaxed else 0.8) <= w / h <= 2.1):
            continue
        if a * sc * sc < 0.7 * w * h:
            continue
        if not relaxed:
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


def _is_search_button(btn, W, H) -> bool:
    """The red cancel button of the in-raid LOOT header ("SEARCHING  x  0:05"): same look as a window's
    close button, but it sits in the header row of the right-hand column, not on a window."""
    x, y, w, h = btn
    return x + w / 2.0 >= 0.6 * W and 0.045 * H <= y + h / 2.0 <= 0.075 * H


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
    for xx in range(xe, min(W - 1, xe + 3)):
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
            # a real window border runs far beyond the title band (a tooltip box over the button
            # has a short one: skip it and keep looking further left)
            col = np.abs(frame[:, xx].astype(np.int16) - bcol).max(axis=1) <= 16
            a_ = cy
            while a_ > 0 and col[a_ - 1]:
                a_ -= 1
            b_ = cy
            while b_ < H - 1 and col[b_ + 1]:
                b_ += 1
            if b_ - a_ >= 4 * bh:
                xl = xx
                break
    if xl is None:
        # a window pushed against the screen edge has no left border: its title bar runs to x = 0
        # (the row next to the button keeps the bar colour all the way)
        row = frame[cy, :bx - 2 * bw].astype(np.int16)
        bar = frame[cy, bx - 2 * bw].astype(np.int16)          # title-bar colour just left of the button
        # (the title text sits at x ~ 10..: look at the few px before it)
        if row.shape[0] > 0.1 * W and (np.abs(row[3:7] - bar).max(axis=1) <= 10).all():
            return _trace_vertical(frame, 0, xr, y0, y1, bcol, bw, bh, cy, edge_left=True)
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
    # a menu / tooltip over the left border cuts its run short: the right border may run further
    if bot - top < 0.3 * H:
        rt, rb = _trace_vertical(frame, xl, xr, y0, y1, bcol, bw, bh, cy, only_run=True)
        if rt >= top - 2 * bh and rb > bot and rt <= top + 2 * bh:
            bot = rb
    w, h = xr - xl + 1, bot - top + 1
    if w < 4 * bw or h < 2 * bh:
        return None
    return int(xl), int(top), int(w), int(h)


def _trace_vertical(frame, xl, xr, y0, y1, bcol, bw, bh, cy, edge_left=False, only_run=False):
    """Top / bottom of a window whose left edge is the screen edge: follow the right border."""
    H = frame.shape[0]

    def run(step):
        yy, gap, last = cy, 0, cy
        while 0 <= yy + step < H:
            yy += step
            if np.abs(frame[yy, xr].astype(np.int16) - bcol).max() <= 16:
                last, gap = yy, 0
            else:
                gap += 1
                if gap > 3:
                    break
        return last
    top, bot = run(-1), run(1)
    if only_run:
        return top, bot
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

ICON_STEP = 46.0 / 1440.0          # vertical pitch of the filter icons, as a fraction of the frame height


@dataclass
class Toolbar:
    x0: int
    x1: int
    y0: int                         # first / last icon (centre)
    y1: int
    n: int                          # icons on the regular lattice
    grid_right: float = 0.0         # vertical border-line coverage right next to it (a grid starts there)


def find_toolbars(frame: np.ndarray, grid_check: bool = True) -> list:
    """Every filter toolbar on the frame, wherever it is: a column of small bright icons on one
    vertical lattice (step 0.032 H), >= 4 of them (a dimmed screen hides most), in a band 0.006-0.016 W wide.  A *stash*
    toolbar is the one with a grid starting right next to it on its right (``grid_right``): the
    trader's SHOWCASE column has the same icons but the grid is on its left."""
    H, W = frame.shape[:2]
    ya, yb = int(0.1 * H), int(0.9 * H)
    g = cv2.cvtColor(frame[ya:yb], cv2.COLOR_BGR2GRAY)
    bright = g > 150
    cs = np.convolve(bright.sum(axis=0).astype(np.float32), np.ones(3) / 3, mode='same')
    on = cs >= 0.05 * (yb - ya)
    step = ICON_STEP * H
    out = []
    x = 0
    while x < W:
        if not on[x]:
            x += 1
            continue
        x2 = x
        while x2 + 1 < W and on[x2 + 1]:
            x2 += 1
        if 0.004 * W <= x2 - x + 1 <= 0.016 * W:
            band = bright[:, x:x2 + 1].any(axis=1).astype(np.uint8)
            nb, _, st, _ = cv2.connectedComponentsWithStats(band.reshape(-1, 1), connectivity=4)
            icons = sorted((s_[1] + s_[3] / 2.0 + ya, s_[3]) for s_ in st[1:] if 0.008 * H <= s_[3] <= 0.04 * H)
            if len(icons) >= 4:
                cy = np.array([c for c, _ in icons])
                best = None
                for ph in cy:                      # the lattice phase that explains most icons
                    r = ((cy - ph) / step + 0.5) % 1.0 - 0.5
                    ok = np.abs(r) <= 0.12
                    if best is None or ok.sum() > best[0]:
                        best = (int(ok.sum()), ok)
                n, ok = best
                if n >= 4:
                    c = cy[ok]
                    out.append(Toolbar(int(x), int(x2), int(c[0]), int(c[-1]), n))
        x = x2 + 1
    if grid_check:
        for t in out:
            t.grid_right = _grid_strip(frame, t, True)
    return out


def _grid_strip(frame: np.ndarray, t: Toolbar, right: bool = True) -> float:
    """Vertical border-line coverage in the strip right next to the toolbar (1.0 = a lattice line
    runs the whole height).  Lines are the EFT cell-border colour."""
    H, W = frame.shape[:2]
    a, b = t.x1 + 3, min(W, t.x1 + int(0.013 * W))
    y0, y1 = max(0, t.y0 - 8), min(H, t.y1 + 8)
    if b - a < 4 or y1 - y0 < 40:
        return 0.0
    lm = line_mask(frame[y0:y1, a:b]).astype(np.uint8)
    k = max(20, int(0.5 * SLOT * H / 1080.0))
    vr = cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return float(vr.mean(axis=0).max())


def pick_stash_toolbar(toolbars: list):
    """The stash toolbar among ``find_toolbars``: the one with a grid on its right."""
    c = [t for t in toolbars if t.grid_right >= 0.15]
    return max(c, key=lambda t: (round(t.grid_right, 1), t.n)) if c else None


def find_toolbar(frame: np.ndarray):
    """(x0, x1) of the stash filter toolbar (a column of small bright icons), or None."""
    t = pick_stash_toolbar(find_toolbars(frame))
    return (t.x0, t.x1) if t else None


def stash_extent(frame: np.ndarray, t: Toolbar, pitch: float):
    """(y_top, y_bottom) of the stash viewport from its border lines, or None.

    The top is the strongest border line near the first icon (extended up along the cell lattice
    when the first icons are hidden).  The bottom is the lowest full-width border line; when the cell borders still run on
    below it (a clipped last row), one more cell is added."""
    H, W = frame.shape[:2]
    x0 = t.x1 + int(0.003 * W) + 4
    x1 = min(W, x0 + int(10 * pitch))
    if x1 - x0 < 4 * pitch:
        return None
    lm = line_mask(frame[:, x0:x1]).astype(np.uint8)
    k = max(20, int(0.45 * pitch))
    hr = cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    cov = hr.mean(axis=1)
    rows = np.nonzero(cov >= 0.3)[0]
    groups = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 2:
            groups[-1].append(r)
        else:
            groups.append([r])
    lines = [(int(np.mean(gp)), float(cov[gp].max())) for gp in groups]
    ys = np.array([y for y, _ in lines])

    cand = [(round(c, 1), -y, y) for y, c in lines if c >= 0.4 and t.y0 - 0.06 * H <= y <= t.y0 + 0.02 * H]
    if not cand:
        return None
    top = max(cand)[2]
    while True:                                  # hidden first icons (a menu over them): follow the lattice up
        up = [y for y, c in lines if c >= 0.4 and abs(y - (top - pitch)) <= 3]
        if not up:
            up = [y for y, c in lines if c >= 0.4 and abs(y - (top - 2 * pitch)) <= 3]
        if not up:
            break
        top = up[0]
    bots = [y for y, c in lines if c >= 0.9 and y >= top + 2 * pitch and y <= 0.995 * H]
    if not bots:
        return None
    bot = max(bots)
    # a clipped last row: the viewport ends at a black row (no border line) within one cell
    strip = frame[bot + 2:min(H, bot + int(pitch)), x0:x1].astype(np.int16)
    if strip.shape[0] > 4:
        gr = strip.mean(axis=2)
        black = (gr.mean(axis=1) < 12) & (gr.std(axis=1) < 4)
        if black.any():
            bot = int(bot + 2 + np.argmax(black))
    return int(top), int(bot)


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
        # the cache key is the bright-text bitmap, not the binarised tile: the blurred scene behind a
        # header changes the Otsu result from frame to frame, the text pixels do not
        bright = (s.min(axis=2) >= 170) if s.ndim == 3 else (s > 127)
        keys.append((g.shape, hash(bright.tobytes()), bright.shape))
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


def _grid_extent(frame: np.ndarray, zone: Region, windows: list, pitch: float, neutral_frames: bool = False):
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
    rv = rh = np.zeros_like(lm)
    if neutral_frames:
        # on the scav loot-transfer screen a grid's outer frame is a bright *neutral* line that misses
        # the border-colour box ((83, 83, 83) on the backpack, whose right column was lost): count
        # strong thin ridges as lines too
        from .owngrid import STRONG, _ridges
        rv, rh = _ridges(cv2.cvtColor(frame[zy:zy + zh, zx:zx + zw], cv2.COLOR_BGR2GRAY).astype(np.int16), STRONG)
        for w in windows:
            wx, wy, ww, wh = w.bbox
            for r_ in (rv, rh):
                r_[max(0, wy - 2 - zy):max(0, wy + wh + 3 - zy), max(0, wx - 2 - zx):max(0, wx + ww + 3 - zx)] = False
    k = max(8, int(0.5 * pitch))
    hr = cv2.morphologyEx((lm | rh).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vr = cv2.morphologyEx((lm | rv).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
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


def _text_mask(strip: np.ndarray, frac: float, floor: int) -> np.ndarray:
    """Bright text of a strip as a binary image (255 = text): a bitmap that does not depend on the
    blurred scene behind it, so the OCR cache hits for the same text."""
    g = strip.min(axis=2) if strip.ndim == 3 else strip
    if g.size == 0:
        return g
    thr = max(floor, frac * float(g.max()))
    return ((g >= thr) * 255).astype(np.uint8)


def _slot_extent(frame: np.ndarray, zone: Region, windows: list, pitch: float):
    """Fallback for ``_grid_extent``: slots drawn with a tinted (brown special slots) frame are not in
    the border colour; the own-grid frame finder (bright ridges) still finds them.  Union bbox of the
    cell groups whose top sits just under the header, or None."""
    from .owngrid import region_panels
    hy = zone.bbox[1] + 0.2 * pitch                       # header bottom
    ps = [p for p in region_panels(frame, zone.bbox, pitch, [w.bbox for w in windows])
          if hy - 0.3 * pitch <= p.y0 <= hy + 1.0 * pitch]
    if not ps:
        return None
    x0, y0 = min(p.x0 for p in ps), min(p.y0 for p in ps)
    x1, y1 = max(p.x1 for p in ps), max(p.y1 for p in ps)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


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


def find_header_candidates(frame: np.ndarray, x_max: int, y_band: tuple | None = None) -> list:
    """Bright uppercase-text blobs (x, y, w, h) that may be slot headers (``y_band`` = (y0, y1) in
    px narrows the search, default 0.07-0.93 H)."""
    H, W = frame.shape[:2]
    y0, y1 = (int(0.07 * H), int(0.93 * H)) if y_band is None else (max(0, int(y_band[0])), min(H, int(y_band[1])))
    if x_max < 8 or y1 - y0 < 8:
        return []
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

def find_banner(frame: np.ndarray):
    """The big centred screen title ("RAID ENDED"): bright text blob (x, y, w, h) in the top
    centre, or None."""
    H, W = frame.shape[:2]
    y1 = int(0.085 * H)
    xa, xb = int(0.3 * W), int(0.7 * W)
    m = (frame[:y1, xa:xb].min(axis=2) >= 190).astype(np.uint8)
    k = max(9, int(round(0.03 * H)))                  # joins the words of the title
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    c = [(int(x + xa), int(y), int(w), int(h)) for x, y, w, h, a in st[1:]
         if 0.022 * H <= h <= 0.05 * H and 0.08 * W <= w <= 0.3 * W]
    return max(c, key=lambda b: b[2] * b[3]) if c else None


def _fuzzy(text: str, word: str, thr: float = 0.75) -> bool:
    t = re.sub(r'[^A-Z]', '', (text or '').upper())
    if not t:
        return False
    if word in t:
        return True
    r = max((difflib.SequenceMatcher(None, t[i:i + len(word)], word).ratio()
             for i in range(0, max(1, len(t) - len(word) + 1))), default=0.0)
    return r >= thr


def _menu_chrome(frame) -> bool:
    try:
        from autoscan.detect import menu_chrome
        return bool(menu_chrome(frame))
    except Exception:
        return False


def analyze_scene(frame_bgr: np.ndarray, *, in_raid=None) -> SceneInfo:
    H, W = frame_bgr.shape[:2]
    pitch = SLOT * min(W / 1920.0, H / 1080.0)
    info = SceneInfo(in_raid=in_raid, pitch=pitch, frame_wh=(W, H))

    # ---- windows --------------------------------------------------------
    wins = []                       # [rect, title_h]

    def add_window(btn):
        if _is_search_button(btn, W, H):
            info.searching = True
            return
        r = trace_window(frame_bgr, btn)
        if r is not None and not any(abs(r[0] - w[0][0]) < 4 and abs(r[1] - w[0][1]) < 4 for w in wins):
            wins.append([r, btn[1] + btn[3] - r[1] + 3])
    for btn in find_close_buttons(frame_bgr):
        add_window(btn)
    for btn in find_close_buttons(frame_bgr, relaxed=True):      # a tooltip may hide most of the button
        if not any(_inside((btn[0] + btn[2] / 2, btn[1] + btn[3] / 2), w[0]) for w in wins):
            add_window(btn)

    # ---- stash toolbar ------------------------------------------------------
    toolbars = find_toolbars(frame_bgr)
    st = pick_stash_toolbar(toolbars)
    tb = (st.x0, st.x1) if st else None
    info.toolbar = tb
    # a stash toolbar exists on lobby screens only (whatever a caller guessed from the menu bar)
    if tb is not None:
        lobby = True
    elif in_raid is not None:
        lobby = not in_raid
    else:
        lobby = _menu_chrome(frame_bgr)
    if st:
        ext = stash_extent(frame_bgr, st, pitch)
        top, bot = ext if ext else (int(0.0715 * H), int(0.883 * H))
        sx = tb[1] + int(0.003 * W)
        info.stash_bbox = (sx, top, int(10 * pitch + 0.006 * W), bot - top)
    other_tb = [t for t in toolbars if st is not None and t is not st and t.x1 < st.x0 - 0.1 * W
                and t.grid_right < 0.15 and t.n >= 7]

    # ---- OCR phase 1: window titles, the SHOWCASE header band, the screen banner ---------
    tstrips = []
    for rect, th in wins:
        x, y, w, h = rect
        tw = int(min(0.4 * w, 7.5 * th))
        tstrips.append(frame_bgr[y + 2:y + th - 3, x + 3:x + 3 + tw])
    bcands = []
    if st and lobby:
        y_top = info.stash_bbox[1]
        bcands = find_header_candidates(frame_bgr, tb[0] - int(0.02 * W), (y_top - 0.04 * H, y_top - 0.003 * H))
        bcands = [c for c in bcands if not any(_inside((c[0] + c[2] / 2, c[1] + c[3] / 2), w[0]) for w in wins)][:6]
    banner = find_banner(frame_bgr) if (st and lobby) else None
    bstrips = [frame_bgr[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] for (x, y, w, h) in bcands]
    if banner is not None:
        x, y, w, h = banner
        bstrips.append(frame_bgr[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3])
    texts = _ocr_strips(tstrips + bstrips)
    ttexts, btexts = texts[:len(tstrips)], texts[len(tstrips):]
    banner_text = btexts[len(bcands)] if banner is not None else ''
    show_hdr = refresh = None
    for c, t in zip(bcands, btexts):
        if show_hdr is None and _fuzzy(t, 'SHOWCASE', 0.78):
            show_hdr = c
        if _fuzzy(t, 'REFRESH', 0.8):
            refresh = c
    raid_ended = banner is not None and _fuzzy(banner_text, 'RAIDENDED', 0.8)

    # ---- window regions -----------------------------------------------------
    lobby_window = False
    for (rect, th), title in zip(wins, ttexts):
        title = _clean_title(title)
        low = re.sub(r'[^a-z]', '', title.lower())
        if difflib.SequenceMatcher(None, low[:9], 'available').ratio() >= 0.75 or low.startswith('availab'):
            role, conf = 'picker', 0.95
        elif difflib.SequenceMatcher(None, low[:13], 'offercreation').ratio() >= 0.8:
            role, conf = 'offer_window', 0.9
        elif difflib.SequenceMatcher(None, low[:8], 'handover').ratio() >= 0.8:
            role, conf = 'handover', 0.9
        elif _has_grid_lines(frame_bgr, rect, th):
            role, conf = 'container_window', 0.85
        else:
            role, conf = 'unknown', 0.6
        lobby_window = lobby_window or role in ('offer_window', 'handover')
        info.windows.append(Region(rect, role, _SIDE.get(role, 'other'), title or None, conf, th))
    if lobby_window:
        lobby = True
    info.in_raid = not lobby
    for w in info.windows:
        if info.in_raid and w.role == 'container_window':
            # a window of the looted container opens on the right, one opened from the backpack /
            # rig next to the player's own grids: the player's gear column ends at ~0.63 W
            w.side = 'loot' if (w.bbox[0] + w.bbox[2] / 2) >= RAID_LOOT_X * W else 'own'
    info.regions.extend(info.windows)

    # ---- in-raid LOOT column (right of the player's gear) -----------------------------
    loot_strips = []
    if info.in_raid:
        from .lootgrid import analyze_loot
        info.loot = analyze_loot(frame_bgr, pitch)
        lo = info.loot
        if lo.origin is not None:
            ox, oy = lo.origin
            if lo.panel is None and lo.placeholder is None:       # nothing drawn: is there a LOOT header at all?
                loot_strips.append(_text_mask(frame_bgr[max(0, int(oy - 0.34 * pitch)):int(oy - 0.04 * pitch),
                                                        ox:int(ox + 1.9 * pitch)], 0.6, 90))
            else:                                                 # the container's name at the top right
                loot_strips.append(_text_mask(frame_bgr[int(0.0455 * H):int(0.0665 * H), int(0.89 * W):int(0.995 * W)],
                                              0.65, 130))

    # ---- stash + SHOWCASE regions -----------------------------------------------------
    side_kind = None
    if st:
        info.regions.append(Region(info.stash_bbox, 'stash', 'stash', None, 0.95))
        if show_hdr is not None or other_tb:
            side_kind = 'trader' if (refresh is not None or (other_tb and show_hdr is None)) else 'showcase'
            gx0 = (show_hdr[0] - int(0.004 * W)) if show_hdr is not None else int(0.0047 * W)
            gw = min(int(10 * pitch + 0.008 * W), tb[0] - 4 - gx0)
            info.side_bbox = (gx0, info.stash_bbox[1], gw, info.stash_bbox[3])
            role = 'trader_goods' if side_kind == 'trader' else 'incoming'
            info.regions.append(Region(info.side_bbox, role, _SIDE[role], 'SHOWCASE', 0.85))

    # ---- OCR phase 2: slot headers (not on trader / receive screens) --------------------
    if side_kind is None:
        x_max = (tb[0] - 4) if (tb and lobby) else int(0.97 * W)
        cands = find_header_candidates(frame_bgr, x_max)
        cands = [c for c in cands if not any(_inside((c[0] + c[2] / 2, c[1] + c[3] / 2), w[0]) for w in wins)]
        strips = [frame_bgr[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] for (x, y, w, h) in cands]
        read = _ocr_strips(strips + loot_strips)
        for c, t in zip(cands, read):
            mh = _match_header(t)
            if mh:
                info.headers.append((mh[0], c))
        if loot_strips:
            txt = read[len(strips)]
            if info.loot.panel is None and info.loot.placeholder is None:
                info.loot.header = _fuzzy(txt, 'LOOT', 0.7)
            else:
                info.loot.name = _clean_title(txt) or None
        _add_loot_regions(info)

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
        if info.in_raid and side == 'own':             # the player's grids end before the LOOT column
            zx1 = min(zx1, int(RAID_LOOT_X * W))
        if kind == 'under':
            for n2, (x2, y2, w2, h2) in info.headers:  # next header to the right on the same row
                if abs(y2 - y) < 0.3 * pitch and x2 > x + pitch:
                    zx1 = min(zx1, x2 - 8)
        info.zones.append(Region((int(zx), int(ybot - 0.2 * pitch), int(zx1 - zx), int(zy1 - ybot + 0.2 * pitch)),
                                 role, side, name, 0.8))
    for z in info.zones:
        if z.role in ('own_other',):
            continue
        bb = _grid_extent(frame_bgr, z, info.windows, pitch, neutral_frames=raid_ended)
        if bb is None and info.in_raid and z.role in ('own_special', 'own_pockets', 'own_rig'):
            bb = _slot_extent(frame_bgr, z, info.windows, pitch)
        if bb is not None:
            info.regions.append(Region(bb, z.role, z.side, z.title, 0.85))

    # ---- scene class ---------------------------------------------------------------
    nh = len(info.headers)
    if info.in_raid:
        loot_h = any(r.side == 'loot' for r in info.regions if r.title in HEADERS)
        loot_side = loot_h or (info.loot is not None and info.loot.found) or             any(r.side == 'loot' for r in info.windows if r.role == 'container_window')
        if nh >= 2:
            info.scene = 'raid_loot' if loot_side else 'raid_inventory'
        else:
            info.scene = 'raid_loot' if (info.windows or loot_side) else 'unknown'
    elif tb is not None:
        offer = any(w.role == 'offer_window' for w in info.windows) or (not info.windows and tb[0] < 0.05 * W)
        if raid_ended:
            info.scene = 'scav_transfer'
            for r in info.regions:                 # the scav's grids are moved to the stash
                if r.role in SCAV_GRID_ROLES:
                    r.role, r.side = 'incoming', 'stash'
        elif side_kind:
            info.scene = side_kind
        elif offer:
            info.scene = 'flea_offer'
        else:
            info.scene = 'lobby_gear' if nh >= 3 else 'lobby_stash'
    elif any(w.role == 'handover' for w in info.windows):
        info.scene = 'flea_handover'
    elif any(w.role == 'offer_window' for w in info.windows):
        info.scene = 'flea_offer'
    elif nh >= 3 or info.windows:
        # a window over the lobby inventory hid the stash toolbar (the menu bar says lobby)
        info.scene = ('lobby_gear' if nh >= 3 else 'lobby_stash') if (not info.in_raid and any(
            w.role == 'container_window' for w in info.windows)) or nh >= 3 and not info.in_raid else 'unknown'
    else:
        info.scene = 'none'
    return info


def _add_loot_regions(info: SceneInfo) -> None:
    """Regions of the LOOT column: the container's grid (cells hidden by the search listed in
    ``hidden``) or the UNSEARCHED stand-in (everything hidden)."""
    lo = info.loot
    if lo is None:
        return
    title = lo.name
    if lo.panel is not None:
        p = lo.panel
        info.regions.append(Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), 'loot', 'loot', title, 0.9,
                                   hidden=tuple(sorted(lo.hidden))))
    elif lo.placeholder is not None:
        info.regions.append(Region(lo.placeholder, 'loot', 'loot', title or 'UNSEARCHED', 0.8, hidden=((0, 0),)))


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
    if sx is not None and scene.stash_bbox is None:        # hand-built SceneInfo: the lobby band
        scene.stash_bbox = (sx, int(0.0715 * H), int(10 * pitch + 0.006 * W), int(0.811 * H))
    heads = []
    for name, (x, y, w, h) in scene.headers:
        role, kind = HEADERS[name]
        heads.append({'name': name, 'role': role, 'kind': kind, 'x0': x - 5, 'ybot': y + h + 5, 'y': y, 'x': x})
    out = []
    lm_cache = {}
    refit = None
    if lobby and scene.stash_bbox is not None and frame_bgr is not None:
        refit = _stash_refit(grid_result, scene, frame_bgr)
    for panel in list(grid_result.panels) + ([refit] if refit is not None else []):
        only_stash = panel is refit
        if only_stash:
            n_orig = len(out)
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
            if only_stash and not (role == 'stash' and wi == -1):
                continue
            if role not in SCANNED_ROLES and not role.startswith(('own', 'loot', 'unknown')):
                continue
            if _needs_line_evidence(role, title) and frame_bgr is not None:
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
                          clip_top=bool(ct), clip_bottom=bool(cb), strength=panel.strength,
                          dim=panel.dim or (role == 'stash' and scene.scene in DIMMING_SCENES))
                reg = Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), role, side, title, conf)
                out.append((p, reg))
    if refit is not None:                  # the fitted stash replaces whatever detect_grid found there
        out = [(p, r) for p, r in out[:n_orig] if not (r.role == 'stash' and r.title is None)] + out[n_orig:]
    if frame_bgr is not None:
        out = _own_gear_pieces(out, grid_result, scene, frame_bgr)
    out.extend(_loot_pieces(scene, wins))
    out.sort(key=lambda t: (t[0].y0, t[0].x0))
    return out


def _loot_box(scene: SceneInfo):
    """(x0, y0, x1, y1) of the LOOT column's grid / stand-in, or None."""
    lo = scene.loot
    if lo is None:
        return None
    if lo.panel is not None:
        p = lo.panel
        return (p.x0, p.y0, p.x1, p.y1)
    if lo.placeholder is not None:
        x, y, w, h = lo.placeholder
        return (x, y, x + w, y + h)
    return None


def _loot_pieces(scene: SceneInfo, wins) -> list:
    """``[(Panel, Region)]`` of the revealed cells of the looted container's grid (found by
    :mod:`identify.lootgrid`, not by ``detect_grid``): cells still hidden by the search, and cells a
    window covers, are left out."""
    lo = scene.loot
    if lo is None or lo.panel is None:
        return []
    pn = lo.panel
    nc, nr = pn.n_cols, pn.n_rows
    m = np.ones((nr, nc), bool)
    for c, r in lo.hidden:
        if r < nr and c < nc:
            m[r, c] = False
    for r in range(nr):
        for c in range(nc):
            if m[r, c] and _touches_window(pn, c, r, wins, None):
                m[r, c] = False
    out = []
    for (c0, c1, r0, r1) in _rects_from_mask(m):
        xs, ys = pn.xs[c0:c1 + 2], pn.ys[r0:r1 + 2]
        p = Panel(x0=int(xs[0]), y0=int(ys[0]), x1=int(xs[-1]) + 1, y1=int(ys[-1]) + 1,
                  ox=float(xs[0]), oy=float(ys[0]), pitch_x=pn.pitch_x, pitch_y=pn.pitch_y,
                  n_cols=c1 - c0 + 1, n_rows=r1 - r0 + 1, xs=list(xs), ys=list(ys),
                  clip_top=False, clip_bottom=bool(pn.clip_bottom and r1 == nr - 1),
                  strength=pn.strength, own=True)
        title = next((r.title for r in scene.regions if r.role == 'loot' and r.bbox[:2] == (pn.x0, pn.y0)), None)
        out.append((p, Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), 'loot', 'loot', title, 0.9)))
    return out


def _stash_cells(panels, bbox) -> int:
    """Number of cells of ``panels`` whose centre lies in the stash ``bbox``."""
    n = 0
    for p in panels:
        for r in range(p.n_rows):
            for c in range(p.n_cols):
                cx, cy = (p.xs[c] + p.xs[c + 1]) / 2.0, (p.ys[r] + p.ys[r + 1]) / 2.0
                if bbox[0] <= cx <= bbox[0] + bbox[2] and bbox[1] <= cy <= bbox[1] + bbox[3]:
                    n += 1
    return n


def _stash_refit(grid_result: GridResult, scene: SceneInfo, frame_bgr: np.ndarray):
    """The stash lattice fitted inside the known stash region (:mod:`identify.stashfit`) when
    ``detect_grid`` found clearly fewer stash cells than that lattice holds (the trader screens dim
    the items the trader does not buy: their borders fall outside the line-colour box).  ``None``
    keeps the ordinary path."""
    from .stashfit import fit_stash_panel
    sb = scene.stash_bbox
    pitch = scene.pitch
    mp = grid_result.pitch_x
    if mp and 0.9 * pitch <= mp <= 1.1 * pitch:
        pitch = float(mp)
    fit = fit_stash_panel(frame_bgr, sb, pitch)
    if fit is None:
        return None
    have = _stash_cells(grid_result.panels, sb)
    if have >= 0.9 * fit.n_cols * fit.n_rows:
        return None
    return fit


def _needs_line_evidence(role: str, title) -> bool:
    """Cells of own-gear style regions need drawn borders (the lattice of a merged panel also spans
    background); stash / window / SHOWCASE grids do not."""
    if role in ('stash', 'container_window'):
        return False
    if role == 'incoming':
        return title in HEADERS                     # the scav's rig / pockets / backpack
    return True


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
        if not _is_gear(reg):
            continue
        for p in region_panels(frame_bgr, reg.bbox, pitch, wins):
            if any(_overlap_frac(p, q) > 0.5 for q, _ in mine):
                continue
            if _in_box(scene.stash_bbox, p) or (reg.title != 'SHOWCASE' and _in_box(scene.side_bbox, p)):
                continue                                  # the stash is never a gear grid
            p.own = True
            mine.append((p, Region((p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0), reg.role, reg.side, reg.title, 0.85)))
    keep = [(p, r) for p, r in out
            if not (_is_gear(r) and any(_overlap_frac(p, q) > 0.3 for q, _ in mine))]
    for p, r in keep:
        if _is_gear(r):
            p.own = True
    return keep + mine


def _in_box(box, p: Panel) -> bool:
    if box is None:
        return False
    cx, cy = (p.x0 + p.x1) / 2.0, (p.y0 + p.y1) / 2.0
    return box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[3]


def _is_gear(reg) -> bool:
    """A grid that needs the faint-line cell finder: the player's own gear, the scav's gear on the
    loot-transfer screen, the (dark, mostly empty) SHOWCASE grid of incoming items."""
    from .owngrid import OWN_ROLES
    return reg.role in OWN_ROLES or reg.role == 'incoming'


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
        if i == skip or w.role == 'offer_window':
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
    if scene.in_raid and not panel.own and not 0.8 * pitch <= panel.pitch_x <= 1.25 * pitch:
        return None                                # a lattice fitted to item art (EFT draws every cell at one pitch)
    for i, w in enumerate(wins):                   # windows first: they sit on top of everything
        wx, wy, ww, wh = w.bbox
        if wx <= cx <= wx + ww and wy <= cy <= wy + wh:
            if w.role == 'offer_window':               # its stash grid is the stash: decided below
                break
            if w.role == 'container_window':
                title_h = 0
                # interior only: below the title band, fully inside the border, clear of other windows
                if y0 >= wy + w.title_h - 2 and x0 >= wx and x1 <= wx + ww + 2 and y1 <= wy + wh + 2 \
                        and not _touches_window(panel, c, r, wins, i):
                    return (w.role, w.side, w.confidence, w.title, i)
            return None
    if _touches_window(panel, c, r, wins, None):   # partly covered by a window: occluded
        return None
    lb = _loot_box(scene)
    if lb is not None and lb[0] - 4 <= cx <= lb[2] + 4 and lb[1] - 4 <= cy <= lb[3] + 4:
        return None                                    # the looted container: read by _loot_pieces
    sb = scene.stash_bbox
    if lobby and sb is not None and sb[0] <= cx <= sb[0] + sb[2] and sb[1] <= cy <= sb[1] + sb[3]:
        return ('stash', 'stash', 0.95, None, -1)
    ob = scene.side_bbox
    if lobby and ob is not None and ob[0] <= cx <= ob[0] + ob[2] and ob[1] <= cy <= ob[1] + ob[3]:
        if scene.scene == 'trader':
            return None                                # the trader's goods are never the player's
        return ('incoming', 'stash', 0.85, 'SHOWCASE', -1)
    if scene.scene in ('trader', 'showcase', 'flea_offer', 'flea_handover'):
        return None                                    # deal panel, offers, ...: not the player's grids
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
        # the nearest column wins (a backpack grid is also inside the 11-cell reach of the equipment
        # headers of the column to its left), then the lowest header
        if ok and (best is None or (h['x0'] // (0.5 * pitch), h['ybot']) > (best['x0'] // (0.5 * pitch), best['ybot'])):
            best = h
    if best is not None:
        role = best['role']
        side = 'own'
        if scene.in_raid and best['x0'] >= 0.55 * W:
            role, side = 'loot', 'loot'
        elif scene.scene == 'scav_transfer' and role in SCAV_GRID_ROLES:
            role, side = 'incoming', 'stash'
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
            'picker': (60, 60, 255), 'unknown': (180, 180, 180), 'incoming': (255, 0, 200),
            'trader_goods': (0, 120, 255), 'offer_window': (200, 200, 0), 'handover': (0, 80, 200)}


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
