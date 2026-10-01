"""
Cheap "is an item grid on screen?" detector for full game frames.

The expensive part of a scan (``identify``) must never run during raid gameplay, so the
auto-scanner first asks this module, ~2 times a second, whether the frame shows an inventory
screen (stash, container window, trader / flea sell screen).  It looks only for what EFT draws
around every inventory cell: a 1 px border in a fixed colour, (84, 81, 73) BGR.  Raid scenes
contain no *long straight lines of exactly that colour at a regular pitch in both directions*.

Cost control - the frame is never processed at full size:

* the vertical-line mask is built on every ``sy``-th row only, the horizontal-line mask on
  every ``sx``-th column only (a 1 px line survives decimation along its own direction;
  averaging it away with an area filter would destroy exactly the signal we look for),
* a morphological opening keeps only runs of >= ``RUN`` samples (art speckle never forms those),
* the mask collapses to two 1-D projections whose thin peaks are fitted with a lattice
  ``x = x0 + k * pitch``.

On a 2560x1440 frame this is a few milliseconds (see ``tests`` / README for measurements).

Also here: the other cheap per-frame signals the trigger logic needs - black / blank frame
detection, the "main menu bottom bar" check that tells a stash from the in-raid inventory,
a stability thumbnail and a perceptual hash for "same view as the last scan".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# Border colour box, BGR.  Same family as identify.grid.line_mask but a plain per-channel box so
# cv2.inRange (SIMD) can evaluate it; the identify detector adds channel-difference tests on top.
LINE_LO = (70, 68, 60)
LINE_HI = (114, 104, 94)

RUN = 6                    # min vertical/horizontal run of line colour, in decimated samples
MIN_LINES = 4              # lattice lines needed on EACH axis (= 3x3 cells)
MIN_HEIGHT = 2.0           # each of the MIN_LINES tallest lattice lines must be this many cells long
PITCH_REF_H = 1080.0       # EFT's UI is laid out for 1080 px height ...
SLOT_REF = 63.0            # ... where one slot is 63 px
PITCH_LO_SCALE = 0.45      # plausible UI scale range, relative to the 1080p layout
PITCH_HI_SCALE = 2.4       # (x allows a 4:3 -> 16:9 stretched render; EFT UI 50-150 %)

BLANK_MEAN = 2.5           # a frame whose mean level is below this ...
BLANK_P99 = 10             # ... and whose 99th percentile is below this is "black"


@dataclass
class AxisFit:
    n_lines: int           # peaks on the lattice
    pitch: float
    first: float           # position of the first / last lattice line (decimated-axis px, full-res units)
    last: float
    n_peaks: int           # strong peaks (lattice or not)
    strength: float        # mean ridge height of the lattice lines, in samples
    ratio: float = 0.0     # share of the strong peaks' height the lattice explains
    heights: tuple = ()    # inlier line heights, tallest first, in cells of line length


@dataclass
class Detection:
    is_inventory: bool
    score: float                       # 0..1, monotone in how grid-like the frame is
    x: AxisFit | None = None
    y: AxisFit | None = None
    menu_chrome: bool = False          # the main-menu bottom bar is on screen (not in a raid)
    reason: str = ''
    ms: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# blank / black frames
# --------------------------------------------------------------------------

def frame_is_blank(frame: np.ndarray) -> bool:
    """True for a (nearly) black or empty capture - what a blocked capture path returns.

    Looks at a sparse sample, so it costs microseconds.  A dark raid scene still has a
    99th percentile well above ``BLANK_P99``; a capture that is black *and* constant does not."""
    if frame is None or frame.size == 0:
        return True
    h, w = frame.shape[:2]
    s = frame[:: max(1, h // 90), :: max(1, w // 160)]
    if float(s.mean()) >= BLANK_MEAN:
        return False
    return float(np.percentile(s, 99)) < BLANK_P99


# --------------------------------------------------------------------------
# lattice fit on one projection
# --------------------------------------------------------------------------

def _peaks(proj: np.ndarray, min_val: float, min_sep: int) -> list[tuple[float, float]]:
    """Thin peaks of a projection -> [(position, height)] strongest-first non-max suppressed.

    A peak must rise above BOTH neighbourhoods 4 samples either side, so plateaus (flat grey
    regions, the edge of a large panel) produce no peak - only 1-2 sample wide lines do."""
    p = proj.astype(np.float32)
    p3 = np.convolve(p, np.ones(3, np.float32), mode='same')
    n = len(p3)
    if n < 12:
        return []
    left = np.empty_like(p3)
    right = np.empty_like(p3)
    left[4:] = p3[:-4]
    left[:4] = p3[:4]
    right[:-4] = p3[4:]
    right[-4:] = p3[-4:]
    ridge = p3 - np.maximum(left, right)
    cand = np.where((ridge >= min_val) & (p3 >= np.roll(p3, 1)) & (p3 > np.roll(p3, -1)))[0]
    cand = [int(i) for i in cand if 2 <= i < n - 2]
    cand.sort(key=lambda i: -ridge[i])
    taken: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_sep for j in taken):
            taken.append(i)
    taken.sort()
    out = []
    for i in taken:
        lo, hi = max(0, i - 1), min(n, i + 2)
        w = p[lo:hi]
        pos = float((np.arange(lo, hi) * w).sum() / w.sum()) if w.sum() > 0 else float(i)
        out.append((pos, float(ridge[i] / 3.0)))
    return out


def fit_lattice(peaks: list[tuple[float, float]], pitch_lo: float, pitch_hi: float,
                step: int = 1, rel_floor: float = 0.15) -> AxisFit | None:
    """Best regular lattice through the peak positions, weighted by peak height.

    Real border lines are long, so they are the tallest peaks; art edges and text are short.
    Candidate pitches are the gaps between nearby strong peaks; the winner is the lattice
    (pitch + anchor) that explains the most peak *height*, ties going to the larger pitch (a
    half-pitch lattice only matches by coincidence, a double-pitch one explains less)."""
    if not peaks:
        return None
    hmax = max(h for _, h in peaks)
    peaks = [(p, h) for p, h in peaks if h >= rel_floor * hmax]
    if len(peaks) < MIN_LINES:
        return None
    pos = np.asarray([p for p, _ in peaks], np.float64)
    hgt = np.asarray([h for _, h in peaks], np.float64)
    n = len(pos)
    gaps = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 4)):
            g = pos[j] - pos[i]
            if pitch_lo <= g <= pitch_hi:
                gaps.append(g)
            elif g > pitch_hi:
                break
    if len(gaps) < MIN_LINES - 1:
        return None
    gaps = np.sort(np.asarray(gaps))
    cands: list[float] = []
    i = 0
    while i < len(gaps) and len(cands) < 12:
        c = gaps[(gaps >= gaps[i]) & (gaps <= gaps[i] * 1.04)]
        if len(c) >= 2:
            cands.append(float(np.median(c)))
        i += max(1, len(c))
    scored = []
    for P in cands:
        tol = max(1.5, 0.035 * P)
        top = (0.0, None)
        for a in range(min(n, 14)):
            res = (pos - pos[a] + P / 2) % P - P / 2
            inl = np.abs(res) <= tol
            if int(inl.sum()) < MIN_LINES:
                continue
            w = float(hgt[inl].sum())
            if w > top[0]:
                top = (w, inl)
        if top[1] is not None:
            scored.append((top[0], P, top[1]))
    if not scored:
        return None
    wmax = max(w for w, _, _ in scored)
    _, P, inl = max(((w, P, inl) for w, P, inl in scored if w >= 0.97 * wmax), key=lambda t: t[1])
    sel = pos[inl]
    kk = np.round((sel - sel[0]) / P)
    if len(np.unique(kk)) >= 2:                # refine the pitch through the inlier lattice indices
        P = float(np.polyfit(kk, sel, 1)[0])
    cell = P / step                            # samples per cell along a line
    hs = tuple(sorted((float(x) / cell for x in hgt[inl]), reverse=True))
    return AxisFit(int(inl.sum()), float(P), float(sel.min()), float(sel.max()), n,
                   float(hgt[inl].mean()), float(hgt[inl].sum() / hgt.sum()), hs)


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------

def _decimation(h: int) -> int:
    """Row/column step so the analysed arrays stay ~360 samples on the short side."""
    return max(1, int(round(h / 360.0)))


def _line_projection(sub: np.ndarray, axis: int) -> np.ndarray:
    """Line-colour runs in a decimated frame -> 1-D projection along the other axis.

    ``axis=0``: vertical lines (sum down the columns); ``axis=1``: horizontal lines."""
    m = cv2.inRange(sub, LINE_LO, LINE_HI)                       # 0 / 255, uint8
    k = (1, RUN) if axis == 0 else (RUN, 1)                      # (width, height) of the run kernel
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    return cv2.reduce(m, axis, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255.0


def menu_chrome(frame: np.ndarray) -> bool:
    """The main-menu bottom bar (MAIN MENU / HIDEOUT / ... / FLEA MARKET ...) is on screen.

    It is present on every lobby screen (stash, traders, flea, hideout) and absent in a raid
    - including the in-raid inventory, which must not be scanned mid-raid.  The bar is a
    ~2.9 %-of-height strip whose top and bottom edges are 2-4 rows of pure black across the
    full width, with lit UI between them."""
    h, w = frame.shape[:2]
    n = max(8, int(h * 0.06))
    sub = frame[h - n:, :: max(1, w // 320)]
    black = (sub.max(axis=2) <= 16).mean(axis=1) >= 0.97         # per bottom row: fully black
    idx = np.where(black)[0]
    if idx.size == 0 or idx[-1] < n - 5:                         # bottom edge must be black
        return False
    lit = ~black
    d_lo, d_hi = 0.022 * h, 0.038 * h
    for top in idx:
        dist = (n - 1) - top
        if not (d_lo <= dist <= d_hi):
            continue
        between = lit[top + 1:n - 4]
        if between.size >= 6 and between.mean() >= 0.6:
            return True
    return False


class InventoryDetector:
    """``detect(frame) -> Detection`` for BGR frames of any size >= ~640x360."""

    def __init__(self, min_lines: int = MIN_LINES, min_height: float = MIN_HEIGHT):
        self.min_lines = min_lines
        self.min_height = min_height

    def detect(self, frame: np.ndarray, with_chrome: bool = True) -> Detection:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if h < 360 or w < 480:
            return Detection(False, 0.0, reason='frame too small')
        if frame_is_blank(frame):
            return Detection(False, 0.0, reason='blank frame', ms=(time.perf_counter() - t0) * 1e3)
        s = _decimation(h)
        ref = SLOT_REF * h / PITCH_REF_H                          # slot px at 100 % UI scale
        lo, hi = ref * PITCH_LO_SCALE, ref * PITCH_HI_SCALE
        min_val = max(5.0, 0.35 * ref / s)                        # a line must span ~1/3 cell
        min_sep = max(4, int(0.4 * lo))
        # vertical lines: every s-th row (columns stay full resolution -> x in real pixels)
        vx = _line_projection(np.ascontiguousarray(frame[::s]), 0)
        # horizontal lines: every s-th column (rows stay full resolution -> y in real pixels)
        hy = _line_projection(np.ascontiguousarray(frame[:, ::s]), 1)
        px = fit_lattice(_peaks(vx, min_val, min_sep), lo, hi, s)
        py = fit_lattice(_peaks(hy, min_val, min_sep), lo, hi, s)
        det = self._decide(px, py)
        det.ms = (time.perf_counter() - t0) * 1e3
        if with_chrome and det.is_inventory:
            det.menu_chrome = menu_chrome(frame)
            det.ms = (time.perf_counter() - t0) * 1e3
        return det

    def _decide(self, px: AxisFit | None, py: AxisFit | None) -> Detection:
        if px is None or py is None:
            return Detection(False, 0.0, px, py, reason='no lattice on ' + ('x' if px is None else 'y'))
        # The k-th tallest lattice line, in cells of line length: a real inventory has >= 4 border
        # lines on each axis that each run for >= 2 cells; raid scenes (measured on ~400 frames of
        # 720p-1440p recordings) never reach 1.5 and real screens never fall below 2.3.
        hx = px.heights[self.min_lines - 1] if len(px.heights) >= self.min_lines else 0.0
        hy = py.heights[self.min_lines - 1] if len(py.heights) >= self.min_lines else 0.0
        score = min(1.0, min(hx, hy) / (2.0 * self.min_height))
        if min(hx, hy) < self.min_height:
            return Detection(False, score, px, py, reason='lattice lines too short')
        return Detection(True, score, px, py)
