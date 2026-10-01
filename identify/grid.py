"""
Grid / panel detection from EFT's drawn border lines.

Why line structure and not autocorrelation of image gradients
-------------------------------------------------------------
The legacy detector autocorrelated Sobel energy.  Item art, labels and the UI
chrome all contribute periodic-looking energy, which is why it needed layers of
"chrome outlier" heuristics and still missed the first row of ``stash1.png``
(grid origin y=141 although a row starts at y=77) and failed outright on the
committed version.

EFT draws every item/cell with a 1 px border in a fixed colour, (84, 81, 73) BGR
(RatEye uses the same constant).  Thresholding for that colour (+ a small
tolerance, to survive blending over bright art) gives a clean line mask whose
structure *is* the grid:

1. ``line_mask``                    colour threshold
2. horizontal / vertical opening    keeps only long straight runs (text and art
                                    anti-aliasing never form >= 1/3-pitch runs)
3. connected components of the mesh each component is one panel (stash, an open
                                    container, an equipment tile, ...)
4. per component: peaks of the line projections -> lattice fit
                                    ``x_k = ox + k * pitch`` (pitch from the minimal
                                    gap cluster, refined by least squares; x and y
                                    independent so a stretched capture still works)
5. cross-checks                     the pitch must be shared by all panels of a frame
                                    (one UI scale per frame); optionally compared
                                    with the scale implied by the screen height.
6. extents                          the panel's vertical extent comes from the border
                                    lines themselves, so partially visible top /
                                    bottom rows (scrolled stash) are included and
                                    flagged ``clip_top`` / ``clip_bottom`` rather than
                                    silently dropped.

Because the lattice is anchored on lines that really exist (not on a comb fit
over noisy energy) the top row cannot be lost: if there are border pixels there,
there is a row there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import LINE_BGR, LINE_TOL, MIN_PITCH, MAX_PITCH, SLOT


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class Panel:
    """One rectangular grid of cells (a stash tab, an open container, ...).

    ``xs`` / ``ys`` are the *observed* pixel positions of the cell boundary
    lines (``len == n_cols+1`` / ``n_rows+1``).  Row 0 is the first *visible* row.
    ``clip_top`` / ``clip_bottom`` say that the first / last row is cut off by the
    panel viewport (its outer boundary is the viewport edge, not a cell line), so
    items there are only partly visible.
    """
    x0: int
    y0: int
    x1: int
    y1: int
    ox: float
    oy: float
    pitch_x: float
    pitch_y: float
    n_cols: int
    n_rows: int
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    clip_top: bool = False
    clip_bottom: bool = False
    strength: float = 0.0

    # --- geometry helpers -------------------------------------------------
    def rect(self, col: int, row: int, w: int = 1, h: int = 1) -> tuple[int, int, int, int]:
        """Pixel rect ``(x, y, width, height)`` of a W x H footprint, *including*
        the closing border line (so a 1x1 at 63 px pitch is 64x64, matching the
        icon geometry)."""
        r0 = row
        x = self.xs[col]
        y = self.ys[r0]
        x1 = self.xs[min(col + w, self.n_cols)]
        y1 = self.ys[min(r0 + h, self.n_rows)]
        return int(x), int(y), int(x1 - x + 1), int(y1 - y + 1)

    @property
    def scale_x(self) -> float:
        return self.pitch_x / SLOT

    @property
    def scale_y(self) -> float:
        return self.pitch_y / SLOT

    def as_grid_dict(self) -> dict:
        """Legacy-style grid dict (cell_w/cell_h/origin_x/origin_y)."""
        return {'cell_w': self.pitch_x, 'cell_h': self.pitch_y,
                'origin_x': float(self.xs[0]), 'origin_y': float(self.ys[0]),
                'x0': self.x0, 'y0': self.y0, 'x1': self.x1, 'y1': self.y1}


@dataclass
class GridResult:
    panels: list
    pitch_x: float | None
    pitch_y: float | None
    warnings: list = field(default_factory=list)
    hint_scale: float | None = None     # pitch implied by the screen size, if known
    mode: str = 'strict'                # line model that produced it: strict | ridge | ridge-soft
    quality: float = 0.0                # sum over panels of cells * lattice consistency


# --------------------------------------------------------------------------
# line mask
# --------------------------------------------------------------------------

def line_mask(img_bgr: np.ndarray, tol: int | None = None) -> np.ndarray:
    """Boolean mask of pixels that look like the EFT border line.

    The line is (84, 81, 73) BGR over most of the stash but is brightened by
    the UI's top gradient to about (107, 94, 84) (measured on stash1.png: the
    first row's lines are *not* the nominal colour, which is exactly why a plain
    colour-distance threshold loses the top row).  So the test is a neutral,
    slightly warm grey *range* rather than a fixed colour: B in 72..112,
    G in 70..102, R in 62..92 with B>=R, B-R<=24 and B-G<=16.  ``tol`` widens/narrows
    the box symmetrically (default 0).  Art with a similar colour passes this per
    pixel, but never as a long continuous run at one exact column - that
    continuity test lives in :mod:`identify.segment` and in the run opening below.
    """
    t = 0 if tol is None else int(tol)
    im = img_bgr.astype(np.int16)
    B, G, R = im[..., 0], im[..., 1], im[..., 2]
    return ((B >= 72 - t) & (B <= 112 + t) & (G >= 70 - t) & (G <= 102 + t)
            & (R >= 62 - t) & (R <= 92 + t)
            & ((B - R) >= 2) & ((B - R) <= 24 + t) & ((B - G) <= 16 + t) & ((G - R) >= 0))


def ridge_masks(img_bgr: np.ndarray, contrast: int | None = 14, neutral_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fallback for captures whose lines were resampled (windowed / scaled /
    stretched), where the exact colour no longer survives: neutral-grey 1-2 px
    ridges that stand out from both neighbours.  Returns (horizontal, vertical)
    line masks.  Noisier than :func:`line_mask`, hence only a fallback."""
    contrast = 14 if contrast is None else contrast
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    neutral = hsv[:, :, 1] < 70
    bright = (g > 40) & (g < 200)
    th_v = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    th_h = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    base = (neutral & bright) if neutral_only else bright
    return (th_h > contrast) & base, (th_v > contrast) & base


def expected_pitch(frame_w: int, frame_h: int) -> float:
    """px/slot implied by the *screen* size (EFT scales its UI by min(w/1920, h/1080)).
    Only meaningful when the frame is the whole screen, not a cropped region."""
    return SLOT * min(frame_w / 1920.0, frame_h / 1080.0)


def _runs(mask: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    """Keep only straight runs >= ``length`` px (closing first bridges 1-2 px gaps
    where art/labels cover the line)."""
    m = mask.astype(np.uint8)
    kc = (5, 1) if horizontal else (1, 5)
    ko = (length, 1) if horizontal else (1, length)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kc))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, ko))
    return m.astype(bool)


# --------------------------------------------------------------------------
# 1-D lattice utilities
# --------------------------------------------------------------------------

def _peaks(proj: np.ndarray, min_val: float, min_sep: int = 6) -> list[tuple[float, float]]:
    """Sub-pixel peaks of a projection as ``[(position, strength)]``; the
    strongest peak wins within ``min_sep`` px."""
    p = np.convolve(proj.astype(np.float64), np.ones(3), mode='same')
    cand = [i for i in range(1, len(p) - 1)
            if p[i] >= min_val and p[i] >= p[i - 1] and p[i] > p[i + 1]]
    cand.sort(key=lambda i: -p[i])
    taken: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_sep for j in taken):
            taken.append(i)
    taken.sort()
    out = []
    for i in taken:
        lo, hi = max(0, i - 1), min(len(proj), i + 2)
        w = proj[lo:hi].astype(np.float64)
        pos = float((np.arange(lo, hi) * w).sum() / w.sum()) if w.sum() > 0 else float(i)
        out.append((pos, float(p[i])))
    return out


def _min_gap_pitch(pos: list[float], lo: float = MIN_PITCH, hi: float = MAX_PITCH) -> float | None:
    """Pitch = typical *smallest* gap between adjacent lines.  Items only ever
    remove interior lines, so every gap is an integer multiple of the pitch and
    the smallest cluster is the pitch itself."""
    if len(pos) < 2:
        return None
    gaps = np.diff(np.asarray(sorted(pos)))
    gaps = gaps[(gaps >= lo) & (gaps <= hi)]
    if gaps.size == 0:
        return None
    g0 = gaps.min()
    cl = gaps[np.abs(gaps - g0) <= 0.12 * g0]
    return float(np.median(cl))


def _lattice_search(pos: list[float], strength: list[float], hint: float | None,
                    lo: float = MIN_PITCH, hi: float = MAX_PITCH) -> float | None:
    """Pitch whose lattice explains the most line energy.

    Score(P) = max over anchor lines of sum_i s_i * exp(-(residual_i / sigma)^2)
    where the residual is the distance of line i to the nearest lattice line
    anchored there.  Sub-multiples (P/2) explain the same lines, multiples (2P)
    explain fewer, so among near-ties the *largest* pitch wins.  This is robust
    to missing interior lines (multi-cell items) and to a few spurious peaks,
    unlike a plain minimum-gap estimate."""
    if len(pos) < 2:
        return None
    x = np.asarray(pos, float)
    w = np.asarray(strength, float)
    w = w / w.max()
    cands = np.arange(lo, hi, 0.25)
    scores = np.zeros(len(cands))
    for ci, P in enumerate(cands):
        d = x[None, :] - x[:, None]                       # anchor j (rows) -> line i (cols)
        kk = np.round(d / P)
        res = d - kk * P
        sig = 0.03 * P + 0.7
        hit = np.abs(res) <= 2.5 * sig
        sc = (w[None, :] * np.exp(-(res / sig) ** 2)).sum(axis=1)
        # lattice slots between the first and last explained line that carry no line at all
        kmax = np.where(hit, kk, -1e9).max(axis=1)
        kmin = np.where(hit, kk, 1e9).min(axis=1)
        misses = (kmax - kmin + 1) - hit.sum(axis=1)
        scores[ci] = (sc - 0.5 * np.maximum(misses, 0)).max()
    smax = scores.max()
    if smax < 1.9:                                        # need >= 2 consistent lines
        return None
    ok = np.where(scores >= 0.93 * smax)[0]
    near = [i for i in ok if hint and abs(cands[i] - hint) / hint < 0.15]
    if near:                       # a plausible screen-implied scale breaks ties between multiples
        best = near[int(np.argmax(scores[near]))]
    else:                          # no/implausible hint: the largest pitch that explains the lines
        best = ok[np.argmax(cands[ok])]
    # refine inside the winning plateau (centroid of contiguous near-max run)
    i0 = best
    j = i0
    while j > 0 and scores[j - 1] >= 0.97 * scores[best]:
        j -= 1
    k = i0
    while k < len(cands) - 1 and scores[k + 1] >= 0.97 * scores[best]:
        k += 1
    return float(cands[(j + k) // 2]) if not near else float(cands[best])


def _fit_lattice(pos: list[float], pitch: float, tol_frac: float = 0.1):
    """Robust ``x = o + k*pitch`` through the positions: the anchor is the peak whose lattice
    explains the most others (so an off-lattice viewport edge cannot poison the fit), then
    least squares with outlier rejection.  Returns ``(origin, pitch, inlier positions)``
    or ``None``."""
    pos = sorted(pos)
    if not pos:
        return None
    arr = np.asarray(pos)
    best_o, best_n = pos[0], -1
    for cand in pos[:16]:
        res = (arr - cand + pitch / 2) % pitch - pitch / 2
        n = int((np.abs(res) <= tol_frac * pitch).sum())
        if n > best_n:
            best_o, best_n = cand, n
    o = best_o
    for _ in range(3):
        k = np.round((arr - o) / pitch)
        res = arr - (o + k * pitch)
        keep = np.abs(res) <= tol_frac * pitch
        if keep.sum() < 2 or len(np.unique(k[keep])) < 2:
            break
        A = np.vstack([np.ones(int(keep.sum())), k[keep]]).T
        sol, *_ = np.linalg.lstsq(A, arr[keep], rcond=None)
        o, pitch = float(sol[0]), float(sol[1])
    k = np.round((arr - o) / pitch)
    res = arr - (o + k * pitch)
    inl = np.abs(res) <= tol_frac * pitch
    return o, pitch, [float(p) for p, i in zip(pos, inl) if i]


def _phase_from_pitch(pos: list[float], pitch: float) -> float:
    """Lattice origin (the residue of positions mod pitch, circular mean)."""
    ang = np.array([(p % pitch) / pitch * 2 * np.pi for p in pos])
    frac = (np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()) / (2 * np.pi)) % 1.0
    return float(frac * pitch)


# --------------------------------------------------------------------------
# axis analysis
# --------------------------------------------------------------------------

@dataclass
class _Axis:
    origin: float     # a lattice line position
    pitch: float
    first: float      # first observed line position (inlier)
    last: float       # last observed line position (inlier)
    n_lines: int
    strength: float
    ratio: float = 1.0   # inlier peaks / all strong peaks (lattice consistency)


def _analyse_axis(proj: np.ndarray, pitch_ref: float | None, hint: float | None) -> _Axis | None:
    """Lattice of the line peaks of one projection.  ``pitch_ref`` (the frame's
    already-known pitch on this axis) wins over estimation; ``hint`` only
    disambiguates when the min-gap pitch is a multiple of the true one."""
    if proj.size == 0 or proj.max() <= 0:
        return None
    sm_max = float(np.convolve(proj.astype(np.float64), np.ones(3), 'same').max())
    pk = _peaks(proj, min_val=max(8.0, 0.25 * sm_max))
    if len(pk) < 2:
        return None
    pos = [p for p, _ in pk]
    if pitch_ref is not None:
        pitch0 = pitch_ref
    else:
        pitch0 = _lattice_search(pos, [s for _, s in pk], hint)
        if pitch0 is None:
            pitch0 = _min_gap_pitch(pos)
        if pitch0 is None:
            return None
    fit = _fit_lattice(pos, pitch0)
    if fit is None:
        return None
    o, pitch, inl = fit
    if len(inl) < 2:
        return None
    if pitch_ref is not None and abs(pitch - pitch_ref) / pitch_ref > 0.02:
        # the frame pitch is authoritative: only the phase is fitted
        pitch = pitch_ref
        o = _phase_from_pitch(inl, pitch)
    ws = [s for p, s in pk if any(abs(p - q) < 1e-6 for q in inl)]
    return _Axis(o, pitch, min(inl), max(inl), len(inl), float(np.mean(ws)) if ws else 0.0,
                 len(inl) / max(1, len(pk)))


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

MODES = (('strict', None), ('ridge', 14), ('ridge-soft', 7))


def detect_grid(img_bgr: np.ndarray, pitch_hint: float | None = None,
                min_cells: int = 4) -> GridResult:
    """Detect every grid panel in ``img_bgr``.

    ``pitch_hint`` is the px/slot expected from the screen/UI scale
    (:func:`expected_pitch`) when known; it is cross-checked, never trusted blindly.

    Three line models are tried in order of fidelity: the exact border colour range
    (pristine PNG), neutral grey ridges (JPEG / resampled captures) and soft ridges
    (blurred).  The pristine model is accepted outright when it explains a large,
    consistent lattice; otherwise the model with the best ``quality`` (cells x lattice
    consistency, so a bogus speckle mesh cannot win) is returned.
    """
    best = None
    for name, contrast in MODES:
        res = _detect(img_bgr, pitch_hint, min_cells, name, contrast)
        if best is None or res.quality > best.quality * 1.15:
            best = res
        if name == 'strict' and res.quality >= 60:
            break
        if name == 'ridge' and res.quality >= 60 and res.quality >= (best.quality if best else 0):
            break
    if best.mode != 'strict':
        best.warnings.append(f"exact line colour not usable; used '{best.mode}' line model "
                             "(compressed / resampled capture?)")
    return best


def _line_masks(img_bgr: np.ndarray, mode: str, contrast):
    if mode == 'strict':
        m = line_mask(img_bgr)
        return m, m
    return ridge_masks(img_bgr, contrast)


def _detect(img_bgr: np.ndarray, pitch_hint: float | None, min_cells: int,
            mode: str, contrast) -> GridResult:
    warnings: list[str] = []
    L = max(10, int(0.3 * pitch_hint)) if pitch_hint else 14
    mh, mv = _line_masks(img_bgr, mode, contrast)
    hm = _runs(mh, L, True)
    vm = _runs(mv, L, False)
    mesh_d = cv2.dilate((hm | vm).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mesh_d, connectivity=8)

    comps = []
    for i in range(1, n):
        _, _, cw, ch, _ = stats[i]
        if cw < MIN_PITCH or ch < MIN_PITCH:
            continue
        comps.append((int(cw) * int(ch), i))
    comps.sort(reverse=True)
    comps = comps[:12]

    # pass 1: every component's own lattice (pitch unknown)
    first = []
    for _, i in comps:
        cm = lab == i
        hmc, vmc = hm & cm, vm & cm
        ax = _analyse_axis(vmc.sum(axis=0), None, pitch_hint)
        ay = _analyse_axis(hmc.sum(axis=1), None, pitch_hint)
        if ax is None or ay is None or ax.ratio < 0.5 or ay.ratio < 0.5:
            continue
        if not (0.6 <= ax.pitch / ay.pitch <= 1.6):
            continue                                   # no real UI is that stretched
        first.append([hmc, vmc, ax, ay])
    if not first:
        return GridResult([], None, None, warnings, pitch_hint, mode, 0.0)

    # frame pitch = the pitch backed by the most line evidence (one UI scale per frame);
    # a bogus mesh (art speckle) is outvoted by the real panel's many consistent lines
    def weight(c):
        return c[2].n_lines * c[2].ratio + c[3].n_lines * c[3].ratio
    best_c = max(first, key=weight)
    pitch_x, pitch_y = best_c[2].pitch, best_c[3].pitch
    panels: list[Panel] = []
    for hmc, vmc, ax, ay in first:
        if (abs(ax.pitch - pitch_x) / pitch_x > 0.04 or abs(ay.pitch - pitch_y) / pitch_y > 0.04):
            continue
        p = _build_panel(hmc, vmc, ax, ay)
        if p is not None and p.n_cols * p.n_rows >= min_cells:
            p.strength = float(min(ax.ratio, ay.ratio))
            panels.append(p)
    # art speckle can form a small bogus mesh *inside* a real panel: drop those
    panels = [p for p in panels
              if not any(q is not p and q.n_cols * q.n_rows > p.n_cols * p.n_rows
                         and q.x0 - 2 <= p.x0 and p.x1 <= q.x1 + 2
                         and q.y0 - 2 <= p.y0 and p.y1 <= q.y1 + 2 for q in panels)]
    panels.sort(key=lambda p: (p.y0, p.x0))

    if pitch_hint and pitch_x and abs(pitch_x - pitch_hint) / pitch_hint > 0.08:
        warnings.append(f'detected pitch {pitch_x:.1f}px differs from screen-implied '
                        f'{pitch_hint:.1f}px (cropped capture or non-default UI scale)')
    if pitch_x and pitch_y and abs(pitch_x - pitch_y) / max(pitch_x, pitch_y) > 0.02:
        warnings.append(f'anisotropic cells {pitch_x:.2f}x{pitch_y:.2f}px (stretched capture)')
    quality = float(sum(p.n_cols * p.n_rows * p.strength for p in panels))
    return GridResult(panels, pitch_x, pitch_y, warnings, pitch_hint, mode, quality)


def _build_panel(hmc: np.ndarray, vmc: np.ndarray, ax: _Axis, ay: _Axis) -> Panel | None:
    px, py = ax.pitch, ay.pitch
    n_cols = int(round((ax.last - ax.first) / px))
    if n_cols < 1:
        return None
    ox = ax.first
    xs = _snap(vmc.sum(axis=0), [ox + k * px for k in range(n_cols + 1)], px)
    # Vertical extent: the rows where at least two lattice-aligned vertical lines are lit.
    # Panel borders / item edges produce that along the whole panel; header and footer
    # chrome essentially never has two verticals at the panel's column spacing, so UI
    # clutter above or below is not counted as extra rows (and one weak outer line, e.g.
    # over a hatched background after JPEG, cannot cut rows off).
    lit_count = np.zeros(vmc.shape[0], np.int32)
    for x in xs:
        lit_count += vmc[:, max(0, x - 1):x + 2].any(axis=1)
    both = lit_count >= 2
    rows_all = _longest_run(both, gap=8)
    if rows_all is None:
        return None
    ymin, ymax = rows_all
    # the top/bottom border itself is a horizontal line, which the vertical-run mask does not
    # contain: pull the extent out to a horizontal line directly adjacent to it
    hrows = np.where(hmc.any(axis=1))[0]
    near_top = hrows[(hrows >= ymin - 3) & (hrows < ymin)]
    near_bot = hrows[(hrows > ymax) & (hrows <= ymax + 3)]
    ymin = int(near_top.min()) if near_top.size else ymin
    ymax = int(near_bot.max()) if near_bot.size else ymax
    if ymax - ymin < 0.5 * py:
        return None

    # Row bookkeeping on the fractional lattice (``ay.origin`` + k*py).  The first
    # visible row is the one containing ``ymin``: if its top line is above ``ymin``
    # the row is a partially scrolled one (clip_top) and starts at the viewport edge.
    k_first = int(np.floor((ymin - ay.origin) / py + 0.03))
    start = ay.origin + k_first * py
    clip_top = (ymin - start) > 0.2 * py     # up to 20% hidden: detection jitter, not a scrolled row
    if clip_top and (start + py - ymin) < 0.2 * py:          # sliver: ignore it
        k_first += 1
        start += py
        clip_top = False
    k_last = int(np.floor((ymax - ay.origin) / py + 0.03))   # last lattice line <= ymax
    clip_bottom = (ymax - (ay.origin + k_last * py)) >= 0.2 * py
    n_rows = k_last - k_first + (1 if clip_bottom else 0)
    if n_rows < 1:
        return None
    oy = start

    ys = _snap(hmc.sum(axis=1), [oy + r * py for r in range(n_rows + 1)], py)
    if clip_top or ys[0] < ymin:
        ys[0] = ymin                      # never start above the visible line evidence
    if clip_bottom:
        ys[-1] = ymax
    return Panel(x0=int(xs[0]), y0=int(ys[0]), x1=int(xs[-1]) + 1, y1=int(ys[-1]) + 1,
                 ox=float(ox), oy=float(oy), pitch_x=px, pitch_y=py,
                 n_cols=n_cols, n_rows=n_rows,
                 xs=[int(v) for v in xs], ys=[int(v) for v in ys],
                 clip_top=clip_top, clip_bottom=clip_bottom,
                 strength=float(min(ax.strength, ay.strength)))


def _longest_run(mask: np.ndarray, gap: int = 8):
    """(first, last) index of the longest run of True, bridging gaps <= ``gap``."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    best = (idx[0], idx[0])
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            if prev - start > best[1] - best[0]:
                best = (start, prev)
            start = i
        prev = i
    if prev - start > best[1] - best[0]:
        best = (start, prev)
    return int(best[0]), int(best[1])


def _snap(proj: np.ndarray, pred: list[float], pitch: float) -> list[int]:
    """Snap predicted boundary positions to the real line pixel within +-2 px (the
    game draws lines on integer pixels, the lattice is fractional).

    At the panel's outer edges the item border line is accompanied by the lighter
    viewport frame one pixel further out; the *item* line (the one icons are drawn
    against) is the inner one, so the first boundary takes the largest strong
    position and the last boundary the smallest."""
    out = []
    n = len(proj)
    last = len(pred) - 1
    for i, p in enumerate(pred):
        c = int(round(p))
        lo, hi = max(0, c - 2), min(n, c + 3)
        if hi <= lo:
            out.append(c)
            continue
        seg = proj[lo:hi]
        m = float(seg.max())
        if m < 0.5 * pitch:
            out.append(c)
            continue
        strong = [lo + j for j in range(len(seg)) if seg[j] >= 0.6 * m]
        if i == 0:
            out.append(max(strong))
        elif i == last:
            out.append(min(strong))
        else:
            out.append(lo + int(np.argmax(seg)))
    return out
