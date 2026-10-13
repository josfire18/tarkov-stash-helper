"""
The looted container's grid on the in-raid Character > GEAR screen (right column, header "LOOT").

Layout (EFT 1.2.0, scales with the frame height): the right column is a dark panel whose left and
top borders are bright lines (grey ~92 / ~60 on a ~20 backdrop).  A container's grid hangs from
that corner: its OUTER frame is bright (right and bottom lines >= ~55 grey), the inner cell lines
are only ~14 levels above the backdrop.  ``detect_grid`` finds this lattice only sometimes (the
borders are not in the stash colour), so the frame is traced here directly:

1. the left border column / top border row (long bright ridges) give the grid's origin,
2. the first lattice column right of the origin that carries a bright full-height line is the
   right frame, the first lattice row with a bright full-width line the bottom frame (a grid
   taller than the viewport has no bottom line: it is clipped),
3. while a container is being searched the cells that are not revealed yet are drawn with a faint
   diagonal hatch: :func:`hatched` finds them (they must not be read).

Searching with nothing revealed shows a big "UNSEARCHED" placeholder instead of a grid: no frame
is found then.
"""
from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass, field

from .grid import Panel

RIDGE = 20          # grey levels a frame line is above both sides (3 px away)
MIN_ABS = 50        # ... and its absolute brightness
BORDER_BGR = np.array([96, 93, 88], np.int16)    # the panel's left border line colour
FRAME_ABS = 86      # the right frame line (~93-97); an item's highlight border is ~79
HATCH_MIN = 0.9     # std of the high-passed per-diagonal means of a hatched cell


def _ridges(gray: np.ndarray):
    H, W = gray.shape
    pad = np.pad(gray, 3, mode='edge')
    left, right = pad[3:3 + H, 0:W], pad[3:3 + H, 6:6 + W]
    up, down = pad[0:H, 3:3 + W], pad[6:6 + H, 3:3 + W]
    v = ((gray - np.maximum(left, right)) >= RIDGE) & (gray >= MIN_ABS)
    h = ((gray - np.maximum(up, down)) >= RIDGE) & (gray >= MIN_ABS)
    return v, h


def hatched(gray_cell: np.ndarray) -> float:
    """Diagonal-stripe energy of one cell interior (an unrevealed cell, ~14 px stripe period):
    the std of the high-passed means along the two diagonal directions.  Flat cells score < 0.3,
    cells holding item art score high too, but those never reach here with a flat dark base:
    :func:`unrevealed` combines this with the cell's overall flatness."""
    g = gray_cell.astype(np.float32)
    h, w = g.shape
    if h < 12 or w < 12:
        return 0.0
    ii, jj = np.mgrid[0:h, 0:w]
    best = 0.0
    for key in (ii + jj, ii - jj + w):
        k = key.ravel()
        n = np.bincount(k)
        m = np.bincount(k, g.ravel()) / np.maximum(n, 1)
        m = m[n >= max(8, min(h, w) // 3)]
        if len(m) < 24:
            continue
        hp = m - cv2.blur(m.reshape(-1, 1), (1, 11)).ravel()
        best = max(best, float(np.std(hp[5:-5])))
    return best


def unrevealed(gray_cell: np.ndarray) -> bool:
    """True for a cell that is still hidden while the container is searched: a flat dark cell with
    the faint diagonal hatch (high-pass std ~3.3, total std < 6; an empty cell scores 1.4, a cell
    with item art >= 4.3 or a total std >= 8)."""
    g = gray_cell.astype(np.float32)
    if g.std() > 6.5:
        return False
    hp = float((g - cv2.GaussianBlur(g, (0, 0), 6)).std())
    return 2.0 <= hp <= 4.4 and hatched(g) >= 1.05


@dataclass
class LootInfo:
    panel: Panel | None = None              # the grid frame (every cell, revealed or not)
    hidden: set = field(default_factory=set)     # (col, row) of cells still hidden by the search
    placeholder: tuple | None = None        # (x, y, w, h) of the big "UNSEARCHED" stand-in (no grid yet)
    origin: tuple | None = None             # (x, y) of the column's top-left corner (the LOOT header sits above)
    header: bool = False                    # the "LOOT" header text was read above the corner (set by scene)
    name: str | None = None                 # the container's name at the top right ("DRAWER"), read by scene

    @property
    def found(self) -> bool:
        """A loot side exists: a grid, the UNSEARCHED stand-in, or just the LOOT header."""
        return self.panel is not None or self.placeholder is not None or self.header


def analyze_loot(frame: np.ndarray, pitch: float, y_bot: float | None = None) -> LootInfo:
    """The right-hand LOOT column: grid frame + unrevealed cells, or the UNSEARCHED placeholder."""
    info = LootInfo()
    p = find_loot_grid(frame, pitch, y_bot, info)
    if p is None:
        if info.origin is not None:
            info.placeholder = _placeholder(frame, pitch, info.origin)
        return info
    info.panel = p
    gray = cv2.cvtColor(frame[p.y0:p.y1, p.x0:p.x1], cv2.COLOR_BGR2GRAY)
    m = max(5, int(0.07 * pitch))
    for r in range(p.n_rows):
        for c in range(p.n_cols):
            x0, x1, y0, y1 = p.xs[c] - p.x0, p.xs[c + 1] - p.x0, p.ys[r] - p.y0, p.ys[r + 1] - p.y0
            cell = gray[y0 + m:y1 - m, x0 + m:x1 - m]
            if cell.size and unrevealed(cell):
                info.hidden.add((c, r))
    return info


def _placeholder(frame, pitch, origin):
    """Bbox of the hatched UNSEARCHED stand-in at the column's corner, or None."""
    H, W = frame.shape[:2]
    xl, yt = origin
    m = max(5, int(0.07 * pitch))
    g = cv2.cvtColor(frame[yt:yt + int(1.0 * pitch), xl:xl + int(1.0 * pitch)], cv2.COLOR_BGR2GRAY)
    if g.shape[0] < pitch * 0.9 or not unrevealed(g[m:-m, m:-m]):
        return None
    full = cv2.cvtColor(frame[yt:min(H, yt + int(4 * pitch)), xl:min(W, xl + int(12 * pitch))], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(full - cv2.GaussianBlur(full, (0, 0), 6))

    def run(profile, gap):
        end, miss = 0, 0
        for i, v in enumerate(profile):
            if v >= 1.9:
                end, miss = i, 0
            else:
                miss += 1
                if miss > gap:
                    break
        return end + 1

    w = run(cv2.blur(hp[int(0.12 * pitch):int(0.88 * pitch)].mean(0).reshape(1, -1), (15, 1)).ravel(), int(0.3 * pitch))
    h = run(cv2.blur(hp[:, int(0.12 * pitch):int(0.88 * pitch)].mean(1).reshape(-1, 1), (1, 15)).ravel(), int(0.3 * pitch))
    return (int(xl), int(yt), int(w), int(h))


def find_loot_grid(frame: np.ndarray, pitch: float, y_bot: float | None = None, info: "LootInfo | None" = None):
    """The looted container's grid frame as a :class:`Panel` (``n_cols`` x ``n_rows``), or None."""
    H, W = frame.shape[:2]
    xa, xb = int(0.64 * W), int(0.985 * W)
    ya, yb = int(0.06 * H), int(0.9 * H) if y_bot is None else int(y_bot)
    if xb - xa < 2 * pitch or yb - ya < 2 * pitch:
        return None
    gray = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.int16)
    sv, sh = _ridges(gray)
    P = float(pitch)
    # --- origin: the panel's left border column (a constant warm grey; the scene behind the panel may
    # be brighter than it, so no ridge test) and its first row
    y_lo, y_hi = int(0.065 * H) - ya, int(0.2 * H) - ya
    x_end = int(0.69 * W) - xa
    near = (np.abs(frame[ya:yb, xa:xa + x_end].astype(np.int16) - BORDER_BGR).max(axis=2) <= 9)
    colsum = near[y_lo:y_hi].sum(0)
    xl = int(np.argmax(colsum))
    if colsum[xl] < 0.9 * (y_hi - y_lo):
        return None
    ys_ = np.nonzero(near[:, xl])[0]
    ys_ = ys_[ys_ >= y_lo - int(0.03 * H)]
    if ys_.size == 0:
        return None
    yt = int(ys_[0]) - 1
    if info is not None:
        info.origin = (xa + xl, ya + yt)
    # --- right frame: first lattice column with a bright line over the first row
    n_max = int((xb - xa - xl) / P)
    r0, r1 = yt + int(0.15 * P), yt + int(0.85 * P)
    xr = None
    for k in range(1, n_max + 1):
        ex = xl + k * P
        lo, hi = int(ex) - 3, int(ex) + 4
        if hi >= sv.shape[1]:
            break
        cov = (sv & (gray >= FRAME_ABS))[r0:r1, lo:hi].mean(0)
        if cov.max() >= 0.8:
            xr, nc = lo + int(np.argmax(cov)), k
            break
    if xr is None:
        return None
    # --- bottom frame: first lattice row with a bright line over the full width
    j_max = int((yb - ya - yt) / P) + 1
    yB, nr, clip = None, 0, False
    for j in range(1, j_max + 1):
        ey = yt + j * P
        lo, hi = int(ey) - 3, int(ey) + 4
        if hi >= sh.shape[0]:
            break
        cov = sh[lo:hi, xl + 4:xr - 3].mean(1)
        if cov.max() >= 0.8:
            yB, nr = lo + int(np.argmax(cov)), j
            break
    if yB is None:                                       # taller than the viewport: the last row is clipped
        nr = max(1, int((yb - ya - yt) / P + 0.15))
        yB = yt + int(nr * P)
        clip = True
    xs = [int(round(xa + xl + i * (xr - xl) / nc)) for i in range(nc + 1)]
    ys = [int(round(ya + yt + j * (yB - yt) / nr)) for j in range(nr + 1)]
    if clip:
        ys = [int(round(ya + yt + j * P)) for j in range(nr + 1)]
    return Panel(x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1, ox=float(xs[0]), oy=float(ys[0]),
                 pitch_x=(xs[-1] - xs[0]) / nc, pitch_y=P if clip else (ys[-1] - ys[0]) / nr,
                 n_cols=nc, n_rows=nr, xs=xs, ys=ys, clip_bottom=clip, strength=1.0)
