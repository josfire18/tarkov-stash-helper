"""
Lattice of the stash fitted *inside the known stash region*, for renderings the colour based
:func:`identify.grid.detect_grid` cannot see.

On the trader screens (sell tab) the game dims every stash item the trader does not buy: the
border lines of those cells drop from (84, 81, 73) to ~(38, 37, 34), a contrast of only ~14 grey
levels over the backdrop, far outside the line-colour box.  The stash region itself is already
known (:mod:`identify.scene`: right of the stash toolbar, 10 columns), and EFT draws every cell
at one pitch, so the lattice is a 1-D phase problem per axis: find the offset at which the most
thin ridges (a pixel brighter than both neighbours 3 px away, by as little as ``MIN_RISE`` grey
levels) line up at multiples of the pitch.  Dim and normal cells share the same lattice, so one
fit covers both.
"""
from __future__ import annotations

import cv2
import numpy as np

from .grid import Panel

MIN_RISE = 5            # grey levels above both neighbours: a dimmed line is still ~13 above the backdrop
MIN_SUPPORT = 0.45      # a lattice line is real when this share of the region carries a ridge there
MIN_LINES_OK = 0.8      # share of the vertical lattice lines that must be supported


def _ridge_profiles(gray: np.ndarray):
    """(vertical-line profile over x, horizontal-line profile over y): per position the share of
    the region's rows / columns that carry a thin ridge there."""
    H, W = gray.shape
    pad = np.pad(gray, 3, mode='edge')
    left, right = pad[3:3 + H, 0:W], pad[3:3 + H, 6:6 + W]
    up, down = pad[0:H, 3:3 + W], pad[6:6 + H, 3:3 + W]
    rv = (gray - np.maximum(left, right)) >= MIN_RISE
    rh = (gray - np.maximum(up, down)) >= MIN_RISE
    return rv.mean(axis=0), rh.mean(axis=1)


def _grow(prof: np.ndarray) -> np.ndarray:
    """Max over +-1 px: the game draws lines on integer pixels, the lattice is fractional."""
    p = np.pad(prof, 1, mode='edge')
    return np.maximum(np.maximum(p[:-2], p[1:-1]), p[2:])


def _phase(prof: np.ndarray, lo: int, pitch: float, n_lines: int):
    """Best (offset, score) of a lattice ``off + k * pitch`` (k = 0..n_lines-1) whose first line lies
    in ``[lo, lo + pitch)``."""
    g = _grow(prof) * 0.3 + prof          # the exact column counts most (frame line 1 px beside the first line)
    n = len(g)
    best = (-1.0, lo)
    for off in range(lo, int(lo + pitch) + 1):
        s = 0.0
        for k in range(n_lines):
            c = int(round(off + k * pitch))
            if 0 <= c < n:
                s += g[c]
        if s > best[0]:
            best = (s, off)
    return best[1], best[0] / n_lines


def fit_stash_panel(frame: np.ndarray, bbox, pitch: float, n_cols: int = 10):
    """Panel of the stash lattice inside ``bbox`` (x, y, w, h) at ``pitch``, or ``None`` when the
    region shows no lattice.  Rows cut by the viewport are flagged ``clip_top`` / ``clip_bottom``."""
    H, W = frame.shape[:2]
    bx, by, bw, bh = (int(v) for v in bbox)
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(W, bx + bw), min(H, by + bh)
    if x1 - x0 < n_cols * pitch * 0.9 or y1 - y0 < 1.5 * pitch:
        return None
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pv, ph = _ridge_profiles(gray)
    # pitch: the frame's (the caller passes the measured one); +-0.6 % covers a fractional scale
    best = None
    for f in (1.0, 0.998, 1.002, 0.996, 1.004, 0.994, 1.006):
        p = pitch * f
        ox, sx = _phase(pv, 2, p, n_cols + 1)
        if best is None or sx > best[0] * 1.02:
            best = (sx, ox, p)
    sx, ox, p = best
    xs = [int(round(ox + k * p)) for k in range(n_cols + 1)]
    sup_v = float(np.mean([_grow(pv)[min(len(pv) - 1, max(0, c))] for c in xs]))
    nv_ok = float(np.mean([_grow(pv)[min(len(pv) - 1, max(0, c))] >= MIN_SUPPORT for c in xs]))
    if nv_ok < MIN_LINES_OK:
        return None
    # rows: lattice lines inside the viewport
    n_rows_max = int((y1 - y0) / p) + 2
    oy, sy = _phase(ph, 0, p, n_rows_max)
    ysl = [int(round(oy + j * p)) for j in range(n_rows_max) if oy + j * p <= (y1 - y0) - 3]
    if len(ysl) < 2 and not ysl:
        return None
    top_gap = ysl[0]                                  # px between the viewport top and the first line
    clip_top = top_gap > 0.2 * p
    ys = list(ysl)
    if clip_top:
        ys.insert(0, 0)
    last_gap = (y1 - y0) - ys[-1]
    clip_bottom = last_gap >= 0.2 * p
    if clip_bottom:
        ys.append((y1 - y0) - 2)
    if len(ys) < 2:
        return None
    ys = [v + y0 for v in ys]
    xs = [v + x0 for v in xs]
    nr = len(ys) - 1
    return Panel(x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1, ox=float(xs[0]), oy=float(ys[0]),
                 pitch_x=p, pitch_y=p, n_cols=n_cols, n_rows=nr, xs=xs, ys=ys,
                 clip_top=clip_top, clip_bottom=clip_bottom, strength=float(sup_v), dim=True)
