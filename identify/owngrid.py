"""
Cell groups of the player's own gear grids (tactical rig, pockets, backpack, pouch, special slots).

``detect_grid`` finds one regular lattice with the stash border colour.  The gear slots differ:

* a tactical rig is a row of separate *slots* (each 1 or 2 cells wide, usually 2 high) with ~9 px
  gaps between them, so no single lattice fits;
* pockets are four separate 1x1 slots with gaps;
* a backpack is a normal grid but its inner lines are only ~10 grey levels above the backdrop
  (the outer frame is bright), and its last row is cut by the viewport.

EFT draws every inventory cell at the SAME pitch, so the pitch measured on the stash of the same
frame (or implied by the frame height) constrains these grids.  Inside each own-gear region found
by :mod:`identify.scene` (from its slot header) this module

1. builds the drawn-border mask of the region: bright thin ridges (>= 25 grey levels above both
   sides) or the stash line colour,
2. keeps only long straight runs (>= half a cell) and groups them into connected frames,
3. snaps every frame to the pitch: columns = round(width / pitch), rows likewise (a missing
   bottom frame line with a fractional last row is the viewport clip), refining the interior
   lattice lines on the faint inner ridges,
4. returns one :class:`~identify.grid.Panel` per frame, so the ordinary segmentation / identification
   path scans it (an empty slot is a flat empty cell: the free-cell count).

Frames touching a floating window, or not snapping to the pitch, are dropped.
"""
from __future__ import annotations

import cv2
import numpy as np

from .grid import Panel

OWN_ROLES = ('own_rig', 'own_pockets', 'own_backpack', 'own_pouch', 'own_special')
STRONG = 25         # grey levels above both sides: a drawn frame line
WEAK = 6            # faint inner lattice line (backpack)
SNAP_TOL = 0.13     # of a pitch: how far a frame may be off an integer number of cells


def _ridges(gray: np.ndarray, level: int):
    """(vertical, horizontal) boolean masks of thin lines at least ``level`` above both sides."""
    H, W = gray.shape
    pad = np.pad(gray, 3, mode='edge')
    left, right = pad[3:3 + H, 0:W], pad[3:3 + H, 6:6 + W]
    up, down = pad[0:H, 3:3 + W], pad[6:6 + H, 3:3 + W]
    return (gray - np.maximum(left, right)) >= level, (gray - np.maximum(up, down)) >= level


def _frames(crop: np.ndarray, pitch: float, blank=()):
    """Connected frames of long drawn lines: [(x0, y0, x1, y1)] bboxes (crop coordinates, x1/y1 =
    the last line pixel) plus the strong masks."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sv, sh = _ridges(gray, STRONG)
    from .grid import line_mask
    lm = line_mask(crop)
    sv, sh = sv | lm, sh | lm
    for (bx, by, bw, bh) in blank:                         # floating windows hide what is below
        sv[max(0, by):max(0, by + bh), max(0, bx):max(0, bx + bw)] = False
        sh[max(0, by):max(0, by + bh), max(0, bx):max(0, bx + bw)] = False
    k = max(8, int(0.5 * pitch))
    vr = cv2.morphologyEx(sv.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    hr = cv2.morphologyEx(sh.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    both = vr | hr
    glue = max(3, int(0.05 * pitch))                       # smaller than the gap between slots
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(both, np.ones((glue, glue), np.uint8)), connectivity=8)
    boxes = []
    for i in range(1, n):
        ys, xs = np.nonzero((lab == i) & (both > 0))
        if xs.size:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes, sh, sv


def _peak(profile: np.ndarray, lo: int, hi: int) -> int:
    lo, hi = max(0, lo), min(len(profile) - 1, hi)
    return lo + int(np.argmax(profile[lo:hi + 1]))


def _run(line: np.ndarray, start: int, gap: int) -> int:
    """Length from ``start`` of the run of True in ``line`` that tolerates gaps of <= ``gap``."""
    end, miss = start, 0
    for i in range(start, len(line)):
        if line[i]:
            end, miss = i, 0
        else:
            miss += 1
            if miss > gap:
                break
    return end - start


def _snap_frame(box, sh, sv, weak_v, weak_h, pitch, top_clip_y=None):
    """Panel lattice (xs, ys, clip_top, clip_bottom) of one frame, or None when it is not a cell
    group.  The extent comes from the *runs* of the top and left frame lines (a long UI bar glued to
    a corner of the frame must not enlarge it)."""
    bx0, by0, bx1, by1 = box
    if bx1 - bx0 < 0.7 * pitch or by1 - by0 < 0.7 * pitch:
        return None
    gap = max(2, int(0.04 * pitch))        # less than the 9 px between rig slots
    xl = _peak(sv[by0:by1 + 1].sum(0), bx0 - 1, bx0 + 2)
    top_clipped = top_clip_y is not None
    yt = int(top_clip_y) if top_clipped else _peak(sh[:, bx0:bx1 + 1].sum(1), by0 - 1, by0 + 2)
    top = sh[max(0, yt - 1):yt + 2].any(0)
    left = sv[:, max(0, xl - 1):xl + 2].any(1)
    lx = _run(top, xl, gap) if not top_clipped else bx1 - xl
    ly = _run(left, yt, gap)
    nc = int(round(lx / pitch))
    if nc < 1 or abs(lx - nc * pitch) > SNAP_TOL * pitch * max(1.0, nc ** 0.5):
        return None
    if ly < 0.7 * pitch:
        return None
    nr = int(round(ly / pitch))
    clip = False
    if top_clipped:                                    # a window hides the top: align to the bottom
        nr = int(np.ceil(ly / pitch - 0.1))
        yb = int(yt + ly)
        xr = _peak(sv.sum(0), int(bx1) - 2, int(bx1) + 2)
        xs = [xl] + [int(round(xl + i * (xr - xl) / nc)) for i in range(1, nc)] + [xr]
        ys = [yt] + [int(round(yb - (nr - j) * pitch)) for j in range(1, nr)] + [yb]
        if nr < 2:
            ys = [yt, yb]
            nr = 1
        return xs, ys, True, False
    if abs(ly - nr * pitch) > SNAP_TOL * pitch * max(1.0, nr ** 0.5) or nr < 1:
        nr = int(np.ceil(ly / pitch - 0.1))
        clip = True                                    # the viewport cuts the last row
        if ly - (nr - 1) * pitch < 0.15 * pitch:
            nr -= 1
            clip = False
            ly = nr * pitch
    xr = _peak(sv.sum(0), int(xl + lx) - 2, int(xl + lx) + 2) if lx >= pitch else int(xl + lx)
    xs = [xl]
    for i in range(1, nc):
        e = int(round(xl + i * (xr - xl) / nc))
        c = _peak(weak_v, e - 2, e + 2)
        xs.append(c if weak_v[c] >= 0.3 * ly else e)
    xs.append(xr)
    yb = int(yt + ly)
    py_ = pitch if clip else ly / nr
    ys = [yt]
    for j in range(1, nr):
        e = int(round(yt + j * py_))
        c = _peak(weak_h, e - 2, e + 2)
        ys.append(c if weak_h[c] >= 0.3 * (xr - xl) else e)
    ys.append(yb)
    return xs, ys, False, clip


def region_panels(frame: np.ndarray, bbox, pitch: float, windows=()):
    """Panels of the cell groups inside ``bbox`` (x, y, w, h) of ``frame``: [Panel]."""
    H, W = frame.shape[:2]
    m = max(4, int(0.1 * pitch))
    x0, y0 = max(0, int(bbox[0]) - m), max(0, int(bbox[1]) - m)
    x1, y1 = min(W, int(bbox[0] + bbox[2]) + m), min(H, int(bbox[1] + bbox[3]) + m)
    if x1 - x0 < pitch or y1 - y0 < pitch:
        return []
    crop = frame[y0:y1, x0:x1]
    blank = [(int(wx - 2 - x0), int(wy - 2 - y0), int(ww + 5), int(wh_ + 5)) for (wx, wy, ww, wh_) in windows]
    boxes, sh, sv = _frames(crop, pitch, blank)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    wv, wh = _ridges(gray, WEAK)
    weak_v = (wv | sv).sum(0)
    weak_h = (wh | sh).sum(1)
    panels = []
    for box in boxes:
        gx0, gy0, gx1, gy1 = box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0
        top_clip_y = None
        hit = False
        for (wx, wy, ww, wh_) in windows:
            ix = min(gx1, wx + ww + 2) - max(gx0, wx - 2)
            iy = min(gy1, wy + wh_ + 2) - max(gy0, wy - 2)
            if ix > 3 and iy > 3:
                if gy0 < wy + wh_ < gy1 - 0.6 * pitch:    # the window only covers the frame's top rows
                    top_clip_y = max(top_clip_y or 0, wy + wh_ + 3 - y0)
                else:
                    hit = True
            elif ix > 0.5 * (gx1 - gx0) and 0 <= gy0 - (wy + wh_) <= 6:
                top_clip_y = gy0 - y0                 # starts right under a window: its top is hidden
        if hit:
            continue
        # per-frame profiles restricted to the frame so neighbouring frames do not leak in
        sub_wv = (wv | sv)[box[1]:box[3] + 1].sum(0)
        sub_wh = (wh | sh)[:, box[0]:box[2] + 1].sum(1)
        snapped = _snap_frame(box, sh, sv, sub_wv, sub_wh, pitch, top_clip_y)
        if snapped is None:
            continue
        xs, ys, clip_top, clip = snapped
        xs = [int(v) + x0 for v in xs]
        ys = [int(v) + y0 for v in ys]
        nc, nr = len(xs) - 1, len(ys) - 1
        panels.append(Panel(x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1, ox=float(xs[0]), oy=float(ys[0]),
                            pitch_x=(xs[-1] - xs[0]) / nc, pitch_y=pitch if (clip or clip_top) else (ys[-1] - ys[0]) / nr,
                            n_cols=nc, n_rows=nr, xs=xs, ys=ys, clip_top=clip_top, clip_bottom=clip, strength=1.0))
    panels.sort(key=lambda p: (p.y0, p.x0))
    return panels
