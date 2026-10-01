"""
Item segmentation from the borders EFT draws.

Every item is framed by a 1 px line of the cell-border colour; lines that would
run *through* a multi-cell item are not drawn.  So, given the panel lattice from
:mod:`identify.grid`, an item footprint is simply a maximal block of cells
joined by edges **without** a line:

* a 2x1 stack of the same ammo type next to another 2x1 of the same type is
  correctly two items, because the shared edge carries a line;
* a gun that the legacy matcher shredded into 1x1 / 2x1 fragments is one block,
  because there is no line inside it;
* rotation changes nothing - the footprint is whatever shape the lines enclose.

Empty cells are single cells whose interior is flat dark colour.

The module also reports items cut off by the panel viewport (a scrolled stash's
first/last row): their visible rectangle is returned with ``clipped_*`` set, and
the matcher compares only the visible part.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid import Panel, line_mask, ridge_masks


@dataclass
class Item:
    """An item footprint inside a panel (cell units + pixel rect)."""
    col: int
    row: int
    w: int
    h: int
    rect: tuple            # (x, y, width, height) pixels, closing line included
    clipped_top: bool = False
    clipped_bottom: bool = False
    empty: bool = False
    suspect: bool = False  # component was not a clean rectangle (line detection doubt)
    panel: int = 0

    @property
    def cells(self) -> int:
        return self.w * self.h


# --------------------------------------------------------------------------
# edge tests
# --------------------------------------------------------------------------

def _edge_fraction(lm: np.ndarray, vertical: bool, pos: int, a: int, b: int, slack: int = 1) -> float:
    """Fraction of a boundary segment covered by the line mask.  ``pos`` is the line's x
    (vertical boundary) or y, ``a..b`` the span along it; ``slack`` px either side absorb
    sub-pixel lattice rounding and the 1-2 px smear of resampled captures."""
    H, W = lm.shape
    a = max(0, a)
    b = max(a + 1, b)
    lo, hi = pos - slack, pos + slack + 1
    if vertical:
        lo, hi = max(0, lo), min(W, hi)
        seg = lm[a:b, lo:hi].any(axis=1) if hi > lo else None
    else:
        lo, hi = max(0, lo), min(H, hi)
        seg = lm[lo:hi, a:b].any(axis=0) if hi > lo else None
    if seg is None or seg.size == 0:
        return 0.0
    return float(seg.mean())


def _otsu_1d(vals: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Otsu threshold of values in [0, 1] and its separability J = sigma_B^2 / sigma_T^2."""
    if vals.size < 4:
        return (lo + hi) / 2, 0.0
    best_t, best_j = (lo + hi) / 2, 0.0
    tot = vals.var()
    if tot <= 1e-9:
        return best_t, 0.0
    for t in np.linspace(0.3, 0.95, 66):
        a, b = vals[vals < t], vals[vals >= t]
        if a.size == 0 or b.size == 0:
            continue
        wa, wb = a.size / vals.size, b.size / vals.size
        j = wa * wb * (a.mean() - b.mean()) ** 2 / tot
        if j > best_j:
            best_j, best_t = j, float(t)
    return min(max(best_t, lo), hi), best_j


def cell_is_empty(img_bgr: np.ndarray, rect: tuple, inset: int = 4) -> bool:
    """Flat, dark interior => empty slot."""
    x, y, w, h = rect
    sub = img_bgr[max(0, y + inset):y + h - inset, max(0, x + inset):x + w - inset]
    if sub.size == 0:
        return False
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # real items measured on stash1: interior grey std >= 16.7 (even a flat black-tint tile has art);
    # an empty slot is a dark backdrop with at most a faint gradient / compression noise
    return bool(g.std() < 8.0 and g.mean() < 70)


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------

class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def segment_panel(img_bgr: np.ndarray, panel: Panel, panel_index: int = 0,
                  ridge: bool | None = None) -> list[Item]:
    """Footprints of every item (and every empty cell) in ``panel``.

    Edge presence is the fraction of a boundary segment covered by a line mask.  Two masks
    are available - the exact border colour range (pristine PNG) and neutral-grey ridges
    (survive JPEG / resampling) - and the threshold is **learned per image**: both masks
    are evaluated on every interior boundary, the distribution of the fractions is
    bimodal (a drawn line scores ~1, none scores low), the mask with the better
    two-class separation wins and Otsu's threshold splits the classes.  ``ridge`` forces
    a mask (True = ridge, False = strict)."""
    nc, nr = panel.n_cols, panel.n_rows
    xs, ys = panel.xs, panel.ys

    def fractions(lm_h, lm_v):
        fv = {(r, c): _edge_fraction(lm_v, True, xs[c], ys[r] + 2, ys[r + 1] - 1)
              for r in range(nr) for c in range(1, nc)}
        fh = {(r, c): _edge_fraction(lm_h, False, ys[r], xs[c] + 2, xs[c + 1] - 1)
              for c in range(nc) for r in range(1, nr)}
        return fv, fh

    strict = line_mask(img_bgr)
    rh, rv = ridge_masks(img_bgr)
    cands = []
    lh, lv = ridge_masks(img_bgr, neutral_only=False)      # any hue: orange 'attention' frames
    models = (('strict', (strict, strict)), ('ridge', (rh, rv)), ('union', (strict | rh, strict | rv)),
              ('luma', (strict | lh, strict | lv)))
    for name, (mh, mv) in models:
        if ridge is not None and (name == 'ridge') != bool(ridge):
            continue
        fv, fh = fractions(mh, mv)
        vals = np.array(list(fv.values()) + list(fh.values()), np.float64)
        t, j = _otsu_1d(vals, 0.3, 0.9)
        cands.append((j, t, fv, fh, name))
    # A model that "separates" the classes only because it finds almost no lines at all (JPEG
    # destroyed the thin line under the exact-colour test) is useless: in a real panel most
    # interior boundaries are drawn (stash1: 77 %), so require a plausible share of lines.
    def share(c):
        vals = list(c[2].values()) + list(c[3].values())
        return sum(1 for f in vals if f >= c[1]) / max(1, len(vals))
    plausible = [c for c in cands if share(c) >= 0.3] or [max(cands, key=share)]
    jmax = max(c[0] for c in plausible)
    # among the models that separate the two classes about equally well, take the most
    # inclusive one (more drawn lines recognised): a hue-agnostic ridge also sees the
    # orange 'attention' frames the neutral models lose after JPEG
    near = [c for c in plausible if c[0] >= jmax - 0.04]
    strict_c = next((c for c in plausible if c[4] == 'strict'), None)
    if strict_c is not None and strict_c[0] >= 0.97:         # pristine capture: the exact model is decisive
        near = [strict_c]
    j, thr, fv, fh, _name = max(near, key=share)

    dsu = _DSU(nc * nr)
    idx = lambda c, r: r * nc + c
    for (r, c), f in fv.items():                 # vertical boundary between col c-1 and c
        if f < thr:
            dsu.union(idx(c - 1, r), idx(c, r))
    for (r, c), f in fh.items():                 # horizontal boundary between row r-1 and r
        if f < thr:
            dsu.union(idx(c, r - 1), idx(c, r))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(nr):
        for c in range(nc):
            groups.setdefault(dsu.find(idx(c, r)), []).append((c, r))

    items: list[Item] = []
    for cells in groups.values():
        cs = [c for c, _ in cells]
        rs = [r for _, r in cells]
        c0, c1, r0, r1 = min(cs), max(cs), min(rs), max(rs)
        w, h = c1 - c0 + 1, r1 - r0 + 1
        if w * h == len(cells):
            blocks = [(c0, r0, w, h, False)]
        else:
            blocks = _split_rects(cells)
        for (c, r, bw, bh, sus) in blocks:
            rect = panel.rect(c, r, bw, bh)
            it = Item(col=c, row=r, w=bw, h=bh, rect=rect, panel=panel_index, suspect=sus,
                      clipped_top=panel.clip_top and r == 0,
                      clipped_bottom=panel.clip_bottom and r + bh == nr)
            it.empty = cell_is_empty(img_bgr, rect)
            items.append(it)
    items.sort(key=lambda i: (i.row, i.col))
    return items


def _split_rects(cells: list[tuple[int, int]]) -> list[tuple[int, int, int, int, bool]]:
    """Greedy decomposition of a non-rectangular cell set into rectangles
    (largest first); every piece is flagged ``suspect``."""
    left = set(cells)
    out = []
    while left:
        c0, r0 = min(left, key=lambda t: (t[1], t[0]))
        w = 1
        while (c0 + w, r0) in left:
            w += 1
        h = 1
        while all((c0 + k, r0 + h) in left for k in range(w)):
            h += 1
        for k in range(w):
            for j in range(h):
                left.discard((c0 + k, r0 + j))
        out.append((c0, r0, w, h, True))
    return out
