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

def _edge_fraction(lm: np.ndarray, vertical: bool, pos: int, a: int, b: int, slack: int = 1,
                   width: int = 1, thin: bool = True) -> float:
    """Coverage of a boundary segment by the line mask: the best *straight* ``width``-px wide
    strip within ``slack`` px of the nominal position (sub-pixel lattice rounding, the 1-2 px smear
    of resampled captures), counting only positions where the line is thin.

    ``pos`` is the line's x (vertical boundary) or y, ``a..b`` the span along it.  A drawn border
    is one straight 1 px line, so (unlike "any mask pixel within +-1 px") art cannot fake it: a
    diagonal hatch pattern or a grey strap crossing the edge covers only part of any single
    column, and a grey *band* (cloth, a container lid) is rejected by ``thin``: a position whose
    mask run continues 3 px away on both sides is a wide patch, not a line."""
    H, W = lm.shape
    a = max(0, a)
    b = max(a + 1, b)
    K = slack + width + 3
    n = W if vertical else H
    if not (0 <= pos < n):
        return 0.0
    lo, hi = pos - K, pos + K + 1
    if vertical:
        strip = lm[a:b, max(0, lo):min(W, hi)]
    else:
        strip = lm[max(0, lo):min(H, hi), a:b].T
    if strip.size == 0:
        return 0.0
    if lo < 0:
        strip = np.pad(strip, ((0, 0), (-lo, 0)))
    if hi > n:
        strip = np.pad(strip, ((0, 0), (0, hi - n)))
    best = 0.0
    for s0 in range(-slack, slack + 1):
        c0 = K + s0
        hit = strip[:, c0]
        for k in range(1, width):
            hit = hit | strip[:, c0 + k]
        if thin:
            hit = hit & ~(strip[:, c0 - 3] & strip[:, c0 + width - 1 + 3])
        best = max(best, float(hit.mean()))
    return best


def _step_fraction(gray: np.ndarray, vertical: bool, pos: int, a: int, b: int, empty_before: bool,
                   rise: float = 22.0, hue_ok: np.ndarray | None = None) -> float:
    """Fraction of a boundary segment where a thin line (1-2 px) is at least ``rise`` grey levels
    brighter than the flat *empty* side (``empty_before``: the empty cell is the one before the
    line, i.e. above / left of it).  ``hue_ok`` (per-pixel: the border's grey-brown hue, see
    :func:`_hue_ok`) keeps a coloured art edge (green camo, a red strap) that merely starts at
    the boundary from passing as a line."""
    H, W = gray.shape
    a, b = max(0, a), max(a + 1, b)
    d = -1 if empty_before else 1
    best = None
    for off in (-1, 0, 1):
        p = pos + off
        q1, q2 = p + d * 3, p + d * 5
        lim = W if vertical else H
        if not (0 <= q1 < lim and 0 <= q2 < lim and 0 <= p < lim):
            continue
        if vertical:
            line, side = gray[a:b, p], 0.5 * (gray[a:b, q1] + gray[a:b, q2])
        else:
            line, side = gray[p, a:b], 0.5 * (gray[q1, a:b] + gray[q2, a:b])
        ok = (line - side) >= rise
        if hue_ok is not None:
            ok &= hue_ok[a:b, p] if vertical else hue_ok[p, a:b]
        fr = float(ok.mean()) if ok.size else 0.0
        best = fr if best is None else max(best, fr)
    return best or 0.0


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


def _hue_ok(img_bgr: np.ndarray) -> np.ndarray:
    """Pixels with the border's hue (B-G 0..14, G-R 4..18), whatever their brightness."""
    im = img_bgr.astype(np.int16)
    bg = im[..., 0] - im[..., 1]
    gr = im[..., 1] - im[..., 2]
    return (bg >= 0) & (bg <= 14) & (gr >= 4) & (gr <= 18)


def _tight_line_mask(img_bgr: np.ndarray) -> np.ndarray:
    """The border colour range of :func:`identify.grid.line_mask`, narrowed to the *hue* the game
    really draws: on the 12 labelled screenshots 99 % of true line pixels have B-G in 2..9 and
    G-R in 6..11 (nominal (84, 81, 73), a little brighter under the top gradient).  The wider box
    the grid finder uses also passes green camo, olive tints and warm browns, which is how a
    backpack's cloth edge became a 'line'."""
    return line_mask(img_bgr) & _hue_ok(img_bgr)


RIDGE_WIDTH = 2     # resampled / JPEG lines smear over 2 px; the exact-colour model needs exactly 1


@dataclass
class EdgeModel:
    """Which edges of a panel carry a drawn line: per-boundary line coverage, the threshold
    that splits "line" from "no line" and the mask model that produced them."""
    name: str
    thr: float
    j: float
    fv: dict          # (row, col) -> coverage of the vertical boundary left of cell (col, row)
    fh: dict          # (row, col) -> coverage of the horizontal boundary above cell (col, row)

    def vertical_line(self, r: int, c: int) -> bool:
        return self.fv[(r, c)] >= self.thr

    def horizontal_line(self, r: int, c: int) -> bool:
        return self.fh[(r, c)] >= self.thr


def edge_model(img_bgr: np.ndarray, panel: Panel, ridge: bool | None = None) -> EdgeModel:
    """Learn, per image, how to tell a drawn border line from art (see :func:`segment_panel`)."""
    nc, nr = panel.n_cols, panel.n_rows
    xs, ys = panel.xs, panel.ys

    def fractions(lm_h, lm_v, width=1):
        fv = {(r, c): _edge_fraction(lm_v, True, xs[c], ys[r] + 2, ys[r + 1] - 1, width=width)
              for r in range(nr) for c in range(1, nc)}
        fh = {(r, c): _edge_fraction(lm_h, False, ys[r], xs[c] + 2, xs[c + 1] - 1, width=width)
              for c in range(nc) for r in range(1, nr)}
        return fv, fh

    strict = _tight_line_mask(img_bgr)
    rh, rv = ridge_masks(img_bgr)
    cands = []
    lh, lv = ridge_masks(img_bgr, neutral_only=False)      # any hue: orange 'attention' frames
    models = (('strict', (strict, strict)), ('ridge', (rh, rv)), ('union', (strict | rh, strict | rv)),
              ('luma', (strict | lh, strict | lv)))
    for name, (mh, mv) in models:
        if ridge is not None and (name == 'ridge') != bool(ridge):
            continue
        fv, fh = fractions(mh, mv, 1 if name == 'strict' else RIDGE_WIDTH)
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
    j, thr, fv, fh, name = max(near, key=share)

    # The border of an item that touches an *empty* slot is alpha-blended over the dark
    # backdrop (stash1: (84,81,73) between items, ~(67,65,59) against an empty cell), so the
    # exact-colour model misses it and the item would be merged with the empties.  Art cannot
    # fake a line there (one side is flat dark), so on those edges a one-sided step test counts:
    # a thin line clearly brighter than the empty side along the whole edge.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hue_ok = _hue_ok(img_bgr)
    empty = {(c, r): cell_is_empty(img_bgr, panel.rect(c, r, 1, 1)) for r in range(nr) for c in range(nc)}

    def _step(vertical, pos, a, b, e0, e1):
        # one empty side: the line must stand out from it.  Both sides flat (two empty slots, or
        # the flat margin of a big icon next to an empty slot): it must stand out from both.
        if e0 and e1:
            return min(_step_fraction(gray, vertical, pos, a, b, True, hue_ok=hue_ok),
                       _step_fraction(gray, vertical, pos, a, b, False, hue_ok=hue_ok))
        return _step_fraction(gray, vertical, pos, a, b, e0, hue_ok=hue_ok)

    if any(empty.values()):
        fv = dict(fv)
        fh = dict(fh)
        for (r, c), f in list(fv.items()):
            e0, e1 = empty[(c - 1, r)], empty[(c, r)]
            if f < thr and (e0 or e1):
                fv[(r, c)] = _step(True, xs[c], ys[r] + 2, ys[r + 1] - 1, e0, e1)
        for (r, c), f in list(fh.items()):
            e0, e1 = empty[(c, r - 1)], empty[(c, r)]
            if f < thr and (e0 or e1):
                fh[(r, c)] = _step(False, ys[r], xs[c] + 2, xs[c + 1] - 1, e0, e1)
    return EdgeModel(name, float(thr), float(j), fv, fh)


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
    em = edge_model(img_bgr, panel, ridge)
    fv, fh, thr = em.fv, em.fh, em.thr

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
