"""
Whole-stash capture: line up stash screenshots taken at different scroll positions.

The stash viewport shows ~14 of up to 68+ rows.  The player scrolls and scans again; this module
works out how far the stash moved between two captures so their items can be put on one
absolute row scale (``StashStitcher``), without stitching a giant image for identification.

Why picture correlation and not the grid lines
----------------------------------------------
Every row draws the same border lines, so the lattice alone is periodic: a shift by any whole
number of rows fits equally well.  The item art is what is unique.  ``vertical_shift`` scores
every shift by the zero-mean normalised correlation of the overlapping part of the two viewports
(all shifts at once with an FFT along y), then refuses the answer when a shift at least half a row
away scores almost as well (two screens of empty cells, a stash of identical ammo boxes).

The scroll bar is the fallback when nothing overlaps (scrolled more than a screen between scans):
``scroll_thumb`` reads the thumb's position and length, which give the first visible row and the
total row count to about half a row.  It is also what tells "row 1 of 68" apart from "row 1 of
what this session happened to see first".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_OVERLAP_ROWS = 1.0       # less shared viewport than this is not trusted
AMBIGUITY_MARGIN = 0.08      # best ZNCC must beat any shift >= half a row away by this much
MIN_SCORE = 0.6              # ...and be at least this good


@dataclass
class Shift:
    dy: int                  # content moved UP by dy px from ``a`` to ``b`` (scrolled down when > 0)
    score: float             # ZNCC of the overlap at dy
    runner_up: float         # best ZNCC at least half a row away
    overlap: int             # rows of pixels the two viewports share at dy


def _gray(frame: np.ndarray, box) -> np.ndarray:
    x, y, w, h = (int(v) for v in box)
    crop = frame[y:y + h, x:x + w]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop.astype(np.float32)


def _zncc_all_shifts(a: np.ndarray, b: np.ndarray, min_overlap: int):
    """ZNCC of a[y + d] vs b[y] over their overlap, for every d in (-H, H).  Returns (ds, scores,
    overlaps).  Sums over the overlap come from cumulative row sums; the cross term from an FFT
    correlation along y per column, summed over columns."""
    H, W = a.shape
    n = 1
    while n < 2 * H:
        n *= 2
    fa = np.fft.rfft(a, n=n, axis=0)
    fb = np.fft.rfft(b, n=n, axis=0)
    cross = np.fft.irfft(fa * np.conj(fb), n=n, axis=0).sum(axis=1)    # cross[d] = sum a[y+d] b[y]
    ra, rb = a.sum(axis=1), b.sum(axis=1)
    ra2, rb2 = (a * a).sum(axis=1), (b * b).sum(axis=1)
    ca = np.concatenate([[0.0], np.cumsum(ra)])
    cb = np.concatenate([[0.0], np.cumsum(rb)])
    ca2 = np.concatenate([[0.0], np.cumsum(ra2)])
    cb2 = np.concatenate([[0.0], np.cumsum(rb2)])
    ds = np.arange(-(H - min_overlap), H - min_overlap + 1)
    scores = np.full(ds.shape, -1.0)
    overlaps = H - np.abs(ds)
    for i, d in enumerate(ds):
        L = H - abs(d)
        a0, b0 = (d, 0) if d >= 0 else (0, -d)        # a rows a0..a0+L pair with b rows b0..b0+L
        npx = float(L * W)
        sa, sb = ca[a0 + L] - ca[a0], cb[b0 + L] - cb[b0]
        sa2, sb2 = ca2[a0 + L] - ca2[a0], cb2[b0 + L] - cb2[b0]
        sab = cross[d % n]
        va, vb = sa2 - sa * sa / npx, sb2 - sb * sb / npx
        if va <= 1e-6 * npx or vb <= 1e-6 * npx:
            continue
        scores[i] = (sab - sa * sb / npx) / np.sqrt(va * vb)
    return ds, scores, overlaps


def vertical_shift(a: np.ndarray, b: np.ndarray, box, pitch: float,
                   min_overlap_rows: float = MIN_OVERLAP_ROWS) -> Shift | None:
    """How far the stash content moved between frame ``a`` and frame ``b`` (same viewport ``box``
    = (x, y, w, h) on screen, same UI scale ``pitch``).  None when the overlap is too small or the
    match is ambiguous (see the module doc)."""
    ga, gb = _gray(a, box), _gray(b, box)
    if ga.shape != gb.shape or ga.shape[0] < 2 * pitch:
        return None
    s = max(1, int(pitch // 28))                      # ~28 px per row is plenty for the coarse pass
    sa, sb = (cv2.resize(g, (g.shape[1] // s, g.shape[0] // s), interpolation=cv2.INTER_AREA)
              for g in (ga, gb))
    ps = pitch / s
    min_ov = max(2, int(min_overlap_rows * ps))
    if sa.shape[0] <= min_ov + 1:
        return None
    ds, scores, overlaps = _zncc_all_shifts(sa, sb, min_ov)
    i = int(np.argmax(scores))
    best = float(scores[i])
    far = np.abs(ds - ds[i]) >= 0.5 * ps
    runner = float(scores[far].max()) if far.any() else -1.0
    if best < MIN_SCORE or best - runner < AMBIGUITY_MARGIN:
        return None
    # refine at full resolution within +-s px (mean absolute difference: the coarse ZNCC already
    # proved the match, this only picks the exact pixel)
    H = ga.shape[0]
    d0 = int(ds[i]) * s
    best_d, best_err = d0, None
    for d in range(d0 - s, d0 + s + 1):
        L = H - abs(d)
        if L < min_ov * s:
            continue
        pa = ga[d:d + L] if d >= 0 else ga[:L]
        pb = gb[:L] if d >= 0 else gb[-d:-d + L]
        err = float(np.abs(pa - pb).mean())
        if best_err is None or err < best_err:
            best_d, best_err = d, err
    return Shift(dy=int(best_d), score=best, runner_up=runner, overlap=int(H - abs(best_d)))


@dataclass
class Thumb:
    top: float               # thumb top as a fraction of the track
    length: float            # thumb length as a fraction of the track

    def first_row(self, total_rows: int) -> float:
        return self.top * total_rows

    def total_rows(self, visible_rows: float) -> float:
        return visible_rows / max(self.length, 1e-6)


def scroll_thumb(frame: np.ndarray, box, pitch: float) -> Thumb | None:
    """The stash scroll bar's thumb: a light, grey, few-px-wide bar just right of the viewport
    ``box``, its track spanning the viewport height.  None when there is no such bar (a stash that
    fits the screen has none)."""
    x, y, w, h = (int(v) for v in box)
    x0, x1 = x + w, min(frame.shape[1], x + w + max(4, int(0.25 * pitch)))
    if x1 - x0 < 2:
        return None
    band = frame[y:y + h, x0:x1].astype(np.int16)
    v = band.max(axis=2)
    sat = v - band.min(axis=2)
    lit = (v >= 110) & (sat <= 40)
    col = int(np.argmax(lit.sum(axis=0)))
    on = lit[:, max(0, col - 1):col + 2].any(axis=1)
    idx = np.where(on)[0]
    if idx.size < 0.05 * h:
        return None
    # longest run, bridging 2 px gaps (JPEG / dithering)
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 3:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    t0, t1 = max(runs, key=lambda r: r[1] - r[0])
    if t1 - t0 < 0.05 * h or t1 - t0 > 0.98 * h:
        return None
    return Thumb(top=t0 / h, length=(t1 - t0 + 1) / h)


# --------------------------------------------------------------------------
# a capture session: many scans of one stash -> one item list
# --------------------------------------------------------------------------

ROW_SNAP = 0.25              # a cell boundary within this fraction of a row counts as on the lattice


@dataclass
class Piece:
    view: np.ndarray         # the stash viewport (BGR crop)
    scroll: int              # content y of the viewport's top edge, in session pixels
    thumb: Thumb | None
    exact: bool              # placed by picture overlap (False: by the scroll bar only)


@dataclass
class AddResult:
    status: str              # 'started' | 'added' | 'unchanged' | 'unplaced' | 'restarted'
    message: str
    rows: tuple = ()         # absolute rows this scan covered (first, last), 1-based for display


class StashStitcher:
    """Items of one stash collected over several scans at different scroll positions.

    ``add(frame, box, pitch, records)`` takes a scan: the full frame, the stash viewport ``box``
    (x, y, w, h), the UI pitch and the scan's stash records (the dicts of
    ``identify.pipeline.Detection.to_record``: ``col``, ``W``, ``px``/``py``/``pw``/``ph`` in frame
    pixels, ``clipped``).  It places the scan against an earlier one (``vertical_shift``; the
    scroll bar when nothing overlaps) and merges its items:

    * the scan is the truth for every row it shows in full: older items lying inside those rows
      are dropped (sold or moved since), and its fully visible items replace whatever they overlap;
    * an item the viewport cuts off is only kept when no other scan saw that spot.

    ``items()`` returns the merged records with ``stash_row`` (0-based, from the top of the stash
    when a scan of the top was seen) and ``px``/``py`` moved onto ``image()``, the stitched
    picture of everything captured.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.box = None
        self.pitch = None
        self.phase = None            # content y of a lattice line, mod pitch
        self.pieces: list[Piece] = []
        self._items: list[dict] = []   # records with session coordinates: _r0/_r1 (rows), _c0/_c1, _top (content px)
        self.top_scroll = None       # session scroll of the stash's first row, once a scan at the top was seen
        self.total_rows = None       # from the scroll bar, rough

    # -- placement ---------------------------------------------------------------------------
    def _same_layout(self, box, pitch) -> bool:
        if self.box is None:
            return False
        return (all(abs(int(a) - int(b)) <= 3 for a, b in zip(self.box, box))
                and abs(self.pitch - pitch) <= 0.03 * self.pitch)

    def _place(self, view, thumb):
        """(scroll, exact) of a new scan, or None."""
        box = (0, 0, view.shape[1], view.shape[0])
        # most recent first: the player usually scrolls on from the last scan
        for p in reversed(self.pieces):
            sh = vertical_shift(p.view, view, box, self.pitch)
            if sh is not None:
                return p.scroll + sh.dy, p.exact
        if thumb is not None:
            ref = next((p for p in reversed(self.pieces) if p.thumb is not None), None)
            if ref is not None:
                total = thumb.total_rows(self.box[3] / self.pitch)
                rows = round((thumb.top - ref.thumb.top) * total)
                return ref.scroll + int(round(rows * self.pitch)), False
        return None

    # -- rows ----------------------------------------------------------------------------------
    def _row_span(self, top: float, height: float) -> tuple[int, int]:
        """Rows [r0, r1) a vertical pixel span covers (content px)."""
        r0 = int(np.floor((top - self.phase) / self.pitch + ROW_SNAP))
        r1 = int(np.ceil((top + height - self.phase) / self.pitch - ROW_SNAP))
        return r0, max(r1, r0 + 1)

    def _full_rows(self, scroll: int) -> tuple[int, int]:
        """Rows [r0, r1) a viewport at ``scroll`` shows from top to bottom line."""
        h = self.box[3]
        r0 = int(np.ceil((scroll - self.phase) / self.pitch - 0.05))
        r1 = int(np.floor((scroll + h - self.phase) / self.pitch + 0.05))
        return r0, r1

    def _phase_from(self, records, scroll):
        tops = [(r['py'] - self.box[1] + scroll) % self.pitch for r in records if not r.get('clipped')]
        if not tops:
            return None
        # circular median: rotate so the values do not straddle the wrap-around
        ref = tops[0]
        rel = sorted(((t - ref + self.pitch / 2) % self.pitch) - self.pitch / 2 for t in tops)
        return (ref + rel[len(rel) // 2]) % self.pitch

    # -- public ----------------------------------------------------------------------------------
    def add(self, frame: np.ndarray, box, pitch: float, records: list) -> AddResult:
        box = tuple(int(v) for v in box)
        x, y, w, h = box
        view = frame[y:y + h, x:x + w].copy()
        thumb = scroll_thumb(frame, box, pitch)
        restarted = False
        if self.pieces and not self._same_layout(box, pitch):
            self.reset()
            restarted = True
        if not self.pieces:
            self.box, self.pitch = box, float(pitch)
            scroll, exact = 0, True
        else:
            placed = self._place(view, thumb)
            if placed is None:
                return AddResult('unplaced', "Couldn't tell where this part of the stash goes. Scroll back "
                                             "until some of the last scan is on screen again, then scan.")
            scroll, exact = placed
        if self.phase is None:
            self.phase = self._phase_from(records, scroll)
            if self.phase is None:                  # no whole item on screen: assume the viewport top is a line
                self.phase = float(scroll % self.pitch)
        if thumb is not None:
            self.total_rows = int(round(thumb.total_rows(h / self.pitch)))
            if thumb.top * (self.total_rows or 1) < 0.3:          # scrolled to the very top
                self.top_scroll = self.phase + round((scroll - self.phase) / self.pitch) * self.pitch
        elif not self.pieces:
            self.top_scroll = None

        unchanged = any(p.scroll == scroll for p in self.pieces)
        self.pieces = [p for p in self.pieces if p.scroll != scroll]
        self.pieces.append(Piece(view=view, scroll=scroll, thumb=thumb, exact=exact))
        self._merge(records, scroll)

        r0, r1 = self._full_rows(scroll)
        rows = (self._display_row(r0), self._display_row(r1 - 1))
        if restarted:
            return AddResult('restarted', 'The stash looks different (another screen size or layout), so '
                                          'the full-stash capture started over.', rows)
        if unchanged:
            return AddResult('unchanged', 'Updated the rows already captured here.', rows)
        if len(self.pieces) == 1:
            return AddResult('started', 'Full-stash capture started. Scroll down and scan again to add more.', rows)
        msg = 'Added to the full-stash capture.'
        if not exact:
            msg += ' Placed by the scroll bar only, so it may be off by a row.'
        return AddResult('added', msg, rows)

    def _merge(self, records, scroll):
        fr0, fr1 = self._full_rows(scroll)
        new_full, new_cut = [], []
        for rec in records:
            top = rec['py'] - self.box[1] + scroll
            r0, r1 = self._row_span(top, rec['ph'])
            item = dict(rec, _top=top, _r0=r0, _r1=r1, _c0=int(rec['col']),
                        _c1=int(rec['col']) + max(1, int(rec.get('W') or 1)), _scroll=scroll)
            (new_cut if rec.get('clipped') else new_full).append(item)

        def overlaps(a, b):
            return a['_r0'] < b['_r1'] and b['_r0'] < a['_r1'] and a['_c0'] < b['_c1'] and b['_c0'] < a['_c1']

        kept = []
        for old in self._items:
            if fr0 <= old['_r0'] and old['_r1'] <= fr1:          # this scan shows that spot in full
                continue
            if any(overlaps(old, n) for n in new_full):
                continue
            kept.append(old)
        kept.extend(new_full)
        for n in new_cut:
            if not any(overlaps(n, k) for k in kept):
                kept.append(n)
        self._items = kept

    def _base_row(self) -> int:
        """Session row shown as row 1: the stash's first row once a scan of the top was seen,
        otherwise the first row any scan showed in full."""
        if self.top_scroll is not None:
            return int(round((self.top_scroll - self.phase) / self.pitch))
        return min(self._full_rows(p.scroll)[0] for p in self.pieces)

    def _display_row(self, r: int) -> int:
        return r - self._base_row() + 1

    def coverage(self) -> dict:
        """Rows captured so far: ``ranges`` [(first, last)] 1-based, ``total`` (scroll bar estimate
        or None), ``anchored`` (row numbers count from the top of the stash)."""
        spans = sorted(self._full_rows(p.scroll) for p in self.pieces)
        merged = []
        for r0, r1 in spans:
            if merged and r0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], r1)
            else:
                merged.append([r0, r1])
        return {'ranges': [(self._display_row(a), self._display_row(b - 1)) for a, b in merged],
                'total': self.total_rows, 'anchored': self.top_scroll is not None,
                'scans': len(self.pieces)}

    def _span(self):
        lo = min(p.scroll for p in self.pieces)
        hi = max(p.scroll + self.box[3] for p in self.pieces)
        return lo, hi

    def image(self) -> np.ndarray | None:
        """Every captured viewport pasted at its scroll position (latest on top)."""
        if not self.pieces:
            return None
        lo, hi = self._span()
        canvas = np.zeros((hi - lo, self.box[2], 3), np.uint8)
        for p in self.pieces:
            canvas[p.scroll - lo:p.scroll - lo + p.view.shape[0]] = p.view
        return canvas

    def items(self) -> list:
        """The merged records, on ``image()`` coordinates, top to bottom."""
        if not self.pieces:
            return []
        lo, _ = self._span()
        base = self._base_row()
        out = []
        for it in sorted(self._items, key=lambda r: (r['_r0'], r['_c0'])):
            rec = {k: v for k, v in it.items() if not k.startswith('_')}
            rec['px'] = int(it['px']) - self.box[0]
            rec['py'] = int(round(it['_top'])) - lo
            rec['row'] = rec['stash_row'] = it['_r0'] - base
            rec['panel'] = 0
            out.append(rec)
        return out
