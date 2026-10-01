"""
Stage 3 support: reading the item's printed short name (and stack count) with
Tesseract, and matching the text against catalog names.

* The game draws the short name right-aligned in the top band of the footprint
  (and the caliber at the bottom-left of weapons / the count at the bottom-right).
* Process start-up dominates Tesseract's cost (~70 ms per call on Windows), so all
  strips of a scan are written to a temp dir and handed to Tesseract as an *image
  list* (one process, one output page per image, split on form feed), in a few
  parallel chunks.  ~95 labels take ~0.3 s instead of ~7 s.
* OCR text is only ever fuzzy evidence.  :func:`fuzzy_score` aligns the reference
  name into the OCR string with glyph-confusion-aware costs (slashed zero, 5/S, J/3 ...),
  ignores junk around the label and accepts truncated labels ("Perfotora",
  "AK-74 gas"), because the game cuts long names to fit the footprint.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .config import SLOT
from .masks import BOTTOM_PX, LABEL_PX

try:                                    # optional at import time
    import pytesseract
except Exception:                       # pragma: no cover
    pytesseract = None

try:
    from rapidfuzz import fuzz, process
except Exception:                       # pragma: no cover
    fuzz = None
    process = None

UP = 5                                  # upscale factor for 63 px/slot glyphs (~9 px caps)
MAX_LABEL_PX = 96                       # label text is right-aligned; never wider than this at 63 px/slot


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------

_avail: bool | None = None


def set_tesseract_cmd(cmd: str | None) -> None:
    global _avail
    if cmd and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = cmd
        _avail = None


def tesseract_available() -> bool:
    global _avail
    if _avail is None:
        try:
            pytesseract.get_tesseract_version()
            _avail = True
        except Exception:
            _avail = False
    return _avail


# --------------------------------------------------------------------------
# strip extraction
# --------------------------------------------------------------------------

def label_strip(img_bgr: np.ndarray, rect: tuple, pitch_x: float, pitch_y: float) -> np.ndarray | None:
    """Grayscale, upscaled, inverted label strip of a footprint (None if off-image)."""
    x, y, w, h = rect
    lh = max(6, int(round(LABEL_PX * pitch_y / SLOT)))
    maxw = int(round(MAX_LABEL_PX * pitch_x / SLOT))
    x0 = max(0, x + max(1, w - 1 - maxw))
    x1 = min(img_bgr.shape[1], x + w - 1)
    y0, y1 = max(0, y + 1), min(img_bgr.shape[0], y + 1 + lh)
    if x1 - x0 < 6 or y1 - y0 < 5:
        return None
    g = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    # normalise each axis to ~5x of the 63 px design size (undoes a stretched 4:3 -> 16:9 capture)
    g = cv2.resize(g, None, fx=UP * SLOT / max(pitch_x, 1.0), fy=UP * SLOT / max(pitch_y, 1.0),
                   interpolation=cv2.INTER_CUBIC)
    return 255 - g


def bottom_strip(img_bgr: np.ndarray, rect: tuple, pitch_x: float, pitch_y: float,
                 left: bool = False) -> np.ndarray | None:
    """Bottom band: the right part (stack count) or left part (caliber text)."""
    x, y, w, h = rect
    bh = max(6, int(round(BOTTOM_PX * pitch_y / SLOT)))
    bw = int(round(MAX_LABEL_PX * pitch_x / SLOT))
    if left:
        x0, x1 = x + 1, min(x + w - 1, x + 1 + bw)
    else:
        x0, x1 = max(x + 1, x + w - 1 - int(round(44 * pitch_x / SLOT))), x + w - 1
    y1 = min(img_bgr.shape[0], y + h - 1)
    y0 = max(0, y1 - bh)
    x0, x1 = max(0, x0), min(img_bgr.shape[1], x1)
    if x1 - x0 < 6 or y1 - y0 < 5:
        return None
    g = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    f = UP * SLOT / max(pitch_y, 1.0)
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return 255 - g


def variants_of(strip: np.ndarray) -> list[np.ndarray]:
    """Two renderings of a label strip: smooth inverted grey (best on clean
    backgrounds) and Otsu-binarised (best when art shows through the band).  Reading
    both and scoring against the *candidate set* is more accurate than choosing one
    up front (plain 74 / Otsu 80 / either-best 91 mean fuzzy score on 21 hand-checked
    labels)."""
    b = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    out = [strip, b]
    iso = isolate_label(strip)
    if iso is not None:
        out.extend(iso)
    return out


THICK_ART = 0.14          # a bright blob whose thickest point is wider than this x the strip height (as an
                          # inscribed radius) is item art, not a stroke (letters measure 0.03-0.10)


def classify_components(stats, h: int, thick=None) -> list:
    """Label each bright connected component of a strip: 'letter', 'piece' or 'art'.

    The label font's strokes are thin: a component only counts as art when it is *thick*
    (``thick`` = the radius of its largest inscribed circle; an ``I``, a ``-``, a fat ``g`` or
    half of a split ``T`` are all thin).  Art also runs off the band's bottom or is taller than
    a glyph.  Letter-sized thin or hollow components are
    letters; what is left (dots, hyphens, bars) are pieces that only count next to letters."""
    kinds = ['art']
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if y + bh >= h - 1 or bh > 0.9 * h or (thick is not None and thick[i] > THICK_ART * h):
            kinds.append('art')
        elif bh >= 0.3 * h:
            kinds.append('letter')
        elif max(bw, bh) <= 0.9 * h and area >= 6:
            kinds.append('piece')
        else:
            kinds.append('art')
    return kinds


def text_run(stats, kinds, h: int, wd: int) -> list | None:
    """Indices of the components that form the label: the run of letters that starts at the
    right edge and continues leftwards while the next letter is within a word gap, sits on
    the same line, and (for pieces) lies inside the letters' vertical range.  Item art and
    dithering to the left of the label never join the run."""
    idx = [i for i in range(1, len(stats)) if kinds[i] != 'art']
    letters = [i for i in idx if kinds[i] == 'letter']
    if not letters:
        return None
    gap_max = max(6, int(round(h * 0.55)))
    anchor = max(letters, key=lambda i: stats[i][0] + stats[i][2])
    if stats[anchor][0] + stats[anchor][2] < wd * 0.6:          # labels start at the right edge
        return None
    run = [anchor]
    left = stats[anchor][0]
    y0, y1 = stats[anchor][1], stats[anchor][1] + stats[anchor][3]
    tol = 0.1 * h
    for i in sorted(idx, key=lambda i: -(stats[i][0] + stats[i][2])):
        if i == anchor:
            continue
        x, y, bw, bh, _ = stats[i]
        if x + bw < left - gap_max:
            break                                         # sorted by right edge: nothing further qualifies
        if kinds[i] == 'letter':
            ov = min(y1, y + bh) - max(y0, y)
            if ov < 0.6 * min(bh, y1 - y0):
                continue
            y0, y1 = min(y0, y), max(y1, y + bh)
        elif y < y0 - tol or y + bh > y1 + tol:
            continue
        run.append(i)
        left = min(left, x)
    return run


def isolate_label(strip: np.ndarray) -> list[np.ndarray] | None:
    """The printed short name alone, without the item art around it.

    Labels are thin bright strokes with a dark outline, right-aligned against
    the cell edge; item art in the band (a silver DVD drive, a white bottle) is
    thick.  Keep the bright components that form the label run (see
    :func:`text_run`) and return that crop as grey and as clean black-on-white
    text.  None if no text run is found.
    """
    g = 255 - strip                                   # back to original brightness
    h, wd = g.shape
    # label white is ~150-200 after scaling, so the cut comes from the strip itself
    thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bright = (g > max(110.0, thr)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    # a component survives erosion by THICK_ART * h only if a disc that wide fits inside it
    r = max(2, int(round(THICK_ART * h)))
    core = cv2.erode(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    thick = np.zeros(n)
    thick[np.unique(lab[core > 0])] = h
    run = text_run(st, classify_components(st, h, thick), h, wd)
    if not run:
        return None
    text = np.isin(lab, run).astype(np.uint8)
    x0 = min(int(st[i][0]) for i in run)
    x1 = max(int(st[i][0] + st[i][2]) for i in run) - 1
    pad = max(4, h // 10)
    a, z = max(0, x0 - pad), min(wd, x1 + pad + 1)
    if z - a < h * 0.4:
        return None
    crop = strip[:, a:z]
    clean = np.full(crop.shape, 255, np.uint8)
    clean[cv2.dilate(text[:, a:z], np.ones((2, 2), np.uint8)) > 0] = 0
    return [crop, clean]


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _clean(txt: str) -> str:
    txt = txt.replace('\n', ' ').strip()
    while txt and not (txt[0].isalnum() or txt[0] == '#'):
        txt = txt[1:]
    return re.sub(r'\s+', ' ', txt).strip()


def _read_one(strip: np.ndarray | None, psm: int = 7, whitelist: str | None = None) -> str:
    if strip is None:
        return ''
    pad = cv2.copyMakeBorder(strip, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    cfg = f'--psm {psm}'
    if whitelist:
        cfg += f' -c tessedit_char_whitelist={whitelist}'
    try:
        return _clean(pytesseract.image_to_string(pad, config=cfg))
    except Exception:
        return ''


def read_strips(strips: list, workers: int = 4, whitelist: str | None = None,
                psm: int = 7) -> list[str]:
    """Read many strips in a few ``tesseract`` processes (see module docstring)."""
    out = [''] * len(strips)
    idx = [i for i, s in enumerate(strips) if s is not None]
    if not idx or not tesseract_available():
        return out
    import os
    import subprocess
    import tempfile
    cmd = pytesseract.pytesseract.tesseract_cmd
    cfg = ['--psm', str(psm)]
    if whitelist:
        cfg += ['-c', f'tessedit_char_whitelist={whitelist}']
    chunks = [idx[i::workers] for i in range(workers) if idx[i::workers]]
    with tempfile.TemporaryDirectory(prefix='tsh_ocr_') as d:
        jobs = []
        for ci, ch in enumerate(chunks):
            paths = []
            for i in ch:
                pad = cv2.copyMakeBorder(strips[i], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
                f = os.path.join(d, f'{ci}_{i}.png')
                cv2.imwrite(f, pad, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                paths.append(f)
            lst = os.path.join(d, f'list{ci}.txt')
            with open(lst, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(paths))
            jobs.append((ch, lst))

        def run(job):
            ch, lst = job
            try:
                r = subprocess.run([cmd, lst, 'stdout'] + cfg, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace',
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                pages = r.stdout.split('\f')
                if len(pages) >= len(ch):
                    return ch, [_clean(p) for p in pages[:len(ch)]]
            except Exception:
                pass
            return ch, [_read_one(strips[i], psm, whitelist) for i in ch]   # per-strip fallback

        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            for ch, texts in ex.map(run, jobs):
                for i, t in zip(ch, texts):
                    out[i] = t
    return out


def parse_count(text: str) -> int | None:
    """Stack count from bottom-right text ('80', '29', '296/400' -> 296)."""
    m = re.search(r'\d+', text.replace('O', '0'))
    return int(m.group()) if m else None


# --------------------------------------------------------------------------
# fuzzy matching
# --------------------------------------------------------------------------

# Glyph pairs Tesseract confuses on EFT's pixel font (slashed zero, 5/S, J/3, ...).
# Substituting within a pair costs 0.4 instead of 1 in the alignment below.
_SIM = {frozenset(p) for p in [
    '08', '0o', '0e', '0a', '0@', '0q', '0d', '0c', '8a', '8e', '83', '8s', '8b', '8@', '5s', '9s', '93',
    '9g', '9q', '3j', '3s', '1l', '1i', '1j', '17', '2z', '6g', '6b', '6c', 'jl', 'ji', '4a', 'hn', 'mn',
    'rn', 'uv', 'vy', 'rp', 'il', 'cg', 'ce', 'ao', 'oe', 'ft', 'tf', 'ky', 'wv', 'u0']}
_FOLD = str.maketrans({'o': '0', 'q': '0', 'd': '0', 'e': '0', 'a': '8', 'b': '8', 's': '5', 'z': '2',
                       'l': '1', 'i': '1', 'j': '3', 'g': '9', 'c': '0', 'h': 'n', 'r': 'n', 'm': 'n',
                       'u': 'v', 'y': 'v', 'w': 'v', 't': '7', 'f': '7', 'p': 'n', 'k': 'n'})


def canon(text: str) -> str:
    """Lowercase alphanumerics with other characters collapsed to single spaces."""
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def canon_nospace(text: str) -> str:
    return canon(text).replace(' ', '')


def fold(text: str) -> str:
    """Aggressive shape folding (used only to *prefilter* candidates quickly)."""
    return canon_nospace(text).translate(_FOLD)


def _align(o: str, r: str, trunc_ok: bool) -> float:
    """Cost of aligning reference ``r`` into OCR text ``o``: semi-global (junk before /
    after the label is free), glyph-confusion aware, optionally allowing the label to
    be a *truncated* prefix of ``r``."""
    m, n = len(r), len(o)
    if m == 0 or n == 0:
        return float(m)
    prev = [0.0] * (n + 1)            # row 0: free text prefix
    rows = [prev]
    for i in range(1, m + 1):
        cur = [float(i)] + [0.0] * n
        ri = r[i - 1]
        for j in range(1, n + 1):
            oj = o[j - 1]
            sub = 0.0 if ri == oj else (0.4 if frozenset((ri, oj)) in _SIM else 1.0)
            cur[j] = min(prev[j - 1] + sub, prev[j] + 1.0, cur[j - 1] + 1.0)
        rows.append(cur)
        prev = cur
    full = min(prev)                  # free text suffix
    best = full
    if trunc_ok:
        for i in range(max(3, int(0.6 * n)), m):
            # the label is a cut-off prefix of the name: text must be fully consumed
            c = rows[i][n] + 0.12 * (m - i)
            best = min(best, c)
    return best


def label_capacity(width_slots: int) -> int:
    """Characters of the game's label font that fit in a footprint ``width_slots`` wide
    (~8 at one slot, ~8.5 more per extra slot).  Longer names are cut to fit."""
    return int(8 + 8.5 * (max(1, width_slots) - 1))


def fuzzy_score(ocr: str, name: str, short: str = '', width: int | None = None) -> float:
    """0..100 agreement between a printed label and an item's names.

    The game prints the *short name*; the full name is a (weaker) second reference.
    Truncated labels are accepted only when the label is as long as the footprint's
    label capacity (``width`` slots; default: >= 6 characters), so a complete label
    like ``HK 416A5`` on a 5-wide gun cannot be mistaken for a cut-off ``416A5 RS``."""
    o = canon_nospace(ocr)
    if len(o) < 2:
        return 0.0
    best = 0.0
    for ref, w in ((short, 1.0), (name, 0.9)):
        r = canon_nospace(ref)
        if not r:
            continue
        trunc = len(o) >= (6 if width is None else label_capacity(width) - 1)
        cost = _align(o, r, trunc)
        # junk around the label is free only for names long enough to be unambiguous: a
        # 2-3 letter name ('ME', 'P', 'L1') must not match inside a longer read ('Meds')
        allow = 0 if len(r) <= 3 else 1
        extra = max(0, len(o) - len(r) - allow)
        # a 1-2 character name ('F-1', 'P') has almost no signal: every substitution costs a
        # large share of the score, so a junk read cannot vouch for it
        sc = max(0.0, 1.0 - cost / max(len(r), 1 if len(r) <= 2 else 3)) * 100.0 - 15.0 * extra
        best = max(best, max(0.0, sc) * w)
    return float(best)


def prefilter(ocr: str, choices: list[str], limit: int = 25, folded: bool = False) -> list[int]:
    """Indices of the most plausible names for an OCR string: the union of the best ``limit``
    by whole-string ratio (right for complete labels, 'M395' -> 'M995') and by partial ratio
    (right for truncated labels).  Cheap C-speed shape-folded matching; the weighted
    alignment then runs on these only."""
    if not choices:
        return []
    if process is None:
        return list(range(min(2 * limit, len(choices))))
    q = fold(ocr)
    if len(q) < 2:
        return []
    ch = choices if folded else [fold(c) for c in choices]
    out = {}
    for scorer in (fuzz.ratio, fuzz.partial_ratio):
        for _, _, i in process.extract(q, ch, scorer=scorer, limit=limit, score_cutoff=55):
            out[i] = True
    return list(out)
