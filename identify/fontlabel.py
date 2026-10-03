"""
Anchor 2: closed-set label match rendered with the game's own text engine.

The stash prints every item's short name on its tile with TextMeshPro, font asset
"Jovanny Lemonad - Bender Outline SDF" (calibrated, see ``docs/accuracy.md``).  Instead of reading
the label with a general OCR engine, this module *renders* the short name of every candidate that
can occupy the tile exactly as the game does (:mod:`identify.tmpfont`: the game's SDF atlas, glyph
metrics, spacing and material, evaluated with TMP's shader maths) at the size, position and colour
calibrated on real strips (:data:`PARAMS_PATH`), and compares the rendering with the observed strip
pixel by pixel.

* **Font**: :func:`ensure_font` extracts the Bender font assets from the installed game
  (``EscapeFromTarkov_Data/resources.assets``) with UnityPy once, into ``data/fonts/`` (gitignored,
  never redistributed).  Without the game or UnityPy the anchor is simply off.
* **Layout** (TMP right alignment, overflow mode Truncate): the text is right-aligned in the label
  box; a name wider than the box loses the characters that do not fit and the visible prefix is
  right-aligned ("Powerbank" -> "Powerban").
* **Score** (:meth:`LabelRenderer.score`): mean absolute level error over the pixels the
  candidate's rendering covers (face + dark outline), compositing over the strip's own background
  (inpainted under the text), plus the observed label ink the candidate leaves unexplained.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DATA_DIR, SLOT
from . import tmpfont

FONT_DIR = os.path.join(DATA_DIR, 'fonts')
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'label_params.json')

GAME_DIRS = [r'C:\Battlestate Games\Escape from Tarkov', r'C:\Battlestate Games\EFT',
             r'D:\Battlestate Games\Escape from Tarkov', r'E:\Battlestate Games\Escape from Tarkov']


# --------------------------------------------------------------------------
# font extraction (first run)
# --------------------------------------------------------------------------

def game_data_dirs(extra: list | None = None) -> list[str]:
    out = []
    for d in list(extra or []) + GAME_DIRS:
        if not d:
            continue
        for cand in (os.path.join(d, 'EscapeFromTarkov_Data'), d):
            if os.path.isfile(os.path.join(cand, 'resources.assets')) and cand not in out:
                out.append(cand)
    try:                                 # the launcher records the install path in the registry
        import winreg
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for sub in (r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov',
                        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov'):
                try:
                    with winreg.OpenKey(root, sub) as k:
                        loc = winreg.QueryValueEx(k, 'InstallLocation')[0]
                except OSError:
                    continue
                cand = os.path.join(loc, 'EscapeFromTarkov_Data')
                if os.path.isfile(os.path.join(cand, 'resources.assets')) and cand not in out:
                    out.append(cand)
    except ImportError:
        pass
    return out


def asset_paths(asset: str, font_dir: str = FONT_DIR) -> tuple[str, str] | None:
    j = os.path.join(font_dir, asset + '_sdf.json')
    a = os.path.join(font_dir, asset + '_atlas.png')
    return (j, a) if os.path.isfile(j) and os.path.isfile(a) else None


def ensure_font(asset: str = 'Bender_Outline', game_dirs: list | None = None, font_dir: str = FONT_DIR,
                log=print) -> tuple[str, str] | None:
    """(json, atlas) of the game's label font asset, extracting it from the installed game on
    first use.  None when the game or UnityPy is not available (Anchor 2 is then off)."""
    p = asset_paths(asset, font_dir)
    if p:
        return p
    for d in game_data_dirs(game_dirs):
        try:
            tmpfont.extract(d, font_dir, log)
        except Exception as e:           # UnityPy missing / unreadable assets: the anchor is off
            log(f'[font] extraction from {d} failed: {e}')
            continue
        p = asset_paths(asset, font_dir)
        if p:
            return p
    return None


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class LabelParams:
    """Label layout at 63 px/slot (scaled linearly with the pitch) and the text colour."""
    asset: str = 'Bender_Outline'
    size: float = 12.0          # TMP font size in px at 63 px/slot
    bold: bool = False
    k: float = 1.0              # shader screen-scale calibration (1 = TMP's own)
    spacing: float = 0.0        # TMP character spacing (em/100) added to the asset's normalSpacingOffset
    shade: dict | None = None   # measured face/outline shading (see tmpfont.TMPRenderer); None = material
    baseline: float = 11.0      # baseline y below the tile's top-left corner (frame line = row 0)
    pad_right: float = 2.0      # gap between the right frame line and the end of the text's advance
    pad_left: float = 2.0       # left edge of the label box (from the left frame line)
    color: tuple = (0.75, 0.73, 0.70)   # vertex colour, BGR 0-1

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d['color'] = [float(c) for c in self.color]
        return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}


def load_params(path: str = PARAMS_PATH) -> LabelParams:
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        return LabelParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()
                              if k in LabelParams.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return LabelParams()


@dataclass
class Fit:
    text: str
    score: float                 # mean abs level error over the candidate's ink
    shift: tuple = (0.0, 0.0)
    unexplained: float = 0.0     # observed face-coloured pixels outside the candidate's ink (count / ink px)
    total: float = 0.0
    n_ink: int = 0


def strip_of(img_bgr: np.ndarray, rect: tuple, pitch_y: float) -> np.ndarray | None:
    """The tile's label band (frame lines included), BGR."""
    x, y, w, h = rect
    lh = int(np.ceil(17 * pitch_y / SLOT)) + 1
    if x < 0 or y < 0 or x + w > img_bgr.shape[1] or y + lh > img_bgr.shape[0]:
        return None
    return img_bgr[y:y + lh, x:x + w]


INK_ALPHA = 0.5                  # pixels whose colour barely depends on the (unknown) background


class LabelRenderer:
    """Renders short names exactly as the stash draws them and scores them against strips."""

    def __init__(self, font: tmpfont.TMPFont, params: LabelParams):
        self.font = font
        self.p = params
        self.tmp = (tmpfont.SpriteRenderer(font, params.shade) if params.shade
                    else tmpfont.TMPRenderer(font, params.k, params.shade))
        self._cache: dict = {}
        self._lock = threading.Lock()

    @classmethod
    def create(cls, params: LabelParams | None = None, font_dir: str = FONT_DIR, log=print):
        p = params or load_params()
        paths = ensure_font(p.asset, font_dir=font_dir, log=log)
        if paths is None:
            return None
        return cls(tmpfont.TMPFont(*paths), p)

    # -- layout -------------------------------------------------------------
    def size(self, pitch: float) -> float:
        return self.p.size * pitch / SLOT

    def box(self, strip_w: int, pitch: float) -> tuple[float, float]:
        s = pitch / SLOT
        return self.p.pad_left * s, (strip_w - 1) - self.p.pad_right * s

    def visible_text(self, text: str, strip_w: int, pitch: float) -> str:
        """TMP overflow mode Truncate: the longest prefix whose advance fits the box."""
        l, r = self.box(strip_w, pitch)
        size = self.size(pitch)
        if self.font.advance(text, size, self.p.spacing) <= r - l:
            return text
        out = ''
        for i in range(1, len(text) + 1):
            if self.font.advance(text[:i], size, self.p.spacing) > r - l:
                break
            out = text[:i]
        return out.rstrip()

    def render(self, text: str, w: int, h: int, pitch: float, dx: float = 0.0, dy: float = 0.0,
               color=None):
        vis = self.visible_text(text, w, pitch)
        size = self.size(pitch)
        _, r = self.box(w, pitch)
        pen = r - self.font.advance(vis, size, self.p.spacing) + dx
        base = self.p.baseline * pitch / SLOT + dy
        key = (vis, w, h, round(pitch, 3), round(pen, 3), round(base, 3), tuple(color or self.p.color))
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self.tmp.render(vis, w, h, size, pen, base, color or self.p.color, self.p.bold, self.p.spacing)
        with self._lock:
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = out
        return out

    # -- scoring ------------------------------------------------------------
    @staticmethod
    def background(obs: np.ndarray, al: np.ndarray) -> np.ndarray:
        """The strip without the text: inpainted under the (dilated) rendered ink."""
        m = cv2.dilate((al > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8))
        m[0, :] = 1                      # the frame lines are not background
        m[:, -1] = 1
        m[:, 0] = 1
        return cv2.inpaint(obs, m, 2, cv2.INPAINT_TELEA).astype(np.float32)

    def residual(self, obs: np.ndarray, text: str, pitch: float, dx: float, dy: float,
                 bg: np.ndarray | None = None) -> tuple[float, int]:
        h, w = obs.shape[:2]
        rgb, al = self.render(text, w, h, pitch, dx, dy)
        m = al > INK_ALPHA
        n = int(m.sum())
        if n == 0:
            return 999.0, 0
        b = obs.astype(np.float32) if bg is None else bg
        pred = b * (1 - al[..., None]) + rgb * 255.0
        return float(np.abs(pred[m] - obs[m].astype(np.float32)).mean()), n

    def align(self, obs: np.ndarray, text: str, pitch: float, rng: float = 2.0, phase=None) -> Fit:
        """Best sub-pixel shift of the candidate's rendering against a strip."""
        if phase is None:
            steps = np.arange(-rng, rng + 1e-6, 0.5)
            grid = [(dx, dy) for dy in steps for dx in steps]
        else:
            grid = [(phase[0] + a, phase[1] + b) for a in (-0.25, 0, 0.25) for b in (-0.25, 0, 0.25)]
        best = (999.0, 0, 0.0, 0.0)
        for dx, dy in grid:
            r, n = self.residual(obs, text, pitch, dx, dy)
            if r < best[0]:
                best = (r, n, dx, dy)
        for step in ((0.25, 0.125) if phase is None else (0.125,)):
            _, _, cx, cy = best
            for a in (-step, 0, step):
                for b in (-step, 0, step):
                    if a or b:
                        r, n = self.residual(obs, text, pitch, cx + a, cy + b)
                        if r < best[0]:
                            best = (r, n, cx + a, cy + b)
        _, _, dx, dy = best
        h, w = obs.shape[:2]
        _, al = self.render(text, w, h, pitch, dx, dy)
        bg = self.background(obs, al)
        r, n = self.residual(obs, text, pitch, dx, dy, bg)
        return Fit(text, r, (dx, dy), n_ink=n)

    def fit_colour(self, items) -> tuple:
        """Least-squares vertex colour from ``[(obs, text, pitch, shift)]``: the rendering is affine
        in the colour (``rgb = rgb0 + c (rgb1 - rgb0)``) for fixed alphas."""
        A, B = [], []
        for obs, text, pitch, (dx, dy) in items:
            h, w = obs.shape[:2]
            rgb1, al = self.render(text, w, h, pitch, dx, dy, color=(1.0, 1.0, 1.0))
            rgb0, _ = self.render(text, w, h, pitch, dx, dy, color=(0.0, 0.0, 0.0))
            m = al > INK_ALPHA
            bg = self.background(obs, al)
            y = obs[m].astype(np.float32) - bg[m] * (1 - al[m])[:, None] - rgb0[m] * 255.0
            x = (rgb1[m] - rgb0[m]) * 255.0
            A.append(x)
            B.append(y)
        A, B = np.concatenate(A), np.concatenate(B)
        col = [(A[:, c] * B[:, c]).sum() / max(1e-6, (A[:, c] ** 2).sum()) for c in range(3)]
        return tuple(float(np.clip(c, 0, 1)) for c in col)

    # -- closed-set match ----------------------------------------------------
    def observed_ink(self, obs: np.ndarray, pitch: float) -> np.ndarray:
        """bool: pixels of the strip that look like label face pixels (the text colour, with
        the dark outline within a pixel) inside the label rows."""
        face = np.float32(self.p.color) * 255.0
        d = np.abs(obs.astype(np.float32) - face[None, None]).max(axis=2)
        dark = cv2.erode(obs.max(axis=2), np.ones((3, 3), np.uint8)) < 60
        m = (d < 40) & dark
        s = pitch / SLOT
        top = int(np.floor((self.p.baseline - 0.75 * self.p.size) * s))
        bot = int(np.ceil((self.p.baseline + 0.5) * s))
        m[:max(0, top)] = False
        m[bot:] = False
        m[:, :1] = False
        m[:, -1:] = False
        return m

    def match(self, obs: np.ndarray, texts: list, pitch: float, shifts=None) -> list:
        """Score every candidate text (deduplicated by what the game would print) against the
        strip; best first.  ``total`` = ink residual + a penalty for observed label ink the
        candidate does not cover."""
        h, w = obs.shape[:2]
        shifts = shifts or SHIFTS
        ink = self.observed_ink(obs, pitch)
        n_obs = int(ink.sum())
        seen, fits = {}, []
        for t in texts:
            vis = self.visible_text(t, w, pitch)
            if not vis or vis in seen:
                continue
            seen[vis] = t
            r, n = self.residual(obs, t, pitch, 0.0, 0.0)
            fits.append(Fit(vis, r, (0.0, 0.0), n_ink=n))
        # sub-pixel search only for candidates that could be in the race (a 0.5 px shift cannot
        # halve a residual); far ones keep their unshifted score and total = score, both of which
        # only make them look *closer* to the winner (conservative for certainty)
        b0 = min((f.score for f in fits), default=0.0)
        near = [f for f in fits if f.score <= 1.6 * b0 + 8.0]
        for f in near:
            best = (f.score, f.n_ink, 0.0, 0.0)
            for dx, dy in shifts:
                if dx == 0.0 and dy == 0.0:
                    continue
                r, n = self.residual(obs, f.text, pitch, dx, dy)
                if r < best[0]:
                    best = (r, n, dx, dy)
            f.score, f.n_ink, f.shift = best[0], best[1], (best[2], best[3])
        fits.sort(key=lambda f: f.score)
        for f in fits[:4]:                       # refine the front runners with the true background
            _, al = self.render(f.text, w, h, pitch, *f.shift)
            bg = self.background(obs, al)
            f.score, f.n_ink = self.residual(obs, f.text, pitch, f.shift[0], f.shift[1], bg)
        for f in fits:                           # unrefined: a lower bound of what a shift could reach
            f.total = FAR_DISCOUNT * f.score
        for f in near:
            _, al = self.render(f.text, w, h, pitch, *f.shift)
            cover = cv2.dilate((al > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            f.unexplained = float((ink & ~cover).sum()) / max(1, n_obs)
            f.total = f.score + UNEXPLAINED_W * f.unexplained
        fits.sort(key=lambda f: f.total)
        return fits


SHIFTS = [(0.0, 0.0), (0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25), (0.5, 0.0), (-0.5, 0.0)]
UNEXPLAINED_W = 60.0
FAR_DISCOUNT = 0.8            # unrefined candidates are scored optimistically (conservative for certainty)
LABEL_EXACT = 24.0            # total of an exact label rendering (calibrated, docs/accuracy.md)
LABEL_MARGIN = 1.6            # the runner-up must score this x worse ...
LABEL_GAP = 8.0               # ... and at least this much worse
LABEL_MAX_UNEXPLAINED = 0.10


def certain(fits: list) -> bool:
    """Anchor 2 decision over a closed candidate set (``match`` output)."""
    if not fits:
        return False
    b = fits[0]
    if b.total > LABEL_EXACT or b.unexplained > LABEL_MAX_UNEXPLAINED:
        return False
    if len(fits) == 1:
        return True
    r = fits[1].total
    return r >= LABEL_MARGIN * b.total and r - b.total >= LABEL_GAP
