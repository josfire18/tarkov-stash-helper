"""
Synthetic EFT-like stash renderer for the identify/ unit tests (no game data, no network).

``render_panel`` draws a grid the way the game does: every item is framed by a 1 px line in
the border colour, lines that would run *inside* a multi-cell item are not drawn, empty cells
are flat dark, each item has a tinted background and a textured "icon".
"""
from __future__ import annotations

import numpy as np
import cv2

LINE = (84, 81, 73)
EMPTY_BG = (22, 22, 22)


def make_icon(seed: int, W: int, H: int, slot: int = 63) -> np.ndarray:
    """Random BGRA icon of (slot*W+1) x (slot*H+1) px with a transparent margin."""
    rng = np.random.default_rng(seed)
    h, w = slot * H + 1, slot * W + 1
    img = np.zeros((h, w, 4), np.uint8)
    for _ in range(4):
        col = tuple(int(c) for c in rng.integers(60, 250, 3))
        x0, y0 = int(rng.integers(8, max(9, w // 2))), int(rng.integers(8, max(9, h // 2)))
        x1, y1 = int(rng.integers(x0 + 6, w - 6)), int(rng.integers(y0 + 6, h - 6))
        cv2.rectangle(img, (x0, y0), (x1, y1), col + (255,), -1)
        cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), int(rng.integers(4, 12)), (255, 255, 255, 255), -1)
    return img


def composite(icon_bgra: np.ndarray, bg) -> np.ndarray:
    a = icon_bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (icon_bgra[:, :, :3] * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)


def render_panel(pitch: float, cols: int, rows: int, items, origin=(40, 30), top_partial: float = 0.0,
                 bottom_partial: float = 0.0, pitch_y: float | None = None, chrome: bool = True,
                 size=None, bg=(10, 10, 10), icons=None, tint=(44, 40, 30)):
    """Draw a stash panel.

    ``items`` = [(col, row, w, h)]; cells not covered are empty.  ``top_partial`` /
    ``bottom_partial`` are the fractions of the first / last row that are *hidden* by the
    viewport.  Returns (image, panel_rect, cell_rects) where cell_rects are the expected
    footprint rectangles (x, y, w, h incl. closing line) clipped to the viewport.
    """
    py = pitch if pitch_y is None else pitch_y
    ox, oy = origin
    # y of grid line k is oy + (k - top_partial) * py  -> first row partially hidden above oy
    full_h = int(round((rows - top_partial - bottom_partial) * py)) + 1
    W = size[0] if size else int(ox + cols * pitch + 60)
    H = size[1] if size else int(oy + full_h + 60)
    img = np.full((H, W, 3), bg, np.uint8)
    rng = np.random.default_rng(1)
    if chrome:                                   # header text bar + a long horizontal rule above the panel
        cv2.rectangle(img, (0, 0), (W, max(2, oy - 8)), (30, 28, 26), -1)
        cv2.line(img, (0, max(1, oy - 12)), (W, max(1, oy - 12)), (70, 68, 66), 1)
    xl = lambda k: int(round(ox + k * pitch))
    yl = lambda k: int(round(oy + (k - top_partial) * py))
    y_top, y_bot = oy, oy + full_h - 1           # visible viewport
    covered = np.zeros((rows, cols), bool)
    rects = []

    def clipv(y):
        return max(y_top, min(y_bot, y))

    allitems = list(items)
    for c in range(cols):
        for r in range(rows):
            if not any(ic <= c < ic + iw and ir <= r < ir + ih for ic, ir, iw, ih in allitems):
                allitems.append((c, r, 1, 1))
                covered[r, c] = True            # mark synthetic empty cells
    empties = {(c, r) for r in range(rows) for c in range(cols) if covered[r, c]}
    for k, (c, r, w, h) in enumerate(allitems):
        x0, x1 = xl(c), xl(c + w)
        y0, y1 = yl(r), yl(r + h)
        if (w, h) == (1, 1) and (c, r) in empties:
            ya, yb = clipv(y0), clipv(y1)
            cv2.rectangle(img, (x0, ya), (x1, yb), EMPTY_BG, -1)
            # empty slots carry the grid lines too (faint)
            cv2.rectangle(img, (x0, ya), (x1, yb), LINE, 1)
            continue
        ya, yb = clipv(y0), clipv(y1)
        # tinted bg + icon
        cv2.rectangle(img, (x0, ya), (x1, yb), tint, -1)
        if icons is not None and (c, r) in icons:
            ic = icons[(c, r)]
        else:
            ic = make_icon(100 + k, w, h, slot=int(round(pitch)))
        comp = composite(cv2.resize(ic, (x1 - x0 + 1, y1 - y0 + 1), interpolation=cv2.INTER_AREA), tint)
        sub = comp[ya - y0:yb - y0 + 1]
        img[ya:yb + 1, x0:x1 + 1] = sub[:, :x1 - x0 + 1]
        # border
        cv2.rectangle(img, (x0, ya), (x1, yb), LINE, 1)
        if y0 < y_top:                          # top cut by viewport: no top line there
            cv2.line(img, (x0, y_top), (x1, y_top), tuple(int(v) for v in img[y_top + 2, x0 + 3]), 1)
        if y1 > y_bot:
            cv2.line(img, (x0, y_bot), (x1, y_bot), tuple(int(v) for v in img[y_bot - 2, x0 + 3]), 1)
        rects.append(((c, r, w, h), (x0, ya, x1 - x0 + 1, yb - ya + 1)))
    return img, (xl(0), y_top, xl(cols), y_bot), rects, empties
