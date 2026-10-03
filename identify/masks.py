"""
Where EFT draws its own overlays on top of an item icon, expressed in 63 px/slot
geometry (verified on ``data/eval/stash1.png`` by overlaying the bands on the
real tiles):

``label band``    the item's short name, drawn right-aligned in the top ~14 px of the
                  footprint (13 px glyph height + 1 px padding)
``bottom band``   bottom ~14 px: stack count / ammo count (bottom right), caliber text
                  of weapons and the mod / "has attachments" badges (bottom left) and the
                  Found-in-Raid check mark (bottom right corner)
``border``        the 1 px frame line on every side

Overlays are *added* on top of the art (white text, coloured badges): the tile is
therefore almost always **brighter** than the clean template inside a band, while
missing art shows up as the tile being **darker**.  Stage-1 therefore does not throw
the bands away (that would delete ~45 % of a 1x1 tile, including the half of an
injector that tells Morphine from Adrenaline) - it only down-weights *positive*
residuals there.  :func:`residual_weights` returns the two weight planes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import SLOT

LABEL_PX = 14          # label band height at 63 px/slot
BOTTOM_PX = 14         # bottom overlay band height at 63 px/slot
BORDER_PX = 1          # frame line
BAND_POS_WEIGHT = 0.12  # weight of "tile brighter than template" inside bands
BODY_POS_WEIGHT = 1.0
NEG_WEIGHT = 1.0       # "tile darker than template" always counts fully


def band_rows(slot: int, H: int) -> tuple[int, int]:
    """(rows in the label band, rows in the bottom band) at ``slot`` px/slot for
    a footprint of H slots (the bands sit on the footprint's outer edges)."""
    top = max(1, int(round(LABEL_PX * slot / SLOT)))
    bot = max(1, int(round(BOTTOM_PX * slot / SLOT)))
    return min(top, slot * H // 3), min(bot, slot * H // 3)


@lru_cache(maxsize=256)
def residual_weights(W: int, H: int, slot: int, vis_rows: int | None = None,
                     clip: str = '') -> tuple[np.ndarray, np.ndarray]:
    """Weight planes ``(w_pos, w_neg)`` of shape ``[h, w]`` for a W x H tile at
    ``slot`` px/slot.

    ``vis_rows`` / ``clip`` describe a tile cut off by the panel viewport
    (``clip`` = 'bottom' or 'top'): only that many rows exist; the band at the cut
    edge does not exist either (nothing is drawn there).
    """
    h = slot * H if vis_rows is None else vis_rows
    w = slot * W
    top, bot = band_rows(slot, H)
    wp = np.full((h, w), BODY_POS_WEIGHT, np.float32)
    wn = np.full((h, w), NEG_WEIGHT, np.float32)
    has_top = clip != 'top'
    has_bottom = clip != 'bottom'
    if has_top:
        wp[:top] = BAND_POS_WEIGHT
    if has_bottom:
        wp[h - bot:] = BAND_POS_WEIGHT
    b = BORDER_PX
    for arr in (wp, wn):
        arr[:, :b] = 0
        arr[:, w - b:] = 0
        if has_top:
            arr[:b] = 0
        if has_bottom:
            arr[h - b:] = 0
    return wp, wn


def label_box(rect: tuple, pitch_y: float, pitch_x: float, clip_top: bool = False) -> tuple[int, int, int, int]:
    """Pixel box ``(x, y, w, h)`` of the item-name label inside a footprint rect."""
    x, y, w, h = rect
    lh = max(6, int(round(LABEL_PX * pitch_y / SLOT)))
    return x, y + 1, w, lh


def bottom_box(rect: tuple, pitch_y: float) -> tuple[int, int, int, int]:
    """Pixel box of the bottom overlay band (count / caliber / badges)."""
    x, y, w, h = rect
    bh = max(6, int(round(BOTTOM_PX * pitch_y / SLOT)))
    return x, y + h - 1 - bh, w, bh


def count_box(rect: tuple, pitch_x: float, pitch_y: float) -> tuple[int, int, int, int]:
    """Pixel box of the stack-count digits (bottom right, up to ~5 characters)."""
    x, y, w, h = rect
    bh = max(6, int(round(BOTTOM_PX * pitch_y / SLOT)))
    bw = min(w, int(round(40 * pitch_x / SLOT)))
    return x + w - 1 - bw, y + h - 1 - bh, bw, bh
