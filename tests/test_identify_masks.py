"""Overlay-band geometry (derived from 63 px slots) and residual weight planes."""
import numpy as np
import pytest

from identify import masks as M
from identify.config import SLOT


def test_bands_at_reference_scale():
    assert M.band_rows(SLOT, 1) == (14, 14)                  # label + bottom overlay bands: 14 px of 63
    assert M.LABEL_PX == 14 and M.BOTTOM_PX == 14


def test_bands_scale_with_the_slot_size():
    top32, bot32 = M.band_rows(32, 1)
    assert top32 == 7 and bot32 == 7                         # 14 * 32/63 = 7.1
    top126, _ = M.band_rows(126, 1)
    assert top126 == 28


def test_weights_shape_and_border_zero():
    wp, wn = M.residual_weights(2, 1, 32)
    assert wp.shape == wn.shape == (32, 64)
    for w in (wp, wn):
        assert (w[0] == 0).all() and (w[-1] == 0).all() and (w[:, 0] == 0).all() and (w[:, -1] == 0).all()


def test_overlays_only_add_so_bands_downweight_positive_residuals_only():
    wp, wn = M.residual_weights(1, 1, 32)
    top, bot = M.band_rows(32, 1)
    mid = wp.shape[0] // 2
    assert wp[mid, 10] == pytest.approx(M.BODY_POS_WEIGHT)
    assert wp[2, 10] == pytest.approx(M.BAND_POS_WEIGHT) and M.BAND_POS_WEIGHT < M.BODY_POS_WEIGHT   # label band
    assert wp[-3, 10] == pytest.approx(M.BAND_POS_WEIGHT)            # bottom band
    assert np.allclose(wn[2:-2, 2:-2], M.NEG_WEIGHT)                    # missing art always counts fully


def test_clipped_tile_has_no_band_at_the_cut_edge():
    wp, wn = M.residual_weights(1, 1, 32, 20, 'bottom')              # only 20 rows visible, bottom cut
    assert wp.shape == (20, 32)
    assert wp[2, 10] == pytest.approx(M.BAND_POS_WEIGHT)             # label band exists at the top
    assert wp[-1, 10] != 0 and wp[-3, 10] == pytest.approx(M.BODY_POS_WEIGHT)   # cut edge: nothing is drawn there
    wp2, _ = M.residual_weights(1, 1, 32, 20, 'top')
    assert wp2[-3, 10] == pytest.approx(M.BAND_POS_WEIGHT) and wp2[2, 10] == pytest.approx(M.BODY_POS_WEIGHT)


def test_boxes_in_pixels():
    rect = (100, 200, 127, 64)                                       # a 2x1 at 63 px pitch
    x, y, w, h = M.label_box(rect, 63, 63)
    assert (x, y, w, h) == (100, 201, 127, 14)
    x, y, w, h = M.bottom_box(rect, 63)
    assert y + h == 200 + 64 - 1 and h == 14
    x, y, w, h = M.count_box(rect, 63, 63)
    assert x + w == 100 + 127 - 1 and w == 40
