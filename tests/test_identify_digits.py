"""Count glyph reader: prototypes round-trip through a synthetic tile."""
import cv2
import numpy as np
import pytest

from identify import digits as D


def _tile_with(text, gw=6, gh=9, colour=(235, 235, 235)):
    protos = D._load()
    if not protos:
        pytest.skip('digit prototypes not trained')
    tile = np.full((64, 64, 3), 30, np.uint8)
    x = 64 - 3 - len(text) * (gw + 1)
    for ch in text:
        g = np.mean(protos[ch], axis=0).reshape(D.GH, D.GW)
        g = (cv2.resize(g, (gw, gh), interpolation=cv2.INTER_LINEAR) > 0.5)
        y = 64 - 4 - gh
        tile[y:y + gh, x:x + gw][g] = colour
        x += gw + 1
    return tile


def test_prototypes_exist_for_every_character():
    protos = D._load()
    assert set(D.CHARS) <= set(protos)


@pytest.mark.parametrize('text,count', [('80', 80), ('29', 29), ('296/400', 296), ('3/3', 3), ('17/17', 17)])
def test_reads_counts(text, count):
    got, raw = D.read_count(_tile_with(text))
    assert got == count and raw == text


def test_red_digits_are_read_too():
    got, _ = D.read_count(_tile_with('2/3', colour=(60, 60, 235)))
    assert got == 2


def test_blank_or_non_text_bands_return_none():
    assert D.read_count(np.full((64, 64, 3), 30, np.uint8)) == (None, '')
    assert D.read_count(None) == (None, '')
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    got, _ = D.read_count(noise)
    assert got is None


def test_training_roundtrip(tmp_path):
    t1, t2 = _tile_with('80'), _tile_with('29')
    info = D.train([t1, t2], ['80', '29'], str(tmp_path / 'd.npz'))
    assert info['glyphs'] == 4
    D._protos = None                             # restore the shipped prototypes for other tests
