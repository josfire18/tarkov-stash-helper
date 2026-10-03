"""Anchors (exact game render / exact game-font label), the TMP font-asset parser and the
per-tile result cache - synthetic data only (the real game font is used when it has been
extracted on this machine, otherwise those tests skip)."""
import json
import os
import struct

import cv2
import numpy as np
import pytest

from identify import anchors as A
from identify import fontlabel as FL
from identify import tmpfont
from identify.learned import LearnedNames
from identify.tilecache import TileCache, tile_key

REAL_FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fonts')


def _render(seed, W=1, H=1):
    """A synthetic 63 px/slot BGRA cache render: opaque random art on a transparent field."""
    rng = np.random.default_rng(seed)
    h, w = 63 * H + 1, 63 * W + 1
    img = np.zeros((h, w, 4), np.uint8)
    art = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3)).astype(np.uint8), (5, 5), 0)
    img[8:h - 8, 8:w - 8, :3] = art[8:h - 8, 8:w - 8]
    img[8:h - 8, 8:w - 8, 3] = 255
    return img


def _on_bg(bgra, bg=(40, 35, 30)):
    a = bgra[:, :, 3:4].astype(np.float32) / 255
    return (bgra[:, :, :3] * a + np.float32(bg) * (1 - a)).astype(np.uint8)


# ---------------------------------------------------------------- Anchor 1
def test_exact_render_native_and_shift():
    r = _render(1)
    tile = _on_bg(r)
    s, n, ph = A.exact_residual(tile, r, 63.0, 63.0, 1)
    assert s < 0.5 and n > 500
    shifted = np.roll(tile, 1, axis=1)
    s2, _, ph2 = A.exact_residual(shifted, r, 63.0, 63.0, 1)
    assert s2 < 0.5 and ph2[0] == 1
    other = _render(2)
    s3, _, _ = A.exact_residual(tile, other, 63.0, 63.0, 1)
    assert s3 > 5 * A.EXACT_MAX[63]


def test_exact_render_resampled_like_the_gpu():
    r = _render(3)
    sc = 84.0 / 63.0
    up = A.warp_render(r, 85, 85, sc, sc, 0.25, -0.25)
    tile = _on_bg(up)
    s, _, ph = A.exact_residual(tile, r, 84.0, 84.0, 1)
    assert s < A.EXACT_MAX[0] / 2
    assert abs(ph[0] - 0.25) <= 0.125 and abs(ph[1] + 0.25) <= 0.125


def _index(tmp_path, renders, assoc=None, learned=None):
    d = tmp_path / 'cache'
    d.mkdir()
    for i, r in enumerate(renders):
        cv2.imwrite(str(d / f'{i + 1}.png'), r)
    return A.CacheIndex(str(d), assoc or {}, learned)


def test_certain_only_when_the_render_is_named(tmp_path):
    ra, rb = _render(10), _render(11)
    idx = _index(tmp_path, [ra, rb])
    res = idx.match(_on_bg(ra), 1, 1, 63.0, 63.0)
    assert not res.certain and res.note == 'exact render of an unnamed item'
    idx.bind(res.render.key, 'item-a')                         # Anchor 2 certified it once
    res = idx.match(_on_bg(ra), 1, 1, 63.0, 63.0)
    assert res.certain and res.item_id == 'item-a'
    # a picture association is trusted only when it is (nearly) the same pixels and unique
    idx2 = A.CacheIndex(idx.dir, {'2.png': ['item-b', 4.2, 15.6]})
    res = idx2.match(_on_bg(rb), 1, 1, 63.0, 63.0)
    assert not res.certain and res.render.hint_id == 'item-b'
    idx3 = A.CacheIndex(idx.dir, {'2.png': ['item-b', 0.4, 9.0]})
    assert idx3.match(_on_bg(rb), 1, 1, 63.0, 63.0).certain


def test_two_items_with_the_same_render_are_never_certain(tmp_path):
    ra = _render(20)
    rb = ra.copy()
    rb[30, 30, :3] = 255 - rb[30, 30, :3]                   # one pixel apart: both renders "exact"
    idx = _index(tmp_path, [ra, rb])
    idx.bind(idx.renders[0].key, 'item-a')
    idx.bind(idx.renders[1].key, 'item-b')
    res = idx.match(_on_bg(ra), 1, 1, 63.0, 63.0)
    assert not res.certain and res.note == 'another item renders the same'


def test_rotated_tile_matches_rotated_render(tmp_path):
    r = _render(30, W=2, H=1)
    idx = _index(tmp_path, [r])
    idx.bind(idx.renders[0].key, 'long-item')
    tile = _on_bg(np.ascontiguousarray(np.rot90(r, -1)))
    res = idx.match(tile, 1, 2, 63.0, 63.0)
    assert res.certain and res.item_id == 'long-item'


def test_learned_certain_binding_is_final(tmp_path):
    ln = LearnedNames(str(tmp_path / 'l.json'))
    ln.bind_certain('k', 'a', 'A')
    assert ln.get('k', min_seen=2) == 'a'
    assert ln.learn('k', 'b', 'B') == 0                    # an ordinary read cannot overwrite it
    ln.bind_certain('k', 'b', 'B')                         # nor can a later conflicting certainty
    assert ln.get('k') == 'a'


# ---------------------------------------------------------------- tile cache
def test_tile_cache_hit_and_miss():
    img = np.random.default_rng(0).integers(0, 255, (200, 300, 3)).astype(np.uint8)
    c = TileCache(maxsize=2)
    k1 = tile_key(img, (10, 10, 64, 64), 63.0, 63.0)
    c.put(k1, {'x': 1})
    assert c.get(tile_key(img, (10, 10, 64, 64), 63.0, 63.0)) == {'x': 1}
    img2 = img.copy()
    img2[40, 40, 0] ^= 1                                     # one changed pixel (a count digit)
    assert c.get(tile_key(img2, (10, 10, 64, 64), 63.0, 63.0)) is None
    assert tile_key(img, (10, 10, 64, 64), 84.0, 84.0) != k1
    assert tile_key(img, (290, 10, 64, 64), 63.0, 63.0) is None
    c.put('a', 1)
    c.put('b', 2)
    assert c.get(k1) is None and len(c) == 2               # LRU eviction


# ---------------------------------------------------------------- TMP font asset parser
def _fake_font_asset():
    def s(txt):
        b = txt.encode()
        return struct.pack('<i', len(b)) + b + b'\0' * ((4 - len(b) % 4) % 4)
    raw = struct.pack('<iq', 0, 0) + b'\x01\0\0\0' + struct.pack('<iq', 1, 4048) + s('Test SDF')
    raw += s('1.1.0') + s('0' * 32) + struct.pack('<iq', 0, 0) + struct.pack('<i', 0)
    raw += struct.pack('<i', 0) + s('Bender') + s('Regular') + struct.pack('<ifi', 48, 1.0, 0)
    raw += struct.pack('<15f', *([1.0] * 15))
    raw += struct.pack('<i', 2)
    raw += struct.pack('<I5f4ifii', 7, 20, 30, 1, 30, 22, 10, 20, 20, 30, 1.0, 0, 0)
    raw += struct.pack('<I5f4ifii', 8, 10, 30, 2, 30, 12, 40, 20, 10, 30, 1.0, 0, 0)
    raw += struct.pack('<i', 2) + struct.pack('<iIIf', 1, ord('A'), 7, 1.0) + struct.pack('<iIIf', 1, ord('l'), 8, 1.0)
    raw += b'\0' * 64 + struct.pack('<4f', 0.0, 10.0, 0.5, 8.0) + bytes([15, 0, 0, 0, 10, 0, 0, 0])
    return raw


def test_parse_tmp_font_asset():
    fa = tmpfont.parse_font_asset(_fake_font_asset())
    assert fa['face']['pointSize'] == 48 and fa['face']['family'] == 'Bender'
    assert fa['chars'][str(ord('A'))] == 7 and fa['glyphs']['8']['adv'] == 12
    assert fa['normalSpacingOffset'] == 10.0 and fa['boldSpacing'] == 8.0


def test_no_game_means_no_label_anchor(tmp_path):
    assert FL.LabelRenderer.create(font_dir=str(tmp_path)) is None


# ---------------------------------------------------------------- Anchor 2 (needs the extracted font)
def _real_renderer():
    p = FL.asset_paths('Bender_Outline', REAL_FONTS)
    if p is None or not os.path.isfile(FL.PARAMS_PATH):
        pytest.skip('game font not extracted on this machine')
    return FL.LabelRenderer(tmpfont.TMPFont(*p), FL.load_params())


def _strip_with(lr, text, w=64, pitch=63.0, bg=(38, 34, 31), noise=2.0):
    h = int(np.ceil(17 * pitch / 63)) + 1
    rng = np.random.default_rng(len(text))
    base = np.full((h, w, 3), bg, np.float32) + rng.normal(0, noise, (h, w, 3))
    base[0, :] = (84, 81, 73)
    base[:, 0] = base[:, -1] = (84, 81, 73)
    rgb, al = lr.render(text, w, h, pitch)
    out = base * (1 - al[..., None]) + rgb * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def test_rendered_label_is_certain_among_confusers():
    lr = _real_renderer()
    st = _strip_with(lr, 'Bolts')
    fits = lr.match(st, ['Nuts', 'Bolts', 'Bolt', 'Belts', 'Tape'], 63.0)
    assert fits[0].text == 'Bolts' and fits[0].total < FL.LABEL_EXACT
    assert FL.certain(fits)


def test_truncated_label_and_print_twins():
    lr = _real_renderer()
    vis = lr.visible_text('Powerbank with a very long name', 64, 63.0)
    assert 'Powerbank with a very long name'.startswith(vis) and len(vis) < 20
    st = _strip_with(lr, vis)
    fits = lr.match(st, ['Powerbank with a very long name', 'Powerbank with a very long name 2'], 63.0)
    assert len(fits) == 1                                    # both print the same: one rendering


def test_label_anchor_scales_to_1440p():
    lr = _real_renderer()
    st = _strip_with(lr, 'Matches', w=85, pitch=84.0)
    fits = lr.match(st, ['Matches', 'Match', 'Watches', 'Patches'], 84.0)
    assert fits[0].text == 'Matches' and FL.certain(fits)
