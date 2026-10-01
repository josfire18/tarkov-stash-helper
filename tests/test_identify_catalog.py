"""Catalog build: compositing, per-size stacks, rotations, cache association, persistence."""
import json
import os

import cv2
import numpy as np
import pytest

from identify import catalog as C
from identify.config import SLOT, STAGE1_SLOT
from synth import make_icon

ITEMS = [
    ('a' * 24, 'Alpha 1x1', 'AL', 1, 1, 'blue', ['meds']),
    ('b' * 24, 'Beta 2x1', 'BE', 2, 1, 'violet', ['mods']),
    ('c' * 24, 'Gamma 2x1', 'GA', 2, 1, 'default', ['gun']),
    ('d' * 24, 'Delta 2x2', 'DE', 2, 2, 'black', ['container']),
    ('e' * 24, 'Epsilon default', 'EP', 3, 1, 'red', ['preset']),
]


@pytest.fixture()
def sources(tmp_path):
    tmpl = tmp_path / 'tmpl'
    tmpl.mkdir()
    cache = tmp_path / 'cache'
    cache.mkdir()
    prices = []
    for k, (iid, name, short, w, h, tint, types) in enumerate(ITEMS):
        cv2.imwrite(str(tmpl / f'{iid}.png'), make_icon(k + 1, w, h))
        prices.append({'id': iid, 'name': name, 'shortName': short, 'width': w, 'height': h,
                       'backgroundColor': tint, 'types': types})
    pp = tmp_path / 'prices.json'
    pp.write_text(json.dumps({'items': prices}), encoding='utf-8')
    return str(pp), str(tmpl), str(cache), tmp_path


def test_category_mapping_covers_every_type_family():
    assert C.category_of(['ammo']) == 'ammo' and C.category_of(['ammoBox']) == 'ammo'
    assert C.category_of(['gun', 'wearable']) == 'weapon' and C.category_of(['preset']) == 'weapon'
    assert C.category_of(['container']) == 'container' and C.category_of(['keys']) == 'keys'
    assert C.category_of(['mods', 'noFlea']) == 'mod' and C.category_of(['barter']) == 'barter'
    assert C.category_of([]) == 'other'


def test_no_category_is_excluded(sources):
    pp, tmpl, cache, _ = sources
    cat = C.build_catalog(pp, tmpl, cache, log=lambda *_: None)
    cats = set(map(str, cat.cats))
    assert {'weapon', 'container', 'meds', 'mod'} <= cats          # the legacy engine dropped weapon/container/ammo
    assert len(cat) == len(ITEMS)


def test_footprint_from_icon_pixel_size():
    assert C.footprint_of_image(np.zeros((64, 64, 4))) == (1, 1)
    assert C.footprint_of_image(np.zeros((127, 316, 4))) == (5, 2)
    assert C.footprint_of_image(np.zeros((128, 128, 4))) == (2, 2)   # some tarkov.dev images use 64 px/slot


def test_stage1_is_premultiplied_and_composites_back(sources):
    pp, tmpl, cache, _ = sources
    cat = C.build_catalog(pp, tmpl, cache, log=lambda *_: None)
    st = cat.stacks[(1, 1)]
    icon = cv2.imread(os.path.join(tmpl, 'a' * 24 + '.png'), cv2.IMREAD_UNCHANGED)
    prem, al = C.to_stage1(icon, 1, 1)
    assert prem.shape == (STAGE1_SLOT, STAGE1_SLOT, 3) and al.shape == (STAGE1_SLOT, STAGE1_SLOT)
    assert np.array_equal(st.prem[0], prem) and np.array_equal(st.alpha[0], al)
    # transparent pixels carry no colour (premultiplied), opaque ones keep theirs
    assert (prem[al == 0] == 0).all()
    # compositing over a background: rgb*alpha + (1-alpha)*bg
    bg = np.array([40, 40, 40], np.float32)
    comp = prem.astype(np.float32) + (1 - al[..., None] / 255.0) * bg
    assert comp[al == 0].mean() == pytest.approx(40, abs=0.5)


def test_rotated_stacks_swap_footprint_and_rotate_pixels(sources):
    pp, tmpl, cache, _ = sources
    cat = C.build_catalog(pp, tmpl, cache, log=lambda *_: None)
    base = cat.stacks[(2, 1)]
    stacks = cat.stacks_for(1, 2)                                   # a 1x2 tile can be a rotated 2x1 item
    rot = [s for s in stacks if s.rotated]
    assert len(rot) == 2 and {s.rot for s in rot} == {-1, 1}
    cw = next(s for s in rot if s.rot == -1)
    assert cw.prem.shape[1:3] == (2 * STAGE1_SLOT, STAGE1_SLOT)
    assert np.array_equal(cw.prem[0], np.rot90(base.prem[0], -1))   # clockwise
    ccw = next(s for s in rot if s.rot == 1)
    assert np.array_equal(ccw.alpha[0], np.rot90(base.alpha[0], 1))
    assert not cat.stacks_for(5, 5)                                  # unknown footprint: nothing, no crash


def test_square_items_are_not_double_counted(sources):
    pp, tmpl, cache, _ = sources
    cat = C.build_catalog(pp, tmpl, cache, log=lambda *_: None)
    assert [s.rotated for s in cat.stacks_for(1, 1, rotations=False)] == [False]


def test_cache_icon_association_and_builds(sources):
    pp, tmpl, cache, tmp = sources
    # (1) a cache icon that is the same render as an api icon -> not a new template (twin of api)
    same = cv2.imread(os.path.join(tmpl, 'b' * 24 + '.png'), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(cache, '10.png'), same)
    # (2) a slightly different render of item d -> a 'cache' variant carrying item d's id
    d = cv2.imread(os.path.join(tmpl, 'd' * 24 + '.png'), cv2.IMREAD_UNCHANGED).copy()
    d[:, :, :3] = np.clip(d[:, :, :3].astype(int) + 9, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(cache, '11.png'), d)
    # (3) something unrelated of footprint 2x2 -> anonymous 'build'
    cv2.imwrite(os.path.join(cache, '12.png'), make_icon(999, 2, 2))
    cat = C.build_catalog(pp, tmpl, cache, log=lambda *_: None)
    srcs = list(map(str, cat.src))
    assert srcs.count('api') == len(ITEMS)
    cached = [i for i, s in enumerate(srcs) if s == 'cache']
    builds = [i for i, s in enumerate(srcs) if s == 'build']
    assert len(cached) == 1 and str(cat.ids[cached[0]]) == 'd' * 24
    assert len(builds) == 1 and str(cat.ids[builds[0]]) == ''
    assert cat.meta['n_cache'] == 1 and cat.meta['n_build'] == 1


def test_persistence_roundtrip_and_rebuild_on_source_change(sources, tmp_path):
    pp, tmpl, cache, tmp = sources
    out = str(tmp_path / 'cat.npz')
    log = []
    c1 = C.load_catalog(out, pp, tmpl, cache, log=log.append)
    assert os.path.exists(out) and log                              # built
    log.clear()
    c2 = C.load_catalog(out, pp, tmpl, cache, log=log.append)
    assert not log                                                   # loaded from disk, not rebuilt
    assert len(c2) == len(c1) and np.array_equal(c2.ids, c1.ids)
    assert np.array_equal(c2.stacks[(2, 1)].prem, c1.stacks[(2, 1)].prem)
    # add a new source file -> signature changes -> rebuild
    cv2.imwrite(os.path.join(cache, '99.png'), make_icon(5, 2, 2))
    c3 = C.load_catalog(out, pp, tmpl, cache, log=log.append)
    assert log and len(c3) == len(c1) + 1


def test_schema_mismatch_forces_rebuild(sources, tmp_path, monkeypatch):
    pp, tmpl, cache, _ = sources
    out = str(tmp_path / 'cat.npz')
    C.load_catalog(out, pp, tmpl, cache, log=lambda *_: None)
    monkeypatch.setattr(C, 'CATALOG_SCHEMA', C.CATALOG_SCHEMA + 1)
    log = []
    C.load_catalog(out, pp, tmpl, cache, log=log.append)
    assert log
