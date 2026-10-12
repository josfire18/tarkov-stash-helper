"""Coverage rules: same-label twins whose drawn shapes differ clearly are settled by a smaller
picture lead than look-alike colour variants (docs/accuracy.md, "Coverage pass")."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from identify import catalog as C
from identify.config import EngineSettings
from identify.pipeline import Engine


def idof(i):
    return f'{i:024x}'


def _icon(kind: str, colour) -> np.ndarray:
    img = np.zeros((64, 64, 4), np.uint8)
    if kind == 'left':
        cv2.rectangle(img, (6, 6), (30, 56), colour + (255,), -1)
    elif kind == 'right':
        cv2.rectangle(img, (34, 6), (58, 56), colour + (255,), -1)
    else:                                              # 'wide'
        cv2.rectangle(img, (6, 6), (58, 56), colour + (255,), -1)
    return img


ITEMS = [  # id, name, short, icon kind, colour
    (idof(21), 'Zed mag', 'ZED', 'left', (200, 80, 60)),
    (idof(22), 'Zed grip', 'ZED', 'right', (60, 80, 200)),          # another shape
    (idof(23), 'Zed mag (green)', 'ZED', 'left', (60, 200, 80)),    # same shape, other colour
]


@pytest.fixture(scope='module')
def eng(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('shape')
    (tmp / 'tmpl').mkdir()
    prices = []
    for iid, name, short, kind, col in ITEMS:
        cv2.imwrite(str(tmp / 'tmpl' / f'{iid}.png'), _icon(kind, col))
        prices.append({'id': iid, 'name': name, 'shortName': short, 'width': 1, 'height': 1,
                       'backgroundColor': 'blue', 'types': ['barter']})
    (tmp / 'p.json').write_text(json.dumps({'items': prices}), encoding='utf-8')
    cat = C.build_catalog(str(tmp / 'p.json'), str(tmp / 'tmpl'), str(tmp / 'nc'), with_cache=False,
                          log=lambda *_: None)
    return Engine(EngineSettings(use_dino=False, use_ocr=False, accelerate=False), catalog=cat)


def _w(eng, ids):
    cands = [SimpleNamespace(row=eng._id_row[i], score=res, rot=0) for i, res in ids]
    return {'cands': cands, 'qe': None, 'sims': {}, 'tile': SimpleNamespace(W=1, H=1, clip='')}


def test_silhouettes_tell_objects_from_colour_variants(eng):
    w = _w(eng, [(idof(21), 4.0), (idof(22), 10.0), (idof(23), 9.0)])
    r = eng._id_row
    assert eng._shape_distinct(w, r[idof(21)], r[idof(22)])
    assert not eng._shape_distinct(w, r[idof(21)], r[idof(23)])


def test_shape_distinct_twin_needs_only_a_smaller_lead(eng):
    r = eng._id_row
    twins = [idof(21), idof(22)]
    tile = SimpleNamespace(W=1, H=1, clip='')
    # lead of 4.5: below the full 6.0 gap, above the shape-distinct 3.0 gap
    w = _w(eng, [(idof(21), 4.0), (idof(22), 8.5)])
    assert eng._twins_settled_by_footprint(idof(21), twins, tile, False, w)
    # the same lead over a colour variant stays unsettled while the tile's colour is in between
    w = _w(eng, [(idof(21), 4.0), (idof(23), 8.5)])
    w['tile'] = _tile_of('left', (130, 140, 70))
    assert not eng._twins_settled_by_footprint(idof(21), [idof(21), idof(23)], w['tile'], False, w)
    # too small a lead is never enough, even for a different shape
    w = _w(eng, [(idof(21), 4.0), (idof(22), 6.0)])
    assert not eng._twins_settled_by_footprint(idof(21), twins, tile, False, w)
    # one close colour variant among the rivals vetoes the pick
    w = _w(eng, [(idof(21), 4.0), (idof(22), 12.0), (idof(23), 8.5)])
    w['tile'] = _tile_of('left', (130, 140, 70))
    assert not eng._twins_settled_by_footprint(idof(21), [idof(21), idof(22), idof(23)], w['tile'], False, w)


def _tile_of(kind, colour):
    """A 1x1 stage-1 tile (32 px/slot) showing the given icon on a dark background."""
    icon = cv2.resize(_icon(kind, colour), (32, 32), interpolation=cv2.INTER_AREA)
    a = icon[:, :, 3:4].astype(np.float32) / 255.0
    img = (icon[:, :, :3] * a + np.float32((30, 30, 30)) * (1 - a)).astype(np.uint8)
    return SimpleNamespace(W=1, H=1, clip='', img=img)


def test_colour_variants_are_settled_by_colour_alone(eng):
    twins = [idof(21), idof(23)]                        # same silhouette, red-ish vs green
    w = _w(eng, [(idof(21), 4.0), (idof(23), 6.5)])     # residual lead 2.5: far below the 6.0 gap
    w['tile'] = _tile_of('left', (200, 80, 60))
    assert eng._colour_gap(w, eng._id_row[idof(21)], eng._id_row[idof(23)]) > 12.0
    assert eng._twins_settled_by_footprint(idof(21), twins, w['tile'], False, w)
    # the tile is the green one: the colour gap points the other way, so the red pick stays unsettled
    w = _w(eng, [(idof(21), 4.0), (idof(23), 6.5)])
    w['tile'] = _tile_of('left', (60, 200, 80))
    assert eng._colour_gap(w, eng._id_row[idof(21)], eng._id_row[idof(23)]) < 0
    assert not eng._twins_settled_by_footprint(idof(21), twins, w['tile'], False, w)
    # the picture must not prefer the rival either
    w = _w(eng, [(idof(21), 6.5), (idof(23), 4.0)])
    w['tile'] = _tile_of('left', (200, 80, 60))
    assert not eng._twins_settled_by_footprint(idof(21), twins, w['tile'], False, w)


# ---------------------------------------------------------------- gun builds proven by their calibre
GUNS = [  # id, name, short, W, H, types
    (idof(31), 'YY-1 12ga pump-action shotgun', 'YY-1', 5, 1, ['gun']),
    (idof(32), 'YY-1 stock', 'YY-1', 4, 1, ['mods']),
    (idof(33), 'YY-1 12ga barrel', 'YY1 barrel', 5, 1, ['mods']),
]


@pytest.fixture(scope='module')
def geng(tmp_path_factory):
    from synth import make_icon
    tmp = tmp_path_factory.mktemp('guns')
    (tmp / 'tmpl').mkdir()
    prices = []
    for k, (iid, name, short, w, h, types) in enumerate(GUNS):
        cv2.imwrite(str(tmp / 'tmpl' / f'{iid}.png'), make_icon(900 + k, w, h))
        prices.append({'id': iid, 'name': name, 'shortName': short, 'width': w, 'height': h,
                       'backgroundColor': 'blue', 'types': types})
    (tmp / 'p.json').write_text(json.dumps({'items': prices}), encoding='utf-8')
    cat = C.build_catalog(str(tmp / 'p.json'), str(tmp / 'tmpl'), str(tmp / 'nc'), with_cache=False,
                          log=lambda *_: None)
    return Engine(EngineSettings(use_dino=False, use_ocr=True, accelerate=False), catalog=cat)


def _decide_gun(geng, monkeypatch, res_gun, res_barrel, label, calibre):
    from identify import ocr as O
    monkeypatch.setattr(geng, '_use_ocr', lambda: True)
    monkeypatch.setattr(O, 'bottom_strip', lambda *a, **k: np.zeros((8, 40), np.uint8))
    monkeypatch.setattr(O, 'read_strips', lambda strips: [calibre] * len(strips))
    pitch = 63.0
    img = np.full((120, 380, 3), 30, np.uint8)
    it = SimpleNamespace(rect=(10, 20, 5 * 63 + 1, 64), w=5, h=1, col=0, row=0, panel=0, clipped_top=False,
                         clipped_bottom=False)
    panel = SimpleNamespace(pitch_x=pitch, pitch_y=pitch, y0=0, y1=500)
    tile = SimpleNamespace(W=5, H=1, clip='', img=np.zeros((32, 160, 3), np.uint8))
    r = geng._id_row
    cands = sorted([SimpleNamespace(row=r[idof(31)], score=res_gun, rot=0, rotated=False, W=5, H=1),
                    SimpleNamespace(row=r[idof(33)], score=res_barrel, rot=0, rotated=False, W=5, H=1)],
                   key=lambda c: c.score)
    ocr_all = [label] if label else []
    w = {'it': it, 'panel': panel, 'tile': tile, 'cands': cands, 'qe': None, 'sims': {}, 'ocr_all': ocr_all,
         'ocr_best': label, 'pool': cands, 'oscores': {c.row: 25.0 for c in cands}}
    return geng._decide(img, w)


def test_gun_build_with_unreadable_label_is_sure_when_its_own_calibre_is_printed(geng, monkeypatch):
    d = _decide_gun(geng, monkeypatch, 2.5, 12.0, 'YY Tactical', '12g')
    assert d.name.startswith('YY-1 12ga') and not d.uncertain
    assert d.evidence.get('verified') == 'weapon+calibre'


def test_gun_build_without_a_matching_calibre_stays_unverified(geng, monkeypatch):
    assert _decide_gun(geng, monkeypatch, 2.5, 12.0, 'YY Tactical', '').evidence.get('verified') != 'weapon+calibre'
    assert _decide_gun(geng, monkeypatch, 2.5, 12.0, 'YY Tactical', '9x19').evidence.get('verified') != 'weapon+calibre'
    # a part that looks about as much like the tile as the gun does vetoes it
    assert _decide_gun(geng, monkeypatch, 2.5, 4.0, 'YY Tactical', '12g').evidence.get('verified') != 'weapon+calibre'
