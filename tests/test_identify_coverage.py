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
    # the same lead over a colour variant stays unsettled
    w = _w(eng, [(idof(21), 4.0), (idof(23), 8.5)])
    assert not eng._twins_settled_by_footprint(idof(21), [idof(21), idof(23)], tile, False, w)
    # too small a lead is never enough, even for a different shape
    w = _w(eng, [(idof(21), 4.0), (idof(22), 6.0)])
    assert not eng._twins_settled_by_footprint(idof(21), twins, tile, False, w)
    # one close colour variant among the rivals vetoes the pick
    w = _w(eng, [(idof(21), 4.0), (idof(22), 12.0), (idof(23), 8.5)])
    assert not eng._twins_settled_by_footprint(idof(21), [idof(21), idof(22), idof(23)], tile, False, w)
