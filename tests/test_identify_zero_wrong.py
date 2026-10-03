"""Zero-wrong rules: same-label parts vs a gun, calibre-verified guns, label twins that the
picture does not separate (docs/accuracy.md)."""
import json
from types import SimpleNamespace

import cv2
import pytest

from identify import catalog as C
from identify.config import EngineSettings
from identify.pipeline import Engine
from synth import make_icon


def idof(i):
    return f'{i:024x}'


TWINS = [  # (id, name, short, W, H, types)
    (idof(11), 'ZX-1 20ga shotgun', 'ZX-1', 4, 1, ['gun']),
    (idof(12), 'ZX-1 stock', 'ZX1', 4, 1, ['mods']),
    (idof(13), 'ZX-1 custom stock', 'ZX1 Custom', 4, 1, ['mods']),
    (idof(14), 'QP9 9x19 upper receiver', 'QP9', 2, 1, ['mods']),
    (idof(15), 'QP9 9x19 30-round magazine', 'QP9', 1, 2, ['mods']),
]


@pytest.fixture(scope='module')
def eng(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('twins')
    (tmp / 'tmpl').mkdir()
    prices = []
    for k, (iid, name, short, w, h, types) in enumerate(TWINS):
        cv2.imwrite(str(tmp / 'tmpl' / f'{iid}.png'), make_icon(800 + k, w, h))
        prices.append({'id': iid, 'name': name, 'shortName': short, 'width': w, 'height': h,
                       'backgroundColor': 'blue', 'types': types})
    (tmp / 'p.json').write_text(json.dumps({'items': prices}), encoding='utf-8')
    cat = C.build_catalog(str(tmp / 'p.json'), str(tmp / 'tmpl'), str(tmp / 'nc'), with_cache=False,
                          log=lambda *_: None)
    return Engine(EngineSettings(use_dino=False, use_ocr=False, accelerate=False), catalog=cat)


def _tile(W, H):
    return SimpleNamespace(W=W, H=H, clip='')


def _w(eng, ids, W=2, H=1):
    cands = [SimpleNamespace(row=eng._id_row[i], score=res, rot=0) for i, res in ids]
    return {'cands': cands, 'qe': None, 'sims': {}, 'tile': _tile(W, H), 'ocr_all': ['QP9']}


def test_calibre_read_makes_a_gun_label_authoritative_over_parts(eng):
    gun = eng._id_row[idof(11)]
    tile = _tile(4, 1)
    got = eng._label_authority(['ZX-1'], tile, True, only_weapons=True)
    assert got is not None and got[0] == gun
    w = {'ocr_all': ['ZX-1']}
    assert eng._calibre_names_gun('20g /5', gun, w, tile, False)       # "20g" is a cut "20ga"
    assert not eng._calibre_names_gun('9x19', gun, w, tile, False)     # another calibre
    assert not eng._calibre_names_gun('', gun, w, tile, False)


def test_calibre_text_cannot_vouch_for_a_gun_when_a_same_label_part_names_it_too(eng):
    # an MP5 upper receiver's art reads "9x19PARA" and its own name carries 9x19 too
    rec = eng._id_row[idof(14)]
    assert not eng._calibre_names_gun('9x19PARA', rec, {'ocr_all': ['QP9']}, _tile(2, 1), False)


def test_exact_label_shared_by_receiver_and_magazine_needs_a_clear_picture(eng):
    mag, rec = idof(15), idof(14)
    assert eng._unresolved_label_twins(_w(eng, [(mag, 5.3), (rec, 6.5)]), _tile(2, 1), False, mag, False)
    # a clear picture lead settles it
    assert not eng._unresolved_label_twins(_w(eng, [(rec, 4.0), (mag, 12.0)]), _tile(2, 1), False, rec, False)
    # no rival visual candidate: nothing to confuse it with
    assert not eng._unresolved_label_twins(_w(eng, [(rec, 4.0)]), _tile(2, 1), False, rec, False)
