"""identify.eft_hash: the game's icon-cache hash, checked against index.json entries of a real
EFT 1.2 icon cache whose renders were label-certified (data/learned_icons.json)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify import eft_hash as E  # noqa: E402

# (template id, index.json key, salt) - plain items, ammo, an empty magazine, a hinged NVG
KNOWN = [
    ('5c1bc5612e221602b5429350', -1754071894, 0),      # Zenit RK-1 tactical foregrip
    ('5888961624597754281f93f3', 399105918, 0),        # Harris S-BRM bipod
    ('590a3efd86f77437d351a25b', 1482547642, 0),       # Gas analyzer
    ('590c5d4b86f774784e1b9c45', -455593815, 0),       # Iskra ration pack
    ('5c0530ee86f774697952d952', 735214668, 0),        # LEDX
    ('56742c284bdc2d98058b456d', 726417346, 0),        # Crickent lighter
    ('5d0375ff86f774186372f685', -128786435, 0),       # Military cable
    ('5d6e68a8a4b9360b6c0d54e2', 1682625896, E.SALT_AMMO),        # 12/70 AP-20 slug
    ('5efb0da7a29a85116f6ea05f', -304564182, E.SALT_AMMO),        # 9x19 PBP
    ('65118f531b90b4fc77015083', -1927576445, E.SALT_MAG_EMPTY),  # VSS/VAL 30-round magazine, empty
    ('5e21a3c67e40bd02257a008a', -370236277, E.SALT_MAG_EMPTY),   # AK PMAG 30 GEN M3, empty
    ('5c0558060db834001b735271', 420777264, E.SALT_HINGED_ALT),   # GPNVG-18
]


@pytest.mark.parametrize('iid,h,salt', KNOWN)
def test_known_pairs(iid, h, salt):
    assert E.item_hash(iid, salt) == h


def test_signed_int32_and_bad_ids():
    for iid, _, _ in KNOWN:
        v = E.template_hash(iid)
        assert -2 ** 31 <= v < 2 ** 31
    assert E.template_hash('not-an-id') is None
    assert E.template_hash('5C1BC5612E221602B5429350') is None     # ids are lower-case hex
    assert E.template_hash('707265736574') is None


def test_exact_ids_salts_gated_by_category(tmp_path):
    index = {h: n for n, (_, h, _) in enumerate(KNOWN, start=1)}
    cats = {'5d6e68a8a4b9360b6c0d54e2': 'ammo', '5efb0da7a29a85116f6ea05f': 'ammo',
            '65118f531b90b4fc77015083': 'mod', '5e21a3c67e40bd02257a008a': 'mod',
            '5c0558060db834001b735271': 'gear'}
    items = {iid: cats.get(iid, 'barter') for iid, _, _ in KNOWN}
    ex = E.exact_ids(index, items)
    assert len(ex) == len(KNOWN)
    for n, (iid, _, salt) in enumerate(KNOWN, start=1):
        assert ex[f'{n}.png'] == (iid, salt)
    # an ammo salt never applies to a non-ammo item
    items['5efb0da7a29a85116f6ea05f'] = 'barter'
    assert '9.png' not in E.exact_ids(index, items)
    # index.json round trip
    (tmp_path / 'index.json').write_text(json.dumps({str(h): n for h, n in index.items()}))
    assert E.read_index(str(tmp_path)) == index
    assert E.read_index(str(tmp_path / 'missing')) == {}


def test_validate_gate():
    exact = {f'{n}.png': (f'{n:024x}', 0) for n in range(100)}
    good = {f'{n}.png': f'{n:024x}' for n in range(100)}
    st = E.validate(exact, good)
    assert st['enabled'] and st['overlap'] == 100 and st['agreement'] == 1.0
    assert E.gate_open(st)
    bad = dict(good, **{f'{n}.png': 'f' * 24 for n in range(10)})
    assert not E.validate(exact, bad)['enabled']                # 90 % is not enough
    assert not E.validate(exact, {k: v for k, v in list(good.items())[:15]})['enabled']   # too little evidence
    assert not E.validate({}, good)['enabled']                  # a formula that names nothing
    assert not E.gate_open(dict(st, provider='something-else'))


def test_status_roundtrip(tmp_path):
    st = E.validate({}, {})
    E.save_status(str(tmp_path), st)
    back = E.load_status(str(tmp_path))
    assert back['provider'] == E.PROVIDER and back['enabled'] is False and 'checked_at' in back
