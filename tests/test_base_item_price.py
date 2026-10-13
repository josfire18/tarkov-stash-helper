"""Price = the BASE item only.

A tile is identified as its base item (the magazine, not the rounds in it; the armour, not its
plates; the container, not what is inside; the gun, not what is built on it) and is priced with
that item's own trader / flea value.  Contents, plates and attachments are never added.
"""
import app
import liveadvice
import sellcalc

CTX = sellcalc.make_context({})


def _item(iid, name, trader, flea=0, types=()):
    sell = [{'vendor': {'name': 'Prapor'}, 'priceRUB': trader, 'currency': 'RUB'}]
    if flea:
        sell.append({'vendor': {'name': 'Flea Market'}, 'priceRUB': flea, 'currency': 'RUB'})
    return {'id': iid, 'name': name, 'basePrice': trader, 'lastLowPrice': flea, 'avg24hPrice': flea,
            'low24hPrice': flea, 'sellFor': sell, 'types': list(types)}


ITEMS = {i['id']: i for i in [
    _item('mag', 'STANAG 30-round magazine', 3000, 6000, ['mods']),
    _item('bp', '5.56x45 BP ammo', 900, 40000, ['ammo']),                   # what a loaded mag holds
    _item('armor', 'Zhuk-3 body armor', 25000, 70000, ['armor', 'wearable']),
    _item('plate', 'Granit 6B13 plate', 8000, 0, ['armorPlate', 'noFlea']),  # plates cannot be sold on the flea
    _item('bag', 'Scav backpack', 4000, 9000, ['backpack']),
    _item('gpu', 'Graphics card', 50000, 160000, ['barter']),               # what a filled bag holds
    _item('gun', 'Colt M4A1 5.56x45 assault rifle', 8278, 29999, ['gun', 'wearable']),
    _item('build', 'Colt M4A1 Default', 99999, 400000, ['preset']),         # a whole build: never the price
]}


def _det(iid, col=0, w=1, h=1):
    return {'item_id': iid, 'count': 1, 'fir': True, 'col': col, 'row': 0, 'panel': 0,
            'px': col * 60, 'py': 0, 'pw': 60 * w, 'ph': 60 * h, 'W': w, 'H': h, 'score': 99}


def _plan(iid):
    sell, keep = sellcalc.plan_entries([_det(iid)], ITEMS, {}, {}, CTX)
    assert len(sell) == 1 and not keep
    return sell[0]


def _unit(row):
    return row['flea_net'] if row['recommend'] == 'flea' else row['trader_price']


def test_loaded_magazine_is_priced_as_the_magazine():
    row = _plan('mag')
    assert row['matched_name'] == 'STANAG 30-round magazine'
    assert row['total'] == _unit(row) < 40000 * 30       # the 30 BP rounds inside add nothing
    assert row['trader_price'] == 3000 and 'base_gun' not in row


def test_plated_armour_is_priced_as_the_armour():
    row = _plan('armor')
    assert row['matched_name'] == 'Zhuk-3 body armor'
    assert row['trader_price'] == 25000
    assert row['total'] == _unit(row) < 25000 + 2 * 8000 + 70000   # plates are not added


def test_a_plate_the_flea_will_not_take_keeps_its_trader_price():
    row = _plan('plate')
    assert row['recommend'] == 'trader' and row['trader_price'] == 8000 and row['total'] == 8000
    assert row['reason'].lower().startswith('banned from flea')


def test_filled_container_is_priced_as_the_container():
    row = _plan('bag')
    assert row['matched_name'] == 'Scav backpack'
    assert row['trader_price'] == 4000
    assert row['total'] == _unit(row) < 4000 + 160000     # what is inside is not read


def test_built_gun_is_priced_as_the_base_gun_and_labelled():
    row = _plan('gun')
    assert row['matched_name'] == 'Colt M4A1 5.56x45 assault rifle'
    assert row['base_gun'] is True
    assert row['trader_price'] == 8278 and row['flea_ref'] == 29999
    assert row['total'] == _unit(row) < 99999           # never the preset's / a build's value


def test_only_guns_carry_the_base_gun_label():
    for iid in ('mag', 'armor', 'bag', 'plate'):
        assert 'base_gun' not in _plan(iid)


def test_base_guns_are_on_the_sell_list_presets_and_dogtags_are_not():
    assert app.skip_badge(ITEMS['gun'], 'weapon') is None
    assert app.skip_badge(ITEMS['build'], 'weapon') == 'GUN'
    for iid in ('mag', 'armor', 'plate', 'bag'):
        assert app.skip_badge(ITEMS[iid]) is None
    assert app.skip_badge({'name': 'Dogtag BEAR', 'types': ['barter', 'noFlea']}) == 'TAG'


def test_raid_grab_values_the_base_item_only():
    dets = [{'item_id': iid, 'role': 'loot', 'col': c, 'row': 0, 'W': 1, 'H': 1, 'panel': 0, 'count': 1,
             'uncertain': False, 'fir': None, 'category': None, 'px': c * 10, 'py': 0, 'pw': 10, 'ph': 10}
            for c, iid in enumerate(['mag', 'armor', 'bag', 'gun'])]
    out = liveadvice.compute_advice(dets, [], ITEMS, {}, [], {'live_min_value_per_slot': 0}, CTX,
                                    scene='raid_loot', in_raid=True)['grab']
    by = {g['item_id']: g for g in out}
    assert by['gun']['base_gun'] is True and by['gun']['value'] == liveadvice.unit_value(ITEMS['gun'], True, {}, CTX)
    assert by['gun']['value'] < 99999
    assert by['mag']['value'] == liveadvice.unit_value(ITEMS['mag'], True, {}, CTX)
    assert by['armor']['value'] == liveadvice.unit_value(ITEMS['armor'], True, {}, CTX)
    assert by['bag']['value'] == liveadvice.unit_value(ITEMS['bag'], True, {}, CTX)
    assert all('base_gun' not in by[i] for i in ('mag', 'armor', 'bag'))
