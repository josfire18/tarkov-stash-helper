"""json.tarkov.dev data source: conversion, ETag caching, source order, refusal of bad payloads,
the background refresher and the new-items -> catalog-rebuild trigger.

Fixtures in tests/fixtures/tarkovdev/ are trimmed copies of the real documents (a handful of
items, two tasks, two hideout stations).  Nothing here touches the network or the real data/."""
import copy
import json
import os
import time

import pytest
import requests

import app
import sellcalc
import tarkovdata as td

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'tarkovdev')
M4A1 = '5447a9cd4bdc2dbd208b4567'
SUDAK = '6a8c4c7999baf8bd5802f7fe'
PEACEKEEPER = '5935c25fb3acc3127c3d8cd9'


def fixture(name):
    with open(os.path.join(FIX, name + '.json'), encoding='utf-8') as f:
        return json.load(f)


def all_docs():
    return {n: fixture(n) for n in ('items', 'items_en', 'tasks', 'tasks_en', 'hideout', 'hideout_en',
                                    'traders', 'traders_en')}


class Resp:
    def __init__(self, status, body=None, etag=None):
        self.status_code, self._body = status, body
        self.headers = {'ETag': etag} if etag else {}

    def json(self):
        if self._body is None:
            raise ValueError('no body')
        return self._body


class FakeNet:
    """Stands in for json.tarkov.dev: serves ``docs`` with ETags, honours If-None-Match,
    records every request, and can be told to fail."""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else all_docs()
        self.etag_n = {}
        self.etags = {}
        self.requests = []          # (doc, If-None-Match or None)
        self.down = False
        self.status = {}            # doc -> forced HTTP status

    def set_doc(self, name, body):
        self.docs[name] = body
        self.etags[name] = f'W/"{name}-{self.etag_n.get(name, 0) + 1}"'
        self.etag_n[name] = self.etag_n.get(name, 0) + 1

    def __call__(self, url, headers, timeout):
        doc = url.rsplit('/', 1)[1]
        self.requests.append((doc, headers.get('If-None-Match')))
        if self.down:
            raise requests.ConnectionError('network down')
        if doc in self.status:
            return Resp(self.status[doc])
        etag = self.etags.setdefault(doc, f'W/"{doc}-0"')
        if headers.get('If-None-Match') == etag:
            return Resp(304)
        return Resp(200, copy.deepcopy(self.docs[doc]), etag)

    def fetched(self, doc):
        return [r for r in self.requests if r[0] == doc]


@pytest.fixture
def net(monkeypatch):
    n = FakeNet()
    monkeypatch.setattr(td, '_http_get', n)
    # the fixtures are a handful of records: scale the "looks complete" floors down
    monkeypatch.setattr(td, 'MIN_ITEMS', 3)
    monkeypatch.setattr(td, 'MIN_TASKS', 1)
    monkeypatch.setattr(td, 'MIN_STATIONS', 1)
    return n


@pytest.fixture
def paths(tmp_path, monkeypatch):
    d = tmp_path / 'data'
    d.mkdir()
    monkeypatch.setattr(app, 'PRICES_PATH', str(d / 'prices_cache.json'))
    monkeypatch.setattr(app, 'TASKS_CACHE_PATH', str(d / 'tasks_cache.json'))
    monkeypatch.setattr(app, 'META_PATH', str(d / 'tarkovdev_meta.json'))
    monkeypatch.setattr(app, '_queue_catalog_update', lambda: None)     # no catalog in unit tests
    for kind in ('prices', 'tasks'):
        app._refresh_state[kind].update({'attempt': 0, 'error': None})
    return d


def read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def trader_names():
    return td.convert_traders(fixture('traders'), fixture('traders_en'))[0]


def convert():
    items, skipped = td.convert_items(fixture('items'), fixture('items_en'), trader_names())
    return {i['id']: i for i in items}, skipped


def test_names_are_translation_keys_resolved_through_items_en():
    raw = fixture('items')['data']['items'][M4A1]
    assert raw['name'] == f'{M4A1} Name'                                  # the JSON carries only a key
    items, skipped = convert()
    assert skipped == 0
    assert items[M4A1]['name'] == 'Colt M4A1 5.56x45 assault rifle'
    assert items[M4A1]['shortName'] == 'M4A1'


def test_an_item_whose_name_cannot_be_resolved_is_skipped_not_guessed():
    en = fixture('items_en')
    del en['data'][f'{M4A1} Name']
    items, skipped = td.convert_items(fixture('items'), en, trader_names())
    assert skipped == 1 and M4A1 not in {i['id'] for i in items}


def test_record_has_the_keys_the_graphql_query_used():
    items, _ = convert()
    assert set(items[M4A1]) == {'id', 'name', 'shortName', 'basePrice', 'avg24hPrice', 'low24hPrice',
                                'lastLowPrice', 'iconLink', 'width', 'height', 'backgroundColor',
                                'gridImageLink', 'baseImageLink', 'types', 'sellFor'}
    assert items[M4A1]['baseImageLink'].endswith(f'{M4A1}-base-image.webp')


def test_sell_to_trader_becomes_sellfor_with_vendor_names_in_tarkov_dev_order():
    sf = convert()[0][M4A1]['sellFor']
    assert [s['vendor']['name'] for s in sf] == ['Prapor', 'Fence', 'Skier', 'Peacekeeper', 'Mechanic', 'Ref',
                                                'Flea Market']
    pk = next(s for s in sf if s['vendor']['name'] == 'Peacekeeper')
    assert (pk['currency'], pk['price'], pk['priceRUB']) == ('USD', 54, 6622)      # priceRUB is what sellcalc reads


def test_flea_entry_is_the_lowest_current_offer_for_flea_legal_items_only():
    items, _ = convert()
    flea = [s for s in items[M4A1]['sellFor'] if s['vendor']['name'] == 'Flea Market']
    assert len(flea) == 1 and flea[0]['priceRUB'] == items[M4A1]['lastLowPrice'] > 0
    assert flea[0]['currency'] == 'RUB'
    assert 'noFlea' in items[SUDAK]['types']
    assert not [s for s in items[SUDAK]['sellFor'] if s['vendor']['name'] == 'Flea Market']


def test_new_item_from_the_json_is_present_with_prices_the_app_reads():
    sudak = convert()[0][SUDAK]
    assert sudak['name'] == 'Sudak-Tudak marine repair kit'
    name, price = sellcalc.best_trader_price(sudak)
    assert (name, price) == ('Peacekeeper', 36000)


def test_converted_records_work_with_the_sell_recommendation():
    m4 = convert()[0]['5449016a4bdc2d6f028b456f']          # roubles: any record must flow through sellcalc
    sellcalc.sell_recommendation(m4)


def test_rules_carry_the_live_flea_rules_and_trader_pay_rates():
    names, trader_rules = td.convert_traders(fixture('traders'), fixture('traders_en'))
    rules = td.convert_rules(fixture('items'), trader_rules)
    flea = rules['flea']
    assert flea['sellOfferFeeRate'] == 0.05 and flea['sellRequirementFeeRate'] == 0.05
    assert flea['foundInRaidRequired'] is False and flea['minPlayerLevel'] == 15
    assert flea['reputationLevels'][0]['offers'] == 1
    assert rules['traders']['Ref']['pay_rates'] == {1: 0.4, 2: 0.45, 3: 0.45, 4: 0.5}
    assert 'Lightkeeper' not in rules['traders']          # placeholder traders (pay rate ~0) carry no rates


def test_live_fir_rule_drives_the_flea_block_when_the_setting_is_null():
    rules = td.convert_rules(fixture('items'), {})
    item = {'types': []}
    # setting null -> follow tarkov.dev: FiR not required right now, a non-FiR copy may be listed
    assert sellcalc.flea_block_reason(item, False, {'flea_requires_fir': None}, rules) is None
    assert sellcalc.make_context({'flea_requires_fir': None}, rules)['fir_required'] is False
    # an explicit user setting wins over the live rule
    assert sellcalc.flea_block_reason(item, False, {'flea_requires_fir': True}, rules) == 'not FiR, cannot list on flea'
    # tarkov.dev turning the rule back on is picked up without touching the setting
    on = copy.deepcopy(rules)
    on['flea']['foundInRaidRequired'] = True
    assert sellcalc.flea_block_reason(item, False, {'flea_requires_fir': None}, on) == 'not FiR, cannot list on flea'
    # no rules at all: the documented default (FiR required)
    assert sellcalc.flea_block_reason(item, False, {'flea_requires_fir': None}, None) == 'not FiR, cannot list on flea'


def test_tasks_convert_to_the_old_shape_with_first_item_fir_and_trader_name():
    names = {'590c695186f7741e566b64a2': ('Augmentin antibiotic pills', 'Augmentin')}
    tasks = {t['id']: t for t in td.convert_tasks(fixture('tasks'), fixture('tasks_en'), trader_names(), names)}
    t = tasks['657315ddab5a49b71f098853']
    assert (t['name'], t['trader'], t['minPlayerLevel']) == ('First in Line', {'name': 'Therapist'}, 1)
    visit, give = t['objectives']
    assert visit == {'id': '65732ac3c67dcd96adffa3c7', 'type': 'visit'}
    assert give['type'] == 'giveItem' and give['count'] == 3 and give['foundInRaid'] is True
    # a multi-item ("any of these") objective keeps its first item, as the deprecated GraphQL field did
    assert give['item'] == {'id': '590c695186f7741e566b64a2', 'name': 'Augmentin antibiotic pills',
                            'shortName': 'Augmentin'}
    kappa = tasks['5979f9ba86f7740f6c3fe9f2']
    assert kappa['kappaRequired'] is True and kappa['name'] == 'Chemical - Part 1'
    assert not [o for o in kappa['objectives'] if o['type'] == 'findQuestItem' and 'item' in o]   # quest items are not stash items


def test_hideout_stations_convert_with_names_and_item_requirements():
    names = {'5449016a4bdc2d6f028b456f': ('Roubles', 'RUB')}
    st = {s['id']: s for s in td.convert_hideout(fixture('hideout'), fixture('hideout_en'), names)}
    med = st['5d484fcd654e7668ec2ec322']
    assert med['name'] == 'Medstation'
    lv1 = med['levels'][0]
    assert lv1['id'] == '5d484fcd654e7668ec2ec322-1' and lv1['level'] == 1
    assert {'count': 50000, 'item': {'id': '5449016a4bdc2d6f028b456f', 'name': 'Roubles', 'shortName': 'RUB'}} \
        in lv1['itemRequirements']


def test_converted_tasks_feed_compute_tasks_view():
    names = td.item_names_from_prices({'items': list(convert()[0].values())})
    cache = {'timestamp': time.time(),
             'tasks': td.convert_tasks(fixture('tasks'), fixture('tasks_en'), trader_names(), names),
             'hideoutStations': td.convert_hideout(fixture('hideout'), fixture('hideout_en'), names)}
    view = app.compute_tasks_view(cache, {'completed_tasks': [], 'completed_hideout': [], 'have': {}})
    assert view['tasks'] and view['stations']
    assert any(r['item_id'] == '590c695186f7741e566b64a2' and r['total_needed'] == 3 and r['fir_needed'] == 3
               for r in view['aggregate'])


# ---------------------------------------------------------------------------
# Refresh: ETags, validation, atomic writes
# ---------------------------------------------------------------------------

def test_first_refresh_writes_the_cache_and_the_etags(net, paths):
    res = app.refresh_prices()
    assert res['status'] == 'updated' and res['source'] == 'json.tarkov.dev'
    cache = read(app.PRICES_PATH)
    assert cache['source'] == 'json.tarkov.dev' and len(cache['items']) == 5
    assert cache['rules']['flea']['foundInRaidRequired'] is False
    meta = read(app.META_PATH)
    assert meta['etags']['items'] == net.etags['items'] and meta['etags']['items_en'] == net.etags['items_en']
    assert meta['sources']['prices'] == 'json.tarkov.dev'
    assert all(etag is None for _, etag in net.requests)             # nothing conditional without a cache
    assert [f for f in os.listdir(paths) if f.endswith('.tmp')] == []


def test_second_refresh_is_a_304_and_rewrites_nothing(net, paths):
    app.refresh_prices()
    before = open(app.PRICES_PATH, 'rb').read()
    mtime = os.stat(app.PRICES_PATH).st_mtime_ns
    net.requests.clear()
    res = app.refresh_prices()
    assert res['status'] == 'unchanged' and res['new_ids'] == []
    assert open(app.PRICES_PATH, 'rb').read() == before and os.stat(app.PRICES_PATH).st_mtime_ns == mtime
    sent = {doc: etag for doc, etag in net.requests}
    assert sent['items'] == net.etags['items'] and sent['items_en'] == net.etags['items_en']     # If-None-Match sent
    assert len(net.fetched('items')) == 1                                                       # no re-download
    meta = read(app.META_PATH)
    assert meta['checked']['prices'] >= os.stat(app.PRICES_PATH).st_mtime - 5                  # "checked" advanced


def test_changed_items_with_unchanged_translations_refetches_the_translations(net, paths):
    app.refresh_prices()
    body = fixture('items')
    body['data']['items'][M4A1]['lastLowPrice'] = 12345
    net.set_doc('items', body)
    net.requests.clear()
    res = app.refresh_prices()
    assert res['status'] == 'updated'
    assert [i for i in res['cache']['items'] if i['id'] == M4A1][0]['lastLowPrice'] == 12345
    # items_en answered 304, but a pair cannot be converted from half: it was fetched again, unconditionally
    assert [e for _, e in net.fetched('items_en')] == [net.etags['items_en'], None]


def test_a_cache_on_disk_is_required_before_anything_is_conditional(net, paths):
    app.refresh_prices()
    os.remove(app.PRICES_PATH)                                      # etags still in the meta
    net.requests.clear()
    res = app.refresh_prices()
    assert res['status'] == 'updated' and os.path.exists(app.PRICES_PATH)
    assert all(etag is None for doc, etag in net.requests if doc.startswith('items'))


def test_bad_payload_never_replaces_a_good_cache(net, paths):
    app.refresh_prices()
    good = open(app.PRICES_PATH, 'rb').read()
    etags = read(app.META_PATH)['etags']
    bad = fixture('items')
    bad['data']['items'] = {}                                        # an empty / emptied data set
    net.set_doc('items', bad)
    with pytest.raises(app.PriceFetchError):
        app.refresh_prices()                                         # (GraphQL is blocked by the fixture: down too)
    assert open(app.PRICES_PATH, 'rb').read() == good
    assert read(app.META_PATH)['etags'] == etags                     # the bad doc's ETag was not remembered
    assert 'json.tarkov.dev' in read(app.META_PATH)['errors']['prices']


def test_partial_payload_is_refused_even_if_above_the_floor(net, paths):
    app.refresh_prices()
    good = open(app.PRICES_PATH, 'rb').read()
    part = fixture('items')
    part['data']['items'] = dict(list(part['data']['items'].items())[:3])     # 3 of 5 = 60 % < 80 %
    net.set_doc('items', part)
    with pytest.raises(app.PriceFetchError, match='partial'):
        app.refresh_prices()
    assert open(app.PRICES_PATH, 'rb').read() == good


def test_untranslated_payload_is_refused(net, paths):
    bad = fixture('items_en')
    bad['data'] = {}
    net.set_doc('items_en', bad)
    with pytest.raises(app.PriceFetchError):
        app.refresh_prices()
    assert not os.path.exists(app.PRICES_PATH)


def test_the_real_floors_reject_a_tiny_data_set(monkeypatch):
    items, _ = convert()
    with pytest.raises(td.SourceError):
        td.validate_items(list(items.values()))                      # 5 items, floor 1000


def test_write_is_atomic_a_failed_replace_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    p = tmp_path / 'c.json'
    td.write_json_atomic(str(p), {'a': 1})

    def boom(a, b):
        raise OSError('disk full')
    monkeypatch.setattr(os, 'replace', boom)
    with pytest.raises(OSError):
        td.write_json_atomic(str(p), {'a': 2})
    monkeypatch.undo()
    assert json.loads(p.read_text(encoding='utf-8')) == {'a': 1}
    assert [f.name for f in tmp_path.iterdir()] == ['c.json']


def test_save_json_is_atomic_too(tmp_path):
    app.save_json(str(tmp_path / 'x.json'), {'k': [1, 2]})
    assert read(str(tmp_path / 'x.json')) == {'k': [1, 2]}
    assert [f.name for f in tmp_path.iterdir()] == ['x.json']


# ---------------------------------------------------------------------------
# Source order: json.tarkov.dev, then GraphQL, then the last good cache
# ---------------------------------------------------------------------------

def graphql_items(n=1200):
    return [{'id': f'g{k}', 'name': f'G{k}', 'shortName': f'G{k}', 'width': 1, 'height': 1,
             'baseImageLink': 'https://img/g', 'types': [],
             'sellFor': [{'vendor': {'name': 'Prapor'}, 'priceRUB': 5}]} for k in range(n)]


class GqlResp:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


@pytest.fixture
def graphql(monkeypatch):
    calls = []
    state = {'items': graphql_items(), 'down': False}

    def post(url, json=None, timeout=None):
        calls.append(json['query'][:30])
        if state['down']:
            return GqlResp({'errors': [{'message': 'GraphQL server unavailable'}]})
        if 'fleaMarket' in json['query']:
            return GqlResp({'data': {'fleaMarket': {'foundInRaidRequired': True}, 'traders': []}})
        return GqlResp({'data': {'items': state['items']}})
    monkeypatch.setattr(app.http_requests, 'post', post)
    state['calls'] = calls
    return state


def test_json_source_is_tried_first_and_graphql_is_not_touched(net, paths, graphql):
    app.refresh_prices()
    assert graphql['calls'] == []


def test_json_failure_falls_back_to_graphql(net, paths, graphql):
    net.down = True
    res = app.refresh_prices()
    assert res['source'] == 'graphql' and read(app.PRICES_PATH)['source'] == 'graphql'
    assert len(read(app.PRICES_PATH)['items']) == 1200
    assert read(app.PRICES_PATH)['rules']['flea']['foundInRaidRequired'] is True
    meta = read(app.META_PATH)
    assert meta['sources']['prices'] == 'graphql' and 'items' not in meta['etags']


def test_http_422_from_json_also_falls_back(net, paths, graphql):
    net.status['items'] = 503
    assert app.refresh_prices()['source'] == 'graphql'


def test_both_sources_down_raises_and_the_stale_cache_is_served(net, paths, graphql, monkeypatch):
    app.refresh_prices()                                             # json: good cache (5 items)
    # age the cache past the TTL
    c = read(app.PRICES_PATH)
    c['timestamp'] = time.time() - 10 ** 6
    app.save_json(app.PRICES_PATH, c)
    meta = read(app.META_PATH)
    meta['checked'] = {}
    app.save_json(app.META_PATH, meta)
    net.down = True
    graphql['down'] = True
    out = app.get_prices()
    assert len(out['items']) == 5 and 'GraphQL' in out['stale_error'] and 'json.tarkov.dev' in out['stale_error']
    assert len(read(app.PRICES_PATH)['items']) == 5                  # untouched
    # a second scan right after does not wait on the dead network again
    net.requests.clear()
    graphql['calls'].clear()
    app.get_prices()
    assert net.requests == [] and graphql['calls'] == []


def test_graphql_fallback_is_validated_too(net, paths, graphql, monkeypatch):
    monkeypatch.setattr(td, 'MIN_ITEMS', 1000)
    net.down = True
    graphql['items'] = graphql_items(10)
    with pytest.raises(app.PriceFetchError, match='refused'):
        app.refresh_prices()
    assert not os.path.exists(app.PRICES_PATH)


def test_json_comes_back_after_a_graphql_period(net, paths, graphql, monkeypatch):
    net.down = True
    app.refresh_prices()                                             # graphql, 1200 items
    net.down = False
    monkeypatch.setattr(td, 'SHRINK_LIMIT', 0)                       # the 5-item fixture is "smaller" than 1200
    res = app.refresh_prices()
    assert res['source'] == 'json.tarkov.dev' and read(app.PRICES_PATH)['source'] == 'json.tarkov.dev'


# ---------------------------------------------------------------------------
# Tasks / hideout
# ---------------------------------------------------------------------------

def test_tasks_refresh_writes_tasks_and_hideout_and_is_conditional_afterwards(net, paths):
    app.refresh_prices()
    res = app.refresh_tasks()
    assert res['status'] == 'updated'
    cache = read(app.TASKS_CACHE_PATH)
    assert {t['name'] for t in cache['tasks']} == {'First in Line', 'Chemical - Part 1'}
    assert {s['name'] for s in cache['hideoutStations']} == {'Medstation', 'Library'}
    names = {i['id']: i['name'] for t in cache['tasks'] for o in t['objectives'] if 'item' in o for i in [o['item']]}
    assert names['590c695186f7741e566b64a2'] == 'Augmentin antibiotic pills'        # named from the prices cache
    net.requests.clear()
    assert app.refresh_tasks()['status'] == 'unchanged'
    assert len(net.fetched('tasks')) == 1 and net.fetched('tasks')[0][1] == net.etags['tasks']


def test_bad_task_payload_keeps_the_good_cache(net, paths):
    app.refresh_prices()
    app.refresh_tasks()
    good = open(app.TASKS_CACHE_PATH, 'rb').read()
    bad = fixture('tasks')
    bad['data']['tasks'] = {}
    net.set_doc('tasks', bad)
    out = app.refresh_tasks()                                        # hideout half still fine and unchanged
    assert out['status'] == 'unchanged'
    assert open(app.TASKS_CACHE_PATH, 'rb').read() == good
    assert 'tasks:' in read(app.META_PATH)['errors']['tasks_partial']


def test_task_outage_falls_back_to_graphql_then_to_the_cache(net, paths, monkeypatch):
    app.refresh_prices()
    app.refresh_tasks()
    old = read(app.TASKS_CACHE_PATH)
    old['timestamp'] = time.time() - 10 ** 6
    app.save_json(app.TASKS_CACHE_PATH, old)
    meta = read(app.META_PATH)
    meta['checked'] = {}
    app.save_json(app.META_PATH, meta)
    net.down = True
    monkeypatch.setattr(app.http_requests, 'post',
                        lambda *a, **k: GqlResp({'errors': [{'message': 'GraphQL server unavailable'}]}))
    got = app.get_tasks()
    assert {t['name'] for t in got['tasks']} == {'First in Line', 'Chemical - Part 1'}      # stale but served
    assert app._refresh_state['tasks']['error']


# ---------------------------------------------------------------------------
# New items -> catalog update
# ---------------------------------------------------------------------------

def test_new_items_are_reported_recorded_and_trigger_the_catalog_update(net, paths, monkeypatch):
    queued = []
    monkeypatch.setattr(app, '_queue_catalog_update', lambda: queued.append(1))
    first = app.refresh_prices()
    assert first['new_ids'] == [] and queued == []                    # first fetch: nothing to compare with
    body = fixture('items')
    new_id = '6a8c4c7999baf8bd5802fff1'
    body['data']['items'][new_id] = {**body['data']['items'][SUDAK], 'id': new_id,
                                     'name': f'{new_id} Name', 'shortName': f'{new_id} ShortName'}
    en = fixture('items_en')
    en['data'][f'{new_id} Name'], en['data'][f'{new_id} ShortName'] = 'BD laptop', 'BD'
    net.set_doc('items', body)
    net.set_doc('items_en', en)
    res = app.refresh_prices()
    assert res['new_ids'] == [new_id] and queued == [1]
    assert read(app.META_PATH)['catalog_pending'] == [new_id]
    assert 'BD laptop' in {i['name'] for i in read(app.PRICES_PATH)['items']}


def test_unchanged_item_set_triggers_nothing(net, paths, monkeypatch):
    queued = []
    monkeypatch.setattr(app, '_queue_catalog_update', lambda: queued.append(1))
    app.refresh_prices()
    body = fixture('items')
    body['data']['items'][M4A1]['avg24hPrice'] = 1                    # a price change only
    net.set_doc('items', body)
    assert app.refresh_prices()['new_ids'] == [] and queued == []


def test_queue_only_acts_on_installs_that_already_have_a_catalog(monkeypatch):
    started = []
    monkeypatch.setattr(app, 'start_icon_db_build', lambda reason: started.append(reason))
    monkeypatch.setattr(app, 'catalog_summary', lambda: None)
    app._queue_catalog_update()
    assert started == []
    monkeypatch.setattr(app, 'catalog_summary', lambda: {'items': 5000})
    app._queue_catalog_update()
    assert started == ['new items']


@pytest.fixture
def build_mocks(monkeypatch, paths):
    log = []
    monkeypatch.setattr(app, 'get_prices', lambda: {'items': []})
    monkeypatch.setattr(app, 'download_missing_base_images',
                        lambda prices, progress_cb=None: (log.append('images') or (3, 0)))
    import identify.catalog as cat
    monkeypatch.setattr(cat, 'load_catalog', lambda force_rebuild=False, **k: log.append(('catalog', force_rebuild)))
    monkeypatch.setattr(app, '_warm_v2_engine', lambda: log.append('swap'))
    monkeypatch.setattr(app, '_startup_done', type('E', (), {'wait': lambda self, t=None: True})())
    app._index_build_state.update({'running': True, 'phase': 'images', 'done': 0, 'total': 0, 'error': None})
    app._catalog_rerun.clear()
    app._scan_state['running'] = False
    yield log
    app._scan_state['running'] = False
    app._index_build_state['running'] = False


def test_catalog_build_downloads_rebuilds_swaps_and_clears_the_pending_ids(build_mocks):
    app.save_json(app.META_PATH, {**app._load_meta(), 'catalog_pending': ['a', 'b']})
    app.run_icon_db_build('new items')
    assert build_mocks == ['images', ('catalog', True), 'swap']
    assert app._load_meta()['catalog_pending'] == []
    assert app._index_build_state['running'] is False and app._index_build_state['error'] is None


def test_failed_image_downloads_keep_the_ids_pending(build_mocks, monkeypatch):
    app.save_json(app.META_PATH, {**app._load_meta(), 'catalog_pending': ['a']})
    monkeypatch.setattr(app, 'download_missing_base_images', lambda prices, progress_cb=None: (0, 1))
    app.run_icon_db_build('new items')
    assert app._load_meta()['catalog_pending'] == ['a']


def test_engine_swap_waits_for_a_running_scan(build_mocks, monkeypatch):
    app._scan_state['running'] = True
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            app._scan_state['running'] = False          # the scan finishes while the build waits
    monkeypatch.setattr(app.time, 'sleep', fake_sleep)
    app.run_icon_db_build('new items')
    assert len(sleeps) == 3
    assert build_mocks[-1] == 'swap'                      # swapped only after the scan was done


def test_items_changing_during_a_build_run_it_once_more(build_mocks, monkeypatch):
    runs = []

    def images(prices, progress_cb=None):
        runs.append(1)
        if len(runs) == 1:
            app._catalog_rerun.set()                    # a refresh found new items mid-build
        return 0, 0
    monkeypatch.setattr(app, 'download_missing_base_images', images)
    app.run_icon_db_build('new items')
    assert len(runs) == 2


def test_start_while_running_requests_a_rerun_instead_of_a_second_build(monkeypatch):
    app._index_build_state['running'] = True
    app._catalog_rerun.clear()
    try:
        assert app.start_icon_db_build('new items') is False
        assert app._catalog_rerun.is_set()
    finally:
        app._index_build_state['running'] = False
        app._catalog_rerun.clear()


def test_catalog_signature_ignores_prices_but_not_new_items(tmp_path):
    from identify.catalog import source_signature
    items = [{'id': 'a', 'name': 'A', 'shortName': 'A', 'width': 1, 'height': 1, 'backgroundColor': 'blue',
              'types': ['barter'], 'avg24hPrice': 100}]
    p = tmp_path / 'p.json'
    p.write_text(json.dumps({'items': items}), encoding='utf-8')
    base = source_signature(str(p), str(tmp_path))
    items[0]['avg24hPrice'] = 999
    p.write_text(json.dumps({'items': items, 'timestamp': 5}), encoding='utf-8')
    assert source_signature(str(p), str(tmp_path)) == base            # hourly price churn: no catalog rebuild
    items.append({'id': 'b', 'name': 'B', 'shortName': 'B', 'width': 1, 'height': 1, 'backgroundColor': 'blue',
                  'types': []})
    p.write_text(json.dumps({'items': items}), encoding='utf-8')
    assert source_signature(str(p), str(tmp_path)) != base


# ---------------------------------------------------------------------------
# Background refresher and status
# ---------------------------------------------------------------------------

@pytest.fixture
def cycle(paths, monkeypatch):
    ran = []
    monkeypatch.setattr(app, 'refresh_prices', lambda force=False: ran.append('prices'))
    monkeypatch.setattr(app, 'refresh_tasks', lambda force=False: ran.append('tasks'))
    monkeypatch.setattr(app, '_index_build_state', {**app._index_build_state, 'running': False})
    return ran


def put(path, ts, **extra):
    app.save_json(path, {'timestamp': ts, **extra})


def test_refresher_refreshes_on_startup_when_the_caches_are_older_than_their_intervals(cycle):
    now = time.time()
    put(app.PRICES_PATH, now - app.PRICE_REFRESH_INTERVAL - 5, items=[{'id': 'x'}])
    put(app.TASKS_CACHE_PATH, now - 10, tasks=[{'id': 'x'}])
    assert app.refresh_cycle(now) == ['prices']                       # prices old, tasks fresh


def test_refresher_does_nothing_while_everything_is_fresh(cycle):
    now = time.time()
    put(app.PRICES_PATH, now - 60, items=[{'id': 'x'}])
    put(app.TASKS_CACHE_PATH, now - 60, tasks=[{'id': 'x'}])
    assert app.refresh_cycle(now) == []


def test_refresher_counts_a_recent_304_check_as_fresh(cycle):
    now = time.time()
    put(app.PRICES_PATH, now - 5 * 3600, items=[{'id': 'x'}])         # file is old, but was re-checked a minute ago
    put(app.TASKS_CACHE_PATH, now - 60, tasks=[{'id': 'x'}])
    app.save_json(app.META_PATH, {**app._load_meta(), 'checked': {'prices': now - 60}})
    assert app.refresh_cycle(now) == []


def test_refresher_with_no_cache_fetches_both_and_prices_first(cycle):
    assert app.refresh_cycle() == ['prices', 'tasks']


def test_refresher_backs_off_after_a_failed_attempt_then_retries(cycle, monkeypatch):
    now = time.time()
    put(app.PRICES_PATH, now - 2 * app.PRICE_REFRESH_INTERVAL, items=[{'id': 'x'}])
    put(app.TASKS_CACHE_PATH, now - 60, tasks=[{'id': 'x'}])
    app.save_json(app.META_PATH, {**app._load_meta(), 'attempted': {'prices': now - 30}})   # failed 30 s ago
    assert app.refresh_cycle(now) == []
    assert app.refresh_cycle(now + app.RETRY_BACKOFF + 1) == ['prices']


def test_refresher_survives_a_failing_refresh(paths, monkeypatch):
    def boom(force=False):
        raise RuntimeError('everything is down')
    monkeypatch.setattr(app, 'refresh_prices', boom)
    monkeypatch.setattr(app, 'refresh_tasks', lambda force=False: None)
    assert app.refresh_cycle() == ['tasks']                           # logged, not raised


def test_refresher_resumes_unfinished_catalog_work_from_a_previous_run(cycle, monkeypatch):
    now = time.time()
    put(app.PRICES_PATH, now - 60, items=[{'id': 'x'}])
    put(app.TASKS_CACHE_PATH, now - 60, tasks=[{'id': 'x'}])
    app.save_json(app.META_PATH, {**app._load_meta(), 'catalog_pending': ['n1']})
    queued = []
    monkeypatch.setattr(app, '_queue_catalog_update', lambda: queued.append(1))
    app.refresh_cycle(now)
    assert queued == [1]


def test_start_refresher_runs_the_first_cycle_at_once_and_is_idempotent(monkeypatch):
    import threading
    hit = threading.Event()
    monkeypatch.setattr(app, 'refresh_cycle', lambda now=None: hit.set())
    old = app._refresher['thread']
    app._refresher['thread'] = None
    try:
        t = app.start_refresher()
        assert hit.wait(5)
        assert app.start_refresher() is t
    finally:
        app._refresher['stop'].set()
        t.join(5)
        app._refresher['thread'] = old
        app._refresher['stop'].clear()


def test_status_endpoint_reports_source_age_count_and_last_error(net, paths):
    c = app.app.test_client()
    st = c.get('/api/prices/status').get_json()
    assert st['cached'] is False and st['count'] == 0
    app.refresh_prices()
    app.refresh_tasks()
    st = c.get('/api/prices/status').get_json()
    assert st['cached'] and st['source'] == 'json.tarkov.dev' and st['count'] == 5
    assert st['age_minutes'] < 1 and st['error'] is None and st['stale'] is False
    assert st['tasks']['count'] == 2 and st['tasks']['source'] == 'json.tarkov.dev'
    # an outage shows up as the last error while the old data is still reported
    net.down = True
    with pytest.raises(app.PriceFetchError):
        app.refresh_prices(force=True)
    st = c.get('/api/prices/status').get_json()
    assert st['cached'] and 'json.tarkov.dev' in st['error']


def test_status_of_a_pre_json_cache_says_graphql(paths):
    put(app.PRICES_PATH, time.time() - 3600 * 24, items=[{'id': 'x'}])
    st = app.app.test_client().get('/api/prices/status').get_json()
    assert st['source'] == 'graphql' and st['stale'] is True


def test_manual_refresh_endpoint_reports_unchanged(net, paths):
    c = app.app.test_client()
    r = c.post('/api/prices/refresh').get_json()
    assert r['ok'] and r['count'] == 5 and r['status'] == 'updated'
    assert c.post('/api/prices/refresh').get_json()['status'] == 'unchanged'
