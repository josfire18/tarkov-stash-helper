"""Game mode (PvP / PvE / PvP Season): detection from the game log, the effective mode, one data set
per mode (no mixing), the json.tarkov.dev base URL per mode, the API and the header chip.

Nothing here touches the network, the real Logs folder or the real data/ (see conftest)."""
import copy
import datetime
import json
import os
import threading
import time

import pytest
import requests

import app
import eftlogs
import tarkovdata as td
from test_tarkovdata import M4A1, FakeNet, Resp, all_docs

PVE_PRICE = 12345


def line(h, m, s, rest):
    return f'2026-10-11 {h:02d}:{m:02d}:{s:02d}.100|1.2.0.0.47965|Info|application|{rest}\n'


def ts(h, m, s):
    return datetime.datetime(2026, 10, 11, h, m, s).timestamp()


def make_session(root, folder='log_2026.10.11_14-47-42_1.2.0.0.47965', first=''):
    d = root / eftlogs.LOG_DIR_NAME / folder
    d.mkdir(parents=True)
    f = d / '2026.10.11_14-47-42_1.2.0.0.47965 application_000.log'
    f.write_text(first + line(14, 47, 43, "Application awaken, updateQueue:'Update'"), encoding='utf-8')
    return f


def append(f, text):
    with open(f, 'a', encoding='utf-8') as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Detection from the log
# ---------------------------------------------------------------------------

def test_session_mode_values_are_parsed():
    text = (line(14, 47, 50, 'Session mode: Regular') + line(15, 0, 0, 'Session mode: Pve')
            + line(16, 0, 0, 'Session mode: PvpSeason'))
    assert [m for _, m in eftlogs.extract_session_modes(text)] == ['pvp', 'pve', 'season']


def test_live_mode_follows_the_latest_line_and_a_switch_mid_session(tmp_path):
    f = make_session(tmp_path)
    rs = eftlogs.RaidState(str(tmp_path))
    assert rs.mode() is None                                          # the session has not said yet
    append(f, line(14, 47, 50, 'Session mode: PvpSeason'))
    assert rs.mode() == {'mode': 'season', 'since': pytest.approx(ts(14, 47, 50)), 'source': 'log'}
    append(f, line(14, 48, 0, 'Matching with group id: ') + line(15, 5, 0, 'Session mode: Pve'))
    assert rs.mode()['mode'] == 'pve' and rs.mode()['since'] == pytest.approx(ts(15, 5, 0))
    assert rs.get()['state'] == 'matching'                            # the raid state machine is unaffected
    append(f, line(15, 9, 0, 'Session mode: Regular'))
    assert rs.mode()['mode'] == 'pvp'


def test_a_newer_session_forgets_the_old_mode_until_it_says_its_own(tmp_path):
    make_session(tmp_path, first=line(14, 47, 42, 'Session mode: Pve'))
    now = [1000.0]
    rs = eftlogs.RaidState(str(tmp_path), clock=lambda: now[0])
    assert rs.mode()['mode'] == 'pve'
    make_session(tmp_path, folder='log_2026.10.12_09-00-00_1.2.0.0.47965')
    now[0] += eftlogs.RAID_FOLDER_RECHECK_S + 1                       # the folder list is re-read every few seconds
    assert rs.mode() is None
    assert rs.get() is not None                                       # (logs readable, mode just unknown)


def test_mode_line_at_the_top_of_a_huge_log_is_still_found(tmp_path, monkeypatch):
    monkeypatch.setattr(eftlogs, 'RAID_FIRST_READ_MAX', 2000)
    f = make_session(tmp_path, first=line(14, 47, 42, 'Session mode: Pve'))
    append(f, ''.join(line(14, 50, i % 60, 'filler ' + 'x' * 60) for i in range(200)))
    rs = eftlogs.RaidState(str(tmp_path))
    assert rs.mode()['mode'] == 'pve'


def test_latest_session_mode_of_the_scanner_records():
    recs = [{'start': 1, 'modes': [[5, 'pvp']]}, {'start': 9, 'modes': [[10, 'pve'], [20, 'season']]},
            {'start': 20, 'modes': [], 'host_mode': None}]
    assert eftlogs.latest_session_mode(recs) == ('season', 20)       # newest session that says one
    assert eftlogs.latest_session_mode([]) == (None, None)
    assert eftlogs.latest_mode([]) == 'pvp'


class StubRaid:
    def __init__(self, mode=None):
        self.m = mode

    def mode(self):
        return None if self.m is None else {'mode': self.m, 'since': 100.0, 'source': 'log'}

    def get(self):
        return None


@pytest.fixture
def detected(monkeypatch):
    """Set what the live log says; ``detected('pve')``."""
    def setter(mode):
        monkeypatch.setattr(app, 'raid_state', StubRaid(mode))
        app._mode_cache['val'] = None
    return setter


def test_detection_falls_back_to_the_last_session_then_to_pvp(detected, monkeypatch):
    detected(None)
    assert app.detect_mode() == {'mode': 'pvp', 'source': 'default', 'since': None}
    monkeypatch.setattr(app._log_scanner, 'records', lambda: [{'start': 3, 'modes': [[4, 'season']]}])
    app._mode_cache['val'] = None
    assert app.detect_mode() == {'mode': 'season', 'source': 'history', 'since': 4}
    detected('pve')
    assert app.detect_mode()['mode'] == 'pve' and app.detect_mode()['source'] == 'log'


# ---------------------------------------------------------------------------
# Effective mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pick,live,want', [
    ('auto', 'season', 'season'), ('auto', 'pve', 'pve'), ('auto', None, 'pvp'),
    ('pve', 'season', 'pve'), ('pvp', 'pve', 'pvp'), ('season', 'pvp', 'season'),
    ('bogus', 'pve', 'pve'),                                            # a hand-edited value means auto
])
def test_effective_mode_is_the_pick_or_the_detected_one(detected, pick, live, want):
    detected(live)
    assert app.effective_mode({'game_mode': pick}) == want
    assert app.effective_mode({}) == (live or 'pvp')


def test_mode_info_flags_a_manual_pick_that_differs_from_the_game(detected):
    detected('season')
    info = app.mode_info({'game_mode': 'pve'})
    assert (info['effective'], info['detected'], info['mismatch'], info['label']) == ('pve', 'season', True, 'PvE')
    assert info['detected_label'] == 'PvP S' and info['data_source'] == 'pve'
    assert info['options'][0] == {'value': 'auto', 'label': 'Auto · detected PvP S'}
    assert [o['label'] for o in info['options'][1:]] == ['PvP', 'PvE', 'PvP Season']
    auto = app.mode_info({'game_mode': 'auto'})
    assert not auto['mismatch'] and auto['effective'] == 'season' and auto['seasonal_prices'] is True
    assert auto['data_source'] == 'regular'                              # no seasonal data: PvP market
    detected(None)
    assert not app.mode_info({'game_mode': 'pve'})['mismatch']          # nothing detected: nothing to disagree with


# ---------------------------------------------------------------------------
# Data per mode
# ---------------------------------------------------------------------------

def test_cache_and_bookkeeping_paths_are_per_data_source():
    assert td.source_for_mode('pvp') == td.source_for_mode('season') == 'regular'
    assert td.source_for_mode('pve') == 'pve'
    assert app._prices_path('regular') == app.PRICES_PATH                # existing installs keep their files
    assert app._tasks_path('regular') == app.TASKS_CACHE_PATH and app._meta_path('regular') == app.META_PATH
    assert app._prices_path('pve').endswith('prices_cache_pve.json')
    assert app._tasks_path('pve').endswith('tasks_cache_pve.json')
    assert app._meta_path('pve').endswith('tarkovdev_meta_pve.json')
    with pytest.raises(ValueError):
        td.source_path('x.json', 'season')


class ModeNet(FakeNet):
    """json.tarkov.dev with a regular/ and a pve/ data set that differ in the M4A1's flea price."""

    def __init__(self):
        super().__init__()
        self.pve_docs = all_docs()
        self.pve_docs['items']['data']['items'][M4A1]['lastLowPrice'] = PVE_PRICE
        self.bases = []

    def __call__(self, url, headers, timeout):
        base = url.rsplit('/', 2)[1]
        doc = url.rsplit('/', 1)[1]
        self.bases.append((base, doc))
        self.requests.append((doc, headers.get('If-None-Match')))
        if self.down:
            raise requests.ConnectionError('network down')
        docs = self.pve_docs if base == 'pve' else self.docs
        etag = f'W/"{base}-{doc}"'
        if headers.get('If-None-Match') == etag:
            return Resp(304)
        return Resp(200, copy.deepcopy(docs[doc]), etag)


@pytest.fixture
def modenet(monkeypatch, tmp_path):
    n = ModeNet()
    monkeypatch.setattr(td, '_http_get', n)
    monkeypatch.setattr(td, 'MIN_ITEMS', 3)
    monkeypatch.setattr(td, 'MIN_TASKS', 1)
    monkeypatch.setattr(td, 'MIN_STATIONS', 1)
    d = tmp_path / 'data'
    d.mkdir()
    monkeypatch.setattr(app, 'PRICES_PATH', str(d / 'prices_cache.json'))
    monkeypatch.setattr(app, 'TASKS_CACHE_PATH', str(d / 'tasks_cache.json'))
    monkeypatch.setattr(app, 'META_PATH', str(d / 'tarkovdev_meta.json'))
    monkeypatch.setattr(app, 'SETTINGS_PATH', str(d / 'settings.json'))
    monkeypatch.setattr(app, '_queue_catalog_update', lambda: None)
    for k in app._refresh_state:
        app._refresh_state[k].update({'attempt': 0, 'error': None})
    n.dir = d
    return n


def test_fetches_use_the_base_url_of_the_data_source(modenet):
    td.fetch_doc('items')
    td.fetch_doc('items', source='pve')
    assert modenet.bases == [('regular', 'items'), ('pve', 'items')]
    with pytest.raises(KeyError):
        td.fetch_doc('items', source='season')                          # there is no seasonal data set


def flea_price(cache):
    item = next(i for i in cache['items'] if i['id'] == M4A1)
    return next(s['price'] for s in item['sellFor'] if s['vendor']['name'] == 'Flea Market')


def test_each_mode_has_its_own_cache_and_etags_and_prices_are_never_mixed(modenet, detected):
    detected('pvp')
    app.refresh_prices()
    app.refresh_tasks()
    assert {b for b, _ in modenet.bases} == {'regular'}
    detected('pve')
    app.refresh_prices()
    app.refresh_tasks()
    assert os.path.exists(app.PRICES_PATH) and os.path.exists(app._prices_path('pve'))
    assert flea_price(json.load(open(app.PRICES_PATH, encoding='utf-8'))) == 39899
    assert flea_price(json.load(open(app._prices_path('pve'), encoding='utf-8'))) == PVE_PRICE
    assert json.load(open(app._prices_path('pve'), encoding='utf-8'))['data_source'] == 'pve'
    assert 'items' in json.load(open(app._meta_path('pve'), encoding='utf-8'))['etags']
    assert 'W/"pve-items"' == json.load(open(app._meta_path('pve'), encoding='utf-8'))['etags']['items']
    assert 'W/"regular-items"' == json.load(open(app.META_PATH, encoding='utf-8'))['etags']['items']
    assert os.path.exists(app._tasks_path('pve')) and os.path.exists(app.TASKS_CACHE_PATH)
    # what the app reads follows the effective mode
    assert flea_price(app.get_prices()) == PVE_PRICE
    detected('pvp')
    assert flea_price(app.get_prices()) == 39899
    detected('season')
    assert flea_price(app.get_prices()) == 39899                        # seasonal = the PvP market
    assert app.get_tasks(allow_fetch=False)['tasks']


def test_pve_refresh_is_conditional_and_leaves_the_regular_etags_alone(modenet, detected):
    detected('pvp')
    app.refresh_prices()
    regular_etags = dict(json.load(open(app.META_PATH, encoding='utf-8'))['etags'])
    detected('pve')
    app.refresh_prices()
    modenet.bases.clear()
    assert app.refresh_prices()['status'] == 'unchanged'                # 304 against the PvE etags
    assert {b for b, _ in modenet.bases} == {'pve'}
    assert json.load(open(app.META_PATH, encoding='utf-8'))['etags'] == regular_etags


def test_a_never_fetched_mode_downloads_in_the_background_and_never_borrows_the_other(modenet, detected, monkeypatch):
    detected('pvp')
    app.refresh_prices()
    detected('pve')
    started = []
    monkeypatch.setattr(app, 'ensure_source_async', lambda src: started.append(src))
    with pytest.raises(app.PriceFetchError, match='PVE prices are still downloading'):
        app.get_prices()                                                # a scan: no blocking fetch, no PvP prices
    assert started == ['pve']
    assert app.get_tasks(allow_fetch=False) is None
    with pytest.raises(RuntimeError, match='still downloading'):
        app.get_tasks()
    assert not any(b == 'pve' for b, _ in modenet.bases)               # nothing was fetched inline


def test_ensure_source_async_fetches_prices_then_tasks_once(modenet):
    assert app.ensure_source_async('regular') is False
    before = set(threading.enumerate())
    assert app.ensure_source_async('pve') is True
    for t in set(threading.enumerate()) - before:
        t.join(10)
    assert os.path.exists(app._prices_path('pve')) and os.path.exists(app._tasks_path('pve'))
    assert app._ensure['running'] == set()
    assert {b for b, _ in modenet.bases} == {'pve'}


def test_the_refresher_keeps_only_the_active_mode_current(modenet, detected):
    detected('pve')
    assert app.refresh_cycle() == ['prices', 'tasks']
    assert {b for b, _ in modenet.bases} == {'pve'}
    assert not os.path.exists(app.PRICES_PATH)


def test_pve_has_no_graphql_fallback(modenet, detected, monkeypatch):
    detected('pve')
    modenet.down = True
    monkeypatch.setattr(app, '_graphql_prices', lambda: pytest.fail('GraphQL has no PvE data here'))
    with pytest.raises(app.PriceFetchError):
        app.refresh_prices()


def test_quest_progress_reads_the_profile_of_the_effective_mode(modenet, detected, monkeypatch):
    seen = []

    class Scanner:
        last_scan_at = 1

        def progress(self, mode, tasks, faction):
            seen.append(mode)
            return {'mode': mode, 'folders': 1, 'completed': [], 'failed': [], 'active': [],
                    'last_event_at': None, 'reset_at': None, 'reset_kind': None}

        def records(self):
            return []

        def scan(self, *a, **k):
            return {}
    monkeypatch.setattr(app, '_log_scanner', Scanner())
    app._log_progress.update(scanned=True, key=None, result=None)
    app.save_json(app.SETTINGS_PATH, {**app.default_settings(), 'game_mode': 'auto'})
    cache = {'timestamp': 1, 'tasks': [], 'hideoutStations': []}
    detected('season')
    app.log_task_progress(cache)
    detected('pve')
    app.log_task_progress(cache)
    app.save_json(app.SETTINGS_PATH, {**app.default_settings(), 'game_mode': 'pvp'})
    app.log_task_progress(cache)
    assert seen == ['season', 'pve', 'pvp']


# ---------------------------------------------------------------------------
# API + header
# ---------------------------------------------------------------------------

@pytest.fixture
def client(modenet, detected, monkeypatch):
    monkeypatch.setattr(app.lifecycle, 'apply_settings', lambda s: None)
    detected('season')
    with app.app.test_client() as c:
        yield c


def test_api_mode_get_reports_detected_and_effective(client):
    m = client.get('/api/mode').get_json()
    assert (m['setting'], m['detected'], m['effective'], m['label']) == ('auto', 'season', 'season', 'PvP S')
    assert m['detected_source'] == 'log' and m['seasonal_prices'] is True and m['data_source'] == 'regular'


def test_api_mode_post_saves_the_setting_and_switches_the_effective_mode(client):
    r = client.post('/api/mode', json={'mode': 'pve'})
    assert r.status_code == 200 and r.get_json()['effective'] == 'pve' and r.get_json()['mismatch'] is True
    assert json.load(open(app.SETTINGS_PATH, encoding='utf-8'))['game_mode'] == 'pve'
    assert client.get('/api/settings').get_json()['game_mode'] == 'pve'
    assert client.post('/api/mode', json={'mode': 'auto'}).get_json()['effective'] == 'season'
    assert client.post('/api/mode', json={'mode': 'creative'}).status_code == 400
    assert client.post('/api/mode', json={}).status_code == 400
    assert json.load(open(app.SETTINGS_PATH, encoding='utf-8'))['game_mode'] == 'auto'


def test_changing_game_mode_in_settings_takes_effect_at_once(client):
    assert client.get('/api/mode').get_json()['effective'] == 'season'
    assert client.post('/api/settings', json={'game_mode': 'pvp'}).status_code == 200
    assert client.get('/api/mode').get_json()['effective'] == 'pvp'


def test_status_endpoint_carries_the_mode(client):
    st = client.get('/api/autoscan/status').get_json()
    assert st['mode']['effective'] == 'season' and st['mode']['detected'] == 'season'


def test_price_status_names_the_data_source(client, modenet):
    client.post('/api/mode', json={'mode': 'pve'})
    app.refresh_prices()
    st = client.get('/api/prices/status').get_json()
    assert st['mode'] == 'pve' and st['data_source'] == 'pve' and st['cached'] is True


def test_header_renders_the_mode_chip_and_menu_on_every_page(client):
    for path in ('/', '/needs', '/settings'):
        html = client.get(path).get_data(as_text=True)
        assert 'id="modechip"' in html and 'id="modemenu"' in html and 'aria-haspopup="menu"' in html
        for mode in ('auto', 'pvp', 'pve', 'season'):
            assert f'data-mode="{mode}"' in html
        assert "tsh:mode" in html


def test_settings_game_mode_select_offers_the_same_modes(client):
    html = client.get('/settings').get_data(as_text=True)
    assert "label: 'PvP Season'" in html and "id: 'mode-detected'" in html
