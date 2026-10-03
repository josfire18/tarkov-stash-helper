"""The three pages (Live, Needs, Settings), the old-route redirects and the tiny endpoints the
Needs / Live pages call.  Data paths live in tmp_path; the real data/ folder is never touched."""
import json

import pytest

import app as tsh_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(tsh_app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    monkeypatch.setattr(tsh_app, 'PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(tsh_app, 'KEEPLIST_PATH', str(tmp_path / 'keep_list.json'))
    monkeypatch.setattr(tsh_app, 'PRICES_PATH', str(tmp_path / 'prices.json'))
    cache = {'timestamp': 0, 'tasks': [], 'hideoutStations': [
        {'id': 'lav', 'name': 'Lavatory', 'levels': [
            {'id': 'lav-1', 'level': 1, 'itemRequirements': []},
            {'id': 'lav-2', 'level': 2, 'itemRequirements': []},
            {'id': 'lav-3', 'level': 3, 'itemRequirements': []}]}]}
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: cache)
    prices = {'timestamp': 9e12, 'items': [
        {'id': 'a', 'name': 'LEDX Skin Transilluminator', 'shortName': 'LEDX'},
        {'id': 'b', 'name': 'Salewa first aid kit', 'shortName': 'Salewa'},
        {'id': 'c', 'name': 'Ledx case', 'shortName': 'LCase'}]}
    (tmp_path / 'prices.json').write_text(json.dumps(prices))
    monkeypatch.setattr(tsh_app, '_load_cache', lambda path: json.load(open(path)) if path == str(tmp_path / 'prices.json') else None)
    monkeypatch.setattr(tsh_app, '_usable_prices', lambda c: c or {})
    with tsh_app.app.test_client() as c:
        c.progress = tmp_path / 'progress.json'
        yield c


@pytest.mark.parametrize('path', ['/', '/live', '/needs', '/settings'])
def test_pages_render_with_shared_header(client, path):
    r = client.get(path)
    assert r.status_code == 200
    body = r.data.decode()
    for label in ('>Live<', '>Needs<', '>Settings<'):
        assert label in body
    assert 'id="pill"' in body and '--s3' in body          # one status pill, shared css inlined
    assert 'Sell Advisor' not in body and 'Keep List' not in body


def test_old_routes_redirect(client):
    for old, new in (('/tasks', '/needs'), ('/keep', '/needs'), ('/sell', '/')):
        r = client.get(old)
        assert r.status_code == 302 and r.headers['Location'].endswith(new)


def test_live_page_has_no_admin_controls(client):
    body = client.get('/').data.decode()
    for gone in ('Set Stash Region', 'Build Icon DB', 'Refresh Prices', 'auto-toggle', 'region-chip'):
        assert gone not in body


def test_hideout_level_sets_every_level_up_to_it(client):
    r = client.post('/api/hideout/level', json={'station_id': 'lav', 'level': 2})
    assert r.get_json()['ok']
    assert json.loads(client.progress.read_text())['completed_hideout'] == ['lav-1', 'lav-2']
    client.post('/api/hideout/level', json={'station_id': 'lav', 'level': 1})
    assert json.loads(client.progress.read_text())['completed_hideout'] == ['lav-1']
    client.post('/api/hideout/level', json={'station_id': 'lav', 'level': 0})
    assert json.loads(client.progress.read_text())['completed_hideout'] == []


def test_hideout_level_errors(client):
    assert client.post('/api/hideout/level', json={'station_id': 'nope', 'level': 1}).status_code == 404
    assert client.post('/api/hideout/level', json={'station_id': 'lav', 'level': 'x'}).status_code == 400


def test_item_search(client):
    names = [i['name'] for i in client.get('/api/items/search?q=ledx').get_json()['items']]
    assert names == ['Ledx case', 'LEDX Skin Transilluminator']       # prefix hits, shortest first
    assert client.get('/api/items/search?q=l').get_json()['items'] == []
    assert client.get('/api/items/search?q=zzzz').get_json()['items'] == []


def test_pin_add_keeps_item_id(client):
    r = client.post('/api/keep-list/add', json={'name': 'Salewa first aid kit', 'category': 'tasks', 'tdev_id': 'b'})
    item = r.get_json()['item']
    assert item['source'] == 'custom' and item['tdev_id'] == 'b'
    kl = client.get('/api/keep-list').get_json()
    assert any(i['id'] == item['id'] for c in kl['categories'] for i in c['items'])


def test_rescan_needs_auto_scan(client, monkeypatch):
    calls = []
    monkeypatch.setattr(tsh_app.autoscanner, 'enabled', lambda: False)
    assert client.post('/api/autoscan/rescan').status_code == 409
    monkeypatch.setattr(tsh_app.autoscanner, 'enabled', lambda: True)
    monkeypatch.setattr(tsh_app, 'rescan_now', lambda: calls.append(1))
    assert client.post('/api/autoscan/rescan').get_json()['ok'] and calls == [1]


def test_settings_page_owns_new_prestige(client):
    body = client.get('/settings').data.decode()
    assert 'New prestige' in body and 'Live viewer' not in body
