"""/api/v1/needs: the versioned read-only needs summary other tools (the Rocko bot) depend on."""
import json

import pytest

import app as tsh_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(tsh_app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    monkeypatch.setattr(tsh_app, 'PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(tsh_app, 'KEEPLIST_PATH', str(tmp_path / 'keep_list.json'))
    cache = {'timestamp': 0, 'tasks': [
        {'id': 't1', 'name': 'Gunsmith', 'kappaRequired': True, 'trader': {'name': 'Mechanic'},
         'objectives': [{'type': 'giveItem', 'count': 2, 'foundInRaid': True,
                         'items': [{'id': 'gpu', 'name': 'Graphics card', 'shortName': 'GPU'}]}]},
    ], 'hideoutStations': [
        {'id': 'med', 'name': 'Medstation', 'levels': [
            {'id': 'med-1', 'level': 1, 'itemRequirements': [
                {'item': {'id': 'gpu', 'name': 'Graphics card', 'shortName': 'GPU'}, 'count': 1},
                {'item': {'id': 'sal', 'name': 'Salewa first aid kit', 'shortName': 'Salewa'}, 'count': 3}]}]}]}
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: cache)
    (tmp_path / 'progress.json').write_text(json.dumps({'have': {'sal': 5, 'gpu': 1}}))
    (tmp_path / 'keep_list.json').write_text(json.dumps({'categories': [
        {'id': 'kappa', 'label': 'Kappa', 'items': [
            {'id': 'tea', 'name': 'English Tea', 'acquired': False},
            {'id': 'axe', 'name': 'Antique axe', 'acquired': True}]}]}))
    with tsh_app.app.test_client() as c:
        yield c


def test_needs_lists_only_what_is_still_short(client):
    body = client.get('/api/v1/needs').get_json()
    assert body['version'] == 1
    assert [i['id'] for i in body['items']] == ['gpu']      # Salewa: have 5 of 3, done
    gpu = body['items'][0]
    assert (gpu['needed'], gpu['fir_needed'], gpu['have'], gpu['still_needed']) == (3, 2, 1, 2)
    assert {s['type'] for s in gpu['sources']} == {'task', 'hideout'}
    assert body['pins'] == [{'name': 'English Tea', 'category': 'Kappa'}]


def test_needs_reports_task_data_errors(client, monkeypatch):
    def boom(allow_fetch=True):
        raise RuntimeError('no task data yet')
    monkeypatch.setattr(tsh_app, 'get_tasks', boom)
    r = client.get('/api/v1/needs')
    assert r.status_code == 503
    assert r.get_json() == {'version': 1, 'error': 'no task data yet'}
