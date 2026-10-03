"""Settings page + /api/settings validation.  The real data/settings.json is never touched and
lifecycle.apply_settings (the Windows start-up entry) is stubbed out."""
import json

import pytest

import app as tsh_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(tsh_app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    applied = []
    monkeypatch.setattr(tsh_app.lifecycle, 'apply_settings', lambda s: applied.append(dict(s)))
    with tsh_app.app.test_client() as c:
        c.applied = applied
        c.path = tmp_path / 'settings.json'
        yield c


def test_page_renders(client):
    r = client.get('/settings')
    assert r.status_code == 200 and b'ignore_task_items' in r.data


def test_round_trip_and_unknown_keys_kept(client):
    r = client.post('/api/settings', json={'ignore_task_items': True, 'my_extra': {'a': 1},
                                           'game_mode': 'Season', 'live_top_n': 99})
    assert r.status_code == 200
    saved = json.loads(client.path.read_text())
    assert saved['ignore_task_items'] is True and saved['my_extra'] == {'a': 1}
    assert saved['game_mode'] == 'season' and saved['live_top_n'] == 20      # canonical + clamped
    assert client.applied and client.applied[-1]['ignore_task_items'] is True
    full = client.get('/api/settings?full=1').get_json()
    assert full['settings']['ignore_task_items'] is True and 'defaults' in full


@pytest.mark.parametrize('bad', [{'auto_scan': 'yes'}, {'faction': 'Scav'}, {'flea_offer_slots': 'x'},
                                 {'trader_levels': {'Ref': 'max'}}])
def test_bad_values_refused_and_nothing_written(client, bad):
    r = client.post('/api/settings', json=bad)
    assert r.status_code == 400 and r.get_json()['ok'] is False
    if client.path.exists():                       # load_json may write the defaults on first read
        saved = json.loads(client.path.read_text())
        for k, v in bad.items():
            assert saved.get(k) != v


def test_partial_update_keeps_other_keys(client):
    client.post('/api/settings', json={'flea_min_gain': 5000})
    client.post('/api/settings', json={'auto_scan': False})
    saved = json.loads(client.path.read_text())
    assert saved['flea_min_gain'] == 5000 and saved['auto_scan'] is False
