"""
test_update.py — offline, no-network tests for the self-update feature:
  * app.py's _ver_tuple / _is_newer version-comparison helpers
  * /api/update-check (GitHub fetch monkeypatched — never hits the network)
  * /api/update-apply refusing to run when not FROZEN

Run with: python -m pytest test_update.py -q
"""

import pytest

import app as tsh_app


@pytest.fixture(autouse=True)
def _reset_update_globals():
    """Every test gets a clean cache/state — the update-check cache and
    server-side download-url state are both module-level globals."""
    tsh_app._update_cache['ts'] = 0
    tsh_app._update_cache['data'] = None
    tsh_app._update_state['download_url'] = None
    tsh_app._update_state['asset_size'] = None
    tsh_app._update_state['tag'] = None
    yield
    tsh_app._update_cache['ts'] = 0
    tsh_app._update_cache['data'] = None
    tsh_app._update_state['download_url'] = None
    tsh_app._update_state['asset_size'] = None
    tsh_app._update_state['tag'] = None


@pytest.fixture
def client():
    tsh_app.app.testing = True
    return tsh_app.app.test_client()


def _fake_release(tag='v9.9.9', with_asset=True, size=123):
    release = {
        'tag_name': tag,
        'html_url': f'https://github.com/josfire18/tarkov-stash-helper/releases/{tag}',
        'assets': [],
    }
    if with_asset:
        release['assets'].append({
            'name': 'TarkovStashHelper.exe',
            'browser_download_url': f'https://github.com/example/{tag}/TarkovStashHelper.exe',
            'size': size,
        })
    return release


# ---------------------------------------------------------------------------
# _ver_tuple
# ---------------------------------------------------------------------------

def test_ver_tuple_normal():
    assert tsh_app._ver_tuple('0.3.1') == (0, 3, 1)


def test_ver_tuple_v_prefixed():
    assert tsh_app._ver_tuple('v0.3.1') == (0, 3, 1)
    assert tsh_app._ver_tuple('V0.3.1') == (0, 3, 1)


def test_ver_tuple_unequal_length():
    assert tsh_app._ver_tuple('0.3') == (0, 3)
    assert tsh_app._ver_tuple('0.3.1') == (0, 3, 1)


def test_ver_tuple_junk_suffix_takes_leading_numeric_run():
    assert tsh_app._ver_tuple('v0.3.0-beta1') == (0, 3, 0)


def test_ver_tuple_garbage_returns_none():
    assert tsh_app._ver_tuple('garbage') is None
    assert tsh_app._ver_tuple('') is None
    assert tsh_app._ver_tuple(None) is None


# ---------------------------------------------------------------------------
# _is_newer
# ---------------------------------------------------------------------------

def test_is_newer_true_normal():
    assert tsh_app._is_newer('0.4.0', '0.3.1') is True


def test_is_newer_false_equal():
    assert tsh_app._is_newer('0.3.1', '0.3.1') is False


def test_is_newer_false_older():
    assert tsh_app._is_newer('0.3.0', '0.3.1') is False


def test_is_newer_v_prefixed():
    assert tsh_app._is_newer('v0.4.0', '0.3.1') is True


def test_is_newer_unequal_length_padding():
    assert tsh_app._is_newer('0.4', '0.3.1') is True
    assert tsh_app._is_newer('0.3', '0.3.1') is False


def test_is_newer_false_when_unparsable():
    assert tsh_app._is_newer('garbage', '0.3.1') is False
    assert tsh_app._is_newer('0.4.0', 'garbage') is False
    assert tsh_app._is_newer(None, '0.3.1') is False


# ---------------------------------------------------------------------------
# /api/update-check
# ---------------------------------------------------------------------------

def test_update_check_update_available(client, monkeypatch):
    monkeypatch.setattr(tsh_app, '_fetch_latest_release', lambda: _fake_release('v9.9.9'))
    res = client.get('/api/update-check').get_json()
    assert res['current'] == tsh_app.APP_VERSION
    assert res['latest'] == 'v9.9.9'
    assert res['update_available'] is True
    assert res['notes_url']
    assert res['error'] is None
    # FROZEN is False under pytest, so can_auto must never be True even
    # though a newer release + matching asset both exist.
    assert res['can_auto'] is False


def test_update_check_no_update_available(client, monkeypatch):
    monkeypatch.setattr(tsh_app, '_fetch_latest_release',
                        lambda: _fake_release(f'v{tsh_app.APP_VERSION}'))
    res = client.get('/api/update-check').get_json()
    assert res['update_available'] is False
    assert res['can_auto'] is False
    assert res['error'] is None


def test_update_check_error_path(client, monkeypatch):
    def boom():
        raise RuntimeError('network unreachable')
    monkeypatch.setattr(tsh_app, '_fetch_latest_release', boom)
    r = client.get('/api/update-check')
    assert r.status_code == 200   # never break the UI
    res = r.get_json()
    assert res['update_available'] is False
    assert 'network unreachable' in res['error']


def test_update_check_cache_behavior_and_force_bypass(client, monkeypatch):
    calls = {'n': 0}
    def fake_fetch():
        calls['n'] += 1
        tag = 'v1.0.0' if calls['n'] == 1 else 'v2.0.0'
        return _fake_release(tag)
    monkeypatch.setattr(tsh_app, '_fetch_latest_release', fake_fetch)

    first = client.get('/api/update-check').get_json()
    assert first['latest'] == 'v1.0.0'
    assert calls['n'] == 1

    # Second call within the cache TTL, no force -> cached result, no re-fetch.
    second = client.get('/api/update-check').get_json()
    assert second['latest'] == 'v1.0.0'
    assert calls['n'] == 1

    # force=1 bypasses the cache and re-fetches.
    third = client.get('/api/update-check?force=1').get_json()
    assert third['latest'] == 'v2.0.0'
    assert calls['n'] == 2


# ---------------------------------------------------------------------------
# /api/update-apply
# ---------------------------------------------------------------------------

def test_update_apply_refuses_when_not_frozen(client, monkeypatch):
    monkeypatch.setattr(tsh_app, 'FROZEN', False)
    r = client.post('/api/update-apply')
    assert r.status_code == 400
    res = r.get_json()
    assert 'git pull' in res['error']
