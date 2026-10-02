"""The "Build Icon DB" button builds what the identification engine needs and nothing else:
tarkov.dev base images in data/tmpl_src/ for every item, then the engine's template catalog.
(It used to also write the legacy 136 MB data/icon_db.npz.)"""
import json
import os
import time

import cv2
import numpy as np
import pytest

import app as tsh_app


class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status


class _Session:
    def __init__(self, png):
        self.png, self.urls = png, []

    def get(self, url, timeout=0):
        self.urls.append(url)
        return _Resp(self.png if 'missing' not in url else b'', 200 if 'missing' not in url else 404)


@pytest.fixture
def tmpl_dir(tmp_path, monkeypatch):
    d = tmp_path / 'tmpl_src'
    d.mkdir()
    monkeypatch.setattr(tsh_app, 'TMPL_SRC_DIR', str(d))
    return d


@pytest.fixture
def session(monkeypatch):
    bgra = np.zeros((64, 64, 4), np.uint8)
    bgra[8:56, 8:56] = (10, 200, 30, 255)
    ok, buf = cv2.imencode('.png', bgra)
    s = _Session(buf.tobytes())
    monkeypatch.setattr(tsh_app, '_get_icon_session', lambda: s)
    return s


def test_download_base_image_writes_bgra_png_once(tmpl_dir, session):
    p = tsh_app.download_base_image('abc', 'https://img/abc.webp')
    assert p == str(tmpl_dir / 'abc.png')
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert img.shape == (64, 64, 4)
    assert not [f for f in os.listdir(tmpl_dir) if 'tmp' in f]       # write-then-rename left nothing behind
    assert tsh_app.download_base_image('abc', 'https://img/abc.webp') == p
    assert len(session.urls) == 1                                     # cached: no second request


def test_download_base_image_failure_returns_none(tmpl_dir, session):
    assert tsh_app.download_base_image('x', 'https://img/missing.webp') is None
    assert tsh_app.download_base_image('y', None) is None
    assert os.listdir(tmpl_dir) == []


def test_download_missing_fetches_every_item_including_ammo_and_guns(tmpl_dir, session):
    # the legacy build skipped ammo/guns/presets/containers; the engine identifies them all
    items = [{'id': 'ammo1', 'baseImageLink': 'https://img/ammo1', 'types': ['ammo']},
             {'id': 'gun1', 'baseImageLink': 'https://img/gun1', 'types': ['gun']},
             {'id': 'have', 'baseImageLink': 'https://img/have', 'types': []},
             {'id': 'nolink', 'types': []},
             {'id': 'gone', 'baseImageLink': 'https://img/missing', 'types': []}]
    (tmpl_dir / 'have.png').write_bytes(b'\x89PNG existing')
    progress = []
    ok, failed = tsh_app.download_missing_base_images({'items': items}, progress_cb=lambda d, t: progress.append((d, t)))
    assert (ok, failed) == (2, 1)
    assert sorted(os.listdir(tmpl_dir)) == ['ammo1.png', 'gun1.png', 'have.png']
    assert (tmpl_dir / 'have.png').read_bytes() == b'\x89PNG existing'     # an existing image is never refetched
    assert progress[-1] == (3, 3)


def test_catalog_summary(tmp_path, monkeypatch):
    from identify import config
    path = tmp_path / 'identify_catalog_v2.npz'
    monkeypatch.setattr(config, 'CATALOG_PATH', str(path))
    assert tsh_app.catalog_summary() is None
    np.savez(path, meta_json=np.array(json.dumps({'n_api': 4321, 'n_cache': 7})))
    assert tsh_app.catalog_summary() == {'items': 4321}
    path.write_bytes(b'not a zip')
    assert tsh_app.catalog_summary() is None


def test_build_route_downloads_images_then_rebuilds_catalog_and_writes_no_legacy_db(tmp_path, monkeypatch):
    import identify.catalog
    calls = []
    prices = {'items': [{'id': 'a', 'baseImageLink': 'u'}]}
    monkeypatch.setattr(tsh_app, 'get_prices', lambda: prices)
    monkeypatch.setattr(tsh_app, 'download_missing_base_images',
                        lambda p, progress_cb=None: calls.append(('images', p)) or (1, 0))
    monkeypatch.setattr(identify.catalog, 'load_catalog',
                        lambda **kw: calls.append(('catalog', kw)))
    monkeypatch.setattr(tsh_app, '_warm_v2_engine', lambda: calls.append(('warm',)))
    monkeypatch.setattr(tsh_app, 'DATA', str(tmp_path))
    client = tsh_app.app.test_client()

    r = client.post('/api/icons/build-index').get_json()
    assert r['ok']
    deadline = time.time() + 5
    while time.time() < deadline and tsh_app._index_build_state['running']:
        time.sleep(0.02)
    assert [c[0] for c in calls] == ['images', 'catalog', 'warm']
    assert calls[1][1] == {'force_rebuild': True}                      # an explicit build always rebuilds
    assert tsh_app._index_build_state['error'] is None
    assert not list(tmp_path.glob('icon_db*'))                          # the legacy 136 MB database is gone


def test_build_route_reports_failure_in_status(monkeypatch):
    monkeypatch.setattr(tsh_app, 'get_prices', lambda: (_ for _ in ()).throw(RuntimeError('offline')))
    client = tsh_app.app.test_client()
    assert client.post('/api/icons/build-index').get_json()['ok']
    deadline = time.time() + 5
    while time.time() < deadline and tsh_app._index_build_state['running']:
        time.sleep(0.02)
    st = client.get('/api/icons/matcher-status').get_json()
    assert st['running'] is False and 'offline' in st['error']


def test_second_build_while_running_is_refused(monkeypatch):
    monkeypatch.setitem(tsh_app._index_build_state, 'running', True)
    r = tsh_app.app.test_client().post('/api/icons/build-index').get_json()
    assert r['ok'] is False
