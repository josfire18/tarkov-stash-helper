import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import pytest


@pytest.fixture(autouse=True)
def _no_network_no_real_data(monkeypatch, tmp_path):
    """Unit tests never touch the network or the real data/ folder: the json.tarkov.dev
    transport raises unless a test installs its own, and the price/task caches and the refresh
    bookkeeping file live in tmp_path (a test that wants other paths monkeypatches them itself)."""
    import tarkovdata

    def _blocked(url, headers, timeout):
        raise AssertionError(f'unit test tried to fetch {url}')
    monkeypatch.setattr(tarkovdata, '_http_get', _blocked)
    app = sys.modules.get('app')
    if app is not None:
        for name, fn in (('META_PATH', 'tarkovdev_meta.json'), ('PRICES_PATH', 'prices_cache.json'),
                         ('TASKS_CACHE_PATH', 'tasks_cache.json')):
            monkeypatch.setattr(app, name, str(tmp_path / 'autouse_data' / fn), raising=False)
        # Never read the real game's Logs folder or write data/eftlogs_cache.json.
        if hasattr(app, '_log_scanner'):
            import eftlogs
            monkeypatch.setattr(app, '_log_scanner', eftlogs.LogScanner(None))
            monkeypatch.setattr(app, '_log_progress',
                                {'key': None, 'result': None, 'error': None, 'scanned': False})
            monkeypatch.setattr(eftlogs, 'find_install_dir', lambda override=None: None)


@pytest.fixture(autouse=True)
def _hermetic_anchors(monkeypatch):
    """Engines built in unit tests never read the real EFT icon cache or the game's font (the
    anchor tests build their own synthetic ones)."""
    import identify.pipeline as pl
    import identify.fontlabel as fl
    monkeypatch.setattr(pl, 'default_cache_dir', lambda: None)
    monkeypatch.setattr(fl, 'ensure_font', lambda *a, **k: None)
