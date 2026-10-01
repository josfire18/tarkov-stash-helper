"""The packaged-build smoke test (selftest.py) must pass from source too, so CI and
developers exercise the same checks the built exe runs."""
import app as tsh_app
import selftest


def test_selftest_required_checks_pass():
    report = selftest.run(tsh_app)
    failed = {k: v['detail'] for k, v in report['checks'].items() if v['required'] and not v['ok']}
    assert not failed, failed
    assert report['ok']


def test_v2_scan_without_item_database_says_what_to_do(tmp_path, monkeypatch):
    import numpy as np
    import pytest
    monkeypatch.setattr(tsh_app, 'PRICES_PATH', str(tmp_path / 'prices_cache.json'))
    monkeypatch.setattr(tsh_app, 'TMPL_SRC_DIR', str(tmp_path))
    with pytest.raises(tsh_app.ScanError, match='Build Icon DB'):
        tsh_app.scan_with_v2(np.zeros((100, 100, 3), np.uint8), {}, [])
