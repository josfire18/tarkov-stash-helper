"""assets/icon.ico is a build input: running the app must never write into assets/."""
import hashlib
import os

import icon_asset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, 'assets')


def snapshot():
    out = {}
    for dp, _dn, fns in os.walk(ASSETS):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, 'rb') as fh:
                out[p] = (hashlib.sha256(fh.read()).hexdigest(), os.path.getmtime(p))
    return out


def test_runtime_icon_helpers_leave_assets_alone(tmp_path):
    before = snapshot()
    assert before, 'assets/icon.ico should be present'
    icon_asset.load_tray_image()
    icon_asset.set_app_id()
    icon_asset.ensure_ico(str(tmp_path / 'data' / 'app.ico'))          # what _run_app does (into data/)
    assert snapshot() == before


def test_ensure_ico_refuses_assets(tmp_path):
    target = os.path.join(ASSETS, 'should_not_exist.ico')
    assert icon_asset.ensure_ico(target) is None
    assert not os.path.exists(target)
    before = snapshot()
    assert icon_asset.ensure_ico(os.path.join(ASSETS, 'icon.ico')) is None
    assert snapshot() == before


def test_bake_only_writes_when_bytes_differ(tmp_path):
    p = str(tmp_path / 'icon.ico')
    assert icon_asset.bake_build_icon(p) is True
    m = os.path.getmtime(p)
    assert icon_asset.bake_build_icon(p) is False and os.path.getmtime(p) == m
