"""Taskbar icon plumbing: own AppUserModelID, multi-size .ico, spec icon."""
import os
import sys

import pytest
from PIL import Image

import icon_asset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ensure_ico_writes_a_multi_size_icon_once(tmp_path):
    p = str(tmp_path / 'sub' / 'app.ico')
    assert icon_asset.ensure_ico(p) == p
    with Image.open(p) as im:
        assert {(16, 16), (32, 32), (256, 256)} <= set(im.info['sizes'])
    m = os.path.getmtime(p)
    assert icon_asset.ensure_ico(p) == p and os.path.getmtime(p) == m          # not rewritten


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows shell API')
def test_set_app_id_is_readable_back_in_this_process():
    # only ever changes this test process's taskbar identity
    assert icon_asset.set_app_id('TarkovStashHelper.App') is True
    assert icon_asset.current_app_id() == 'TarkovStashHelper.App'


def test_the_spec_sets_the_exe_icon_and_the_ico_exists():
    assert "icon=os.path.join('assets', 'icon.ico')" in open(os.path.join(ROOT, 'TarkovStashHelper.spec'), encoding='utf-8').read()
    assert os.path.isfile(os.path.join(ROOT, 'assets', 'icon.ico'))
