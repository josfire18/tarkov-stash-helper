"""
Regression tests for the icon-DB save/load race (v0.3.0 "File is not a zip
file" bug): the status endpoints poll get_icon_db() while save_icon_db is
still writing, and np.load on a half-written archive raises BadZipFile.
Fixed by write-then-rename in save_icon_db plus not touching the disk while
a build is in flight.
"""
import os

import numpy as np
import pytest

import app as tsh


def _tiny_raw_db():
    """Smallest possible by_size_raw: one 1x1 template."""
    d = tsh.CANONICAL_PER_SLOT * tsh.CANONICAL_PER_SLOT * 3
    return {(1, 1): {
        'ids':     ['item-a'],
        'names':   ['Item A'],
        'sources': ['api'],
        'rotated': [False],
        'tmpls':   [np.zeros(d, dtype=np.uint8)],
        'masks':   [np.ones(d, dtype=np.uint8)],
    }}


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    p = str(tmp_path / 'icon_db.npz')
    monkeypatch.setattr(tsh, 'ICON_DB_PATH', p)
    monkeypatch.setattr(tsh, '_icon_db', None)
    monkeypatch.setattr(tsh, '_icon_db_error', None)
    return p


def test_save_then_load_roundtrip(db_path):
    tsh.save_icon_db(_tiny_raw_db())
    assert os.path.exists(db_path)
    assert not os.path.exists(db_path + '.tmp.npz')   # tmp cleaned up
    db = tsh.load_icon_db()
    assert db is not None and (1, 1) in db
    assert tsh._icon_db_error is None


def test_save_is_atomic_over_existing_file(db_path):
    """os.replace must swap the file in one step — the destination is never
    truncated/partial, even briefly, while the new archive is being written."""
    tsh.save_icon_db(_tiny_raw_db())
    before = os.path.getsize(db_path)
    tsh.save_icon_db(_tiny_raw_db())
    assert os.path.getsize(db_path) == before   # replaced, not appended/truncated
    assert tsh.load_icon_db() is not None


def test_partial_file_sets_error_and_good_load_clears_it(db_path):
    tsh.save_icon_db(_tiny_raw_db())
    with open(db_path, 'rb') as f:
        blob = f.read()
    # Simulate what a reader used to see mid-save: a truncated archive.
    with open(db_path, 'wb') as f:
        f.write(blob[:len(blob) // 2])
    assert tsh.load_icon_db() is None
    assert 'Icon DB load failed' in (tsh._icon_db_error or '')
    # Restore the full file — a subsequent good load clears the sticky error.
    with open(db_path, 'wb') as f:
        f.write(blob)
    assert tsh.load_icon_db() is not None
    assert tsh._icon_db_error is None


def test_get_icon_db_skips_disk_while_build_running(db_path, monkeypatch):
    tsh.save_icon_db(_tiny_raw_db())
    calls = []
    monkeypatch.setattr(tsh, 'load_icon_db', lambda: calls.append(1) or None)
    monkeypatch.setitem(tsh._index_build_state, 'running', True)
    try:
        assert tsh.get_icon_db() is None
        assert calls == []                       # no disk read during a build
    finally:
        monkeypatch.setitem(tsh._index_build_state, 'running', False)
    tsh.get_icon_db()
    assert calls == [1]                          # normal path reads again
