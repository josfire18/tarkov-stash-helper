"""Cold start: the first scan after launch must not pay for one-time work, the first look is
published before the expensive phases, and a player who changes view never starves the display."""
import os
import threading
import time

import numpy as np
import pytest

from identify import dino as dino_mod
from identify.pipeline import Engine, ScanCancelled
from identify.config import EngineSettings
from test_identify_pipeline import engine, world  # noqa: F401  (fixture)
from test_progressive import _prog_scanner, _synthetic, _wait
from tests.test_autoscan import run


def _fake_embed(imgs, batch=64):
    """Deterministic stand-in for DINO: a unit vector derived from the picture."""
    out = []
    for im in imgs:
        v = np.zeros(768, np.float32)
        v[int(im.mean()) % 768] = 1.0
        out.append(v)
    return np.stack(out) if out else np.zeros((0, 768), np.float32)


@pytest.fixture
def fake_dino(monkeypatch):
    calls = {'n': 0}

    def embed(imgs, batch=64):
        calls['n'] += len(imgs)
        return _fake_embed(imgs, batch)
    monkeypatch.setattr(dino_mod, 'embed', embed)
    return calls


# ---- order of phases ------------------------------------------------------------------------

def test_first_look_is_published_before_dino_and_certification(world, monkeypatch):
    cat, img, n = _synthetic(world)
    eng = engine(cat)
    order = []
    monkeypatch.setattr(eng, '_use_dino', lambda: True)
    monkeypatch.setattr(eng, '_dino_pass', lambda *a, **k: order.append('dino'))
    real_certify = eng._certify_all
    monkeypatch.setattr(eng, '_certify_all', lambda *a, **k: (order.append('certify'), real_certify(*a, **k)))
    eng.scan(img, on_provisional=lambda r: order.append('prov'))
    assert order[0] == 'prov' and order.index('dino') < order.index('certify')


def test_cancel_after_the_first_look_stops_before_dino(world, monkeypatch):
    cat, img, n = _synthetic(world)
    eng = engine(cat)
    ran = []
    monkeypatch.setattr(eng, '_use_dino', lambda: True)
    monkeypatch.setattr(eng, '_dino_pass', lambda *a, **k: ran.append('dino'))
    got = []
    with pytest.raises(ScanCancelled):
        eng.scan(img, on_provisional=got.append, cancel=lambda: bool(got))
    assert len(got) == 1 and ran == []


# ---- service: a new view never cancels a scan before its first look ---------------------------

def test_a_new_view_waits_for_the_running_scans_first_look_before_cancelling_it():
    gate = threading.Event()
    log = []

    def scan_fn(frame, publish, cancel):
        n = len(log) + 1
        log.append(('start', n))
        if n == 1:
            gate.wait(5)                                            # still working on its first look
            log.append(('cancelled_before_look', cancel()))
            publish({'results': [{'view': 1}], 'grid': {}, 'provisional': True})
            for _ in range(500):
                if cancel():
                    raise ScanCancelled()
                time.sleep(0.01)
        return {'results': [{'view': n}], 'grid': {}, 'grid_failed': False}
    s = _prog_scanner(scan_fn)
    run(s, 4)
    assert _wait(lambda: ('start', 1) in log)
    s.trigger.reset_view()
    t = threading.Thread(target=lambda: run(s, 4), daemon=True)     # the new view: the service thread blocks, it does not cancel
    t.start()
    time.sleep(0.3)
    assert ('cancelled_before_look', True) not in log and ('start', 2) not in log
    gate.set()
    t.join(10)
    assert ('cancelled_before_look', False) in log                  # scan 1 reached its first look un-cancelled
    assert _wait(lambda: s.status()['state'] == 'scanned')
    assert s.status()['seq'] >= 2                                   # the first look of view 1 and view 2's result are out
    s.stop()


def test_a_cancelled_scans_first_look_is_still_published_when_nothing_newer_exists():
    release = threading.Event()

    def scan_fn(frame, publish, cancel):
        release.wait(5)
        publish({'results': [{'view': 1}], 'grid': {}, 'provisional': True})   # cancel() is already true here
        raise ScanCancelled()
    s = _prog_scanner(scan_fn)
    run(s, 4)
    assert s.scanning()
    s._job[1].set()                                                 # e.g. superseded by a view that then never published
    release.set()
    assert _wait(lambda: s.status()['seq'] == 1)
    assert s.result(0)['results'] == [{'view': 1}] and s.result(0)['provisional']
    s.stop()


def test_stop_does_not_wait_for_a_first_look():
    never = threading.Event()

    def scan_fn(frame, publish, cancel):
        while not cancel():                                         # no first look, ever; but it does notice a cancel
            time.sleep(0.01)
        raise ScanCancelled()
    s = _prog_scanner(scan_fn)
    run(s, 4)
    assert s.scanning()
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 5


# ---- embedding cache -------------------------------------------------------------------------

def test_embeddings_are_persisted_and_reloaded_without_recomputing(world, fake_dino, tmp_path):
    cat, _ = world
    path = str(tmp_path / 'emb.npz')
    st = dino_mod.EmbeddingStore(cat, path)
    rows = list(range(len(cat)))
    first = st.get(rows)
    n = fake_dino['n']
    assert n == len(rows) and os.path.exists(path)
    st2 = dino_mod.EmbeddingStore(cat, path)
    assert st2.loaded == len(rows)
    again = st2.get(rows)
    assert fake_dino['n'] == n                                      # nothing re-embedded
    assert np.allclose(first, again, atol=1e-3)
    assert st2.by_key[next(iter(st2.by_key))].dtype == np.float16


def test_embedding_cache_survives_a_catalog_rebuild_and_only_changed_icons_are_redone(world, fake_dino, tmp_path):
    cat, _ = world
    path = str(tmp_path / 'emb.npz')
    dino_mod.EmbeddingStore(cat, path).precompute(log=lambda *a: None)
    base = fake_dino['n']
    cat.meta['built'] = time.time() + 1000                          # a rebuilt catalog: new stamp, same icon files
    st = dino_mod.EmbeddingStore(cat, path)
    assert st.loaded and st.missing(range(len(cat))) == []
    row = 0
    p = cat.meta['paths'][row]
    os.utime(p, (time.time() + 50, time.time() + 50))               # the icon file changed
    st = dino_mod.EmbeddingStore(cat, path)
    assert st.missing(range(len(cat))) == [row]
    st.precompute(log=lambda *a: None)
    assert fake_dino['n'] == base + 1


def test_embedding_cache_is_ignored_for_another_model_or_version(world, fake_dino, tmp_path, monkeypatch):
    cat, _ = world
    path = str(tmp_path / 'emb.npz')
    dino_mod.EmbeddingStore(cat, path).precompute(log=lambda *a: None)
    monkeypatch.setattr(dino_mod, 'MODEL_ID', 'other/model')
    assert dino_mod.EmbeddingStore(cat, path).loaded == 0
    monkeypatch.undo()
    monkeypatch.setattr(dino_mod, 'EMB_VERSION', dino_mod.EMB_VERSION + 1)
    assert dino_mod.EmbeddingStore(cat, path).loaded == 0


def test_embedding_runs_in_bounded_chunks(world, fake_dino, tmp_path, monkeypatch):
    cat, _ = world
    sizes = []
    real = dino_mod.embed
    monkeypatch.setattr(dino_mod, 'embed', lambda imgs, batch=64: (sizes.append(len(imgs)), real(imgs))[1])
    monkeypatch.setattr(dino_mod, '_CHUNK', 5)
    dino_mod.EmbeddingStore(cat, str(tmp_path / 'e.npz')).precompute(log=lambda *a: None)
    assert sizes and max(sizes) <= 5 and sum(sizes) == len(cat)


# ---- warm-up ---------------------------------------------------------------------------------

def test_warm_does_the_one_time_work_so_the_first_scan_does_none(world, fake_dino, tmp_path, monkeypatch):
    cat, img, n = _synthetic(world)
    eng = Engine(EngineSettings(use_dino=True, use_ocr=False, accelerate=False), catalog=cat)
    monkeypatch.setattr(eng, '_use_dino', lambda: True)
    eng.store = dino_mod.EmbeddingStore(cat, str(tmp_path / 'emb.npz'))
    out = eng.warm(log=lambda *a: None)
    assert out['dino'] and out['footprints'] == len(cat.stacks) and out['embedded'] >= 0
    assert eng._builds_done and eng.store.missing(range(len(cat))) == []
    before = fake_dino['n']
    stacks_before = len(cat._rot_cache)
    eng.scan(img)
    # the scan embedded only the tiles it was asked about (queries + rotated / clipped candidates), never catalog icons
    assert eng.store.missing(range(len(cat))) == [] and len(cat._rot_cache) == stacks_before
    assert fake_dino['n'] - before <= 3 * len(cat)


def test_warm_without_dino_still_primes_stage_one(world):
    cat, _ = world
    eng = engine(cat)                                               # use_dino=False
    out = eng.warm(log=lambda *a: None)
    assert out['dino'] is False and out['footprints'] == len(cat.stacks)
    assert len(cat._rot_cache) > 0 or len(cat.stacks) > 0
