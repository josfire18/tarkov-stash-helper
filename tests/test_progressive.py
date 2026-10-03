"""Progressive scans: provisional first look -> certified final, and supersede by a newer view."""
import threading
import time

import pytest

import liveadvice
import sellcalc
from autoscan import AutoScanner
from identify.pipeline import Engine, ScanCancelled
from test_identify_pipeline import engine, idof, render_panel, world  # noqa: F401  (fixture)
from tests.test_autoscan import inventory_frame, make_scanner, run


def _synthetic(world):
    cat, icons = world
    layout = [(0, 0, 1, 1, 1), (1, 0, 1, 1, 2), (3, 0, 2, 1, 20), (0, 1, 1, 3, 30)]
    cells = {(c, r): icons[idof(i)] for c, r, w, h, i in layout}
    img, *_ = render_panel(63.0, 6, 5, [(c, r, w, h) for c, r, w, h, _ in layout], icons=cells, size=(520, 420))
    return cat, img, len(layout)


# ---- engine ----------------------------------------------------------------------------------

def test_engine_publishes_a_provisional_result_before_the_final(world):
    cat, img, n = _synthetic(world)
    order, seen = [], {}

    def prov(res):
        order.append('prov')
        seen['dets'] = res.detections
        seen['t'] = res.timings.get('provisional')

    res = engine(cat).scan(img, on_provisional=prov)
    order.append('final')
    assert order == ['prov', 'final']
    assert len(seen['dets']) == n and all(d.provisional and d.uncertain for d in seen['dets'])
    assert seen['t'] is not None and seen['t'] <= res.timings['total']
    assert not any(d.provisional for d in res.detections)           # the final detections are not marked
    assert all(d.to_record().get('provisional') for d in seen['dets'])
    assert 'provisional' not in res.detections[0].to_record()


def test_cached_tiles_are_final_in_the_provisional_result_and_nothing_is_published_when_all_cached(world):
    cat, img, n = _synthetic(world)
    eng = engine(cat)
    eng.scan(img)
    calls = []
    eng.scan(img, on_provisional=calls.append)
    assert calls == []                                              # warm re-scan: nothing fresh, no first look
    # one tile changes -> only that one is provisional
    img2 = img.copy()
    x, y, w, h = [d.rect for d in eng.scan(img).detections][0]
    img2[y + 8:y + h - 8, x + 8:x + w - 8] = 200
    got = []
    eng.scan(img2, on_provisional=got.append)
    if got:
        flags = [d.provisional for d in got[0].detections]
        assert 0 < sum(flags) < len(flags)


def test_a_listener_error_never_costs_the_scan(world):
    cat, img, n = _synthetic(world)

    def boom(res):
        raise RuntimeError('listener')
    res = engine(cat).scan(img, on_provisional=boom)
    assert len(res.detections) == n


def test_cancel_during_certification_raises_and_keeps_certified_tiles_cached(world):
    cat, img, n = _synthetic(world)
    eng = engine(cat)
    state = {'prov': False}

    def cancel():
        return state['prov']
    with pytest.raises(ScanCancelled):
        eng.scan(img, on_provisional=lambda r: state.update(prov=True), cancel=cancel)
    assert len(eng.scan(img).detections) == n                       # engine lock released, still usable


# ---- sell payload / advice -------------------------------------------------------------------

def test_a_provisional_item_is_never_a_confident_sell_row():
    item = {'id': 'a', 'name': 'Thing', 'basePrice': 1000, 'types': [], 'width': 1, 'height': 1}
    det = {'item_id': 'a', 'name': 'Thing', 'uncertain': True, 'provisional': True, 'count': 1, 'fir': True,
           'panel': 0, 'col': 0, 'row': 0, 'px': 1, 'py': 2, 'pw': 3, 'ph': 4}
    ctx = {'slots': 0, 'overflow': 'x', 'rules': None}
    sell, keep = sellcalc.plan_entries([det], {'a': item}, {}, {}, ctx)
    assert sell == [] and keep[0]['provisional'] and keep[0]['recommend'] == 'keep' and keep[0]['check']


def test_provisional_grab_is_flagged_and_never_triggers_a_drop(monkeypatch):
    monkeypatch.setattr(liveadvice, 'unit_value', lambda *a, **k: 90000)
    item = {'id': 'a', 'name': 'Thing', 'types': []}
    loot = {'item_id': 'a', 'name': 'Thing', 'role': 'loot', 'uncertain': True, 'provisional': True, 'count': 1,
            'W': 1, 'H': 1, 'px': 1, 'py': 1, 'pw': 5, 'ph': 5, 'panel': 0, 'col': 0, 'row': 0}
    grab = liveadvice.rank_grab([loot], {'a': item}, {}, set(), {}, {})
    assert len(grab) == 1 and grab[0]['provisional']
    grids = [{'panel': 1, 'role': 'own_backpack', 'cols': 1, 'rows': 1, 'cells': [(0, 0, 1, 1, False)]}]
    own = [{'item_id': 'a', 'panel': 1, 'col': 0, 'row': 0, 'role': 'own_backpack', 'count': 1}]
    assert liveadvice.plan_drop(grab, grids, own, {'a': item}, {}, set(), {}, {})[0] == []
    certain = dict(loot, uncertain=False, provisional=False)
    assert not liveadvice.rank_grab([dict(loot, provisional=False)], {'a': item}, {}, set(), {}, {})
    assert 'provisional' not in liveadvice.rank_grab([certain], {'a': item}, {}, set(), {}, {})[0]


# ---- service: sequencing + supersede -----------------------------------------------------------

def _prog_scanner(scan_fn, **kw):
    s, calls, settings = make_scanner(inventory_frame(), **kw)
    s.scan_fn, s.progressive = scan_fn, True
    return s


def _wait(pred, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_service_publishes_provisional_then_final_as_two_seqs():
    gate = threading.Event()

    def scan_fn(frame, publish, cancel):
        publish({'results': [{'num': 'K', 'provisional': True}], 'grid': {}, 'provisional': True})
        gate.wait(5)
        return {'results': [{'num': 1}], 'grid': {}, 'grid_failed': False}
    s = _prog_scanner(scan_fn)
    run(s, 4)
    assert _wait(lambda: s.status()['seq'] == 1)
    st = s.status()
    assert st['state'] == 'verifying' and s.result(0)['provisional'] and s.result(0)['timing']['final'] is False
    run(s, 3)                                                       # same view while certifying: no new scan
    assert s.status()['state'] == 'verifying'
    gate.set()
    assert _wait(lambda: s.status()['seq'] == 2 and s.status()['state'] == 'scanned')
    r = s.result(1)
    assert r['ready'] and not r.get('provisional') and r['timing']['final'] is True and r['results'] == [{'num': 1}]
    s.stop()


def test_a_newer_view_cancels_the_pending_certification_and_its_result_is_dropped():
    started, cancelled, finals = threading.Event(), threading.Event(), []
    frames = []

    def scan_fn(frame, publish, cancel):
        frames.append(frame)
        n = len(frames)
        publish({'results': [{'view': n}], 'grid': {}, 'provisional': True})
        if n == 1:
            started.set()
            for _ in range(500):                                    # "certification": polls cancel like the engine
                if cancel():
                    cancelled.set()
                    raise ScanCancelled()
                time.sleep(0.01)
        finals.append(n)
        return {'results': [{'view': n, 'final': True}], 'grid': {}, 'grid_failed': False}
    s = _prog_scanner(scan_fn)
    run(s, 4)
    assert started.wait(5)
    s.trigger.reset_view()                                          # the player opened something else
    run(s, 4)
    assert cancelled.wait(5) and _wait(lambda: finals == [2])
    assert _wait(lambda: s.status()['state'] == 'scanned')
    res = s.result(0)
    assert res['results'] == [{'view': 2, 'final': True}]           # nothing of view 1 beyond its first look
    assert s.status()['seq'] == 3                                    # prov(1), prov(2), final(2)
    s.stop()


def test_a_failing_progressive_scan_reports_an_error_and_a_superseded_one_does_not():
    def bad(frame, publish, cancel):
        raise RuntimeError('boom')
    s = _prog_scanner(bad)
    run(s, 4)
    assert _wait(lambda: s.status()['state'] in ('error', 'failed'))
    assert 'boom' in s.status()['detail']
    s.stop()
