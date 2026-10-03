"""Live view: raid advice (grab / drop), tile cache, SSE stream and the live trigger timing."""
import json

import numpy as np
import pytest

import liveadvice as adv
import sellcalc
from autoscan import trigger as trg
from autoscan.service import AutoScanner, make_blueprint
from identify import tilecache

CTX = sellcalc.make_context({})
SETTINGS = {'live_top_n': 5, 'live_min_value_per_slot': 10000}


def item(iid, name, trader=0, flea=0, types=()):
    sell = []
    if trader:
        sell.append({'vendor': {'name': 'Therapist'}, 'priceRUB': trader, 'currency': 'RUB'})
    if flea:
        sell.append({'vendor': {'name': 'Flea Market'}, 'priceRUB': flea, 'currency': 'RUB'})
    return {'id': iid, 'name': name, 'basePrice': trader or flea, 'lastLowPrice': flea,
            'avg24hPrice': flea, 'sellFor': sell, 'types': list(types)}


ITEMS = {i['id']: i for i in [
    item('gpu', 'Graphics card', trader=50000, flea=160000),
    item('lion', 'Golden lion', trader=20000, flea=90000),
    item('bolt', 'Bolts', trader=1000, flea=2000),
    item('tushonka', 'Tushonka', trader=3000, flea=6000),
    item('cpu', 'CPU fan', trader=7000, flea=0),
    item('big', 'Big thing', trader=60000, flea=0),
    item('nofl', 'Banned thing', trader=40000, flea=900000, types=['noFlea']),
    item('key', 'Quest key', trader=500, flea=800),
    item('cell', 'Hideout item', trader=400, flea=700),
    item('rock', 'Pebble', trader=100, flea=0),
    item('docs', 'Documents', trader=9000, flea=0),
    item('gun', 'Some gun', trader=70000, flea=0, types=['gun']),
]}


def det(iid, role, col=0, row=0, w=1, h=1, panel=0, count=1, uncertain=False, fir=None, category=None):
    return {'item_id': iid, 'role': role, 'col': col, 'row': row, 'W': w, 'H': h, 'panel': panel,
            'count': count, 'uncertain': uncertain, 'fir': fir, 'category': category,
            'px': col * 10 + panel * 1000, 'py': row * 10, 'pw': w * 10, 'ph': h * 10}


def run(dets, grids=None, protected=None, any_of=None, settings=None):
    return adv.compute_advice(dets, grids or [], ITEMS, protected or {}, any_of or [],
                              settings or SETTINGS, CTX, scene='raid_loot', in_raid=True)


# ---- grab ------------------------------------------------------------------------------------

def test_not_a_raid_means_no_advice():
    assert adv.compute_advice([det('gpu', 'loot')], [], ITEMS, {}, [], SETTINGS, CTX, scene='lobby_stash') is None


def test_grab_ranks_by_value_per_slot_best_of_trader_and_flea():
    dets = [det('lion', 'loot', col=0, w=1, h=1), det('gpu', 'loot', col=1, w=1, h=1),
            det('big', 'loot', col=2, w=2, h=2), det('bolt', 'loot', col=4)]
    out = run(dets)['grab']
    names = [g['name'] for g in out]
    assert names[0] == 'Graphics card'                         # flea net beats the trader's 50k
    assert 'Bolts' not in names                                # below 10k per slot
    big = next(g for g in out if g['name'] == 'Big thing')
    assert big['slots'] == 4 and big['per_slot'] == 15000      # trader-only item: 60k over 4 slots
    assert names.index('Golden lion') < names.index('Big thing')


def test_flea_not_counted_for_noflea_items_and_guns_not_ranked():
    out = run([det('nofl', 'loot'), det('gun', 'loot', col=1, category='weapon')])['grab']
    assert [g['name'] for g in out] == ['Banned thing']
    assert out[0]['value'] == 40000                            # trader only, the 900k flea price is not available


def test_stacks_multiply_and_top_n_and_threshold_come_from_settings():
    dets = [det('tushonka', 'loot', count=5), det('lion', 'loot', col=1), det('gpu', 'loot', col=2)]
    out = run(dets)['grab']
    assert next(g for g in out if g['name'] == 'Tushonka')['value'] >= 5 * 3000
    assert len(run(dets, settings={**SETTINGS, 'live_top_n': 2})['grab']) == 2
    assert len(run(dets, settings={**SETTINGS, 'live_min_value_per_slot': 10 ** 6})['grab']) == 0


def test_quest_items_come_first_whatever_they_are_worth_and_are_flagged():
    protected = {'key': {'kinds': ['task'], 'need': 1}, 'cell': {'kinds': ['hideout'], 'need': 1}}
    dets = [det('gpu', 'loot'), det('key', 'loot', col=1), det('cell', 'loot', col=2), det('rock', 'loot', col=3)]
    out = run(dets, protected=protected)['grab']
    assert [g['name'] for g in out][:3] == ['Quest key', 'Hideout item', 'Graphics card'] or \
        [g['name'] for g in out][:3] == ['Hideout item', 'Quest key', 'Graphics card']
    flags = {g['name']: g['flag'] for g in out}
    assert flags['Quest key'] == 'Quest' and flags['Hideout item'] == 'Hideout' and flags['Graphics card'] is None
    assert 'Pebble' not in flags


def test_any_of_objective_items_count_as_quest_items():
    out = run([det('docs', 'loot', col=0), det('gpu', 'loot', col=1)],
              any_of=[{'items': [{'id': 'rock'}, {'id': 'cpu'}]}], settings={**SETTINGS, 'live_min_value_per_slot': 10 ** 6})
    assert out['grab'] == []
    out = run([det('cpu', 'loot')], any_of=[{'items': [{'id': 'cpu'}]}], settings={**SETTINGS, 'live_min_value_per_slot': 10 ** 6})
    assert [g['flag'] for g in out['grab']] == ['Quest']


def test_uncertain_identifications_are_never_ranked():
    out = run([det('gpu', 'loot', uncertain=True), det('lion', 'loot', col=1)])['grab']
    assert [g['name'] for g in out] == ['Golden lion']
    out = run([det('key', 'loot', uncertain=True)], protected={'key': {'kinds': ['task']}})['grab']
    assert out == []


def test_only_loot_regions_are_grab_candidates():
    out = run([det('gpu', 'own_backpack'), det('gpu', 'stash', col=1), det('lion', 'container_window', col=2)])['grab']
    assert [g['name'] for g in out] == ['Golden lion']


# ---- drop ------------------------------------------------------------------------------------

def grid(role, cols, rows, items, panel=0):
    """items: [(col, row, w, h, iid or None)] -> (grid dict, own detections)."""
    cells, dets = [], []
    taken = set()
    for col, row, w, h, iid in items:
        cells.append((col, row, w, h, False))
        for r in range(row, row + h):
            for c in range(col, col + w):
                taken.add((c, r))
        if iid:
            dets.append(det(iid, role, col, row, w, h, panel=panel))
    for r in range(rows):
        for c in range(cols):
            if (c, r) not in taken:
                cells.append((c, r, 1, 1, True))
    return {'panel': panel, 'role': role, 'cols': cols, 'rows': rows, 'cells': cells}, dets


def full_rig(extra=()):
    # 4x1 rig completely full of cheap things
    return grid('own_rig', 4, 1, [(0, 0, 1, 1, 'bolt'), (1, 0, 1, 1, 'tushonka'), (2, 0, 1, 1, 'rock'),
                                  (3, 0, 1, 1, 'cpu')] + list(extra))


def test_no_drop_while_there_is_room():
    g, own = grid('own_rig', 4, 1, [(0, 0, 1, 1, 'bolt')])
    out = run([det('gpu', 'loot')] + own, [g])
    assert out['drop'] == [] and out['free_cells'] == 3


def test_drop_the_cheapest_per_slot_when_full_and_the_grab_is_worth_more():
    g, own = full_rig()
    out = run([det('gpu', 'loot')] + own, [g])
    assert [d['name'] for d in out['drop']] == ['Pebble']     # one cell is enough, the cheapest per slot goes
    assert out['grab'][0]['name'] == 'Graphics card'


def test_drop_enough_items_for_a_bigger_grab_but_not_more():
    g, own = full_rig()
    out = run([det('big', 'loot', w=2, h=2)] + own, [g], settings={**SETTINGS, 'live_min_value_per_slot': 0})
    # no 2x2 block exists in a 4x1 rig even when it is empty -> nothing can make room
    assert out['drop'] == [] and 'enough room' in out['note']
    g, own = grid('own_backpack', 2, 2, [(0, 0, 1, 1, 'bolt'), (1, 0, 1, 1, 'rock'), (0, 1, 1, 1, 'tushonka'), (1, 1, 1, 1, 'cpu')])
    out = run([det('big', 'loot', w=1, h=2)] + own, [g])
    names = [d['name'] for d in out['drop']]
    assert len(names) == 2 and set(names) <= {'Pebble', 'Bolts', 'Tushonka'} and 'CPU fan' not in names


def test_never_drop_pouch_special_slots_quest_items_or_guesses():
    pouch, p_own = grid('own_pouch', 2, 1, [(0, 0, 1, 1, 'rock'), (1, 0, 1, 1, 'bolt')], panel=1)
    special, s_own = grid('own_special', 3, 1, [(0, 0, 1, 1, 'rock')], panel=2)
    rig, own = grid('own_rig', 2, 1, [(0, 0, 1, 1, 'key'), (1, 0, 1, 1, 'rock')])
    own[1]['uncertain'] = True                                 # unsure what the pebble is: not touchable
    out = run([det('gpu', 'loot')] + own + p_own + s_own, [pouch, special, rig],
              protected={'key': {'kinds': ['task']}})
    assert out['drop'] == [] and out['free_cells'] == 0
    own[1]['uncertain'] = False
    out = run([det('gpu', 'loot')] + own + p_own + s_own, [pouch, special, rig], protected={'key': {'kinds': ['task']}})
    assert [(d['name'], d['panel']) for d in out['drop']] == [('Pebble', 0)]      # the rig's own pebble, never the pouch's


def test_no_drop_when_the_grab_is_not_worth_more_than_what_goes():
    g, own = grid('own_rig', 2, 1, [(0, 0, 1, 1, 'gpu'), (1, 0, 1, 1, 'gpu')])
    out = run([det('lion', 'loot')] + own, [g])
    assert out['drop'] == [] and 'worth swapping' in out['note']


def test_a_quest_grab_is_always_worth_a_swap():
    g, own = full_rig()
    out = run([det('key', 'loot')] + own, [g], protected={'key': {'kinds': ['task']}})
    assert [d['name'] for d in out['drop']] == ['Pebble']


# ---- tile cache ------------------------------------------------------------------------------

def test_tile_key_changes_with_pixels_not_with_pitch_jitter():
    img = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    k = tilecache.tile_key(img, (10, 10, 64, 64), 63.002, 63.004)
    assert k == tilecache.tile_key(img, (10, 10, 64, 64), 63.0, 63.0)
    assert k != tilecache.tile_key(img, (10, 10, 64, 64), 70.0, 63.0)
    img2 = img.copy()
    img2[20, 20] ^= 1
    assert k != tilecache.tile_key(img2, (10, 10, 64, 64), 63.0, 63.0)
    assert tilecache.tile_key(img, (190, 190, 64, 64), 63.0, 63.0) is None


def test_tile_cache_hit_miss_and_lru_eviction():
    c = tilecache.TileCache(maxsize=3)
    assert c.get('a') is None and c.misses == 1
    for k in 'abc':
        c.put(k, {'v': k})
    assert c.get('a') == {'v': 'a'} and c.hits == 1             # touching 'a' makes 'b' the oldest
    c.put('d', {'v': 'd'})
    assert len(c) == 3 and c.get('b') is None and c.get('a') and c.get('d')
    c.put(None, 1)
    assert c.get(None) is None and len(c) == 3
    c.clear()
    assert len(c) == 0 and c.hits == 0


# ---- trigger timing --------------------------------------------------------------------------

def _thumb(seed, patch=None):
    t = np.random.default_rng(seed).integers(0, 255, trg.THUMB[::-1], dtype=np.uint8)
    if patch:
        y, x, hh, ww, v = patch
        t[y:y + hh, x:x + ww] = v
    return t


class Feed:
    def __init__(self, **kw):
        self.t, self.now = trg.Trigger(**kw), 0.0

    def step(self, thumb, dt=0.25, **kw):
        self.now += dt
        d = self.t.observe(trg.Observation(True, kw.get('chrome', True), thumb), self.now)
        if d.scan:
            self.t.scan_started(thumb, self.now)
        return d.state


def test_view_is_scanned_two_polls_after_it_stops_changing():
    f = Feed()
    a = _thumb(1)
    assert f.step(a) == 'settling' and f.step(a) == 'scan'      # 0.25 s apart: settled in half a second
    assert f.step(a) == 'scanned'


def test_a_small_change_rescans_without_the_old_3_second_gap():
    f = Feed()
    a = _thumb(1)
    f.step(a), f.step(a)
    # one cell's worth of pixels changes (a moved item): hash barely moves, pixels do
    b = a.copy()
    b[10:15, 10:15] = 255 - b[10:15, 10:15]
    assert f.step(b) == 'settling'
    assert f.step(b) == 'scan'                                  # 0.5 s after the previous scan, not 3 s


def test_tiny_jitter_does_not_rescan():
    f = Feed()
    a = _thumb(1)
    f.step(a), f.step(a)
    b = a.copy()
    b[0, 0] = 255 - b[0, 0]
    assert [f.step(b) for _ in range(3)] == ['scanned'] * 3


def test_raid_inventory_waits_unless_allowed():
    a = _thumb(1)
    f = Feed(allow_raid=False)
    assert f.step(a, chrome=False) == 'raid_inventory'
    f = Feed(allow_raid=True)
    f.step(a, chrome=False)
    assert f.step(a, chrome=False) == 'scan'


# ---- SSE -------------------------------------------------------------------------------------

class _Cap:
    backend_name = 'fake'

    def close(self):
        pass


def make_app(scanner=None):
    from flask import Flask
    scanner = scanner or AutoScanner(lambda f: {'results': []}, lambda: {'auto_scan': True}, capture=_Cap(), win32=object())
    app = Flask(__name__)
    app.register_blueprint(make_blueprint(scanner, lambda: {}, lambda s: None))
    return scanner, app


def test_stream_announces_the_current_seq_and_a_new_result():
    import threading
    scanner, app = make_app()
    scanner._seq, scanner._result, scanner._last_scan_wall = 4, {'scene': 'raid_loot', 'results': []}, 1.0

    def publish():
        import time
        time.sleep(0.15)
        with scanner._lock:
            scanner._seq = 5
        with scanner._cond:
            scanner._cond.notify_all()
    threading.Thread(target=publish).start()
    body = app.test_client().get('/api/autoscan/stream?max_s=0.6').get_data(as_text=True)
    events = [json.loads(l[6:]) for l in body.splitlines() if l.startswith('data: ')]
    assert [e['seq'] for e in events] == [4, 5] and events[0]['scene'] == 'raid_loot'
    assert body.startswith('retry:')


def test_stream_is_sse_and_result_is_lean_with_a_cached_frame():
    scanner, app = make_app()
    scanner._seq, scanner._result, scanner._last_scan_wall = 2, {'results': [], 'image': None}, 1.0
    scanner._frame = np.full((40, 60, 3), 90, np.uint8)
    c = app.test_client()
    r = c.get('/api/autoscan/stream?max_s=0.1')
    assert r.mimetype == 'text/event-stream' and r.headers['Cache-Control'] == 'no-cache'
    res = c.get('/api/autoscan/result').get_json()
    assert res['seq'] == 2 and res['frame_url'] == '/api/autoscan/frame?seq=2' and not res.get('image')
    f = c.get(res['frame_url'])
    assert f.status_code == 200 and f.mimetype == 'image/jpeg' and f.data[:2] == b'\xff\xd8'
    assert 'immutable' in f.headers['Cache-Control']
    assert c.get('/api/autoscan/frame?seq=1').status_code == 404      # an old picture is gone


def test_service_scans_the_raid_inventory_when_live_in_raid_is_on():
    from tests.test_autoscan import make_scanner, inventory_frame, run as run_polls
    s, calls, _ = make_scanner(inventory_frame(chrome=False), settings={'auto_scan': True, 'live_in_raid': True})
    run_polls(s, 6)
    assert len(calls) == 1 and s.status()['state'] == 'scanned'
