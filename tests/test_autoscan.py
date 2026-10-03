"""autoscan: window/monitor selection, black frames, inventory detector, stability/dedup,
trigger state machine, capture fallback, service loop and endpoints (no game, no GPU needed)."""
import contextlib
import os

import cv2
import numpy as np
import pytest

from autoscan import AutoScanner, make_blueprint
from autoscan import capture as cap
from autoscan import trigger as trg
from autoscan.detect import InventoryDetector, frame_is_blank, menu_chrome
from autoscan.winapi import (GameWindow, MonitorInfo, WinInfo, locate_game, our_window_covers,
                             pick_game_window, pick_monitor)
from autoscan_frames import inventory_frame, raid_frame

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
M1 = MonitorInfo(1, r'\\.\DISPLAY1', (0, 0, 2560, 1440), True)
M2 = MonitorInfo(2, r'\\.\DISPLAY2', (2560, 0, 4480, 1080), False)


def win(hwnd, pid, cls='UnityWndClass', vis=True, iconic=False, client=(0, 0, 2560, 1440), title='EscapeFromTarkov'):
    return WinInfo(hwnd, pid, cls, title, vis, iconic, client, client)


class FakeWin32:
    def __init__(self, pids=(100,), windows=(), monitors=(M1, M2), fg=0, mon_of=None):
        self.pids, self.windows, self._mons, self.fg, self.mon_of = list(pids), list(windows), list(monitors), fg, mon_of

    def find_pids(self, exe):
        return self.pids if exe.lower() == 'escapefromtarkov.exe' else []

    def top_level_windows(self):
        return self.windows

    def monitors(self):
        return self._mons

    def monitor_of(self, hwnd):
        return self.mon_of

    def foreground(self):
        return self.fg

    def physical_pixels(self):
        return contextlib.nullcontext()


# ---------------------------------------------------------------- window / monitor selection
def test_picks_the_visible_unity_window_not_the_helpers():
    helpers = [win(1, 100, 'InvisibleWindowClassNvPresent', vis=False, client=(130, 130, 314, 291)),
               win(2, 100, 'IME', vis=False, client=(0, 0, 0, 0))]
    assert pick_game_window(helpers + [win(3, 100)], {100}).hwnd == 3


def test_ignores_other_processes_and_small_windows():
    assert pick_game_window([win(1, 999)], {100}) is None
    assert pick_game_window([win(1, 100, client=(0, 0, 300, 200))], {100}) is None


def test_visible_beats_minimised_and_bigger_beats_smaller():
    a = win(1, 100, iconic=True)
    b = win(2, 100, cls='Other', client=(0, 0, 1280, 720))
    c = win(3, 100, cls='Other', client=(0, 0, 1920, 1080))
    assert pick_game_window([a, b], {100}).hwnd == 2
    assert pick_game_window([b, c], {100}).hwnd == 3


def test_monitor_is_the_one_showing_most_of_the_window():
    assert pick_monitor((2500, 0, 3500, 800), [M1, M2]).device == M2.device
    assert pick_monitor((0, 0, 2560, 1440), [M2, M1]).device == M1.device


def test_region_is_relative_to_the_monitor_and_clipped():
    g = GameWindow(1, 100, 't', (2560, 0, 4480, 1080), M2, False)
    assert g.region_on_monitor() == (0, 0, 1920, 1080)
    g2 = GameWindow(1, 100, 't', (-100, 10, 1000, 700), M1, False)
    assert g2.region_on_monitor() == (0, 10, 1000, 700)


def test_locate_game_end_to_end_with_fake_win32():
    g = locate_game(FakeWin32(windows=[win(3, 100)], mon_of=M1, fg=3), our_pid=555)
    assert (g.hwnd, g.monitor.device, g.iconic, g.covered_by_us) == (3, M1.device, False, False)
    assert locate_game(FakeWin32(pids=()), our_pid=555) is None
    assert locate_game(FakeWin32(windows=[win(1, 100, vis=False, client=(0, 0, 0, 0))]), our_pid=555) is None


def test_our_own_window_covering_the_game_is_detected():
    ours = WinInfo(9, 555, 'Chrome_WidgetWin_1', 'Tarkov Stash Helper', True, False, (100, 100, 1300, 960), (100, 100, 1300, 960))
    assert our_window_covers(ours, 555, (0, 0, 2560, 1440))
    away = WinInfo(9, 555, 'x', 't', True, False, (2600, 0, 3800, 900), (2600, 0, 3800, 900))
    assert not our_window_covers(away, 555, (0, 0, 2560, 1440))          # on the second monitor
    assert not our_window_covers(ours, 777, (0, 0, 2560, 1440))          # someone else's window
    w = FakeWin32(windows=[win(3, 100), ours], mon_of=M1, fg=9)
    assert locate_game(w, our_pid=555).covered_by_us


# ---------------------------------------------------------------- black frames
def test_blank_frame_detection():
    assert frame_is_blank(np.zeros((1440, 2560, 3), np.uint8))
    assert frame_is_blank(None)
    assert not frame_is_blank(raid_frame(1280, 720))
    assert not frame_is_blank(raid_frame(1280, 720, dark=True))          # a dark scene is not a black capture


# ---------------------------------------------------------------- inventory detector
@pytest.mark.parametrize('w,h', [(1280, 720), (1920, 1080), (2560, 1440)])
def test_inventory_detected_at_several_resolutions(w, h):
    det = InventoryDetector().detect(inventory_frame(w, h))
    assert det.is_inventory and det.menu_chrome
    assert det.x.pitch == pytest.approx(63.0 * h / 1080, rel=0.03)


def test_fixtures_on_disk():
    d = InventoryDetector()
    assert d.detect(cv2.imread(os.path.join(FIX, 'autoscan_inventory_720p.png'))).is_inventory
    assert not d.detect(cv2.imread(os.path.join(FIX, 'autoscan_raid_720p.png'))).is_inventory


@pytest.mark.parametrize('seed', range(6))
def test_raid_like_frames_are_never_inventories(seed):
    assert not InventoryDetector().detect(raid_frame(seed=seed)).is_inventory
    assert not InventoryDetector().detect(raid_frame(seed=seed, dark=True)).is_inventory


def test_small_or_black_frames_are_rejected():
    d = InventoryDetector()
    black = d.detect(np.zeros((1440, 2560, 3), np.uint8))
    assert not black.is_inventory and black.reason == 'blank frame'
    assert not d.detect(np.full((200, 300, 3), 80, np.uint8)).is_inventory


def test_menu_bar_separates_lobby_from_raid_inventory():
    assert menu_chrome(inventory_frame(chrome=True))
    assert not menu_chrome(inventory_frame(chrome=False))
    assert not InventoryDetector().detect(inventory_frame(chrome=False)).menu_chrome
    assert not menu_chrome(raid_frame(dark=True))        # black rows but no lit bar between them


def test_detector_is_fast_enough():
    f, d = inventory_frame(), InventoryDetector()
    d.detect(f)
    assert min(d.detect(f).ms for _ in range(5)) < 40    # loose CI bound; ~7-10 ms on the dev machine


# ---------------------------------------------------------------- stability / dedup
def _scrolled(dy):
    f = inventory_frame()
    return np.roll(f[:1300], dy, axis=0) if dy else f[:1300]


def test_stability_tolerates_a_blinking_cursor_but_not_a_scroll():
    a = trg.thumbnail(inventory_frame())
    b = inventory_frame()
    cv2.rectangle(b, (1000, 600), (1030, 630), (255, 255, 255), -1)
    assert trg.is_stable(a, trg.thumbnail(b))
    assert not trg.is_stable(a, trg.thumbnail(_scrolled(84)))
    assert not trg.is_stable(None, a)


def test_same_view_hash_ignores_tooltips_but_sees_a_scroll():
    f = inventory_frame()
    h0 = trg.view_hash(trg.thumbnail(f))
    tip = f.copy()
    cv2.rectangle(tip, (1300, 500), (1550, 640), (12, 12, 12), -1)
    assert trg.same_view(h0, trg.view_hash(trg.thumbnail(tip)))
    assert not trg.same_view(h0, trg.view_hash(trg.thumbnail(_scrolled(84))))
    assert not trg.same_view(None, h0)


# ---------------------------------------------------------------- trigger state machine
_frames = {'A': inventory_frame(), 'B': _scrolled(84), 'C': inventory_frame(origin=(500, 300)), 'raid': raid_frame()}
_thumbs = {k: trg.thumbnail(v) for k, v in _frames.items()}


class Feed:
    """Drives Trigger with synthetic observations on a fake clock."""

    def __init__(self, **kw):
        self.t, self.now = trg.Trigger(**kw), 0.0

    def step(self, key, inventory=True, chrome=True, blank=False, dt=0.5):
        self.now += dt
        d = self.t.observe(trg.Observation(inventory, chrome, _thumbs[key], blank), self.now)
        if d.scan:
            self.t.scan_started(_thumbs[key], self.now)
        return d.state


def test_scan_fires_once_when_the_view_settles_then_goes_quiet():
    f = Feed()
    assert f.step('A') == 'settling'                  # first sighting: nothing to compare with
    assert f.step('A') == 'scan'                      # two identical polls
    assert [f.step('A') for _ in range(4)] == ['scanned'] * 4


def test_scrolling_gives_one_scan_per_resting_position():
    f = Feed(min_gap_s=0.0)
    states = [f.step(k) for k in 'AAABBBCCC']         # rest, scroll, rest, scroll, rest
    assert states.count('scan') == 3
    f2 = Feed(min_gap_s=0.0)
    assert 'scan' not in [f2.step(k) for k in 'ABCABCABC']      # continuous scrolling: never stable


def test_min_gap_between_scans():
    f = Feed(min_gap_s=3.0)
    f.step('A')
    assert f.step('A') == 'scan'
    f.step('B')
    assert f.step('B') == 'settling'                  # stable and new, but only 1 s after the last scan
    assert f.step('B', dt=3.0) == 'scan'


def test_raid_gameplay_and_raid_inventory_never_scan():
    f = Feed()
    assert [f.step('raid', inventory=False) for _ in range(5)] == ['waiting'] * 5
    assert [f.step('A', chrome=False) for _ in range(5)] == ['raid_inventory'] * 5
    g = Feed(allow_raid=True)
    g.step('A', chrome=False)
    assert g.step('A', chrome=False) == 'scan'


def test_blank_frames_are_reported_not_scanned():
    assert Feed().step('A', blank=True) == 'blank'


def test_failed_scan_retries_only_after_the_retry_delay_or_a_view_change():
    f = Feed(retry_s=30.0)
    f.step('A')
    assert f.step('A') == 'scan'
    f.t.scan_failed(f.now)
    assert f.step('A') == 'failed'                    # same view: the failure is not retried at once
    assert f.step('A', dt=40) == 'scan'


# ---------------------------------------------------------------- capture fallback
class StubBackend:
    def __init__(self, name, frames=None, error=None):
        self.name, self.frames, self.error, self.closed = name, list(frames or []), error, 0
        self.last = None

    def grab(self, monitor, region):
        if self.error:
            raise cap.BackendError(self.error)
        self.last = self.frames.pop(0) if self.frames else self.last
        return cap.CaptureResult(self.last, self.name, blank=frame_is_blank(self.last))

    def close(self):
        self.closed += 1


BLACK, LIVE = np.zeros((720, 1280, 3), np.uint8), raid_frame(1280, 720)


def test_falls_back_when_the_primary_errors():
    m = cap.CaptureManager([StubBackend('dxgi', error='access lost'), StubBackend('gdi', [LIVE])])
    r = m.grab(M1, None)
    assert r.backend == 'gdi' and r.frame is not None and m.backend_name == 'gdi'
    assert 'access lost' in m.last_error


def test_falls_back_after_black_frames_and_retries_the_primary_later():
    now = [0.0]
    a, b = StubBackend('dxgi', [BLACK]), StubBackend('gdi', [LIVE])
    m = cap.CaptureManager([a, b], blank_limit=3, retry_s=60, clock=lambda: now[0])
    assert m.grab(M1, None).blank and m.grab(M1, None).blank      # 1st, 2nd black: stay on the primary
    r = m.grab(M1, None)                                          # 3rd: demote, gdi delivers
    assert r.backend == 'gdi' and not r.blank
    now[0] = 61.0
    a.frames = [LIVE]
    assert m.grab(M1, None).backend == 'dxgi'


def test_black_everywhere_is_reported_as_blank():
    m = cap.CaptureManager([StubBackend('dxgi', [BLACK]), StubBackend('gdi', [BLACK])], blank_limit=1)
    assert m.grab(M1, None).blank


def test_all_backends_failing_reports_an_error():
    r = cap.CaptureManager([StubBackend('a', error='x'), StubBackend('b', error='y')]).grab(M1, None)
    assert r.frame is None and r.error == 'y'


# ---------------------------------------------------------------- service + endpoints
class FakeCapture:
    backend_name = 'fake'

    def __init__(self, frame):
        self.frame, self.closed = frame, 0

    def grab(self, monitor, region):
        return cap.CaptureResult(self.frame, 'fake', blank=frame_is_blank(self.frame))

    def close(self):
        self.closed += 1


def make_scanner(frame, scan=None, settings=None, win32=None, busy=None):
    settings = settings if settings is not None else {'auto_scan': True, 'live_in_raid': False}
    clock, calls = {'t': 0.0}, []

    def scan_fn(f):
        calls.append(f.shape)
        return scan(f) if scan else {'image': 'img', 'results': [{'num': 1}], 'grid': {}, 'grid_failed': False}
    s = AutoScanner(scan_fn, lambda: settings, busy_fn=busy,
                    win32=win32 or FakeWin32(windows=[win(3, 100)], mon_of=M1), capture=FakeCapture(frame),
                    clock=lambda: clock['t'], wall=lambda: 1000.0 + clock['t'])
    s._clock = clock
    return s, calls, settings


def run(s, n, dt=0.5):
    out = []
    for _ in range(n):
        s._clock['t'] += dt
        out.append(s.step())
    return out


def test_service_scans_a_settled_stash_exactly_once_and_publishes_the_result():
    s, calls, _ = make_scanner(inventory_frame())
    run(s, 6)
    assert calls == [(1440, 2560, 3)]                 # the full frame, no region, once
    st = s.status()
    assert st['state'] == 'scanned' and st['seq'] == 1 and st['message'] == 'Stash scanned'
    assert s.result(0)['ready'] and s.result(0)['results'] == [{'num': 1}]
    assert not s.result(1)['ready']


@pytest.mark.parametrize('frame,state', [(raid_frame(), 'waiting'), (np.zeros((1440, 2560, 3), np.uint8), 'blank'),
                                         (inventory_frame(chrome=False), 'raid_inventory')])
def test_service_never_scans_gameplay_black_frames_or_the_raid_inventory(frame, state):
    s, calls, _ = make_scanner(frame)
    run(s, 6)
    assert calls == [] and s.status()['state'] == state


def test_service_states_for_missing_game_minimised_disabled_busy():
    s, _, _ = make_scanner(inventory_frame(), win32=FakeWin32(pids=()))
    run(s, 1)
    assert s.status()['state'] == 'no_game'
    s, _, _ = make_scanner(inventory_frame(), win32=FakeWin32(windows=[win(3, 100, iconic=True)], mon_of=M1))
    run(s, 1)
    assert s.status()['state'] == 'minimized'
    s, _, st = make_scanner(inventory_frame(), settings={'auto_scan': False})
    run(s, 1)
    assert s.status()['state'] == 'disabled' and s.capture.closed
    s, calls, _ = make_scanner(inventory_frame(), busy=lambda: True)
    run(s, 4)
    assert s.status()['state'] == 'busy' and calls == []


def test_polling_never_exceeds_2hz_and_backs_off_in_gameplay():
    delays = run(make_scanner(raid_frame())[0], 40)
    assert min(delays) >= 0.5 and delays[0] == 1.0 and delays[-1] == 2.0


def test_failed_scan_is_reported_and_not_published():
    s, _, _ = make_scanner(inventory_frame(), scan=lambda f: {'error': 'boom', 'image': None, 'results': []})
    run(s, 4)
    assert s.status()['state'] == 'failed' and 'boom' in s.status()['detail'] and not s.result(0)['ready']


def test_endpoints():
    from flask import Flask
    store = {'auto_scan': True}
    s, _, _ = make_scanner(inventory_frame(), settings=store)
    run(s, 6)
    app = Flask(__name__)
    app.register_blueprint(make_blueprint(s, lambda: dict(store), lambda d: (store.clear(), store.update(d))))
    c = app.test_client()
    st = c.get('/api/autoscan/status').get_json()
    assert st['state'] == 'scanned' and st['enabled'] and st['seq'] == 1
    assert c.get('/api/autoscan/result?since=0').get_json()['ready'] is True
    assert c.get('/api/autoscan/result?since=1').get_json()['ready'] is False
    assert c.get('/api/autoscan/result?since=zzz').get_json()['ready'] is True
    assert c.post('/api/autoscan/toggle', json={'enabled': False}).get_json() == {'ok': True, 'enabled': False}
    assert store['auto_scan'] is False
    assert c.get('/api/autoscan/status').get_json()['enabled'] is False
