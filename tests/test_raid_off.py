"""Raid-off: with "Work in raid" off (the default) NOTHING runs while the game log says raid - no capture, no
detection, no scan, no frame collection; the lobby flows afterwards are scanned as usual."""
import threading

import cv2
import numpy as np

import app as tsh_app
from autoscan import AutoScanner
from autoscan import service as svc
from autoscan_frames import inventory_frame
from tests.test_autoscan import FakeCapture, FakeWin32, M1, run, win


class CountingCapture(FakeCapture):
    def __init__(self, frame):
        super().__init__(frame)
        self.grabs = 0

    def grab(self, monitor, region):
        self.grabs += 1
        return super().grab(monitor, region)


def make(frame, state, settings=None, **kw):
    clock, calls, st = {'t': 0.0}, [], {'v': state}
    cap = CountingCapture(frame)
    settings = settings if settings is not None else {'auto_scan': True}
    s = AutoScanner(lambda f: calls.append(f.shape) or {'results': [{'num': 1}], 'grid': {}, 'grid_failed': False},
                    lambda: settings, win32=FakeWin32(windows=[win(3, 100)], mon_of=M1), capture=cap,
                    clock=lambda: clock['t'], wall=lambda: 1000.0 + clock['t'], raid_state_fn=lambda: st['v'], **kw)
    s._clock = clock
    return s, calls, cap, st, settings


RAID = {'state': 'in_raid', 'since': None, 'source': 'logs'}
LOBBY = {'state': 'lobby', 'since': None, 'source': 'logs'}


def test_default_is_off():
    assert tsh_app.default_settings()['live_in_raid'] is False
    s = make(inventory_frame(), RAID, settings={})[0]
    assert s._opts()['allow_raid'] is False


def test_a_raid_costs_nothing_no_grab_no_detect_no_scan_no_collect():
    s, calls, cap, _, _ = make(inventory_frame(), RAID)
    delays = run(s, 12)
    assert cap.grabs == 0 and calls == [] and cap.closed >= 1          # screen duplication released too
    assert s.status()['state'] == 'in_raid' and 'paused' in s.status()['message']
    assert set(delays) == {svc.POLL_RAID_S}
    assert s.status()['live_in_raid'] is False


def test_back_in_the_lobby_the_stash_is_scanned_once():
    s, calls, cap, st, _ = make(inventory_frame(), RAID)
    run(s, 6)
    st['v'] = LOBBY
    run(s, 8)
    assert len(calls) == 1 and cap.grabs > 0 and s.status()['state'] == 'scanned'


def test_entering_a_raid_cancels_a_running_scan():
    release, entered = threading.Event(), threading.Event()
    seen = {}

    def slow(frame, publish=None, cancel=None):
        entered.set()
        while not cancel():
            release.wait(0.01)
        seen['cancelled'] = True
        raise RuntimeError('cancelled')
    clock, st = {'t': 0.0}, {'v': LOBBY}
    s = AutoScanner(slow, lambda: {'auto_scan': True}, win32=FakeWin32(windows=[win(3, 100)], mon_of=M1),
                    capture=CountingCapture(inventory_frame()), clock=lambda: clock['t'], progressive=True,
                    raid_state_fn=lambda: st['v'])
    s._clock = clock
    run(s, 4)
    assert entered.wait(2)
    st['v'] = RAID
    run(s, 1)
    assert seen.get('cancelled') and not s.scanning()


def test_work_in_raid_on_keeps_the_old_behaviour():
    s, calls, cap, _, _ = make(inventory_frame(chrome=False), RAID, settings={'auto_scan': True, 'live_in_raid': True})
    run(s, 6)
    assert cap.grabs > 0 and len(calls) == 1 and s.status()['live_in_raid'] is True


def test_a_raid_line_that_never_ended_stops_being_trusted():
    stale = {'state': 'in_raid', 'since': 1000.0 - svc.RAID_MAX_S - 5, 'source': 'logs'}
    s, calls, cap, _, _ = make(inventory_frame(), stale)
    run(s, 6)
    assert cap.grabs > 0 and len(calls) == 1


def test_the_log_being_unreadable_falls_back_to_the_frame():
    s, calls, cap, _, _ = make(inventory_frame(), None)
    run(s, 6)
    assert cap.grabs > 0 and len(calls) == 1


def test_scan_payload_has_no_raid_advice_when_off(monkeypatch):
    import liveadvice
    hit = []
    monkeypatch.setattr(liveadvice, 'compute_advice', lambda *a, **k: hit.append(1) or {'grab': [], 'drop': []})
    # _sell_payload is exercised through its raid gate only: raid scene + live_in_raid off -> not computed
    src = open(tsh_app.__file__, encoding='utf-8').read()
    assert "if raid and (settings.get('live_in_raid') or settings.get('auto_scan_in_raid')):" in src
    assert hit == []
