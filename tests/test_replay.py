"""The replay harness (autoscan/replay.py): timeline lookup, Live-page grouping of a payload, shown-at-time
summary, and a real AutoScanner driven over a short timeline with a fake progressive scan function."""
import threading
import time

import numpy as np

from autoscan.replay import (Event, ReplayLog, Step, Timeline, live_groups, provisional_sellable, replay,
                             screen_at, shown_at, summarize)
from autoscan_frames import inventory_frame


def test_timeline_picks_the_step_on_screen_and_keeps_the_last():
    a, b = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
    tl = Timeline([Step('a', a, 2.0), Step('b', b, 3.0)])
    assert tl.total == 5.0
    assert [tl.at(t).label for t in (0, 1.99, 2.0, 4.9, 99)] == ['a', 'a', 'b', 'b', 'b']


def _row(rec, **kw):
    return {'matched_name': 'x', 'recommend': rec, **kw}


def test_live_groups_mirror_the_live_page():
    p = {'results': [_row('flea'), _row('keep', trader_name=None), _row('keep', check=True),
                     _row('keep', provisional=True, uncertain=True), _row('trader', trader_name='Skier'),
                     _row('none'), {'matched_name': 'Roubles', 'recommend': 'keep'}],
         'on_you': [{}, {}], 'advice': {'grab': [1], 'drop': []}}
    g = live_groups(p)
    assert g == {'sell': 2, 'nosell': 1, 'keep': 1, 'check': 1, 'pending': 1, 'own': 2, 'grab': 1, 'drop': 0}


def test_provisional_rows_never_carry_sell_advice():
    ok = {'results': [_row('keep', provisional=True, uncertain=True)]}
    bad = {'results': [_row('flea', provisional=True)]}
    assert provisional_sellable(ok) == 0 and provisional_sellable(bad) == 1


def test_shown_at_and_summary_report_staleness():
    log = ReplayLog(total=6.0, steps=[(0.0, 'a'), (3.0, 'b')])
    log.events = [Event(1.0, 1, False, 'a', {'pending': 4}, 'a'), Event(2.0, 2, True, 'a', {'sell': 4}, 'a')]
    assert shown_at(log, 0.5) is None and shown_at(log, 1.5).seq == 1 and shown_at(log, 5).seq == 2
    assert screen_at(log, 2.9) == 'a' and screen_at(log, 3.0) == 'b'
    rows = summarize(log, 1.0)
    assert rows[0]['kind'] is None and rows[1]['kind'] == 'first look' and rows[2]['kind'] == 'final'
    assert not rows[2]['stale'] and rows[4]['stale']                       # screen is 'b', Live still shows 'a'


def _payload(final, n=3):
    rows = [{'matched_name': 'i', 'recommend': 'keep' if not final else 'flea', 'provisional': not final,
             'uncertain': not final} for _ in range(n)]
    return {'results': rows, 'grid': {}, 'grid_failed': False, 'on_you': [], 'warnings': []}


def test_replay_drives_a_real_autoscanner_first_look_then_final_and_a_new_view_replaces_it():
    a = inventory_frame()
    b = a.copy()
    b[200:600, 300:900] = 40           # a different, stable view (a container opened over the stash)
    seen = []

    def fake_scan(frame, publish=None, cancel=None):
        seen.append(threading.current_thread().name)
        publish(_payload(False))
        time.sleep(0.15)
        return _payload(True)

    log = replay(fake_scan, [Step('stash', a, 1.6), Step('other', b, 1.6)], poll=0.02)
    kinds = [(e.frame_label, e.final) for e in log.events]
    assert ('stash', False) in kinds and ('stash', True) in kinds       # first look then final for the first view
    assert kinds[-1] == ('other', True)                                 # the newer view's final replaced it
    assert [lab for _, lab in log.calls][0] == 'stash' and 'other' in [lab for _, lab in log.calls]
    assert all(provisional_sellable(_payload(False)) == 0 for _ in log.events)
    rows = summarize(log, 0.5)
    assert rows[-1]['shown'] == 'other' and rows[-1]['kind'] == 'final'
