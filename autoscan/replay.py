"""
Replay harness: feed recorded frames through a REAL :class:`AutoScanner` (its trigger, supersede rules,
progressive first look -> final) with a fake capture that follows a timeline, and record what the Live
page would have shown at every moment (the latest published result), against what was on screen.

    steps = [Step('stash', frame_a, 14.0), Step('scrolled', frame_b, 6.0), ...]
    log = replay(scan_fn, steps)            # real time: the scan function takes whatever it takes
    print(format_log(log))

Pure helpers (``Timeline``, ``live_groups``, ``summarize``, ``shown_at``) are unit-tested without any
engine.  Used by ``scripts/replay_real_frames.py`` and ``tests/test_replay.py``.
"""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field

from . import capture as cap
from .detect import frame_is_blank
from .winapi import GameWindow, MonitorInfo, WinInfo

MONITOR = MonitorInfo(1, r'\\.\DISPLAY1', (0, 0, 2560, 1440), True)


@dataclass
class Step:
    label: str
    frame: object          # BGR frame
    seconds: float         # how long this view stays on screen


class Timeline:
    """Which step is on screen ``t`` seconds after the start (the last step stays)."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.total = sum(s.seconds for s in self.steps)

    def at(self, t: float) -> Step:
        acc = 0.0
        for s in self.steps:
            acc += s.seconds
            if t < acc:
                return s
        return self.steps[-1]


class TimelineCapture:
    backend_name = 'replay'

    def __init__(self, timeline: Timeline, now=time.monotonic):
        self.timeline, self.now, self.t0 = timeline, now, None
        self.closed = 0

    def start(self):
        self.t0 = self.now()

    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else self.now() - self.t0

    def grab(self, monitor, region):
        fr = self.timeline.at(self.elapsed()).frame
        return cap.CaptureResult(fr, 'replay', blank=frame_is_blank(fr))

    def close(self):
        self.closed += 1


class ReplayWin32:
    """Just enough Win32 for ``locate_game``: one Unity window of one process on one monitor."""

    def __init__(self, size=(2560, 1440)):
        self.size = size

    def find_pids(self, exe):
        return [100] if exe.lower() == 'escapefromtarkov.exe' else []

    def top_level_windows(self):
        return [WinInfo(3, 100, 'UnityWndClass', 'EscapeFromTarkov', True, False, (0, 0, *self.size), (0, 0, *self.size))]

    def monitors(self):
        return [MONITOR]

    def monitor_of(self, hwnd):
        return MONITOR

    def foreground(self):
        return 0

    def physical_pixels(self):
        return contextlib.nullcontext()


# ---- what the Live page makes of a payload (mirrors templates/live.html ``classify``) -----------

MONEY = ('Roubles', 'Dollars', 'Euros')


def live_groups(payload: dict) -> dict:
    """Row counts per Live-page group: sell (flea + traders), nosell, keep, check, pending (a provisional
    first-look row: 'Verifying'), own (On you), grab / drop (raid advice)."""
    g = {'sell': 0, 'nosell': 0, 'keep': 0, 'check': 0, 'pending': 0, 'own': 0, 'grab': 0, 'drop': 0}
    for r in payload.get('results') or []:
        if r.get('matched_name') in MONEY:
            continue
        rec = r.get('recommend')
        if r.get('provisional') and rec == 'keep':
            g['pending'] += 1
        elif rec == 'keep':
            g['check' if (r.get('check') or r.get('uncertain')) else 'keep'] += 1
        elif rec == 'flea' or r.get('trader_name'):
            g['sell'] += 1
        else:
            g['nosell'] += 1
    g['own'] = len(payload.get('on_you') or [])
    adv = payload.get('advice') or {}
    g['grab'], g['drop'] = len(adv.get('grab') or []), len(adv.get('drop') or [])
    return g


def provisional_sellable(payload: dict) -> int:
    """Provisional rows that carry sell advice (must always be 0)."""
    return sum(1 for r in (payload.get('results') or [])
               if r.get('provisional') and (r.get('recommend') in ('flea', 'trader') or r.get('trader_name')))


@dataclass
class Event:
    t: float
    seq: int
    final: bool
    frame_label: str       # the view the published result was computed from
    groups: dict
    screen_label: str      # what was on screen when it was published
    ms: int = 0


@dataclass
class ReplayLog:
    events: list = field(default_factory=list)
    states: list = field(default_factory=list)        # [(t, scanner state)]
    steps: list = field(default_factory=list)         # [(t_start, label)]
    calls: list = field(default_factory=list)         # [(t, label)] scan_fn invocations
    total: float = 0.0


def shown_at(log: ReplayLog, t: float):
    """The event the Live page shows at time ``t`` (the latest published one), or None."""
    cur = None
    for e in log.events:
        if e.t <= t:
            cur = e
    return cur


def screen_at(log: ReplayLog, t: float) -> str:
    cur = log.steps[0][1] if log.steps else ''
    for t0, lab in log.steps:
        if t0 <= t:
            cur = lab
    return cur


def summarize(log: ReplayLog, every: float = 1.0) -> list:
    """One row per ``every`` seconds: (t, screen, shown label, 'final'/'first look'/'-', groups, stale?)."""
    rows, t = [], 0.0
    while t <= log.total + 1e-9:
        scr, ev = screen_at(log, t), shown_at(log, t)
        rows.append({'t': round(t, 1), 'screen': scr,
                     'shown': ev.frame_label if ev else None,
                     'kind': None if ev is None else ('final' if ev.final else 'first look'),
                     'seq': ev.seq if ev else 0,
                     'groups': ev.groups if ev else None,
                     'stale': bool(ev and ev.frame_label != scr)})
        t += every
    return rows


def format_log(log: ReplayLog, every: float = 2.0) -> str:
    out = []
    for r in summarize(log, every):
        g = r['groups'] or {}
        gs = ' '.join(f'{k}={v}' for k, v in g.items() if v and k in ('sell', 'keep', 'check', 'pending', 'nosell', 'own'))
        out.append(f"{r['t']:6.1f}s screen={r['screen']:<10} shows={str(r['shown']):<10} "
                   f"{r['kind'] or '-':<10}{' STALE' if r['stale'] else '      '} {gs}")
    return '\n'.join(out)


def replay(scan_fn, steps, settings=None, poll=0.05, tail=0.0, progressive=True, make_scanner=None, wall=time.monotonic):
    """Run a real AutoScanner over ``steps`` in real time.  ``scan_fn`` is the app's scan function
    (``scan_fn(frame, publish=..., cancel=...)`` when progressive).  Returns a :class:`ReplayLog`."""
    from . import AutoScanner
    tl = Timeline(steps)
    capture = TimelineCapture(tl, wall)
    settings = {'auto_scan': True, 'live_in_raid': False, 'collect_scene_frames': False, **(settings or {})}
    log = ReplayLog(total=tl.total + tail)
    label_of = {id(s.frame): s.label for s in steps}
    lock = threading.Lock()

    def wrapped(frame, publish=None, cancel=None):
        with lock:
            log.calls.append((capture.elapsed(), label_of.get(id(frame), '?')))
        return scan_fn(frame, publish=publish, cancel=cancel) if progressive else scan_fn(frame)

    mk = make_scanner or (lambda **kw: AutoScanner(**kw))
    sc = mk(scan_fn=wrapped, get_settings=lambda: settings, win32=ReplayWin32(), capture=capture,
            progressive=progressive, collect_dir=None)
    seen, last_state = 0, None
    capture.start()
    sc.start()
    try:
        while capture.elapsed() < log.total:
            t = capture.elapsed()
            res = sc.result(seen)
            if res.get('ready'):
                seen = res['seq']
                fl = label_of.get(id(sc._frame), '?')
                log.events.append(Event(t, seen, not (res.get('timing') or {}).get('final') is False,
                                        fl, live_groups(res), tl.at(t).label, (res.get('timing') or {}).get('scan_ms', 0)))
            st = sc.status().get('state')
            if st != last_state:
                log.states.append((round(t, 2), st))
                last_state = st
            time.sleep(poll)
    finally:
        sc.stop()
    acc = 0.0
    for s in steps:
        log.steps.append((acc, s.label))
        acc += s.seconds
    return log
