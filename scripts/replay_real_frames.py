"""
Replay the user's OWN recorded session through the real auto-scanner and print what the Live page
showed at every moment.

    python scripts/replay_real_frames.py FIRST_TS LAST_TS [--tail 25] [--every 2] [--frames DIR]

FIRST_TS / LAST_TS are unix timestamps (``ts`` in data/scenes/live/index.jsonl, or the number at the start
of a frame's file name / 1000).  Each recorded frame is on screen from its timestamp until the next one's
(the collector saves a view once it has been stable for two polls), the last one for ``--tail`` seconds.
The scan function is app._autoscan_scan - the exact function the running app uses - and everything runs in
REAL time, so a scan that takes 10 s really blocks the next view the way it does for the player.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from autoscan.replay import Step, format_log, replay  # noqa: E402


def steps_from_index(frames_dir, t0, t1, tail=25.0):
    rows = [json.loads(l) for l in open(os.path.join(frames_dir, 'index.jsonl'), encoding='utf-8') if l.strip()]
    rows = [r for r in rows if t0 <= r['ts'] <= t1 and os.path.exists(os.path.join(frames_dir, r['file']))]
    rows.sort(key=lambda r: r['ts'])
    steps = []
    for i, r in enumerate(rows):
        dur = (rows[i + 1]['ts'] - r['ts']) if i + 1 < len(rows) else tail
        steps.append(Step(r['file'].split('-')[0][-6:], cv2.imread(os.path.join(frames_dir, r['file'])), dur))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('t0', type=float)
    ap.add_argument('t1', type=float)
    ap.add_argument('--tail', type=float, default=25.0)
    ap.add_argument('--every', type=float, default=2.0)
    ap.add_argument('--frames', default=os.path.join(ROOT, 'data', 'scenes', 'live'))
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    import app
    app._warm_v2_engine()
    steps = steps_from_index(a.frames, a.t0, a.t1, a.tail)
    print('timeline:', ', '.join(f'{s.label}({s.seconds:.0f}s)' for s in steps))
    log = replay(app._autoscan_scan, steps)
    print(format_log(log, a.every))
    print('scan starts:', [(round(t, 1), l) for t, l in log.calls])
    print('states:', log.states)
    print('events:', [(round(e.t, 1), e.seq, 'final' if e.final else 'first look', e.frame_label, e.ms) for e in log.events])
    if a.json:
        json.dump({'events': [e.__dict__ for e in log.events], 'calls': log.calls, 'states': log.states}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
