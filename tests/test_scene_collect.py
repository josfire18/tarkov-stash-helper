import json
import os

import numpy as np

from autoscan.collect import SceneCollector


def _wait(c):
    if c._thread is not None:
        c._thread.join(10)


def test_keeps_stable_new_views_only(tmp_path):
    c = SceneCollector(str(tmp_path), max_frames=2)
    fr = np.zeros((40, 60, 3), np.uint8)
    a, b, d = (np.zeros(64, bool), np.ones(64, bool), np.r_[np.ones(32, bool), np.zeros(32, bool)])
    assert not c.observe(fr, a, in_raid=False)          # first sight: not settled yet
    assert c.observe(fr, a, in_raid=False)               # stable -> kept
    assert not c.observe(fr, a, in_raid=False)           # already kept
    c.observe(fr, b, True); assert c.observe(fr, b, True)
    c.observe(fr, d, True); assert c.observe(fr, d, True)
    _wait(c)
    pngs = sorted(f for f in os.listdir(tmp_path) if f.endswith('.png'))
    assert len(pngs) == 2 and pngs[-1].endswith('-raid.png')          # ring of 2, oldest dropped
    lines = [json.loads(x) for x in open(tmp_path / 'index.jsonl')]
    assert [x['in_raid'] for x in lines] == [False, True, True]
