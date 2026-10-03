"""
Measure Anchor 1 (pixel-exact game render) on the labelled eval tiles.

For every un-clipped labelled tile: the aligned residual against the best render of the truth
item (when the icon cache has one) and against the best render of any *other* item, per pitch,
plus what the certification rule says (certain right / certain WRONG / not certain, by reason).

    python scripts/anchor_measure.py [--interp linear|area|cubic] [--limit N] [images...]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from identify import anchors as A  # noqa: E402
from identify.catalog import default_cache_dir, load_catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*')
    ap.add_argument('--interp', default='linear')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--json')
    a = ap.parse_args()
    interp = {'linear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'cubic': cv2.INTER_CUBIC}[a.interp]
    cat = load_catalog()
    t = time.time()
    from identify.pipeline import Engine
    from identify.config import EngineSettings
    eng = Engine(EngineSettings(use_dino=False, use_ocr=False, accelerate=False), catalog=cat)
    canon = {k: str(cat.ids[v]) for k, v in eng.preset_base.items()}
    idx = A.CacheIndex(default_cache_dir(), cat.meta.get('cache_assoc') or {}, canon=canon)
    print(f'index: {len(idx)} renders, {sum(1 for r in idx.renders if r.item_id)} named, {time.time() - t:.1f}s')
    names = {str(i): str(n) for i, n in zip(cat.ids, cat.names)}
    imgs = a.images or sorted(p[:-len('.truth.full.json')] + '.png' for p in
                              glob.glob(os.path.join(ROOT, 'data', 'eval', '**', '*.truth.full.json'), recursive=True))
    rows = []
    stats = collections.Counter()
    for p in imgs:
        img = cv2.imread(p)
        truth = json.load(open(p[:-4] + '.truth.full.json', encoding='utf-8'))
        ps = [(r['rect'][2] - 1) / r['W'] for r in truth if r.get('W') and not r.get('uncertain')]
        if not ps:
            continue
        snap = 63.0 if np.median(ps) < 70 else 84.0       # grid pitch of the screenshot
        n = 0
        for t_ in truth:
            if t_.get('uncertain') or not t_.get('item_id') or t_.get('clipped') or not t_.get('W') or not t_.get('H'):
                continue
            x, y, w, h = t_['rect']
            W, H = t_['W'], t_['H']
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                continue
            px = py = snap
            tile = img[y:y + h, x:x + w]
            res = idx.match(tile, W, H, px, py, k=A.SHORTLIST)
            want = canon.get(t_['item_id'], t_['item_id'])
            row = {'img': os.path.basename(p), 'name': t_['name'], 'pitch': round(py, 1), 'W': W, 'H': H}
            if res is None:
                stats['no render shortlisted'] += 1
                rows.append(row)
                continue
            true_s = min((s for s, r, _ in res.ranked if (r.item_id or r.hint_id) == want), default=None)
            wrong_s = min((s for s, r, _ in res.ranked if (r.item_id or r.hint_id) and (r.item_id or r.hint_id) != want), default=None)
            anon_s = min((s for s, r, _ in res.ranked if not (r.item_id or r.hint_id)), default=None)
            row.update(best=round(res.score, 2), true=true_s and round(true_s, 2), wrong=wrong_s and round(wrong_s, 2),
                       anon=anon_s and round(anon_s, 2), got=names.get(res.item_id, res.item_id), note=res.note,
                       certain=res.certain, ok=res.item_id == want, n_px=res.n_px, phase=res.phase)
            if res.certain:
                stats['certain right' if res.item_id == want else 'CERTAIN WRONG'] += 1
                if res.item_id != want:
                    print('  CERTAIN WRONG', row)
            else:
                stats[res.note] += 1
            rows.append(row)
            n += 1
            if a.limit and n >= a.limit:
                break
        print(os.path.basename(p), n, dict(stats))
    for pitch in sorted({r['pitch'] for r in rows}):
        rs = [r for r in rows if r['pitch'] == pitch]
        tr = np.array([r['true'] for r in rs if r.get('true') is not None])
        wr = np.array([r['wrong'] for r in rs if r.get('wrong') is not None])
        q = lambda v: np.percentile(v, [0, 50, 90, 99, 100]).round(2).tolist() if len(v) else []
        print(f'pitch {pitch}: tiles {len(rs)} | true-render residual n={len(tr)} min/med/p90/p99/max {q(tr)}'
              f' | best wrong-item render n={len(wr)} {q(wr)}')
    print(dict(stats))
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=0, default=str)


if __name__ == '__main__':
    main()
