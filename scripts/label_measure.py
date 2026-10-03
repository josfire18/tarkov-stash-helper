"""
Measure Anchor 2 (closed-set label match in the game's font) on the labelled eval tiles.

For every un-clipped labelled non-weapon tile the candidate set is the truth short name plus the
CONFUSERS nearest short names (by edit distance of what the game would print) among all items that
can occupy the footprint - an adversarial set, harder than what the pipeline hands the anchor.
Reports the true-text score distribution, the best wrong-text score, and what the certification
rule says (certain right / certain WRONG / not certain).

    python -u scripts/label_measure.py [--limit N] [--confusers 25] [images...]
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
from rapidfuzz import process, distance  # noqa: E402

from identify import fontlabel as FL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--confusers', type=int, default=25)
    ap.add_argument('--json')
    a = ap.parse_args()
    lr = FL.LabelRenderer.create()
    items = json.load(open(os.path.join(ROOT, 'data', 'prices_cache.json'), encoding='utf-8'))['items']
    by_fp = collections.defaultdict(list)
    for it in items:
        if it.get('shortName') and it.get('width'):
            by_fp[(it['width'], it['height'])].append(it)
    info = {it['id']: it for it in items}
    imgs = a.images or sorted(p[:-len('.truth.full.json')] + '.png' for p in
                              glob.glob(os.path.join(ROOT, 'data', 'eval', '**', '*.truth.full.json'), recursive=True))
    rows, stats = [], collections.Counter()
    t0 = time.time()
    for p in imgs:
        img = cv2.imread(p)
        truth = json.load(open(p[:-4] + '.truth.full.json', encoding='utf-8'))
        ps = [(r['rect'][2] - 1) / r['W'] for r in truth if r.get('W') and not r.get('uncertain')]
        if not ps:
            continue
        pitch = 63.0 if np.median(ps) < 70 else 84.0
        n = 0
        for t in truth:
            if (t.get('uncertain') or not t.get('item_id') or t.get('clipped') or not t.get('W')
                    or t.get('category') == 'weapon' or t['item_id'] not in info):
                continue
            st = FL.strip_of(img, t['rect'], pitch)
            if st is None:
                continue
            W, H = t['W'], t['H']
            pool = {it['shortName'] for it in by_fp[(W, H)] + by_fp[(H, W)]}
            want = info[t['item_id']]['shortName']
            wv = lr.visible_text(want, st.shape[1], pitch)
            pool.discard(want)
            pool = [s for s in pool if lr.visible_text(s, st.shape[1], pitch) != wv]   # print-identical = twins
            near = process.extract(want, pool, scorer=distance.Levenshtein.distance, limit=a.confusers)
            texts = [want] + [s for s, _, _ in near]
            fits = FL.LabelRenderer.match(lr, st, texts, pitch)
            tf = next(f for f in fits if f.text == wv)
            wrong = next((f for f in fits if f.text != wv), None)
            cert = FL.certain(fits)
            ok = fits[0].text == wv
            stats['certain right' if cert and ok else 'CERTAIN WRONG' if cert else 'not certain'] += 1
            row = dict(img=os.path.basename(p), want=want, vis=wv, pitch=pitch, true=round(tf.total, 2),
                       true_res=round(tf.score, 2), unexpl=round(tf.unexplained, 3),
                       wrong=wrong and round(wrong.total, 2), wrong_text=wrong and wrong.text, cert=cert, ok=ok)
            if cert and not ok:
                print('  CERTAIN WRONG', row)
            rows.append(row)
            n += 1
            if a.limit and n >= a.limit:
                break
        print(os.path.basename(p), n, dict(stats), f'{time.time() - t0:.0f}s')
    for pitch in sorted({r['pitch'] for r in rows}):
        rs = [r for r in rows if r['pitch'] == pitch]
        q = lambda v: np.percentile(v, [0, 50, 90, 99, 100]).round(2).tolist() if len(v) else []
        tr = np.array([r['true'] for r in rs])
        wr = np.array([r['wrong'] for r in rs if r['wrong'] is not None])
        print(f'pitch {pitch}: n={len(rs)} true total min/med/p90/p99/max {q(tr)} | best wrong {q(wr)}')
    print(dict(stats))
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=0)


if __name__ == '__main__':
    main()
