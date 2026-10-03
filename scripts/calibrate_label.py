"""
Calibrate the label renderer (identify/fontlabel.py, TMP SDF) against real label strips.

Samples: labelled eval tiles whose short name fits its label box (not truncated).  For a grid of
font assets / sizes / bold the per-strip best sub-pixel shift and the residual over the text's ink
are measured; the text (vertex) colour is refit by least squares.  Writes
identify/assets/label_params.json with --write.

    python scripts/calibrate_label.py [--n 25] [--assets Bender_Outline] [--sizes 12,12.5] [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from identify import fontlabel as FL  # noqa: E402


RNG = 2.0


def samples(n_per_pitch: int, lr, shorts: dict, truncated: bool = False):
    out = {}
    for tp in sorted(glob.glob(os.path.join(ROOT, 'data', 'eval', '**', '*.truth.full.json'), recursive=True)):
        img = cv2.imread(tp[:-len('.truth.full.json')] + '.png')
        rows = json.load(open(tp, encoding='utf-8'))
        ps = [(r['rect'][2] - 1) / r['W'] for r in rows if r.get('W') and not r.get('uncertain')]
        if not ps:
            continue
        pitch = 63.0 if np.median(ps) < 70 else 84.0     # the two UI scales of the eval set
        for t in rows:
            if (t.get('uncertain') or not t.get('item_id') or t.get('clipped') or not t.get('W')
                    or t.get('category') == 'weapon'):
                continue
            text = shorts.get(t['item_id'])
            if not text:
                continue
            st = FL.strip_of(img, t['rect'], pitch)
            if st is None:
                continue
            fits = lr.visible_text(text, st.shape[1], pitch) == text
            if fits == truncated:
                continue
            lst = out.setdefault(pitch, [])
            if len(lst) < n_per_pitch:
                lst.append((st, text, pitch, os.path.basename(tp)[:12]))
    return out


def evaluate(lr, data, iters=2, rng=None):
    rng = RNG if rng is None else rng
    for _ in range(iters):
        fits = [lr.align(st, text, pitch, rng=rng) for st, text, pitch, _ in data]
        lr.p.color = lr.fit_colour([(st, text, pitch, f.shift) for (st, text, pitch, _), f in zip(data, fits)])
    fits = [lr.align(st, text, pitch, rng=rng) for st, text, pitch, _ in data]
    return np.array([f.score for f in fits]), fits


def fit_shade(data, p0, rounds: int = 3, log=print):
    """Nelder-Mead over (size, spacing, ef, wf, eu, wu, au) on fixed per-strip alignments,
    re-aligning between rounds; the text colour is refit by least squares each round."""
    import warnings
    warnings.filterwarnings('ignore')
    from scipy.optimize import minimize
    p = FL.LabelParams(**p0.to_json())
    if p.shade is None:
        p.shade = {'ef': -0.5, 'wf': 1.0, 'eu': -5.0, 'wu': 1.0, 'au': 0.8}
    keys = ['ef', 'wf', 'eu', 'wu', 'au']

    def make(x):
        q = FL.LabelParams(**p.to_json())
        q.size, q.spacing = float(x[0]), float(x[1])
        q.shade = {k: float(v) for k, v in zip(keys, x[2:])}
        q.shade['wf'] = max(0.2, q.shade['wf'])
        q.shade['wu'] = max(0.2, q.shade['wu'])
        q.shade['au'] = min(1.0, max(0.0, q.shade['au']))
        return FL.LabelRenderer.create(q)

    for rd in range(rounds):
        lr = FL.LabelRenderer.create(p)
        fits = [lr.align(st, text, pitch, rng=RNG if rd == 0 else 1.0) for st, text, pitch, _ in data]
        p.color = lr.fit_colour([(st, text, pitch, f.shift) for (st, text, pitch, _), f in zip(data, fits)])
        lr = FL.LabelRenderer.create(p)
        bgs = []
        for (st, text, pitch, _), f in zip(data, fits):
            h, w = st.shape[:2]
            bgs.append(lr.background(st, lr.render(text, w, h, pitch, *f.shift)[1]))

        def obj(x):
            q = make(x)
            tot = 0.0
            for (st, text, pitch, _), f, bg in zip(data, fits, bgs):
                r, _ = q.residual(st, text, pitch, f.shift[0], f.shift[1], bg)
                tot += r
            return tot / len(data)
        x0 = [p.size, p.spacing] + [p.shade[k] for k in keys]
        r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': 250, 'xatol': 0.01, 'fatol': 0.01})
        q = make(r.x).p
        p.size, p.spacing, p.shade = q.size, q.spacing, q.shade
        log(f'round {rd}: residual {r.fun:.2f}  size {p.size:.3f} spacing {p.spacing:.2f} shade '
            f'{ {k: round(v, 3) for k, v in p.shade.items()} } colour {tuple(round(c, 3) for c in p.color)}')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--assets', default='Bender_Outline')
    ap.add_argument('--sizes', default='11.5,12,12.5')
    ap.add_argument('--bold', default='0,1')
    ap.add_argument('--k', default='1.0')
    ap.add_argument('--spacing', default='0')
    ap.add_argument('--pitch', type=float, default=63.0)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--rng', type=float, default=2.0)
    ap.add_argument('--truncated', action='store_true', help='evaluate on truncated names instead')
    ap.add_argument('--fit-shade', action='store_true', help='fit size/spacing/shading (Nelder-Mead)')
    a = ap.parse_args()
    global RNG
    RNG = a.rng
    items = json.load(open(os.path.join(ROOT, 'data', 'prices_cache.json'), encoding='utf-8'))['items']
    shorts = {it['id']: it.get('shortName') or '' for it in items}
    base = FL.load_params()
    lr0 = FL.LabelRenderer.create(base)
    data_by_pitch = samples(a.n, lr0, shorts, a.truncated)
    print({k: len(v) for k, v in data_by_pitch.items()})
    data = data_by_pitch.get(a.pitch, [])
    if a.fit_shade:
        p = base
        p.size = float(a.sizes.split(',')[0])
        p.spacing = float(a.spacing.split(',')[0])
        p = fit_shade(data, p)
        lr = FL.LabelRenderer.create(p)
        fits = [lr.align(st, text, pitch, rng=1.0) for st, text, pitch, _ in data]
        p.pad_right -= float(np.median([f.shift[0] for f in fits])) * 63.0 / a.pitch
        p.baseline += float(np.median([f.shift[1] for f in fits])) * 63.0 / a.pitch
        lr = FL.LabelRenderer.create(p)
        for pitch, d in sorted(data_by_pitch.items()):
            res, fits = evaluate(lr, d, iters=0)
            print(f'pitch {pitch}: n={len(d)} res mean {res.mean():.2f} med {np.median(res):.2f} max {res.max():.2f} '
                  f'shift med ({np.median([f.shift[0] for f in fits]):+.2f},{np.median([f.shift[1] for f in fits]):+.2f})')
        print('params', p.to_json())
        if a.write:
            os.makedirs(os.path.dirname(FL.PARAMS_PATH), exist_ok=True)
            with open(FL.PARAMS_PATH, 'w', encoding='utf-8') as fh:
                json.dump(p.to_json(), fh, indent=1)
            print('wrote', FL.PARAMS_PATH)
        return
    best = None
    for asset in a.assets.split(','):
        for size in (float(s) for s in a.sizes.split(',')):
            for bold in (bool(int(b)) for b in a.bold.split(',')):
                for k, sp in ((float(x), float(y)) for x in a.k.split(',') for y in a.spacing.split(',')):
                    p = FL.LabelParams(**{**base.to_json(), 'asset': asset, 'size': size, 'bold': bold, 'k': k,
                                          'spacing': sp})
                    lr = FL.LabelRenderer.create(p)
                    res, fits = evaluate(lr, data)
                    dx = float(np.median([f.shift[0] for f in fits]))
                    dy = float(np.median([f.shift[1] for f in fits]))
                    print(f'{asset:<16} size {size:5.2f} bold={int(bold)} k={k} sp={sp} res mean {res.mean():.2f} med {np.median(res):.2f}'
                          f' p90 {np.percentile(res, 90):.2f} max {res.max():.2f} | shift med ({dx:+.2f},{dy:+.2f})'
                          f' colour {tuple(round(c, 3) for c in lr.p.color)}')
                    if best is None or res.mean() < best[0]:
                        best = (res.mean(), lr.p, dx, dy)
    _, p, dx, dy = best
    p.pad_right -= dx * 63.0 / a.pitch
    p.baseline += dy * 63.0 / a.pitch
    print('best', p.to_json())
    lr = FL.LabelRenderer.create(p)
    for pitch, data in sorted(data_by_pitch.items()):
        res, fits = evaluate(lr, data, iters=0)
        print(f'pitch {pitch}: n={len(data)} res mean {res.mean():.2f} med {np.median(res):.2f} max {res.max():.2f} '
              f'shift med ({np.median([f.shift[0] for f in fits]):+.2f},{np.median([f.shift[1] for f in fits]):+.2f})')
        worst = np.argsort(-res)[:3]
        print('   worst:', [(data[i][1], data[i][3], round(float(res[i]), 2)) for i in worst])
    if a.write:
        os.makedirs(os.path.dirname(FL.PARAMS_PATH), exist_ok=True)
        with open(FL.PARAMS_PATH, 'w', encoding='utf-8') as fh:
            json.dump(p.to_json(), fh, indent=1)
        print('wrote', FL.PARAMS_PATH)


if __name__ == '__main__':
    main()
