"""
Strict accuracy report: every labelled tile is CORRECT (right item, not flagged), WRONG (wrong
item, not flagged - the dangerous case), UNCERTAIN (flagged, counts as a failure too) or MISSED
(not segmented).  Coverage = (correct + wrong) / labelled.

    python scripts/accuracy_report.py [--no-dino] [--no-ocr] [--json out.json] [images...]

Without images every ``data/eval/**/<name>.png`` that has a ``<name>.truth.full.json`` is used.
Truth rows marked uncertain / without an item id are excluded.  Guns count as correct when they
are recognised as a weapon (they are skipped by the sell list), dogtags when any dogtag is named.
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

import test_scan as T  # noqa: E402


def eval_images():
    out = []
    for tp in sorted(glob.glob(os.path.join(ROOT, 'data', 'eval', '**', '*.truth.full.json'), recursive=True)):
        img = tp[:-len('.truth.full.json')] + '.png'
        if os.path.exists(img):
            out.append(img)
    return out


def is_ok(t, d, cat):
    if cat == 'weapon':
        return d.get('category') == 'weapon'
    if t.get('name', '').startswith('Dogtag ') and 'case' not in t.get('name', ''):
        return d.get('name', '').startswith('Dogtag ') and 'case' not in d.get('name', '')
    return d['item_id'] == t['item_id'] or (bool(d.get('name')) and d.get('name') == t.get('name'))


_ARRS = ('ids', 'names', 'shorts', 'cats', 'tint', 'src')
_snap: dict = {}


def run_clean(img, use_dino, use_ocr, exe=False):
    """One scan from a clean engine state: the learned-icon store and the in-memory renames it
    causes are reset before every image, so a result never depends on which screenshots (or a
    warm-up scan of the same screenshot) were seen before."""
    key = (use_dino, use_ocr)
    if key not in T._v2_engines:
        from identify.config import EngineSettings
        from identify.pipeline import Engine
        kw = dict(dino_backend='onnx', accelerate=False) if exe else {}
        eng = Engine(EngineSettings(use_dino=use_dino, use_ocr=use_ocr,
                                    extra={'learned_path': T._EVAL_LEARNED}, **kw))
        eng.learned.data = {}
        T._v2_engines[key] = eng
        _snap[key] = {k: getattr(eng.cat, k).copy() for k in _ARRS}   # before any scan
        eng.scan(img)                                                  # model load / CUDA init
    eng = T._v2_engines[key]
    for k in _ARRS:
        getattr(eng.cat, k)[:] = _snap[key][k]
    eng.learned.data = {}
    eng._builds_done = False          # re-run the (deterministic) build association
    return T.run_v2(img, use_dino, use_ocr, warm=False)


def grade(truth, dets, cats):
    mt = T.match_rects([t['rect'] for t in truth], [d['rect'] for d in dets])
    res = {'correct': 0, 'wrong': 0, 'uncertain': 0, 'missed': 0, 'n': 0, 'fail': []}
    for i, t in enumerate(truth):
        if t.get('uncertain') or not t.get('item_id'):
            continue
        res['n'] += 1
        j = mt.get(i)
        if j is None:
            res['missed'] += 1
            res['fail'].append({'kind': 'missed', 'idx': i, 'rect': t['rect'], 'want': t.get('name')})
            continue
        d = dets[j]
        cat = t.get('category') or cats.get(t['item_id'], 'other')
        ok = is_ok(t, d, cat)
        ev = d['det'].evidence
        info = {'idx': i, 'rect': t['rect'], 'want': t.get('name'), 'got': d['name'], 'conf': round(d['conf'], 3),
                'ocr': ev.get('ocr_text'), 'res': ev.get('residual'), 'dino': ev.get('dino'),
                'src': ev.get('source'), 'note': ev.get('note', ''),
                'alts': [a['name'] for a in ev.get('alternatives', [])]}
        if d.get('uncertain'):
            res['uncertain'] += 1
            res['fail'].append({'kind': 'uncertain' + ('' if ok else '(wrong)'), **info})
        elif ok:
            res['correct'] += 1
        else:
            res['wrong'] += 1
            res['fail'].append({'kind': 'WRONG', **info})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*')
    ap.add_argument('--no-dino', action='store_true')
    ap.add_argument('--no-ocr', action='store_true')
    ap.add_argument('--exe', action='store_true',
                    help='what the packaged exe runs: stage 2 on onnxruntime (CPU), stage 1 on numpy')
    ap.add_argument('--json')
    ap.add_argument('--quiet', action='store_true', help='no per-failure lines')
    a = ap.parse_args()
    imgs = a.images or eval_images()
    cats = T._categories()
    tot = {'correct': 0, 'wrong': 0, 'uncertain': 0, 'missed': 0, 'n': 0, 'time': 0.0}
    out = {}
    print(f"{'screenshot':<26}{'n':>5}{'correct':>9}{'wrong':>7}{'uncert':>8}{'missed':>8}{'cover':>8}{'s/scan':>8}")
    for p in imgs:
        img = cv2.imread(p)
        truth, _ = T.load_truth(p, 'full')
        if img is None or truth is None:
            continue
        dets, dt, _ = run_clean(img, not a.no_dino, not a.no_ocr, a.exe)
        g = grade(truth, dets, cats)
        name = os.path.basename(p)
        cov = (g['correct'] + g['wrong']) / g['n'] if g['n'] else 0.0
        print(f"{name[:25]:<26}{g['n']:>5}{g['correct']:>9}{g['wrong']:>7}{g['uncertain']:>8}{g['missed']:>8}"
              f"{100 * cov:>7.1f}%{dt:>8.2f}")
        if not a.quiet:
            for f in g['fail']:
                print(f"     {f['kind']:<16} #{f['idx']} want '{f.get('want')}' got '{f.get('got', '')}' "
                      f"ocr='{f.get('ocr', '')}' res={f.get('res')} dino={f.get('dino')} conf={f.get('conf')}")
        for k in ('correct', 'wrong', 'uncertain', 'missed', 'n'):
            tot[k] += g[k]
        tot['time'] += dt
        out[name] = {**{k: v for k, v in g.items()}, 'time': dt}
    n = max(1, tot['n'])
    print(f"{'TOTAL':<26}{tot['n']:>5}{tot['correct']:>9}{tot['wrong']:>7}{tot['uncertain']:>8}{tot['missed']:>8}"
          f"{100 * (tot['correct'] + tot['wrong']) / n:>7.1f}%{tot['time'] / max(1, len(out)):>8.2f}")
    print(f"strict accuracy (correct / labelled): {100 * tot['correct'] / n:.2f}%")
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'total': tot, 'per_image': out}, f, indent=1, default=str)


if __name__ == '__main__':
    main()
