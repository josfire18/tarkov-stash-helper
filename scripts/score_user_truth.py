"""
Score the REAL live path (app._autoscan_scan: the function the running app calls) on the user's own labelled
frames (data/eval/user/*.truth.full.json, built by scripts/build_user_truth.py).

    python scripts/score_user_truth.py [--cold] [stem ...]

For every frame it records what the Live page would get twice: the provisional first look (published before
the scan finishes) and the final result, and grades each labelled tile (uncertain / '?' truth rows are skipped):

    correct  the row names the right item and is not flagged Check
    wrong    the row names another item and is NOT flagged (the dangerous case; must be 0)
    check    flagged Check / uncertain (counted apart: check-right = a right answer the app would not commit to)
    missed   no row at that tile at all

The provisional pass grades the name each tile would be listed under ("Looks like X").
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

USER = os.path.join(ROOT, 'data', 'eval', 'user')


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def payload_rows(p):
    """Every tile row of a payload as (rect, item_id, name, flagged, provisional)."""
    out = []
    for r in p.get('results') or []:
        if r.get('px') is None:
            continue
        out.append(((r['px'], r['py'], r['pw'], r['ph']), r.get('item_id'), r.get('matched_name'),
                    bool(r.get('check') or r.get('uncertain') or r.get('provisional')), bool(r.get('provisional'))))
    for r in p.get('on_you') or []:
        out.append(((r['px'], r['py'], r['pw'], r['ph']), r.get('item_id'), r.get('matched_name'),
                    bool(r.get('uncertain') or r.get('provisional')), bool(r.get('provisional'))))
    for r in p.get('skipped') or []:
        out.append(((r['px'], r['py'], r['pw'], r['ph']), None, r.get('matched_name'), False, False))
    return out


def grade(truth, payload, items):
    rows = payload_rows(payload)
    res = dict(n=0, correct=0, wrong=0, check_right=0, check_wrong=0, missed=0, wrong_list=[])
    for t in truth:
        if t.get('uncertain') or not t.get('item_id'):
            continue
        res['n'] += 1
        best, bi = None, 0.0
        for rr in rows:
            v = iou(t['rect'], rr[0])
            if v > bi:
                best, bi = rr, v
        if best is None or bi < 0.5:
            res['missed'] += 1
            continue
        _, iid, name, flagged, _prov = best
        ok = iid == t['item_id'] or (iid is None and name == t.get('name')) or (
            bool(name) and name == t.get('name'))
        if ok:
            res['check_right' if flagged else 'correct'] += 1
        elif flagged:
            res['check_wrong'] += 1
        else:
            res['wrong'] += 1
            res['wrong_list'].append((t['name'], name, t['role']))
    return res


def main(argv):
    cold = '--cold' in argv
    stems = [a for a in argv if not a.startswith('--')]
    paths = sorted(glob.glob(os.path.join(USER, '*.truth.full.json')))
    if stems:
        paths = [p for p in paths if any(s in os.path.basename(p) for s in stems)]
    import app
    app._warm_v2_engine()
    eng = app._v2_engine_for(app._engine_settings(app._load_settings()))
    out = {}
    for tp in paths:
        stem = os.path.basename(tp)[:-len('.truth.full.json')]
        truth = json.load(open(tp, encoding='utf-8'))
        img = cv2.imread(os.path.join(USER, stem + '.png'))
        if cold and hasattr(eng, 'reset_anchors'):
            eng.reset_anchors()
        provs = []
        t0 = time.time()
        final = app._autoscan_scan(img, publish=lambda p: provs.append((time.time() - t0, p)))
        t_final = time.time() - t0
        gf = grade(truth, final, None)
        gp = grade(truth, provs[0][1], None) if provs else None
        out[stem] = {'final': gf, 'prov': gp, 't_first': provs[0][0] if provs else None, 't_final': t_final}
        f = gf
        line = (f"{stem[-18:]:<18} n={f['n']:>3}  FINAL correct {f['correct']:>3} wrong {f['wrong']} check {f['check_right']}+{f['check_wrong']}"
                f" missed {f['missed']:>2}  ({t_final:4.1f}s)")
        if gp:
            line += (f" | FIRST LOOK ({provs[0][0]:.1f}s) right-name {gp['check_right']} wrong-name {gp['check_wrong']}"
                     f" unflagged-wrong {gp['wrong']} missed {gp['missed']}")
        print(line)
        for w in f['wrong_list']:
            print('   WRONG (final):', w)
    tot = lambda k, which: sum(v[which][k] for v in out.values() if v[which])
    for which in ('final', 'prov'):
        print(which.upper(), {k: tot(k, which) for k in ('n', 'correct', 'wrong', 'check_right', 'check_wrong', 'missed')})
    return out


if __name__ == '__main__':
    main(sys.argv[1:])
