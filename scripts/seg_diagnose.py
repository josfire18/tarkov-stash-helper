#!/usr/bin/env python3
"""
Per-error segmentation diagnosis (grid + segmentation only: no catalog, OCR or GPU needed).

For every labelled image it runs ``detect_grid`` + ``segment_panel``, matches the non-empty
footprints against the truth rectangles exactly like ``test_scan.py --score`` (IoU >= 0.9,
one-to-one) and writes, for every NOT SEGMENTED truth row and every unmatched detection, a
crop with the rectangles overlaid:

    green  = truth rectangle        red   = detection (solid = unmatched, thin = matched)
    cyan   = panel lattice lines    white = panel outline

and prints one line per error with the overlapping rectangles so the cause can be classified
(split item / merged items / empty cell / UI chrome / missed).

    python scripts/seg_diagnose.py [images...] [--out _seg_debug] [--json out.json]

With no images it uses the 12 labelled images of ``data/eval``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from identify.grid import detect_grid
from identify.segment import segment_panel


def _default_images() -> list[str]:
    ev = os.path.join(ROOT, 'data', 'eval')
    paths = [os.path.join(ev, 'stash1.png')]
    paths += sorted(glob.glob(os.path.join(ev, 'joey', '*.png')))
    paths += sorted(glob.glob(os.path.join(ev, 'web', 'w*.png')))
    return [p for p in paths if os.path.exists(os.path.splitext(p)[0] + '.truth.full.json')]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _match(truth, dets, thr=0.9):
    pairs = sorted(((_iou(t, d), i, j) for i, t in enumerate(truth) for j, d in enumerate(dets)
                    if _iou(t, d) >= thr), reverse=True)
    mt, md = {}, {}
    for _, i, j in pairs:
        if i not in mt and j not in md:
            mt[i], md[j] = j, i
    return mt, md


def _overlap(a, b) -> float:
    """Intersection over the smaller rectangle (1.0 = one contains the other)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    m = min(aw * ah, bw * bh)
    return ix * iy / m if m else 0.0


def analyse(path: str, out_dir: str | None = None) -> dict:
    img = cv2.imread(path)
    base = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.splitext(path)[0] + '.truth.full.json', encoding='utf-8') as f:
        truth = json.load(f)
    grid = detect_grid(img)
    items = []
    for pi, panel in enumerate(grid.panels):
        for it in segment_panel(img, panel, pi):
            items.append(it)
    dets = [it for it in items if not it.empty]
    trect = [tuple(t['rect']) for t in truth]
    drect = [tuple(d.rect) for d in dets]
    mt, md = _match(trect, drect)
    rep = {'image': base, 'truth': len(truth), 'dets': len(dets), 'matched': len(mt),
           'missed': [], 'extra': [], 'panels': [(p.x0, p.y0, p.x1, p.y1, round(p.pitch_x, 2)) for p in grid.panels]}

    def overlay(rect, name, tag):
        if out_dir is None:
            return None
        x, y, w, h = rect
        pad = int(1.6 * (grid.pitch_x or 64))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        crop = img[y0:y1, x0:x1].copy()
        for p in grid.panels:                                   # lattice
            for gx in p.xs:
                if x0 <= gx < x1:
                    cv2.line(crop, (gx - x0, 0), (gx - x0, crop.shape[0] - 1), (255, 255, 0), 1)
            for gy in p.ys:
                if y0 <= gy < y1:
                    cv2.line(crop, (0, gy - y0), (crop.shape[1] - 1, gy - y0), (255, 255, 0), 1)
        # upscale first so overlay strokes stay thin on the item art
        sc = 2 if max(crop.shape[:2]) < 700 else 1
        if sc > 1:
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        for j, d in enumerate(drect):
            if _overlap(d, (x0, y0, x1 - x0, y1 - y0)) > 0.3:
                col = (0, 0, 255)
                th = 1 if j in md else 2
                dx, dy, dw, dh = d
                cv2.rectangle(crop, ((dx - x0) * sc, (dy - y0) * sc), ((dx - x0 + dw) * sc, (dy - y0 + dh) * sc), col, th)
        for i, t in enumerate(trect):
            if _overlap(t, (x0, y0, x1 - x0, y1 - y0)) > 0.3:
                tx, ty, tw, th_ = t
                cv2.rectangle(crop, ((tx - x0) * sc + 3, (ty - y0) * sc + 3),
                              ((tx - x0 + tw) * sc - 3, (ty - y0 + th_) * sc - 3), (0, 255, 0), 1 if i in mt else 2)
        fn = os.path.join(out_dir, f'{base}_{tag}.png')
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(fn, crop)
        return fn

    for i, t in enumerate(truth):
        if i in mt:
            continue
        r = trect[i]
        ov = [(j, drect[j]) for j in range(len(drect)) if _overlap(r, drect[j]) > 0.3]
        rep['missed'].append({'idx': i, 'rect': list(r), 'name': t.get('name'), 'W': t['W'], 'H': t['H'],
                              'overlapping_dets': [list(d) for _, d in ov],
                              'overlapping_cells': [[dets[j].w, dets[j].h, bool(dets[j].suspect)] for j, _ in ov],
                              'img': overlay(r, t.get('name'), f'T{i}')})
    for j, d in enumerate(drect):
        if j in md:
            continue
        ov = [(i, trect[i]) for i in range(len(trect)) if _overlap(d, trect[i]) > 0.3]
        it = dets[j]
        rep['extra'].append({'idx': j, 'rect': list(d), 'cells': [it.w, it.h], 'suspect': bool(it.suspect),
                             'clipped': [bool(it.clipped_top), bool(it.clipped_bottom)],
                             'overlapping_truth': [{'idx': i, 'rect': list(r), 'name': truth[i].get('name')} for i, r in ov],
                             'img': overlay(d, 'det', f'D{j}')})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='*')
    ap.add_argument('--out', default=os.path.join(ROOT, '_seg_debug'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--no-crops', action='store_true')
    a = ap.parse_args()
    paths = a.images or _default_images()
    reps = []
    tm = td = tmatch = 0
    for p in paths:
        r = analyse(p, None if a.no_crops else a.out)
        reps.append(r)
        tm += r['truth']; td += r['dets']; tmatch += r['matched']
        print(f"{r['image']:24s} truth {r['truth']:4d}  det {r['dets']:4d}  matched {r['matched']:4d}  "
              f"recall {r['matched'] / max(1, r['truth']):.1%}  precision {r['matched'] / max(1, r['dets']):.1%}")
        for m in r['missed']:
            print(f"   MISSED  #{m['idx']} {m['rect']} {m['W']}x{m['H']} '{m['name']}'  "
                  f"covered by {len(m['overlapping_dets'])} det {m['overlapping_dets'][:4]}")
        for e in r['extra']:
            names = [(t['rect'], t['name']) for t in e['overlapping_truth']]
            print(f"   EXTRA   #{e['idx']} {e['rect']} {e['cells'][0]}x{e['cells'][1]}"
                  f"{' suspect' if e['suspect'] else ''}{' clip' if any(e['clipped']) else ''}  overlaps truth {names[:3]}")
    print(f"TOTAL recall {tmatch / max(1, tm):.1%} ({tmatch}/{tm})  precision {tmatch / max(1, td):.1%} ({tmatch}/{td})")
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(reps, f, indent=1)


if __name__ == '__main__':
    main()
