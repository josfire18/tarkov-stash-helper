#!/usr/bin/env python3
"""
Segmentation-only robustness sweep (grid + segmentation, no catalog / OCR / GPU).

Degrades each labelled screenshot (JPEG quality ladder, blur, noise, rescale, 4:3 stretch - the same
variants as ``test_scan.py --robustness``) and reports segmentation recall / precision against the
rectangle truth scaled the same way, so a change to ``identify/grid.py`` or ``identify/segment.py``
can be checked for fragility in seconds instead of minutes.

    python scripts/seg_robust.py [images...] [--kinds png,jpg85,...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import cv2

from identify.grid import detect_grid
from identify.segment import segment_panel
from seg_diagnose import _default_images, _match

KINDS = ['png', 'jpg95', 'jpg85', 'jpg70', 'jpg50', 'blur', 'noise', 'scale0.83', 'scale1.33', 'stretch1.3333']


def variant(img, kind):
    import numpy as np
    if kind == 'png':
        return img, (1.0, 1.0)
    if kind.startswith('stretch'):
        fx = float(kind[7:] or 1.3333)
        return cv2.resize(img, None, fx=fx, fy=1.0, interpolation=cv2.INTER_CUBIC), (fx, 1.0)
    if kind.startswith('jpg'):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(kind[3:])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), (1.0, 1.0)
    if kind == 'blur':
        return cv2.GaussianBlur(img, (0, 0), 1.0), (1.0, 1.0)
    if kind == 'noise':
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8), (1.0, 1.0)
    if kind.startswith('scale'):
        f = float(kind[5:])
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC), (f, f)
    raise ValueError(kind)


def run(path, kind):
    img = cv2.imread(path)
    with open(os.path.splitext(path)[0] + '.truth.full.json', encoding='utf-8') as f:
        truth = json.load(f)
    v, (fx, fy) = variant(img, kind)
    tr = [(round(t['rect'][0] * fx), round(t['rect'][1] * fy), round(t['rect'][2] * fx), round(t['rect'][3] * fy))
          for t in truth]
    g = detect_grid(v)
    dets = [tuple(it.rect) for i, p in enumerate(g.panels) for it in segment_panel(v, p, i) if not it.empty]
    mt, _ = _match(tr, dets)
    return len(tr), len(dets), len(mt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='*')
    ap.add_argument('--kinds', default=','.join(KINDS))
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    paths = a.images or _default_images()
    kinds = a.kinds.split(',')
    table = {}
    print(f"{'image':24s}" + ''.join(f'{k:>16s}' for k in kinds))
    tot = {k: [0, 0, 0] for k in kinds}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        row = f'{name:24s}'
        for k in kinds:
            nt, nd, nm = run(p, k)
            table.setdefault(name, {})[k] = [nt, nd, nm]
            for i, v in enumerate((nt, nd, nm)):
                tot[k][i] += v
            row += f'  {nm / max(1, nt):5.1%}/{nm / max(1, nd):5.1%}'
        print(row)
    print(f"{'TOTAL recall/precision':24s}" + ''.join(f'  {tot[k][2] / max(1, tot[k][0]):5.1%}/{tot[k][2] / max(1, tot[k][1]):5.1%}' for k in kinds))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(table, f, indent=1)


if __name__ == '__main__':
    main()
