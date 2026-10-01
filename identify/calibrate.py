"""
Fit the logistic confidence calibration used by :mod:`identify.pipeline`.

Usage::

    python -m identify.calibrate data/eval/stash1.png [--truth full]

The pipeline reports ``confidence = sigmoid(b + sum(w_k * feature_k))`` over four
evidence features (see ``pipeline._decide``).  To get *errors* to learn from (a clean
PNG of one stash has almost none) the labelled screenshot is degraded in the ways real
captures degrade - JPEG at several qualities, sensor noise, and resampling - and the
detections of every variant are pooled.  To make the fit well-posed the stages are also switched off (no OCR / no DINO / neither), which
produces many more errors with the feature distributions the engine will have when Tesseract or
torch is missing.  Weights are fitted by L2-regularised IRLS on
"was the top-1 id right?" and the reliability table printed afterwards shows how well the
resulting probabilities match the observed accuracy.

Caveat (also in the report): the pool comes from one stash, so the calibration is fitted
in-sample; collect more labelled screenshots and re-run before trusting the absolute
numbers.  The *flag* (``uncertain``) only needs the ranking to be right.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

KEYS = ('margin', 'res', 'ocr', 'dino')


CONFIGS = ((True, True), (False, True), (True, False), (False, False))   # (DINO, OCR) availability


def collect(img_paths, which='full', kinds=('png', 'jpg95', 'jpg85', 'jpg70', 'jpg50', 'noise', 'stretch1.3333',
                                            'scale0.83', 'scale1.33')):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import test_scan as T
    X, y, meta = [], [], []
    for path in img_paths:
        img = cv2.imread(path)
        truth, _ = T.load_truth(path, which)
        for kind in kinds:
            v, f = T.make_variant(img, kind)
            for dino, ocr in CONFIGS:
                dets, _, _ = T.run_v2(v, dino, ocr)
                tr = [dict(t, rect=T._scale_rect(t['rect'], f)) for t in truth]
                mt = T.match_rects([t['rect'] for t in tr], [d['rect'] for d in dets])
                for i, j in mt.items():
                    t, d = tr[i], dets[j]
                    if t.get('uncertain') or not t.get('item_id'):
                        continue
                    feats = d['det'].evidence.get('features')
                    if not feats:
                        continue
                    X.append([feats[k] for k in KEYS])
                    y.append(1.0 if d['item_id'] == t['item_id'] else 0.0)
                    meta.append((os.path.basename(path), kind, int(dino) * 2 + int(ocr)))
    return np.array(X), np.array(y), meta


def fit_logistic(X, y, l2=1.0, iters=50):
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    R = np.eye(A.shape[1]) * l2
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        W = p * (1 - p) + 1e-6
        g = A.T @ (y - p) - R @ w
        H = (A * W[:, None]).T @ A + R
        w = w + np.linalg.solve(H, g)
    return w


def reliability(p, y, bins=(0, .5, .8, .9, .95, .99, 1.01)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--truth', default='full')
    a = ap.parse_args()
    X, y, _ = collect(a.images, a.truth)
    print(f'{len(y)} samples, {int((y == 0).sum())} wrong')
    w = fit_logistic(X, y)
    names = ('b',) + KEYS
    print('CALIB =', json.dumps({n: round(float(v), 3) for n, v in zip(names, w)}))
    p = 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))
    print('reliability (predicted bin: n, mean p, observed accuracy)')
    for lo, hi, n, mp, acc in reliability(p, y):
        print(f'  [{lo:.2f},{min(hi, 1):.2f})  n={n:4d}  p={mp:.3f}  acc={acc:.3f}')


if __name__ == '__main__':
    main()
