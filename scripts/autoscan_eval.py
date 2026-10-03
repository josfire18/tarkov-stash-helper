"""
Precision / recall of the auto-scan inventory detector.

Positives: labelled screenshots (data/eval/stash1.png, joey/*.png, web/*.png), each also pasted onto a
dark 2560x1440 canvas (with a main-menu bar for crops) to simulate a full-screen frame.
Negatives: frames sampled from raid recordings (any folders of .mp4/.mkv you pass, or a folder of
already-extracted PNG/JPG frames).

    python scripts/autoscan_eval.py --video-glob "D:/Videos/Geforce Captures/Escape From Tarkov/*.mp4" --step 4
    python scripts/autoscan_eval.py --frames-dir some/frames
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from autoscan.detect import InventoryDetector      # noqa: E402
from autoscan_frames import menu_bar               # noqa: E402


def positives():
    paths = ['data/eval/stash1.png'] + sorted(glob.glob('data/eval/joey/*.png')) + [
        p for p in sorted(glob.glob('data/eval/web/w*.png')) if 'sheet' not in p]
    for p in paths:
        im = cv2.imread(os.path.join(ROOT, p))
        if im is not None:
            yield os.path.basename(p), im
            if im.shape[0] < 1400 or im.shape[1] < 2500:          # crop: also as a full 1440p frame
                c = np.full((1440, 2560, 3), (26, 28, 28), np.uint8)
                h, w = min(im.shape[0], 1380), min(im.shape[1], 2400)
                c[20:20 + h, 60:60 + w] = im[:h, :w]
                yield os.path.basename(p) + ' (on canvas)', menu_bar(c)


def video_frames(pattern, step):
    for f in sorted(glob.glob(pattern)):
        cap = cv2.VideoCapture(f)
        fps, n = cap.get(5) or 60.0, int(cap.get(7) or 0)
        stride = max(1, int(round(step * fps)))
        for i in range(n):
            if not cap.grab():
                break
            if i % stride == stride // 2:
                ok, fr = cap.retrieve()
                if ok:
                    yield f'{os.path.basename(f)}@{i}', fr
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-glob', action='append', default=[])
    ap.add_argument('--frames-dir', action='append', default=[])
    ap.add_argument('--step', type=float, default=4.0)
    a = ap.parse_args()
    det = InventoryDetector()
    tp = fn = 0
    ms = []
    print('POSITIVES')
    for name, im in positives():
        r = det.detect(im)
        ms.append(r.ms)
        tp, fn = tp + r.is_inventory, fn + (not r.is_inventory)
        print(f'  {"hit " if r.is_inventory else "MISS"} {name:42s} score {r.score:.2f} menu-bar {r.menu_chrome} {r.ms:.1f} ms')
    fp = tn = 0
    nms = []

    def neg(name, im):
        nonlocal fp, tn
        r = det.detect(im)
        nms.append(r.ms)
        if r.is_inventory:
            fp += 1
            print(f'  FALSE POSITIVE {name} score {r.score:.2f}')
        else:
            tn += 1
    print('NEGATIVES')
    for pat in a.video_glob:
        for name, fr in video_frames(pat, a.step):
            neg(name, fr)
    for d in a.frames_dir:
        for p in sorted(glob.glob(os.path.join(d, '*.png')) + glob.glob(os.path.join(d, '*.jpg'))):
            neg(os.path.basename(p), cv2.imread(p))
    prec = tp / (tp + fp) if tp + fp else float('nan')
    print(f'\nrecall {tp}/{tp + fn} = {tp / max(1, tp + fn):.1%}   precision {tp}/{tp + fp} = {prec:.1%}   '
          f'false positives {fp}/{fp + tn} negative frames')
    print(f'cost per frame: positives mean {np.mean(ms):.1f} ms (max {max(ms):.1f}); '
          f'negatives mean {np.mean(nms) if nms else 0:.1f} ms (max {max(nms) if nms else 0:.1f})')


if __name__ == '__main__':
    main()
