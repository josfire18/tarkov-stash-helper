"""
Mine inventory frames from raid recordings for scene-understanding work (identify/scene.py).

Samples every ``--step`` seconds of each video (seeking, so a 60 s clip costs ~30 decodes), keeps
frames where the cheap auto-scan detector (autoscan.detect) sees an item grid, drops near-duplicate
views (32x32 difference hash, same test the auto-scan trigger uses) and writes PNGs plus an
``index.json`` to a shared folder.  The index is rewritten after every video (atomic replace), so
readers can use the folder while mining is still running, and a re-run skips finished videos.

    python scripts/mine_scenes.py                                  # all default sources
    python scripts/mine_scenes.py --limit 10 --workers 6           # quick first pass
    python scripts/mine_scenes.py --video-glob "D:/Videos/Medal Clips/*.mp4"

index.json::

    {"version": 1,
     "frames": [{"file": "gf_Kaban_Eats_t034.0.png", "video": "<abs path>", "t": 34.0,
                 "w": 2560, "h": 1440, "is_inventory": true, "score": 1.0,
                 "menu_chrome": false, "in_raid": true}, ...],
     "negatives": [{"file": "neg/...png", ...}],      # a few non-inventory frames per video
     "videos_done": ["<abs path>", ...]}

``in_raid`` is the detector's lobby signal (no main-menu bottom bar => in raid); it is a heuristic,
not ground truth.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from autoscan.detect import InventoryDetector          # noqa: E402
from autoscan.trigger import thumbnail, view_hash      # noqa: E402

DEFAULT_OUT = r'D:\Documents\.. Scripting\Python\Tarkov Helper\data\scenes'
DEFAULT_GLOBS = [
    r'D:\Videos\Captures\EscapeFromTarkov\*.mkv',
    r'D:\Videos\Geforce Captures\**\*.mp4',
    r'D:\Videos\Medal Clips\**\*.mp4',
    r'D:\Videos\NVIDIA\**\*.mp4',
]
DUP_BITS = 0.06          # <= 6 % differing hash bits -> same view (trigger uses 7 %)


def _tag(path: str) -> str:
    p = path.replace('\\', '/').lower()
    src = ('cap' if '/captures/' in p else 'gf' if 'geforce' in p else 'medal' if 'medal' in p
           else 'nv' if 'nvidia' in p else 'vid')
    stem = re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(path))[0]).strip('_')[:48]
    return f'{src}_{stem}'


def _write_png(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ok:
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            fh.write(buf.tobytes())
        os.replace(tmp, path)


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which('ffmpeg')


def _probe_size(path: str):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _frames(path: str, step: float):
    """(t, frame) for the video's keyframes, at most one per ``step`` s.  Keyframes only
    (ffmpeg -skip_frame nokey): OBS .mkv recordings report a bogus duration and ignore
    OpenCV seeks, and keyframe-only decoding reads a 1 h 1080p60 recording in about a minute.
    ``t`` is keyframe-time from the stream's pts.  A still image yields itself once."""
    import subprocess
    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        im = cv2.imread(path)
        if im is not None:
            yield 0.0, im
        return
    exe = _ffmpeg()
    w, h = _probe_size(path)
    if not exe or w <= 0 or h <= 0:
        return
    cmd = [exe, '-loglevel', 'error', '-skip_frame', 'nokey', '-i', path, '-an', '-sn',
           '-vf', 'showinfo', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
    size = w * h * 3
    k, last = 0, -1e9
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            t = k * 2.0                          # OBS default keyint is 2 s; exact t is not needed
            k += 1
            if t - last < step:
                continue
            last = t
            yield t, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.kill()


def mine_video(args):
    path, out_dir, step, dense, n_neg = args
    det = InventoryDetector()
    tag = _tag(path)
    kept, negs, hashes = [], [], []
    neg_pool = []
    t0 = time.time()
    n = 0
    gen = _frames(path, step)
    send = None
    try:
        while True:
            t, fr = gen.send(send) if send is not None else next(gen)
            send = None
            n += 1
            d = det.detect(fr)
            if not d.is_inventory:
                if d.reason != 'blank frame' and len(neg_pool) < 16 and n % 7 == 0:
                    neg_pool.append((t, d, cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2))))
                continue
            send = dense
            h = view_hash(thumbnail(fr))
            if any(np.count_nonzero(h != q) <= DUP_BITS * h.size for q in hashes):
                continue
            hashes.append(h)
            name = f'{tag}_t{t:05.1f}.png'
            _write_png(os.path.join(out_dir, name), fr)
            kept.append({'file': name, 'video': path, 't': round(t, 2), 'w': int(fr.shape[1]),
                         'h': int(fr.shape[0]), 'is_inventory': True, 'score': round(d.score, 3),
                         'menu_chrome': bool(d.menu_chrome), 'in_raid': not d.menu_chrome})
    except StopIteration:
        pass
    # a few gameplay (non-inventory) frames per video: analyze_scene must say 'none' on them
    if neg_pool and n_neg > 0:
        rng = np.random.default_rng(abs(hash(path)) % (2 ** 32))
        picks = rng.choice(len(neg_pool), size=min(n_neg, len(neg_pool)), replace=False)
        for i in sorted(int(k) for k in picks):
            t, d, fr = neg_pool[i]
            name = f'neg/{tag}_t{t:05.1f}.png'
            _write_png(os.path.join(out_dir, name), fr)
            negs.append({'file': name, 'video': path, 't': round(t, 2), 'w': int(fr.shape[1]),
                         'h': int(fr.shape[0]), 'is_inventory': False, 'reason': d.reason})
    return path, kept, negs, n, time.time() - t0


def _load_index(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        if isinstance(d, dict) and 'frames' in d:
            d.setdefault('negatives', [])
            d.setdefault('videos_done', [])
            return d
    except Exception:
        pass
    return {'version': 1, 'frames': [], 'negatives': [], 'videos_done': []}


def _save_index(path: str, idx: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(idx, fh, indent=1)
    os.replace(tmp, path)


def _interleave(lists):
    out = []
    its = [list(x) for x in lists if x]
    while its:
        for x in list(its):
            out.append(x.pop(0))
            if not x:
                its.remove(x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video-glob', action='append', default=None)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--dense', type=float, default=1.0, help='step after an inventory hit')
    ap.add_argument('--neg', type=int, default=1, help='non-inventory frames kept per video')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'neg'), exist_ok=True)
    idx_path = os.path.join(a.out, 'index.json')
    idx = _load_index(idx_path)
    done = set(idx['videos_done'])
    groups = []
    for g in (a.video_glob or DEFAULT_GLOBS):
        groups.append(sorted(set(glob.glob(g, recursive=True))))
    todo = [p for p in _interleave(groups) if p not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(todo)} videos to mine -> {a.out}', flush=True)
    jobs = [(p, a.out, a.step, a.dense, a.neg) for p in todo]
    t0 = time.time()
    with Pool(max(1, a.workers)) as pool:
        for k, (path, kept, negs, n, dt) in enumerate(pool.imap_unordered(mine_video, jobs), 1):
            idx['frames'] += kept
            idx['negatives'] += negs
            idx['videos_done'].append(path)
            _save_index(idx_path, idx)
            print(f'[{k}/{len(jobs)}] {os.path.basename(path)}: {n} samples, {len(kept)} kept '
                  f'({sum(f["in_raid"] for f in kept)} raid) in {dt:.0f}s; total {len(idx["frames"])} '
                  f'frames, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
