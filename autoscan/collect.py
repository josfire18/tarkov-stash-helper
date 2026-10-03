"""Keep the settled inventory views the auto-scan sees (lobby and raid) as a local dataset.

The scene layer (which grid is the player's own gear, which is loot, which is the stash, which is
an open container window) is built and tested against real frames, and the only source of real
in-raid inventory frames is the player's own sessions.  Frames never leave the machine.

A view is kept once it has been stable for two polls (same 32x32 difference hash) and differs from
every view kept in this session.  Writing happens on a worker thread at the poller's priority, the
folder is a ring (oldest files go first) and ``index.jsonl`` gets one line per frame.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time

import cv2
import numpy as np

MAX_FRAMES = 300
DUP_BITS = 0.06


class SceneCollector:
    def __init__(self, out_dir: str, max_frames: int = MAX_FRAMES):
        self.out_dir = out_dir
        self.max_frames = max_frames
        self._prev = None
        self._kept: list[np.ndarray] = []
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._thread = None

    @staticmethod
    def _same(a, b) -> bool:
        return a is not None and b is not None and np.count_nonzero(a != b) <= DUP_BITS * a.size

    def observe(self, frame, view_hash, in_raid: bool, meta: dict | None = None) -> bool:
        """Called for every inventory poll.  Returns True when the frame was queued for saving."""
        stable = self._same(view_hash, self._prev)
        self._prev = view_hash
        if not stable or any(self._same(view_hash, k) for k in self._kept):
            return False
        self._kept.append(view_hash)
        if len(self._kept) > 512:
            self._kept = self._kept[-512:]
        try:
            self._q.put_nowait((frame.copy(), bool(in_raid), dict(meta or {}), time.time()))
        except queue.Full:
            return False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._drain, name='scene-collect', daemon=True)
            self._thread.start()
        return True

    def _drain(self):
        while True:
            try:
                frame, in_raid, meta, ts = self._q.get(timeout=5)
            except queue.Empty:
                return
            try:
                self._save(frame, in_raid, meta, ts)
            except Exception as e:                       # a full disk must not stop the auto-scan
                print(f'[collect] could not save a scene frame: {e}')

    def _save(self, frame, in_raid, meta, ts):
        os.makedirs(self.out_dir, exist_ok=True)
        name = f"{int(ts * 1000)}-{'raid' if in_raid else 'lobby'}.png"   # time first: the ring drops the oldest
        ok, buf = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return
        with open(os.path.join(self.out_dir, name), 'wb') as fh:
            fh.write(buf.tobytes())
        with open(os.path.join(self.out_dir, 'index.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'file': name, 'ts': round(ts, 3), 'in_raid': in_raid,
                                 'w': int(frame.shape[1]), 'h': int(frame.shape[0]), **meta}) + '\n')
        pngs = sorted(f for f in os.listdir(self.out_dir) if f.endswith('.png'))
        for f in pngs[:max(0, len(pngs) - self.max_frames)]:
            try:
                os.remove(os.path.join(self.out_dir, f))
            except OSError:
                pass
