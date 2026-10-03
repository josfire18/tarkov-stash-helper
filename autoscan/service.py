"""
The auto-scan service: a below-normal-priority thread that polls the game window at <= 2 Hz,
runs the cheap inventory detector on the passively captured frame and - only when an inventory
screen has settled - hands the full frame to the app's normal scan function.

Cost rules (Tarkov is main-thread bound; frame drops are the one thing this must not cause):
  * poll 2 Hz while an inventory is on screen, 1 Hz otherwise, 0.5 Hz after ~30 s of gameplay,
    one process lookup every 3 s while the game is not running;
  * detection runs on decimated data (autoscan.detect), a few ms per poll;
  * the identify pipeline runs only on a settled, menu-context inventory frame, never in a raid;
  * the thread runs at THREAD_PRIORITY_BELOW_NORMAL; the duplication is released when the game
    is gone, the feature is switched off or the app quits.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time

from .capture import CaptureManager
from .detect import InventoryDetector
from .collect import SceneCollector
from .trigger import Observation, Trigger, thumbnail, view_hash
from .winapi import GAME_EXE, Win32, locate_game

POLL_ACTIVE_S = 0.5          # inventory (or candidate) on screen
POLL_IDLE_S = 1.0            # menus / gameplay
POLL_RAID_S = 2.0            # after RAID_BACKOFF idle polls with no menu bar: gameplay
POLL_NO_GAME_S = 3.0
RAID_BACKOFF = 30
RELOCATE_S = 5.0

MESSAGES = {
    'disabled': 'Auto-scan off',
    'starting': 'Auto-scan starting',
    'no_game': 'Waiting for Tarkov to start',
    'minimized': 'Tarkov is minimised',
    'covered': 'Helper window is covering the game',
    'blank': 'Capture returned black frames',
    'waiting': 'Waiting for stash',
    'raid_inventory': 'Raid inventory open - not scanned mid-raid',
    'settling': 'Inventory found - waiting for it to settle',
    'scanning': 'Scanning...',
    'scanned': 'Stash scanned',
    'busy': 'Another scan is running',
    'failed': 'Last scan failed',
    'error': 'Capture error',
}


def _below_normal_priority():
    try:
        k = ctypes.WinDLL('kernel32')
        k.GetCurrentThread.restype = ctypes.c_void_p
        k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k.SetThreadPriority(k.GetCurrentThread(), -1)       # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:
        pass


class AutoScanner:
    def __init__(self, scan_fn, get_settings, busy_fn=None, win32=None, capture=None,
                 detector=None, clock=time.monotonic, wall=time.time, collect_dir=None):
        self.scan_fn = scan_fn                    # frame_bgr -> sell-scan payload dict
        self.get_settings = get_settings
        self.busy_fn = busy_fn or (lambda: False)
        self.win = win32 if win32 is not None else Win32()
        self.capture = capture if capture is not None else CaptureManager()
        self.detector = detector or InventoryDetector()
        self.clock, self.wall = clock, wall
        self.trigger = Trigger()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._game = None
        self._game_ts = -1e9
        self._idle_polls = 0
        self._status = {'state': 'starting'}
        self._result = None
        self._seq = 0
        self._last_scan_wall = 0.0
        self._last_detect_ms = 0.0
        self._last_error = ''
        self._last_grab_ms = 0.0
        # settled inventory views (lobby + raid) kept as the scene dataset (collect.py)
        self.collector = SceneCollector(collect_dir) if collect_dir else None

    # -- settings ----------------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool((self.get_settings() or {}).get('auto_scan', True))

    def _opts(self) -> dict:
        s = self.get_settings() or {}
        return {'exe': s.get('auto_scan_exe') or GAME_EXE, 'allow_raid': bool(s.get('auto_scan_in_raid', False))}

    # -- public ------------------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='autoscan', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self.capture.close()

    def poke(self):
        """Settings changed - re-evaluate now."""
        self._wake.set()

    def status(self) -> dict:
        with self._lock:
            st = dict(self._status)
            seq, ts = self._seq, self._last_scan_wall
        st.update(enabled=self.enabled(), seq=seq, last_scan_ts=ts,
                  last_scan_age=(self.wall() - ts) if ts else None,
                  message=MESSAGES.get(st.get('state'), ''),
                  backend=self.capture.backend_name, detect_ms=round(self._last_detect_ms, 1),
                  grab_ms=round(self._last_grab_ms, 1))
        return st

    def result(self, since: int = 0) -> dict:
        with self._lock:
            if self._result is None or self._seq <= since:
                return {'ready': False, 'seq': self._seq}
            return {'ready': True, 'seq': self._seq, 'ts': self._last_scan_wall, **self._result}

    # -- loop --------------------------------------------------------------------------------
    def _run(self):
        _below_normal_priority()
        with self.win.physical_pixels():
            while not self._stop.is_set():
                try:
                    delay = self.step()
                except Exception as e:                         # never let the thread die
                    self._set('error', detail=str(e))
                    delay = POLL_NO_GAME_S
                self._wake.wait(delay)
                self._wake.clear()
        self.capture.close()

    def _set(self, state: str, **kw):
        with self._lock:
            self._status = {'state': state, **kw}

    def _locate(self, now: float, exe: str):
        if self._game is None or now - self._game_ts >= RELOCATE_S:
            self._game = locate_game(self.win, exe, os.getpid())
            self._game_ts = now
        return self._game

    def step(self) -> float:
        """One poll.  Returns the delay until the next one."""
        now = self.clock()
        if not self.enabled():
            self.capture.close()
            self._set('disabled')
            return POLL_NO_GAME_S
        opts = self._opts()
        self.trigger.allow_raid = opts['allow_raid']
        game = self._locate(now, opts['exe'])
        if game is None:
            self.capture.close()
            self._set('no_game')
            return POLL_NO_GAME_S
        game_info = {'pid': game.pid, 'size': game.size, 'monitor': game.monitor.device}
        if game.iconic:
            self.capture.close()
            self._set('minimized', game=game_info)
            return POLL_IDLE_S
        if game.covered_by_us:
            self._set('covered', game=game_info)
            return POLL_IDLE_S
        if self.busy_fn():
            self._set('busy', game=game_info)
            return POLL_IDLE_S

        t0 = time.perf_counter()
        res = self.capture.grab(game.monitor, game.region_on_monitor())
        self._last_grab_ms = (time.perf_counter() - t0) * 1e3
        if res.error:
            self._set('error', detail=res.error, game=game_info)
            return POLL_NO_GAME_S
        if res.frame is None:                    # compositor had nothing new: view unchanged
            return POLL_ACTIVE_S
        det = self.detector.detect(res.frame)
        self._last_detect_ms = det.ms
        thumb = thumbnail(res.frame) if det.is_inventory else None
        if (thumb is not None and self.collector is not None
                and (self.get_settings() or {}).get('collect_scene_frames', True)):
            try:
                self.collector.observe(res.frame, view_hash(thumb), in_raid=not det.menu_chrome,
                                       meta={'monitor': game.monitor.device})
            except Exception as e:
                print(f'[collect] {e}')
        dec = self.trigger.observe(Observation(det.is_inventory, det.menu_chrome, thumb, res.blank), now)
        if dec.scan:
            self._scan(res.frame, thumb, now, game_info)
            return POLL_ACTIVE_S
        extra = {'detail': self._last_error} if dec.state == 'failed' else {}
        self._set(dec.state, game=game_info, score=round(det.score, 2), **extra)
        if dec.state in ('settling', 'scanned', 'failed', 'raid_inventory'):
            self._idle_polls = 0
            return POLL_ACTIVE_S
        self._idle_polls += 1
        return POLL_RAID_S if self._idle_polls >= RAID_BACKOFF and not det.menu_chrome else POLL_IDLE_S

    def _scan(self, frame, thumb, now, game_info):
        self._set('scanning', game=game_info)
        self.trigger.scan_started(thumb, now)
        try:
            payload = self.scan_fn(frame)
        except Exception as e:
            self.trigger.scan_failed(self.clock())
            self._last_error = f'scan failed: {e}'
            self._set('error', detail=self._last_error, game=game_info)
            return
        if not isinstance(payload, dict) or payload.get('error') or payload.get('grid_failed'):
            self.trigger.scan_failed(self.clock())
            msg = (payload or {}).get('error') or 'stash grid not detected'
            self._last_error = msg
            self._set('error', detail=msg, game=game_info)
            return
        with self._lock:
            self._seq += 1
            self._last_scan_wall = self.wall()
            self._result = payload
        self._set('scanned', game=game_info)


def make_blueprint(scanner: AutoScanner, get_settings, save_settings):
    """Flask routes: /api/autoscan/status, /api/autoscan/result?since=N, POST /api/autoscan/toggle."""
    from flask import Blueprint, jsonify, request
    bp = Blueprint('autoscan', __name__)

    @bp.route('/api/autoscan/status')
    def status():
        return jsonify(scanner.status())

    @bp.route('/api/autoscan/result')
    def result():
        try:
            since = int(request.args.get('since', 0))
        except ValueError:
            since = 0
        return jsonify(scanner.result(since))

    @bp.route('/api/autoscan/toggle', methods=['POST'])
    def toggle():
        on = bool((request.get_json(silent=True) or {}).get('enabled', True))
        s = get_settings()
        s['auto_scan'] = on
        save_settings(s)
        scanner.poke()
        return jsonify({'ok': True, 'enabled': on})

    return bp
