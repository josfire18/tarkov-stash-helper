"""
The scan service.  Two ways in: the scan hotkey (``request_scan``: one capture of what is on screen,
scanned at once, the screen duplication released again when auto-scan is off) and auto-scan (off by
default).  Auto-scan is a below-normal-priority thread that polls the game window at <= 2 Hz,
runs the cheap inventory detector on the passively captured frame and - only when an inventory
screen has settled - hands the full frame to the app's normal scan function.

Cost rules (Tarkov is main-thread bound; frame drops are the one thing this must not cause):
  * poll 4 Hz while an inventory is on screen (settled = two identical polls, ~0.5 s), 1 Hz otherwise, 0.5 Hz after ~30 s of gameplay,
    one process lookup every 3 s while the game is not running;
  * detection runs on decimated data (autoscan.detect), a few ms per poll;
  * the identify pipeline runs only on a settled, menu-context inventory frame, never in a raid;
  * while the game log says the player is matching, loading or in a raid NOTHING runs: no capture,
    detection, scan or frame collection, the screen duplication is released, the loop only re-reads
    the log tail every POLL_RAID_S.  This is a stash helper; a raid never pays for it;
  * the poll thread and the scan worker run at THREAD_PRIORITY_BELOW_NORMAL (the app also drops
    its whole process to below normal, so OpenCV's worker threads give way to the game too); the
    duplication is released when the game is gone, the feature is switched off or the app quits.
"""
from __future__ import annotations

import math
import ctypes
import json
import os
import threading
import time

import cv2

from .capture import CaptureManager
from .detect import InventoryDetector, resolve_in_raid
from .collect import SceneCollector
from .trigger import Observation, Trigger, thumbnail, view_hash
from .winapi import GAME_EXE, Win32, locate_game

POLL_ACTIVE_S = 0.25         # inventory (or candidate) on screen: live view
POLL_IDLE_S = 1.0            # menus / gameplay
POLL_RAID_S = 2.0            # after RAID_BACKOFF idle polls with no menu bar: gameplay
POLL_NO_GAME_S = 3.0
RAID_BACKOFF = 30
RELOCATE_S = 5.0
RAID_MAX_S = 2 * 3600.0      # a log that has said "in raid" for longer than this lost its exit line: stop trusting it
PAUSE_STATES = ('matching', 'loading', 'in_raid')   # game log states in which nothing at all runs
PROV_WAIT_S = 60.0          # a new view waits at most this long for the running scan's first look before cancelling it

# states a hotkey scan leaves behind that "auto-scan off" must not paint over
MANUAL_STATES = ('scanning', 'verifying', 'scanned', 'error', 'blank')

MESSAGES = {
    'disabled': 'Auto-scan off - press the scan hotkey in the game to scan',
    'starting': 'Auto-scan starting',
    'no_game': 'Waiting for Tarkov to start',
    'minimized': 'Tarkov is minimised',
    'covered': 'Helper window is covering the game',
    'blank': 'Capture returned black frames',
    'waiting': 'Waiting for stash',
    'raid_inventory': 'Raid inventory open - only the stash is scanned',
    'in_raid': 'In a raid - paused until you are back in the lobby',
    'settling': 'Inventory found - waiting for it to settle',
    'scanning': 'Scanning...',
    'verifying': 'First look ready - verifying items...',
    'scanned': 'Stash scanned',
    'busy': 'Another scan is running',
    'failed': 'Last scan failed',
    'error': 'Capture error',
}


def below_normal_process():
    """Drop the whole process to BELOW_NORMAL_PRIORITY_CLASS: every thread of the helper (OpenCV's
    and numpy's worker pools included) then gives way to the game the moment it wants a core."""
    try:
        k = ctypes.WinDLL('kernel32')
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x4000))   # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


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
                 detector=None, clock=time.monotonic, wall=time.time, collect_dir=None, progressive=False,
                 raid_state_fn=None, mode_fn=None):
        # frame_bgr -> sell-scan payload dict.  With ``progressive`` the scan runs in a worker thread
        # and is called as ``scan_fn(frame, publish=cb, cancel=fn)``: ``cb(payload)`` publishes a
        # provisional result (own seq), the return value is the final one (next seq); ``fn()`` turns
        # true when a newer view superseded the scan, which then raises / returns and is dropped.
        self.scan_fn = scan_fn
        self.raid_state_fn = raid_state_fn        # () -> {'state', 'since', 'source'} from the game's log, or None
        self.mode_fn = mode_fn                    # () -> game-mode info dict (detected / effective), for the status
        self._raid_state = None
        self._was_in_raid = False
        self.progressive = progressive
        self._job = None                          # (thread, cancel Event) of the running progressive scan
        self._prov = threading.Event()            # the running scan has published (or no longer can publish) its first look
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
        self._manual = threading.Event()          # the scan hotkey was pressed: scan the screen once
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
        self._cond = threading.Condition()        # wakes the SSE streams when seq / state change
        self._frame = None                         # BGR frame of the last published result
        self._frame_jpeg = None                    # (seq, bytes) - encoded lazily on first request
        # settled inventory views (lobby + raid) kept as the scene dataset (collect.py)
        self.collector = SceneCollector(collect_dir) if collect_dir else None

    # -- settings ----------------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool((self.get_settings() or {}).get('auto_scan', False))

    def _opts(self) -> dict:
        s = self.get_settings() or {}
        return {'exe': s.get('auto_scan_exe') or GAME_EXE}

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
        self._supersede(wait_first_look=False)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self.capture.close()

    def poke(self):
        """Settings changed - re-evaluate now."""
        self._wake.set()

    def request_scan(self):
        """Scan what is on screen now (the hotkey), whether auto-scan is on or off.  Runs on the
        scan thread; returns at once."""
        self._manual.set()
        self._wake.set()

    def status(self) -> dict:
        with self._lock:
            st = dict(self._status)
            seq, ts = self._seq, self._last_scan_wall
            scene = (self._result or {}).get('scene')
            timing = (self._result or {}).get('timing')
        st.update(scene=scene, timing=timing, enabled=self.enabled(), seq=seq, last_scan_ts=ts,
                  last_scan_age=(self.wall() - ts) if ts else None,
                  message=MESSAGES.get(st.get('state'), ''), raid_state=self._log_raid_state(),
                  backend=self.capture.backend_name, detect_ms=round(self._last_detect_ms, 1),
                  grab_ms=round(self._last_grab_ms, 1))
        st['mode'] = self._mode_info()
        st['hotkey'] = (self.get_settings() or {}).get('hotkey')
        return st

    def _mode_info(self):
        if self.mode_fn is None:
            return None
        try:
            return self.mode_fn()
        except Exception:
            return None

    def result(self, since: int = 0) -> dict:
        with self._lock:
            if self._result is None or self._seq <= since:
                return {'ready': False, 'seq': self._seq}
            return {'ready': True, 'seq': self._seq, 'ts': self._last_scan_wall,
                    'frame_url': f'/api/autoscan/frame?seq={self._seq}', **self._result}

    def frame_jpeg(self, seq=None):
        """The frame the published result was computed from, JPEG-encoded on first request (the
        result payload stays small; the browser caches the picture by ``seq``)."""
        with self._lock:
            cur, frame = self._seq, self._frame
            cached = self._frame_jpeg
        if seq is not None and seq != cur:
            return cached[1] if cached and cached[0] == seq else None
        if frame is None:
            return None
        if cached and cached[0] == cur:
            return cached[1]
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        data = buf.tobytes()
        with self._lock:
            if self._seq == cur:
                self._frame_jpeg = (cur, data)
        return data

    def state_key(self) -> tuple:
        with self._lock:
            return (self._seq, self._status.get('state'))

    def wait_change(self, seen, timeout: float) -> tuple:
        """Block until ``(seq, state)`` differs from ``seen`` (or ``timeout``); returns the current
        pair.  Feeds the SSE stream."""
        with self._cond:
            cur = self.state_key()
            if cur == seen:
                self._cond.wait(timeout)
                cur = self.state_key()
        return cur

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
            changed = self._status.get('state') != state
            self._status = {'state': state, **kw}
        if changed:
            with self._cond:
                self._cond.notify_all()

    def _log_raid_state(self):
        """The game log's raid state (None when unavailable); kept for the status endpoint."""
        st = None
        if self.raid_state_fn is not None:
            try:
                st = self.raid_state_fn()
            except Exception:
                st = None
        self._raid_state = st
        return st

    def _in_raid_by_log(self) -> bool:
        """The game's own log says the player is matching, loading or in a raid (False when the log is
        unreadable or that state is older than RAID_MAX_S, i.e. the exit line was never logged)."""
        st = self._log_raid_state()
        if not st or st.get('state') not in PAUSE_STATES:
            return False
        since = st.get('since')
        return not (since and self.wall() - since > RAID_MAX_S)

    def _pause_for_raid(self, game_info):
        """Release everything a raid must not pay for: the running scan, the screen duplication."""
        if not self._was_in_raid:
            self._was_in_raid = True
            self._supersede(wait_first_look=False)
            self.capture.close()
            self.trigger.reset_view()                 # the first lobby view afterwards is scanned afresh
            self._idle_polls = 0
        self._set('in_raid', game=game_info)

    def _locate(self, now: float, exe: str):
        if self._game is None or now - self._game_ts >= RELOCATE_S:
            self._game = locate_game(self.win, exe, os.getpid())
            self._game_ts = now
        return self._game

    def step(self) -> float:
        """One poll.  Returns the delay until the next one."""
        now = self.clock()
        manual = self._manual.is_set()
        self._manual.clear()
        auto = self.enabled()
        if not auto and not manual:
            self.capture.close()                  # off: nothing is captured until the hotkey is pressed
            if not self.scanning() and self._status.get('state') not in MANUAL_STATES:
                self._set('disabled')
            return POLL_NO_GAME_S
        opts = self._opts()
        if manual:
            self._game_ts = -1e9                  # the window may have moved since the last look
        game = self._locate(now, opts['exe'])
        if game is None:
            self.capture.close()
            self._set('no_game')
            return POLL_NO_GAME_S
        game_info = {'pid': game.pid, 'size': game.size, 'monitor': game.monitor.device}
        if self._in_raid_by_log():
            # Raid: no capture, no detection, no scan, no frame collection - nothing runs next to the
            # game (raids are won by frame time; views change every few seconds, so a scan would never
            # finish anyway).  Only the log tail (a stat() every POLL_RAID_S) is read, to notice the exit.
            self._pause_for_raid(game_info)
            return POLL_RAID_S
        self._was_in_raid = False
        if game.iconic:
            self.capture.close()
            self._set('minimized', game=game_info)
            return POLL_IDLE_S
        if game.covered_by_us:
            self._set('covered', game=game_info)
            return POLL_IDLE_S
        if self.busy_fn() and not self.scanning():
            self._set('busy', game=game_info)
            return POLL_IDLE_S
        if manual:
            return self._manual_scan(game, game_info, now, release=not auto)

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
        in_raid = resolve_in_raid(self._log_raid_state(), det.menu_chrome)
        thumb = thumbnail(res.frame) if det.is_inventory else None
        if (thumb is not None and not in_raid and self.collector is not None
                and (self.get_settings() or {}).get('collect_scene_frames', True)):
            try:
                self.collector.observe(res.frame, view_hash(thumb), in_raid=in_raid,
                                       meta={'monitor': game.monitor.device})
            except Exception as e:
                print(f'[collect] {e}')
        dec = self.trigger.observe(Observation(det.is_inventory, not in_raid, thumb, res.blank), now)
        if dec.scan:
            self._scan(res.frame, thumb, now, game_info)
            return POLL_ACTIVE_S
        if self.scanning() and dec.state in ('scanned', 'settling'):
            return POLL_ACTIVE_S                  # certification of the current view is still running
        extra = {'detail': self._last_error} if dec.state == 'failed' else {}
        self._set(dec.state, game=game_info, score=round(det.score, 2), **extra)
        if dec.state == 'raid_inventory':          # the log could not tell; the missing menu bar did
            return POLL_RAID_S
        if dec.state in ('settling', 'scanned', 'failed'):
            self._idle_polls = 0
            return POLL_ACTIVE_S
        self._idle_polls += 1
        return POLL_RAID_S if self._idle_polls >= RAID_BACKOFF and in_raid else POLL_IDLE_S

    def _manual_scan(self, game, game_info, now, release: bool) -> float:
        """The hotkey: capture the game once and scan that frame, inventory detector or not."""
        region = game.region_on_monitor()
        t0 = time.perf_counter()
        res = self.capture.grab(game.monitor, region)
        if res.frame is None and not res.error:   # "nothing new since the last grab": open afresh
            self.capture.close()
            res = self.capture.grab(game.monitor, region)
        self._last_grab_ms = (time.perf_counter() - t0) * 1e3
        if release:
            self.capture.close()                  # auto-scan off: hold nothing between hotkey presses
        if res.error or res.frame is None:
            self._set('error', detail=res.error or 'no frame from the screen capture', game=game_info)
            return POLL_NO_GAME_S
        if res.blank:
            self._set('blank', game=game_info)
            return POLL_NO_GAME_S
        self._scan(res.frame, thumbnail(res.frame), now, game_info)
        return POLL_NO_GAME_S if release else POLL_ACTIVE_S

    def scanning(self) -> bool:
        """A progressive scan (first look published, certification pending) is running."""
        job = self._job
        return bool(job and job[0].is_alive())

    def _supersede(self, wait_first_look: bool = True):
        """Cancel the pending certification (a newer view arrived / shutdown) and wait for it.

        A scan that has not yet published its first look is let alone until it has (bounded by
        ``PROV_WAIT_S``): otherwise a player who keeps changing view would cancel every scan
        before anything was ever shown.  Once the first look is out, cancelling is free."""
        job = self._job
        if job is None:
            return
        if wait_first_look and job[0] is not threading.current_thread():
            deadline = time.monotonic() + PROV_WAIT_S
            while job[0].is_alive() and not self._prov.is_set() and time.monotonic() < deadline:
                self._prov.wait(0.05)
        with self._lock:
            job[1].set()                          # under the lock: a publish in flight is either done or dropped
        if job[0] is not threading.current_thread():
            job[0].join(timeout=15)
        self._job = None

    def _publish(self, payload, frame, t0, game_info, final, cancel=None, born_seq=None):
        payload['timing'] = {'scan_ms': round((time.perf_counter() - t0) * 1e3), 'final': final}
        with self._lock:
            if cancel is not None and cancel.is_set():
                # superseded while the payload was being built.  Its first look is still better than
                # nothing while no newer result has been published (the newer scan is still working)
                if final or born_seq is None or self._seq != born_seq or self._stop.is_set():
                    return
            self._seq += 1
            self._last_scan_wall = self.wall()
            self._result = payload
            self._frame = frame
            self._frame_jpeg = None
        self._set('scanned' if final else 'verifying', game=game_info)
        with self._cond:
            self._cond.notify_all()

    def _work(self, frame, cancel, t0, game_info, prov=None):
        _below_normal_priority()                  # the game wins every core it wants while we scan
        prov = prov if prov is not None else self._prov
        born_seq = self._seq

        def publish(payload):
            if isinstance(payload, dict):
                try:
                    self._publish(payload, frame, t0, game_info, False, cancel, born_seq)
                finally:
                    prov.set()

        try:
            payload = self.scan_fn(frame, publish=publish, cancel=cancel.is_set)
        except Exception as e:
            prov.set()
            if cancel.is_set():
                return                            # superseded: not a failure
            self._fail(f'scan failed: {e}', game_info)
            return
        prov.set()                                # a scan with nothing to show first has nothing to wait for
        if cancel.is_set():
            return
        if not isinstance(payload, dict) or payload.get('error') or payload.get('grid_failed'):
            self._fail((payload or {}).get('error') or 'stash grid not detected', game_info)
            return
        self._publish(payload, frame, t0, game_info, True, cancel)

    def _fail(self, msg, game_info):
        self.trigger.scan_failed(self.clock())
        self._last_error = msg
        self._set('error', detail=msg, game=game_info)

    def _scan(self, frame, thumb, now, game_info):
        t0 = time.perf_counter()
        if self.progressive:
            self._supersede()
            self._set('scanning', game=game_info)
            self.trigger.scan_started(thumb, now)
            cancel = threading.Event()
            self._prov = prov = threading.Event()
            th = threading.Thread(target=self._work, args=(frame, cancel, t0, game_info, prov),
                                  name='autoscan-scan', daemon=True)
            self._job = (th, cancel)
            th.start()
            return
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
        payload['timing'] = {'scan_ms': round((time.perf_counter() - t0) * 1e3)}
        with self._lock:
            self._seq += 1
            self._last_scan_wall = self.wall()
            self._result = payload
            self._frame = frame
            self._frame_jpeg = None
        self._set('scanned', game=game_info)
        with self._cond:
            self._cond.notify_all()


def make_blueprint(scanner: AutoScanner, get_settings, save_settings):
    """Flask routes: /api/autoscan/status, /api/autoscan/result?since=N, POST /api/autoscan/toggle."""
    from flask import Blueprint, Response, jsonify, request
    bp = Blueprint('autoscan', __name__)

    @bp.route('/api/autoscan/frame')
    def frame():
        try:
            seq = int(request.args['seq']) if 'seq' in request.args else None
        except ValueError:
            seq = None
        data = scanner.frame_jpeg(seq)
        if data is None:
            return Response(status=404)
        # the picture of one result never changes: the browser may keep it for good
        return Response(data, mimetype='image/jpeg',
                        headers={'Cache-Control': 'private, max-age=31536000, immutable'})

    @bp.route('/api/autoscan/stream')
    def stream():
        """Server-Sent Events: one ``data: {"seq": N, "state": "..."}`` event each time the result
        sequence number or the scanner state changes (and once on connect).  The page then GETs
        /api/autoscan/result?since=N; polling stays the fallback.  ``max_s`` ends the stream after
        that many seconds (tests); a ``: ping`` comment every 15 s keeps idle connections open."""
        try:
            max_s = float(request.args.get('max_s', 0))  # nosemgrep: nan-injection -- non-finite values rejected just below
            if not math.isfinite(max_s):
                raise ValueError
        except ValueError:
            max_s = 0.0

        def gen():
            t_end = time.monotonic() + max_s if max_s > 0 else None
            seen = None
            last = time.monotonic()
            yield 'retry: 2000\n\n'
            while True:
                left = 15.0 if t_end is None else min(15.0, t_end - time.monotonic())
                if left <= 0:
                    return
                cur = scanner.wait_change(seen, left) if seen is not None else scanner.state_key()
                if cur != seen:
                    seen = cur
                    st = scanner.status()
                    yield 'data: ' + json.dumps({'seq': cur[0], 'state': cur[1], 'scene': st.get('scene')}) + '\n\n'
                    last = time.monotonic()
                elif time.monotonic() - last >= 15:
                    yield ': ping\n\n'
                    last = time.monotonic()
        return Response(gen(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

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
