"""
Passive frame capture with a primary method, a fallback and black-frame detection.

Primary: DXGI Desktop Duplication through the ``dxcam`` package.  It is pull based (we ask for a
frame ~2x a second; nothing is delivered per vsync), copies only the game's region from the GPU
and, unlike GDI, is the API OBS "Display Capture" / Discord use.  Measured on the target machine:
~3-9 ms per 2560x1440 grab versus ~75 ms for GDI.  Windows 10 runs DX11 "exclusive fullscreen"
games through Fullscreen Optimisations (a flip-model surface the DWM still composes), which is
why Desktop Duplication sees them; a game that really bypasses the DWM yields black frames,
which :func:`autoscan.detect.frame_is_blank` catches.

Fallback: ``mss`` (GDI BitBlt of the composed desktop) - slower, but needs nothing but the stdlib
ctypes and works when DXGI duplication is unavailable (RDP session, GPU driver reset, ...).

Windows.Graphics.Capture was evaluated and rejected: it delivers a callback per composed frame
(165/s on this monitor, each one a GIL round trip) and Microsoft states it cannot be relied on for
exclusive fullscreen either.

Passive only: this module reads pixels the compositor already produced.  No game memory, no
hooks, no injection, no input.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .detect import frame_is_blank


class BackendError(Exception):
    """The backend cannot capture right now (not installed, access lost, ...)."""


@dataclass
class CaptureResult:
    frame: np.ndarray | None        # BGR uint8, or None (no new frame / failure)
    backend: str
    blank: bool = False             # an all-black frame (blocked capture path)
    unchanged: bool = False         # the compositor had no new frame since the last grab
    error: str | None = None


class DxgiBackend:
    name = 'dxgi'

    def __init__(self):
        self._cam = None
        self._device = None
        self._have_frame = False

    @staticmethod
    def _output_index(dxcam, device_name: str, size: tuple, primary: bool):
        """(device_idx, output_idx) of the DXGI output for a monitor: by device name, else by
        resolution, else the primary output."""
        factory = getattr(dxcam, '__factory', None)
        outs = getattr(factory, 'outputs', None)
        if outs:
            for di, row in enumerate(outs):
                for oi, o in enumerate(row):
                    if getattr(o, 'devicename', None) == device_name:
                        return di, oi
            for di, row in enumerate(outs):
                for oi, o in enumerate(row):
                    if tuple(o.resolution) == tuple(size):
                        return di, oi
        return 0, None if primary else 0

    def _open(self, monitor):
        try:
            import dxcam
        except Exception as e:                                   # not installed / COM failure
            raise BackendError(f'dxcam unavailable: {e}') from e
        self.close()
        di, oi = self._output_index(dxcam, monitor.device, monitor.size, monitor.primary)
        try:
            self._cam = dxcam.create(device_idx=di, output_idx=oi, output_color='BGR')
        except Exception as e:
            raise BackendError(f'DXGI duplication failed: {e}') from e
        self._device = monitor.device
        self._have_frame = False

    def grab(self, monitor, region) -> CaptureResult:
        if self._cam is None or self._device != monitor.device:
            self._open(monitor)
        try:
            frame = self._cam.grab(region=tuple(region))
            if frame is None and not self._have_frame:           # first grab after open
                for _ in range(25):
                    time.sleep(0.02)
                    frame = self._cam.grab(region=tuple(region))
                    if frame is not None:
                        break
        except Exception as e:
            self.close()
            raise BackendError(f'DXGI grab failed: {e}') from e
        if frame is None:
            return CaptureResult(None, self.name, unchanged=self._have_frame)
        self._have_frame = True
        return CaptureResult(frame, self.name, blank=frame_is_blank(frame))

    def close(self):
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass


class GdiBackend:
    name = 'gdi'

    def __init__(self):
        self._sct = None

    def grab(self, monitor, region) -> CaptureResult:
        try:
            import mss
            if self._sct is None:
                self._sct = mss.mss()
            l, t, r, b = region
            raw = self._sct.grab({'left': monitor.rect[0] + l, 'top': monitor.rect[1] + t,
                                  'width': r - l, 'height': b - t})
        except Exception as e:
            self.close()
            raise BackendError(f'GDI grab failed: {e}') from e
        frame = np.frombuffer(raw.bgra, np.uint8).reshape(raw.height, raw.width, 4)[:, :, :3]
        frame = np.ascontiguousarray(frame)
        return CaptureResult(frame, self.name, blank=frame_is_blank(frame))

    def close(self):
        sct, self._sct = self._sct, None
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass


class CaptureManager:
    """Uses the first backend that delivers a non-black frame; drops to the next after errors or
    ``blank_limit`` consecutive black frames and re-tries the primary every ``retry_s`` seconds."""

    def __init__(self, backends=None, blank_limit: int = 3, retry_s: float = 60.0, clock=time.monotonic):
        self.backends = backends if backends is not None else [DxgiBackend(), GdiBackend()]
        self.blank_limit = blank_limit
        self.retry_s = retry_s
        self.clock = clock
        self.active = 0
        self._blanks = 0
        self._demoted_at = 0.0
        self.last_error: str | None = None

    @property
    def backend_name(self) -> str:
        return self.backends[self.active].name

    def _demote(self, why: str):
        self.last_error = why
        self._blanks = 0
        if self.active + 1 < len(self.backends):
            self.backends[self.active].close()
            self.active += 1
            self._demoted_at = self.clock()

    def grab(self, monitor, region) -> CaptureResult:
        if self.active > 0 and self.clock() - self._demoted_at >= self.retry_s:
            self.active, self._demoted_at = 0, self.clock()       # give the primary another chance
        for _ in range(len(self.backends)):
            be = self.backends[self.active]
            try:
                res = be.grab(monitor, region)
            except BackendError as e:
                if self.active + 1 >= len(self.backends):
                    self.last_error = str(e)
                    return CaptureResult(None, be.name, error=str(e))
                self._demote(str(e))
                continue
            if res.blank:
                self._blanks += 1
                if self._blanks >= self.blank_limit and self.active + 1 < len(self.backends):
                    self._demote(f'{be.name} returned black frames')
                    continue
            else:
                self._blanks = 0
            return res
        return CaptureResult(None, self.backend_name, error=self.last_error)

    def close(self):
        for b in self.backends:
            b.close()
