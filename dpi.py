"""
Per-monitor DPI awareness for the whole process (stdlib + ctypes only).

On a 125 % / 150 % desktop a DPI-unaware process is "virtualised": Windows lies about the screen size and
screen capture (mss / DXGI) then returns a rescaled or cropped picture, so the stash grid would be
measured on the wrong pixels.  With multiple monitors of different scale (game on the 4K screen at 150 %,
the main screen at 100 %) even *system* awareness - which is all pywebview asks for - gives wrong
coordinates for the secondary screen.  So the process declares itself PER_MONITOR_AWARE_V2 (Windows 10
1703+), falling back to per-monitor v1 (8.1) and then to system aware, before any window or capture exists.
Call :func:`ensure` first thing in the process; it is idempotent and never raises.
"""
from __future__ import annotations

import ctypes
import sys

PER_MONITOR_AWARE_V2 = -4          # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
PER_MONITOR_AWARE = 2              # PROCESS_PER_MONITOR_DPI_AWARE (shcore)


def ensure(user32=None, shcore=None) -> str:
    """Make the process per-monitor DPI aware.  Returns which level was reached:
    ``'per-monitor-v2'``, ``'per-monitor'``, ``'system'``, ``'already-set'`` (some code set it
    first) or ``'unavailable'`` (not Windows / every call failed)."""
    if user32 is None:
        if sys.platform != 'win32':
            return 'unavailable'
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        try:
            shcore = ctypes.WinDLL('shcore', use_last_error=True)
        except OSError:
            shcore = None
    try:
        fn = user32.SetProcessDpiAwarenessContext
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_bool
        if fn(ctypes.c_void_p(PER_MONITOR_AWARE_V2)):
            return 'per-monitor-v2'
        if ctypes.get_last_error() == 5:               # ERROR_ACCESS_DENIED: awareness is already set
            return 'already-set'
    except (AttributeError, OSError):
        pass
    try:
        if shcore is not None:
            hr = shcore.SetProcessDpiAwareness(PER_MONITOR_AWARE)
            if hr == 0:
                return 'per-monitor'
            if hr & 0xFFFFFFFF == 0x80070005:          # E_ACCESSDENIED
                return 'already-set'
    except (AttributeError, OSError):
        pass
    try:
        if user32.SetProcessDPIAware():
            return 'system'
    except (AttributeError, OSError):
        pass
    return 'unavailable'


def current() -> str:
    """The awareness the process has now: ``'per-monitor-v2'`` / ``'per-monitor'`` / ``'system'`` /
    ``'unaware'`` / ``'unknown'``."""
    if sys.platform != 'win32':
        return 'unknown'
    try:
        user32 = ctypes.WinDLL('user32')
        user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        ctx = user32.GetThreadDpiAwarenessContext()
        a = user32.GetAwarenessFromDpiAwarenessContext(ctx)      # 0 unaware, 1 system, 2 per-monitor
        if a == 2:
            user32.AreDpiAwarenessContextsEqual.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            v2 = user32.AreDpiAwarenessContextsEqual(ctx, ctypes.c_void_p(PER_MONITOR_AWARE_V2))
            return 'per-monitor-v2' if v2 else 'per-monitor'
        return 'system' if a == 1 else 'unaware'
    except (AttributeError, OSError):
        return 'unknown'
