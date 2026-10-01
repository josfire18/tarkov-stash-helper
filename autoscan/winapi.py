"""
Find the game: process -> main window -> monitor.  Read-only Win32 queries, nothing else.

Everything that touches Windows lives in :class:`Win32`; the decisions (which of a process's
windows is "the game", which monitor to capture, whether our own window is covering it) are pure
functions over plain dataclasses so they are unit-tested with a fake ``Win32``.

Anti-cheat note: this module only asks the window manager for the same public information a
taskbar or Alt-Tab shows (process image name, window rectangle, minimised flag, monitor).  It never
opens the game process, never sends it messages and never moves, resizes or focuses its window.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
from ctypes import wintypes as wt
from dataclasses import dataclass

GAME_EXE = 'EscapeFromTarkov.exe'
MIN_GAME_W, MIN_GAME_H = 640, 360          # smaller windows are launcher / overlay helper windows


@dataclass(frozen=True)
class WinInfo:
    hwnd: int
    pid: int
    cls: str
    title: str
    visible: bool
    iconic: bool
    rect: tuple            # window rect, virtual-screen px (left, top, right, bottom)
    client: tuple          # client rect, same space


@dataclass(frozen=True)
class MonitorInfo:
    handle: int
    device: str            # '\\\\.\\DISPLAY1'
    rect: tuple            # (left, top, right, bottom) in virtual-screen px
    primary: bool

    @property
    def size(self) -> tuple:
        return self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    pid: int
    title: str
    client: tuple          # client rect in virtual-screen px
    monitor: MonitorInfo
    iconic: bool
    covered_by_us: bool = False

    @property
    def size(self) -> tuple:
        return self.client[2] - self.client[0], self.client[3] - self.client[1]

    def region_on_monitor(self) -> tuple:
        """Client rect relative to its monitor's origin and clipped to it: the region to grab."""
        ml, mt, mr, mb = self.monitor.rect
        l, t, r, b = self.client
        return (max(l, ml) - ml, max(t, mt) - mt, min(r, mr) - ml, min(b, mb) - mt)


# --------------------------------------------------------------------------
# pure decisions
# --------------------------------------------------------------------------

def _area(rc: tuple) -> int:
    return max(0, rc[2] - rc[0]) * max(0, rc[3] - rc[1])


def _intersection(a: tuple, b: tuple) -> int:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def pick_game_window(windows: list[WinInfo], pids: set[int]) -> WinInfo | None:
    """The game's main window among everything the process owns.

    The game process also owns invisible helper windows (IME, NvPresent, ...).  Candidates are
    the process's windows with a usable client area; visible, non-minimised ones win, then the
    Unity window class, then the largest area.  ``None`` when the process has no real window."""
    cands = [w for w in windows
             if w.pid in pids and (w.client[2] - w.client[0]) >= MIN_GAME_W
             and (w.client[3] - w.client[1]) >= MIN_GAME_H]
    if not cands:
        return None
    return max(cands, key=lambda w: (w.visible and not w.iconic, w.cls == 'UnityWndClass',
                                     _area(w.client)))


def pick_monitor(client: tuple, monitors: list[MonitorInfo], hint: MonitorInfo | None = None) -> MonitorInfo | None:
    """The monitor showing most of the window (``hint`` - what Windows says - wins ties)."""
    if not monitors:
        return None
    best = max(monitors, key=lambda m: (_intersection(client, m.rect), m is hint or m == hint))
    return best if _intersection(client, best.rect) > 0 else (hint or monitors[0])


def our_window_covers(foreground: WinInfo | None, our_pid: int, client: tuple, frac: float = 0.05) -> bool:
    """Our own (pywebview) window is foreground and overlaps the game's screen area by more than
    ``frac`` of it: a capture now would contain our UI, so it must not be scanned."""
    if foreground is None or foreground.pid != our_pid or not foreground.visible or foreground.iconic:
        return False
    return _intersection(foreground.rect, client) > frac * max(1, _area(client))


def locate_game(win: 'Win32', exe: str = GAME_EXE, our_pid: int | None = None) -> GameWindow | None:
    """Process -> main window -> monitor, as one :class:`GameWindow` (or ``None``: not running)."""
    pids = set(win.find_pids(exe))
    if not pids:
        return None
    windows = win.top_level_windows()
    w = pick_game_window(windows, pids)
    if w is None:
        return None
    mon = pick_monitor(w.client, win.monitors(), win.monitor_of(w.hwnd))
    if mon is None:
        return None
    fg = next((x for x in windows if x.hwnd == win.foreground()), None)
    covered = our_window_covers(fg, our_pid if our_pid is not None else os.getpid(), w.client)
    return GameWindow(w.hwnd, w.pid, w.title, w.client, mon, w.iconic or not w.visible, covered)


# --------------------------------------------------------------------------
# the real thing (ctypes)
# --------------------------------------------------------------------------

class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', wt.LONG), ('dwFlags', wt.DWORD), ('szExeFile', wt.WCHAR * 260)]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [('cbSize', wt.DWORD), ('rcMonitor', wt.RECT), ('rcWork', wt.RECT),
                ('dwFlags', wt.DWORD), ('szDevice', wt.WCHAR * 32)]


_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)      # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2


class Win32:
    """Thin read-only wrapper over user32 / kernel32 (Windows only)."""

    def __init__(self):
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        u, k = self.user32, self.kernel32
        k.CreateToolhelp32Snapshot.restype = wt.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        k.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
        k.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
        k.CloseHandle.argtypes = [wt.HANDLE]
        u.MonitorFromWindow.restype = wt.HMONITOR
        u.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
        u.GetMonitorInfoW.argtypes = [wt.HMONITOR, ctypes.POINTER(_MONITORINFOEXW)]
        u.GetForegroundWindow.restype = wt.HWND
        u.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u.IsWindowVisible.argtypes = [wt.HWND]
        u.IsIconic.argtypes = [wt.HWND]
        u.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
        u.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
        u.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
        u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        self._enum_proc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        self._mon_proc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM)

    @contextlib.contextmanager
    def physical_pixels(self):
        """Make this thread per-monitor DPI aware so every rectangle below is in physical
        pixels (what the capture API uses) even when the process itself is DPI-virtualised."""
        try:
            fn = self.user32.SetThreadDpiAwarenessContext
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
            old = fn(_PER_MONITOR_AWARE_V2)
        except (AttributeError, OSError):          # pre-1703 Windows 10
            old = None
        try:
            yield
        finally:
            if old:
                self.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(old))

    # -- processes -----------------------------------------------------------------------
    def find_pids(self, exe: str) -> list[int]:
        k = self.kernel32
        snap = k.CreateToolhelp32Snapshot(0x2, 0)             # TH32CS_SNAPPROCESS
        if not snap or snap == wt.HANDLE(-1).value:
            return []
        pe = _PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(pe)
        out = []
        try:
            ok = k.Process32FirstW(snap, ctypes.byref(pe))
            while ok:
                if pe.szExeFile.lower() == exe.lower():
                    out.append(int(pe.th32ProcessID))
                ok = k.Process32NextW(snap, ctypes.byref(pe))
        finally:
            k.CloseHandle(snap)
        return out

    # -- windows -------------------------------------------------------------------------
    def _info(self, hwnd: int) -> WinInfo:
        u = self.user32
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        buf = ctypes.create_unicode_buffer(256)
        u.GetClassNameW(hwnd, buf, 256)
        cls = buf.value
        u.GetWindowTextW(hwnd, buf, 256)
        title = buf.value
        r, c = wt.RECT(), wt.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(r))
        u.GetClientRect(hwnd, ctypes.byref(c))
        origin = wt.POINT(0, 0)
        u.ClientToScreen(hwnd, ctypes.byref(origin))
        client = (origin.x, origin.y, origin.x + c.right, origin.y + c.bottom)
        return WinInfo(int(hwnd), int(pid.value), cls, title, bool(u.IsWindowVisible(hwnd)),
                       bool(u.IsIconic(hwnd)), (r.left, r.top, r.right, r.bottom), client)

    def top_level_windows(self) -> list[WinInfo]:
        found: list[int] = []

        def cb(hwnd, _):
            found.append(int(hwnd) if hwnd else 0)
            return True
        self.user32.EnumWindows(self._enum_proc(cb), 0)
        return [self._info(h) for h in found if h]

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    # -- monitors ------------------------------------------------------------------------
    def _monitor(self, hmon) -> MonitorInfo:
        mi = _MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(mi)
        self.user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        rc = mi.rcMonitor
        return MonitorInfo(int(hmon or 0), mi.szDevice, (rc.left, rc.top, rc.right, rc.bottom),
                           bool(mi.dwFlags & 1))

    def monitors(self) -> list[MonitorInfo]:
        out: list[MonitorInfo] = []

        def cb(hmon, hdc, rc, _):
            out.append(self._monitor(hmon))
            return True
        self.user32.EnumDisplayMonitors(None, None, self._mon_proc(cb), 0)
        return out

    def monitor_of(self, hwnd: int) -> MonitorInfo | None:
        hmon = self.user32.MonitorFromWindow(hwnd, 2)         # MONITOR_DEFAULTTONEAREST
        return self._monitor(hmon) if hmon else None
