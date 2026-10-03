"""
Open and close with Tarkov: the app has no business running while the game is not.

Two processes cooperate (both are this same program, selected by command line):

  watcher   ``app.py --watch``   Registered in the Windows startup list (HKCU ...\\Run).  A dormant
            loop (below-normal priority, ~3 s poll, a few MB) whose only job is to notice
            ``EscapeFromTarkov.exe`` and start the full app with ``--with-tarkov``.  It imports
            nothing heavy: this module is stdlib + ctypes only, and ``app.py`` dispatches to it
            *before* its own numpy / cv2 / flask imports.
  app       the normal process.  When setting ``follow_tarkov`` is on and it was started with the
            game (``--with-tarkov``) or has seen the game during this session, it polls for the
            game and exits cleanly once the game has been gone for FOLLOW_GONE_S.  A copy started
            by hand while the game is not running never closes until it has seen the game and the
            game then exits.

Settings this module owns (data/settings.json, applied at once by ``apply_settings``):

  start_with_windows          True   register + run the watcher (False: unregister, stop it)
  follow_tarkov               True   open with the game / close when it exits (False: the watcher
                                     starts the app once when you sign in and it stays until quit)
  show_window_on_game_start   True   False: the app starts to the tray with no window

Single instance everywhere is by named kernel mutex (``Local\\TarkovStashHelperApp``,
``Local\\TarkovStashHelperWatch``); the watcher also waits on a named event so the app can stop it
instantly (``stop_watcher``: before a self-update replaces the exe, or when auto-launch is turned off).
When TSH_PORT is set the object names get a ``.<port>`` suffix, so a smoke test never collides with
a real running instance.  TSH_RUN_KEY (a registry sub-key path) redirects the startup registration
for tests.

Anti-cheat stance (README): the only thing done to the game is listing process names, the same
read-only snapshot Task Manager takes.  It is never opened, moved, resized or focused.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from ctypes import wintypes as wt

try:                                    # Windows only; tests replace it with a fake
    import winreg
except ImportError:                     # pragma: no cover
    winreg = None

GAME_EXE = 'EscapeFromTarkov.exe'
DEFAULT_PORT = 8877
RUN_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'
RUN_VALUE = 'TarkovStashHelper'

WATCH_POLL_S = 3.0         # watcher: how often it looks for the game
FOLLOW_POLL_S = 5.0        # app: how often it looks for the game
FOLLOW_GONE_S = 15.0       # app: the game must be gone this long before the app closes
TRIM_EVERY_TICKS = 100     # watcher: hand idle memory back to Windows every ~5 minutes

DEFAULTS = {'start_with_windows': True, 'follow_tarkov': True, 'show_window_on_game_start': True}

FROZEN = getattr(sys, 'frozen', False)
BASE = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))

# modules the watcher must never import (they are what makes the app 100+ MB); checked by --once
HEAVY_MODULES = ('numpy', 'cv2', 'flask', 'PIL', 'requests', 'mss', 'pytesseract', 'rapidfuzz',
                 'pynput', 'pystray', 'webview', 'waitress', 'identify', 'autoscan',
                 'sellcalc', 'tarkovdata')


# ---------------------------------------------------------------------------
# Names, paths, settings (stdlib only)
# ---------------------------------------------------------------------------

def _port() -> int:
    try:
        return int(os.environ.get('TSH_PORT', DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT


def _object_name(base: str) -> str:
    port = _port()
    return 'Local\\' + base + ('' if port == DEFAULT_PORT else f'.{port}')


def app_mutex_name() -> str:
    return _object_name('TarkovStashHelperApp')


def watch_mutex_name() -> str:
    return _object_name('TarkovStashHelperWatch')


def watch_stop_name() -> str:
    return _object_name('TarkovStashHelperWatchStop')


def run_key() -> str:
    return os.environ.get('TSH_RUN_KEY') or RUN_KEY


def data_dir() -> str:
    return os.path.join(BASE, 'data')


def settings_path() -> str:
    return os.path.join(data_dir(), 'settings.json')


def with_defaults(settings) -> dict:
    """``settings`` with the three lifecycle keys filled in (existing installs have none of them)."""
    out = dict(settings or {})
    for k, v in DEFAULTS.items():
        if out.get(k) is None:
            out[k] = v
    return out


def game_exe(settings) -> str:
    """The game's process name; ``auto_scan_exe`` (README, Auto-scan) overrides it."""
    return str((settings or {}).get('auto_scan_exe') or GAME_EXE)


class SettingsFile:
    """data/settings.json read cheaply: re-parsed only when the file changes; a missing or
    half-written file keeps the last good copy (defaults at first)."""

    def __init__(self, path: str | None = None):
        self.path = path
        self._sig = None
        self._data: dict = {}

    def __call__(self) -> dict:
        path = self.path or settings_path()
        try:
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
            if sig != self._sig:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._data, self._sig = data, sig
        except (OSError, ValueError):
            pass
        return with_defaults(self._data)


def log(msg: str) -> None:
    """Append to data/lifecycle.log (kept small).  Never raises: a read-only folder must not stop
    the watcher.  Events only - nothing is written per poll."""
    try:
        os.makedirs(data_dir(), exist_ok=True)
        p = os.path.join(data_dir(), 'lifecycle.log')
        if os.path.exists(p) and os.path.getsize(p) > 200_000:
            with open(p, 'rb') as f:
                f.seek(-50_000, os.SEEK_END)
                tail = f.read()
            with open(p, 'wb') as f:
                f.write(tail)
        with open(p, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {msg}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Win32 (ctypes, kernel32 only unless a window is touched)
# ---------------------------------------------------------------------------

class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', wt.LONG), ('dwFlags', wt.DWORD), ('szExeFile', wt.WCHAR * 260)]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t)]


_ERROR_ALREADY_EXISTS = 183
_SYNCHRONIZE = 0x00100000
_EVENT_MODIFY_STATE = 0x0002
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_WS_EX_NOACTIVATE = 0x08000000
_GWL_EXSTYLE = -20


class Win32:
    """The handful of kernel32 calls the watcher needs.  Every named object created here stays
    open for the life of the process (that is what makes a named mutex a single-instance lock)."""

    def __init__(self):
        k = self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateToolhelp32Snapshot.restype = wt.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        k.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
        k.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
        k.CloseHandle.argtypes = [wt.HANDLE]
        k.CreateMutexW.restype = wt.HANDLE
        k.CreateMutexW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.LPCWSTR]
        k.OpenMutexW.restype = wt.HANDLE
        k.OpenMutexW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
        k.CreateEventW.restype = wt.HANDLE
        k.CreateEventW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR]
        k.OpenEventW.restype = wt.HANDLE
        k.OpenEventW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
        k.SetEvent.argtypes = [wt.HANDLE]
        k.WaitForSingleObject.restype = wt.DWORD
        k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        k.GetCurrentProcess.restype = wt.HANDLE
        k.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
        k.K32EmptyWorkingSet.argtypes = [wt.HANDLE]
        k.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), wt.DWORD]
        self._events: dict[str, int] = {}
        self._u = None

    # -- processes -----------------------------------------------------------------------
    def find_pids(self, exe: str) -> list[int]:
        """PIDs whose image name is ``exe`` (read-only process snapshot).  Raises OSError when the
        snapshot itself fails, so a hiccup is not mistaken for "the game exited"."""
        k = self.k
        snap = k.CreateToolhelp32Snapshot(0x2, 0)               # TH32CS_SNAPPROCESS
        if not snap or snap == wt.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), 'CreateToolhelp32Snapshot failed')
        pe = _PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(pe)
        want = exe.lower()
        out: list[int] = []
        try:
            ok = k.Process32FirstW(snap, ctypes.byref(pe))
            while ok:
                if pe.szExeFile.lower() == want:
                    out.append(int(pe.th32ProcessID))
                ok = k.Process32NextW(snap, ctypes.byref(pe))
        finally:
            k.CloseHandle(snap)
        return out

    # -- named kernel objects ------------------------------------------------------------
    def create_mutex(self, name: str) -> bool:
        """Create (and keep) the named mutex.  True when it ALREADY existed, i.e. another
        process holds the lock - then this process must not run."""
        h = self.k.CreateMutexW(None, False, name)
        existed = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
        if not h:
            raise OSError(ctypes.get_last_error(), f'CreateMutex {name} failed')
        if existed:
            self.k.CloseHandle(h)
        else:
            self._events[name] = h
        return existed

    def mutex_exists(self, name: str) -> bool:
        h = self.k.OpenMutexW(_SYNCHRONIZE, False, name)
        if h:
            self.k.CloseHandle(h)
            return True
        return False

    def create_event(self, name: str):
        h = self.k.CreateEventW(None, True, False, name)        # manual-reset, unsignalled
        if not h:
            raise OSError(ctypes.get_last_error(), f'CreateEvent {name} failed')
        self._events[name] = h
        return h

    def signal_event(self, name: str) -> bool:
        h = self.k.OpenEventW(_EVENT_MODIFY_STATE, False, name)
        if not h:
            return False
        try:
            return bool(self.k.SetEvent(h))
        finally:
            self.k.CloseHandle(h)

    def wait(self, handle, timeout_ms: int) -> bool:
        """Sleep up to ``timeout_ms`` (no CPU); True when the event was signalled."""
        return self.k.WaitForSingleObject(handle, int(timeout_ms)) == 0

    # -- this process --------------------------------------------------------------------
    def lower_priority(self) -> None:
        self.k.SetPriorityClass(self.k.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS)

    def trim_memory(self) -> None:
        """Ask Windows to take back this idle process's working set (pages fault back in on use)."""
        self.k.K32EmptyWorkingSet(self.k.GetCurrentProcess())

    def memory(self) -> dict:
        pc = _PROCESS_MEMORY_COUNTERS_EX()
        pc.cb = ctypes.sizeof(pc)
        if not self.k.K32GetProcessMemoryInfo(self.k.GetCurrentProcess(), ctypes.byref(pc), pc.cb):
            return {}
        mb = 1 << 20
        return {'working_set_mb': round(pc.WorkingSetSize / mb, 1),
                'peak_working_set_mb': round(pc.PeakWorkingSetSize / mb, 1),
                'private_mb': round(pc.PrivateUsage / mb, 1)}

    # -- windows (only the app's own, for allow_activation) -------------------------------
    @property
    def user32(self):
        if self._u is None:
            u = self._u = ctypes.WinDLL('user32', use_last_error=True)
            u.GetWindowLongW.restype = wt.LONG
            u.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
            u.SetWindowLongW.restype = wt.LONG
            u.SetWindowLongW.argtypes = [wt.HWND, ctypes.c_int, wt.LONG]
            u.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
            u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
            self._enum_proc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        return self._u

    def windows_of(self, pid: int, title: str) -> list[int]:
        """Top-level windows of process ``pid`` titled ``title``."""
        u = self.user32
        found: list[int] = []

        def cb(hwnd, _):
            owner = wt.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid:
                buf = ctypes.create_unicode_buffer(256)
                u.GetWindowTextW(hwnd, buf, 256)
                if buf.value == title:
                    found.append(int(hwnd))
            return True
        u.EnumWindows(self._enum_proc(cb), 0)
        return found

    def set_no_activate(self, hwnd: int, on: bool) -> None:
        u = self.user32
        ex = u.GetWindowLongW(hwnd, _GWL_EXSTYLE) & 0xFFFFFFFF
        ex = (ex | _WS_EX_NOACTIVATE) if on else (ex & ~_WS_EX_NOACTIVATE)
        u.SetWindowLongW(hwnd, _GWL_EXSTYLE, ctypes.c_int32(ex & 0xFFFFFFFF).value)


_win_singleton: Win32 | None = None


def get_win() -> Win32:
    global _win_singleton
    if _win_singleton is None:
        _win_singleton = Win32()
    return _win_singleton


def game_pids(settings=None, win=None) -> list[int]:
    return (win or get_win()).find_pids(game_exe(settings))


def game_running(settings=None, win=None) -> bool:
    try:
        return bool(game_pids(settings, win))
    except OSError:
        return False


def app_running(win=None) -> bool:
    return (win or get_win()).mutex_exists(app_mutex_name())


def watcher_running(win=None) -> bool:
    return (win or get_win()).mutex_exists(watch_mutex_name())


# ---------------------------------------------------------------------------
# Command lines + Windows startup registration (HKCU Run)
# ---------------------------------------------------------------------------

def _gui_python() -> str:
    """pythonw.exe next to the running interpreter when there is one (no console window)."""
    exe = sys.executable
    d, name = os.path.split(exe)
    if name.lower() == 'python.exe':
        w = os.path.join(d, 'pythonw.exe')
        if os.path.exists(w):
            return w
    return exe


def command_args(*extra: str) -> list[str]:
    """argv that starts this program again: the exe itself when packaged, else pythonw + app.py."""
    if FROZEN:
        return [sys.executable, *extra]
    return [_gui_python(), os.path.join(BASE, 'app.py'), *extra]


def run_command() -> str:
    """The Run-key value: the watcher's command line, every path quoted."""
    return ' '.join(f'"{a}"' if (' ' in a or not a.startswith('-')) else a for a in command_args('--watch'))


def registered_command() -> str | None:
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key(), 0, winreg.KEY_READ) as k:
            value, _kind = winreg.QueryValueEx(k, RUN_VALUE)
        return str(value)
    except OSError:
        return None


def is_registered() -> bool:
    return registered_command() is not None


def register() -> bool:
    """Make Windows start the watcher at sign-in.  Idempotent, and re-points the value when the
    exe moved (called on every app launch).  Returns True when something was written."""
    if winreg is None:
        return False
    cmd = run_command()
    if registered_command() == cmd:
        return False
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, run_key(), 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, RUN_VALUE, 0, winreg.REG_SZ, cmd)
    log(f'registered startup entry: {cmd}')
    return True


def unregister() -> bool:
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key(), 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, RUN_VALUE)
    except OSError:
        return False
    log('removed startup entry')
    return True


# ---------------------------------------------------------------------------
# Starting processes
# ---------------------------------------------------------------------------

def _child_env() -> dict:
    """Environment for a new, independent copy of this exe.  A onefile PyInstaller child would
    otherwise inherit the parent's extraction-dir variables and try to reuse a temp folder that
    disappears when the parent exits."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('_PYI_') and k != '_MEIPASS2'}
    env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
    return env


def spawn(*extra: str):
    """Start this program detached (own process group, no console, nothing inherited)."""
    import subprocess
    flags = (getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
             | getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return subprocess.Popen(command_args(*extra), creationflags=flags, close_fds=True, cwd=BASE,
                            env=_child_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def spawn_watcher() -> None:
    spawn('--watch')
    log('started the watcher')


def stop_watcher(win=None, wait_s: float = 2.0) -> bool:
    """Tell a running watcher to exit and wait (briefly) until it has, so its exe is unlocked.
    True when no watcher is left."""
    win = win or get_win()
    if not win.mutex_exists(watch_mutex_name()):
        return True
    win.signal_event(watch_stop_name())
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if not win.mutex_exists(watch_mutex_name()):
            return True
        time.sleep(0.05)
    return not win.mutex_exists(watch_mutex_name())


# ---------------------------------------------------------------------------
# The watcher  (python app.py --watch)
# ---------------------------------------------------------------------------

class Watcher:
    """One poll of the watcher.  Edge triggered per game PID: the app is started when a game
    process we have not handled yet appears, and never again for that same process - so a user
    who quits the app mid-session is not fought with."""

    def __init__(self, win, read_settings, start_app, log_fn=log):
        self.win = win
        self.read_settings = read_settings
        self.start_app = start_app          # start_app(*flags) -> starts the full app
        self.log = log_fn
        self.handled: set[int] = set()
        self.login_done = False

    def tick(self) -> str | None:
        """Returns 'stop' when the watcher should exit, else None."""
        s = self.read_settings()
        if not s['start_with_windows']:
            self.log('start_with_windows is off - watcher exits')
            return 'stop'
        if s['follow_tarkov']:
            try:
                pids = set(game_pids(s, self.win))
            except OSError:
                return None                                         # snapshot hiccup: look again next tick
            self.handled &= pids                                    # forget processes that exited
            new = pids - self.handled
            if not new:
                return None
            if self.win.mutex_exists(app_mutex_name()):
                self.log(f'game {sorted(new)} is running; the app is already open')
            else:
                self.log(f'game {sorted(new)} started - opening the app')
                self.start_app('--with-tarkov')
            self.handled |= new                                     # (a failed start raised above: retried)
        elif not self.login_done:
            # Not following the game: "start with Windows" means "start the app at sign-in".
            self.login_done = True
            if not self.win.mutex_exists(app_mutex_name()):
                self.log('follow_tarkov is off - opening the app at sign-in')
                self.start_app('--background')
        return None


def _heavy_loaded() -> list[str]:
    return [m for m in HEAVY_MODULES if m in sys.modules]


def _watch_once(argv: list[str], win) -> int:
    """``--watch --once [report.json]``: one look, no loop, no mutex, never starts anything.
    The packaged-build smoke test (CI) and a quick way to see the watcher's footprint."""
    s = SettingsFile()()
    try:
        pids = game_pids(s, win)
        err = None
    except OSError as e:
        pids, err = [], str(e)
    heavy = _heavy_loaded()
    win.trim_memory()
    rep = {'ok': err is None and not heavy, 'frozen': FROZEN, 'game_exe': game_exe(s),
           'game_running': bool(pids), 'error': err, 'app_running': app_running(win),
           'watcher_running': watcher_running(win), 'registered': is_registered(),
           'run_command': run_command(), 'heavy_imported': heavy,
           'modules_loaded': len(sys.modules), 'memory': win.memory()}
    paths = [a for a in argv if not a.startswith('-')]
    if paths:
        try:
            with open(paths[0], 'w', encoding='utf-8') as f:
                json.dump(rep, f, indent=2)
        except OSError:
            pass
    if sys.stdout is not None:
        print(json.dumps(rep, indent=2))
    return 0 if rep['ok'] else 1


def watch_main(argv=None, win=None) -> int:
    """Entry point of ``app.py --watch``."""
    argv = list(argv or [])
    win = win or get_win()
    if '--once' in argv:
        return _watch_once(argv, win)
    if win.create_mutex(watch_mutex_name()):
        return 0                                                    # a watcher is already running
    stop = win.create_event(watch_stop_name())
    win.lower_priority()
    win.trim_memory()
    read_settings = SettingsFile()
    w = Watcher(win, read_settings, spawn)
    log(f'watcher started (poll {WATCH_POLL_S:g}s, game {game_exe(read_settings())})')
    ticks = 0
    while True:
        try:
            if w.tick() == 'stop':
                break
        except Exception as e:                                      # noqa: BLE001 - the loop must survive
            log(f'watcher tick failed: {type(e).__name__}: {e}')
        ticks += 1
        if ticks % TRIM_EVERY_TICKS == 0:
            win.trim_memory()
        if win.wait(stop, int(WATCH_POLL_S * 1000)):
            log('watcher told to stop')
            break
    return 0


# ---------------------------------------------------------------------------
# The full app's side
# ---------------------------------------------------------------------------

class Follower:
    """Decides when the app should close because the game is gone.

    ``armed`` = the app follows this game session: follow_tarkov is on AND (it was started with
    the game, or the game has been seen running while follow_tarkov was on).  Only an armed app
    closes, and only after the game has been absent for FOLLOW_GONE_S in a row."""

    def __init__(self, win, read_settings, started_with_tarkov: bool, clock=time.monotonic):
        self.win = win
        self.read_settings = read_settings
        self.clock = clock
        self._initial = bool(started_with_tarkov)
        self.armed = False
        self.gone_since: float | None = None
        self.game_running = False

    def tick(self) -> bool:
        """True when the app should shut down now."""
        s = self.read_settings()
        try:
            running = bool(game_pids(s, self.win))
        except OSError:
            return False                                            # unknown is not "gone"
        now = self.clock()
        self.game_running = running
        if not s['follow_tarkov']:
            self.armed, self.gone_since, self._initial = False, None, False
            return False
        if self._initial:
            self.armed, self._initial = True, False
        if running:
            self.armed, self.gone_since = True, None
            return False
        if self.gone_since is None:
            self.gone_since = now
        return self.armed and now - self.gone_since >= FOLLOW_GONE_S


_session = {'managed': False, 'with_tarkov': False, 'background': False, 'follower': None,
            'settings': None}


def ensure_autostart(settings, win=None) -> bool:
    """start_with_windows on: the startup entry points at this exe and a watcher is running (so
    after one manual launch everything is automatic, no reboot needed).  True when the startup
    entry did not exist before (the app mentions that once)."""
    if not with_defaults(settings)['start_with_windows']:
        return False
    first = not is_registered()
    register()
    if not watcher_running(win):
        spawn_watcher()
    return first


def apply_settings(settings) -> dict | None:
    """Make the lifecycle settings take effect now (called by POST /api/settings).  Inert until
    the app has begun its session, so unit tests that save settings never touch the registry.
    Never raises."""
    if not _session['managed']:
        return None
    try:
        s = with_defaults(settings)
        if s['start_with_windows']:
            ensure_autostart(s)
        else:
            unregister()
            stop_watcher()
    except Exception as e:                                          # noqa: BLE001
        log(f'apply_settings failed: {type(e).__name__}: {e}')
    return status(settings)


def show_running_instance() -> bool:
    """Ask the already-running app to bring its window up (second launch by hand)."""
    import urllib.request
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{_port()}/api/lifecycle/show', method='POST')
        urllib.request.urlopen(req, timeout=3).read()
        return True
    except Exception:                                               # noqa: BLE001
        return False


def parse_flags(argv) -> tuple[bool, bool]:
    """(started by the watcher because the game is running, started to the tray at sign-in)."""
    argv = list(argv or [])
    return '--with-tarkov' in argv, '--background' in argv


def begin_app_session(argv, settings_fn, on_gone, win=None, start_thread: bool = True) -> dict | None:
    """Called once by the full app before it opens anything.  Takes the single-instance lock (None
    when another copy is already running: that one is told to show itself and this one must exit),
    makes sure the startup entry + watcher exist, and starts watching for the game to exit."""
    win = win or get_win()
    with_tarkov, background = parse_flags(argv)
    if win.create_mutex(app_mutex_name()):
        log('another copy is already running')
        if not (with_tarkov or background):
            show_running_instance()
        return None
    s = with_defaults(settings_fn())
    _session.update(managed=True, with_tarkov=with_tarkov, background=background, settings=settings_fn)
    first = False
    try:
        first = ensure_autostart(s, win)
    except Exception as e:                                          # noqa: BLE001
        log(f'ensure_autostart failed: {type(e).__name__}: {e}')
    follower = Follower(win, lambda: with_defaults(settings_fn()), with_tarkov)
    _session['follower'] = follower
    if start_thread:
        import threading

        def loop():
            while True:
                try:
                    if follower.tick():
                        log('Tarkov has exited - closing the app')
                        on_gone()
                        return
                except Exception as e:                              # noqa: BLE001
                    log(f'follower failed: {type(e).__name__}: {e}')
                time.sleep(FOLLOW_POLL_S)
        threading.Thread(target=loop, daemon=True, name='tarkov-follower').start()
    hidden = background or (with_tarkov and not s['show_window_on_game_start'])
    return {'with_tarkov': with_tarkov, 'background': background, 'hidden': hidden,
            'quiet': with_tarkov or background, 'first_registration': first}


def status(settings=None, win=None) -> dict:
    s = with_defaults(settings)
    win = win or get_win()
    follower = _session.get('follower')
    return {'registered': is_registered(),
            'watcher_running': watcher_running(win),
            'game_running': game_running(s, win),
            'follow_tarkov': bool(s['follow_tarkov']),
            'started_with_tarkov': bool(_session['with_tarkov']),
            'following': bool(follower and follower.armed)}


def make_blueprint(settings_fn, show_fn=None):
    """GET /api/lifecycle/status, POST /api/lifecycle/show (the second-launch hand-over)."""
    from flask import Blueprint, jsonify
    bp = Blueprint('lifecycle', __name__)

    @bp.route('/api/lifecycle/status', methods=['GET'])
    def lifecycle_status():
        return jsonify(status(settings_fn()))

    @bp.route('/api/lifecycle/show', methods=['POST'])
    def lifecycle_show():
        shown = False
        if show_fn is not None:
            try:
                show_fn()
                shown = True
            except Exception as e:                                  # noqa: BLE001
                log(f'show failed: {type(e).__name__}: {e}')
        return jsonify({'ok': shown})
    return bp


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

def window_kwargs(accepted, quiet: bool, hidden: bool) -> dict:
    """pywebview create_window options so the window never takes the focus from the game:
    ``focus=False`` (older pywebview: start minimized), ``hidden`` for a tray-only start.
    ``accepted`` = the parameter names the installed create_window understands."""
    kw: dict = {}
    if quiet or hidden:
        if 'focus' in accepted:
            kw['focus'] = False
        elif 'minimized' in accepted:
            kw['minimized'] = True
    if hidden and 'hidden' in accepted:
        kw['hidden'] = True
    return kw


def allow_activation(window, title: str = 'Tarkov Stash Helper', win=None) -> bool:
    """pywebview's ``focus=False`` sets WS_EX_NOACTIVATE for the window's whole life: it could
    never take keyboard focus, even when the user clicks into it.  Once the window is up, lift that
    (the game keeps the focus it has; the user's own click may now activate the window)."""
    if getattr(window, 'focus', True):
        return False
    window.focus = True                      # pywebview re-applies the style on activation unless this is set
    try:
        win = win or get_win()
        for hwnd in win.windows_of(os.getpid(), title):
            win.set_no_activate(hwnd, False)
    except Exception as e:                                          # noqa: BLE001
        log(f'allow_activation: {type(e).__name__}: {e}')
    return True
