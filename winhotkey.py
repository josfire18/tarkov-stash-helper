"""
The scan hotkey through Win32 ``RegisterHotKey``.

pynput listens through a low-level keyboard hook (WH_KEYBOARD_LL).  Windows does not deliver
hooked keys to a normal process while a window running as administrator has the focus (UIPI), and
Tarkov with BattlEye often is that window - so the old hotkey did nothing in game.  A hotkey
registered with ``RegisterHotKey`` is matched by Windows itself and posted to this thread as
WM_HOTKEY whatever window has the focus.  It also never sees any other key: nothing is hooked.

The combo text is the one the Settings page stores (pynput syntax): ``<ctrl>+<f1>``,
``<ctrl>+<shift>+s``.  :func:`parse` is pure so it is tested anywhere; :class:`Win32Hotkey`
needs Windows.
"""
from __future__ import annotations

import ctypes
import string
import threading

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
HOTKEY_ID = 1

MODIFIERS = {'ctrl': MOD_CONTROL, 'ctrl_l': MOD_CONTROL, 'ctrl_r': MOD_CONTROL,
             'alt': MOD_ALT, 'alt_l': MOD_ALT, 'alt_r': MOD_ALT, 'alt_gr': MOD_ALT | MOD_CONTROL,
             'shift': MOD_SHIFT, 'shift_l': MOD_SHIFT, 'shift_r': MOD_SHIFT,
             'cmd': MOD_WIN, 'cmd_l': MOD_WIN, 'cmd_r': MOD_WIN, 'win': MOD_WIN}

NAMED_KEYS = {'space': 0x20, 'esc': 0x1B, 'enter': 0x0D, 'tab': 0x09, 'backspace': 0x08,
              'insert': 0x2D, 'delete': 0x2E, 'home': 0x24, 'end': 0x23,
              'page_up': 0x21, 'page_down': 0x22, 'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
              'pause': 0x13, 'print_screen': 0x2C, 'scroll_lock': 0x91}
NAMED_KEYS.update({f'f{n}': 0x6F + n for n in range(1, 25)})          # F1 = 0x70 ... F24 = 0x87

# US layout punctuation (VK_OEM_*); other characters go through VkKeyScanW on Windows
PUNCT = {';': 0xBA, '=': 0xBB, ',': 0xBC, '-': 0xBD, '.': 0xBE, '/': 0xBF, '`': 0xC0,
         '[': 0xDB, '\\': 0xDC, ']': 0xDD, "'": 0xDE}


def _char_vk(ch: str) -> int | None:
    if ch in string.ascii_lowercase or ch in string.digits:
        return ord(ch.upper())
    if ch in PUNCT:
        return PUNCT[ch]
    try:
        vk = ctypes.windll.user32.VkKeyScanW(ord(ch))
        vk = int(vk) & 0xFF
        return vk if 0 < vk < 0xFF else None
    except Exception:
        return None


def parse(combo: str) -> tuple[int, int]:
    """``(modifiers, virtual key)`` for a combo such as ``<ctrl>+<f1>``.  Raises ValueError."""
    if not isinstance(combo, str) or not combo.strip():
        raise ValueError('empty key combination')
    mods, vk = 0, None
    for raw in combo.strip().lower().split('+'):
        part = raw.strip()
        if not part:
            raise ValueError(f'"{combo}" has an empty part')
        name = part[1:-1] if part.startswith('<') and part.endswith('>') and len(part) > 2 else None
        if name in MODIFIERS:
            mods |= MODIFIERS[name]
            continue
        if vk is not None:
            raise ValueError(f'"{combo}" has more than one main key')
        if name is not None:
            if name not in NAMED_KEYS:
                raise ValueError(f'unknown key <{name}>')
            vk = NAMED_KEYS[name]
        elif len(part) == 1:
            vk = _char_vk(part)
            if vk is None:
                raise ValueError(f'unknown key "{part}"')
        else:
            raise ValueError(f'unknown key "{part}"')
    if vk is None:
        raise ValueError(f'"{combo}" has no main key, only Ctrl/Alt/Shift')
    return mods, vk


def pretty(combo: str) -> str:
    """``<ctrl>+<f1>`` -> ``Ctrl+F1`` for messages."""
    out = []
    for p in (combo or '').split('+'):
        p = p.strip().strip('<>')
        out.append(p.upper() if len(p) <= 3 and (len(p) == 1 or p[0] == 'f') else p.capitalize())
    return '+'.join(out)


class HotkeyTaken(ValueError):
    """Another program already registered this combo."""


class Win32Hotkey:
    """One registered hotkey on its own message-loop thread.  ``on_fire`` runs on that thread,
    so it must return quickly (start a thread for real work)."""

    def __init__(self, combo: str, on_fire):
        self.combo = combo
        self.mods, self.vk = parse(combo)
        self.on_fire = on_fire
        self._thread = None
        self._tid = None
        self._ready = threading.Event()
        self._error = None

    def start(self, timeout: float = 3.0):
        """Register and start listening; raises HotkeyTaken / OSError when Windows refuses."""
        self._thread = threading.Thread(target=self._run, name='scan-hotkey', daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise OSError('the hotkey thread did not start')
        if self._error is not None:
            raise self._error

    def _run(self):
        from ctypes import wintypes as wt
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        kernel32 = ctypes.WinDLL('kernel32')
        user32.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
        user32.RegisterHotKey.restype = wt.BOOL
        user32.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        self._tid = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, HOTKEY_ID, self.mods | MOD_NOREPEAT, self.vk):
            err = ctypes.get_last_error()
            if err == ERROR_HOTKEY_ALREADY_REGISTERED:
                self._error = HotkeyTaken(f'{pretty(self.combo)} is already used by another program')
            else:
                self._error = OSError(f'Windows refused the hotkey (error {err})')
            self._ready.set()
            return
        self._ready.set()
        msg = wt.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:     # 0 = WM_QUIT, -1 = error
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    try:
                        self.on_fire()
                    except Exception as e:
                        print(f'[hotkey] handler failed: {e}')
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    def stop(self):
        if self._tid is not None and self._thread is not None and self._thread.is_alive():
            ctypes.WinDLL('user32').PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            self._thread.join(timeout=2)
