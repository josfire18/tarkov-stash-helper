"""Generates the app/tray icon in-memory so no binary asset needs to be
hand-authored or committed as an opaque blob. Run this file directly to
also bake out assets/icon.ico for the PyInstaller build.
"""
from __future__ import annotations

from PIL import Image, ImageDraw


def _draw(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(1, size // 16)
    bg = (30, 29, 28, 255)      # matches EFT_BG_TINTS['grey']
    accent = (196, 149, 60, 255)  # amber, Tarkov-ish rarity gold
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size // 6, fill=bg)
    inset = size // 4
    d.rectangle([inset, inset, size - inset, size - inset], outline=accent, width=max(1, size // 16))
    return img


def load_tray_image(size=64):
    return _draw(size)


APP_ID = 'TarkovStashHelper.App'
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def set_app_id(app_id: str = APP_ID) -> bool:
    """Give the process its own Windows AppUserModelID *before any window exists*, so the taskbar
    groups / labels the window as this app instead of as python.exe (whose icon it would borrow)."""
    try:
        import ctypes
        shell = ctypes.windll.shell32
        if shell.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id)) != 0:
            return False
        return current_app_id() == app_id
    except Exception:                      # not Windows / old shell: harmless
        return False


def current_app_id() -> str | None:
    """The process's explicit AppUserModelID (None when unset / not Windows)."""
    try:
        import ctypes
        p = ctypes.c_void_p()
        if ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(p)) != 0 or not p.value:
            return None
        try:
            return ctypes.wstring_at(p.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(p)
    except Exception:
        return None


def apply_window_icon(hwnd: int, ico_path: str) -> bool:
    """WM_SETICON with the right-sized images (big = the taskbar / Alt-Tab size at the current DPI,
    small = the caption).  WinForms' ``Form.Icon`` alone yields one 16 px image, blurry on the taskbar."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.LoadImageW.restype = ctypes.c_void_p
        u.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
        ok = False
        for which, metric in ((1, 11), (0, 49)):            # ICON_BIG / SM_CXICON, ICON_SMALL / SM_CXSMICON
            px = u.GetSystemMetrics(metric)
            h = u.LoadImageW(None, ico_path, 1, px, px, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
            if h:
                u.SendMessageW(hwnd, 0x80, which, h)             # WM_SETICON
                ok = True
        return ok
    except Exception as e:
        print(f'[icon] WM_SETICON failed: {e}')
        return False


def write_ico(path: str) -> None:
    """Multi-size .ico.  PIL embeds only the sizes that fit its *base* image, so save the biggest."""
    _draw(max(ICO_SIZES)).save(path, format='ICO', sizes=[(s, s) for s in ICO_SIZES])


def ensure_ico(path: str) -> str | None:
    """Write the multi-size window icon to ``path`` if it is not there yet (the packaged exe ships
    no loose .ico; data/ is writable).  Returns the path, or None when it cannot be written."""
    import os
    try:
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            write_ico(path)
        return path
    except Exception as e:
        print(f'[icon] could not write {path}: {e}')
        return None


if __name__ == '__main__':
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'icon.ico')
    write_ico(out_path)
    print(f"Wrote {out_path}")
