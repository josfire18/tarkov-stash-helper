"""
Finding Tesseract OCR, which the app needs to read item names off the screen (stdlib only).

The release exe carries its own copy (``tesseract.zip`` inside the bundle: tesseract.exe, its DLLs and
``tessdata/eng.traineddata``; Apache-2.0, built by UB Mannheim - see scripts/vendor_tesseract.py and
the spec), so nobody has to install anything.  It is unpacked once into ``<data dir>/tesseract``.
Used in this order:

  1. the bundled copy, or ``vendor/tesseract`` in a source checkout
  2. ``tesseract`` on PATH
  3. a system install: the registry key the UB Mannheim installer writes, then the usual folders on
     every drive

``configure`` points pytesseract at the result and, for the bundled copy, sets ``TESSDATA_PREFIX`` so
it finds its own language data.
"""
from __future__ import annotations

import os
import shutil
import sys

EXE = 'tesseract.exe'


def _usable(d) -> bool:
    return (os.path.isfile(os.path.join(d, EXE)) and os.path.isfile(os.path.join(d, 'tessdata', 'eng.traineddata')))


def unpack_bundle(zip_path: str, dest: str) -> str | None:
    """Unpack the bundled ``tesseract.zip`` into ``dest`` (once: a ``.stamp`` file inside the zip
    says which Tesseract it holds and the unpacked folder is only redone when that changes or the
    folder is incomplete).  Returns ``dest``, or None if it could not be unpacked."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as z:
            stamp = z.read('.stamp').decode('utf-8', 'replace').strip()
            try:
                with open(os.path.join(dest, '.stamp'), encoding='utf-8') as f:
                    if f.read().strip() == stamp and _usable(dest):
                        return dest
            except OSError:
                pass
            shutil.rmtree(dest, ignore_errors=True)
            tmp = dest + '.part'
            shutil.rmtree(tmp, ignore_errors=True)
            z.extractall(tmp)
        os.replace(tmp, dest)
        return dest if _usable(dest) else None
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def bundled_dir(data_dir=None, meipass=None, source_dir=None) -> str | None:
    """The folder holding the bundled tesseract.exe, or None.  In the packaged exe the copy is a zip
    inside the bundle, unpacked once into ``<data dir>/tesseract``; from a source checkout it is the
    unpacked ``vendor/tesseract`` (scripts/vendor_tesseract.py)."""
    meipass = meipass if meipass is not None else getattr(sys, '_MEIPASS', None)
    if meipass and data_dir:
        z = os.path.join(meipass, 'tesseract.zip')
        if os.path.isfile(z):
            d = unpack_bundle(z, os.path.join(data_dir, 'tesseract'))
            if d:
                return d
    d = os.path.join(source_dir or os.path.dirname(os.path.abspath(__file__)), 'vendor', 'tesseract')
    return d if _usable(d) else None


def _registry_dirs():
    try:
        import winreg
    except ImportError:
        return []
    out = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for sub in (r'SOFTWARE\Tesseract-OCR', r'SOFTWARE\WOW6432Node\Tesseract-OCR'):
            try:
                with winreg.OpenKey(hive, sub) as k:
                    out.append(str(winreg.QueryValueEx(k, 'InstallDir')[0]))
            except OSError:
                continue
    return out


def _common_dirs():
    if sys.platform != 'win32':
        return []
    drives = [f'{c}:\\' for c in 'CDEFGH' if os.path.isdir(f'{c}:\\')]
    return [os.path.join(d, p) for d in drives
            for p in (r'Program Files\Tesseract-OCR', r'Program Files (x86)\Tesseract-OCR', 'Tesseract-OCR')]


def find(data_dir=None, meipass=None, source_dir=None) -> tuple[str | None, str | None]:
    """``(tesseract.exe path, tessdata dir to force or None)``; ``(None, None)`` when not found."""
    b = bundled_dir(data_dir, meipass, source_dir)
    if b:
        return os.path.join(b, EXE), os.path.join(b, 'tessdata')
    on_path = shutil.which('tesseract')
    if on_path:
        return on_path, None
    for d in _registry_dirs() + _common_dirs():
        exe = os.path.join(d, EXE)
        if os.path.isfile(exe):
            return exe, None
    return None, None


def configure(pytesseract_module, environ=None, **kw) -> str | None:
    """Point pytesseract at the Tesseract that ``find`` returns.  Returns its path (None: none)."""
    environ = os.environ if environ is None else environ
    exe, tessdata = find(**kw)
    if exe:
        pytesseract_module.pytesseract.tesseract_cmd = exe
    if tessdata:
        environ['TESSDATA_PREFIX'] = tessdata
    return exe
