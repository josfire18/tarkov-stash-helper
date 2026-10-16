"""
Prepare the Tesseract OCR copy that the release exe carries (``vendor/tesseract/``, gitignored).

    python scripts/vendor_tesseract.py                 # download the pinned UB Mannheim installer, extract it
    python scripts/vendor_tesseract.py --source DIR    # use an installed / extracted Tesseract folder

Nothing is installed: the installer is only unpacked with 7-Zip (``7z`` on PATH or in Program
Files).  From the unpacked folder only what ``tesseract.exe`` needs at run time is kept: the exe,
the DLLs it (transitively) imports - found by reading the PE import tables - ``tessdata/eng.traineddata``,
``tessdata/configs`` and the Apache-2.0 licence.  That is a third of the installer (no training tools,
documentation, Java viewer or the OSD model).  Tesseract is Apache-2.0; the DLLs are the usual MSYS2
runtime libraries shipped by the same installer.

TarkovStashHelper.spec bundles ``vendor/tesseract.zip``; tess.py unpacks it once into the data folder.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

VERSION = '5.4.0.20240606'
URL = ('https://github.com/UB-Mannheim/tesseract/releases/download/'
       f'v{VERSION}/tesseract-ocr-w64-setup-{VERSION}.exe')
SHA256 = 'c885fff6998e0608ba4bb8ab51436e1c6775c2bafc2559a19b423e18678b60c9'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'vendor', 'tesseract')
ZIP = os.path.join(ROOT, 'vendor', 'tesseract.zip')
STAMP = f'tesseract-{VERSION}'
LANGS = ('eng',)


def find_7z() -> str:
    for c in (shutil.which('7z'), r'C:\Program Files\7-Zip\7z.exe', r'C:\Program Files (x86)\7-Zip\7z.exe'):
        if c and os.path.isfile(c):
            return c
    raise SystemExit('7-Zip not found (needed to unpack the Tesseract installer)')


def download(dest: str) -> None:
    print(f'downloading {URL}')
    with urllib.request.urlopen(URL, timeout=120) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)
    got = hashlib.sha256(open(dest, 'rb').read()).hexdigest()
    if got != SHA256:
        raise SystemExit(f'installer hash mismatch: {got}')


def dll_closure(folder: str, exe: str = 'tesseract.exe') -> list[str]:
    """Names of the DLLs in ``folder`` that ``exe`` needs, directly or through each other."""
    import pefile
    have = {f.lower(): f for f in os.listdir(folder) if f.lower().endswith('.dll')}
    todo, seen = [exe], set()
    while todo:
        name = todo.pop()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        pe = pefile.PE(os.path.join(folder, name), fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
        for imp in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
            dep = imp.dll.decode().lower()
            if dep in have and dep not in seen:                # system DLLs are not in the folder
                todo.append(have[dep])
        pe.close()
    return sorted(have[n] for n in seen if n in have)


def build(src: str, out: str = OUT) -> str:
    if not os.path.isfile(os.path.join(src, 'tesseract.exe')):
        raise SystemExit(f'no tesseract.exe in {src}')
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'tessdata'))
    shutil.copy2(os.path.join(src, 'tesseract.exe'), out)
    for d in dll_closure(src):
        shutil.copy2(os.path.join(src, d), out)
    for lang in LANGS:
        shutil.copy2(os.path.join(src, 'tessdata', f'{lang}.traineddata'), os.path.join(out, 'tessdata'))
    if os.path.isdir(os.path.join(src, 'tessdata', 'configs')):
        shutil.copytree(os.path.join(src, 'tessdata', 'configs'), os.path.join(out, 'tessdata', 'configs'))
    for lic in ('LICENSE', 'AUTHORS'):
        p = os.path.join(src, 'doc', lic)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, f'TESSERACT-{lic}.txt'))
    with open(os.path.join(out, '.stamp'), 'w') as f:
        f.write(STAMP)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    zip_up(out, ZIP)
    print(f'{out}: {size / 1e6:.1f} MB unpacked, {os.path.getsize(ZIP) / 1e6:.1f} MB zipped ({ZIP})')
    return out


def zip_up(folder: str, dest: str) -> None:
    """One zip of the folder: the exe bundles this single file and tess.py unpacks it once into the
    data folder (a onefile exe would otherwise copy 125 MB of DLLs to %TEMP% on every launch)."""
    import zipfile
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dp, _, fs in os.walk(folder):
            for f in sorted(fs):
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, folder))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', help='an installed or extracted Tesseract folder (skips the download)')
    a = ap.parse_args(argv)
    if a.source:
        build(a.source)
        return 0
    with tempfile.TemporaryDirectory(prefix='tsh_tess_') as tmp:
        setup = os.path.join(tmp, 'setup.exe')
        download(setup)
        ex = os.path.join(tmp, 'x')
        subprocess.run([find_7z(), 'x', '-y', f'-o{ex}', setup, '-bso0', '-bsp0'], check=True)
        build(ex)
    return 0


if __name__ == '__main__':
    sys.exit(main())
