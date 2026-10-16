"""
Where the app keeps its data (settings, price/task caches, the item catalog, logs, the extracted
game font) - stdlib only, so the sign-in watcher (lifecycle.py) can use it too.

The rule is that *where the exe sits must not matter*:

* packaged exe: ``%LOCALAPPDATA%\\TarkovStashHelper`` - always writable, survives moving or
  re-downloading the exe, survives self-updates.  An older install that kept ``data\\`` next to the
  exe is moved there once (``migrate``); if that fails the old folder keeps being used.
* run from source: ``<repo>\\data`` (as before), or the LOCALAPPDATA folder if that is read-only.
* ``TSH_DATA_DIR`` overrides everything (tests, portable use).
* a folder that turns out not to be writable is skipped, ending at the system temp folder, so the
  app always starts.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

APP_DIR_NAME = 'TarkovStashHelper'
# files that mean "this folder holds a real install's data"
_MARKERS = ('settings.json', 'prices_cache.json', 'identify_catalog_v2.npz', 'progress.json')


def writable(path: str) -> bool:
    """True if ``path`` can be created and written to."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f'.write-test-{os.getpid()}')
        with open(probe, 'w') as f:
            f.write('x')
        os.remove(probe)
        return True
    except OSError:
        return False


def local_app_dir(env=None) -> str:
    env = os.environ if env is None else env
    base = env.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    return os.path.join(base, APP_DIR_NAME)


def _has_data(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, m)) for m in _MARKERS)


def migrate_dir(src: str, dst: str) -> bool:
    """Move every entry of ``src`` into ``dst`` (created).  True when ``src`` ended up empty.
    A same-drive rename is instant; across drives it copies and then deletes.  Never raises; an
    entry that cannot be moved stays where it was (and the caller keeps using ``src``)."""
    try:
        os.makedirs(dst, exist_ok=True)
        left = 0
        for name in os.listdir(src):
            a, b = os.path.join(src, name), os.path.join(dst, name)
            if os.path.exists(b):
                continue
            try:
                shutil.move(a, b)
            except (OSError, shutil.Error):
                left += 1
        return left == 0
    except OSError:
        return False


def resolve_data_dir(frozen: bool, exe_dir: str, source_dir: str, env=None, migrate: bool = True) -> str:
    """The data folder for this run (see the module docstring).  ``migrate=False`` (the watcher)
    never moves anything: it returns whichever folder currently holds the data."""
    env = os.environ if env is None else env
    forced = (env.get('TSH_DATA_DIR') or '').strip().strip('"')
    if forced:
        os.makedirs(forced, exist_ok=True)
        return os.path.abspath(forced)
    local = local_app_dir(env)
    adjacent = os.path.join(exe_dir if frozen else source_dir, 'data')
    if frozen:
        if os.path.isdir(adjacent) and _has_data(adjacent) and not _has_data(local):
            if migrate and writable(local):
                migrate_dir(adjacent, local)
            if _has_data(adjacent) and not _has_data(local):
                return adjacent if writable(adjacent) else local   # could not move: keep using it
        order = [local, adjacent]
    else:
        order = [adjacent, local]
    for cand in order:
        if writable(cand):
            return cand
    fallback = os.path.join(tempfile.gettempdir(), APP_DIR_NAME)
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _exe_dir() -> str:
    return os.path.dirname(sys.executable)


def default_data_dir(migrate: bool = True) -> str:
    frozen = bool(getattr(sys, 'frozen', False))
    return resolve_data_dir(frozen, _exe_dir() if frozen else '',
                            os.path.dirname(os.path.abspath(__file__)), migrate=migrate)
