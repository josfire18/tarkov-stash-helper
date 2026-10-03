# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for TarkovStashHelper.exe - the single definition build.bat and
.github/workflows/build.yml both use, so a local build and a release build cannot drift.

    pyinstaller --noconfirm TarkovStashHelper.spec

Flavour (environment variable TSH_FLAVOUR):

  lean  (default, the release flavour)
        No torch / transformers.  Stage 1 runs on the CPU (numpy), stage 2 (DINOv2) is off,
        stage 3 (Tesseract OCR) is unchanged.  ~0.9 point less accurate than the full engine
        and about twice as slow as a GPU stage 1, but a ~74 MB download instead of 237 MB (CPU torch) or 2.9 GB (CUDA).
  dino  (experiment; never released)
        Bundles whatever torch / transformers the *build* Python has installed, plus the
        modules transformers imports by name at run time.  Used to measure what the heavy
        flavours would cost - see the packaging notes in README.md.

Everything that must be *read* at run time and is not a Python module is listed in ``datas``:
templates/ (Flask templates) and identify/assets/ (digit prototypes read via
``os.path.dirname(__file__)`` in identify/digits.py - without it stack counts silently stop
being read).  data/ is NOT bundled: it lives next to the exe (identify/config.py ROOT, app.py BASE).
"""
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

FLAVOUR = os.environ.get('TSH_FLAVOUR', 'lean').strip().lower()
if FLAVOUR not in ('lean', 'dino'):
    raise SystemExit(f"TSH_FLAVOUR must be 'lean' or 'dino', not {FLAVOUR!r}")

HEAVY = ['torch', 'torchvision', 'torchaudio', 'transformers', 'tokenizers', 'safetensors',
         'huggingface_hub', 'onnxruntime', 'tensorflow', 'jax', 'sklearn', 'scipy', 'pandas',
         'matplotlib', 'numba', 'sympy', 'IPython', 'jupyter', 'notebook']
# stdlib / dev packages nothing here imports (Tk is not used: the UI is a webview)
UNUSED = ['tkinter', '_tkinter', 'pytest', '_pytest', 'pydoc', 'pydoc_data', 'lib2to3']

hidden = []
excludes = list(UNUSED)
if FLAVOUR == 'lean':
    excludes += HEAVY
else:
    # transformers resolves model classes by *name* (lazy modules + auto-class mappings), which
    # PyInstaller's import scan cannot see.
    hidden += collect_submodules('transformers.models.dinov2')
    hidden += collect_submodules('transformers.models.auto')
    hidden += ['transformers.models.bit', 'transformers.models.dinov2.modeling_dinov2']
    excludes += ['tensorflow', 'jax', 'sklearn', 'pandas', 'matplotlib', 'numba', 'sympy',
                 'IPython', 'jupyter', 'notebook', 'onnxruntime', 'torchaudio']

# Modules app.py (or code it calls) imports lazily or by name, which PyInstaller's static scan
# can miss.  Listed explicitly and only when present, so this one spec builds every branch:
#   sellcalc          top-level module (sell calculator)             - branch feat/sell
#   autoscan          package (screen watcher; uses dxcam, else mss) - branch feat/autoscan
#   dxcam, comtypes   imported lazily inside autoscan; comtypes builds COM wrappers at run time
#   lifecycle         `app.py --watch` (the Windows-startup watcher) imports it before anything heavy;
#                     winreg is its only non-obvious import (stdlib, startup registration)
import importlib.util

# UnityPy: one-time extraction of the game's label font (identify/fontlabel.py, Anchor 2). It reads
# type trees from its bundled resources (.tpk) and imports its parsers by name.
unity_datas = []
if importlib.util.find_spec('UnityPy'):
    hidden += collect_submodules('UnityPy')
    unity_datas = collect_data_files('UnityPy')

hidden += ['lifecycle', 'winreg']
for mod in ('sellcalc',):
    if os.path.exists(os.path.join(SPECPATH, mod + '.py')):
        hidden.append(mod)
for pkg in ('autoscan', 'dxcam', 'comtypes'):
    local = os.path.isdir(os.path.join(SPECPATH, pkg))
    if local or importlib.util.find_spec(pkg):
        hidden += collect_submodules(pkg)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'),
           (os.path.join('identify', 'assets'), os.path.join('identify', 'assets'))] + unity_datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TarkovStashHelper',
    icon=os.path.join('assets', 'icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # windowed: no console window (app.py routes stdout to data/app.log)
    disable_windowed_traceback=False,
)
