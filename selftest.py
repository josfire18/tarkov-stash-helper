"""
Smoke test for a packaged build:  ``TarkovStashHelper.exe --selftest [report.json]``
(also ``python app.py --selftest``).

A PyInstaller exe can lose things a source checkout never does - a data file that was not
bundled, a module imported by name, a DLL - and most of them fail *silently* (for example
identify/digits.py treats a missing ``assets/digits.npz`` as "no prototypes" and simply stops
reading stack counts).  This runs every such check without opening the window, the tray icon
or the hotkey listener, writes a JSON report (a windowed exe has no console) and exits 0 when
every *required* check passed.  CI runs it against the freshly built exe before publishing it.

Checks marked ``required`` fail the run; the others (Tesseract, torch) are reported only,
because a lean build is *meant* to run without torch and CI has no Tesseract.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


def _check(checks: dict, name: str, fn, required: bool = True):
    t = time.perf_counter()
    try:
        detail = fn()
        checks[name] = {'ok': True, 'required': required, 'detail': detail,
                                  'sec': round(time.perf_counter() - t, 2)}
    except Exception as e:                                   # noqa: BLE001 - report everything
        checks[name] = {'ok': False, 'required': required,
                                  'detail': f'{type(e).__name__}: {e}',
                                  'trace': traceback.format_exc(limit=4),
                                  'sec': round(time.perf_counter() - t, 2)}


def run(app_mod) -> dict:
    """Run every check against the already-imported ``app`` module (in a frozen exe the entry
    script is ``__main__``, not importable as ``app``, so the caller passes it in)."""
    from identify import config

    report: dict = {'version': getattr(app_mod, 'APP_VERSION', '?'), 'frozen': bool(getattr(sys, 'frozen', False)),
                    'python': sys.version.split()[0], 'executable': sys.executable, 'checks': {}}
    C = report['checks']

    def paths():
        # data/ must be the same folder for the app and the engine, and next to the exe when frozen
        assert os.path.normcase(config.DATA_DIR) == os.path.normcase(app_mod.DATA), \
            f'engine data dir {config.DATA_DIR} != app data dir {app_mod.DATA}'
        if report['frozen']:
            assert os.path.normcase(app_mod.BASE) == os.path.normcase(os.path.dirname(sys.executable))
        return {'root': config.ROOT, 'data': config.DATA_DIR, 'bundle': app_mod.BUNDLE}
    _check(C, 'paths', paths)

    def data_writable():
        p = os.path.join(config.DATA_DIR, '.selftest')
        with open(p, 'w') as f:
            f.write('x')
        os.remove(p)
        return config.DATA_DIR
    _check(C, 'data_dir_writable', data_writable)

    def ui():
        client = app_mod.app.test_client()
        out = {}
        for route in ('/', '/sell', '/api/health'):
            r = client.get(route)
            assert r.status_code == 200, f'GET {route} -> {r.status_code}'
            out[route] = len(r.data)
        return out
    _check(C, 'ui_templates_and_routes', ui)

    def digits():
        from identify import digits as D
        protos = D._load()
        n = sum(len(v) for v in protos.values())
        assert n > 0, f'no digit prototypes loaded from {D.ASSET} (identify/assets not bundled?)'
        return {'prototypes': n, 'asset': D.ASSET}
    _check(C, 'digit_prototypes', digits)

    def engine_modules():
        import importlib
        for m in ('identify.pipeline', 'identify.grid', 'identify.segment', 'identify.catalog',
                  'identify.match', 'identify.ocr', 'identify.dino', 'identify.learned'):
            importlib.import_module(m)
        import numpy as np
        from identify.grid import detect_grid
        blank = np.full((600, 800, 3), (44, 42, 38), np.uint8)
        g = detect_grid(blank)                    # exercises cv2 + numpy; a blank frame has no panels
        return {'blank_frame_panels': len(g.panels)}
    _check(C, 'engine_imports_and_grid', engine_modules)

    def desktop_shell():
        import importlib
        for m in ('webview', 'pystray', 'pynput.keyboard', 'waitress', 'mss', 'rapidfuzz', 'pytesseract'):
            importlib.import_module(m)
        return 'ok'
    _check(C, 'desktop_shell_imports', desktop_shell)

    def catalog():
        have = os.path.exists(config.CATALOG_PATH)
        return {'catalog_present': have, 'prices_present': os.path.exists(config.PRICES_PATH),
                'tmpl_src_images': (len([f for f in os.listdir(config.TMPL_SRC_DIR) if f.endswith('.png')])
                                    if os.path.isdir(config.TMPL_SRC_DIR) else 0)}
    _check(C, 'catalog_sources', catalog)

    def tesseract():
        ok = bool(app_mod.tesseract_available())
        assert ok, 'Tesseract not found (install it: winget install UB-Mannheim.TesseractOCR)'
        return app_mod.pytesseract.pytesseract.tesseract_cmd
    _check(C, 'tesseract', tesseract, required=False)

    def accel():
        from identify import dino, match
        try:
            import torch
        except Exception as e:                               # noqa: BLE001 - lean build: expected
            return {'torch': None, 'cuda': False, 'stage1_gpu': False, 'dino_available': False,
                    'torch_error': f'{type(e).__name__}: {e}'}
        out = {'torch': torch.__version__, 'cuda': bool(torch.cuda.is_available()),
               'stage1_gpu': bool(match.enable_torch())}
        match.disable_torch()
        out['dino_available'] = bool(dino.available())
        out['dino_error'] = dino._state.get('err') or None
        return out
    _check(C, 'acceleration', accel, required=False)

    report['ok'] = all(c['ok'] for c in C.values() if c['required'])
    return report


def main(app_mod, argv: list[str]) -> int:
    out = argv[0] if argv and not argv[0].startswith('-') else os.path.join(app_mod.DATA, 'selftest.json')
    report = run(app_mod)
    try:
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
    except OSError:
        pass
    if sys.stdout is not None:
        print(json.dumps(report, indent=2, default=str))
    return 0 if report['ok'] else 1
