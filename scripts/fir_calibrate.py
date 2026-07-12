"""
scripts/fir_calibrate.py — one-time human-eyeballing harness for tuning the
Found-in-Raid (FiR) checkmark detector (app.detect_fir, A3).

For every data/eval/*.png screenshot, detects the stash grid, walks every
occupied cell, and dumps an ENLARGED PNG tile of the exact top-right corner
window detect_fir samples — the same window, same size — so a human can
eyeball whether FIR_BRIGHT / FIR_MIN_PX / FIR_MAX_FRAC (app.py) need
retuning. The current verdict (t/f/n) is baked into each tile's filename so
misclassifications jump out just from a directory listing / thumbnail grid.

This is diagnostic-only: it never asserts anything and never fails CI. Run
directly:  python scripts/fir_calibrate.py
"""

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import detect_stash_grid, detect_fir, _build_occupancy   # noqa: E402


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


VERDICT_TAG = {True: 't', False: 'f', None: 'n'}
TILE_SCALE = 8   # enlarge the (small, ~5-8px) corner window for legibility


def _corner_window_bounds(img_bgr, grid, col, row):
    """
    Mirrors detect_fir's own window math exactly (see app.py) so the dumped
    tile is the literal pixels the detector scored — not an approximation.
    """
    x2 = grid['origin_x'] + (col + 1) * grid['cell_w']
    y1 = grid['origin_y'] + row * grid['cell_h']
    c = round(18 * grid['cell_w'] / 63)
    sh, sw = img_bgr.shape[:2]
    wx1, wy1 = max(0, x2 - c), max(0, y1)
    wx2, wy2 = min(sw, x2), min(sh, y1 + c)
    return wx1, wy1, wx2, wy2


def dump_tiles_for_screenshot(path, out_root):
    img_bgr = cv2.imread(path)
    if img_bgr is None or img_bgr.size == 0:
        print(f"[fir_calibrate] could not read {path} — skipping")
        return 0

    grid = detect_stash_grid(img_bgr)
    if not grid:
        print(f"[fir_calibrate] grid detection failed for {path} — skipping")
        return 0

    sh, sw = img_bgr.shape[:2]
    cw, ch = grid['cell_w'], grid['cell_h']
    ox, oy = grid['origin_x'], grid['origin_y']
    n_cols = max(0, (sw - ox) // cw)
    n_rows = max(0, (sh - oy) // ch)
    if n_cols == 0 or n_rows == 0:
        print(f"[fir_calibrate] empty grid for {path} — skipping")
        return 0

    occ = _build_occupancy(img_bgr, grid, n_rows, n_cols)

    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if not occ[row, col]:
                continue
            verdict = detect_fir(img_bgr, grid, col, row, 1, 1)
            tag = VERDICT_TAG[verdict]

            wx1, wy1, wx2, wy2 = _corner_window_bounds(img_bgr, grid, col, row)
            if wy2 - wy1 < 1 or wx2 - wx1 < 1:
                continue   # nothing to dump (window fell entirely off-image)
            tile = img_bgr[wy1:wy2, wx1:wx2]
            tile = cv2.resize(tile, None, fx=TILE_SCALE, fy=TILE_SCALE,
                              interpolation=cv2.INTER_NEAREST)

            out_path = os.path.join(out_dir, f'{col}x{row}_{tag}.png')
            cv2.imwrite(out_path, tile)
            n_written += 1

    print(f"[fir_calibrate] {stem}: wrote {n_written} tile(s) -> {out_dir}")
    return n_written


def main():
    eval_dir = os.path.join(_project_root(), 'data', 'eval')
    out_root = os.path.join(eval_dir, 'fir_tiles')
    screenshots = sorted(glob.glob(os.path.join(eval_dir, '*.png')))

    if not screenshots:
        print(f"[fir_calibrate] no PNG screenshots found in {eval_dir} — "
              "nothing to do (add some data/eval/*.png captures and re-run)")
        return

    total = 0
    for path in screenshots:
        total += dump_tiles_for_screenshot(path, out_root)
    print(f"[fir_calibrate] done — {total} tile(s) total across "
          f"{len(screenshots)} screenshot(s)")


if __name__ == '__main__':
    main()
