#!/usr/bin/env python3
"""
Offline stash-scan evaluation harness.

Runs grid detection + NCC identification on saved screenshots without needing
Tarkov running or Flask started, and measures accuracy against labelled ground
truth so every tuning change is provable.

Usage
-----
  # 1. Save 2-3 real full-stash screenshots (native resolution) here:
  #        data/eval/<name>.png

  # 2. Generate a ground-truth template you correct once:
  python test_scan.py --label data/eval/myshot.png
  #    → writes data/eval/myshot.truth.json  (list of {col,row,W,H,item_id})
  #    → also writes data/eval/myshot.label.html — open it, eyeball each crop
  #      against the guessed name, fix wrong item_ids in the .truth.json.

  # 3. Score detection against the corrected truth (per-failure dump included):
  python test_scan.py --score data/eval/*.png
  #    → precision / recall / accuracy per image and overall.

Legacy (no flag): dumps detections for a single image (default data/test_stash.png).

  python test_scan.py [path/to/stash.png] [col row W H]   # + optional top-5 probe
"""
import sys
import os
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles default to cp1252 which can't encode the arrows / box-drawing
# used in the failure dumps — force UTF-8 so --score never crashes mid-report.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
import numpy as np

from app import (
    detect_stash_grid,
    resolve_grid,
    validate_grid,
    identify_items_by_icon,
    build_label_matcher,
    tesseract_available,
    load_icon_db,
    load_json,
    default_settings,
    SETTINGS_PATH,
    PRICES_PATH,
    _cell_block,
    _native_cell_vec,
    _slot_px,
    get_db_at_pitch,
    _masked_ncc_scores,
    _best_with_margin,
    ICON_MATCH_MIN_SCORE,
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'eval')


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _load_grid(img_bgr):
    settings = load_json(SETTINGS_PATH, default_settings)
    grid, src = resolve_grid(img_bgr, settings)
    return grid, src


_label_matcher_cache = [False, None]   # [initialized?, matcher-or-None]


def _get_label_matcher():
    """
    The OCR-label matcher the live scans use, built from the offline price
    cache (no network).  None when Tesseract or the price cache is missing —
    the scan then measures NCC alone, which is NOT the shipped pipeline.
    """
    if not _label_matcher_cache[0]:
        _label_matcher_cache[0] = True
        prices = load_json(PRICES_PATH, lambda: {})
        if not tesseract_available():
            print('NOTE : Tesseract missing — scoring NCC-only (not the live pipeline)')
        elif not prices.get('items'):
            print('NOTE : no price cache — scoring NCC-only (not the live pipeline)')
        else:
            _label_matcher_cache[1] = build_label_matcher(prices)
    return _label_matcher_cache[1]


def scan_image(img_bgr, raw_db):
    """
    Grid detection + pitch resample + NCC identification + OCR-label fusion —
    the same pipeline the live scans run.  Returns (detections, grid,
    grid_src, matcher_db) — matcher_db is the pitch-matched DB needed by
    top_matches for the same image.
    """
    grid, src = _load_grid(img_bgr)
    print(f"Grid[{src}]: cell={grid['cell_w']:.2f}×{grid['cell_h']:.2f}px  "
          f"origin=({grid['origin_x']:.1f},{grid['origin_y']:.1f})"
          + (f"  strength={grid['strength']}" if 'strength' in grid else ''))
    mdb = get_db_at_pitch(raw_db, *_slot_px(grid))
    detections = identify_items_by_icon(img_bgr, grid, mdb,
                                        label_matcher=_get_label_matcher())
    return detections, grid, src, mdb


def top_matches(img_bgr, col, row, W, H, grid, mdb, n=5):
    """Top-N NCC matches for a specific footprint (debugging / failure dump)."""
    spw, sph = _slot_px(grid)
    vec = _native_cell_vec(img_bgr, col, row, W, H, grid, spw, sph)
    if vec is None or (W, H) not in mdb:
        return []
    bucket = mdb[(W, H)]
    scores = _masked_ncc_scores(vec, bucket)
    idxs = np.argsort(scores)[::-1][:n]
    return [(float(scores[i]), bucket['names'][i], bucket['ids'][i],
             bucket['sources'][i], bool(bucket['rotated'][i])) for i in idxs]


def _require_db():
    icon_db = load_icon_db()
    if icon_db is None:
        print("ERROR: No icon DB found (or version mismatch).")
        print("  → Open the web app, click Build Icon DB, then retry.")
        sys.exit(1)
    total = sum(len(b['ids']) for b in icon_db.values())
    print(f"DB   : {len(icon_db)} size buckets, {total} templates")
    return icon_db


def _truth_path(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(os.path.dirname(img_path), stem + '.truth.json')


# ---------------------------------------------------------------------------
# --label : produce an editable ground-truth + visual HTML
# ---------------------------------------------------------------------------

def _crop_data_uri(img_bgr, col, row, W, H, grid):
    crop = _cell_block(img_bgr, col, row, W, H, grid)
    if crop is None or crop.size == 0:
        return ''
    ok, buf = cv2.imencode('.png', crop)
    if not ok:
        return ''
    import base64
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()


def label_mode(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: cannot load '{img_path}'")
        sys.exit(1)
    icon_db = _require_db()
    detections, grid, _, mdb = scan_image(img_bgr, icon_db)
    detections = sorted(detections, key=lambda d: (d['row'], d['col']))

    truth = [{'col': d['col'], 'row': d['row'], 'W': d['W'], 'H': d['H'],
              'item_id': d['item_id'], 'name': d['name'],
              'rotated': d.get('rotated', False)} for d in detections]

    tp = _truth_path(img_path)
    if os.path.exists(tp):
        print(f"NOTE : {tp} already exists — writing guesses to {tp}.new instead "
              "(merge manually so you don't lose corrections).")
        tp = tp + '.new'
    with open(tp, 'w', encoding='utf-8') as f:
        json.dump(truth, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(truth)} guesses → {tp}")

    # Visual HTML: crop + guessed name + top-5 so you can verify/fix quickly.
    rows = []
    for d in detections:
        uri = _crop_data_uri(img_bgr, d['col'], d['row'], d['W'], d['H'], grid)
        tops = top_matches(img_bgr, d['col'], d['row'], d['W'], d['H'], grid, mdb)
        alt = '<br>'.join(f"{sc:.3f} {name} [{src}{'/rot' if rot else ''}]"
                          for sc, name, _id, src, rot in tops)
        rot = ' (rot)' if d.get('rotated') else ''
        rows.append(
            f"<tr><td><img src='{uri}' style='max-height:96px;border:1px solid #333'></td>"
            f"<td>({d['col']},{d['row']}) {d['W']}×{d['H']}{rot}</td>"
            f"<td><b>{d['name']}</b><br><code>{d['item_id']}</code><br>"
            f"{d['score']}%</td><td style='font-size:11px;color:#888'>{alt}</td></tr>")
    html = ("<html><body style='background:#111;color:#ccc;font-family:sans-serif'>"
            f"<h3>{os.path.basename(img_path)} — {len(detections)} detections</h3>"
            "<p>Fix wrong <code>item_id</code>s in the .truth.json, delete false "
            "positives, add rows for missed items.</p>"
            "<table cellpadding=6 style='border-collapse:collapse'>"
            "<tr><th>crop</th><th>cell</th><th>guess</th><th>top-5</th></tr>"
            + ''.join(rows) + "</table></body></html>")
    hp = os.path.join(os.path.dirname(img_path),
                      os.path.splitext(os.path.basename(img_path))[0] + '.label.html')
    with open(hp, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Wrote visual check → {hp}  (open in a browser)")


# ---------------------------------------------------------------------------
# --score : precision / recall / accuracy vs. truth
# ---------------------------------------------------------------------------

def score_mode(img_paths):
    icon_db = _require_db()
    grand = {'tp': 0, 'fp': 0, 'fn': 0, 'wrong': 0, 'truth': 0}

    for img_path in img_paths:
        tp_path = _truth_path(img_path)
        if not os.path.exists(tp_path):
            print(f"\n{os.path.basename(img_path)} — no {os.path.basename(tp_path)}; "
                  "run --label first, correct it, then --score.")
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"\n{os.path.basename(img_path)} — cannot load; skipping.")
            continue
        with open(tp_path, encoding='utf-8') as f:
            truth = json.load(f)

        print(f"\n{'='*66}\n{os.path.basename(img_path)}  ({len(truth)} labelled items)")
        detections, grid, _, mdb = scan_image(img_bgr, icon_db)

        # Index truth by anchor cell (col,row).
        truth_by_cell = {(t['col'], t['row']): t for t in truth}
        det_by_cell = {(d['col'], d['row']): d for d in detections}

        tp = fp = fn = wrong = 0
        failures = []
        for cell, t in truth_by_cell.items():
            d = det_by_cell.get(cell)
            if d is None:
                fn += 1
                failures.append(('MISSED', cell, t, None))
            elif d['item_id'] == t['item_id']:
                tp += 1
            else:
                wrong += 1
                failures.append(('WRONG', cell, t, d))
        for cell, d in det_by_cell.items():
            if cell not in truth_by_cell:
                fp += 1
                failures.append(('EXTRA', cell, None, d))

        n = len(truth)
        acc = tp / n if n else 0
        prec = tp / (tp + wrong + fp) if (tp + wrong + fp) else 0
        rec = tp / (tp + wrong + fn) if (tp + wrong + fn) else 0
        print(f"  accuracy {acc*100:5.1f}%   precision {prec*100:5.1f}%   "
              f"recall {rec*100:5.1f}%")
        print(f"  correct={tp}  wrong={wrong}  missed={fn}  extra={fp}")

        for kind, cell, t, d in failures[:40]:
            if kind == 'MISSED':
                print(f"    MISSED ({cell[0]:2d},{cell[1]:2d}) {t['W']}×{t['H']}  "
                      f"want '{t['name']}'")
                for sc, name, _id, src, rot in top_matches(
                        img_bgr, cell[0], cell[1], t['W'], t['H'], grid, mdb):
                    hit = ' ←WANT' if _id == t['item_id'] else ''
                    print(f"        {sc:6.3f} {name} [{src}{'/rot' if rot else ''}]{hit}")
            elif kind == 'WRONG':
                print(f"    WRONG  ({cell[0]:2d},{cell[1]:2d})  got '{d['name']}' "
                      f"({d['score']}%)  want '{t['name']}'")
                for sc, name, _id, src, rot in top_matches(
                        img_bgr, cell[0], cell[1], t['W'], t['H'], grid, mdb):
                    hit = ' ←WANT' if _id == t['item_id'] else ''
                    print(f"        {sc:6.3f} {name} [{src}{'/rot' if rot else ''}]{hit}")
            else:  # EXTRA
                print(f"    EXTRA  ({cell[0]:2d},{cell[1]:2d}) {d['W']}×{d['H']}  "
                      f"got '{d['name']}' ({d['score']}%)")

        grand['tp'] += tp; grand['fp'] += fp; grand['fn'] += fn
        grand['wrong'] += wrong; grand['truth'] += n

    n = grand['truth']
    if n:
        tp, wrong, fn, fp = grand['tp'], grand['wrong'], grand['fn'], grand['fp']
        acc = tp / n
        prec = tp / (tp + wrong + fp) if (tp + wrong + fp) else 0
        rec = tp / (tp + wrong + fn) if (tp + wrong + fn) else 0
        print(f"\n{'='*66}\nOVERALL  {n} items across {len(img_paths)} image(s)")
        print(f"  accuracy {acc*100:5.1f}%   precision {prec*100:5.1f}%   "
              f"recall {rec*100:5.1f}%")
        print(f"  correct={tp}  wrong={wrong}  missed={fn}  extra={fp}")
        print(f"  TARGET ≥95% accuracy — {'PASS' if acc >= 0.95 else 'not yet'}")


# ---------------------------------------------------------------------------
# Legacy single-image dump
# ---------------------------------------------------------------------------

def dump_mode(argv):
    img_path = argv[0] if argv else os.path.join(
        os.path.dirname(__file__), 'data', 'test_stash.png')
    print(f"Image: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: cannot load '{img_path}'  (save one under data/eval/)")
        sys.exit(1)
    print(f"Size : {img_bgr.shape[1]}×{img_bgr.shape[0]}")
    icon_db = _require_db()
    detections, grid, _, mdb = scan_image(img_bgr, icon_db)

    print(f"\n{'─'*66}\n{'col':>4} {'row':>4}  {'size':>5}  {'rot':>3}  "
          f"{'src':>5}  {'score':>6}  item\n{'─'*66}")
    for d in sorted(detections, key=lambda x: (x['row'], x['col'])):
        print(f"{d['col']:>4} {d['row']:>4}  {d['W']}×{d['H']:<3}  "
              f"{'Y' if d.get('rotated') else '·':>3}  {d.get('source','?'):>5}  "
              f"{d['score']:>5.1f}%  {d['name']}")
    print(f"{'─'*66}\nTotal: {len(detections)} items")

    if len(argv) >= 3:
        col, row = int(argv[1]), int(argv[2])
        W = int(argv[3]) if len(argv) > 3 else 1
        H = int(argv[4]) if len(argv) > 4 else 1
        print(f"\nTop-5 for ({col},{row}) {W}×{H}:")
        for sc, name, _id, src, rot in top_matches(img_bgr, col, row, W, H, grid, mdb):
            flag = ' ← ACCEPTED' if sc >= ICON_MATCH_MIN_SCORE else ''
            print(f"  {sc:6.3f} {name} [{src}{'/rot' if rot else ''}]{flag}")


def main():
    args = sys.argv[1:]
    if args and args[0] == '--label':
        if len(args) < 2:
            print("usage: python test_scan.py --label data/eval/shot.png")
            sys.exit(1)
        label_mode(args[1])
    elif args and args[0] == '--score':
        paths = []
        for a in args[1:]:
            paths.extend(sorted(glob.glob(a)) if any(c in a for c in '*?[') else [a])
        if not paths:
            print("usage: python test_scan.py --score data/eval/*.png")
            sys.exit(1)
        score_mode(paths)
    else:
        dump_mode(args)


if __name__ == '__main__':
    main()
