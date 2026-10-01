#!/usr/bin/env python3
"""
Offline stash-scan evaluation harness (no Tarkov, no Flask needed).

Two engines can be scored side by side on the same labelled screenshots:

  --engine v2        the new ``identify`` package (default)
  --engine legacy    the old app.py masked-NCC pipeline
  --engine both      run both and print a comparison table

Usage
-----
  # score (segmentation / identification / per-category / uncertain / end-to-end)
  python test_scan.py --score data/eval/stash1.png [--engine v2|legacy|both]
                      [--truth full|orig|<path>] [--no-dino] [--no-ocr] [--json out.json]

  # label a new screenshot (v2 prefill + contact sheets), then correct and re-score
  python test_scan.py --prefill data/eval/shot.png
  python test_scan.py --relabel data/eval/shot.png corrections.json
  python test_scan.py --score data/eval/shot.png --truth full

  # legacy helpers (unchanged): --label <img> writes the old truth+HTML; no flag dumps one image
  python test_scan.py --label data/eval/shot.png
  python test_scan.py [path/to/stash.png] [col row W H]

What is measured (``--score``)
  segmentation    footprint rectangles, IoU >= 0.9, one-to-one: precision / recall
  identification  top-1 accuracy on correctly segmented, non-"uncertain" truth items,
                  overall and per category (ammo, weapon, mod, meds, ...)
  uncertain       share of detections the engine flagged uncertain, how many of its wrong
                  answers it flagged, how many flagged answers were in fact right
  end-to-end      truth items found with the right rectangle AND the right item id
"""
import sys
import os
import glob
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles default to cp1252 which can't encode the symbols used in reports.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'eval')
PRICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prices_cache.json')
IOU_MIN = 0.9


# ---------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------

def _stem(img_path):
    return os.path.splitext(img_path)[0]


def truth_path(img_path, which='full'):
    if which in ('full', 'orig'):
        return _stem(img_path) + ('.truth.full.json' if which == 'full' else '.truth.json')
    return which


def load_truth(img_path, which='full'):
    p = truth_path(img_path, which)
    if which == 'full' and not os.path.exists(p):
        p = truth_path(img_path, 'orig')
    if not os.path.exists(p):
        return None, p
    with open(p, encoding='utf-8') as f:
        return json.load(f), p


def _categories():
    """item_id -> category (same mapping the catalog uses)."""
    from identify.catalog import category_of
    with open(PRICES, encoding='utf-8') as f:
        return {it['id']: category_of(it.get('types')) for it in json.load(f)['items']}


def _legacy_truth_to_rects(truth, panel0):
    """Original ``stash1.truth.json`` rows have (col,row) on the legacy grid (origin y=141,
    i.e. the *second* line of the v2 lattice) and no rect: map them through the v2 panel."""
    if not truth or 'rect' in truth[0]:
        return truth
    off = int(round((141.0 - panel0.ys[0]) / panel0.pitch_y))
    out = []
    for t in truth:
        t = dict(t)
        c, r, w, h = t['col'], t['row'] + off, t['W'], t['H']
        t['rect'] = list(panel0.rect(c, r, w, h))
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# engines -> normalised detections
#   each detection: {'rect': [x,y,w,h], 'item_id', 'name', 'uncertain', 'conf'}
# ---------------------------------------------------------------------------

_v2_engines = {}
PITCH_HINT = None          # set by --pitch-hint (labelling aid for lattices the engine cannot lock by itself)


def run_v2(img_bgr, use_dino=True, use_ocr=True, warm=True):
    from identify.config import EngineSettings
    from identify.pipeline import Engine
    key = (use_dino, use_ocr)
    if key not in _v2_engines:
        _v2_engines[key] = Engine(EngineSettings(use_dino=use_dino, use_ocr=use_ocr, pitch_hint=PITCH_HINT))
        if warm:                       # first call pays model load / CUDA init: don't time it
            _v2_engines[key].scan(img_bgr)
    t = time.perf_counter()
    res = _v2_engines[key].scan(img_bgr)
    dt = time.perf_counter() - t
    dets = [{'rect': list(d.rect), 'item_id': d.item_id, 'name': d.name, 'uncertain': d.uncertain,
             'conf': d.confidence, 'det': d} for d in res.detections]
    return dets, dt, res


def run_legacy(img_bgr):
    import app as A
    icon_db = A.load_icon_db()
    if icon_db is None:
        raise SystemExit('legacy engine needs data/icon_db.npz (Build Icon DB in the app)')
    settings = dict(A.load_json(A.SETTINGS_PATH, A.default_settings))
    settings.pop('grid', None)
    prices = A.load_json(A.PRICES_PATH, lambda: {})
    lm = A.build_label_matcher(prices) if (A.tesseract_available() and prices.get('items')) else None
    t = time.perf_counter()
    panels, _src = A.resolve_panels(img_bgr, settings, persist_fn=None)
    mdb = A.get_db_at_pitch(icon_db, *A._slot_px(panels[0]))
    raw = A.scan_all_panels(img_bgr, panels, mdb, label_matcher=lm)
    dt = time.perf_counter() - t
    dets = [{'rect': [d['px'], d['py'], d['pw'], d['ph']], 'item_id': d['item_id'], 'name': d['name'],
             'uncertain': False, 'conf': d['score'] / 100.0} for d in raw]
    return dets, dt, None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def match_rects(truth_rects, det_rects, thr=IOU_MIN):
    """Greedy one-to-one IoU matching -> {truth_idx: det_idx}."""
    pairs = []
    for i, t in enumerate(truth_rects):
        for j, d in enumerate(det_rects):
            v = iou(t, d)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    mt, md = {}, {}
    for v, i, j in pairs:
        if i not in mt and j not in md:
            mt[i] = j
            md[j] = i
    return mt


def score(truth, dets, cats, verbose=True):
    """Compute the metric dict for one image."""
    tr = [t['rect'] for t in truth]
    dr = [d['rect'] for d in dets]
    mt = match_rects(tr, dr)
    n_t, n_d, n_m = len(truth), len(dets), len(mt)
    seg = {'truth': n_t, 'detections': n_d, 'matched': n_m,
           'recall': n_m / n_t if n_t else 0.0, 'precision': n_m / n_d if n_d else 0.0}

    ident = {'n': 0, 'ok': 0}
    percat = {}
    wrong = []
    caught = needless = flagged = 0
    e2e_ok = 0
    e2e_n = 0
    for i, t in enumerate(truth):
        if t.get('uncertain') or not t.get('item_id'):
            continue
        e2e_n += 1
        j = mt.get(i)
        if j is None:
            continue
        d = dets[j]
        cat = t.get('category') or cats.get(t['item_id'], 'other')
        c = percat.setdefault(cat, {'n': 0, 'ok': 0})
        ident['n'] += 1
        c['n'] += 1
        ok = d['item_id'] == t['item_id'] or (bool(d.get('name')) and d.get('name') == t.get('name'))   # twin ids share a name
        if ok:
            ident['ok'] += 1
            c['ok'] += 1
            e2e_ok += 1
        else:
            wrong.append((i, t, d))
        if d.get('uncertain'):
            flagged += 1
            if ok:
                needless += 1
            else:
                caught += 1
    n_unc = sum(1 for d in dets if d.get('uncertain'))
    wrong_n = ident['n'] - ident['ok']
    return {
        'segmentation': seg,
        'identification': {**ident, 'acc': ident['ok'] / ident['n'] if ident['n'] else 0.0},
        'per_category': {k: {**v, 'acc': v['ok'] / v['n'] if v['n'] else 0.0} for k, v in sorted(percat.items())},
        'uncertain': {'rate': n_unc / n_d if n_d else 0.0, 'flagged': n_unc,
                      'wrong_flagged': caught, 'wrong_total': wrong_n,
                      'flagged_but_right': needless},
        'end_to_end': {'ok': e2e_ok, 'n': e2e_n, 'acc': e2e_ok / e2e_n if e2e_n else 0.0},
        '_wrong': wrong, '_unmatched': [i for i in range(n_t) if i not in mt],
    }


def _pct(x):
    return f'{100 * x:5.1f}%'


def print_report(name, m, dt, truth, dets, show_wrong=True):
    s, i, u, e = m['segmentation'], m['identification'], m['uncertain'], m['end_to_end']
    print(f'\n--- {name}  ({dt:.2f}s) ---')
    print(f"  segmentation   recall {_pct(s['recall'])} ({s['matched']}/{s['truth']})   "
          f"precision {_pct(s['precision'])} ({s['matched']}/{s['detections']})")
    print(f"  identification top-1 {_pct(i['acc'])} ({i['ok']}/{i['n']}) on segmented, non-uncertain truth")
    for k, v in m['per_category'].items():
        print(f"      {k:<10} {_pct(v['acc'])} ({v['ok']}/{v['n']})")
    print(f"  uncertain      {_pct(u['rate'])} of detections flagged; catches {u['wrong_flagged']}/{u['wrong_total']} "
          f"wrong answers; {u['flagged_but_right']} flagged answers were right")
    print(f"  end-to-end     {_pct(e['acc'])} ({e['ok']}/{e['n']})  right rectangle AND right item")
    if show_wrong:
        for idx, t, d in m['_wrong'][:30]:
            print(f"    WRONG #{idx} {t['rect']} want '{t.get('name')}'  got '{d['name']}'"
                  f"{'  [flagged uncertain]' if d.get('uncertain') else ''}")
        for idx in m['_unmatched'][:20]:
            t = truth[idx]
            print(f"    NOT SEGMENTED #{idx} {t['rect']} '{t.get('name')}'")


def _strip(m):
    return {k: v for k, v in m.items() if not k.startswith('_')}


# ---------------------------------------------------------------------------
# --score
# ---------------------------------------------------------------------------

def score_mode(paths, engine, which, use_dino, use_ocr, json_out):
    cats = _categories()
    engines = ['v2', 'legacy'] if engine == 'both' else [engine]
    results = {}
    agg = {e: {'truth': 0, 'matched': 0, 'dets': 0, 'ok': 0, 'n': 0, 'e2e_ok': 0, 'e2e_n': 0,
               'time': 0.0, 'cat': {}} for e in engines}
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'cannot load {path}')
            continue
        truth, tp = load_truth(path, which)
        if truth is None:
            print(f'\n{os.path.basename(path)}: no truth file ({tp}); run --prefill first')
            continue
        print(f"\n{'=' * 70}\n{os.path.basename(path)}  truth={os.path.basename(tp)}  ({len(truth)} rows, "
              f"{sum(1 for t in truth if t.get('uncertain'))} uncertain)")
        if truth and 'rect' not in truth[0]:
            from identify.grid import detect_grid
            gp = detect_grid(img).panels
            truth = _legacy_truth_to_rects(truth, gp[0])
        for e in engines:
            if e == 'v2':
                dets, dt, _ = run_v2(img, use_dino, use_ocr)
            else:
                dets, dt, _ = run_legacy(img)
            m = score(truth, dets, cats)
            print_report(e, m, dt, truth, dets)
            results.setdefault(os.path.basename(path), {})[e] = {**_strip(m), 'time': dt}
            a = agg[e]
            a['truth'] += m['segmentation']['truth']; a['matched'] += m['segmentation']['matched']
            a['dets'] += m['segmentation']['detections']
            a['ok'] += m['identification']['ok']; a['n'] += m['identification']['n']
            a['e2e_ok'] += m['end_to_end']['ok']; a['e2e_n'] += m['end_to_end']['n']
            a['time'] += dt
            for k, v in m['per_category'].items():
                c = a['cat'].setdefault(k, [0, 0])
                c[0] += v['ok']; c[1] += v['n']
    if len(paths) > 1 or engine == 'both':
        print(f"\n{'=' * 70}\nOVERALL")
        hdr = f"  {'':<22}" + ''.join(f'{e:>12}' for e in engines)
        print(hdr)

        def row(label, fn):
            print(f'  {label:<22}' + ''.join(f'{fn(agg[e]):>12}' for e in engines))
        row('seg recall', lambda a: _pct(a['matched'] / a['truth']) if a['truth'] else '-')
        row('seg precision', lambda a: _pct(a['matched'] / a['dets']) if a['dets'] else '-')
        row('identification', lambda a: _pct(a['ok'] / a['n']) if a['n'] else '-')
        row('end-to-end', lambda a: _pct(a['e2e_ok'] / a['e2e_n']) if a['e2e_n'] else '-')
        row('time/scan (s)', lambda a: f"{a['time'] / max(1, len(paths)):.2f}")
        allcats = sorted({k for e in engines for k in agg[e]['cat']})
        for k in allcats:
            row(f'  {k}', lambda a, k=k: (f"{_pct(a['cat'][k][0] / a['cat'][k][1])} ({a['cat'][k][0]}/{a['cat'][k][1]})"
                                           if k in a['cat'] and a['cat'][k][1] else '-'))
    if json_out:
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1, default=str)
        print(f'\nwrote {json_out}')
    return results


# ---------------------------------------------------------------------------
# --robustness: how do JPEG / blur / rescaling hurt each stage?
# ---------------------------------------------------------------------------

def make_variant(img, kind):
    """Degraded copy of a screenshot + the geometric scale applied to it ((fx, fy) or one float)."""
    if kind == 'png':
        return img, 1.0
    if kind.startswith('stretch'):            # 4:3 -> 16:9 style horizontal stretch (the user's setup)
        fx = float(kind[7:] or 1.3333)
        return cv2.resize(img, None, fx=fx, fy=1.0, interpolation=cv2.INTER_CUBIC), (fx, 1.0)
    if kind.startswith('jpg'):
        q = int(kind[3:])
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), 1.0
    if kind == 'blur':
        return cv2.GaussianBlur(img, (0, 0), 1.0), 1.0
    if kind == 'noise':
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8), 1.0
    if kind.startswith('scale'):
        f = float(kind[5:])
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC), f
    raise ValueError(kind)


ROBUST_KINDS = ['png', 'jpg95', 'jpg85', 'jpg70', 'jpg50', 'jpg30', 'blur', 'noise', 'scale0.83', 'scale1.33',
                'stretch1.3333']
ROBUST_CONFIGS = [('pixels only', False, False), ('+DINO', True, False), ('+OCR', False, True), ('full', True, True)]


def _scale_rect(rect, f):
    fx, fy = (f, f) if not isinstance(f, tuple) else f
    x, y, w, h = rect
    return [int(round(x * fx)), int(round(y * fy)), int(round(w * fx)), int(round(h * fy))]


def robustness_mode(img_path, which, kinds=None, json_out=None):
    cats = _categories()
    img = cv2.imread(img_path)
    truth, _ = load_truth(img_path, which)
    cols = kinds or ROBUST_KINDS
    table = {}
    print(f"\nrobustness on {os.path.basename(img_path)} ({len(truth)} truth rows): "
          "identification top-1 / segmentation recall")
    print(f"  {'variant':<11}" + ''.join(f'{c[0]:>20}' for c in ROBUST_CONFIGS))
    for kind in cols:
        v, f = make_variant(img, kind)
        tr = [dict(t, rect=_scale_rect(t['rect'], f)) for t in truth]
        row = []
        for name, dino, ocr in ROBUST_CONFIGS:
            dets, dt, _ = run_v2(v, dino, ocr)
            m = score(tr, dets, cats)
            row.append((m['identification']['acc'], m['segmentation']['recall'], dt))
            table.setdefault(kind, {})[name] = {'id_acc': m['identification']['acc'],
                                               'seg_recall': m['segmentation']['recall'], 'time': dt}
        print(f"  {kind:<11}" + ''.join(f'{100 * a:>12.1f}% /{100 * r:>4.0f}%' for a, r, _ in row))
    if json_out:
        with open(json_out, 'w', encoding='utf-8') as fh:
            json.dump(table, fh, indent=1)
    return table


# ---------------------------------------------------------------------------
# --prefill / --relabel
# ---------------------------------------------------------------------------

def prefill_mode(img_path, use_dino=True):
    from identify import evaltools as ET
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f'cannot load {img_path}')
    dets, dt, res = run_v2(img, use_dino, True, warm=False)
    eng = _v2_engines[(use_dino, True)]
    rows = ET.truth_from_detections([d['det'] for d in dets], res.items)
    out = truth_path(img_path, 'full')
    if os.path.exists(out):
        out += '.new'
        print(f'NOTE: {truth_path(img_path, "full")} exists; writing the draft to {out}')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    sheets = ET.contact_sheets(img, rows, [d['det'] for d in dets], eng.cat, _stem(img_path) + '.sheet')
    print(f'{len(rows)} footprints ({len(dets)} identified) in {dt:.1f}s -> {out}')
    print('contact sheets:\n  ' + '\n  '.join(sheets))
    print('Check each sheet by eye, write corrections as {"<index>": {"name": "...", "uncertain": false, '
          '"note": "..."}} and apply with --relabel.')


def relabel_mode(img_path, corr_path):
    from identify import evaltools as ET
    p = truth_path(img_path, 'full')
    if not os.path.exists(p) and os.path.exists(p + '.new'):
        p += '.new'
    with open(p, encoding='utf-8') as f:
        rows = json.load(f)
    with open(corr_path, encoding='utf-8') as f:
        corr = json.load(f)
    items = ET.load_items(PRICES)
    cats = _categories()
    for k, c in corr.items():
        r = rows[int(k)]
        if 'name' in c or 'item_id' in c:
            q = c.get('item_id') or c['name']
            if q in ('', '?', 'none'):
                r['item_id'], r['name'] = '', c.get('label', '')
            else:
                hits = [items[q]] if q in items else [it for it in items.values() if it['name'].lower() == q.lower()]
                if not hits:
                    hits = ET.find_items(items, q)
                if len(hits) > 1 and len({h['name'] for h in hits}) == 1:
                    hits = hits[:1]        # twin ids with an identical name (tarkov.dev lists some items twice)
                if len(hits) != 1:
                    raise SystemExit(f'#{k}: "{q}" matched {len(hits)} items: {[h["name"] for h in hits]}')
                r['item_id'], r['name'] = hits[0]['id'], hits[0]['name']
                r['category'] = cats.get(hits[0]['id'], 'other')
        for key in ('uncertain', 'note', 'category', 'W', 'H', 'rotated'):
            if key in c:
                r[key] = c[key]
        r.pop('pred_conf', None)
    out = truth_path(img_path, 'full')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'wrote {out} ({len(rows)} rows, {sum(1 for r in rows if r.get("uncertain"))} uncertain)')


# ---------------------------------------------------------------------------
# legacy modes (--label, dump): unchanged behaviour, lazy app import
# ---------------------------------------------------------------------------

def _legacy_api():
    import app as A
    return A


def legacy_label_mode(img_path):
    A = _legacy_api()
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: cannot load '{img_path}'")
        sys.exit(1)
    icon_db = A.load_icon_db()
    settings = dict(A.load_json(A.SETTINGS_PATH, A.default_settings))
    settings.pop('grid', None)
    panels, _ = A.resolve_panels(img_bgr, settings, persist_fn=None)
    mdb = A.get_db_at_pitch(icon_db, *A._slot_px(panels[0]))
    prices = A.load_json(A.PRICES_PATH, lambda: {})
    lm = A.build_label_matcher(prices) if A.tesseract_available() and prices.get('items') else None
    detections = A.scan_all_panels(img_bgr, panels, mdb, label_matcher=lm)
    detections = sorted(detections, key=lambda d: (d.get('panel', 0), d['row'], d['col']))
    truth = [{'panel': d.get('panel', 0), 'col': d['col'], 'row': d['row'], 'W': d['W'], 'H': d['H'],
              'item_id': d['item_id'], 'name': d['name'], 'rotated': d.get('rotated', False)}
             for d in detections]
    tp = truth_path(img_path, 'orig')
    if os.path.exists(tp):
        tp += '.new'
    with open(tp, 'w', encoding='utf-8') as f:
        json.dump(truth, f, indent=2, ensure_ascii=False)
    print(f'Wrote {len(truth)} legacy guesses -> {tp}  (prefer --prefill: the v2 tool also labels the whole stash)')


def dump_mode(argv):
    img_path = argv[0] if argv else os.path.join(EVAL_DIR, 'stash1.png')
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: cannot load '{img_path}'")
        sys.exit(1)
    dets, dt, res = run_v2(img)
    print(f'Image {img_path}  {img.shape[1]}x{img.shape[0]}  scan {dt:.2f}s  {res.timings}')
    for d in res.detections:
        print(f"{d.panel:>2} ({d.col:>2},{d.row:>2}) {d.w}x{d.h} {'R' if d.rotated else ' '} "
              f"{d.confidence:.2f}{'?' if d.uncertain else ' '} fir={d.fir!s:<5} n={d.count!s:<5} {d.name}")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--score', nargs='+', metavar='IMG')
    ap.add_argument('--engine', default='v2', choices=['v2', 'legacy', 'both'])
    ap.add_argument('--truth', default='full', help="'full' (default), 'orig' or a path")
    ap.add_argument('--no-dino', action='store_true')
    ap.add_argument('--no-ocr', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--pitch-hint', type=float, help='px per slot, for --prefill of lattices the engine mis-locks')
    ap.add_argument('--robustness', metavar='IMG')
    ap.add_argument('--kinds', help='comma separated variants for --robustness')
    ap.add_argument('--prefill', metavar='IMG')
    ap.add_argument('--relabel', nargs=2, metavar=('IMG', 'CORRECTIONS.json'))
    ap.add_argument('--label', metavar='IMG')
    ap.add_argument('rest', nargs='*')
    a = ap.parse_args()
    if a.pitch_hint:
        globals()["PITCH_HINT"] = a.pitch_hint
    if a.score:
        paths = []
        for p in a.score:
            paths.extend(sorted(glob.glob(p)) if any(c in p for c in '*?[') else [p])
        score_mode(paths, a.engine, a.truth, not a.no_dino, not a.no_ocr, a.json)
    elif a.robustness:
        robustness_mode(a.robustness, a.truth, a.kinds.split(',') if a.kinds else None, a.json)
    elif a.prefill:
        prefill_mode(a.prefill, not a.no_dino)
    elif a.relabel:
        relabel_mode(*a.relabel)
    elif a.label:
        legacy_label_mode(a.label)
    else:
        dump_mode(a.rest)


if __name__ == '__main__':
    main()
