"""
Truth for the user's OWN frames (data/scenes/live, 1440p), built from two independent sources:

  (a) the exact oracle - a tile whose pixels match a render of the game's icon cache (the hash maps
      renders to item ids) near pixel-perfectly, with a clear gap to the runner-up, IS that item.
      Needs no picture comparison against tarkov.dev art and no OCR; it only shares the grid /
      segmentation with the live path.
  (b) a human reading the printed label on a contact sheet.  Those decisions live in
      ``data/eval/user/<stem>.manual.json`` ({"<row index>": "<item name | id | ?>"}) and are merged
      on every run, so the oracle part can be regenerated at any time.

    python scripts/build_user_truth.py FRAME.png [FRAME2.png ...]      # frames from data/scenes/live
        -> data/eval/user/<stem>.png (copy), <stem>.truth.full.json, <stem>.sheet_NN.png (tiles the
           oracle could not name; look at them, then write <stem>.manual.json and run again)

Rows follow identify/evaltools.py (panel, col, row, W, H, rect, item_id, name, category, uncertain,
clipped, note); extra fields: role (stash / own_* / loot / ...), src ('oracle' | 'manual').
Nothing here is ever committed: data/ is gitignored (private stash screenshots).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ORACLE_MAX = 1.5            # aligned residual of a pixel-exact render (the pipeline's own bound is 3.0 / 4.5)
ORACLE_RUNNER_X = 2.5       # runner-up (any other item's render) must be this x worse, and > 3.0
OUT_DIR = os.path.join(ROOT, 'data', 'eval', 'user')
SRC_DIR = os.environ.get('TSH_LIVE_FRAMES') or os.path.join(ROOT, 'data', 'scenes', 'live')


def load_engine():
    import app
    return app, app._v2_engine_for(app._engine_settings(app._load_settings()))


def oracle(engine, img, it, panel):
    """-> (item_id, residual, runner) or None: the tile is exactly a render of a named item."""
    idx = engine.anchor_idx
    if idx is None or not len(idx) or it.empty:
        return None
    if it.clipped_top and it.clipped_bottom:
        return None
    x, y, w, h = it.rect
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return None
    crop = img[y:y + h, x:x + w]
    px, py = float(panel.pitch_x), float(panel.pitch_y)
    clip = 'top' if (it.clipped_top and not it.clipped_bottom) else 'bottom' if it.clipped_bottom else ''
    a = idx.match_clipped(crop, int(it.w), clip, px, py, k=16) if clip else idx.match(crop, int(it.w), int(it.h), px, py, k=16)
    if a is None or not a.item_id or not a.certain:
        return None
    if a.score > ORACLE_MAX or a.runner <= max(3.0, ORACLE_RUNNER_X * a.score):
        return None
    return a.item_id, float(a.score), float(a.runner)


def expand_manual(m: dict) -> dict:
    """{"3-9": "=", "12": "M4A1"} -> one key per row index ('=' = accept the live path's answer by eye)."""
    out = {}
    for k, v in m.items():
        a, _, b = str(k).partition('-')
        for i in range(int(a), int(b or a) + 1):
            out[str(i)] = v
    return out


def build(frame, items_db, by_name):
    from identify.scene import analyze_scene
    app, engine = load_engine()
    path = frame if os.path.isabs(frame) else os.path.join(SRC_DIR, frame)
    img = cv2.imread(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copyfile(path, os.path.join(OUT_DIR, stem + '.png'))
    scene = analyze_scene(img, in_raid=False)
    res = engine.scan(img, scene)
    dets = {tuple(d.rect): d for d in res.detections}
    manual_p = os.path.join(OUT_DIR, stem + '.manual.json')
    manual = expand_manual(json.load(open(manual_p, encoding='utf-8'))) if os.path.exists(manual_p) else {}
    rows, pending = [], []
    for it in sorted((i for i in res.items if not i.empty), key=lambda i: (i.panel, i.row, i.col)):
        panel = res.grid.panels[it.panel]
        reg = res.regions[it.panel] if it.panel < len(res.regions) else None
        d = dets.get(tuple(it.rect))
        row = {'panel': it.panel, 'col': it.col, 'row': it.row, 'W': it.w, 'H': it.h, 'rect': list(it.rect),
               'clipped': bool(it.clipped_top or it.clipped_bottom), 'role': reg.role if reg else '',
               'item_id': '', 'name': '', 'category': d.category if d else '', 'uncertain': True,
               'rotated': bool(d.rotated) if d else False, 'src': '', 'note': ''}
        o = oracle(engine, img, it, panel)
        key = str(len(rows))
        if o:
            iid, s, r = o
            row.update(item_id=iid, name=(items_db.get(iid) or {}).get('name', ''), uncertain=False, src='oracle',
                       note=f'exact render {s:.2f} vs runner {r:.1f}')
        elif key in manual and manual[key] not in ('', '?'):
            m = manual[key]
            if m == '=':                                   # "the live path's answer is right": freeze its id now
                m = manual[key] = (d.item_id if d else '')
            iid = m if m in items_db else (by_name.get(m.lower()) or '')
            row.update(item_id=iid, name=(items_db.get(iid) or {}).get('name', m), uncertain=not iid,
                       src='manual', note='read from the contact sheet by eye')
        elif key in manual:
            row['note'] = 'unsure: a human could not tell'
        else:
            pending.append((len(rows), row, d))
        if o and key in manual and manual[key] not in ('', '?', '='):
            m = manual[key]
            iid = m if m in items_db else (by_name.get(m.lower()) or '')
            row['note'] += f'; manual says {m}' + ('' if iid == row['item_id'] else '  <-- DISAGREES')
        rows.append(row)
    if manual:
        json.dump(manual, open(manual_p, 'w', encoding='utf-8'), indent=0)
    json.dump(rows, open(os.path.join(OUT_DIR, stem + '.truth.full.json'), 'w', encoding='utf-8'), indent=1)
    n_or = sum(r['src'] == 'oracle' for r in rows)
    n_man = sum(r['src'] == 'manual' for r in rows)
    print(f'{stem}: {len(rows)} tiles, oracle {n_or}, manual {n_man}, still to look at {len(pending)}')
    for old in glob.glob(os.path.join(OUT_DIR, stem + '.sheet_*.png')):
        os.remove(old)
    if pending:
        sheets(img, pending, engine, os.path.join(OUT_DIR, stem + '.sheet'))
    return rows


def sheets(img, pending, engine, prefix, per=20):
    from identify.dino import TINT_BGR, composite_icon
    cell_w, cell_h = 560, 118
    font = cv2.FONT_HERSHEY_SIMPLEX
    for si in range(0, len(pending), per):
        chunk = pending[si:si + per]
        sheet = np.full((cell_h * ((len(chunk) + 1) // 2), cell_w * 2, 3), 24, np.uint8)
        for k, (idx, row, d) in enumerate(chunk):
            ox, oy = (k % 2) * cell_w, (k // 2) * cell_h
            x, y, w, h = row['rect']
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            sc = min(3.0, 110.0 / crop.shape[0], 150.0 / crop.shape[1])
            c = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            sheet[oy + 4:oy + 4 + c.shape[0], ox + 40:ox + 40 + c.shape[1]] = c
            ex = ox + 40 + c.shape[1] + 8
            cat = engine.cat
            r_ = d.evidence.get('catalog_row') if d else None
            if r_ is not None:
                full = cat.full_icon(int(r_))
                if full is not None:
                    comp = composite_icon(full, TINT_BGR.get(str(cat.tint[int(r_)]), TINT_BGR['default']),
                                          d.evidence.get('rot', 0))
                    s2 = min(110.0 / comp.shape[0], 150.0 / comp.shape[1], 2.0)
                    comp = cv2.resize(comp, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
                    ee = min(ex + comp.shape[1], ox + cell_w)
                    sheet[oy + 4:oy + 4 + comp.shape[0], ex:ee] = comp[:, :ee - ex]
                    ex += comp.shape[1] + 8
            cv2.putText(sheet, str(idx), (ox + 2, oy + 16), font, 0.5, (90, 220, 255), 1)
            lines = [f"{row['W']}x{row['H']} {row['role']}" + (' CLIP' if row['clipped'] else ''),
                     (d.name if d else '?')[:34]]
            if d:
                alts = [a['name'] for a in (d.evidence.get('alternatives') or ())][:3]
                lines += [f"conf {d.confidence:.2f}{' CHECK' if d.uncertain else ''} ocr '{str(d.evidence.get('ocr_text', ''))[:16]}'"]
                lines += [('or ' + ' / '.join(alts))[:44]] if alts else []
            for li, t in enumerate(lines):
                cv2.putText(sheet, t, (max(ex, ox + 200), oy + 18 + li * 17), font, 0.42,
                            (140, 235, 150) if li == 1 else (225, 225, 225), 1)
        cv2.imwrite(f'{prefix}_{si // per + 1:02d}.png', sheet)


def main(argv):
    from identify.evaltools import load_items
    prices = os.path.join(ROOT, 'data', 'prices_cache.json')
    items_db = load_items(prices)
    by_name = {}
    for i in items_db.values():
        by_name.setdefault(i['name'].lower(), i['id'])
        by_name.setdefault((i.get('shortName') or '').lower(), i['id'])
    for f in argv:
        build(f, items_db, by_name)


if __name__ == '__main__':
    main(sys.argv[1:])
