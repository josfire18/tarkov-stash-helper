"""
Ground-truth tooling: prefill a ``<name>.truth.full.json`` from the pipeline and render
contact sheets (crop | predicted icon | name) to check it by eye.

Workflow (see ``data/eval/README.txt``)::

    python test_scan.py --prefill data/eval/shot.png        # writes the draft + sheets
    # open data/eval/shot.sheet_01.png ... ; fix wrong rows with --relabel
    python test_scan.py --relabel data/eval/shot.png 12="Morphine injector" 15=?
    python test_scan.py --score data/eval/shot.png --truth full

Truth row fields::

    panel, col, row, W, H         footprint in cells (row 0 = first *visible* row)
    rect [x, y, w, h]             pixel rectangle (used for IoU matching, so the truth
                                  survives a change of grid origin convention)
    item_id, name                 verified identity ('' when the item is not in the catalog)
    category                      ammo | weapon | mod | meds | container | barter | keys | ...
    uncertain                     true when the crop alone does not settle the identity
                                  (excluded from ID accuracy, still counted for segmentation)
    clipped                       footprint cut by the viewport
    note                          free text (how it was verified)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .catalog import Catalog
from .dino import TINT_BGR, composite_icon


def _font(size: int):
    for p in ('C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def truth_from_detections(dets, items) -> list[dict]:
    """Draft truth rows for every segmented footprint (empty cells excluded)."""
    rows = []
    det_by_rect = {tuple(d.rect): d for d in dets}
    for it in items:
        if it.empty:
            continue
        d = det_by_rect.get(tuple(it.rect))
        rows.append({
            'panel': it.panel, 'col': it.col, 'row': it.row, 'W': it.w, 'H': it.h,
            'rect': list(it.rect), 'clipped': bool(it.clipped_top or it.clipped_bottom),
            'item_id': d.item_id if d else '', 'name': d.name if d else '',
            'category': d.category if d else '', 'uncertain': False if d else True,
            'rotated': bool(d.rotated) if d else False,
            'note': 'prefill (unverified)',
            'pred_conf': round(d.confidence, 3) if d else 0.0,
        })
    return rows


def contact_sheets(img_bgr: np.ndarray, rows: list[dict], dets, cat: Catalog, out_prefix: str,
                   per_sheet: int = 14) -> list[str]:
    """Render ``<out_prefix>_01.png`` ... : index | crop | predicted icon | text."""
    det_by_rect = {tuple(d.rect): d for d in dets}
    f_small, f_big = _font(13), _font(15)
    paths = []
    cell_w, cell_h = 640, 150
    for si in range(0, len(rows), per_sheet):
        chunk = rows[si:si + per_sheet]
        n_rows = (len(chunk) + 1) // 2
        sheet = Image.new('RGB', (cell_w * 2, cell_h * n_rows), (24, 24, 28))
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            idx = si + k
            ox, oy = (k % 2) * cell_w, (k // 2) * cell_h
            x, y, w, h = r['rect']
            crop = img_bgr[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            sc = min(2.0, 130.0 / crop.shape[0], 250.0 / crop.shape[1])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
            sheet.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (ox + 44, oy + 8))
            # predicted icon
            ex = ox + 44 + crop.shape[1] + 10
            d = det_by_rect.get(tuple(r['rect']))
            row = d.evidence.get('catalog_row') if d else None
            rot = d.evidence.get('rot', 0) if d else 0
            if row is not None and r.get('item_id') == (d.item_id if d else None):
                full = cat.full_icon(int(row))
                if full is not None:
                    bg = TINT_BGR.get(str(cat.tint[int(row)]), TINT_BGR['default'])
                    comp = composite_icon(full, bg, rot)
                    s2 = min(130.0 / comp.shape[0], 250.0 / comp.shape[1], 2.0)
                    comp = cv2.resize(comp, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
                    sheet.paste(Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)), (ex, oy + 8))
                    ex += comp.shape[1] + 10
            tx = max(ex, ox + 44 + 240)
            dr.text((ox + 4, oy + 8), f'{idx}', fill=(255, 220, 90), font=f_big)
            lines = [f"({r['col']},{r['row']}) {r['W']}x{r['H']}"
                     + (' ROT' if r.get('rotated') else '') + (' CLIP' if r.get('clipped') else ''),
                     (r.get('name') or '?')[:44]]
            if d:
                ev = d.evidence
                lines += [f"conf {d.confidence:.2f}{' UNCERTAIN' if d.uncertain else ''}",
                          f"ocr '{ev.get('ocr_text', '')[:22]}' {ev.get('ocr')}",
                          f"res {ev.get('residual')} dino {ev.get('dino')}",
                          f"src {ev.get('source')}  fir {d.fir}  n {d.count}"]
            for li, t in enumerate(lines):
                dr.text((tx, oy + 8 + li * 18), t, fill=(235, 235, 235) if li != 1 else (120, 230, 140), font=f_small)
        path = f'{out_prefix}_{si // per_sheet + 1:02d}.png'
        sheet.save(path)
        paths.append(path)
    return paths


def load_items(prices_path: str) -> dict:
    with open(prices_path, encoding='utf-8') as f:
        return {it['id']: it for it in json.load(f)['items']}


def find_items(items: dict, query: str, limit: int = 8) -> list[dict]:
    q = query.lower()
    out = [it for it in items.values()
           if q == it['id'] or q in it['name'].lower() or q == (it.get('shortName') or '').lower()]
    return out[:limit]
