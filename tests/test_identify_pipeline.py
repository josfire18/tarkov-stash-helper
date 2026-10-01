"""Stage 1 + pipeline on a synthetic stash with a synthetic catalog (no game data, no OCR/DINO)."""
import json
import os

import cv2
import numpy as np
import pytest

from identify import catalog as C
from identify.config import EngineSettings
from identify.fir import detect_fir
from identify.match import normalize_tile, stage1
from identify.pipeline import Engine, Detection
from synth import make_icon, render_panel

# (id, name, W, H)
ITEMS = ([(f'{i:024x}', f'Item {i}', 1, 1) for i in range(1, 9)]
         + [(f'{i:024x}', f'Long {i}', 2, 1) for i in range(20, 24)]
         + [(f'{i:024x}', f'Tall {i}', 1, 3) for i in range(30, 32)]
         + [(f'{i:024x}', f'Big {i}', 3, 2) for i in range(40, 42)])


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('world')
    tmpl = tmp / 'tmpl'
    tmpl.mkdir()
    prices, icons = [], {}
    for k, (iid, name, w, h) in enumerate(ITEMS):
        ic = make_icon(500 + k, w, h)
        icons[iid] = ic
        cv2.imwrite(str(tmpl / f'{iid}.png'), ic)
        prices.append({'id': iid, 'name': name, 'shortName': name, 'width': w, 'height': h,
                       'backgroundColor': 'blue', 'types': ['mods']})
    pp = tmp / 'prices.json'
    pp.write_text(json.dumps({'items': prices}), encoding='utf-8')
    cat = C.build_catalog(str(pp), str(tmpl), str(tmp / 'nocache'), with_cache=False, log=lambda *_: None)
    return cat, icons


def engine(cat):
    return Engine(EngineSettings(use_dino=False, use_ocr=False, accelerate=False), catalog=cat)


def idof(i):
    return f'{i:024x}'


def test_stage1_finds_the_right_icon_over_an_unknown_background(world):
    cat, icons = world
    # tile = icon 3 over a tint the catalog has never seen, plus a bright fake label in the top band
    ic = icons[idof(3)]
    tile = np.zeros((64, 64, 3), np.uint8)
    tile[:] = (50, 70, 40)
    a = ic[:, :, 3:4] / 255.0
    tile = (ic[:, :, :3] * a + tile * (1 - a)).astype(np.uint8)
    tile[3:12, 20:60] = 235                                          # label text blob
    cv2.rectangle(tile, (0, 0), (63, 63), (84, 81, 73), 1)
    t = normalize_tile(tile, (0, 0, 64, 64), 1, 63.0, 63.0)
    c = stage1(cat, t, top_k=3)
    assert str(cat.ids[c[0].row]) == idof(3)
    assert c[0].score < c[1].score - 1.0


def test_stage1_matches_rotated_items(world):
    cat, icons = world
    ic = np.rot90(icons[idof(20)], -1)                               # 2x1 'Long 20' rotated clockwise -> 1x2
    tile = np.zeros(ic.shape[:2] + (3,), np.uint8)
    tile[:] = (45, 40, 30)
    a = ic[:, :, 3:4] / 255.0
    tile = (ic[:, :, :3] * a + tile * (1 - a)).astype(np.uint8)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (84, 81, 73), 1)
    t = normalize_tile(tile, (0, 0, tile.shape[1], tile.shape[0]), 1, 63.0, 63.0)
    assert (t.W, t.H) == (1, 2)
    best = stage1(cat, t, top_k=2)[0]
    assert str(cat.ids[best.row]) == idof(20) and best.rotated and best.rot == -1


def test_full_pipeline_on_synthetic_stash(world):
    cat, icons = world
    layout = [(0, 0, 1, 1, 1), (1, 0, 1, 1, 2), (2, 0, 1, 1, 1),            # (col,row,w,h,item) - two identical neighbours
              (3, 0, 2, 1, 20), (0, 1, 1, 3, 30), (1, 1, 3, 2, 40),
              (4, 1, 1, 1, 5)]
    cells = {(c, r): icons[idof(i)] for c, r, w, h, i in layout}
    img, rect, expected, _ = render_panel(63.0, 6, 5, [(c, r, w, h) for c, r, w, h, _ in layout],
                                          icons=cells, size=(520, 420))
    res = engine(cat).scan(img)
    by_rect = {tuple(d.rect): d for d in res.detections}
    for (c, r, w, h, i), (_, rc) in zip(layout, [e for e in expected if e[0][2:] != (1, 1) or True][:len(layout)]):
        pass
    want = {(c, r, w, h): idof(i) for c, r, w, h, i in layout}
    got = {(d.col, d.row, d.w, d.h): d.item_id for d in res.detections}
    for k, v in want.items():
        assert got.get(k) == v, (k, got.get(k), v)
    assert res.grid.panels and abs(res.grid.panels[0].pitch_x - 63) < 1
    assert 'stage1' in res.timings and res.timings['total'] > 0


def test_adjacent_identical_stacks_are_two_detections(world):
    cat, icons = world
    cells = {(0, 0): icons[idof(1)], (1, 0): icons[idof(1)]}
    img, _, _, _ = render_panel(63.0, 4, 3, [(0, 0, 1, 1), (1, 0, 1, 1)], icons=cells, size=(380, 300))
    dets = engine(cat).scan(img).detections
    ones = [d for d in dets if (d.col, d.row) in ((0, 0), (1, 0))]
    assert len(ones) == 2 and all(d.item_id == idof(1) for d in ones)


def test_clipped_bottom_row_still_identifies(world):
    cat, icons = world
    cells = {(0, 2): icons[idof(2)], (1, 2): icons[idof(3)]}
    img, _, _, _ = render_panel(63.0, 4, 3, [(0, 2, 1, 1), (1, 2, 1, 1)], icons=cells,
                                bottom_partial=0.3, size=(380, 300))
    res = engine(cat).scan(img)
    last = [d for d in res.detections if d.row == 2]
    assert {d.item_id for d in last} >= {idof(2), idof(3)}
    assert all(d.clipped for d in last)


def test_detection_has_the_legacy_result_shape(world):
    cat, icons = world
    img, _, _, _ = render_panel(63.0, 3, 2, [(0, 0, 1, 1)], icons={(0, 0): icons[idof(1)]}, size=(300, 240))
    d = engine(cat).scan(img).detections[0]
    assert isinstance(d, Detection)
    leg = d.to_legacy()
    for key in ('col', 'row', 'W', 'H', 'item_id', 'name', 'rotated', 'score', 'fir', 'panel',
                'px', 'py', 'pw', 'ph', 'uncertain'):
        assert key in leg
    assert 0 <= leg['score'] <= 100
    assert set(d.evidence) >= {'residual', 'margin', 'features', 'catalog_row'}


def test_a_stack_not_in_the_catalog_is_flagged_not_guessed(world):
    cat, icons = world
    stranger = make_icon(31337, 1, 1)
    img, _, _, _ = render_panel(63.0, 3, 2, [(0, 0, 1, 1)], icons={(0, 0): stranger}, size=(300, 240))
    d = engine(cat).scan(img).detections[0]
    assert d.uncertain


def test_fir_detector_three_valued():
    tile = np.full((64, 64, 3), 25, np.uint8)
    assert detect_fir(tile) is False                                  # flat dark corner: confidently not FiR
    assert detect_fir(None) is None and detect_fir(tile[:10]) is None
    from identify import fir as F
    g = np.clip(F._T, 0, 255).astype(np.uint8)
    for corner_y in (64 - 3 - g.shape[0], 64 - 17 - g.shape[0]):      # at the corner, or one row up above a stack count
        t2 = np.full((64, 64, 3), 25, np.uint8)
        t2[corner_y:corner_y + g.shape[0], 64 - 3 - g.shape[1]:64 - 3] = g[..., None]
        assert detect_fir(t2) is True
