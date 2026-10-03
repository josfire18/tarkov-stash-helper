"""Own-gear grids (tactical rig slots with gaps, pockets, backpack, pouch): cell groups found by
identify.owngrid, wired through scene.split_panels, counted free by liveadvice.

Real-frame tests use the 3 lobby Character > GEAR frames (2560x1440) in the gitignored
``data/frames/f1..f3.png`` (or ``data/debug/scan-*/frame.png``) and are skipped when absent:
f1 = tactical rig (2 magazines + 3 empty slots) + empty backpack, "Available" picker open;
f2 = pouch 3x3 (Gamma) + pockets partly behind the SICC window; f3 = four pockets."""
import glob
import os

import cv2
import numpy as np
import pytest

import liveadvice
from identify import scene as S
from identify.grid import GridResult, detect_grid
from identify.owngrid import region_panels
from identify.segment import segment_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frame_paths():
    p = [os.path.join(ROOT, 'data', 'frames', f'f{i}.png') for i in (1, 2, 3)]
    if all(os.path.exists(x) for x in p):
        return p
    p = sorted(glob.glob(os.path.join(ROOT, 'data', 'debug', 'scan-*', 'frame.png')))[:3]
    return p if len(p) == 3 else []


PATHS = _frame_paths()
real = pytest.mark.skipif(not PATHS, reason='real lobby frames not present')
_cache = {}


def _own(i):
    """{role: [(Panel, items)]} of the own-gear pieces of real frame i."""
    if i not in _cache:
        im = cv2.imread(PATHS[i])
        sc = S.analyze_scene(im, in_raid=False)
        pieces = S.split_panels(detect_grid(im), sc, im)
        out = {}
        for pi, (p, r) in enumerate(pieces):
            if r.role.startswith('own'):
                out.setdefault(r.role, []).append((p, segment_panel(im, p, pi)))
        _cache[i] = (im, sc, pieces, out)
    return _cache[i]


def _free(items):
    return sum(it.w * it.h for it in items if it.empty)


# --------------------------------------------------------------------------- real frames
@real
def test_rig_slots_with_gaps_on_frame_1():
    im, sc, pieces, own = _own(0)
    rig = own['own_rig']
    assert len(rig) == 5                                           # five slots, 9 px apart
    assert all(p.n_cols == 1 and p.n_rows == 2 for p, _ in rig)
    full = [its for _, its in rig if any(not it.empty for it in its)]
    assert len(full) == 2                                          # STANAG + GEN M3
    assert sum(_free(its) for _, its in rig) == 6                  # 3 empty 1x2 slots
    gaps = [rig[i + 1][0].x0 - rig[i][0].x1 for i in range(4)]
    assert all(5 <= g <= 14 for g in gaps)


@real
def test_backpack_grid_empty_on_frame_1():
    im, sc, pieces, own = _own(0)
    (p, items), = own['own_backpack']
    assert p.n_cols == 5 and p.n_rows == 4 and p.clip_bottom      # last row cut by the viewport
    assert all(it.empty and it.w == it.h == 1 for it in items)
    assert _free(items) == 20


@real
def test_free_cells_of_frame_1_for_the_drop_advice():
    im, sc, pieces, own = _own(0)
    grids = []
    for role, lst in own.items():
        for pi, (p, items) in enumerate(lst):
            grids.append({'panel': pi, 'role': role, 'cols': p.n_cols, 'rows': p.n_rows,
                          'cells': [(it.col, it.row, it.w, it.h, bool(it.empty)) for it in items]})
    assert liveadvice.free_cells(grids) == 26


@real
def test_pouch_3x3_on_frame_2():
    im, sc, pieces, own = _own(1)
    (p, items), = own['own_pouch']
    assert p.n_cols == 3 and p.n_rows == 3
    assert p.clip_top                                              # the SICC window hides the top
    assert sum(1 for it in items if not it.empty) == 4             # case, SICC, injector case, vaseline
    assert _free(items) == 3


@real
def test_pockets_four_cells_on_frame_3():
    im, sc, pieces, own = _own(2)
    pk = own['own_pockets']
    assert len(pk) == 4 and all(p.n_cols == p.n_rows == 1 for p, _ in pk)
    assert all(its[0].empty for _, its in pk)


@real
def test_occluded_pockets_are_dropped_on_frame_2():
    im, sc, pieces, own = _own(1)
    assert len(own['own_pockets']) == 2                            # cells 3 / 4 are under the window


@real
def test_stash_pieces_unchanged_by_own_grids():
    im, sc, pieces, own = _own(0)
    stash = [p for p, r in pieces if r.role == 'stash']
    assert stash and all(p.n_cols in (8, 10) for p in stash)


# --------------------------------------------------------------------------- synthetic
PITCH = 63
BORDER = (97, 98, 81)           # BGR, a rig slot's drawn border


def _canvas(w=900, h=420):
    img = np.full((h, w, 3), 14, np.uint8)
    return img


def _box(img, x, y, nc, nr, pitch=PITCH, col=BORDER, inner=False):
    cv2.rectangle(img, (x, y), (x + nc * pitch, y + nr * pitch), col, 1)
    if inner:
        for i in range(1, nc):
            cv2.line(img, (x + i * pitch, y), (x + i * pitch, y + nr * pitch), (35, 35, 35), 1)
        for j in range(1, nr):
            cv2.line(img, (x, y + j * pitch), (x + nc * pitch, y + j * pitch), (35, 35, 35), 1)


def test_gapped_rig_slots_become_one_panel_each():
    img = _canvas()
    xs = [100 + i * (PITCH + 9) for i in range(5)]
    for x in xs:
        _box(img, x, 100, 1, 2)
    ps = region_panels(img, (95, 95, 5 * (PITCH + 9), 2 * PITCH + 12), PITCH)
    assert [(p.x0, p.n_cols, p.n_rows) for p in ps] == [(x, 1, 2) for x in xs]
    assert all(not p.clip_top and not p.clip_bottom for p in ps)


def test_wide_slot_snaps_to_two_columns():
    img = _canvas()
    _box(img, 100, 100, 2, 2)
    _box(img, 100 + 2 * PITCH + 9, 100, 1, 2)
    ps = region_panels(img, (95, 95, 3 * PITCH + 30, 2 * PITCH + 12), PITCH)
    assert [(p.n_cols, p.n_rows) for p in ps] == [(2, 2), (1, 2)]


def test_group_that_is_not_a_whole_number_of_cells_is_rejected():
    img = _canvas()
    cv2.rectangle(img, (100, 100), (100 + int(1.5 * PITCH), 100 + PITCH), BORDER, 1)
    assert region_panels(img, (95, 95, 140, 90), PITCH) == []


def test_faint_inner_lines_and_viewport_clip():
    img = _canvas()
    _box(img, 100, 100, 4, 3, inner=True)
    img[100 + 2 * PITCH + 30:, :] = 14               # the viewport cuts the last row (no bottom frame line)
    ps = region_panels(img, (95, 95, 4 * PITCH + 12, 3 * PITCH + 12), PITCH)
    assert len(ps) == 1
    p = ps[0]
    assert p.n_cols == 4 and p.n_rows == 3 and p.clip_bottom
    assert [x - p.x0 for x in p.xs] == [i * PITCH for i in range(5)]


def test_window_overlap_drops_the_cell_and_top_clip_is_flagged():
    img = _canvas()
    for x in (100, 100 + PITCH + 9):
        _box(img, x, 100, 1, 1)
    win = (168, 60, 300, 90)                         # covers the second cell, right of the first
    ps = region_panels(img, (95, 95, 2 * PITCH + 20, PITCH + 12), PITCH, [win])
    assert [p.x0 for p in ps] == [100]
    img2 = _canvas()
    _box(img2, 100, 100, 3, 3, inner=True)
    img2[100:130, 150:] = 60                         # a window sitting on the top rows (right part)
    win2 = (150, 90, 400, 40)
    ps = region_panels(img2, (95, 95, 3 * PITCH + 12, 3 * PITCH + 12), PITCH, [win2])
    assert len(ps) == 1 and ps[0].clip_top and ps[0].n_cols == 3


def test_split_panels_adds_own_pieces_with_their_role():
    img = _canvas()
    for i in range(3):
        _box(img, 100 + i * (PITCH + 9), 100, 1, 2)
    sc = S.SceneInfo(scene='lobby_gear', in_raid=False, pitch=float(PITCH), frame_wh=(900, 420))
    sc.regions.append(S.Region((95, 95, 3 * (PITCH + 9), 2 * PITCH + 12), 'own_rig', 'own', 'TACTICAL RIG', 0.85))
    pieces = S.split_panels(GridResult([], None, None), sc, img)
    assert [r.role for _, r in pieces] == ['own_rig'] * 3
    assert all(p.own and r.side == 'own' and r.title == 'TACTICAL RIG' for p, r in pieces)


def test_own_panel_splits_empty_cells_one_by_one():
    img = _canvas()
    _box(img, 100, 100, 3, 2, inner=True)
    ps = region_panels(img, (95, 95, 3 * PITCH + 12, 2 * PITCH + 12), PITCH)
    ps[0].own = True
    items = segment_panel(img, ps[0], 0)
    assert len(items) == 6 and all(it.empty and it.w == it.h == 1 for it in items)


def test_free_cells_counts_area_not_footprints():
    g = [{'role': 'own_rig', 'cells': [(0, 0, 1, 2, True), (0, 0, 2, 2, False)], 'rows': 2, 'cols': 2, 'panel': 0}]
    assert liveadvice.free_cells(g) == 2
