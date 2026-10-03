"""Scene layer: windows (red close button), stash toolbar, slot headers, panel splitting.

The real-frame tests use the user's 3 lobby Character > GEAR frames (2560x1440), copied into the
gitignored ``data/frames/f1..f3.png`` (or read from ``data/debug/scan-*/frame.png``); they are
skipped when absent.  f1 = "Available" item picker open, f2 = "SICC" container window open,
f3 = Messenger (trader chat) covering part of the stash."""
import glob
import os

import cv2
import numpy as np
import pytest

from identify import scene as S
from identify.grid import GridResult, Panel, detect_grid

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


def _analysis(i):
    if i not in _cache:
        im = cv2.imread(PATHS[i])
        sc = S.analyze_scene(im, in_raid=False)
        g = detect_grid(im)
        _cache[i] = (im, sc, S.split_panels(g, sc, im))
    return _cache[i]


def _overlap(a, b):
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, ix) * max(0, iy)


def _pbox(p):
    return (p.x0 + 2, p.y0 + 2, p.x1 - p.x0 - 4, p.y1 - p.y0 - 4)     # without the closing border lines


# --------------------------------------------------------------------------- real frames
@real
@pytest.mark.parametrize('i', [0, 1, 2])
def test_one_stash_region_ten_columns(i):
    im, sc, pieces = _analysis(i)
    assert sc.scene == 'lobby_gear'
    stash = [r for r in sc.regions if r.role == 'stash']
    assert len(stash) == 1
    pitch = sc.pitch
    assert abs(stash[0].bbox[2] - 10 * pitch) < 0.1 * pitch + 12
    sp = [p for p, r in pieces if r.role == 'stash']
    assert sp
    cols = {round((x - stash[0].bbox[0] - 8) / pitch) for p in sp for x in p.xs[:-1]}
    assert cols <= set(range(10)) and 0 in cols
    assert max(p.n_cols for p in sp) <= 10
    assert all(p.x0 >= stash[0].bbox[0] for p in sp)                 # nothing left of the toolbar
    assert all(r.side == 'stash' for p, r in pieces if r.role == 'stash')


@real
def test_picker_found_and_excluded():
    im, sc, pieces = _analysis(0)
    pk = [w for w in sc.windows if w.role == 'picker']
    assert len(pk) == 1 and (pk[0].title or '').lower().startswith('available')
    for p, r in pieces:
        assert r.role != 'picker'
        assert _overlap(_pbox(p), pk[0].bbox) == 0                # at most a sliver of border
    # the stash cells that sit under the picker are not scanned
    sp = [p for p, r in pieces if r.role == 'stash']
    assert not any(_overlap(_pbox(p), pk[0].bbox) > 0 and p.x0 < pk[0].bbox[0] + pk[0].bbox[2] - 8 and
                   p.y0 < pk[0].bbox[1] + pk[0].bbox[3] - 8 and p.y1 > pk[0].bbox[1] + 8 for p in sp)


@real
def test_container_window_with_title():
    im, sc, pieces = _analysis(1)
    cw = [w for w in sc.windows if w.role == 'container_window']
    assert len(cw) == 1 and (cw[0].title or '').upper().replace(' ', '') == 'SICC'
    assert cw[0].side == 'stash'
    cp = [(p, r) for p, r in pieces if r.role == 'container_window']
    assert cp and sum(p.n_cols * p.n_rows for p, _ in cp) == 10         # the 5x2 content of the pouch
    for p, _ in cp:
        assert _overlap(_pbox(p), cw[0].bbox) == (p.x1 - p.x0 - 4) * (p.y1 - p.y0 - 4)


@real
def test_own_grids_labelled_own():
    _, sc1, _ = _analysis(0)
    roles1 = {r.role: r for r in sc1.regions}
    assert roles1['own_rig'].side == 'own' and roles1['own_backpack'].side == 'own'
    rig, bp = roles1['own_rig'].bbox, roles1['own_backpack'].bbox
    assert 1000 < rig[0] < 1100 and 200 < rig[1] < 260 and rig[2] > 400          # six rig slots
    assert 1000 < bp[0] < 1100 and 940 < bp[1] < 1000 and bp[3] > 250            # backpack grid
    _, sc2, pieces2 = _analysis(1)
    assert 'own_pouch' in {r.role for r in sc2.regions} and 'own_pockets' in {r.role for r in sc2.regions}
    pouch = [p for p, r in pieces2 if r.role == 'own_pouch']
    assert pouch and pouch[0].n_cols == 3 and pouch[0].n_rows == 3


@real
def test_messenger_occludes_stash():
    im, sc, pieces = _analysis(2)
    win = [w for w in sc.windows if w.role == 'unknown']
    assert len(win) == 1 and 'ragman' in (win[0].title or '').lower()
    sp = [p for p, r in pieces if r.role == 'stash']
    assert sp and len(sp) >= 3
    for p in sp:
        assert _overlap(_pbox(p), win[0].bbox) == 0
    assert min(p.x0 for p in sp) == 1687 or min(p.x0 for p in sp) > 1600


@real
def test_scene_is_cheap_once_headers_are_cached():
    import time
    im = cv2.imread(PATHS[1])
    S.analyze_scene(im, in_raid=False)
    t = time.perf_counter()
    S.analyze_scene(im, in_raid=False)
    assert time.perf_counter() - t < 0.25


# --------------------------------------------------------------------------- synthetic
def _synthetic_window(H=1080, W=1920, btn=(900, 300, 40, 25), rect=(500, 290, 450, 300)):
    img = np.full((H, W, 3), 60, np.uint8)
    img[::7] = 75
    x, y, w, h = rect
    img[y:y + h, x:x + w] = (22, 22, 22)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (96, 93, 88), 1)
    img[y + 1:y + 38, x + 1:x + w - 1] = (36, 35, 34)                      # title bar
    bx, by, bw, bh = btn
    img[by:by + bh, bx:bx + bw] = (14, 14, 68)
    cv2.line(img, (bx + 12, by + 6), (bx + bw - 12, by + bh - 6), (230, 230, 230), 2)
    cv2.line(img, (bx + bw - 12, by + 6), (bx + 12, by + bh - 6), (230, 230, 230), 2)
    return img


def test_close_button_and_window_trace():
    img = _synthetic_window()
    btns = S.find_close_buttons(img)
    assert len(btns) == 1 and all(abs(a - b) <= 2 for a, b in zip(btns[0], (900, 300, 40, 25)))
    x, y, w, h = S.trace_window(img, btns[0])
    assert abs(x - 500) <= 2 and abs(y - 290) <= 2 and abs(x + w - 950) <= 3 and abs(y + h - 590) <= 3


def test_red_without_glyph_or_wrong_shape_is_not_a_button():
    img = _synthetic_window()
    img[300:325, 900:940] = (14, 14, 68)                                   # no white X
    assert S.find_close_buttons(img) == []
    img2 = np.full((1080, 1920, 3), 60, np.uint8)
    img2[100:160, 100:120] = (14, 14, 68)                                  # tall red bar
    assert S.find_close_buttons(img2) == []
    img3 = _synthetic_window()
    img3[300:325, 900:940] = (120, 120, 120)                               # grey button
    assert S.find_close_buttons(img3) == []


def test_two_windows_scale_with_resolution():
    img = _synthetic_window(H=1440, W=2560, btn=(1200, 400, 54, 33), rect=(660, 385, 600, 400))
    assert len(S.find_close_buttons(img)) == 1
    tiny = _synthetic_window(H=720, W=1280, btn=(600, 200, 27, 17), rect=(330, 192, 300, 200))
    assert len(S.find_close_buttons(tiny)) == 1


def test_rects_from_mask_covers_cells_exactly():
    m = np.ones((6, 10), bool)
    m[2:5, 0:4] = False                                                    # a window in the corner
    rects = S._rects_from_mask(m)
    cover = np.zeros_like(m)
    for c0, c1, r0, r1 in rects:
        assert not cover[r0:r1 + 1, c0:c1 + 1].any()
        cover[r0:r1 + 1, c0:c1 + 1] = True
    assert (cover == m).all() and len(rects) == 3


def test_split_drops_cells_under_a_window():
    pitch = 84.0
    xs = [int(100 + k * pitch) for k in range(7)]
    ys = [int(100 + k * pitch) for k in range(5)]
    p = Panel(x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1, ox=xs[0], oy=ys[0], pitch_x=pitch, pitch_y=pitch,
              n_cols=6, n_rows=4, xs=xs, ys=ys)
    sc = S.SceneInfo(scene='lobby_gear', in_raid=False, frame_wh=(1920, 1080), pitch=pitch, toolbar=(20, 40))
    sc.windows.append(S.Region((100, 184, 3 * 84 - 6, 78), 'picker', 'other', 'Available', 0.95))
    out = S.split_panels(GridResult([p], pitch, pitch), sc, None)
    assert all(r.role == 'stash' for _, r in out)
    cells = sum(q.n_cols * q.n_rows for q, _ in out)
    assert cells == 24 - 3                                                 # exactly the 3 covered cells go
    for q, _ in out:
        assert _overlap(_pbox(q), sc.windows[0].bbox) == 0
