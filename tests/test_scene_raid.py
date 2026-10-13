"""Scene layer on the in-raid Character > GEAR screen while looting (game 1.2.0): the player's own
grids in the middle, the looted container ("LOOT" header, name top right) in the right column.

The real-frame tests read ``data/frames_ref/raid/*.png`` (gitignored, 2560x1440; 41 frames) and are
skipped when absent.  Sorted indices: 0-6 receive-all SHOWCASE screens, 8-9 lobby stash / gear and
34-40 flea "Offer creation" were mislabelled "raid" by an old build and must stay lobby scenes;
7 and 10-33 are the true in-raid frames.  A synthetic frame covers the loot grid finder without them."""
import glob
import os
import time

import cv2
import numpy as np
import pytest

import liveadvice as adv
from identify import scene as S
from identify.grid import detect_grid
from identify.lootgrid import BORDER_BGR, analyze_loot, unrevealed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = sorted(glob.glob(os.path.join(ROOT, 'data', 'frames_ref', 'raid', '*.png')))
real = pytest.mark.skipif(len(FRAMES) < 41, reason='real raid frames not present')
RAID = [7] + list(range(10, 34))
P = 63 * 1440 / 1080.0

# (cols, rows, hidden cells) of the container grid, or the UNSEARCHED stand-in
GRIDS = {7: (2, 2, 0), 10: (2, 2, 0), 19: (2, 2, 0), 27: (2, 2, 0), 21: (4, 3, 1), 22: (3, 3, 3),
         23: (3, 3, 1), 24: (2, 2, 2), 26: (2, 2, 2), 30: (2, 2, 1), 33: (2, 2, 1)}
UNSEARCHED = [11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 28, 29, 31, 32]
_cache = {}


def _scene(i, hint=True):
    k = (i, hint)
    if k not in _cache:
        im = cv2.imread(FRAMES[i])
        sc = S.analyze_scene(im, in_raid=hint)
        _cache[k] = (im, sc)
    return _cache[k]


# --------------------------------------------------------------------------- synthetic
def _synthetic(hidden=((1, 0),), cols=3, rows=2):
    """Right column of a 2560x1440 raid frame: border lines, a cols x rows grid, hatched cells."""
    im = np.full((1440, 2560, 3), 12, np.uint8)
    xl, yt = 1686, 104
    im[yt:1264, xl] = BORDER_BGR
    im[yt, xl:2466] = (70, 69, 67)
    xr, yb = int(xl + cols * P), int(yt + rows * P)
    im[yt:yb + 1, xl + 1:xr] = 24                              # grid backdrop
    for c in range(1, cols):
        im[yt:yb, int(xl + c * P)] = 36
    for r in range(1, rows):
        im[int(yt + r * P), xl:xr] = 36
    im[yt:yb + 1, xr] = (98, 94, 89)
    im[yb, xl:xr + 1] = (62, 60, 57)
    im[yt, xl:xr + 1] = (70, 69, 67)
    for c, r in hidden:
        x0, y0 = int(xl + c * P) + 2, int(yt + r * P) + 2
        yy, xx = np.mgrid[0:int(P) - 3, 0:int(P) - 3]
        stripes = 34 + 7 * (((xx + yy) // 7) % 2)
        im[y0:y0 + stripes.shape[0], x0:x0 + stripes.shape[1]] = stripes[..., None].astype(np.uint8)
    return im


def test_synthetic_loot_grid_and_hidden_cells():
    lo = analyze_loot(_synthetic(), P)
    p = lo.panel
    assert p is not None and (p.n_cols, p.n_rows) == (3, 2) and p.x0 == 1686 and abs(p.y0 - 104) <= 1
    assert abs((p.x1 - p.x0) - 3 * P) <= 3 and abs((p.y1 - p.y0) - 2 * P) <= 3
    assert lo.hidden == {(1, 0)} and lo.found and lo.placeholder is None


def test_synthetic_no_hatch_nothing_hidden():
    lo = analyze_loot(_synthetic(hidden=()), P)
    assert lo.panel is not None and lo.hidden == set()


def test_flat_and_art_cells_are_not_unrevealed():
    flat = np.full((70, 70), 24, np.uint8)
    rng = np.random.default_rng(1)
    art = (rng.integers(0, 255, (70, 70))).astype(np.uint8)
    assert not unrevealed(flat) and not unrevealed(art)
    yy, xx = np.mgrid[0:70, 0:70]
    assert unrevealed((34 + 7 * (((xx + yy) // 7) % 2)).astype(np.uint8))


def test_search_button_is_not_a_window():
    # the red cancel button of "SEARCHING x 0:05" in the LOOT header row, not a window's close button
    assert S._is_search_button((1975, 78, 24, 20), 2560, 1440)
    assert not S._is_search_button((1975, 300, 24, 20), 2560, 1440)      # lower: a real window
    assert not S._is_search_button((300, 78, 24, 20), 2560, 1440)        # left half


def test_grab_ignores_container_window_of_own_bag():
    from tests.test_live import ITEMS, SETTINGS, CTX
    d = {'item_id': 'gpu', 'role': 'container_window', 'col': 0, 'row': 0, 'W': 1, 'H': 1, 'panel': 0,
         'count': 1, 'uncertain': False, 'fir': None, 'category': None, 'px': 0, 'py': 0, 'pw': 10, 'ph': 10}
    run = lambda side: adv.rank_grab([dict(d, side=side)], ITEMS, {}, set(), SETTINGS, CTX)
    assert len(run('loot')) == 1 and run('own') == []


# --------------------------------------------------------------------------- real frames
@real
@pytest.mark.parametrize('hint', [None, True])
@pytest.mark.parametrize('i', list(range(7)) + [8, 9] + list(range(34, 41)))
def test_mislabelled_lobby_frames_stay_lobby(i, hint):
    im, sc = _scene(i, hint)
    expect = 'showcase' if i < 7 else 'lobby_stash' if i == 8 else 'lobby_gear' if i == 9 else 'flea_offer'
    assert sc.scene == expect and sc.in_raid is False
    assert not any(r.role == 'loot' for r in sc.regions)


@real
@pytest.mark.parametrize('i', RAID)
def test_raid_frame_has_loot_side_and_own_grids(i):
    im, sc = _scene(i)
    W = im.shape[1]
    assert sc.scene == 'raid_loot' and sc.in_raid is True and not sc.windows
    roles = {r.role for r in sc.regions}
    assert {'own_backpack', 'own_pouch', 'loot'} <= roles
    loot = [r for r in sc.regions if r.role == 'loot']
    assert len(loot) == 1 and loot[0].side == 'loot' and abs(loot[0].bbox[0] - 1686) <= 3
    for r in sc.regions:
        if r.role.startswith('own_'):
            assert r.bbox[0] + r.bbox[2] <= S.RAID_LOOT_X * W + 8            # never reaches the loot column
            assert r.side == 'own'
    if i == 7:
        assert {'own_special', 'own_pockets'} <= roles


@real
@pytest.mark.parametrize('i', sorted(GRIDS))
def test_loot_grid_cells_and_hidden(i):
    im, sc = _scene(i)
    nc, nr, nh = GRIDS[i]
    r = [r for r in sc.regions if r.role == 'loot'][0]
    assert abs(r.bbox[2] - nc * sc.pitch) < 0.08 * sc.pitch * nc and abs(r.bbox[3] - nr * sc.pitch) < 0.08 * sc.pitch * nr
    assert len(r.hidden) == nh and r.title in ('DRAWER', 'SAFE', 'DUFFLE BAG')
    pieces = S.split_panels(detect_grid(im), sc, im)
    cells = sum(p.n_cols * p.n_rows for p, rg in pieces if rg.role == 'loot')
    assert cells == nc * nr - nh                      # hidden cells are not read
    for p, rg in pieces:
        if rg.role == 'loot':
            for c, row in r.hidden:                   # no piece covers a hidden cell
                cx = r.bbox[0] + (c + 0.5) * sc.pitch
                cy = r.bbox[1] + (row + 0.5) * sc.pitch
                assert not (p.x0 <= cx <= p.x1 and p.y0 <= cy <= p.y1)


@real
@pytest.mark.parametrize('i', UNSEARCHED)
def test_unsearched_container_is_all_hidden(i):
    im, sc = _scene(i)
    r = [r for r in sc.regions if r.role == 'loot'][0]
    assert r.hidden and 5 * sc.pitch < r.bbox[2] < 7 * sc.pitch and 1.7 * sc.pitch < r.bbox[3] < 2.3 * sc.pitch
    pieces = S.split_panels(detect_grid(im), sc, im)
    assert not any(rg.role == 'loot' for _, rg in pieces)


@real
def test_raid_pieces_are_clean():
    """No lattice fitted to item art (21 px 'cells'), and the backpack column is not worn gear."""
    for i in (7, 22, 30):
        im, sc = _scene(i)
        pieces = S.split_panels(detect_grid(im), sc, im)
        assert all(0.8 * sc.pitch <= p.pitch_x <= 1.25 * sc.pitch for p, _ in pieces)
        mid = [(p, r) for p, r in pieces if p.x0 >= 1000 and p.x1 <= 1600]
        assert mid and all(r.role.startswith('own_') and r.role != 'own_other' for p, r in mid)


@real
def test_raid_scene_is_fast():
    im = cv2.imread(FRAMES[22])
    S.analyze_scene(im, in_raid=True)
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        S.analyze_scene(im, in_raid=True)
        t.append(time.perf_counter() - t0)
    assert min(t) < 0.25
