"""Scene layer on the other lobby screens: flea "Offer creation", trader, receive-all SHOWCASE,
hand-over window, scav loot transfer (game version 1.2.0 frames).

The real-frame tests read the live collector's frames from the gitignored ``data/scenes/live/``
(2560x1440) and are skipped when absent.  The ``in_raid`` flags the collector stored for some of
them are wrong (lobby screens flagged as raid): analyze_scene must not depend on that hint."""
import os
import time

import cv2
import pytest

from identify import scene as S
from identify.grid import GridResult, Panel, detect_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(ROOT, 'data', 'scenes', 'live')
LIVE_FRAMES = {
    'gear': '1791730346167-lobby.png', 'flea_offer': '1791730379737-raid.png',
    'trader': '1791730400469-lobby.png', 'handover': '1791731210892-lobby.png',
    'picker': '1791731263265-lobby.png', 'showcase': '1791731298039-raid.png',
    'scav': '1791734087953-lobby.png', 'scav_empty_rig': '1791734092427-lobby.png',
    'scav_items': '1791734081088-lobby.png', 'trader_dim': '1791730353083-lobby.png',
}
live = pytest.mark.skipif(not all(os.path.exists(os.path.join(LIVE, f)) for f in LIVE_FRAMES.values()),
                          reason='live lobby frames not present')
_live = {}


def _lv(name, in_raid=None):
    k = (name, in_raid)
    if k not in _live:
        im = cv2.imread(os.path.join(LIVE, LIVE_FRAMES[name]))
        sc = S.analyze_scene(im, in_raid=in_raid)
        _live[k] = (im, sc, S.split_panels(detect_grid(im), sc, im))
    return _live[k]


def _inside(inner, outer, slack=4):
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack and
            inner[0] + inner[2] <= outer[0] + outer[2] + slack and inner[1] + inner[3] <= outer[1] + outer[3] + slack)


# --------------------------------------------------------------------------- real frames
@live
@pytest.mark.parametrize('hint', [None, True, False])
def test_flea_offer_stash_found_inside_the_window(hint):
    im, sc, pieces = _lv('flea_offer', hint)
    assert sc.scene == 'flea_offer' and sc.in_raid is False
    win = [w for w in sc.windows if w.role == 'offer_window']
    assert len(win) == 1 and (win[0].title or '').lower().startswith('offer')
    stash = [r for r in sc.regions if r.role == 'stash']
    assert len(stash) == 1 and _inside(stash[0].bbox, win[0].bbox)
    assert stash[0].bbox[0] < 0.05 * im.shape[1]                      # the toolbar sits at the screen edge
    assert abs(stash[0].bbox[2] - 10 * sc.pitch) < 0.1 * sc.pitch + 12
    sp = [(p, r) for p, r in pieces if r.role == 'stash']
    assert sum(p.n_cols * p.n_rows for p, _ in sp) >= 50
    assert all(p.x1 <= stash[0].bbox[0] + stash[0].bbox[2] + 6 for p, _ in sp)       # nothing of the offer list
    assert all(r.role == 'stash' for _, r in pieces)


@live
@pytest.mark.parametrize('hint', [None, True])
def test_showcase_screen_has_stash_and_incoming(hint):
    im, sc, pieces = _lv('showcase', hint)
    assert sc.scene == 'showcase' and sc.in_raid is False
    roles = {r.role: r for r in sc.regions}
    assert roles['stash'].side == 'stash' and roles['incoming'].side == 'stash'
    assert roles['incoming'].bbox[0] < roles['stash'].bbox[0]
    inc = [p for p, r in pieces if r.role == 'incoming']
    assert sum(p.n_cols * p.n_rows for p in inc) == 140 and max(p.n_cols for p in inc) == 10
    assert all(p.x1 < roles['stash'].bbox[0] for p in inc)
    assert sum(p.n_cols * p.n_rows for p, r in pieces if r.role == 'stash') >= 100
    assert 'incoming' in S.ADVISED_ROLES


@live
def test_trader_goods_are_never_the_players():
    im, sc, pieces = _lv('trader', True)
    assert sc.scene == 'trader' and sc.in_raid is False
    roles = {r.role: r for r in sc.regions}
    assert 'trader_goods' in roles and roles['trader_goods'].side == 'other'
    assert roles['trader_goods'].bbox[0] < 0.05 * im.shape[1] and roles['stash'].bbox[0] > 0.6 * im.shape[1]
    assert 'trader_goods' not in S.ADVISED_ROLES and 'trader_goods' not in S.SCANNED_ROLES
    assert pieces and all(r.role == 'stash' for _, r in pieces)           # goods + deal panel are cut away
    assert all(p.x0 >= roles['stash'].bbox[0] for p, _ in pieces)


@live
@pytest.mark.parametrize('name', ['trader_dim', 'trader'])
def test_dimmed_trader_stash_is_found_whole(name):
    """The sell tab dims the stash items the trader does not buy (border ~(38, 38, 36)): the whole 10 x 12
    lattice must still be found (frame 1 had no stash cell, frame 6 only the bright half), and the
    dimmed items must not be called empty."""
    from identify.segment import segment_panel
    im, sc, pieces = _lv(name, None)
    st = [(p, r) for p, r in pieces if r.role == 'stash']
    assert sum(p.n_cols * p.n_rows for p, _ in st) >= 120 and all(p.n_cols == 10 for p, _ in st)
    assert all(p.x0 >= sc.stash_bbox[0] and p.x1 <= sc.stash_bbox[0] + sc.stash_bbox[2] + 4 for p, _ in st)
    items = [it for i, (p, _) in enumerate(st) for it in segment_panel(im, p, i)]
    assert sum(1 for it in items if not it.empty) >= 20          # frame 1: ~22 items, 8 empty blobs


@live
def test_scav_backpack_found_whole():
    im, sc, pieces = _lv('scav_items', None)
    bp = [p for p, r in pieces if r.role == 'incoming' and r.title == 'BACKPACK']
    assert [(p.n_cols, p.n_rows) for p in bp] == [(4, 5)]      # the dark 4th column used to be cut off
    im2, sc2, pieces2 = _lv('scav', None)
    assert [(p.n_cols, p.n_rows) for p, r in pieces2 if r.title == 'BACKPACK'] == [(4, 5)]


@live
def test_gear_screen_unchanged():
    im, sc, pieces = _lv('gear', None)
    assert sc.scene == 'lobby_gear' and not sc.in_raid
    assert {r.role for r in sc.regions} == {'stash', 'own_pockets', 'own_pouch'}
    st = [r for r in sc.regions if r.role == 'stash'][0]
    assert 1650 < st.bbox[0] < 1700 and 90 < st.bbox[1] < 120 and 1100 < st.bbox[3] < 1200
    cells = {}
    for p, r in pieces:
        cells[r.role] = cells.get(r.role, 0) + p.n_cols * p.n_rows
    assert cells == {'stash': 140, 'own_pockets': 4, 'own_pouch': 9}


@live
def test_handover_window_is_not_scanned():
    im, sc, pieces = _lv('handover', None)       # a tooltip hides the window's red close button
    assert sc.scene == 'flea_handover' and sc.in_raid is False
    assert [w.role for w in sc.windows] == ['handover']
    assert pieces == []


@live
def test_picker_window_traced_to_its_real_height():
    im, sc, pieces = _lv('picker', None)         # a context menu covers the left border
    pk = [w for w in sc.windows if w.role == 'picker']
    assert len(pk) == 1 and pk[0].bbox[3] > 350
    assert [r.role for _, r in pieces].count('picker') == 0


@live
def test_scav_loot_transfer_grids_are_incoming():
    im, sc, pieces = _lv('scav', None)
    assert sc.scene == 'scav_transfer' and sc.in_raid is False
    assert {r.role for r in sc.regions} == {'stash', 'incoming'}
    inc = [r for r in sc.regions if r.role == 'incoming']
    assert {r.title for r in inc} == {'TACTICAL RIG', 'POCKETS', 'BACKPACK'} and all(r.side == 'stash' for r in inc)
    assert {r.role for _, r in pieces} == {'stash', 'incoming'}
    assert all(p.own for p, r in pieces if r.role == 'incoming')


@live
def test_scav_rig_empty_slots_stay_empty():
    from identify.segment import segment_panel
    im, sc, pieces = _lv('scav_empty_rig', None)
    rig = [(i, p) for i, (p, r) in enumerate(pieces) if r.role == 'incoming' and r.title == 'TACTICAL RIG']
    tall = [(i, p) for i, p in rig if (p.n_cols, p.n_rows) == (1, 2)]
    assert len(tall) == 4                                   # the four 1x2 rig slots are found ...
    for i, p in tall:
        assert all(it.empty for it in segment_panel(im, p, i))      # ... and are empty (no item)
    pk = [(i, p) for i, (p, r) in enumerate(pieces) if r.role == 'incoming' and r.title == 'POCKETS']
    assert sum(1 for i, p in pk for it in segment_panel(im, p, i) if not it.empty) == 4
    im2, sc2, pieces2 = _lv('scav_items', None)                 # slots holding items are found too
    rig2 = [(i, p) for i, (p, r) in enumerate(pieces2) if r.role == 'incoming' and r.title == 'TACTICAL RIG'
            and (p.n_cols, p.n_rows) == (1, 2)]
    full = [1 for i, p in rig2 for it in segment_panel(im2, p, i) if not it.empty]
    assert len(rig2) == 4 and len(full) == 2


@live
def test_lobby_scene_is_cheap_warm():
    for name in ('gear', 'flea_offer', 'trader', 'showcase', 'scav'):
        im = cv2.imread(os.path.join(LIVE, LIVE_FRAMES[name]))
        S.analyze_scene(im)
        t = time.perf_counter()
        S.analyze_scene(im)
        assert time.perf_counter() - t < 0.25, name


# --------------------------------------------------------------------------- synthetic: roles of new screens
def _grid_panel(x0, y0, nc, nr, pitch=84.0):
    xs = [int(x0 + k * pitch) for k in range(nc + 1)]
    ys = [int(y0 + k * pitch) for k in range(nr + 1)]
    return Panel(x0=xs[0], y0=ys[0], x1=xs[-1] + 1, y1=ys[-1] + 1, ox=xs[0], oy=ys[0], pitch_x=pitch, pitch_y=pitch,
                 n_cols=nc, n_rows=nr, xs=xs, ys=ys)


def _lobby_scene(kind):
    sc = S.SceneInfo(scene=kind, in_raid=False, frame_wh=(2560, 1440), pitch=84.0, toolbar=(1651, 1671))
    sc.stash_bbox = (1678, 350, 855, 992)
    sc.side_bbox = (12, 350, 860, 992)
    return sc


@pytest.mark.parametrize('kind,role', [('trader', None), ('showcase', 'incoming')])
def test_side_grid_roles_synthetic(kind, role):
    sc = _lobby_scene(kind)
    side, stash, deal = _grid_panel(12, 350, 10, 3), _grid_panel(1690, 350, 10, 3), _grid_panel(940, 350, 6, 3)
    out = S.split_panels(GridResult([side, stash, deal], 84.0, 84.0), sc, None)
    roles = sorted({r.role for _, r in out})
    assert roles == (['stash'] if role is None else ['incoming', 'stash'])
    assert all(p.x0 > 1600 for p, r in out if r.role == 'stash')
    assert not any(900 < p.x0 < 1600 for p, _ in out)                        # the deal panel is never read
    if role:
        assert all(r.side == 'stash' for _, r in out)


def test_stash_grid_inside_offer_window_is_stash():
    sc = S.SceneInfo(scene='flea_offer', in_raid=False, frame_wh=(2560, 1440), pitch=84.0, toolbar=(14, 31))
    sc.stash_bbox = (38, 280, 855, 1148)
    sc.windows.append(S.Region((0, 146, 1601, 1294), 'offer_window', 'other', 'Offer creation', 0.9, 40))
    stash, offers = _grid_panel(48, 280, 10, 4), _grid_panel(1100, 280, 4, 4)
    out = S.split_panels(GridResult([stash, offers], 84.0, 84.0), sc, None)
    assert [r.role for _, r in out] == ['stash'] and out[0][0].n_cols == 10 and out[0][0].n_rows == 4


def test_handover_window_cells_dropped_synthetic():
    sc = S.SceneInfo(scene='flea_handover', in_raid=False, frame_wh=(2560, 1440), pitch=84.0)
    sc.windows.append(S.Region((800, 300, 900, 800), 'handover', 'other', 'Hand over', 0.9, 40))
    out = S.split_panels(GridResult([_grid_panel(850, 450, 5, 4)], 84.0, 84.0), sc, None)
    assert out == []


def test_advised_roles_cover_incoming_never_trader_goods():
    assert set(S.ADVISED_ROLES) == {'stash', 'container_window', 'incoming'}
    assert not {'trader_goods', 'handover', 'offer_window', 'picker'} & set(S.SCANNED_ROLES)
