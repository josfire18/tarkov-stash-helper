"""Whole-stash capture (identify/stitch.py): scans at different scroll positions line up and merge
into one item list.  Frames are cut from one tall synthetic stash, as the game's viewport would show
it, with a scroll bar thumb drawn beside the viewport."""
import numpy as np
import pytest

from identify.stitch import StashStitcher, scroll_thumb, vertical_shift
from tests.synth import render_panel

PITCH = 63
COLS, ROWS = 10, 40
VIEW_ROWS = 12
BOX = (100, 50, COLS * PITCH + 1, VIEW_ROWS * PITCH)      # viewport on screen (x, y, w, h)


def _layout():
    rng = np.random.default_rng(7)
    taken = np.zeros((ROWS, COLS), bool)
    items = []
    for r in range(ROWS):
        if r in (20, 21):                    # two empty rows in the middle
            continue
        for c in range(COLS):
            if taken[r, c] or rng.random() < 0.3:
                continue
            w, h = (1, 1) if rng.random() < 0.7 else (int(rng.integers(1, 3)), int(rng.integers(1, 4)))
            w = min(w, COLS - c)
            h = min(h, ROWS - r)
            if r < 20 <= r + h - 1:          # keep the empty rows empty
                h = 20 - r
            if taken[r:r + h, c:c + w].any():
                w = h = 1
            taken[r:r + h, c:c + w] = True
            items.append((c, r, w, h))
    return items


ITEMS = _layout()
STASH, _, _, _ = render_panel(PITCH, COLS, ROWS, ITEMS, origin=(0, 0), chrome=False, size=(COLS * PITCH + 1, ROWS * PITCH + 1))


def frame_at(scroll, items=ITEMS, stash=None, thumb=True):
    """The screen with the stash scrolled ``scroll`` px down, and the records a scan would give."""
    stash = STASH if stash is None else stash
    x, y, w, h = BOX
    frame = np.full((y + h + 40, x + w + 40, 3), 12, np.uint8)
    frame[y:y + h, x:x + w] = stash[scroll:scroll + h, :w]
    if thumb:
        total = ROWS * PITCH
        t0 = y + int(round(scroll / total * h))
        t1 = y + int(round((scroll + h) / total * h)) - 1
        frame[t0:t1 + 1, x + w + 2:x + w + 6] = (160, 160, 160)
    recs = []
    for k, (c, r, iw, ih) in enumerate(items):
        top, bot = r * PITCH - scroll, (r + ih) * PITCH - scroll
        if bot <= 0 or top >= h:
            continue
        vt, vb = max(0, top), min(h, bot)
        if vb - vt < 0.2 * PITCH:
            continue
        recs.append({'col': c, 'row': 0, 'W': iw, 'H': ih, 'item_id': f'it{c}_{r}', 'name': f'it{c}_{r}',
                     'px': x + c * PITCH, 'py': y + vt, 'pw': iw * PITCH, 'ph': vb - vt,
                     'clipped': top < 0 or bot > h, 'role': 'stash'})
    return frame, recs


def _ids(st):
    return {r['item_id'] for r in st.items()}


def _expected(r_lo, r_hi, items=ITEMS):
    """Items lying in rows [r_lo, r_hi) at least partly."""
    return {f'it{c}_{r}' for c, r, w, h in items if r < r_hi and r + h > r_lo}


def test_shift_found_for_whole_and_part_rows():
    for d in (3 * PITCH, 7 * PITCH, int(2.5 * PITCH), -4 * PITCH):
        a, _ = frame_at(8 * PITCH)
        b, _ = frame_at(8 * PITCH + d)
        sh = vertical_shift(a, b, BOX, PITCH)
        assert sh is not None and sh.dy == d, (d, sh)


def test_no_overlap_gives_no_shift():
    a, _ = frame_at(0)
    b, _ = frame_at(VIEW_ROWS * PITCH + PITCH)
    assert vertical_shift(a, b, BOX, PITCH) is None


def test_empty_area_is_ambiguous():
    blank = render_panel(PITCH, COLS, ROWS, [], origin=(0, 0), chrome=False,
                         size=(COLS * PITCH + 1, ROWS * PITCH + 1))[0]
    a, _ = frame_at(0, items=[], stash=blank)
    b, _ = frame_at(2 * PITCH, items=[], stash=blank)
    assert vertical_shift(a, b, BOX, PITCH) is None


def test_scroll_thumb_reads_position_and_size():
    f, _ = frame_at(10 * PITCH)
    t = scroll_thumb(f, BOX, PITCH)
    assert t is not None
    assert abs(t.total_rows(VIEW_ROWS) - ROWS) <= 1
    assert abs(t.first_row(ROWS) - 10) <= 0.5


def test_scrolling_down_collects_the_whole_stash():
    st = StashStitcher()
    scrolls = sorted(set(range(0, (ROWS - VIEW_ROWS) * PITCH + 1, 5 * PITCH)) | {(ROWS - VIEW_ROWS) * PITCH})
    for s in scrolls:
        f, recs = frame_at(s)
        res = st.add(f, BOX, PITCH, recs)
        assert res.status in ('started', 'added'), res
    assert _ids(st) == _expected(0, ROWS)
    rows = {r['item_id']: r['stash_row'] for r in st.items()}
    for c, r, w, h in ITEMS:
        assert rows[f'it{c}_{r}'] == r
    cov = st.coverage()
    assert cov['ranges'] == [(1, ROWS)] and cov['anchored'] and abs(cov['total'] - ROWS) <= 1
    img = st.image()
    assert img.shape[0] == ROWS * PITCH
    # every merged item sits where it is in the stash picture
    for rec in st.items():
        assert rec['py'] == rec['stash_row'] * PITCH


def test_part_row_scrolls_and_scrolling_up():
    st = StashStitcher()
    for s in (20 * PITCH, int(16.5 * PITCH), int(9.5 * PITCH), 5 * PITCH):
        f, recs = frame_at(s)
        assert st.add(f, BOX, PITCH, recs).status in ('started', 'added')
    assert _ids(st) == _expected(5, 20 + VIEW_ROWS)
    for rec in st.items():
        c, r = (int(v) for v in rec['item_id'][2:].split('_'))
        assert rec['col'] == c
    # not anchored to the stash top: rows count from the first fully visible row seen
    assert not st.coverage()['anchored']
    assert st.coverage()['ranges'] == [(1, 20 + VIEW_ROWS - 5)]


def test_clipped_item_does_not_replace_a_full_view():
    st = StashStitcher()
    f, recs = frame_at(0)
    st.add(f, BOX, PITCH, recs)
    f, recs = frame_at(int(4.5 * PITCH))
    st.add(f, BOX, PITCH, recs)
    by_id = {r['item_id']: r for r in st.items()}
    full_in_first = [f'it{c}_{r}' for c, r, w, h in ITEMS if r + h <= VIEW_ROWS and r < 4.5 <= r + h]
    assert full_in_first
    for iid in full_in_first:
        assert not by_id[iid].get('clipped')


def test_sold_item_disappears_on_rescan():
    st = StashStitcher()
    f, recs = frame_at(0)
    st.add(f, BOX, PITCH, recs)
    f, recs = frame_at(6 * PITCH)
    st.add(f, BOX, PITCH, recs)
    gone = next((c, r, w, h) for c, r, w, h in ITEMS if 7 <= r and r + h <= 11)
    left = [it for it in ITEMS if it != gone]
    stash = render_panel(PITCH, COLS, ROWS, left, origin=(0, 0), chrome=False,
                         size=(COLS * PITCH + 1, ROWS * PITCH + 1))[0]
    f, recs = frame_at(3 * PITCH, items=left, stash=stash)
    assert st.add(f, BOX, PITCH, recs).status == 'added'
    assert f'it{gone[0]}_{gone[1]}' not in _ids(st)


def test_rescanning_the_same_spot_updates_it():
    st = StashStitcher()
    f, recs = frame_at(0)
    st.add(f, BOX, PITCH, recs)
    assert st.add(f, BOX, PITCH, recs).status == 'unchanged'
    assert len(st.pieces) == 1 and _ids(st) == _expected(0, VIEW_ROWS)


def test_far_jump_uses_the_scroll_bar_or_asks():
    st = StashStitcher()
    f, recs = frame_at(0)
    st.add(f, BOX, PITCH, recs)
    f, recs = frame_at(26 * PITCH)
    res = st.add(f, BOX, PITCH, recs)
    assert res.status == 'added' and 'scroll bar' in res.message
    assert {r['item_id']: r['stash_row'] for r in st.items()}[sorted(_expected(27, 30))[0]] >= 26

    st = StashStitcher()
    f, recs = frame_at(0, thumb=False)
    st.add(f, BOX, PITCH, recs)
    f, recs = frame_at(26 * PITCH, thumb=False)
    assert st.add(f, BOX, PITCH, recs).status == 'unplaced'
    assert len(st.pieces) == 1


def test_layout_change_starts_over():
    st = StashStitcher()
    f, recs = frame_at(0)
    st.add(f, BOX, PITCH, recs)
    x, y, w, h = BOX
    res = st.add(f, (x, y, w, h - 2 * PITCH), PITCH, recs)
    assert res.status == 'restarted' and len(st.pieces) == 1


# --------------------------------------------------------------------------
# app.py: the Full stash switch on the Live page
# --------------------------------------------------------------------------

@pytest.fixture
def stitched_app(monkeypatch):
    import app as tsh_app
    frames = {}

    def fake_scan(img_bgr, settings, warnings, scene_out=None, in_raid=None, on_provisional=None, cancel=None):
        assert on_provisional is None or not tsh_app._stash_pass['on']    # no single-screen first look while stitching
        scene_out.update({'scene': 'lobby_stash', 'in_raid': False, 'regions': [],
                          'stash_bbox': BOX, 'pitch': float(PITCH)})
        return frames[id(img_bgr)], [{'cell_w': PITCH, 'cell_h': PITCH, 'origin_x': 0, 'origin_y': 0}], False

    def fake_payload(raw, panels, grid_failed, warnings, scene_info, settings, img, img_bgr, lean, final=True):
        return {'results': raw, 'grid': panels[0], 'warnings': warnings, 'image': None,
                'size': [img_bgr.shape[1], img_bgr.shape[0]]}

    monkeypatch.setattr(tsh_app, 'scan_with_v2', fake_scan)
    monkeypatch.setattr(tsh_app, '_sell_payload', fake_payload)
    monkeypatch.setattr(tsh_app, '_load_settings', lambda: {})
    monkeypatch.setattr(tsh_app.raid_state, 'in_raid', lambda: False)
    tsh_app._stash_pass['on'] = False
    tsh_app._stash_pass['stitcher'].reset()

    def scan(scroll):
        f, recs = frame_at(scroll)
        frames[id(f)] = recs
        with tsh_app.app.app_context():
            return tsh_app._sell_scan_inner(frame_bgr=f, lean=True).get_json()
    yield tsh_app, scan
    tsh_app._stash_pass['on'] = False
    tsh_app._stash_pass['stitcher'].reset()


def test_full_stash_is_off_until_turned_on(stitched_app):
    tsh_app, scan = stitched_app
    c = tsh_app.app.test_client()
    assert c.get('/api/stash-pass').get_json()['on'] is False
    res = scan(0)
    assert 'stitch' not in res
    assert c.get('/api/stash-pass').get_json()['scans'] == 0


def test_full_stash_merges_scans_and_serves_the_picture(stitched_app):
    tsh_app, scan = stitched_app
    c = tsh_app.app.test_client()
    assert c.post('/api/stash-pass', json={'on': True}).get_json()['on'] is True
    scan(0)
    res = scan(9 * PITCH)
    assert res['stitch']['ranges'] == [[1, 9 + VIEW_ROWS]]
    assert {r['item_id'] for r in res['results']} == _expected(0, 9 + VIEW_ROWS)
    assert res['size'] == [BOX[2], (9 + VIEW_ROWS) * PITCH]
    img = c.get(res['frame_url'])
    assert img.status_code == 200 and img.mimetype == 'image/jpeg'
    info = c.post('/api/stash-pass', json={'reset': True}).get_json()
    assert info['scans'] == 0 and info['on'] is True
    assert c.post('/api/stash-pass', json={'on': False}).get_json()['on'] is False
