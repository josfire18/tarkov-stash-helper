"""The scan hotkey: auto-scan is off by default, the hotkey scans what is on screen once (and holds no
capture between presses), and the hotkey is a Win32 RegisterHotKey combo parsed from the stored text."""
import pytest

import app as tsh_app
import winhotkey
from autoscan import AutoScanner
from autoscan_frames import inventory_frame, raid_frame
from tests.test_raid_off import CountingCapture, LOBBY, RAID, make
from tests.test_autoscan import FakeWin32, M1, run, win


def test_auto_scan_is_off_by_default_and_the_hotkey_is_ctrl_f1():
    d = tsh_app.default_settings()
    assert d['auto_scan'] is False and d['hotkey'] == '<ctrl>+<f1>'
    assert AutoScanner(lambda f: {}, lambda: {}, win32=object(), capture=CountingCapture(None)).enabled() is False


def test_off_means_nothing_is_captured_until_the_hotkey():
    s, calls, cap, _, _ = make(inventory_frame(), LOBBY, settings={'auto_scan': False})
    run(s, 10)
    assert cap.grabs == 0 and calls == [] and s.status()['state'] == 'disabled'


def test_the_hotkey_scans_the_screen_once_and_releases_the_capture():
    s, calls, cap, _, _ = make(inventory_frame(), LOBBY, settings={'auto_scan': False, 'hotkey': '<ctrl>+<f1>'})
    run(s, 2)
    closed = cap.closed
    s.request_scan()
    run(s, 1)
    assert calls == [(1440, 2560, 3)] and cap.grabs == 1 and cap.closed > closed
    st = s.status()
    assert st['state'] == 'scanned' and st['seq'] == 1 and st['hotkey'] == '<ctrl>+<f1>'
    run(s, 10)                                       # the result stays up, nothing more is captured
    assert cap.grabs == 1 and len(calls) == 1 and s.status()['state'] == 'scanned'


def test_the_hotkey_scans_any_view_even_one_the_detector_would_skip():
    s, calls, cap, _, _ = make(raid_frame(), LOBBY, settings={'auto_scan': False})
    s.request_scan()
    run(s, 1)
    assert len(calls) == 1


def test_the_hotkey_with_auto_scan_on_scans_now_even_an_unchanged_view():
    s, calls, cap, _, _ = make(inventory_frame(), LOBBY)
    run(s, 6)
    assert len(calls) == 1
    s.request_scan()
    run(s, 1)
    assert len(calls) == 2


def test_the_hotkey_does_nothing_in_a_raid():
    s, calls, cap, _, _ = make(inventory_frame(), RAID, settings={'auto_scan': False})
    s.request_scan()
    run(s, 1)
    assert cap.grabs == 0 and calls == [] and s.status()['state'] == 'in_raid'


def test_the_hotkey_without_the_game_running_reports_it():
    s, calls, cap, _, _ = make(inventory_frame(), LOBBY, settings={'auto_scan': False})
    s.win = FakeWin32(pids=())
    s.request_scan()
    run(s, 1)
    assert calls == [] and s.status()['state'] == 'no_game'


@pytest.mark.parametrize('combo,mods,vk', [
    ('<ctrl>+<f1>', winhotkey.MOD_CONTROL, 0x70),
    ('<ctrl>+<shift>+s', winhotkey.MOD_CONTROL | winhotkey.MOD_SHIFT, ord('S')),
    ('<alt>+<f12>', winhotkey.MOD_ALT, 0x7B),
    ('<ctrl>+<alt>+<page_down>', winhotkey.MOD_CONTROL | winhotkey.MOD_ALT, 0x22),
    ('<shift>+5', winhotkey.MOD_SHIFT, ord('5')),
    ('<f9>', 0, 0x78),
])
def test_parse(combo, mods, vk):
    assert winhotkey.parse(combo) == (mods, vk)


@pytest.mark.parametrize('bad', ['', '<ctrl>', '<ctrl>+<shift>', '<ctrl>+<nope>', '<ctrl>+a+b', '<ctrl>++a', 'abc'])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        winhotkey.parse(bad)


def test_pretty():
    assert winhotkey.pretty('<ctrl>+<f1>') == 'Ctrl+F1'
    assert winhotkey.pretty('<ctrl>+<shift>+s') == 'Ctrl+Shift+S'


def test_old_settings_move_to_the_hotkey_once():
    s = {'auto_scan': True, 'hotkey': '<ctrl>+<shift>+s'}
    assert tsh_app.migrate_scan_settings(s) is True
    assert s['auto_scan'] is False and s['hotkey'] == '<ctrl>+<f1>'
    s['auto_scan'] = True                            # turned back on afterwards: stays on
    assert tsh_app.migrate_scan_settings(s) is False and s['auto_scan'] is True


def test_a_hotkey_the_user_picked_is_kept():
    s = {'auto_scan': True, 'hotkey': '<alt>+<f5>'}
    tsh_app.migrate_scan_settings(s)
    assert s['hotkey'] == '<alt>+<f5>' and s['auto_scan'] is False


def test_new_installs_start_migrated():
    s = tsh_app.default_settings()
    assert tsh_app.migrate_scan_settings(s) is False and s['auto_scan'] is False


def test_hotkey_problems_are_plain_messages():
    assert tsh_app.hotkey_problem(winhotkey.HotkeyTaken('x'), '<ctrl>+<f1>') == \
        'Ctrl+F1 is already used by another program. Pick another key.'
    assert 'not a key combination' in tsh_app.hotkey_problem(ValueError('x'), '<ctrl>+<nope>')
    assert tsh_app.hotkey_problem(OSError('x'), '<ctrl>+<f1>') == 'Windows refused the hotkey. Try another key.'
