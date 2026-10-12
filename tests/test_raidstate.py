"""eftlogs.RaidState: lobby / matching / loading / in_raid from the live application log."""
import datetime
import os

import pytest

import eftlogs
from autoscan.detect import resolve_in_raid

REAL_INSTALL = r'C:\Battlestate Games\Escape from Tarkov'
REAL_FOLDER = 'log_2026.10.11_14-47-42_1.2.0.0.47965'


def ts(h, m, s):
    return datetime.datetime(2026, 10, 11, h, m, s).timestamp()


def line(h, m, s, rest):
    return f'2026-10-11 {h:02d}:{m:02d}:{s:02d}.100|1.2.0.0.47965|Info|application|{rest}\n'


def make_session(tmp_path, folder='log_2026.10.11_14-47-42_1.2.0.0.47965'):
    d = tmp_path / eftlogs.LOG_DIR_NAME / folder
    d.mkdir(parents=True)
    f = d / '2026.10.11_14-47-42_1.2.0.0.47965 application_000.log'
    f.write_text(line(14, 47, 43, "Application awaken, updateQueue:'Update'"), encoding='utf-8')
    return f


def append(f, text):
    with open(f, 'a', encoding='utf-8') as fh:
        fh.write(text)


RAID = (line(14, 52, 45, 'Matching with group id: ') +
        line(14, 52, 54, 'MatchingCompleted:5.76 real:8.62 diff:2.85') +
        line(14, 52, 56, 'LocationLoaded:7.63 real:11.01 diff:3.38') +
        line(14, 52, 57, "TRACE-NetworkGameCreate profileStatus: 'Profileid: x, Status: Busy, RaidMode: Online'") +
        line(14, 53, 46, 'GameStarted:53.98(53.98) real:60.84(60.84) diff:6.86'))


def test_lobby_matching_loading_raid_exit(tmp_path):
    f = make_session(tmp_path)
    rs = eftlogs.RaidState(str(tmp_path))
    assert rs.get()['state'] == 'lobby'
    append(f, line(14, 48, 22, 'BEClient exit'))                     # right after launch: still lobby
    assert rs.get()['state'] == 'lobby'
    append(f, line(14, 52, 45, 'Matching with group id: '))
    assert rs.get() == {'state': 'matching', 'since': ts(14, 52, 45), 'source': 'logs'}
    append(f, line(14, 52, 56, 'LocationLoaded:7.63 real:11.01 diff:3.38'))
    assert rs.get()['state'] == 'loading'
    append(f, line(14, 53, 46, 'GameStarted:53.98(53.98) real:60.84(60.84) diff:6.86'))
    st = rs.get()
    assert st['state'] == 'in_raid' and st['source'] == 'logs'
    assert st['since'] == pytest.approx(ts(14, 53, 46))
    assert rs.in_raid() is True
    append(f, line(15, 3, 43, 'BEClient exit') + line(15, 3, 43, 'BEClient exit successfully'))
    assert rs.get()['state'] == 'lobby' and rs.in_raid() is False
    assert rs.state_at(ts(14, 59, 0)) == 'in_raid' and rs.state_at(ts(15, 12, 0)) == 'lobby'


def test_network_game_create_alone_reaches_raid(tmp_path):
    f = make_session(tmp_path)
    rs = eftlogs.RaidState(str(tmp_path))
    append(f, line(14, 52, 57, "TRACE-NetworkGameCreate profileStatus: 'Status: Busy'"))
    assert rs.get()['state'] == 'loading'
    append(f, line(14, 53, 46, 'TRACE-NetworkGameCreate 6'))
    assert rs.get()['state'] == 'in_raid'


def test_only_appended_bytes_are_read_and_partial_lines_wait(tmp_path, monkeypatch):
    f = make_session(tmp_path)
    rs = eftlogs.RaidState(str(tmp_path))
    rs.get()
    reads = []
    real_open = open

    def spy(path, mode='r', *a, **k):
        fh = real_open(path, mode, *a, **k)
        if 'b' in mode and str(path).endswith('.log'):
            orig = fh.read
            fh.read = lambda n=-1: (reads.append(n), orig(n))[1]
        return fh
    monkeypatch.setattr('builtins.open', spy)
    rs.get()
    assert reads == []                                               # nothing appended: no read
    half = line(14, 53, 46, 'GameStarted:1')
    append(f, half[:30])
    assert rs.get()['state'] == 'lobby'                              # incomplete line is not consumed
    append(f, half[30:])
    assert rs.get()['state'] == 'in_raid'
    assert all(n < len(half) + 2 for n in reads)                     # only the new bytes


def test_new_folder_takes_over(tmp_path, monkeypatch):
    f = make_session(tmp_path)
    clock = [1000.0]
    rs = eftlogs.RaidState(str(tmp_path), clock=lambda: clock[0])
    append(f, RAID)
    assert rs.get()['state'] == 'in_raid'
    f2 = make_session(tmp_path, 'log_2026.10.11_16-00-00_1.2.0.0.47965')       # game restarted mid-"raid"
    assert rs.get()['state'] == 'in_raid'                            # folder list is only re-checked every few s
    clock[0] += eftlogs.RAID_FOLDER_RECHECK_S + 1
    assert rs.get()['state'] == 'lobby'
    append(f2, line(16, 5, 0, 'GameStarted:1'))
    assert rs.get()['state'] == 'in_raid'


def test_unreadable_returns_none(tmp_path):
    assert eftlogs.RaidState(None).get() is None
    assert eftlogs.RaidState(str(tmp_path / 'nope')).get() is None
    (tmp_path / eftlogs.LOG_DIR_NAME).mkdir()
    rs = eftlogs.RaidState(str(tmp_path))
    assert rs.get() is None and rs.in_raid() is None                 # no session folder at all
    d = tmp_path / eftlogs.LOG_DIR_NAME / 'log_2026.10.11_14-47-42_1'
    d.mkdir()
    assert rs.get()['state'] == 'lobby'                              # session started, nothing logged yet


def test_truncated_log_restarts(tmp_path):
    f = make_session(tmp_path)
    rs = eftlogs.RaidState(str(tmp_path))
    append(f, RAID)
    assert rs.get()['state'] == 'in_raid'
    f.write_text(line(14, 47, 43, 'Application awaken'), encoding='utf-8')
    assert rs.get()['state'] == 'lobby'


def test_resolve_in_raid_prefers_logs_with_menu_bar_as_sanity():
    raid, lobby = {'state': 'in_raid'}, {'state': 'lobby'}
    assert resolve_in_raid(lobby, bar_visible=False) is False        # flea offer / trader reward screen
    assert resolve_in_raid(raid, bar_visible=False) is True
    assert resolve_in_raid(raid, bar_visible=True) is False          # bar on screen cannot be a raid
    assert resolve_in_raid(None, bar_visible=False) is True          # logs unreadable: old heuristic
    assert resolve_in_raid(None, bar_visible=True) is False


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_INSTALL, 'Logs', REAL_FOLDER)),
                    reason='real EFT log folder not present')
def test_real_log_folder_three_raids():
    # tail the real folder (not the newest one: a game may be running); copy-free, read-only
    rs = eftlogs.RaidState(REAL_INSTALL)
    rs._folder = REAL_FOLDER
    rs._checked = 1e18                                               # pin the folder
    rs.get()
    for (a, b) in (((14, 53, 50), (15, 3, 40)), ((15, 25, 40), (15, 38, 20)), ((15, 43, 20), (15, 57, 50))):
        assert rs.state_at(ts(*a)) == 'in_raid' and rs.state_at(ts(*b)) == 'in_raid'
    for t in ((14, 50, 0), (15, 12, 0), (15, 20, 0), (15, 39, 0)):
        assert rs.state_at(ts(*t)) == 'lobby', t
    assert rs.state_at(ts(14, 52, 50)) == 'matching'
    assert rs.state_at(ts(14, 52, 56)) == 'loading'
