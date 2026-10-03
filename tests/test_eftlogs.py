"""Automatic task progress from EFT's log folders (eftlogs.py) and its merge into the app's
progress (app.effective_progress).  Synthetic log folders in the real formats."""
import json
import os

import eftlogs

Q1, Q2, Q3, Q4 = ('a' * 24, 'b' * 24, 'c' * 24, 'd' * 24)
TRADER = 'f' * 24


def _notif(when, typ, qid, dt, suffix='successMessageText'):
    block = {'type': 'new_message', 'eventId': '1', 'dialogId': TRADER,
             'message': {'_id': 'x', 'uid': TRADER, 'type': typ, 'dt': dt, 'text': 'quest started',
                         'templateId': f'{qid} {suffix}'}}
    return (f'{when}|1.1.5.1.47510|Info|push-notifications|Got notification | ChatMessageReceived\n'
            + json.dumps(block, indent=2) + '\n')


def _folder(root, name, notifs=(), mode='PvpSeason', backend=(), host='gw-pvp-season'):
    d = root / 'Logs' / name
    d.mkdir(parents=True)
    stamp = name[4:].replace('log_', '')
    when = '2026-09-01 10:00:00.000'
    (d / f'{stamp} application_000.log').write_text(
        f'{when}|1.1.5.1.47510|Info|application|Session mode: {mode}\n')
    lines = [f'{when}|1.1.5.1.47510|Info|backend|---> Request HTTPS, id [1]: URL: '
             f'https://{host}.escapefromtarkov.com/client/game/start. \n']
    for t, what in backend:
        lines.append(f'{t}|1.1.5.1.47510|Info|backend|---> Request HTTPS, id [9]: URL: '
                     f'https://{host}.escapefromtarkov.com/client/{what}. \n')
    (d / f'{stamp} backend_000.log').write_text(''.join(lines))
    (d / f'{stamp} push-notifications_000.log').write_text(''.join(notifs))
    return d


def _scan(root, mode='auto', tasks=None, faction='auto'):
    s = eftlogs.LogScanner(str(root / 'cache.json'))
    s.scan(str(root))
    return s, s.progress(mode, tasks, faction)


def test_start_complete_fail(tmp_path):
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-01 10:01:00.000', 10, Q1, 1000, 'description'),
        _notif('2026-09-01 10:02:00.000', 12, Q1, 1100),
        _notif('2026-09-01 10:03:00.000', 10, Q2, 1200, 'description'),
        _notif('2026-09-01 10:04:00.000', 11, Q3, 1300, 'failMessageText'),
    ])
    _s, p = _scan(tmp_path)
    assert p['mode'] == 'season'
    assert p['completed'] == [Q1] and p['active'] == [Q2] and p['failed'] == [Q3]


def test_fail_then_restart_is_active(tmp_path):
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-01 10:01:00.000', 11, Q1, 1000, 'failMessageText'),
        _notif('2026-09-01 10:02:00.000', 10, Q1, 2000, 'description'),
    ])
    _s, p = _scan(tmp_path)
    assert p['active'] == [Q1] and p['failed'] == []


def test_prestige_wipes_earlier_progress(tmp_path):
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-01 10:01:00.000', 12, Q1, 1000)], mode='Regular', host='gw-pvp')
    later = eftlogs.parse_log_time('2026-09-02', '10:00:00')
    _folder(tmp_path, 'log_2026.09.02_10-00-00_1.1.5.1.47510',
            [_notif('2026-09-02 10:05:00.000', 12, Q2, int(later) + 300)],
            mode='Regular', host='gw-pvp', backend=[('2026-09-02 10:00:00.000', 'prestige/obtain')])
    _s, p = _scan(tmp_path, mode='pvp')
    assert p['completed'] == [Q2] and p['reset_kind'] == 'prestige'


def test_modes_are_separate_profiles(tmp_path):
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-01 10:01:00.000', 12, Q1, 1000)], mode='Pve', host='gw-pve-01')
    _folder(tmp_path, 'log_2026.09.02_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-02 10:01:00.000', 12, Q2, 2000)], mode='PvpSeason')
    s, auto = _scan(tmp_path)
    assert auto['mode'] == 'season' and auto['completed'] == [Q2]       # latest session's profile
    assert s.progress('pve')['completed'] == [Q1]


def test_prerequisites_and_faction(tmp_path):
    tasks = [
        {'id': Q1, 'name': 'One', 'factionName': 'Any', 'taskRequirements': []},
        {'id': Q2, 'name': 'Two', 'factionName': 'Any',
         'taskRequirements': [{'task': {'id': Q1}, 'status': ['complete']}]},
        {'id': Q3, 'name': 'Usec only', 'factionName': 'USEC', 'taskRequirements': []},
        {'id': Q4, 'name': 'Bear only', 'factionName': 'BEAR', 'taskRequirements': []},
    ]
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [
        _notif('2026-09-01 10:01:00.000', 12, Q2, 1000),                       # Q1 never logged
        _notif('2026-09-01 10:02:00.000', 10, Q3, 1100, 'description'),
    ])
    _s, p = _scan(tmp_path, tasks=tasks)
    assert set(p['completed']) == {Q1, Q2} and p['inferred'] == [Q1]
    assert p['faction'] == 'USEC' and p['other_faction'] == [Q4]


def test_rescan_reads_only_the_live_folder(tmp_path):
    _folder(tmp_path, 'log_2026.09.01_10-00-00_1.1.5.1.47510', [_notif('2026-09-01 10:01:00.000', 12, Q1, 1)])
    live = _folder(tmp_path, 'log_2026.09.02_10-00-00_1.1.5.1.47510', [])
    s = eftlogs.LogScanner(str(tmp_path / 'cache.json'))
    assert s.scan(str(tmp_path))['read'] == 2
    notif = next(p for p in os.listdir(live) if 'push-notifications' in p)
    with open(live / notif, 'a') as f:
        f.write(_notif('2026-09-02 10:01:00.000', 12, Q2, 2))
    assert s.scan(str(tmp_path), full=False)['read'] == 1
    assert set(s.progress()['completed']) == {Q1, Q2}
    s2 = eftlogs.LogScanner(str(tmp_path / 'cache.json'))                     # disk cache reused
    assert s2.scan(str(tmp_path))['read'] == 1


def test_effective_progress_merges_logs_and_manual_overrides(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, 'PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    (tmp_path / 'progress.json').write_text(json.dumps({
        'completed_tasks': [Q4], 'completed_hideout': [], 'have': {},
        'manual_overrides': {Q1: 'open', Q3: 'done'}}))
    monkeypatch.setattr(app, 'log_task_progress', lambda cache=None: {
        'completed': [Q1, Q2], 'failed': [], 'active': [], 'other_faction': [], 'mode': 'season',
        'faction': 'USEC', 'last_event_at': 1, 'reset_at': None, 'reset_kind': None, 'folders': 1})
    p = app.effective_progress({'tasks': []})
    assert set(p['completed_tasks']) == {Q2, Q3, Q4}      # Q1 reopened by the player, Q3 forced done
    assert p['auto_done'] == [Q2]
    assert p['log_status']['mode'] == 'season'


def test_effective_progress_off(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, 'PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setattr(app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    (tmp_path / 'settings.json').write_text(json.dumps({'auto_task_progress': False}))
    p = app.effective_progress({'tasks': []})
    assert p['completed_tasks'] == [] and p['log_status'] is None


def test_migration_turns_kappa_only_off_once(tmp_path):
    import app
    p = tmp_path / 'settings.json'
    p.write_text(json.dumps({'kappa_only_tasks': True}))
    assert app.migrate_settings(str(p)) == ['kappa_scope_all_tasks']
    assert json.loads(p.read_text())['kappa_only_tasks'] is False
    p.write_text(json.dumps({**json.loads(p.read_text()), 'kappa_only_tasks': True}))   # user re-enables
    assert app.migrate_settings(str(p)) == []
    assert json.loads(p.read_text())['kappa_only_tasks'] is True
