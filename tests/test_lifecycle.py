"""lifecycle.py: open and close with Tarkov.

Everything that touches Windows is replaced: the registry by ``FakeWinreg`` and the process
snapshot / named mutexes / events by ``FakeWin`` (an autouse fixture installs both, and a test
that forgets cannot reach the real Run key).  The two real-process tests at the bottom run a child
Python with TSH_RUN_KEY pointing at a key that does not exist and only ever *read*.
"""
import json
import os
import subprocess
import sys

import pytest

import lifecycle as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeWinreg:
    """Just the winreg surface lifecycle uses; values live in a dict keyed by (key path, name)."""
    HKEY_CURRENT_USER = 'HKCU'
    KEY_READ, KEY_SET_VALUE, REG_SZ = 1, 2, 1

    def __init__(self):
        self.values = {}
        self.writes = []

    class _Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def CreateKeyEx(self, root, path, reserved, access):
        assert root == self.HKEY_CURRENT_USER
        return self._Key(path)

    def OpenKey(self, root, path, reserved, access):
        assert root == self.HKEY_CURRENT_USER
        return self._Key(path)

    def SetValueEx(self, key, name, reserved, kind, value):
        self.values[(key.path, name)] = value
        self.writes.append((key.path, name, value))

    def QueryValueEx(self, key, name):
        try:
            return self.values[(key.path, name)], self.REG_SZ
        except KeyError:
            raise FileNotFoundError(name) from None

    def DeleteValue(self, key, name):
        try:
            del self.values[(key.path, name)]
        except KeyError:
            raise FileNotFoundError(name) from None


class FakeWin:
    """Process list + named mutexes/events, same method names as lifecycle.Win32."""

    def __init__(self):
        self.procs = {}                  # exe (lower) -> [pids]
        self.mutexes = set()
        self.events = {}                 # name -> signalled?
        self.snapshot_fails = False
        self.stop_after_waits = None     # signal the stop event after N waits (loop tests)
        self.waits = 0
        self.trims = 0
        self.priority_lowered = False

    def find_pids(self, exe):
        if self.snapshot_fails:
            raise OSError('snapshot failed')
        return list(self.procs.get(exe.lower(), []))

    def create_mutex(self, name):
        existed = name in self.mutexes
        self.mutexes.add(name)
        return existed

    def mutex_exists(self, name):
        return name in self.mutexes

    def create_event(self, name):
        self.events[name] = False
        return name

    def signal_event(self, name):
        if name not in self.events:
            return False
        self.events[name] = True
        return True

    def wait(self, handle, timeout_ms):
        self.waits += 1
        if self.stop_after_waits is not None and self.waits >= self.stop_after_waits:
            self.events[handle] = True
        return self.events.get(handle, False)

    def lower_priority(self):
        self.priority_lowered = True

    def trim_memory(self):
        self.trims += 1

    def memory(self):
        return {'working_set_mb': 1.0}

    def start_game(self, pid=100, exe=L.GAME_EXE):
        self.procs.setdefault(exe.lower(), []).append(pid)

    def stop_game(self, exe=L.GAME_EXE):
        self.procs.pop(exe.lower(), None)


@pytest.fixture(autouse=True)
def fakes(monkeypatch, tmp_path):
    reg, win = FakeWinreg(), FakeWin()
    monkeypatch.setattr(L, 'winreg', reg)
    monkeypatch.setattr(L, '_win_singleton', win)
    monkeypatch.setattr(L, 'BASE', str(tmp_path))                 # data/lifecycle.log + settings.json live here
    monkeypatch.delenv('TSH_PORT', raising=False)
    monkeypatch.delenv('TSH_RUN_KEY', raising=False)
    monkeypatch.setitem(L._session, 'managed', False)
    monkeypatch.setitem(L._session, 'follower', None)
    monkeypatch.setitem(L._session, 'with_tarkov', False)
    monkeypatch.setitem(L._session, 'background', False)
    monkeypatch.setitem(L._session, 'settings', None)
    spawned = []
    monkeypatch.setattr(L, 'spawn', lambda *extra: spawned.append(extra))   # nothing real is ever started
    reg.spawned = spawned
    win.reg = reg
    return win


@pytest.fixture
def reg(fakes):
    return fakes.reg


def settings(**kw):
    return L.with_defaults(kw)


# ---------------------------------------------------------------------------
# settings / names / command lines
# ---------------------------------------------------------------------------

def test_lifecycle_keys_are_in_default_settings_and_with_defaults():
    import app as tsh_app
    d = tsh_app.default_settings()
    assert d['start_with_windows'] is True
    assert d['follow_tarkov'] is True
    assert d['show_window_on_game_start'] is True
    # an existing settings.json without the keys, and one with an explicit False, are both honoured
    assert L.with_defaults({'hotkey': 'x'})['follow_tarkov'] is True
    assert L.with_defaults({'follow_tarkov': False})['follow_tarkov'] is False
    assert L.with_defaults(None) == L.DEFAULTS


def test_game_exe_default_and_override():
    assert L.game_exe({}) == 'EscapeFromTarkov.exe'
    assert L.game_exe({'auto_scan_exe': 'Other.exe'}) == 'Other.exe'


def test_object_names_get_a_port_suffix_only_off_the_default_port(monkeypatch):
    assert L.app_mutex_name() == 'Local\\TarkovStashHelperApp'
    assert L.watch_mutex_name() == 'Local\\TarkovStashHelperWatch'
    monkeypatch.setenv('TSH_PORT', '8899')
    assert L.app_mutex_name() == 'Local\\TarkovStashHelperApp.8899'
    assert L.watch_stop_name() == 'Local\\TarkovStashHelperWatchStop.8899'


def test_command_line_source_and_frozen(monkeypatch):
    monkeypatch.setattr(L, 'FROZEN', True)
    monkeypatch.setattr(sys, 'executable', r'C:\Apps\Tarkov Stash Helper\TarkovStashHelper.exe')
    assert L.command_args('--watch') == [sys.executable, '--watch']
    assert L.run_command() == r'"C:\Apps\Tarkov Stash Helper\TarkovStashHelper.exe" --watch'
    monkeypatch.setattr(L, 'FROZEN', False)
    monkeypatch.setattr(L, 'BASE', r'D:\src\tsh')
    monkeypatch.setattr(L, '_gui_python', lambda: r'C:\Python312\pythonw.exe')
    assert L.run_command() == r'"C:\Python312\pythonw.exe" "D:\src\tsh\app.py" --watch'


def test_settings_file_reads_defaults_then_changes_and_survives_garbage(tmp_path):
    p = tmp_path / 'settings.json'
    f = L.SettingsFile(str(p))
    assert f() == L.DEFAULTS                                          # no file yet
    p.write_text(json.dumps({'follow_tarkov': False}), encoding='utf-8')
    assert f()['follow_tarkov'] is False and f()['start_with_windows'] is True
    p.write_text('{ half written', encoding='utf-8')
    assert f()['follow_tarkov'] is False                              # keeps the last good copy


# ---------------------------------------------------------------------------
# startup registration (fake registry)
# ---------------------------------------------------------------------------

def test_register_writes_the_run_value_once_and_is_idempotent(reg):
    assert not L.is_registered()
    assert L.register() is True
    assert L.is_registered()
    assert L.registered_command() == L.run_command()
    assert reg.writes and reg.writes[0][0] == L.RUN_KEY and reg.writes[0][1] == 'TarkovStashHelper'
    assert L.register() is False                                      # nothing to change
    assert len(reg.writes) == 1


def test_register_repoints_a_stale_path(reg):
    stale = '"C:\\old\\place\\TarkovStashHelper.exe" --watch'          # the exe was moved since last launch
    reg.values[(L.RUN_KEY, 'TarkovStashHelper')] = stale
    assert L.is_registered()
    assert L.register() is True
    assert L.registered_command() == L.run_command() != stale


def test_unregister_removes_the_value_and_tolerates_absence(reg):
    L.register()
    assert L.unregister() is True
    assert not L.is_registered()
    assert L.unregister() is False                                    # already gone: no error


def test_registration_uses_the_overridden_key_for_tests(reg, monkeypatch):
    monkeypatch.setenv('TSH_RUN_KEY', r'Software\TarkovStashHelperE2E\Run')
    L.register()
    assert (r'Software\TarkovStashHelperE2E\Run', 'TarkovStashHelper') in reg.values
    assert (L.RUN_KEY, 'TarkovStashHelper') not in reg.values


# ---------------------------------------------------------------------------
# the watcher
# ---------------------------------------------------------------------------

def make_watcher(win, **kw):
    started = []
    box = {'s': settings(**kw)}
    w = L.Watcher(win, lambda: box['s'], lambda *flags: started.append(flags), log_fn=lambda m: None)
    w.box, w.started = box, started
    return w


def test_watcher_idle_without_the_game(fakes):
    w = make_watcher(fakes)
    for _ in range(5):
        assert w.tick() is None
    assert w.started == []


def test_watcher_starts_the_app_once_when_the_game_appears(fakes):
    w = make_watcher(fakes)
    w.tick()
    fakes.start_game(pid=500)
    assert w.tick() is None
    assert w.started == [('--with-tarkov',)]
    for _ in range(5):
        w.tick()
    assert len(w.started) == 1                                        # same process: no second start


def test_watcher_does_not_fight_a_user_who_quit_the_app_mid_session(fakes):
    w = make_watcher(fakes)
    fakes.start_game(pid=500)
    w.tick()
    assert len(w.started) == 1
    # the app was quit from the tray: no app mutex, game still running
    for _ in range(5):
        w.tick()
    assert len(w.started) == 1


def test_watcher_starts_the_app_again_for_the_next_game_process(fakes):
    w = make_watcher(fakes)
    fakes.start_game(pid=500)
    w.tick()
    fakes.stop_game()
    w.tick()
    fakes.start_game(pid=777)
    w.tick()
    assert w.started == [('--with-tarkov',), ('--with-tarkov',)]


def test_watcher_does_nothing_when_the_app_is_already_open(fakes):
    w = make_watcher(fakes)
    fakes.mutexes.add(L.app_mutex_name())                             # opened by hand before the game
    fakes.start_game()
    w.tick()
    assert w.started == []
    fakes.mutexes.clear()                                             # ...then closed: still the same game process
    w.tick()
    assert w.started == []


def test_watcher_game_already_running_at_startup_opens_the_app(fakes):
    fakes.start_game()                                                # e.g. sign-in with the game up, or first run mid-raid
    w = make_watcher(fakes)
    w.tick()
    assert w.started == [('--with-tarkov',)]


def test_watcher_retries_a_failed_start_on_the_next_tick(fakes):
    fakes.start_game()
    calls = []

    def flaky(*flags):
        calls.append(flags)
        if len(calls) == 1:
            raise OSError('exe is being replaced')
    w = L.Watcher(fakes, lambda: settings(), flaky, log_fn=lambda m: None)
    with pytest.raises(OSError):
        w.tick()
    w.tick()
    assert len(calls) == 2
    w.tick()
    assert len(calls) == 2                                            # now handled


def test_watcher_survives_a_snapshot_failure(fakes):
    w = make_watcher(fakes)
    fakes.snapshot_fails = True
    assert w.tick() is None
    assert w.started == []


def test_watcher_exits_when_start_with_windows_is_turned_off(fakes):
    w = make_watcher(fakes)
    w.box['s'] = settings(start_with_windows=False)
    assert w.tick() == 'stop'


def test_watcher_without_follow_opens_the_app_once_at_signin_to_the_tray(fakes):
    w = make_watcher(fakes, follow_tarkov=False)
    fakes.start_game()                                                # the game is irrelevant in this mode
    w.tick()
    w.tick()
    assert w.started == [('--background',)]


def test_watcher_without_follow_leaves_an_already_open_app_alone(fakes):
    fakes.mutexes.add(L.app_mutex_name())
    w = make_watcher(fakes, follow_tarkov=False)
    w.tick()
    assert w.started == []


def test_watch_main_second_watcher_exits_at_once(fakes):
    fakes.mutexes.add(L.watch_mutex_name())
    assert L.watch_main([], win=fakes) == 0
    assert not fakes.priority_lowered                                  # it never got as far as the loop


def test_watch_main_loops_until_the_stop_event(fakes, monkeypatch):
    fakes.start_game()
    fakes.stop_after_waits = 3
    started = []
    monkeypatch.setattr(L, 'spawn', lambda *flags: started.append(flags))
    assert L.watch_main([], win=fakes) == 0
    assert started == [('--with-tarkov',)]
    assert fakes.priority_lowered and fakes.trims >= 1
    assert fakes.waits == 3
    assert L.watch_mutex_name() in fakes.mutexes                      # held for the life of the process


def test_watch_main_loop_survives_an_exception(fakes, monkeypatch):
    fakes.stop_after_waits = 3
    boom = {'n': 0}

    def bad_tick(self):
        boom['n'] += 1
        raise RuntimeError('boom')
    monkeypatch.setattr(L.Watcher, 'tick', bad_tick)
    assert L.watch_main([], win=fakes) == 0
    assert boom['n'] == 3


def test_stop_watcher_signals_the_event_and_reports_when_gone(fakes, monkeypatch):
    assert L.stop_watcher(win=fakes) is True                          # nothing running
    fakes.mutexes.add(L.watch_mutex_name())
    fakes.events[L.watch_stop_name()] = False

    real = fakes.signal_event

    def signal_and_exit(name):
        ok = real(name)
        fakes.mutexes.discard(L.watch_mutex_name())                   # the watcher reacts and exits
        return ok
    monkeypatch.setattr(fakes, 'signal_event', signal_and_exit)
    assert L.stop_watcher(win=fakes) is True
    assert fakes.events[L.watch_stop_name()] is True


# ---------------------------------------------------------------------------
# the app's follower
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def make_follower(win, started_with_tarkov=False, **kw):
    clock = Clock()
    box = {'s': settings(**kw)}
    f = L.Follower(win, lambda: box['s'], started_with_tarkov, clock=clock)
    f.box, f.clock_ = box, clock
    return f


def run_for(f, seconds, step=5):
    """Tick every ``step`` s for ``seconds``; True if it asked to quit at any point."""
    for _ in range(int(seconds // step)):
        f.clock_.advance(step)
        if f.tick():
            return True
    return False


def test_manual_launch_without_the_game_never_closes(fakes):
    f = make_follower(fakes)
    assert run_for(f, 3600) is False
    assert not f.armed


def test_manual_launch_then_game_seen_then_gone_closes_after_the_grace(fakes):
    f = make_follower(fakes)
    run_for(f, 60)
    fakes.start_game()
    assert run_for(f, 30) is False
    assert f.armed and f.game_running
    fakes.stop_game()
    f.clock_.advance(5)
    assert f.tick() is False                                          # 0 s gone
    assert run_for(f, L.FOLLOW_GONE_S - 5 - 1) is False               # still inside the grace window
    assert run_for(f, 10) is True


def test_a_game_restart_within_the_grace_does_not_close_the_app(fakes):
    f = make_follower(fakes)
    fakes.start_game()
    f.tick()
    fakes.stop_game()
    run_for(f, 10)
    fakes.start_game(pid=900)
    assert run_for(f, 60) is False
    fakes.stop_game()
    assert run_for(f, 10) is False                                    # the grace restarts from zero
    assert run_for(f, 10) is True


def test_started_with_tarkov_closes_even_if_it_never_saw_the_game(fakes):
    f = make_follower(fakes, started_with_tarkov=True)               # game exited before the first poll
    assert run_for(f, 10) is False
    assert run_for(f, 10) is True


def test_follow_off_never_closes_even_when_started_with_tarkov(fakes):
    f = make_follower(fakes, started_with_tarkov=True, follow_tarkov=False)
    fakes.start_game()
    f.tick()
    fakes.stop_game()
    assert run_for(f, 600) is False


def test_turning_follow_on_after_the_game_is_gone_does_not_close_the_app(fakes):
    f = make_follower(fakes, follow_tarkov=False)
    fakes.start_game()
    run_for(f, 30)
    fakes.stop_game()
    run_for(f, 120)
    f.box['s'] = settings(follow_tarkov=True)                         # the user is in the settings page
    assert run_for(f, 600) is False
    fakes.start_game()                                                # but the next game session is followed
    run_for(f, 10)
    fakes.stop_game()
    assert run_for(f, 60) is True


def test_turning_follow_off_mid_session_disarms(fakes):
    f = make_follower(fakes, started_with_tarkov=True)
    fakes.start_game()
    run_for(f, 10)
    assert f.armed
    f.box['s'] = settings(follow_tarkov=False)
    fakes.stop_game()
    assert run_for(f, 600) is False


def test_a_snapshot_failure_is_not_taken_for_the_game_exiting(fakes):
    f = make_follower(fakes, started_with_tarkov=True)
    fakes.start_game()
    f.tick()
    fakes.stop_game()
    fakes.snapshot_fails = True
    assert run_for(f, 600) is False
    fakes.snapshot_fails = False
    assert run_for(f, 10) is False
    assert run_for(f, 20) is True


# ---------------------------------------------------------------------------
# the app session / settings application
# ---------------------------------------------------------------------------

def begin(win, argv=(), **kw):
    return L.begin_app_session(list(argv), lambda: settings(**kw), lambda: None, win=win, start_thread=False)


def test_first_launch_registers_and_starts_the_watcher(fakes, reg):
    session = begin(fakes)
    assert session == {'with_tarkov': False, 'background': False, 'hidden': False, 'quiet': False,
                       'first_registration': True}
    assert L.is_registered()
    assert reg.spawned == [('--watch',)]                              # automatic from now on, no reboot
    assert L.app_mutex_name() in fakes.mutexes


def test_later_launch_does_not_start_a_second_watcher_or_announce_again(fakes, reg):
    L.register()
    fakes.mutexes.add(L.watch_mutex_name())
    session = begin(fakes)
    assert session['first_registration'] is False
    assert reg.spawned == []


def test_start_with_windows_off_registers_nothing(fakes, reg):
    begin(fakes, start_with_windows=False)
    assert not L.is_registered()
    assert reg.spawned == []


def test_second_manual_copy_asks_the_first_to_show_itself_and_exits(fakes, monkeypatch):
    fakes.mutexes.add(L.app_mutex_name())
    shown = []
    monkeypatch.setattr(L, 'show_running_instance', lambda: shown.append(1) or True)
    assert begin(fakes) is None
    assert shown == [1]


def test_copy_started_by_the_watcher_never_pops_up_the_other_window(fakes, monkeypatch):
    fakes.mutexes.add(L.app_mutex_name())
    shown = []
    monkeypatch.setattr(L, 'show_running_instance', lambda: shown.append(1) or True)
    assert begin(fakes, ['--with-tarkov']) is None
    assert shown == []


def test_window_policy_follows_the_flags_and_settings(fakes):
    s = begin(fakes, ['--with-tarkov'])
    assert s['with_tarkov'] and s['quiet'] and not s['hidden']        # opens, but without taking the focus
    fakes.mutexes.clear()
    s = begin(fakes, ['--with-tarkov'], show_window_on_game_start=False)
    assert s['quiet'] and s['hidden']                                 # straight to the tray
    fakes.mutexes.clear()
    s = begin(fakes, ['--background'])
    assert s['quiet'] and s['hidden'] and not s['with_tarkov']
    fakes.mutexes.clear()
    s = begin(fakes)
    assert not s['quiet'] and not s['hidden']                         # a normal launch by hand


def test_apply_settings_is_inert_before_the_app_session(fakes, reg):
    assert L.apply_settings(settings(start_with_windows=True)) is None
    assert not L.is_registered() and reg.spawned == []


def test_apply_settings_toggles_registration_and_the_watcher(fakes, reg, monkeypatch):
    begin(fakes)                                                      # session starts, registers, spawns once
    reg.spawned.clear()
    fakes.mutexes.discard(L.watch_mutex_name())
    L.unregister()

    L.apply_settings(settings(start_with_windows=True))
    assert L.is_registered() and reg.spawned == [('--watch',)]

    fakes.mutexes.add(L.watch_mutex_name())
    fakes.events[L.watch_stop_name()] = False
    L.apply_settings(settings(start_with_windows=True, hotkey='x'))   # any other save: nothing more to do
    assert reg.spawned == [('--watch',)]

    real = fakes.signal_event
    monkeypatch.setattr(fakes, 'signal_event', lambda n: (fakes.mutexes.discard(L.watch_mutex_name()), real(n))[1])
    L.apply_settings(settings(start_with_windows=False))
    assert not L.is_registered()
    assert fakes.events[L.watch_stop_name()] is True                  # the watcher was told to exit


def test_status_reports_the_five_documented_keys(fakes):
    begin(fakes, ['--with-tarkov'])
    fakes.start_game()
    st = L.status(settings())
    assert {k: st[k] for k in ('registered', 'watcher_running', 'game_running', 'follow_tarkov',
                               'started_with_tarkov')} == {
        'registered': True, 'watcher_running': False, 'game_running': True, 'follow_tarkov': True,
        'started_with_tarkov': True}


def test_blueprint_status_and_show(fakes):
    from flask import Flask
    shown = []
    app = Flask(__name__)
    app.register_blueprint(L.make_blueprint(lambda: settings(follow_tarkov=False), lambda: shown.append(1)))
    c = app.test_client()
    r = c.get('/api/lifecycle/status')
    assert r.status_code == 200
    assert r.get_json()['follow_tarkov'] is False
    assert set(r.get_json()) >= {'registered', 'watcher_running', 'game_running', 'follow_tarkov',
                                 'started_with_tarkov'}
    assert c.post('/api/lifecycle/show').get_json() == {'ok': True} and shown == [1]


def test_save_settings_route_applies_the_lifecycle_settings(monkeypatch, tmp_path):
    import app as tsh_app
    monkeypatch.setattr(tsh_app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    applied = []
    monkeypatch.setattr(tsh_app.lifecycle, 'apply_settings', lambda s: applied.append(dict(s)))
    c = tsh_app.app.test_client()
    body = {'hotkey': '<ctrl>+x', 'start_with_windows': False, 'follow_tarkov': True}
    assert c.post('/api/settings', json=body).get_json()['ok'] is True
    assert len(applied) == 1 and all(applied[0][k] == v for k, v in body.items())   # merged, validated
    got = c.get('/api/settings').get_json()
    assert got['start_with_windows'] is False and got['follow_tarkov'] is True
    assert got['show_window_on_game_start'] is True                   # filled in: the file never had it


# ---------------------------------------------------------------------------
# window options
# ---------------------------------------------------------------------------

def test_window_kwargs():
    new = {'focus', 'hidden', 'minimized', 'width'}
    assert L.window_kwargs(new, quiet=False, hidden=False) == {}
    assert L.window_kwargs(new, quiet=True, hidden=False) == {'focus': False}
    assert L.window_kwargs(new, quiet=True, hidden=True) == {'focus': False, 'hidden': True}
    old = {'hidden', 'minimized'}                                     # a pywebview without `focus`
    assert L.window_kwargs(old, quiet=True, hidden=False) == {'minimized': True}
    assert L.window_kwargs(set(), quiet=True, hidden=True) == {}


def test_allow_activation_lifts_the_no_activate_style_once(fakes):
    calls = []

    class W:
        focus = False

    class Win:
        def windows_of(self, pid, title):
            return [4242]

        def set_no_activate(self, hwnd, on):
            calls.append((hwnd, on))
    w = W()
    assert L.allow_activation(w, win=Win()) is True
    assert calls == [(4242, False)] and w.focus is True
    assert L.allow_activation(w, win=Win()) is False                  # a normal window: nothing to do


# ---------------------------------------------------------------------------
# real (child) processes: read-only
# ---------------------------------------------------------------------------

def _child_env():
    env = dict(os.environ)
    env['TSH_RUN_KEY'] = r'Software\TarkovStashHelperNoSuchKey\Run'
    env['TSH_PORT'] = '8898'
    env.pop('PYTEST_CURRENT_TEST', None)
    return env


def test_importing_lifecycle_loads_nothing_heavy():
    code = ('import sys; sys.path.insert(0, %r); import lifecycle; '
            'print(",".join(m for m in lifecycle.HEAVY_MODULES if m in sys.modules))' % ROOT)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60,
                         env=_child_env())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '', f'lifecycle pulled in: {out.stdout.strip()}'


def test_watch_once_via_app_py_stays_light_and_starts_nothing(tmp_path):
    report = tmp_path / 'watch.json'
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'app.py'), '--watch', '--once', str(report)],
                         capture_output=True, text=True, timeout=60, env=_child_env(), cwd=str(tmp_path))
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.loads(report.read_text(encoding='utf-8'))
    assert rep['ok'] and rep['heavy_imported'] == []                  # dispatched before numpy / cv2 / flask
    assert rep['registered'] is False and rep['watcher_running'] is False and rep['app_running'] is False
    assert rep['run_command'].endswith('--watch')
    assert isinstance(rep['game_running'], bool)
