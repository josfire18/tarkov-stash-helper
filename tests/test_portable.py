"""A stranger's first run on another machine: where data lives, finding the game (BSG launcher / Steam /
other drives), the client language, the bundled Tesseract, DPI awareness and the first-run banner.

Nothing here touches the real registry, the real %APPDATA%, the network or the real data/."""
import copy
import json
import os
import sys
import zipfile

import pytest

import app
import dpi
import eftlogs
import gamelang
import lifecycle
import paths
import tarkovdata as td
import tess
from identify.grid import expected_pitch

from test_tarkovdata import FakeNet, fixture            # noqa: E402 - tests/ is on sys.path

REAL_FIND = eftlogs.find_install_dir      # conftest replaces eftlogs.find_install_dir per test (hermetic)


def touch(path, text=''):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return str(path)


# ---------------------------------------------------------------------------
# data folder: the exe's location must not matter
# ---------------------------------------------------------------------------

def env_for(tmp_path, **extra):
    return {'LOCALAPPDATA': str(tmp_path / 'Local'), **extra}


def test_packaged_exe_keeps_data_in_localappdata_wherever_it_sits(tmp_path):
    env = env_for(tmp_path)
    a = paths.resolve_data_dir(True, str(tmp_path / 'Downloads'), '', env)
    b = paths.resolve_data_dir(True, str(tmp_path / 'Desktop'), '', env)       # the exe was moved
    assert a == b == str(tmp_path / 'Local' / 'TarkovStashHelper')
    assert os.path.isdir(a)
    assert not (tmp_path / 'Downloads' / 'data').exists()                       # nothing created next to the exe


def test_moving_the_exe_keeps_the_data(tmp_path):
    env = env_for(tmp_path)
    d = paths.resolve_data_dir(True, str(tmp_path / 'Downloads'), '', env)
    touch(os.path.join(d, 'settings.json'), '{"hotkey": "f9"}')
    moved = paths.resolve_data_dir(True, str(tmp_path / 'Elsewhere' / 'deep'), '', env)
    assert moved == d and os.path.isfile(os.path.join(moved, 'settings.json'))


def test_folder_that_is_not_writable_falls_back_to_localappdata(tmp_path, monkeypatch):
    # run from source into a read-only checkout (or Program Files): the data goes to %LOCALAPPDATA%
    readonly = str(tmp_path / 'ProgramFiles' / 'tsh' / 'data')
    real = paths.writable
    monkeypatch.setattr(paths, 'writable', lambda p: False if os.path.normcase(p) == os.path.normcase(readonly) else real(p))
    got = paths.resolve_data_dir(False, '', str(tmp_path / 'ProgramFiles' / 'tsh'), env_for(tmp_path))
    assert got == str(tmp_path / 'Local' / 'TarkovStashHelper')


def test_nothing_writable_still_starts_in_the_temp_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'writable', lambda p: 'TarkovStashHelper' in p and os.path.dirname(p) != str(tmp_path / 'Local'))
    got = paths.resolve_data_dir(True, str(tmp_path / 'x'), '', env_for(tmp_path))
    assert os.path.isdir(got)
    assert os.path.normcase(got).startswith(os.path.normcase(os.path.dirname(paths.tempfile.mkdtemp())))


def test_env_override_wins(tmp_path):
    forced = tmp_path / 'fresh'
    got = paths.resolve_data_dir(True, str(tmp_path / 'e'), '', env_for(tmp_path, TSH_DATA_DIR=str(forced)))
    assert got == str(forced) and forced.is_dir()


def test_old_install_next_to_the_exe_is_moved_once(tmp_path):
    exe_dir = tmp_path / 'Games' / 'tsh'
    touch(exe_dir / 'data' / 'settings.json', '{"a": 1}')
    touch(exe_dir / 'data' / 'tmpl_src' / 'x.png', 'png')
    env = env_for(tmp_path)
    got = paths.resolve_data_dir(True, str(exe_dir), '', env)
    assert got == str(tmp_path / 'Local' / 'TarkovStashHelper')
    assert open(os.path.join(got, 'settings.json')).read() == '{"a": 1}'
    assert os.path.isfile(os.path.join(got, 'tmpl_src', 'x.png'))
    assert not (exe_dir / 'data' / 'settings.json').exists()
    assert paths.resolve_data_dir(True, str(exe_dir), '', env) == got              # idempotent


def test_failed_migration_keeps_using_the_old_folder(tmp_path, monkeypatch):
    exe_dir = tmp_path / 'tsh'
    touch(exe_dir / 'data' / 'settings.json', '{}')
    monkeypatch.setattr(paths, 'migrate_dir', lambda s, d: False)
    got = paths.resolve_data_dir(True, str(exe_dir), '', env_for(tmp_path))
    assert got == str(exe_dir / 'data')


def test_the_watcher_never_moves_anything(tmp_path):
    exe_dir = tmp_path / 'tsh'
    touch(exe_dir / 'data' / 'settings.json', '{}')
    got = paths.resolve_data_dir(True, str(exe_dir), '', env_for(tmp_path), migrate=False)
    assert got == str(exe_dir / 'data') and (exe_dir / 'data' / 'settings.json').exists()


def test_every_module_agrees_on_the_data_folder():
    from identify import config
    assert os.path.normcase(app.DATA) == os.path.normcase(config.DATA_DIR) == os.path.normcase(paths.default_data_dir())


# ---------------------------------------------------------------------------
# game install: BSG launcher, Steam, other drives, the setting
# ---------------------------------------------------------------------------

class FakeWinreg:
    HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER = 'HKLM', 'HKCU'

    def __init__(self, keys):
        self.keys = keys                       # {(hive, path): {value: data}}

    class _Key:
        def __init__(self, hive, path, values):
            self.hive, self.path, self.values = hive, path, values

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def OpenKey(self, hive, path, *a):
        if (hive, path) not in self.keys:
            raise FileNotFoundError(path)
        return self._Key(hive, path, self.keys[(hive, path)])

    def QueryValueEx(self, key, name):
        if name not in key.values:
            raise FileNotFoundError(name)
        return key.values[name], 1

    def EnumKey(self, key, i):
        subs = sorted(p[len(key.path) + 1:] for (h, p) in self.keys
                      if h == key.hive and p.startswith(key.path + '\\') and '\\' not in p[len(key.path) + 1:])
        if i >= len(subs):
            raise OSError
        return subs[i]


UNINSTALL = r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'


@pytest.fixture
def nowhere(monkeypatch):
    """No registry, no APPDATA, no drives: only what a test adds exists."""
    monkeypatch.setattr(eftlogs, 'winreg', None)
    monkeypatch.setattr(eftlogs, '_common_install_dirs', lambda: [])
    monkeypatch.delenv('APPDATA', raising=False)


def make_install(root, logs=True, exe=True):
    if exe:
        touch(os.path.join(str(root), 'EscapeFromTarkov.exe'), 'MZ')
    os.makedirs(os.path.join(str(root), 'EscapeFromTarkov_Data'), exist_ok=True)
    if logs:
        os.makedirs(os.path.join(str(root), 'Logs'), exist_ok=True)
    return str(root)


def test_bsg_launcher_registry_install(tmp_path, monkeypatch, nowhere):
    inst = make_install(tmp_path / 'Games' / 'Escape from Tarkov')
    monkeypatch.setattr(eftlogs, 'winreg', FakeWinreg({
        ('HKLM', UNINSTALL + r'\EscapeFromTarkov'): {'InstallLocation': inst}}))
    assert REAL_FIND() == os.path.normpath(inst)


def test_bsg_launcher_games_root_setting_finds_an_install_on_another_drive(tmp_path, monkeypatch, nowhere):
    inst = make_install(tmp_path / 'E_drive' / 'Escape from Tarkov')
    touch(tmp_path / 'AppData' / 'Battlestate Games' / 'BsgLauncher' / 'settings',
          json.dumps({'gamesRootDir': str(tmp_path / 'E_drive'), 'language': 'en'}))
    monkeypatch.setenv('APPDATA', str(tmp_path / 'AppData'))
    assert REAL_FIND() == os.path.normpath(inst)


def write_steam(tmp_path, monkeypatch, library_games, installdir='Escape from Tarkov', manifest=True):
    steam = tmp_path / 'Steam'
    lib2 = tmp_path / 'SteamLibrary'
    vdf = ('"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
           % (str(steam).replace('\\', '\\\\'), str(lib2).replace('\\', '\\\\')))
    touch(steam / 'steamapps' / 'libraryfolders.vdf', vdf)
    lib = lib2 if library_games == 'second' else steam
    inst = make_install(lib / 'steamapps' / 'common' / installdir, logs=False)
    if manifest:
        touch(lib / 'steamapps' / 'appmanifest_3932890.acf',
              '"AppState"\n{\n\t"appid"\t"3932890"\n\t"name"\t"Escape from Tarkov"\n\t"installdir"\t"%s"\n}\n' % installdir)
    monkeypatch.setattr(eftlogs, 'winreg', FakeWinreg({
        ('HKCU', r'Software\Valve\Steam'): {'SteamPath': str(steam).replace('\\', '/')}}))
    return inst


def test_steam_install_in_a_second_library_folder(tmp_path, monkeypatch, nowhere):
    inst = write_steam(tmp_path, monkeypatch, 'second')
    assert eftlogs.install_candidates() == [os.path.normpath(inst)]
    assert REAL_FIND(require_logs=False) == os.path.normpath(inst)
    assert REAL_FIND() is None             # never played yet: no Logs folder, so no log reading


def test_steam_install_found_through_its_manifest_when_the_folder_was_renamed(tmp_path, monkeypatch, nowhere):
    inst = write_steam(tmp_path, monkeypatch, 'main', installdir='EFT live')
    assert REAL_FIND(require_logs=False) == os.path.normpath(inst)


def test_steam_app_uninstall_key(tmp_path, monkeypatch, nowhere):
    inst = make_install(tmp_path / 'weird' / 'place', logs=False)
    monkeypatch.setattr(eftlogs, 'winreg', FakeWinreg({
        ('HKLM', UNINSTALL): {},
        ('HKLM', UNINSTALL + r'\Steam App 3932890'): {'DisplayName': 'Escape from Tarkov', 'InstallLocation': inst},
        ('HKLM', UNINSTALL + r'\Steam App 1'): {'DisplayName': 'Other Game', 'InstallLocation': str(tmp_path)}}))
    assert REAL_FIND(require_logs=False) == os.path.normpath(inst)


def test_arena_is_not_mistaken_for_the_game(tmp_path, monkeypatch, nowhere):
    arena = make_install(tmp_path / 'Escape from Tarkov Arena')
    monkeypatch.setattr(eftlogs, 'winreg', FakeWinreg({
        ('HKLM', UNINSTALL + r'\EscapeFromTarkov'): {'InstallLocation': arena}}))
    assert not eftlogs.is_install(arena)
    assert eftlogs.install_candidates() == [] or arena not in eftlogs.install_candidates()
    assert REAL_FIND(require_logs=False) is None


def test_setting_may_point_at_the_logs_folder_and_beats_detection(tmp_path, nowhere):
    inst = make_install(tmp_path / 'mine')
    assert REAL_FIND(os.path.join(inst, 'Logs')) == os.path.normpath(inst)
    assert REAL_FIND(f'"{inst}"') == os.path.normpath(inst)


def test_install_with_logs_is_preferred_over_one_without(tmp_path, monkeypatch, nowhere):
    a = make_install(tmp_path / 'a' / 'Escape from Tarkov', logs=False)
    b = make_install(tmp_path / 'b' / 'Escape from Tarkov', logs=True)
    monkeypatch.setattr(eftlogs, '_common_install_dirs', lambda: [a, b])
    assert REAL_FIND() == os.path.normpath(b)
    assert REAL_FIND(require_logs=False) == os.path.normpath(b)


def test_a_broken_source_does_not_hide_the_others(tmp_path, monkeypatch, nowhere):
    inst = make_install(tmp_path / 'ok')
    monkeypatch.setattr(eftlogs, '_steam_install_dirs', lambda: 1 / 0)
    monkeypatch.setattr(eftlogs, '_common_install_dirs', lambda: [inst])
    assert REAL_FIND() == os.path.normpath(inst)


def test_choose_folder_accepts_an_install_and_rejects_other_folders(tmp_path, monkeypatch):
    inst = make_install(tmp_path / 'Escape from Tarkov', logs=False)
    saved = {}
    monkeypatch.setattr(app, '_save_settings', lambda s, mode=None: saved.update(s))
    monkeypatch.setattr(app, 'scan_task_logs', lambda full=False: None)
    bad = app.set_install_dir(str(tmp_path / 'Documents'))
    assert bad['ok'] is False and 'EscapeFromTarkov.exe' in bad['error']
    for picked in (inst, str(tmp_path)):                 # the install itself, or the folder that holds it
        saved.clear()
        res = app.set_install_dir(picked)
        assert res['ok'] and saved['eft_install_dir'] == os.path.normpath(inst)


# ---------------------------------------------------------------------------
# client language
# ---------------------------------------------------------------------------

def write_game_ini(tmp_path, lang):
    touch(tmp_path / 'Battlestate Games' / 'Escape from Tarkov' / 'Settings' / 'Game.ini',
          json.dumps({'Version': 1, 'Language': lang, 'FieldOfView': 69}))
    return str(tmp_path)


def test_language_read_from_game_ini(tmp_path):
    assert gamelang.detect(appdata=write_game_ini(tmp_path, 'ge')) == \
        {'code': 'ge', 'name': 'German', 'doc': 'de', 'supported': True, 'label_ok': True}
    assert gamelang.detect(appdata=write_game_ini(tmp_path, 'en'))['doc'] is None


def test_launcher_language_is_the_fallback_before_the_game_ever_ran(tmp_path):
    touch(tmp_path / 'Battlestate Games' / 'BsgLauncher' / 'settings', json.dumps({'language': 'fr'}))
    assert gamelang.detect(appdata=str(tmp_path))['doc'] == 'fr'


def test_no_config_means_english(tmp_path):
    d = gamelang.detect(appdata=str(tmp_path))
    assert d['code'] == 'en' and d['label_ok'] and gamelang.notice(d) is None


@pytest.mark.parametrize('code', ['ru', 'jp', 'kr', 'ch', 'xx'])
def test_unreadable_scripts_fall_back_to_picture_only_with_one_notice(code):
    d = gamelang.describe(code)
    assert not d['label_ok']
    note = gamelang.notice(d)
    assert note and 'picture' in note and note.count('.') == 1


def test_every_client_language_maps_to_an_existing_translation_document():
    published = {'cs', 'de', 'en', 'es', 'fr', 'hu', 'it', 'ja', 'ko', 'pl', 'pt', 'ro', 'ru', 'sk', 'tr', 'zh'}
    assert {v[0] for v in gamelang.LANGUAGES.values()} <= published


def test_picture_only_engine_settings_for_an_unreadable_language(monkeypatch):
    monkeypatch.setattr(app, '_game_language', lambda s=None: gamelang.describe('ru'))
    es = app._engine_settings({})
    assert es.use_ocr is False and es.extra['label_anchor'] is False
    monkeypatch.setattr(app, '_game_language', lambda s=None: gamelang.describe('de'))
    monkeypatch.setattr(app, '_font_ready', lambda s=None: True)
    es = app._engine_settings({})
    assert es.use_ocr and 'label_anchor' not in es.extra


def fake_lang_doc(en_body, translations):
    body = copy.deepcopy(en_body)
    for k, v in translations.items():
        body['data'][k] = v
    return body


def test_prices_carry_the_labels_in_the_clients_language(monkeypatch, tmp_path):
    net = FakeNet()
    net.docs['items_de'] = fake_lang_doc(net.docs['items_en'],
                                         {'5447a9cd4bdc2dbd208b4567 ShortName': 'M4A1 (de)',
                                          '590c695186f7741e566b64a2 ShortName': 'Augmentin'})
    monkeypatch.setattr(td, '_http_get', net)
    for k, v in (('MIN_ITEMS', 3), ('MIN_TASKS', 1), ('MIN_STATIONS', 1)):
        monkeypatch.setattr(td, k, v)
    meta = td.load_meta(str(tmp_path / 'm.json'))
    out = td.refresh_prices(str(tmp_path / 'p.json'), meta, None, lang='de')
    items = {i['id']: i for i in out['cache']['items']}
    assert out['cache']['label_lang'] == 'de'
    assert items['5447a9cd4bdc2dbd208b4567']['labelShort'] == 'M4A1 (de)'
    assert items['5447a9cd4bdc2dbd208b4567']['shortName'] == 'M4A1'                # the English name stays
    assert 'labelShort' not in items['590c695186f7741e566b64a2']                   # same text: no extra field
    # unchanged next time, and a different language re-converts instead of answering "unchanged"
    assert td.refresh_prices(str(tmp_path / 'p.json'), meta, out['cache'], lang='de')['status'] == 'unchanged'
    plain = td.refresh_prices(str(tmp_path / 'p.json'), meta, out['cache'], lang=None)
    assert plain['status'] == 'updated' and 'label_lang' not in plain['cache']
    assert all('labelShort' not in i for i in plain['cache']['items'])


def test_a_missing_translation_degrades_to_english_instead_of_failing(monkeypatch, tmp_path):
    net = FakeNet()
    net.status['items_pl'] = 404
    monkeypatch.setattr(td, '_http_get', net)
    for k, v in (('MIN_ITEMS', 3), ('MIN_TASKS', 1), ('MIN_STATIONS', 1)):
        monkeypatch.setattr(td, k, v)
    meta = td.load_meta(str(tmp_path / 'm.json'))
    out = td.refresh_prices(str(tmp_path / 'p.json'), meta, None, lang='pl')
    assert out['status'] == 'updated' and 'label_lang' not in out['cache']
    assert 'items_pl' in out['cache']['label_lang_error']
    assert all('labelShort' not in i for i in out['cache']['items'])


def test_catalog_uses_the_label_text_as_the_short_name():
    from identify import catalog
    src = open(catalog.__file__, encoding='utf-8').read()
    assert src.count("it.get('labelShort') or it.get('shortName')") >= 3


# ---------------------------------------------------------------------------
# Tesseract inside the exe
# ---------------------------------------------------------------------------

def make_tess_zip(path, stamp='tesseract-test-1'):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('tesseract.exe', 'MZ')
        z.writestr('tessdata/eng.traineddata', 'x')
        z.writestr('.stamp', stamp)
    return str(path)


def test_bundled_tesseract_is_unpacked_once_into_the_data_folder(tmp_path):
    meipass = tmp_path / 'MEI'
    meipass.mkdir()
    make_tess_zip(meipass / 'tesseract.zip')
    data = tmp_path / 'data'
    exe, tessdata = tess.find(str(data), str(meipass), str(tmp_path / 'src'))
    assert exe == str(data / 'tesseract' / 'tesseract.exe') and os.path.isfile(exe)
    assert tessdata == str(data / 'tesseract' / 'tessdata')
    marker = data / 'tesseract' / 'marker'
    marker.write_text('x')
    tess.find(str(data), str(meipass), str(tmp_path / 'src'))
    assert marker.exists()                                                  # same stamp: not unpacked again
    make_tess_zip(meipass / 'tesseract.zip', 'tesseract-test-2')            # a newer exe bundles a newer Tesseract
    tess.find(str(data), str(meipass), str(tmp_path / 'src'))
    assert not marker.exists() and (data / 'tesseract' / 'tesseract.exe').exists()


def test_damaged_bundle_falls_back_to_a_system_install(tmp_path, monkeypatch):
    meipass = tmp_path / 'MEI'
    meipass.mkdir()
    (meipass / 'tesseract.zip').write_bytes(b'not a zip')
    monkeypatch.setattr(tess.shutil, 'which', lambda n: r'C:\somewhere\tesseract.exe')
    assert tess.find(str(tmp_path / 'data'), str(meipass), str(tmp_path / 'src')) == (r'C:\somewhere\tesseract.exe', None)


def test_configure_points_pytesseract_and_tessdata_at_the_bundle(tmp_path):
    meipass = tmp_path / 'MEI'
    meipass.mkdir()
    make_tess_zip(meipass / 'tesseract.zip')

    class PT:
        class pytesseract:
            tesseract_cmd = 'tesseract'
    env = {}
    cmd = tess.configure(PT, env, data_dir=str(tmp_path / 'd'), meipass=str(meipass), source_dir=str(tmp_path))
    assert PT.pytesseract.tesseract_cmd == cmd and env['TESSDATA_PREFIX'].endswith('tessdata')


# ---------------------------------------------------------------------------
# DPI, resolutions
# ---------------------------------------------------------------------------

class FakeFn:
    def __init__(self, ret, err=0):
        self.ret, self.err = ret, err
        self.argtypes = self.restype = None
        self.calls = []

    def __call__(self, *a):
        self.calls.append(a)
        return self.ret


def test_dpi_prefers_per_monitor_v2():
    u = type('U', (), {'SetProcessDpiAwarenessContext': FakeFn(True)})()
    assert dpi.ensure(u, None) == 'per-monitor-v2'
    import ctypes
    assert u.SetProcessDpiAwarenessContext.calls[0][0].value == ctypes.c_void_p(-4).value


def test_dpi_falls_back_through_the_older_apis():
    class U:
        SetProcessDpiAwarenessContext = FakeFn(False)
        SetProcessDPIAware = FakeFn(True)

    class S:
        SetProcessDpiAwareness = FakeFn(0x80004005)             # fails
    assert dpi.ensure(U(), S()) == 'system'

    class S2:
        SetProcessDpiAwareness = FakeFn(0)
    assert dpi.ensure(U(), S2()) == 'per-monitor'
    assert dpi.ensure(object(), None) == 'unavailable'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
def test_a_real_process_becomes_per_monitor_v2_aware():
    import subprocess
    out = subprocess.run([sys.executable, '-c',
                          'import sys; sys.path.insert(0, %r); import dpi; a = dpi.ensure(); print(a, dpi.current())'
                          % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))],
                         capture_output=True, text=True, timeout=30)
    assert out.stdout.split() == ['per-monitor-v2', 'per-monitor-v2'], out.stdout + out.stderr


@pytest.mark.parametrize('w,h,pitch', [
    (1280, 720, 42.0), (1920, 1080, 63.0), (2560, 1440, 84.0), (3840, 2160, 126.0),
    (3440, 1440, 84.0), (5120, 1440, 84.0), (2560, 1080, 63.0),     # ultrawide / super-ultrawide scale by height
    (1600, 1200, 52.5),                                             # 4:3 scales by width
    (1536, 864, 50.4),                                              # 1080p at 125 % DPI as the process sees it unaware
])
def test_the_ui_scale_is_relative_never_a_fixed_resolution(w, h, pitch):
    assert expected_pitch(w, h) == pytest.approx(pitch)


def test_no_module_hard_codes_a_resolution():
    """Numeric literals 1440 / 2560 in the engine would mean a layout that only fits one screen.  (The
    scene module's icon spacing is a fraction of the frame height, written as 46 / 1440.)"""
    import tokenize
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for sub in ('identify', 'autoscan', ''):
        d = os.path.join(root, sub)
        for fn in os.listdir(d):
            if not fn.endswith('.py'):
                continue
            with open(os.path.join(d, fn), 'rb') as f:
                for tok in tokenize.tokenize(f.readline):
                    if tok.type == tokenize.NUMBER and tok.string.rstrip('.0') in ('2560', '1440', '2560.', '1440.'):
                        line = tok.line.strip()
                        if '/ 1440' in line:                 # a fraction of the frame height
                            continue
                        bad.append(f'{sub}/{fn}:{tok.start[0]}: {line}')
    assert not bad, bad


# ---------------------------------------------------------------------------
# first run: nobody clicks anything
# ---------------------------------------------------------------------------

def test_first_run_builds_the_catalog_by_itself_and_backs_off_after_failures(monkeypatch):
    started = []
    monkeypatch.setattr(app, '_has_catalog', lambda: False)
    monkeypatch.setattr(app, 'start_icon_db_build', lambda reason='manual': started.append(reason) or True)
    monkeypatch.setitem(app._index_build_state, 'running', False)
    monkeypatch.setattr(app, '_first_build', {'at': 0.0, 'tries': 0})
    assert app._maybe_first_build(1000.0) and started == ['first run']
    assert not app._maybe_first_build(1100.0)                      # still inside the 5 min pause
    assert app._maybe_first_build(1000.0 + app.RETRY_BACKOFF + 1)
    monkeypatch.setattr(app, '_has_catalog', lambda: True)
    assert not app._maybe_first_build(10 ** 6) and len(started) == 2


def test_setup_status_walks_through_the_first_run(monkeypatch, tmp_path):
    monkeypatch.setattr(app, '_has_catalog', lambda: False)
    monkeypatch.setattr(app, 'tesseract_available', lambda: True)
    monkeypatch.setattr(app, '_eft_install_dir', lambda: 'C:\\Games\\EFT')
    monkeypatch.setattr(app, '_game_language', lambda s=None: gamelang.describe('en'))
    monkeypatch.setattr(app, 'PRICES_PATH', str(tmp_path / 'none.json'))
    st = dict(app._index_build_state)
    try:
        app._index_build_state.update(running=False, phase=None, error=None, done=0, total=0)
        assert app.setup_status()['busy'] == 'Downloading item prices'
        app._index_build_state.update(running=True, phase='images', done=50, total=200)
        s = app.setup_status()
        assert s['busy'] == 'Downloading item pictures' and s['pct'] == 25 and s['problem'] is None
        app._index_build_state.update(running=True, phase='catalog')
        assert app.setup_status()['busy'] == 'Building the item database'
        app._index_build_state.update(running=False, phase=None, error='HTTPSConnectionPool(host=...): Max retries exceeded')
        s = app.setup_status()
        assert s['problem']['button'] == {'label': 'Try again', 'action': 'build'}
        assert 'internet' in s['problem']['text'] and 'Pool' not in s['problem']['text']
        monkeypatch.setattr(app, '_has_catalog', lambda: True)
        assert app.setup_status() == {'busy': None, 'pct': None, 'problem': None, 'notices': []}     # done: banner disappears
    finally:
        app._index_build_state.update(st)


def test_setup_problems_are_one_sentence_with_at_most_one_button(monkeypatch):
    monkeypatch.setattr(app, '_has_catalog', lambda: True)
    monkeypatch.setattr(app, 'tesseract_available', lambda: True)
    monkeypatch.setattr(app, '_eft_install_dir', lambda: None)
    monkeypatch.setattr(eftlogs, 'find_install_dir', lambda override=None, require_logs=True: None)
    p = app.setup_status()['problem']
    assert p == {'text': 'Tarkov not found.', 'button': {'label': 'Choose folder', 'action': 'choose-folder'}}
    monkeypatch.setattr(app, 'tesseract_available', lambda: False)
    p = app.setup_status()['problem']
    assert p['button'] is None and 'Releases' in p['text'] and 'winget' not in p['text']


def test_setup_endpoints(monkeypatch):
    c = app.app.test_client()
    assert set(c.get('/api/setup').get_json()) == {'busy', 'pct', 'problem', 'notices'}
    r = c.post('/api/setup/choose-folder').get_json()                  # no window under test: a sentence, not a traceback
    assert r['ok'] is False and 'window' in r['error']
    started = []
    monkeypatch.setattr(app, 'start_icon_db_build', lambda reason='manual': started.append(reason) or True)
    assert c.post('/api/setup/retry').get_json() == {'ok': True} and started == ['manual']


def test_unreadable_language_notice_shows_on_the_banner(monkeypatch):
    monkeypatch.setattr(app, '_has_catalog', lambda: True)
    monkeypatch.setattr(app, 'tesseract_available', lambda: True)
    monkeypatch.setattr(app, '_eft_install_dir', lambda: 'x')
    monkeypatch.setattr(app, '_game_language', lambda s=None: gamelang.describe('ru'))
    assert 'Russian' in app.setup_status()['notices'][0]


def test_live_banner_has_no_manual_steps():
    html = open(os.path.join(app.BUNDLE, 'templates', 'live.html'), encoding='utf-8').read()
    assert 'winget' not in html and 'Copy install command' not in html and 'Build now' not in html
    assert '/api/setup' in html


# ---------------------------------------------------------------------------
# auto-start follows the exe
# ---------------------------------------------------------------------------

def test_moving_the_exe_replaces_the_old_watcher_so_the_old_file_is_unlocked(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, 'is_registered', lambda: True)
    monkeypatch.setattr(lifecycle, 'register', lambda: True)           # the Run value pointed somewhere else
    running = [True]
    monkeypatch.setattr(lifecycle, 'watcher_running', lambda win=None: running[0])
    monkeypatch.setattr(lifecycle, 'stop_watcher', lambda win=None, wait_s=2.0: calls.append('stop') or running.__setitem__(0, False))
    monkeypatch.setattr(lifecycle, 'spawn_watcher', lambda: calls.append('spawn'))
    assert lifecycle.ensure_autostart({'start_with_windows': True}) is False
    assert calls == ['stop', 'spawn']


def test_a_catalog_built_before_the_pictures_arrived_does_not_count(monkeypatch):
    monkeypatch.setattr(app, 'catalog_summary', lambda: {'items': 0})
    assert app._has_catalog() is False
    monkeypatch.setattr(app, 'catalog_summary', lambda: {'items': 5476})
    assert app._has_catalog() is True
    monkeypatch.setattr(app, 'catalog_summary', lambda: None)
    assert app._has_catalog() is False


def test_warm_up_waits_for_the_pictures_on_a_clean_install(tmp_path, monkeypatch):
    prices = tmp_path / 'prices_cache.json'
    prices.write_text('{}')
    empty = tmp_path / 'tmpl_src'
    empty.mkdir()
    monkeypatch.setattr(app, 'PRICES_PATH', str(prices))
    monkeypatch.setattr(app, 'TMPL_SRC_DIR', str(empty))
    monkeypatch.setattr(app, '_v2_engine', [None, None])
    built = []
    import identify.pipeline as pl
    monkeypatch.setattr(pl, 'Engine', lambda es: built.append(es) or object())
    app._warm_v2_engine()
    assert built == []                      # no engine (and no pointless empty catalog) before the first build


def test_failed_translation_download_is_mentioned_once(tmp_path, monkeypatch):
    prices = tmp_path / 'prices_cache.json'
    prices.write_text(json.dumps({'items': [{'id': 'x'}], 'label_lang_error': 'items_de: HTTP 404'}))
    monkeypatch.setattr(app, 'PRICES_PATH', str(prices))
    monkeypatch.setattr(app, '_has_catalog', lambda: True)
    monkeypatch.setattr(app, 'tesseract_available', lambda: True)
    monkeypatch.setattr(app, '_eft_install_dir', lambda: 'x')
    monkeypatch.setattr(app, '_game_language', lambda s=None: gamelang.describe('ge'))
    notes = app.setup_status()['notices']
    assert notes == ["Couldn't download the item names in German, so English names are used."]


def test_lifecycle_data_dir_matches_the_app_unless_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv('TSH_DATA_DIR', str(tmp_path / 'forced'))
    lifecycle._data_dir_cache.clear()
    assert lifecycle.data_dir() == str(tmp_path / 'forced')
    lifecycle._data_dir_cache.clear()
