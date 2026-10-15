"""Profiles: PvP, PvP Season and PvE are separate characters.  Progress, the keep list and the
character settings live per profile (data/profiles/<mode>/), the app's preferences stay global.

Every path is redirected to tmp_path; the real data/ is never touched."""
import json
import os

import pytest

import app

CHARACTER = {'prestige': 5, 'faction': 'USEC', 'trader_levels': {'Ref': 3}, 'intel_center_level': 2,
             'hideout_management_level': 12, 'flea_offer_slots': 7}


@pytest.fixture
def env(monkeypatch, tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    monkeypatch.setattr(app, 'SETTINGS_PATH', str(d / 'settings.json'))
    monkeypatch.setattr(app, 'PROGRESS_PATH', str(d / 'progress.json'))
    monkeypatch.setattr(app, 'KEEPLIST_PATH', str(d / 'keep_list.json'))
    monkeypatch.setattr(app, 'KAPPA_WIKI_PATH', str(d / 'kappa_wiki.json'))
    monkeypatch.setattr(app, 'PROFILES_DIR', str(d / 'profiles'))
    monkeypatch.setattr(app, '_mig_clean', {'sig': None})
    monkeypatch.setattr(app.lifecycle, 'apply_settings', lambda s: None)
    monkeypatch.setattr(app, 'ensure_source_async', lambda source: False)
    det = {'mode': 'season', 'source': 'history', 'since': None}
    monkeypatch.setattr(app, 'detect_mode', lambda: det)
    cache = {'timestamp': 0, 'tasks': [
        {'id': 't1', 'name': 'Gunsmith', 'kappaRequired': True, 'trader': {'name': 'Mechanic'},
         'objectives': [{'type': 'giveItem', 'count': 2, 'foundInRaid': True,
                         'items': [{'id': 'gpu', 'name': 'Graphics card', 'shortName': 'GPU'}]}]},
    ], 'hideoutStations': [
        {'id': 'lav', 'name': 'Lavatory', 'levels': [
            {'id': 'lav-1', 'level': 1, 'itemRequirements': []},
            {'id': 'lav-2', 'level': 2, 'itemRequirements': []}]}]}
    monkeypatch.setattr(app, 'get_tasks', lambda allow_fetch=True: cache)
    monkeypatch.setattr(app, 'log_task_progress', lambda cache=None: None)

    class Env:
        dir = d
        detected = det

        def read(self, path):
            return json.loads(open(path, encoding='utf-8').read())

        def pfile(self, mode, name):
            return str(d / 'profiles' / mode / (name + '.json'))

        def pick(self, mode):
            """The header chip / Settings: choose the profile."""
            app._mode_cache['val'] = None
            return self.client.post('/api/mode', json={'mode': mode})

    e = Env()
    with app.app.test_client() as c:
        e.client = c
        yield e


def write(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def legacy_install(env):
    write(app.SETTINGS_PATH, {**app.default_settings(), **CHARACTER, 'flea_min_gain': 9000, 'game_mode': 'auto'})
    write(app.PROGRESS_PATH, {'completed_tasks': ['t1'], 'completed_hideout': ['lav-1'], 'have': {'gpu': 1}})
    write(app.KEEPLIST_PATH, {'categories': [{'id': 'kappa', 'label': 'K', 'items': [
        {'id': 'tea', 'name': 'English Tea', 'acquired': True}]}]})


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_moves_everything_to_the_detected_profile_and_keeps_backups(env):
    legacy_install(env)
    before = {p: open(p, 'rb').read() for p in (app.SETTINGS_PATH, app.PROGRESS_PATH, app.KEEPLIST_PATH)}
    assert sorted(app.migrate_profiles()) == ['keep_list', 'progress', 'settings']
    # the old originals are kept byte for byte as *.migrated.bak
    for p, raw in before.items():
        assert open(p + '.migrated.bak', 'rb').read() == raw
    assert not os.path.exists(app.PROGRESS_PATH) and not os.path.exists(app.KEEPLIST_PATH)
    assert env.read(env.pfile('season', 'progress'))['completed_tasks'] == ['t1']
    assert env.read(env.pfile('season', 'keep_list'))['categories'][0]['items'][0]['acquired'] is True
    assert env.read(env.pfile('season', 'settings')) == CHARACTER        # only the character keys
    g = env.read(app.SETTINGS_PATH)
    assert not set(g) & set(app.PROFILE_SETTING_KEYS) and g['flea_min_gain'] == 9000
    for other in ('pvp', 'pve'):                                          # the other profiles start fresh
        assert not os.path.exists(os.path.join(app.PROFILES_DIR, other))
    assert 'profile season' in open(os.path.join(app.PROFILES_DIR, 'migration.log'), encoding='utf-8').read()


def test_migration_is_idempotent_and_never_loses_data(env):
    legacy_install(env)
    app.migrate_profiles()
    snap = {n: open(env.pfile('season', n), 'rb').read() for n in ('progress', 'keep_list', 'settings')}
    g = open(app.SETTINGS_PATH, 'rb').read()
    assert app.migrate_profiles() == [] and app.migrate_profiles() == []
    assert {n: open(env.pfile('season', n), 'rb').read() for n in snap} == snap
    assert open(app.SETTINGS_PATH, 'rb').read() == g
    baks = sorted(f for f in os.listdir(env.dir) if '.migrated' in f)
    assert baks == ['keep_list.json.migrated.bak', 'progress.json.migrated.bak', 'settings.json.migrated.bak']


def test_a_resurrected_old_file_never_overwrites_the_profile(env):
    legacy_install(env)
    app.migrate_profiles()
    app.save_progress({'completed_tasks': ['newer'], 'completed_hideout': [], 'have': {}})
    write(app.PROGRESS_PATH, {'completed_tasks': ['stale'], 'completed_hideout': [], 'have': {}})
    assert app.migrate_profiles() == ['progress']
    assert app.load_progress()['completed_tasks'] == ['newer']
    assert len([f for f in os.listdir(env.dir) if f.startswith('progress.json.migrated')]) == 2   # both kept


def test_migration_follows_a_manual_pick_and_falls_back_to_pvp(env):
    legacy_install(env)
    write(app.SETTINGS_PATH, {**app.default_settings(), **CHARACTER, 'game_mode': 'pve'})
    app.migrate_profiles()
    assert os.path.exists(env.pfile('pve', 'progress')) and not os.path.exists(env.pfile('season', 'progress'))


def test_migration_with_nothing_known_lands_on_pvp(env, monkeypatch):
    legacy_install(env)
    monkeypatch.setattr(app, 'detect_mode', lambda: {'mode': 'pvp', 'source': 'default', 'since': None})
    app.migrate_profiles()
    assert env.read(env.pfile('pvp', 'progress'))['completed_tasks'] == ['t1']


def test_first_read_migrates_without_an_explicit_call(env):
    legacy_install(env)
    assert app.load_progress()['completed_tasks'] == ['t1']               # lazily, on first use
    assert app._load_settings()['prestige'] == 5 and app._load_settings()['flea_min_gain'] == 9000


def test_fresh_install_has_no_legacy_files_to_migrate(env):
    assert app.migrate_profiles() == []
    assert app.load_progress() == app.default_progress()
    assert not [f for f in os.listdir(env.dir) if f.endswith('.bak')]


# ---------------------------------------------------------------------------
# Isolation + switching
# ---------------------------------------------------------------------------

def test_a_profile_never_sees_another_profiles_progress(env):
    c = env.client
    env.pick('season')
    c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True})
    c.post('/api/hideout/level', json={'station_id': 'lav', 'level': 2})
    c.post('/api/tasks/have', json={'item_id': 'gpu', 'delta': 2})
    tea = app.load_keep_list()['categories'][0]['items'][0]
    c.post('/api/keep-list/toggle', json={'id': tea['id']})
    c.post('/api/settings', json={'prestige': 4, 'faction': 'BEAR'})

    r = env.pick('pve')
    assert r.get_json()['profile'] == {'mode': 'pve', 'label': 'PvE'}
    assert app.load_progress() == app.default_progress()                 # a blank character
    assert all(not i['acquired'] for cat in app.load_keep_list()['categories'] for i in cat['items'])
    assert app._load_settings()['prestige'] == app.default_settings()['prestige']
    c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True})
    c.post('/api/prestige/advance')                                      # PvE prestige 3 -> 4, resets PvE only
    c.post('/api/hideout/level', json={'station_id': 'lav', 'level': 1})

    env.pick('season')                                                   # back: exactly what it was
    p = app.load_progress()
    assert p['completed_tasks'] == ['t1'] and p['completed_hideout'] == ['lav-1', 'lav-2'] and p['have'] == {'gpu': 2}
    assert next(i for cat in app.load_keep_list()['categories'] for i in cat['items'] if i['id'] == tea['id'])['acquired']
    s = app._load_settings()
    assert (s['prestige'], s['faction']) == (4, 'BEAR')

    env.pick('pvp')
    assert app.load_progress() == app.default_progress()                 # PvP was never touched by PvE or Season
    assert app.load_keep_list()['categories'][0]['items'][0]['acquired'] is False


def test_prestige_in_one_profile_leaves_the_others_alone(env):
    c = env.client
    env.pick('season')
    c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True})
    tea = app.load_keep_list()['categories'][0]['items'][0]
    c.post('/api/keep-list/toggle', json={'id': tea['id']})
    env.pick('pvp')
    c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True})
    assert c.post('/api/prestige/advance').get_json() == {'prestige': 4, 'profile': {'mode': 'pvp', 'label': 'PvP'}}
    assert app.load_progress('pvp') == app.default_progress()
    assert app.load_progress('season')['completed_tasks'] == ['t1']
    assert app.load_keep_list('season')['categories'][0]['items'][0]['acquired'] is True
    assert app.load_profile_settings('season')['prestige'] == app.default_settings()['prestige']


def test_auto_mode_follows_the_detected_game(env):
    c = env.client
    env.pick('auto')
    c.post('/api/tasks/have', json={'item_id': 'gpu', 'delta': 1})       # lands in season (detected)
    env.detected.update(mode='pvp')
    app._mode_cache['val'] = None
    assert app.load_progress()['have'] == {}
    assert c.get('/api/tasks').get_json()['profile'] == {'mode': 'pvp', 'label': 'PvP'}
    assert app.load_progress('season')['have'] == {'gpu': 1}


def test_effective_progress_reads_the_active_profile(env):
    env.pick('season')
    app.save_progress({'completed_tasks': ['t1'], 'completed_hideout': [], 'have': {}})
    assert app.effective_progress({'tasks': []})['completed_tasks'] == ['t1']
    env.pick('pve')
    p = app.effective_progress({'tasks': []})
    assert p['completed_tasks'] == [] and p['profile'] == 'pve'


def test_sell_context_uses_the_active_profiles_hideout(env, monkeypatch):
    seen = []
    monkeypatch.setattr(app.sellcalc, 'intel_center_level', lambda st, done: seen.append(list(done or ())) or 0)
    env.pick('season')
    app.save_progress({'completed_tasks': [], 'completed_hideout': ['lav-1'], 'have': {}})
    app.build_sell_context(app._load_settings(), {'rules': None})
    env.pick('pvp')
    app.build_sell_context(app._load_settings(), {'rules': None})
    assert seen == [['lav-1'], []]


def test_protected_plan_follows_the_profile(env, monkeypatch):
    prices = {'items': [{'id': 'gpu', 'name': 'Graphics card', 'shortName': 'GPU'}]}
    env.pick('season')
    app.save_progress({'completed_tasks': ['t1'], 'completed_hideout': [], 'have': {}})
    assert 'gpu' not in app.get_protected_plan({'categories': []}, {})[0]    # task done in Season
    env.pick('pvp')
    assert app.get_protected_plan({'categories': []}, {})[0]['gpu']['need'] == 2   # still open in PvP


# ---------------------------------------------------------------------------
# Settings split
# ---------------------------------------------------------------------------

def test_profile_keys_go_to_the_profile_file_and_global_keys_stay_global(env):
    env.pick('season')
    r = env.client.post('/api/settings', json={**CHARACTER, 'flea_min_gain': 4000, 'auto_scan': False})
    assert r.status_code == 200
    assert env.read(env.pfile('season', 'settings')) == CHARACTER
    g = env.read(app.SETTINGS_PATH)
    assert not set(g) & set(app.PROFILE_SETTING_KEYS)
    assert g['flea_min_gain'] == 4000 and g['auto_scan'] is False and g['game_mode'] == 'season'
    merged = env.client.get('/api/settings').get_json()
    assert merged['prestige'] == 5 and merged['flea_min_gain'] == 4000


def test_every_default_setting_is_classified(env):
    g, p = app.split_settings(app.default_settings())
    assert set(p) == set(app.PROFILE_SETTING_KEYS) == set(app.profile_defaults())
    assert not set(g) & set(p) and set(g) == set(app.global_defaults())
    for k in ('skip_traders', 'flea_min_gain', 'flea_overflow', 'flea_requires_fir', 'ignore_task_items',
              'ignore_hideout_items', 'game_mode', 'auto_scan', 'hotkey', 'region', 'kappa_only_tasks'):
        assert k in g


def test_character_settings_differ_per_profile_and_global_ones_do_not(env):
    c = env.client
    env.pick('season')
    c.post('/api/settings', json={'prestige': 6, 'trader_levels': {'Ref': 4}, 'flea_min_gain': 1234})
    env.pick('pve')
    s = c.get('/api/settings').get_json()
    assert s['prestige'] == app.default_settings()['prestige'] and s['trader_levels'] == {}
    assert s['flea_min_gain'] == 1234                                    # a preference, shared
    c.post('/api/settings', json={'prestige': 1})
    env.pick('season')
    assert c.get('/api/settings').get_json()['prestige'] == 6


def test_changing_the_mode_and_a_character_key_in_one_save_edits_the_profile_on_screen(env):
    env.pick('season')
    r = env.client.post('/api/settings', json={'game_mode': 'pve', 'prestige': 2})
    body = r.get_json()
    assert body['profile'] == {'mode': 'pve', 'label': 'PvE'}
    assert app.load_profile_settings('season')['prestige'] == 2           # the page showed Season when saved
    assert body['settings']['prestige'] == app.default_settings()['prestige']   # ... the response shows PvE


def test_autoscan_toggle_and_hotkey_saves_keep_the_split(env):
    env.pick('season')
    env.client.post('/api/settings', json={'prestige': 2})
    env.client.post('/api/autoscan/toggle', json={'enabled': False})
    assert not set(env.read(app.SETTINGS_PATH)) & set(app.PROFILE_SETTING_KEYS)
    assert env.read(env.pfile('season', 'settings'))['prestige'] == 2


# ---------------------------------------------------------------------------
# Keep list
# ---------------------------------------------------------------------------

def test_a_new_profile_starts_from_defaults_plus_the_cached_collector_list(env):
    write(app.KAPPA_WIKI_PATH, {'timestamp': 1, 'names': ['Brand new wiki item']})
    kl = app.load_keep_list('pve')
    names = [i['name'] for i in kl['categories'][0]['items']]
    assert 'Brand new wiki item' in names and 'Antique axe' not in names   # wiki-sourced list replaces the defaults
    assert os.path.exists(env.pfile('pve', 'keep_list'))
    assert all(not i['acquired'] for i in kl['categories'][0]['items'])


def test_a_new_profile_without_a_wiki_cache_gets_the_built_in_defaults(env):
    kl = app.load_keep_list('pvp')
    assert any(i['name'] == 'Antique axe' for i in kl['categories'][0]['items'])


def test_kappa_sync_updates_every_profile_and_keeps_each_hand_in_state(env, monkeypatch):
    env.pick('season')
    axe = next(i for i in app.load_keep_list()['categories'][0]['items'] if i['name'] == 'Antique axe')
    env.client.post('/api/keep-list/toggle', json={'id': axe['id']})
    app.load_keep_list('pve')
    monkeypatch.setattr(app, 'fetch_kappa_names', lambda force=False: ['Antique axe', 'Golden egg'] * 12
                        + [f'Filler {i}' for i in range(30)])
    monkeypatch.setattr(app, 'get_prices', lambda *a, **k: {'items': []})
    app.kappa_sync()
    s = {i['name']: i for i in app.load_keep_list('season')['categories'][0]['items']}
    p = {i['name']: i for i in app.load_keep_list('pve')['categories'][0]['items']}
    assert s['Antique axe']['acquired'] is True and p['Antique axe']['acquired'] is False
    assert 'Filler 3' in s and 'Filler 3' in p


def test_keep_list_routes_edit_only_the_active_profile(env):
    c = env.client
    env.pick('season')
    item = c.post('/api/keep-list/add', json={'name': 'My pin', 'category': 'tasks'}).get_json()['item']
    env.pick('pvp')
    assert not any(i['id'] == item['id'] for cat in c.get('/api/keep-list').get_json()['categories'] for i in cat['items'])
    env.pick('season')
    kl = c.get('/api/keep-list').get_json()
    assert kl['profile'] == {'mode': 'season', 'label': 'PvP S'}
    assert any(i['id'] == item['id'] for cat in kl['categories'] for i in cat['items'])
    c.post('/api/reset')                                                 # resets the Season list only
    assert not any(i['id'] == item['id'] for cat in app.load_keep_list('season')['categories'] for i in cat['items'])


# ---------------------------------------------------------------------------
# API shapes
# ---------------------------------------------------------------------------

def test_api_responses_name_the_profile(env):
    c = env.client
    env.pick('season')
    prof = {'mode': 'season', 'label': 'PvP S'}
    assert c.get('/api/mode').get_json()['profile'] == prof
    assert c.get('/api/tasks').get_json()['profile'] == prof
    assert c.get('/api/v1/needs').get_json()['profile'] == prof
    full = c.get('/api/settings?full=1').get_json()
    assert full['profile'] == {**prof, 'keys': list(app.PROFILE_SETTING_KEYS)}
    assert c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True}).get_json()['profile'] == prof
    assert c.post('/api/tasks/have', json={'item_id': 'gpu', 'delta': 1}).get_json()['profile'] == prof
    assert c.post('/api/hideout/level', json={'station_id': 'lav', 'level': 1}).get_json()['profile'] == prof
    assert c.post('/api/settings', json={'auto_scan': True}).get_json()['profile'] == prof


def test_existing_response_shapes_are_unchanged(env):
    c = env.client
    assert c.post('/api/tasks/complete', json={'type': 'task', 'id': 't1', 'done': True}).get_json()['ok'] is True
    assert c.post('/api/tasks/have', json={'item_id': 'gpu', 'delta': 1}).get_json()['have'] == 1
    assert c.post('/api/hideout/level', json={'station_id': 'lav', 'level': 2}).get_json()['level'] == 2
    view = c.get('/api/tasks').get_json()
    assert {'aggregate', 'task_progress', 'auto_done'} <= set(view)
    assert 'categories' in c.get('/api/keep-list').get_json()


def test_needs_page_and_settings_page_carry_the_profile_ui(env):
    needs = env.client.get('/needs').data.decode()
    assert "profile · " in needs and 'view.profile' in needs
    s = env.client.get('/settings').data.decode()
    assert "id: 'character'" in s and 'Your character' in s and 'paintProfile' in s
