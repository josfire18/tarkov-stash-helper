"""The Settings page's "pure money" switches: ignore_task_items / ignore_hideout_items drop whole
need sources from the protected plan, so those items are routed to the flea or a trader."""
import json

import app as tsh_app


def _setup(monkeypatch, tmp_path, **settings):
    sp = tmp_path / 'settings.json'
    sp.write_text(json.dumps({**tsh_app.default_settings(), **settings}))
    monkeypatch.setattr(tsh_app, 'SETTINGS_PATH', str(sp))
    monkeypatch.setattr(tsh_app, 'PROGRESS_PATH', str(tmp_path / 'progress.json'))
    cache = {
        'timestamp': 0,
        'tasks': [{'id': 't1', 'name': 'Task One', 'kappaRequired': True, 'trader': {'name': 'Prapor'},
                   'objectives': [{'type': 'giveItem', 'count': 2, 'foundInRaid': True,
                                   'item': {'id': 'task_item', 'name': 'Task Item'}}]}],
        'hideoutStations': [{'id': 's1', 'name': 'Lavatory', 'levels': [
            {'id': 'l1', 'level': 1, 'itemRequirements': [
                {'count': 1, 'item': {'id': 'hideout_item', 'name': 'Hideout Item'}}]}]}],
    }
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: cache)
    keep_list = {'categories': [
        {'id': 'kappa', 'label': 'Kappa', 'items': [
            {'id': 'k', 'name': 'Kappa Item', 'aliases': [], 'acquired': False}]},
        {'id': 'custom', 'label': 'Mine', 'items': [
            {'id': 'c', 'name': 'Custom Item', 'aliases': [], 'acquired': False}]},
    ]}
    price_idx = {'kappa item': {'id': 'kappa_item', 'name': 'Kappa Item'},
                 'custom item': {'id': 'custom_item', 'name': 'Custom Item'}}
    return tsh_app.get_protected_ids(keep_list, price_idx)


def test_default_keeps_every_source(monkeypatch, tmp_path):
    p = _setup(monkeypatch, tmp_path)
    assert {'task_item', 'hideout_item', 'kappa_item', 'custom_item'} <= set(p)


def test_ignore_task_items(monkeypatch, tmp_path):
    p = _setup(monkeypatch, tmp_path, ignore_task_items=True)
    assert 'task_item' not in p and 'kappa_item' not in p
    assert 'hideout_item' in p and 'custom_item' in p


def test_ignore_hideout_items(monkeypatch, tmp_path):
    p = _setup(monkeypatch, tmp_path, ignore_hideout_items=True)
    assert 'hideout_item' not in p
    assert 'task_item' in p and 'kappa_item' in p


def test_pure_money_mode(monkeypatch, tmp_path):
    p = _setup(monkeypatch, tmp_path, ignore_task_items=True, ignore_hideout_items=True)
    assert set(p) == {'custom_item'}      # only the player's own keep list is still honoured
