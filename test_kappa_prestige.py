"""
test_kappa_prestige.py — offline, no-network tests for v0.3.0's:
  * merge_kappa_into_keep_list deletion semantics (D)
  * default_keep_list() fresh-install hygiene (D)
  * compute_tasks_view kappa_only scoping (B)
  * the Prestige wiki-table parser, _parse_prestige_html (C)

Run with: python -m pytest test_kappa_prestige.py -q
"""

import time

import pytest

import app as tsh_app


# ---------------------------------------------------------------------------
# merge_kappa_into_keep_list — deletion semantics
# ---------------------------------------------------------------------------

def _kl(items):
    return {'categories': [{'id': 'kappa', 'label': 'Kappa (Collector)', 'items': items}]}


def test_wiki_sourced_unmatched_entry_is_removed():
    keep_list = _kl([
        {'id': 'a', 'name': 'Old Wiki Item', 'aliases': [], 'acquired': False, 'source': 'wiki'},
    ])
    summary = tsh_app.merge_kappa_into_keep_list(keep_list, names=[])
    cat = keep_list['categories'][0]
    assert cat['items'] == []
    assert summary['removed'] == ['Old Wiki Item']


def test_custom_sourced_unmatched_entry_survives():
    keep_list = _kl([
        {'id': 'b', 'name': 'My Custom Item', 'aliases': [], 'acquired': False, 'source': 'custom'},
    ])
    summary = tsh_app.merge_kappa_into_keep_list(keep_list, names=[])
    cat = keep_list['categories'][0]
    assert [it['id'] for it in cat['items']] == ['b']
    assert summary['removed'] == []


def test_rename_rescue_keeps_state_and_gains_alias():
    keep_list = _kl([
        {'id': 'c', 'name': 'Press pass (NoiceGuy)', 'aliases': [], 'acquired': True, 'source': 'wiki'},
    ])
    summary = tsh_app.merge_kappa_into_keep_list(
        keep_list, names=['Press pass (issued for NoiceGuy)'])
    cat = keep_list['categories'][0]
    assert len(cat['items']) == 1
    entry = cat['items'][0]
    assert entry['id'] == 'c'                       # same entry, not recreated
    assert entry['acquired'] is True                # state survives
    assert entry['name'] == 'Press pass (issued for NoiceGuy)'
    assert 'Press pass (NoiceGuy)' in entry['aliases']
    assert summary['removed'] == []


def test_surviving_entries_have_no_stale_key():
    keep_list = _kl([
        {'id': 'd', 'name': 'Still Here', 'aliases': [], 'acquired': False,
         'source': 'wiki', 'stale': True},
    ])
    tsh_app.merge_kappa_into_keep_list(keep_list, names=['Still Here'])
    cat = keep_list['categories'][0]
    entry = cat['items'][0]
    assert 'stale' not in entry


def test_summary_has_removed_not_stale():
    keep_list = _kl([])
    summary = tsh_app.merge_kappa_into_keep_list(keep_list, names=['Brand New Item'])
    assert 'removed' in summary
    assert 'stale' not in summary
    assert summary['added'] == ['Brand New Item']


# ---------------------------------------------------------------------------
# default_keep_list() — fresh-install hygiene
# ---------------------------------------------------------------------------

def test_default_keep_list_ships_clean():
    kl = tsh_app.default_keep_list()

    acquired_true = [it for cat in kl['categories'] for it in cat['items']
                     if it.get('acquired') is True]
    assert acquired_true == []

    tasks_cat = next(c for c in kl['categories'] if c['id'] == 'tasks')
    assert tasks_cat['items'] == []

    kappa_cat = next(c for c in kl['categories'] if c['id'] == 'kappa')
    assert len(kappa_cat['items']) > 0
    assert kappa_cat['label'] == tsh_app.KAPPA_LABEL


# ---------------------------------------------------------------------------
# compute_tasks_view — kappa_only scoping
# ---------------------------------------------------------------------------

def _fake_tasks_cache():
    return {
        'timestamp': time.time(),
        'tasks': [
            {
                'id': 't-kappa', 'name': 'Kappa Task', 'minPlayerLevel': 1,
                'kappaRequired': True, 'trader': {'name': 'Prapor'},
                'objectives': [{
                    'id': 'o1', 'type': 'giveItem', 'count': 2, 'foundInRaid': False,
                    'item': {'id': 'shared_item', 'name': 'Widget', 'shortName': 'Wdg'},
                }],
            },
            {
                'id': 't-side', 'name': 'Side Task', 'minPlayerLevel': 1,
                'kappaRequired': False, 'trader': {'name': 'Prapor'},
                'objectives': [{
                    'id': 'o2', 'type': 'giveItem', 'count': 3, 'foundInRaid': False,
                    'item': {'id': 'shared_item', 'name': 'Widget', 'shortName': 'Wdg'},
                }],
            },
        ],
        'hideoutStations': [],
    }


def test_compute_tasks_view_kappa_only_true_excludes_non_kappa_from_aggregate():
    cache = _fake_tasks_cache()
    progress = tsh_app.default_progress()
    view = tsh_app.compute_tasks_view(cache, progress, kappa_only=True)

    assert view['kappa_only'] is True
    assert len(view['tasks']) == 2   # both hand-in tasks still listed

    agg = {r['item_id']: r for r in view['aggregate']}
    assert agg['shared_item']['total_needed'] == 2   # only the kappa task's count


def test_compute_tasks_view_kappa_only_false_includes_both():
    cache = _fake_tasks_cache()
    progress = tsh_app.default_progress()
    view = tsh_app.compute_tasks_view(cache, progress, kappa_only=False)

    assert view['kappa_only'] is False
    assert len(view['tasks']) == 2

    agg = {r['item_id']: r for r in view['aggregate']}
    assert agg['shared_item']['total_needed'] == 5   # both tasks' counts (2 + 3)


# ---------------------------------------------------------------------------
# Prestige wiki-table parser
# ---------------------------------------------------------------------------

_PRESTIGE_HEADER = (
    '<tr><th>Prestige level</th><th>PMC level</th><th>Quests completed</th>'
    '<th>Objectives</th><th>Skills</th><th>Hideout upgrades</th><th>Items</th></tr>'
)


def _prestige_row(level, pmc, quests, objectives, skills, hideout, items):
    return (f'<tr><td>{level}</td><td>{pmc}</td><td>{quests}</td><td>{objectives}</td>'
           f'<td>{skills}</td><td>{hideout}</td><td>{items}</td></tr>')


def test_parse_prestige_html_maps_columns_by_header():
    rows = [
        _prestige_row(1, 15, 10, 20, 'Level 5 in all skills', 'Level 2 all stations', '500,000 roubles'),
        _prestige_row(2, 20, 20, 40, 'Level 10 in all skills', 'Level 3 all stations', '1,000,000 roubles'),
        _prestige_row(3, 25, 30, 60, 'Level 15 in all skills', 'Level 4 all stations', '1,500,000 roubles'),
        _prestige_row(4, 30, 40, 80, 'Level 20 in all skills', 'Level 5 all stations', '2,000,000 roubles'),
        _prestige_row(5, 35, 50, 100, 'Level 25 in all skills', 'Level 6 all stations', '2,500,000 roubles'),
        _prestige_row(6, 40, 60, 120, 'Level 30 in all skills', 'Level 7 all stations', '3,000,000 roubles'),
    ]
    html = f'<table>{_PRESTIGE_HEADER}{"".join(rows)}</table>'

    levels = tsh_app._parse_prestige_html(html)
    assert len(levels) == 6

    lvl1 = next(l for l in levels if l['level'] == 1)
    assert lvl1['pmc_level'] == '15'
    assert lvl1['quests'] == '10'
    assert lvl1['objectives'] == '20'
    assert lvl1['skills'] == 'Level 5 in all skills'
    assert lvl1['hideout'] == 'Level 2 all stations'
    assert lvl1['items'] == '500,000 roubles'

    lvl6 = next(l for l in levels if l['level'] == 6)
    assert lvl6['pmc_level'] == '40'
    assert lvl6['items'] == '3,000,000 roubles'


def test_parse_prestige_html_raises_on_too_few_rows():
    rows = [
        _prestige_row(1, 15, 10, 20, 'Level 5', 'Level 2', '500,000 roubles'),
        _prestige_row(2, 20, 20, 40, 'Level 10', 'Level 3', '1,000,000 roubles'),
        _prestige_row(3, 25, 30, 60, 'Level 15', 'Level 4', '1,500,000 roubles'),
    ]
    html = f'<table>{_PRESTIGE_HEADER}{"".join(rows)}</table>'

    with pytest.raises(ValueError):
        tsh_app._parse_prestige_html(html)
