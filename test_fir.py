"""
test_fir.py — offline, no-network, no-Tesseract tests for the Found-in-Raid
(FiR) keep-list rules:
  * app.py's get_protected_ids fir_only return shape
(the checkmark detector itself is identify/fir.py, tested in tests/test_identify_pipeline.py)

Run with: python -m pytest test_fir.py -q
"""

import app as tsh_app


# ---------------------------------------------------------------------------
# get_protected_ids — fir_only shape
# ---------------------------------------------------------------------------

def test_get_protected_ids_fir_only_shape(monkeypatch):
    # Offline: no cached task data, so the aggregate (task/hideout) pass is
    # skipped entirely and only the keep-list pass runs.
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: None)

    keep_list = {
        'categories': [
            {'id': 'kappa', 'label': 'Kappa (Collector)', 'items': [
                {'id': 'kappa_item', 'name': 'Kappa Test Item',
                 'aliases': [], 'acquired': False},
            ]},
            {'id': 'tasks', 'label': 'Task Items (manual)', 'items': [
                {'id': 'task_item', 'name': 'Task Test Item',
                 'aliases': [], 'acquired': False},
            ]},
        ]
    }
    price_idx = {
        'kappa test item': {'id': 'tid_kappa', 'name': 'Kappa Test Item'},
        'task test item':  {'id': 'tid_task',  'name': 'Task Test Item'},
    }

    protected = tsh_app.get_protected_ids(keep_list, price_idx)

    assert protected['tid_kappa']['reason'] == 'On keep list'
    assert protected['tid_kappa']['fir_only'] is True
    assert protected['tid_task']['reason'] == 'On keep list'
    assert protected['tid_task']['fir_only'] is False
    # one copy each; the Kappa hand-in must be FiR, the manual entry takes any copy
    assert (protected['tid_kappa']['need'], protected['tid_kappa']['fir_need']) == (1, 1)
    assert (protected['tid_task']['need'], protected['tid_task']['fir_need']) == (1, 0)
    assert 'Kappa (Collector)' in protected['tid_kappa']['why'][0]


def test_get_protected_ids_skips_acquired_entries(monkeypatch):
    monkeypatch.setattr(tsh_app, 'get_tasks', lambda allow_fetch=True: None)

    keep_list = {
        'categories': [
            {'id': 'kappa', 'label': 'Kappa (Collector)', 'items': [
                {'id': 'kappa_done', 'name': 'Already Acquired',
                 'aliases': [], 'acquired': True},
            ]},
        ]
    }
    price_idx = {
        'already acquired': {'id': 'tid_done', 'name': 'Already Acquired'},
    }

    protected = tsh_app.get_protected_ids(keep_list, price_idx)
    assert 'tid_done' not in protected
