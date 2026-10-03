"""Learned cache-icon names: keyed by pixels, persisted, and never re-taught by accident."""
import json

import cv2
import numpy as np

from identify.learned import LearnedNames, content_key


def _icon(path, value):
    img = np.full((64, 64, 4), value, np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_key_follows_pixels_not_file_name(tmp_path):
    a = _icon(tmp_path / '1.png', 40)
    b = _icon(tmp_path / '2.png', 40)          # same pixels, recycled under a new number
    c = _icon(tmp_path / '3.png', 90)
    assert content_key(a) == content_key(b)
    assert content_key(a) != content_key(c)


def test_learn_persists_and_counts_repeats(tmp_path):
    p = tmp_path / 'learned.json'
    ln = LearnedNames(str(p))
    assert ln.learn("k1", "id-plug", "T-Shaped plug") == 1
    assert ln.learn("k1", "id-plug", "T-Shaped plug") == 2      # second confirmation
    data = json.loads(p.read_text(encoding='utf-8'))
    assert data['k1']['id'] == 'id-plug' and data['k1']['seen'] == 2
    assert LearnedNames(str(p)).get('k1') == 'id-plug'               # reload from disk


def test_relearn_overrides_a_wrong_name(tmp_path):
    ln = LearnedNames(str(tmp_path / 'learned.json'))
    ln.learn('k1', 'id-mask', 'Lower half-mask (Moss)')
    assert ln.learn("k1", "id-egg", "Golden egg") == 1          # a different item restarts the count
    assert ln.get('k1') == 'id-egg'


def test_missing_or_corrupt_store_starts_empty(tmp_path):
    p = tmp_path / 'learned.json'
    p.write_text('{not json', encoding='utf-8')
    assert LearnedNames(str(p)).get('k1') is None
    assert LearnedNames(str(tmp_path / 'absent.json')).data == {}


def test_min_seen_hides_unconfirmed_pairings(tmp_path):
    ln = LearnedNames(str(tmp_path / 'learned.json'))
    ln.learn('k1', 'id-x', 'X')
    assert ln.get('k1', min_seen=2) is None
    ln.learn('k1', 'id-x', 'X')
    assert ln.get('k1', min_seen=2) == 'id-x'
