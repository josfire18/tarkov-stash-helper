"""
test_eft_hash.py — offline, no-network, no-Tesseract tests for:
  * eft_hash.py's C# string hash / RatStash item hash reimplementation
  * eft_hash.py's validation gate (build_exact_map / validate)
  * app.py's _ocr_tokens_contained OCR-override guard (A2)

Run with: python -m pytest test_eft_hash.py -q
"""

import ctypes

import eft_hash


# ---------------------------------------------------------------------------
# csharp_string_hash / ratstash_item_hash
# ---------------------------------------------------------------------------

def _ref_csharp_hash(s):
    """
    Independent reference implementation of the same algorithm, using
    ctypes.c_int32 (rather than eft_hash._int32's bit-masking) for the
    wraparound arithmetic — cross-checks eft_hash's own int32 handling
    rather than merely re-running the same code.
    """
    def i32(v):
        return ctypes.c_int32(v).value

    h1 = 5381
    h2 = 5381
    for i, ch in enumerate(s):
        c = ord(ch)
        if i % 2 == 0:
            h1 = i32(i32((h1 << 5) + h1) ^ c)
        else:
            h2 = i32(i32((h2 << 5) + h2) ^ c)
    return i32(h1 + h2 * 1566083941)


def test_csharp_hash_empty_string_hand_computable():
    # No loop iterations: h1 = h2 = 5381 unchanged.
    # result = int32(5381 + 5381 * 1566083941)
    expected = ctypes.c_int32(5381 + 5381 * 1566083941).value
    assert eft_hash.csharp_string_hash('') == expected


def test_csharp_hash_matches_independent_reference_implementation():
    samples = ['', 'a', 'ab', 'abc', 'Hello', 'test-item-id-123',
               '5c0e3f8386f7745e5f0c8bcf', 'x' * 37]
    for s in samples:
        assert eft_hash.csharp_string_hash(s) == _ref_csharp_hash(s), s


def test_csharp_hash_is_deterministic():
    s = 'some-tarkov-item-id'
    assert eft_hash.csharp_string_hash(s) == eft_hash.csharp_string_hash(s)


def test_csharp_hash_is_int32_range():
    for s in ['', 'a', 'zzzzzzzzzzzzzzzzzzzz', '5c0e3f8386f7745e5f0c8bcf']:
        h = eft_hash.csharp_string_hash(s)
        assert -(2 ** 31) <= h <= (2 ** 31 - 1)


def test_csharp_hash_differing_strings_differ():
    ids = ['5c0e3f8386f7745e5f0c8bcf', '5c0e3f8386f7745e5f0c8bd0',
           'some-item', 'some-items', 'Some-Item']
    hashes = {eft_hash.csharp_string_hash(s) for s in ids}
    assert len(hashes) == len(ids)


def test_ratstash_item_hash_applies_17_xor_and_salt():
    item_id = '5c0e3f8386f7745e5f0c8bcf'
    base = eft_hash.csharp_string_hash(item_id)
    assert eft_hash.ratstash_item_hash(item_id, salt=0) == eft_hash._int32(17 ^ base)
    salt = 23
    assert eft_hash.ratstash_item_hash(item_id, salt=salt) == eft_hash._int32(17 ^ (base ^ salt))


def test_int32_wraparound_matches_ctypes():
    for v in [0, 1, -1, 2 ** 31, 2 ** 31 - 1, -(2 ** 31), 2 ** 32, -(2 ** 32) - 5, 5381 * 1566083941]:
        assert eft_hash._int32(v) == ctypes.c_int32(v).value


def test_providers_registry_has_ratstash_2022():
    assert 'ratstash-2022' in eft_hash.PROVIDERS
    item_id = '5c0e3f8386f7745e5f0c8bcf'
    assert eft_hash.PROVIDERS['ratstash-2022'](item_id) == eft_hash.ratstash_item_hash(item_id, salt=0)


# ---------------------------------------------------------------------------
# Validation gate — synthetic maps
# ---------------------------------------------------------------------------

def test_validate_disabled_when_agreement_too_low():
    # 100 overlapping files, only 90 agree -> 0.90 agreement, below 0.95 gate.
    exact_map = {f'{i}.png': f'item{i}' for i in range(100)}
    cache_map = {}
    for i in range(100):
        agree_id = f'item{i}' if i < 90 else f'other{i}'
        cache_map[f'{i}.png'] = {'item_id': agree_id, 'score': 0.90, 'unknown': False, 'preset': False}
    result = eft_hash.validate(exact_map, cache_map, index_json={str(i): i for i in range(2000)})
    assert result['overlap'] == 100
    assert 0.89 < result['agreement'] < 0.91
    assert result['enabled'] is False


def test_validate_disabled_when_overlap_too_low():
    # Perfect agreement but only 10 overlapping strong pairs -> below overlap>=50 gate.
    exact_map = {f'{i}.png': f'item{i}' for i in range(10)}
    cache_map = {f'{i}.png': {'item_id': f'item{i}', 'score': 0.95, 'unknown': False, 'preset': False}
                 for i in range(10)}
    result = eft_hash.validate(exact_map, cache_map, index_json={str(i): i for i in range(1000)})
    assert result['agreement'] == 1.0
    assert result['overlap'] == 10
    assert result['enabled'] is False


def test_validate_disabled_when_coverage_too_low():
    # Perfect agreement, plenty of overlap, but exact_map only covers a tiny
    # slice of a huge index.json -> coverage < 0.05 gate.
    exact_map = {f'{i}.png': f'item{i}' for i in range(60)}
    cache_map = {f'{i}.png': {'item_id': f'item{i}', 'score': 0.95, 'unknown': False, 'preset': False}
                 for i in range(60)}
    huge_index = {str(i): i for i in range(100000)}   # coverage = 60/100000 << 0.05
    result = eft_hash.validate(exact_map, cache_map, index_json=huge_index)
    assert result['agreement'] == 1.0
    assert result['overlap'] == 60
    assert result['coverage'] < 0.05
    assert result['enabled'] is False


def test_validate_enabled_when_all_thresholds_met():
    exact_map = {f'{i}.png': f'item{i}' for i in range(60)}
    cache_map = {f'{i}.png': {'item_id': f'item{i}', 'score': 0.95, 'unknown': False, 'preset': False}
                 for i in range(60)}
    index_json = {str(i): i for i in range(1000)}   # coverage = 60/1000 = 0.06
    result = eft_hash.validate(exact_map, cache_map, index_json=index_json)
    assert result['agreement'] == 1.0
    assert result['overlap'] == 60
    assert result['coverage'] >= 0.05
    assert result['enabled'] is True


def test_validate_empty_exact_map_disabled_with_zero_coverage():
    result = eft_hash.validate({}, {'x.png': {'item_id': 'a', 'score': 0.99}},
                                index_json={str(i): i for i in range(1000)})
    assert result['overlap'] == 0
    assert result['agreement'] == 0.0
    assert result['coverage'] == 0.0
    assert result['enabled'] is False


def test_validate_ignores_weak_and_unknown_cache_entries():
    # Files below the strong cutoff, or flagged unknown/preset, must not
    # count toward overlap even though they exist in both maps.
    exact_map = {'a.png': 'item-a', 'b.png': 'item-b', 'c.png': 'item-c'}
    cache_map = {
        'a.png': {'item_id': 'item-a', 'score': 0.50, 'unknown': False, 'preset': False},  # too weak
        'b.png': {'item_id': 'item-b', 'score': 0.99, 'unknown': True, 'preset': False},   # unknown
        'c.png': {'item_id': 'item-c', 'score': 0.99, 'unknown': False, 'preset': True},   # preset
    }
    result = eft_hash.validate(exact_map, cache_map, index_json={str(i): i for i in range(100)})
    assert result['overlap'] == 0


def test_build_exact_map_drops_hash_collisions():
    # Two distinct item ids that happen to hash identically (forced by
    # monkeypatching the provider) must both be dropped, not silently
    # resolved to one winner.
    items = [{'id': 'item-a'}, {'id': 'item-b'}, {'id': 'item-c'}]

    def fake_hash(item_id):
        return 111 if item_id in ('item-a', 'item-b') else 222

    orig = eft_hash.PROVIDERS.get('__test__')
    eft_hash.PROVIDERS['__test__'] = fake_hash
    try:
        index_json = {'111': 1, '222': 2}
        exact_map = eft_hash.build_exact_map(items, index_json, provider='__test__')
        assert '1.png' not in exact_map      # collided hash -> dropped
        assert exact_map.get('2.png') == 'item-c'
    finally:
        if orig is None:
            del eft_hash.PROVIDERS['__test__']
        else:
            eft_hash.PROVIDERS['__test__'] = orig


def test_build_exact_map_skips_hashes_absent_from_index():
    items = [{'id': 'item-a'}]

    def fake_hash(item_id):
        return 999

    eft_hash.PROVIDERS['__test2__'] = fake_hash
    try:
        exact_map = eft_hash.build_exact_map(items, {'123': 1}, provider='__test2__')
        assert exact_map == {}
    finally:
        del eft_hash.PROVIDERS['__test2__']


def test_get_status_missing_file_returns_none(tmp_path):
    assert eft_hash.get_status(str(tmp_path)) is None


def test_compute_and_persist_status_roundtrip(tmp_path):
    items = [{'id': f'item{i}'} for i in range(60)]

    def fake_hash(item_id):
        return int(item_id.replace('item', ''))

    eft_hash.PROVIDERS['__test3__'] = fake_hash
    try:
        index_json = {str(i): i for i in range(1000)}
        cache_map = {f'{i}.png': {'item_id': f'item{i}', 'score': 0.95,
                                  'unknown': False, 'preset': False} for i in range(60)}
        status, exact_map = eft_hash.compute_and_persist_status(
            items, index_json, cache_map, str(tmp_path), provider='__test3__')
        assert status['enabled'] is True
        assert status['provider'] == '__test3__'
        reloaded = eft_hash.get_status(str(tmp_path))
        assert reloaded == status
    finally:
        del eft_hash.PROVIDERS['__test3__']


# ---------------------------------------------------------------------------
# A2 OCR override guard — _ocr_tokens_contained (imported from app.py)
#
# Importing app.py only runs its module-level setup (flask/cv2/pytesseract
# imports, constant definitions, route registration) — no scan/Tesseract/
# display calls happen at import time, so this is safe offline.
# ---------------------------------------------------------------------------

import app as tsh_app   # noqa: E402  (after eft_hash import block above)


def test_ocr_tokens_contained_blocks_unrelated_token():
    # Reproduces the original failure: a PMAG label OCR'd as 'gen m3' should
    # NOT be considered contained in the Benelli M3 name — 'gen' has no
    # relationship to any Benelli token.
    assert tsh_app._ocr_tokens_contained(
        'gen m3', 'Benelli M3 Super 90 dual-mode charging handle', 'M3 SUPER90 ch'
    ) is False


def test_ocr_tokens_contained_true_for_exact_tokens():
    assert tsh_app._ocr_tokens_contained(
        'pmag 30 gen m3', 'Magpul PMAG 30 GEN M3 5.56x45 magazine (Black)', 'GEN M3'
    ) is True


def test_ocr_tokens_contained_empty_text_is_false():
    assert tsh_app._ocr_tokens_contained('', 'Anything', 'Anything') is False
    assert tsh_app._ocr_tokens_contained('   ', 'Anything', 'Anything') is False


def test_ocr_tokens_contained_prefix_match():
    # 'benel' is a genuine prefix of 'benelli' -> should pass.
    assert tsh_app._ocr_tokens_contained(
        'benel m3', 'Benelli M3 Super 90 dual-mode charging handle', 'M3 SUPER90 ch'
    ) is True


def test_ocr_tokens_contained_rejects_non_prefix_partial():
    # 'eli' is a substring of 'benelli' but NOT a prefix of any reference
    # token, so it must be rejected even though rapidfuzz might still score
    # this pair highly on raw string similarity.
    assert tsh_app._ocr_tokens_contained(
        'eli m3', 'Benelli M3 Super 90 dual-mode charging handle', 'M3 SUPER90 ch'
    ) is False
