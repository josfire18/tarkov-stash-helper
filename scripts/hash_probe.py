"""
scripts/hash_probe.py — standalone research harness for the EFT icon-cache
item-hash question.

This codifies a FALSIFIED HYPOTHESIS so it stays reproducible rather than
folklore: the "ratstash-2022" algorithm (dual-DJB2 C# string hash, see
eft_hash.py — ported from the-hideout/tarkov-image-generator's
hash-calculator.js) was published a few years ago as BSG's item-icon-cache
hash. As of 2026-07, probing it against a real, live-game index.json
snapshot scores 0/15,420 hits — it no longer matches. RatScanner, the tool
that popularized this algorithm, has since retired its icon-hash
identification feature for the same reason: BSG evidently changed something
upstream and no public replacement algorithm has surfaced yet.

Recovery routes, roughly in order of effort:
  1. Decompile Assembly-CSharp.dll with ILSpy and locate the current
     item-icon hash function directly. BSG's EULA permits personal
     inspection/research use — do not redistribute decompiled game code or
     ship it in this repo.
  2. Watch RatScanner (https://github.com/RatScanner/RatScanner) or
     the-hideout/tarkov-image-generator for an updated algorithm landing in
     their source, and port it the same way eft_hash.ratstash_item_hash was
     ported — register it in eft_hash.PROVIDERS and re-run this probe.

Run directly:  python scripts/hash_probe.py
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import icon_cache   # reuse its cache-folder discovery + cache_map loader
import eft_hash


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_index_json():
    """
    Locate + load the EFT icon-cache index.json using icon_cache's own
    cache-folder discovery logic, degrading gracefully (printing, not
    raising) when the cache folder isn't present — e.g. EFT isn't installed
    on this machine, or has never rendered any icons yet.
    """
    cache_dir = icon_cache.find_cache_dir()
    if not cache_dir:
        print("[hash_probe] icon-cache folder not found — skipping "
              "(EFT may not be installed, or hasn't rendered any icons yet)")
        return None, None
    idx_path = os.path.join(cache_dir, 'index.json')
    if not os.path.exists(idx_path):
        print(f"[hash_probe] {idx_path} missing — skipping")
        return None, None
    with open(idx_path, encoding='utf-8') as f:
        index_json = json.load(f)
    return cache_dir, index_json


def _load_items():
    """
    Pull the tarkov.dev item catalogue from the app's cached price data
    (data/prices_cache.json) so this script needs no network access. Returns
    [] (with a message) if the app has never fetched prices.
    """
    prices_path = os.path.join(_project_root(), 'data', 'prices_cache.json')
    if os.path.exists(prices_path):
        with open(prices_path, encoding='utf-8') as f:
            cache = json.load(f)
        items = cache.get('items', [])
        if items:
            return items
    print("[hash_probe] no cached data/prices_cache.json items found — "
          "run the app once (it fetches prices on startup) first")
    return []


def probe(hash_fn, index_json, items, cache_map=None):
    """
    Score an arbitrary item-id -> int32 hash function against a real
    index.json and the existing NCC-verified cache_map.

    Returns a dict:
        hits              - count of items whose hash exists in index_json
        total_index       - len(index_json)
        collision_histogram - {n_items_sharing_a_hash: count_of_such_hashes}
        agreement, overlap, coverage, enabled - see eft_hash.validate()
    `coverage` here means (# index.json entries explained by a
    non-colliding item hash) / (total index.json entries).
    """
    if cache_map is None:
        cache_map = icon_cache.load_cache_map()

    by_hash = {}
    for it in items:
        h = str(hash_fn(it['id']))
        by_hash.setdefault(h, []).append(it['id'])

    collision_hist = Counter(len(ids) for ids in by_hash.values())
    hits = sum(1 for h in by_hash if h in index_json)

    # Build the same {filename: item_id} mapping build_exact_map would,
    # without needing this ad-hoc hash_fn registered as a named provider.
    exact_map = {}
    for h, ids in by_hash.items():
        file_num = index_json.get(h)
        if file_num is None or len(ids) != 1:
            continue
        exact_map[f'{file_num}.png'] = ids[0]

    result = eft_hash.validate(exact_map, cache_map, index_json)
    result.update({
        'hits': hits,
        'total_index': len(index_json),
        'collision_histogram': dict(collision_hist),
    })
    return result


if __name__ == '__main__':
    cache_dir, index_json = _load_index_json()
    items = _load_items()
    if not (cache_dir and index_json and items):
        print("[hash_probe] nothing to probe (see messages above) — exiting")
    else:
        cache_map = icon_cache.load_cache_map()
        result = probe(eft_hash.PROVIDERS['ratstash-2022'], index_json, items, cache_map)
        print(f"[hash_probe] provider=ratstash-2022  "
              f"hits={result['hits']}/{result['total_index']}  "
              f"coverage={result['coverage']:.4f}  "
              f"overlap={result['overlap']}  agreement={result['agreement']:.4f}  "
              f"enabled={result['enabled']}")
        print(f"[hash_probe] collision histogram "
              f"(n_items_sharing_a_hash -> count_of_hashes): "
              f"{result['collision_histogram']}")
        # Persist the gate verdict — this probe is the only code path that
        # can flip data/eft_hash_status.json to enabled, so running it after
        # registering a new provider in eft_hash.PROVIDERS is how the exact-ID
        # pipeline lights up (icon_cache/app.py just read the status file).
        status, _ = eft_hash.compute_and_persist_status(
            items, index_json, cache_map,
            os.path.join(_project_root(), 'data'))
        print(f"[hash_probe] persisted gate status: {status}")
