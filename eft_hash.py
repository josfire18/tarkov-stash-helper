"""
eft_hash.py — validation-gated "exact icon identity" pipeline.

Escape From Tarkov's local icon cache (see icon_cache.py) writes an
`index.json` mapping `{bsgItemHash: fileNumber}` for every icon it has ever
rendered.  If we could reproduce that hash from a tarkov.dev item id, we could
map cache icons to items with zero ambiguity — no NCC, no OCR, no guessing.

RatStash (and the tool built on it, RatScanner) published exactly such an
algorithm a few years ago ("ratstash-2022" below).  As of 2026-07 it no longer
matches the live game: probing it against a real index.json snapshot scores
0/15,420 hits (see scripts/hash_probe.py), and RatScanner itself has since
retired the icon-hash identification feature. BSG evidently changed something
upstream (item-hash salt, string layout, or the hash function itself) and no
public replacement has surfaced yet.

Rather than deleting this work or shipping it unconditionally (which would
silently corrupt identification the moment it's wrong), the whole pipeline is
VALIDATION-GATED:
  - build_exact_map() computes exact_map = {icon filename: item_id} for
    whichever provider is registered in PROVIDERS.
  - validate() cross-checks exact_map against the existing NCC-verified
    associations in data/cache_map.json and only flips `enabled = True` if
    the two independently-derived sources agree on damn near everything.
  - Callers (icon_cache.associate_cache, app.py) only ever consult the exact
    map when the persisted gate (data/eft_hash_status.json) says enabled.

Today the gate is closed (ratstash-2022 has 0 coverage against the live game),
so every consumer of this module behaves byte-identically to before it
existed.  If a future provider validates, the gate opens itself the next time
`compute_and_persist_status` runs — no code changes needed elsewhere.
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# C# string hash (the .NET Framework / "legacy" string.GetHashCode())
# ---------------------------------------------------------------------------


def _int32(v):
    """Wrap an arbitrary Python int into signed 32-bit two's-complement range."""
    v &= 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def csharp_string_hash(s):
    """
    Reimplementation of the classic (pre-.NET-Core, non-randomized)
    `System.String.GetHashCode()`: a dual-DJB2 hash with seeds h1 = h2 = 5381,
    even-index characters folded into h1 and odd-index characters folded into
    h2 (`h = ((h << 5) + h) ^ ord(c)`), combined as `h1 + h2 * 1566083941`.
    All arithmetic wraps at int32.

    This is the specific variant BSG's Unity/.NET runtime is assumed to use
    for item-hashing (per RatStash); it predates the salted string hashing
    .NET Core introduced by default, which is why it's stable across runs and
    worth trying to reproduce at all.
    """
    h1 = 5381
    h2 = 5381
    for i, ch in enumerate(s):
        c = ord(ch)
        if i % 2 == 0:
            h1 = _int32(((h1 << 5) + h1) ^ c)
        else:
            h2 = _int32(((h2 << 5) + h2) ^ c)
    return _int32(h1 + h2 * 1566083941)


# ---------------------------------------------------------------------------
# RatStash item hash
#
# Ported from the-hideout/tarkov-image-generator `hash-calculator.js` (MIT
# licensed): https://github.com/the-hideout/tarkov-image-generator
# That project in turn documents it as BSG's item-icon cache hash as reverse
# engineered by the RatStash / RatScanner project circa 2022 — hence the
# provider name "ratstash-2022" below.
# ---------------------------------------------------------------------------


def ratstash_item_hash(item_id, salt=0):
    """Base RatStash item hash: `17 ^ (stringHash(item_id) ^ salt)`, int32."""
    return _int32(17 ^ (csharp_string_hash(item_id) ^ salt))


# Salt table for hashing item *variant* renders (accessory/attachment states).
# build_exact_map() only ever uses the base hash (salt=0) — variant salts are
# documented here for completeness / future use, since disambiguating base
# item identity (not attachment state) is all NCC/OCR fusion currently needs.
SALT_NVG_THERMAL_HINGED = 23          # NVG / thermal / hinged-component render
SALT_FOLDABLE_STOCK     = 23 << 1     # stock rendered in its folded state
SALT_MAGAZINE_EMPTY     = 24 << 2     # magazine rendered empty


def salt_magazine_loaded(visible_ammo_count):
    """Salt for a magazine rendered with `visible_ammo_count` visible rounds."""
    return (23 + visible_ammo_count) << 2


SALT_AMMO = 27 * 56   # standalone ammo round render


# Registry so a future validated algorithm can be dropped in alongside (or in
# place of) ratstash-2022 without touching call sites.  Each entry maps to a
# fn(item_id) -> base item hash (salt=0).
PROVIDERS = {
    'ratstash-2022': lambda item_id: ratstash_item_hash(item_id, salt=0),
}


# ---------------------------------------------------------------------------
# Exact map construction + validation gate
# ---------------------------------------------------------------------------


def build_exact_map(items, index_json, provider='ratstash-2022'):
    """
    Compute {icon filename: item_id} by hashing every item's id with
    `provider` and looking the hash up in the icon cache's index.json
    (`{str(hash): fileNumber}`).

    When two items collide on the same hash, BOTH are dropped (neither can be
    trusted) and the collision is logged — silently keeping one at random
    would be worse than not mapping either.
    """
    hash_fn = PROVIDERS[provider]

    by_hash = {}
    for it in items:
        h = str(hash_fn(it['id']))
        by_hash.setdefault(h, []).append(it['id'])

    exact_map = {}
    n_collisions = 0
    for h, ids in by_hash.items():
        file_num = index_json.get(h)
        if file_num is None:
            continue
        if len(ids) > 1:
            n_collisions += 1
            print(f"[eft_hash] collision on hash {h}: items {ids} — dropped")
            continue
        filename = f'{file_num}.png'
        if filename in exact_map:
            # Two different hashes resolved (via index.json) to the same
            # file number — shouldn't happen, but never trust either.
            n_collisions += 1
            print(f"[eft_hash] collision on file {filename} — dropped")
            del exact_map[filename]
            continue
        exact_map[filename] = ids[0]

    if n_collisions:
        print(f"[eft_hash] {n_collisions} hash collision(s) dropped "
              f"({len(exact_map)} unambiguous mappings kept)")
    return exact_map


def validate(exact_map, cache_map, index_json=None, strong_cutoff=0.86):
    """
    Decide whether `exact_map` (provider hash -> item id, via build_exact_map)
    can be trusted, by cross-checking it against the existing NCC-verified
    visual association in data/cache_map.json (see icon_cache.py — entries
    have `item_id`, `score`, and `unknown`/`preset` flags).

    Only files with BOTH an exact-map id and a *strong* visual id (NCC score
    >= strong_cutoff, i.e. status is neither 'unknown' nor merely 'preset')
    are counted, since those are the only pairs where the visual side is
    itself trustworthy enough to arbitrate.

    Returns {agreement, overlap, coverage, enabled}:
      agreement = fraction of overlap files where exact_map and cache_map agree
      overlap   = count of such files
      coverage  = len(exact_map) / len(index_json)  (0.0 if index_json omitted)
      enabled   = agreement >= 0.95 and overlap >= 50 and coverage >= 0.05
    """
    overlap = 0
    agree = 0
    for fn, item_id in exact_map.items():
        entry = cache_map.get(fn)
        if not entry:
            continue
        if entry.get('unknown') or entry.get('preset'):
            continue
        if entry.get('score', 0) < strong_cutoff:
            continue
        overlap += 1
        if entry.get('item_id') == item_id:
            agree += 1

    agreement = (agree / overlap) if overlap else 0.0
    coverage = (len(exact_map) / len(index_json)) if index_json else 0.0
    enabled = agreement >= 0.95 and overlap >= 50 and coverage >= 0.05
    return {'agreement': agreement, 'overlap': overlap, 'coverage': coverage,
            'enabled': enabled}


# ---------------------------------------------------------------------------
# Persistence — data/eft_hash_status.json
# ---------------------------------------------------------------------------


def _status_path(data_dir):
    return os.path.join(data_dir, 'eft_hash_status.json')


def get_status(data_dir):
    """Return the persisted gate status dict, or None if never computed."""
    path = _status_path(data_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def compute_and_persist_status(items, index_json, cache_map, data_dir,
                                provider='ratstash-2022'):
    """
    Build the exact map for `provider`, validate it against `cache_map`, and
    persist the gate status to data/eft_hash_status.json.

    Returns (status_dict, exact_map) — callers that only care about the gate
    can ignore exact_map; icon_cache.associate_cache uses both.
    """
    exact_map = build_exact_map(items, index_json, provider=provider)
    result = validate(exact_map, cache_map, index_json)
    status = {
        'provider':   provider,
        'enabled':    result['enabled'],
        'agreement':  round(result['agreement'], 4),
        'overlap':    result['overlap'],
        'coverage':   round(result['coverage'], 4),
        'checked_at': time.time(),
    }
    with open(_status_path(data_dir), 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    return status, exact_map
