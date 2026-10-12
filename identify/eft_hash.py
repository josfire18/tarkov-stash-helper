"""Exact item ids for EFT icon-cache renders, from the game's own icon hash.

The game keeps every icon it draws in ``Icon Cache/live/<n>.png`` plus an ``index.json``
``{"<signed int32 hash>": n}``.  The hash is computed from the item, so reproducing it turns a
cache file number into an item id with no picture comparison at all.

Plain items (EFT 1.x, IL2CPP; reverse-derived from the cache itself, see
``scripts/hash_probe.py`` and docs/accuracy.md "Icon hash")::

    template_hash(id) = 17 ^ t ^ (p * 3637) ^ (q * 5807)      (mod 2^32, signed)

where the 24-hex template id is read as three big-endian uint32 words ``t | p | q`` (the
MongoID's timestamp and the two halves of its 64-bit counter).  The leading ``17 ^`` and the
render-state salts below are the same as in the 2022 RatStash algorithm; what changed is the id
hash itself (the template id became a ``MongoID`` struct instead of a string).

A render-state salt is XOR-ed in for items drawn in a non-default state:

    ====  ===============================================================
    1512  ammunition (27 * 56)
      96  empty magazine (24 << 2)
      23  NVG / thermal / hinged face shield or headwear
      24  the same kind of item in its other hinge state
      46  folded stock / weapon (23 << 1)
    ====  ===============================================================

Built items (weapons with mods, loaded magazines, armour with plates, filled containers) fold the
hashes of their parts and slot names in as well; those are *not* reproduced, so their renders stay
unnamed (the label anchor still names them).

Everything is validation-gated: :func:`validate` checks the exact ids against independent
evidence (label-certified bindings, unambiguous picture associations) and :func:`gate_status`
persists the verdict in ``data/eft_hash_status.json``; consumers only use exact ids while the
gate is open.
"""
from __future__ import annotations

import json
import os
import re
import time

PROVIDER = 'mongo-3637-5807'          # bump when the formula changes

SALT_PLAIN = 0
SALT_HINGED = 23
SALT_HINGED_ALT = 24
SALT_FOLDED = 46
SALT_MAG_EMPTY = 96
SALT_AMMO = 1512

# salt -> catalog categories (identify.catalog.category_of) an item must have for that salt to
# apply.  None = any.  Restricting the salts keeps chance collisions (~1e-6 per pair) out.
SALTS: dict[int, frozenset | None] = {
    SALT_PLAIN: None,
    SALT_AMMO: frozenset({'ammo'}),
    SALT_MAG_EMPTY: frozenset({'mod'}),
    SALT_HINGED: frozenset({'gear', 'mod', 'weapon'}),
    SALT_HINGED_ALT: frozenset({'gear', 'mod', 'weapon'}),
    SALT_FOLDED: frozenset({'mod', 'weapon'}),
}

_HEX24 = re.compile(r'[0-9a-f]{24}')
_M = 0xFFFFFFFF


def _int32(v: int) -> int:
    v &= _M
    return v - 0x100000000 if v >= 0x80000000 else v


def template_hash(item_id: str) -> int | None:
    """Icon hash of a plain (unsalted, unbuilt) item, as the signed int32 used as the key of
    ``index.json``.  None for an id that is not a 24-hex MongoID."""
    if not isinstance(item_id, str) or not _HEX24.fullmatch(item_id):
        return None
    t, p, q = int(item_id[:8], 16), int(item_id[8:16], 16), int(item_id[16:], 16)
    return _int32(17 ^ t ^ ((p * 3637) & _M) ^ ((q * 5807) & _M))


def item_hash(item_id: str, salt: int = SALT_PLAIN) -> int | None:
    h = template_hash(item_id)
    return None if h is None else _int32(h ^ salt)


def read_index(cache_dir: str | None) -> dict[int, int]:
    """``{signed hash: file number}`` from the cache's ``index.json`` ({} when unreadable)."""
    if not cache_dir:
        return {}
    try:
        with open(os.path.join(cache_dir, 'index.json'), encoding='utf-8') as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def exact_ids(index: dict[int, int], items: dict[str, str]) -> dict[str, tuple[str, int]]:
    """``{'<n>.png': (item id, salt)}`` for every index entry whose hash reproduces exactly one
    item.  ``items`` = ``{item id: catalog category}`` (the category gates the salts).  A file
    claimed by two different items (never seen; it would take a 32-bit collision) is dropped."""
    by_hash: dict[int, list[tuple[str, int]]] = {}
    for iid, cat in items.items():
        h0 = template_hash(iid)
        if h0 is None:
            continue
        for salt, cats in SALTS.items():
            if cats is None or cat in cats:
                by_hash.setdefault(_int32(h0 ^ salt), []).append((iid, salt))
    out: dict[str, tuple[str, int]] = {}
    bad: set[str] = set()
    for h, n in index.items():
        hits = by_hash.get(h)
        if not hits:
            continue
        fname = f'{n}.png'
        ids = {iid for iid, _ in hits}
        prev = out.get(fname)
        if len(ids) > 1 or (prev is not None and prev[0] not in ids):
            bad.add(fname)
            continue
        out[fname] = hits[0]
    for fname in bad:
        out.pop(fname, None)
    return out


def validate(exact: dict[str, tuple[str, int]], evidence: dict[str, str]) -> dict:
    """Agreement of the exact ids with independent evidence ``{'<n>.png': item id}``.

    Open (``enabled``) when the two agree on >= 95 % of at least 20 shared files.  The gate guards
    against the game changing its hash: a wrong formula agrees on ~0 % (a chance match of a 32-bit
    hash into a ~5k-item set is ~1e-6), so modest evidence suffices, and the evidence itself is
    imperfect (picture associations can confuse look-alikes)."""
    overlap = agree = 0
    disagree: list[tuple[str, str, str]] = []
    for fname, iid in evidence.items():
        e = exact.get(fname)
        if e is None or not iid:
            continue
        overlap += 1
        if e[0] == iid:
            agree += 1
        else:
            disagree.append((fname, e[0], iid))
    agreement = agree / overlap if overlap else 0.0
    return {'provider': PROVIDER, 'enabled': overlap >= 20 and agreement >= 0.95,
            'agreement': round(agreement, 4), 'overlap': overlap, 'n_exact': len(exact),
            'disagree': disagree[:20]}


def status_path(data_dir: str) -> str:
    return os.path.join(data_dir, 'eft_hash_status.json')


def save_status(data_dir: str, status: dict) -> None:
    st = dict(status, checked_at=int(time.time()))
    try:
        os.makedirs(data_dir, exist_ok=True)
        tmp = status_path(data_dir) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, status_path(data_dir))
    except OSError:
        pass


def load_status(data_dir: str) -> dict | None:
    try:
        with open(status_path(data_dir), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def gate_open(status: dict | None) -> bool:
    return bool(status and status.get('enabled') and status.get('provider') == PROVIDER)
