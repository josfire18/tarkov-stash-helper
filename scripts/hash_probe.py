"""
Check the icon-cache hash (identify/eft_hash.py) against this machine's icon cache and open or
close its validation gate (data/eft_hash_status.json).

Evidence the exact ids are compared with:
  * label-certified bindings (data/learned_icons.json, matched to cache files by pixel content)
  * the catalog's unambiguous picture associations (render ~ tarkov.dev icon, nothing else close)

    python scripts/hash_probe.py [--cache DIR] [--verbose]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from identify import eft_hash  # noqa: E402
from identify.anchors import ASSOC_TRUST_D, ASSOC_TRUST_RATIO, content_key_of  # noqa: E402
from identify.catalog import category_of, default_cache_dir, load_catalog  # noqa: E402
from identify.config import DATA_DIR, LEARNED_PATH, PRICES_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    cdir = a.cache or default_cache_dir()
    if not cdir:
        sys.exit('no EFT icon cache found')
    with open(PRICES_PATH, encoding='utf-8') as f:
        items = {it['id']: category_of(it.get('types')) for it in json.load(f)['items']}
    index = eft_hash.read_index(cdir)
    exact = eft_hash.exact_ids(index, items)
    files = {f for f in os.listdir(cdir) if f.endswith('.png')}
    on_disk = {k: v for k, v in exact.items() if k in files}
    salts = collections.Counter(s for _, s in on_disk.values())
    print(f'index: {len(index)} hashes, {len(files)} files; exact ids: {len(exact)} entries, '
          f'{len(on_disk)} files on disk ({len({i for i, _ in on_disk.values()})} distinct items); '
          f'salts {dict(salts)}')

    evidence: dict[str, str] = {}
    try:
        with open(LEARNED_PATH, encoding='utf-8') as f:
            learned = json.load(f)
    except (OSError, ValueError):
        learned = {}
    n_learned = 0
    if learned:
        for fname in sorted(on_disk):
            img = cv2.imread(os.path.join(cdir, fname), cv2.IMREAD_UNCHANGED)
            e = learned.get(content_key_of(img)) if img is not None else None
            if e and (e.get('certain') or e.get('seen', 1) >= 2):
                evidence[fname] = e['id']
                n_learned += 1
    cat = load_catalog(log=lambda *x: None)
    n_assoc = 0
    for fname, (iid, d, d2, *_) in (cat.meta.get('cache_assoc') or {}).items():
        if iid and fname not in evidence and d <= ASSOC_TRUST_D and d2 >= ASSOC_TRUST_RATIO * max(d, 0.3):
            evidence[fname] = iid
            n_assoc += 1
    st = eft_hash.validate(exact, evidence)
    print(f'evidence: {n_learned} learned + {n_assoc} picture associations; '
          f'overlap {st["overlap"]}, agreement {st["agreement"]:.2%} -> gate {"OPEN" if st["enabled"] else "closed"}')
    if a.verbose or not st['enabled']:
        names = {str(i): str(n) for i, n in zip(cat.ids, cat.names)}
        for fname, e, v in st['disagree']:
            print(f'  {fname}: hash {names.get(e, e)}  vs evidence {names.get(v, v)}')
    eft_hash.save_status(DATA_DIR, st)


if __name__ == '__main__':
    main()
