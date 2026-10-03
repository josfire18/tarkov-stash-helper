"""
Raid advice for the Live page: what is worth grabbing, and what to drop to make room.

Pure functions (no I/O, stdlib + sellcalc) so the rules are unit-tested without a game
(tests/test_live.py).  app.py feeds in the scan's detections, the scene's grids and the sell
context and puts the result into the payload as ``advice``.

Rules
-----
* Only identified items count.  An *uncertain* identification is never ranked as treasure and
  never suggested for dropping.  A *provisional* one (fast first look of a progressive scan,
  certification pending) may be listed as a grab, flagged ``provisional``, but no drop is ever
  planned for it until the final result replaces it.
* grab  = detections in ``loot`` / ``container_window`` regions, valued per slot as the better of
  the trader's net and (when the item may be sold on the flea) the flea net after the fee.  Items
  needed for an open quest / hideout upgrade come first, flagged "Quest" / "Hideout", whatever they
  are worth; the rest must reach ``live_min_value_per_slot``; at most ``live_top_n`` in total.
  Guns are never priced (their value is their parts), so they are not ranked.
* drop  = only when the best grab does not fit into the free space of the tactical rig, pockets or
  backpack (the secure container, special slots and worn equipment are never touched): the
  cheapest-per-slot items on you - never one a quest / the hideout needs - whose removal makes room,
  and only when the grab is worth more than what is dropped (total and per slot; a quest item
  always is).
"""
from __future__ import annotations

import sellcalc

DROP_ROLES = ('own_rig', 'own_pockets', 'own_backpack')      # where a grab can be put, and taken from
RAID_SCENES = ('raid_inventory', 'raid_loot')
LOOT_ROLES = ('loot', 'container_window')
DEFAULT_TOP_N = 5
DEFAULT_MIN_PER_SLOT = 10000


def is_raid_scene(scene, in_raid=None) -> bool:
    return bool(in_raid) or (scene in RAID_SCENES)


def _unpriced(item, category=None) -> bool:
    """A gun (value = its parts) is never ranked or dropped."""
    return category == 'weapon' or 'gun' in ((item or {}).get('types') or ())


def unit_value(item, fir, settings, ctx) -> int:
    """Roubles one unit fetches: the better of trader and (if allowed) flea, after the flea fee."""
    blocked = sellcalc.flea_block_reason(item, fir, settings, ctx.get('rules'))
    rec = sellcalc.sell_recommendation(item, flea_blocked=blocked, count=1, ctx=ctx)
    return max(rec['trader_price'] or 0, rec['flea_net'] or 0)


def need_flag(item_id, protected, any_of_ids=()):
    """'Quest' / 'Hideout' when an open objective still needs this item, else None."""
    p = (protected or {}).get(item_id)
    if p is None:
        return 'Quest' if item_id in any_of_ids else None
    kinds = set(p.get('kinds') or ())
    return 'Hideout' if kinds == {'hideout'} else 'Quest'


def _entry(d, item, flag, value, per_slot, slots, why=''):
    return {'item_id': d['item_id'], 'name': item.get('name') or d.get('name'), 'count': d.get('count') or 1,
            'slots': slots, 'value': value, 'per_slot': per_slot, 'flag': flag, 'why': why,
            'px': d['px'], 'py': d['py'], 'pw': d['pw'], 'ph': d['ph'], 'w': _wh(d)[0], 'h': _wh(d)[1],
            'panel': d.get('panel'), 'col': d.get('col'), 'row': d.get('row')}


def _wh(d):
    return int(d.get('W') or d.get('w') or 1), int(d.get('H') or d.get('h') or 1)


def rank_grab(dets, id_to_item, protected, any_of_ids, settings, ctx, top_n=DEFAULT_TOP_N,
              min_per_slot=DEFAULT_MIN_PER_SLOT):
    """The loot worth taking, best first (see module doc)."""
    quest, rest = [], []
    for d in dets:
        # a provisional read (fast first look, certification pending) may be ranked, flagged;
        # an uncertain one never counts
        if d.get('role') not in LOOT_ROLES or (d.get('uncertain') and not d.get('provisional')):
            continue
        item = id_to_item.get(d['item_id'])
        if not item or _unpriced(item, d.get('category')):
            continue
        w, h = _wh(d)
        slots = max(1, w * h)
        count = d.get('count') or 1
        unit = unit_value(item, True, settings, ctx)           # looted in raid -> Found in Raid
        value = unit * count
        per_slot = value / slots
        flag = need_flag(d['item_id'], protected, any_of_ids)
        e = _entry(d, item, flag, value, round(per_slot), slots)
        if d.get('provisional'):
            e['provisional'] = True
        if flag:
            e['why'] = 'Needed for a quest' if flag == 'Quest' else 'Needed for the hideout'
            quest.append(e)
        elif per_slot >= max(0, min_per_slot) and value > 0:
            rest.append(e)
    key = lambda e: (-e['per_slot'], -e['value'], e['name'] or '')
    ordered = sorted(quest, key=key) + sorted(rest, key=key)
    return ordered[:max(1, int(top_n))]


# ---- free space ------------------------------------------------------------------------------

def _occupancy(grid):
    """rows x cols booleans (True = taken) from ``grid['cells']`` = [(col, row, w, h, empty)]."""
    occ = [[False] * grid['cols'] for _ in range(grid['rows'])]
    for col, row, w, h, empty in grid['cells']:
        if empty:
            continue
        for r in range(row, min(row + h, grid['rows'])):
            for c in range(col, min(col + w, grid['cols'])):
                occ[r][c] = True
    return occ


def _fits(occ, w, h) -> bool:
    """A w x h (or, rotated, h x w) block of free cells exists."""
    rows, cols = len(occ), len(occ[0]) if occ else 0
    for bw, bh in {(w, h), (h, w)}:
        for r in range(rows - bh + 1):
            for c in range(cols - bw + 1):
                if all(not occ[rr][cc] for rr in range(r, r + bh) for cc in range(c, c + bw)):
                    return True
    return False


def free_cells(grids) -> int:
    return sum(1 for g in grids if g.get('role') in DROP_ROLES
               for cell in g['cells'] if cell[4])


def plan_drop(grab, grids, own_dets, id_to_item, protected, any_of_ids, settings, ctx):
    """``(drop_entries, note)`` for the best grab (see module doc)."""
    if not grab:
        return [], ''
    best = grab[0]
    if best.get('provisional'):
        return [], 'checking items...'                          # no swap for an unverified read
    w, h = best['w'], best['h']
    cand_grids = [g for g in grids if g.get('role') in DROP_ROLES and g.get('rows') and g.get('cols')]
    if not cand_grids:
        return [], ''
    occs = {id(g): _occupancy(g) for g in cand_grids}
    if any(_fits(o, w, h) for o in occs.values()):
        return [], ''                                           # room already: nothing to drop
    by_pos = {(d.get('panel'), d.get('col'), d.get('row')): d for d in own_dets}
    plans = []
    for g in cand_grids:
        cands = []
        for col, row, cw, ch, empty in g['cells']:
            d = by_pos.get((g['panel'], col, row))
            if empty or d is None or d.get('uncertain'):
                continue
            item = id_to_item.get(d['item_id'])
            if not item or _unpriced(item, d.get('category')) or need_flag(d['item_id'], protected, any_of_ids):
                continue
            slots = max(1, cw * ch)
            value = unit_value(item, d.get('fir'), settings, ctx) * (d.get('count') or 1)
            cands.append((value / slots, value, col, row, cw, ch, d, item, slots))
        cands.sort(key=lambda c: (c[0], c[1]))
        occ = [r[:] for r in occs[id(g)]]
        taken = []
        for c in cands:
            if _fits(occ, w, h):
                break
            _free(occ, c)
            taken.append(c)
        if not _fits(occ, w, h):
            continue                                            # even clearing every candidate is not enough
        for c in list(reversed(taken)):                         # keep only what is needed
            _take(occ, c)
            if _fits(occ, w, h):
                taken.remove(c)
            else:
                _free(occ, c)
        plans.append((sum(c[1] for c in taken), taken))
    if not plans:
        return [], 'nothing you can drop would free enough room'
    lost, taken = min(plans, key=lambda p: (p[0], len(p[1])))
    if not (best['flag'] or (best['value'] > lost and best['per_slot'] > max(c[0] for c in taken))):
        return [], 'nothing on you is cheap enough to be worth swapping'
    out = []
    for per_slot, value, col, row, cw, ch, d, item, slots in sorted(taken, key=lambda c: c[0]):
        e = _entry(d, item, None, value, round(per_slot), slots, f'makes room for {best["name"]}')
        out.append(e)
    return out, ''


def _free(occ, c):
    _, _, col, row, cw, ch = c[:6]
    for r in range(row, row + ch):
        for cc in range(col, col + cw):
            if r < len(occ) and cc < len(occ[0]):
                occ[r][cc] = False


def _take(occ, c):
    _, _, col, row, cw, ch = c[:6]
    for r in range(row, row + ch):
        for cc in range(col, col + cw):
            if r < len(occ) and cc < len(occ[0]):
                occ[r][cc] = True


def compute_advice(dets, grids, id_to_item, protected, any_of, settings, ctx, scene=None, in_raid=None):
    """The payload's ``advice``: ``None`` outside a raid, else
    ``{'grab': [...], 'drop': [...], 'free_cells': n, 'note': str}``."""
    if not is_raid_scene(scene, in_raid):
        return None
    top_n = _num(settings.get('live_top_n'), DEFAULT_TOP_N, 1, 20)
    min_ps = _num(settings.get('live_min_value_per_slot'), DEFAULT_MIN_PER_SLOT, 0, 10 ** 9)
    any_ids = {a['id'] for g in (any_of or ()) for a in g.get('items', ())}
    grab = rank_grab(dets, id_to_item, protected, any_ids, settings, ctx, top_n, min_ps)
    own = [d for d in dets if str(d.get('role') or '').startswith('own_')]
    drop, note = plan_drop(grab, grids or [], own, id_to_item, protected, any_ids, settings, ctx)
    return {'grab': grab, 'drop': drop, 'free_cells': free_cells(grids or []), 'note': note}


def _num(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default
