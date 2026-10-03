"""tarkov.dev data source: json.tarkov.dev documents -> the records the app consumes.

tarkov.dev's GraphQL API (api.tarkov.dev) went down in July 2026 and the app's item
list froze.  tarkov.dev's own website reads static JSON documents instead
(``https://json.tarkov.dev/regular/<doc>``, gzip, ETag + If-None-Match, regenerated
within the hour), and those stayed up.  This module fetches them and converts them to
EXACTLY the record shapes the rest of the app already consumed from its GraphQL
queries (``app.PRICE_QUERY`` / ``app.TASKS_QUERY`` / ``app.RULES_QUERY``), so nothing
downstream (sellcalc, the task views, the identification catalog) changes:

* items  -> ``{id, name, shortName, basePrice, avg24hPrice, low24hPrice, lastLowPrice,
  iconLink, width, height, backgroundColor, gridImageLink, baseImageLink, types,
  sellFor: [{vendor: {name}, price, currency, priceRUB}]}``.  ``sellFor`` is the
  JSON's ``sellToTrader`` (trader ids resolved to names) plus a ``Flea Market`` entry
  at ``lastLowPrice`` - the same derivation tarkov.dev's own GraphQL resolver used
  (only when the item is not ``noFlea`` and a lowest offer exists).
* ``fleaMarket`` + ``traders`` -> the ``rules`` blob sellcalc reads (live flea fee
  rates, the Found-in-Raid rule, trader pay rates).
* tasks / hideout -> ``tasks`` and ``hideoutStations`` with the old shape (an item
  objective carries ``items`` = the JSON's whole ``items`` list and ``item`` = its first, as the
  deprecated GraphQL field did).

Names are translation keys in the JSON (``"<id> Name"``) resolved through the
``*_en`` documents.

Safety rules (a refresh must never make things worse):

* every document is fetched with its ETag, so an unchanged data set costs a few
  hundred bytes;
* a converted data set is validated against what is on disk and refused when it
  looks partial (see ``validate_*``); the caller keeps the last good cache;
* files are written to a temp file and moved into place (``write_json_atomic``);
* ETags are committed to the meta file only after the cache write succeeded.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import requests

JSON_BASE = 'https://json.tarkov.dev/regular/'
SOURCE_NAME = 'json.tarkov.dev'
HTTP_TIMEOUT = 60
USER_AGENT = 'TarkovStashHelper (+https://github.com/josfire18/tarkov-stash-helper)'

# Sanity floors for validation (the live data sets are ~5400 items / ~500 tasks / 26 stations).
MIN_ITEMS = 1000
MIN_TASKS = 50
MIN_STATIONS = 10
SHRINK_LIMIT = 0.8            # a new data set smaller than this fraction of the old one is refused
MIN_SELLFOR_SHARE = 0.5       # share of items that must carry at least one sellFor entry
MIN_IMAGE_SHARE = 0.9         # ... and a base image link

# Trader ids -> names, the last resort when traders/traders_en cannot be fetched and the
# meta file has nothing (ids are stable game ids).  Pay rates are not needed for names.
FALLBACK_TRADER_NAMES = {
    '54cb50c76803fa8b248b4571': 'Prapor',
    '54cb57776803fa99248b456e': 'Therapist',
    '579dc571d53a0658a154fbec': 'Fence',
    '58330581ace78e27b8b10cee': 'Skier',
    '5935c25fb3acc3127c3d8cd9': 'Peacekeeper',
    '5a7c2eca46aef81a7ca2145d': 'Mechanic',
    '5ac3b934156ae10c4430e83c': 'Ragman',
    '5c0647fdd443bc2504c2d371': 'Jaeger',
    '6617beeaa9cfa777ca915b7c': 'Ref',
}

ITEM_OBJECTIVES = ('giveItem', 'findItem', 'plantItem')   # GraphQL's TaskObjectiveItem
# Cache layout version of tasks_cache.json.  2 = item objectives carry ``items`` (every
# accepted alternative) next to the legacy ``item`` (the first one).  3 = tasks carry
# ``taskRequirements`` (the prerequisite tasks) and ``factionName`` (BEAR/USEC-only tasks), which
# the automatic task progress (eftlogs) needs.  A cache written before the current schema is
# re-fetched even when the ETags say "unchanged".
TASKS_SCHEMA = 3
FLEA_KEYS = ('minPlayerLevel', 'enabled', 'sellOfferFeeRate', 'sellRequirementFeeRate',
             'foundInRaidRequired', 'reputationLevels')


class SourceError(Exception):
    """json.tarkov.dev did not give usable data (network, HTTP status, bad/partial payload)."""


# ---------------------------------------------------------------------------
# HTTP + persistence
# ---------------------------------------------------------------------------

def _http_get(url, headers, timeout):
    """The one place the network is touched (tests replace it)."""
    return requests.get(url, headers=headers, timeout=timeout)


def write_json_atomic(path, data, indent=None):
    """Write JSON to a temp file next to ``path`` and move it into place, so a crash or a
    concurrent reader never sees a half-written file."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_meta(path):
    """The refresh bookkeeping (ETags, last check times, trader names, pending catalog work).
    A missing or unreadable file is an empty meta: everything is then fetched unconditionally."""
    try:
        with open(path, encoding='utf-8') as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError('not an object')
    except (OSError, ValueError):
        meta = {}
    for key in ('etags', 'checked', 'attempted', 'errors', 'sources'):
        if not isinstance(meta.get(key), dict):
            meta[key] = {}
    if not isinstance(meta.get('catalog_pending'), list):
        meta['catalog_pending'] = []
    return meta


def save_meta(path, meta):
    write_json_atomic(path, meta, indent=1)


def fetch_doc(doc, etag=None, timeout=HTTP_TIMEOUT):
    """GET one document.  Returns ``('unchanged', None, etag)`` on 304, ``('fresh', body, etag)``
    on 200.  Anything else raises SourceError."""
    headers = {'Accept-Encoding': 'gzip', 'User-Agent': USER_AGENT}
    if etag:
        headers['If-None-Match'] = etag
    try:
        r = _http_get(JSON_BASE + doc, headers, timeout)
    except requests.RequestException as e:
        raise SourceError(f'{doc}: {e}') from e
    if r.status_code == 304:
        return 'unchanged', None, etag
    if r.status_code != 200:
        raise SourceError(f'{doc}: HTTP {r.status_code}')
    try:
        body = r.json()
    except ValueError as e:
        raise SourceError(f'{doc}: response is not JSON') from e
    if not isinstance(body, dict) or not isinstance(body.get('data'), dict):
        raise SourceError(f"{doc}: unexpected payload (no 'data' object)")
    return 'fresh', body, (r.headers or {}).get('ETag')


def fetch_pair(meta, doc, en_doc, conditional):
    """A document and its English translations, conditionally.  ``None`` when both are
    unchanged.  When only one changed the other is fetched again unconditionally (a data
    set cannot be converted from half a pair).  Returns ``(body, en_body, etags)``; the
    caller commits ``etags`` to the meta once the converted result is safely written."""
    etags = meta['etags'] if conditional else {}
    s1, b1, t1 = fetch_doc(doc, etags.get(doc))
    s2, b2, t2 = fetch_doc(en_doc, etags.get(en_doc))
    if s1 == 'unchanged' and s2 == 'unchanged':
        return None
    if s1 == 'unchanged':
        s1, b1, t1 = fetch_doc(doc)
    if s2 == 'unchanged':
        s2, b2, t2 = fetch_doc(en_doc)
    return b1, b2, {doc: t1, en_doc: t2}


def _commit_etags(meta, etags):
    for k, v in etags.items():
        if v:
            meta['etags'][k] = v
        else:
            meta['etags'].pop(k, None)


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

def translator(en_body):
    """``tr(key, default=None)`` over an ``*_en`` document.  A key with an empty or missing
    translation gives ``default``."""
    table = (en_body or {}).get('data') or {}

    def tr(key, default=None):
        if not key:
            return default
        v = table.get(key)
        if v:
            return v
        return default
    return tr


# ---------------------------------------------------------------------------
# Traders
# ---------------------------------------------------------------------------

def convert_traders(traders_body, traders_en_body):
    """-> ``(names, pay_rates)``: ``{trader id: name}`` and
    ``{name: {'currency', 'pay_rates': {level: rate}}}`` (the ``rules['traders']`` shape).
    Placeholder traders with a ~0 pay rate (Lightkeeper, BTR driver, ...) have no rates."""
    tr = translator(traders_en_body)
    names, rules = {}, {}
    for tid, t in ((traders_body or {}).get('data') or {}).items():
        name = tr(t.get('name')) or FALLBACK_TRADER_NAMES.get(tid)
        if not name:
            continue
        names[tid] = name
        rates = {lv['level']: lv['payRate'] for lv in t.get('levels') or []
                 if lv.get('level') and (lv.get('payRate') or 0) >= 0.01}
        if rates:
            rules[name] = {'currency': t.get('currency'), 'pay_rates': rates}
    return names, rules


def refresh_traders(meta):
    """Trader names + pay rates, from the JSON (conditional) or the meta's last copy.
    Never raises: item prices must refresh even when traders cannot."""
    try:
        got = fetch_pair(meta, 'traders', 'traders_en', conditional=bool(meta.get('trader_names')))
        if got is not None:
            names, rules = convert_traders(got[0], got[1])
            if names:
                meta['trader_names'], meta['trader_rules'] = names, rules
                _commit_etags(meta, got[2])
    except SourceError:
        pass
    return dict(meta.get('trader_names') or FALLBACK_TRADER_NAMES), dict(meta.get('trader_rules') or {})


# ---------------------------------------------------------------------------
# Items / prices / rules
# ---------------------------------------------------------------------------

def convert_items(items_body, en_body, trader_names):
    """The ``items`` document -> ``(records, skipped)`` in the old ``PRICE_QUERY`` shape.
    Items whose name cannot be resolved are skipped (counted) rather than guessed."""
    tr = translator(en_body)
    out, skipped = [], 0
    for iid, it in ((items_body.get('data') or {}).get('items') or {}).items():
        name = tr(it.get('name'))
        if not name:
            skipped += 1
            continue
        short = tr(it.get('shortName')) or name
        types = list(it.get('types') or [])
        sell_for = []
        for s in it.get('sellToTrader') or []:
            vendor = trader_names.get(s.get('trader'))
            price_rub = s.get('priceRUB')
            if not vendor or not price_rub:
                continue
            sell_for.append({'vendor': {'name': vendor}, 'price': s.get('price'),
                             'currency': s.get('currency'), 'priceRUB': price_rub})
        last_low = it.get('lastLowPrice')
        if last_low and 'noFlea' not in types:
            sell_for.append({'vendor': {'name': 'Flea Market'}, 'price': last_low,
                             'currency': 'RUB', 'priceRUB': last_low})
        out.append({
            'id': it.get('id') or iid, 'name': name, 'shortName': short,
            'basePrice': it.get('basePrice'), 'avg24hPrice': it.get('avg24hPrice'),
            'low24hPrice': it.get('low24hPrice'), 'lastLowPrice': last_low,
            'iconLink': it.get('iconLink'), 'width': it.get('width'), 'height': it.get('height'),
            'backgroundColor': it.get('backgroundColor'), 'gridImageLink': it.get('gridImageLink'),
            'baseImageLink': it.get('baseImageLink'), 'types': types, 'sellFor': sell_for,
        })
    return out, skipped


def convert_rules(items_body, trader_rules):
    """``fleaMarket`` (from the items document) + trader pay rates -> sellcalc's ``rules`` blob,
    or None when there is nothing usable."""
    fm = ((items_body.get('data') or {}).get('fleaMarket')) or {}
    flea = {k: fm[k] for k in FLEA_KEYS if k in fm}
    traders = dict(trader_rules or {})
    if not flea and not traders:
        return None
    return {'flea': flea or None, 'traders': traders}


def validate_items(items, previous_items=None, skipped=0):
    """Raise SourceError unless ``items`` looks like a complete data set."""
    n = len(items)
    if n < MIN_ITEMS:
        raise SourceError(f'only {n} items (expected thousands)')
    if previous_items and n < SHRINK_LIMIT * len(previous_items):
        raise SourceError(f'{n} items, but the cache has {len(previous_items)} (looks partial)')
    if skipped > 0.02 * (n + skipped):
        raise SourceError(f'{skipped} items had no resolvable name (translations incomplete)')
    if sum(1 for i in items if i.get('sellFor')) < MIN_SELLFOR_SHARE * n:
        raise SourceError('too few items carry trader prices')
    if sum(1 for i in items if i.get('baseImageLink') and i.get('width') and i.get('height')) < MIN_IMAGE_SHARE * n:
        raise SourceError('too few items carry image links and sizes')


def refresh_prices(prices_path, meta, previous, now=None):
    """Bring ``prices_path`` up to date from json.tarkov.dev.

    ``previous`` is the cache currently on disk (None if absent or unusable; then nothing is
    conditional).  Returns ``{'status': 'unchanged'|'updated', 'cache': dict, 'new_ids': [...]}``.
    Raises SourceError and leaves the cache and the stored ETags untouched when the data is
    unreachable or fails validation.  The caller saves the meta."""
    now = time.time() if now is None else now
    prev_items = (previous or {}).get('items') or None
    conditional = bool(prev_items)
    trader_names, trader_rules = refresh_traders(meta)
    got = fetch_pair(meta, 'items', 'items_en', conditional)
    if got is None:
        meta['checked']['prices'] = now
        return {'status': 'unchanged', 'cache': previous, 'new_ids': []}
    items_body, en_body, etags = got
    items, skipped = convert_items(items_body, en_body, trader_names)
    validate_items(items, prev_items, skipped)
    cache = {'timestamp': now, 'source': SOURCE_NAME, 'items': items}
    rules = convert_rules(items_body, trader_rules) or (previous or {}).get('rules')
    if rules:
        cache['rules'] = rules
    write_json_atomic(prices_path, cache)
    _commit_etags(meta, etags)
    meta['checked']['prices'] = now
    meta['sources']['prices'] = SOURCE_NAME
    old_ids = {i['id'] for i in prev_items or ()}
    new_ids = [i['id'] for i in items if i['id'] not in old_ids] if old_ids else []
    return {'status': 'updated', 'cache': cache, 'new_ids': new_ids}


# ---------------------------------------------------------------------------
# Tasks / hideout
# ---------------------------------------------------------------------------

def _item_ref(item_id, item_names):
    name, short = (item_names or {}).get(item_id, (None, None))
    return {'id': item_id, 'name': name or item_id, 'shortName': short or name or item_id}


def objective_items(obj):
    """Every item an item objective accepts, as ``[{'id', 'name', 'shortName'}]`` (de-duplicated,
    order kept).  Reads the ``items`` list; a cache written before it existed (or a GraphQL
    answer with only ``item``) falls back to the single ``item``.  [] for non-item objectives."""
    refs = [r for r in (obj.get('items') or ()) if r and r.get('id')]
    if not refs and (obj.get('item') or {}).get('id'):
        refs = [obj['item']]
    seen, out = set(), []
    for r in refs:
        if r['id'] not in seen:
            seen.add(r['id'])
            out.append(r)
    return out


def convert_tasks(tasks_body, en_body, trader_names, item_names):
    """The ``tasks`` document -> the old ``TASKS_QUERY`` task list."""
    tr = translator(en_body)
    out = []
    for tid, t in ((tasks_body.get('data') or {}).get('tasks') or {}).items():
        objectives = []
        for o in t.get('objectives') or []:
            rec = {'id': o.get('id'), 'type': o.get('type')}
            ids = o.get('items') or []
            if o.get('type') in ITEM_OBJECTIVES and ids:
                refs = [_item_ref(i, item_names) for i in dict.fromkeys(ids)]   # any ONE of these
                rec.update({'count': o.get('count') or 0, 'foundInRaid': bool(o.get('foundInRaid')),
                            'item': refs[0], 'items': refs})
            objectives.append(rec)
        out.append({
            'id': t.get('id') or tid, 'name': tr(t.get('name'), t.get('normalizedName') or tid),
            'minPlayerLevel': t.get('minPlayerLevel') or 0,
            'kappaRequired': bool(t.get('kappaRequired')),
            'factionName': t.get('factionName') or 'Any',
            'taskRequirements': [{'task': {'id': r['task']}, 'status': list(r.get('status') or ())}
                                 for r in t.get('taskRequirements') or () if r.get('task')],
            'trader': {'name': trader_names.get(t.get('trader')) or '?'},
            'objectives': objectives,
        })
    return out


def convert_hideout(hideout_body, en_body, item_names):
    """The ``hideout`` document -> the old ``hideoutStations`` list."""
    tr = translator(en_body)
    out = []
    for sid, s in (hideout_body.get('data') or {}).items():
        levels = []
        for lv in s.get('levels') or []:
            reqs = [{'count': r.get('count') or 0, 'item': _item_ref(r.get('item'), item_names)}
                    for r in lv.get('itemRequirements') or [] if r.get('item')]
            levels.append({'id': lv.get('id'), 'level': lv.get('level'), 'itemRequirements': reqs})
        out.append({'id': s.get('id') or sid,
                    'name': tr(s.get('name'), s.get('normalizedName') or sid), 'levels': levels})
    return out


def validate_tasks(tasks, previous_tasks=None):
    if len(tasks) < MIN_TASKS:
        raise SourceError(f'only {len(tasks)} tasks')
    if previous_tasks and len(tasks) < SHRINK_LIMIT * len(previous_tasks):
        raise SourceError(f'{len(tasks)} tasks, but the cache has {len(previous_tasks)} (looks partial)')
    if not any(o.get('item') for t in tasks for o in t['objectives']):
        raise SourceError('no item objectives in the task data')
    if any(not t.get('name') for t in tasks):
        raise SourceError('tasks without names (translations incomplete)')


def validate_stations(stations, previous=None):
    if len(stations) < MIN_STATIONS:
        raise SourceError(f'only {len(stations)} hideout stations')
    if previous and len(stations) < SHRINK_LIMIT * len(previous):
        raise SourceError(f'{len(stations)} stations, but the cache has {len(previous)} (looks partial)')
    if not any(r for s in stations for lv in s['levels'] for r in lv['itemRequirements']):
        raise SourceError('no item requirements in the hideout data')


def refresh_tasks(tasks_path, meta, previous, item_names, now=None):
    """Bring the tasks/hideout cache up to date.  Tasks and hideout are independent halves:
    an unchanged half is carried over, a half that fails validation keeps its previous
    content (error raised only if nothing usable results).  Same return contract as
    :func:`refresh_prices` (without ``new_ids``)."""
    now = time.time() if now is None else now
    prev_tasks = (previous or {}).get('tasks') or None
    prev_stations = (previous or {}).get('hideoutStations') or None
    trader_names, _ = refresh_traders(meta)

    tasks, stations, etags, errors = prev_tasks, prev_stations, {}, []
    changed = False
    old_layout = (previous or {}).get('schema', 1) < TASKS_SCHEMA     # no ``items`` lists yet
    got = fetch_pair(meta, 'tasks', 'tasks_en', bool(prev_tasks) and not old_layout)
    if got is not None:
        try:
            new = convert_tasks(got[0], got[1], trader_names, item_names)
            validate_tasks(new, prev_tasks)
            tasks, changed = new, True
            etags.update(got[2])
        except SourceError as e:
            errors.append(f'tasks: {e}')
    got = fetch_pair(meta, 'hideout', 'hideout_en', bool(prev_stations))
    if got is not None:
        try:
            new = convert_hideout(got[0], got[1], item_names)
            validate_stations(new, prev_stations)
            stations, changed = new, True
            etags.update(got[2])
        except SourceError as e:
            errors.append(f'hideout: {e}')
    if errors and not (tasks and stations):
        raise SourceError('; '.join(errors))
    meta['checked']['tasks'] = now
    if not changed:
        if errors:                       # something was refused but the old data stands
            meta['errors']['tasks_partial'] = '; '.join(errors)
        return {'status': 'unchanged', 'cache': previous}
    cache = {'timestamp': now, 'source': SOURCE_NAME, 'schema': TASKS_SCHEMA,
             'tasks': tasks, 'hideoutStations': stations}
    write_json_atomic(tasks_path, cache)
    _commit_etags(meta, etags)
    meta['sources']['tasks'] = SOURCE_NAME
    if errors:
        meta['errors']['tasks_partial'] = '; '.join(errors)
    else:
        meta['errors'].pop('tasks_partial', None)
    return {'status': 'updated', 'cache': cache}


def item_names_from_prices(cache):
    """{item id: (name, shortName)} from a prices cache, for naming task/hideout items."""
    return {i['id']: (i.get('name'), i.get('shortName')) for i in (cache or {}).get('items') or ()}
