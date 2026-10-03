import os
import sys

if __name__ == '__main__' and '--watch' in sys.argv:
    # Watcher mode (lifecycle.py): the few-MB process Windows starts at sign-in that launches the
    # real app when Tarkov starts.  Dispatched HERE, before the numpy / cv2 / flask imports below,
    # so it never loads them (also true of the packaged exe: a PyInstaller bundle imports lazily).
    import lifecycle
    sys.exit(lifecycle.watch_main(sys.argv[1:]))

import shutil
import json
import base64
import uuid
import threading
import time
import subprocess
from io import BytesIO

import re
import requests as http_requests

from flask import Flask, jsonify, request, render_template, redirect
import mss
from PIL import Image, ImageDraw
import pytesseract
from rapidfuzz import process as rfuzz
from pynput import keyboard
import cv2
import numpy as np

import sellcalc
import tarkovdata
import lifecycle    # open/close with the game (stdlib + ctypes only)
# The sell-advice economics live in sellcalc.py (pure functions); the historical
# names stay importable from here (test_scan.py scores sell decisions via app.*).
from sellcalc import (best_trader_price, calc_flea_fee, price_420, flea_block_reason,  # noqa: F401
                      sell_recommendation, order_for_selling, TRADER_ORDER)

APP_VERSION = '0.3.1'

FROZEN = getattr(sys, 'frozen', False)

# When run from source, data/ lives next to app.py. When packaged with
# PyInstaller (--onefile), __file__ resolves inside the temp extraction dir,
# so data/ must instead live next to the .exe or user settings/caches would
# vanish every launch.
if FROZEN:
    BASE = os.path.dirname(sys.executable)
    # templates/ (and any other bundled read-only assets) still ship inside
    # the PyInstaller bundle.
    BUNDLE = getattr(sys, '_MEIPASS', BASE)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLE = BASE

app = Flask(__name__, template_folder=os.path.join(BUNDLE, 'templates'))

DATA = os.path.join(BASE, 'data')
SETTINGS_PATH  = os.path.join(DATA, 'settings.json')
KEEPLIST_PATH  = os.path.join(DATA, 'keep_list.json')
PRICES_PATH    = os.path.join(DATA, 'prices_cache.json')
KAPPA_WIKI_PATH  = os.path.join(DATA, 'kappa_wiki.json')   # cached Collector item names from the wiki
PRESTIGE_WIKI_PATH = os.path.join(DATA, 'prestige_wiki.json')  # cached Prestige requirements from the wiki
TASKS_CACHE_PATH = os.path.join(DATA, 'tasks_cache.json')  # cached tasks + hideout requirements (tarkov.dev)
META_PATH        = os.path.join(DATA, 'tarkovdev_meta.json')  # ETags, last-check times, pending catalog work
PROGRESS_PATH    = os.path.join(DATA, 'progress.json')     # user task/hideout completion + have-counts
TMPL_SRC_DIR   = os.path.join(DATA, 'tmpl_src')        # transparent per-slot base images (BGRA PNG)
os.makedirs(DATA, exist_ok=True)
os.makedirs(TMPL_SRC_DIR, exist_ok=True)

# Tesseract-OCR is an external (non-pip) dependency the user must install
# separately. pytesseract only finds it automatically if it's on PATH; fall
# back to the default Windows install location so a fresh setup doesn't
# require manually editing PATH.
if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
    _default_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(_default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _default_tesseract

PRICE_CACHE_TTL      = 1800  # seconds (30 min): older and a scan refreshes on demand (the refresher normally beats this)
PRICE_REFRESH_INTERVAL = 900   # seconds (15 min): background refresh of items/prices (a 304 when nothing changed)
TASKS_REFRESH_INTERVAL = 3 * 3600  # seconds (3 h): background refresh of tasks/hideout/traders
RETRY_BACKOFF        = 300   # seconds: after a failed refresh, wait this long before trying again
KAPPA_WIKI_TTL       = 86400 # seconds (24 h) — Collector list changes rarely
PRESTIGE_WIKI_TTL    = 7 * 86400 # seconds (7 d) — Prestige requirements change only per major patch
TASKS_CACHE_TTL      = 86400 # seconds (24 h) — task/hideout requirements change per patch
FLEA_MIN_PROFIT      = sellcalc.FLEA_MIN_GAIN  # recommend flea only if net > trader by this much (setting: flea_min_gain)
TARKOV_API           = 'https://api.tarkov.dev/graphql'
# Self-update: owner/repo are baked in here and NEVER taken from the client —
# the download URL that ends up in _update_state always traces back to this
# exact GitHub API call.
GITHUB_RELEASES_API  = 'https://api.github.com/repos/josfire18/tarkov-stash-helper/releases/latest'
UPDATE_ASSET_NAME    = 'TarkovStashHelper.exe'
UPDATE_CACHE_TTL     = 6 * 3600  # seconds (6 h)
# The raw wiki page (escapefromtarkov.fandom.com/wiki/Collector) sits behind a
# Cloudflare bot challenge and returns "Just a moment..." to plain HTTP clients.
# The MediaWiki API serves the identical rendered HTML unchallenged — always
# fetch through it, never the page URL.
COLLECTOR_WIKI_API   = ('https://escapefromtarkov.fandom.com/api.php'
                        '?action=parse&page=Collector&prop=text&format=json&formatversion=2')
# Same Cloudflare-avoidance pattern as COLLECTOR_WIKI_API — the Prestige page
# must be fetched through the MediaWiki API too, never the plain page URL.
PRESTIGE_WIKI_API    = ('https://escapefromtarkov.fandom.com/api.php'
                        '?action=parse&page=Prestige&prop=text&format=json&formatversion=2')


# Per-trader badge colours (RGB — PIL's ImageDraw, unlike the OpenCV/BGR
# pipeline above, takes RGB(A) tuples) so the sell-scan screenshot's numbered
# badges visually batch by trader instead of all sharing one gold colour.
# Picked to stay distinct from the flea badge (30,150,30 green) and the KEEP
# badge (0,140,180 cyan) used elsewhere in draw_badge calls in this module.
TRADER_COLORS_RGB = {
    'Prapor':      (204, 80, 40),    # orange-red
    'Therapist':   (0, 150, 136),    # teal
    'Skier':       (66, 133, 244),   # blue
    'Peacekeeper': (94, 92, 230),    # indigo
    'Mechanic':    (154, 205, 50),   # yellow-green
    'Ragman':      (216, 27, 96),    # magenta
    'Jaeger':      (34, 139, 34),    # forest green
    'Ref':         (230, 190, 40),   # yellow
    'Fence':       (120, 120, 120),  # grey
}
DEFAULT_TRADER_BADGE_RGB = (180, 120, 20)  # fallback — the old uniform trader-gold
FLEA_RGB = (30, 150, 30)
FLEA_QUEUE_RGB = (95, 125, 95)   # flea picks beyond the offer slots: same family, visibly dimmer


PRICE_QUERY = '''{
  items {
    id name shortName basePrice avg24hPrice low24hPrice lastLowPrice iconLink width height
    backgroundColor gridImageLink baseImageLink types
    sellFor { vendor { name } priceRUB price currency }
  }
}'''

# The flea market's live rules and what each trader pays, from the game's own
# globals via tarkov.dev (schema checked 2026-10-01).  A separate, best-effort
# query: if it fails the item prices still refresh and sellcalc's documented
# constants apply.  Rates are fractions (0.05 = 5 %).
RULES_QUERY = '''{
  fleaMarket {
    minPlayerLevel enabled sellOfferFeeRate sellRequirementFeeRate foundInRaidRequired
    reputationLevels { offers offersSpecialEditions minRep maxRep }
  }
  traders { name currency { shortName } levels { level payRate } }
}'''

TASKS_QUERY = '''{
  tasks {
    id name minPlayerLevel kappaRequired
    trader { name }
    objectives {
      id type
      ... on TaskObjectiveItem { count foundInRaid item { id name shortName } items { id name shortName } }
    }
  }
  hideoutStations {
    id name
    levels { id level itemRequirements { count item { id name shortName } } }
  }
}'''


class ScanError(Exception):
    """User-actionable scan failure (no region set, capture failed, ...)."""


# Shared state for hotkey-triggered scans
_last_scan = {'image': None, 'detections': [], 'ts': 0, 'grid_failed': False,
              'error': None, 'warnings': [], 'checklist_matches': []}
_scan_lock = threading.Lock()

# Live progress of the currently running scan, polled by the frontend.
_scan_state = {'running': False, 'phase': None, 'done': 0, 'total': 0, 'ts': 0}

# /api/update-check result, cached UPDATE_CACHE_TTL seconds so the UI's
# poll-on-load doesn't hit GitHub on every page load.
_update_cache = {'ts': 0, 'data': None}
# The exe asset's download URL/size for the latest checked release, kept
# server-side only — /api/update-apply reads this instead of trusting
# anything the client might send.
_update_state = {'download_url': None, 'asset_size': None, 'tag': None}

# The region-picker modal (on both pages) already grabs a full-monitor frame
# via /api/calibration-screenshot to show the user something to drag a
# rectangle over. Stashing that same frame here lets a scan triggered right
# after a region save reuse it instead of live-grabbing — a live re-grab at
# that moment would capture the helper's own window sitting on top of the
# game. Only used when the caller opts in (from_calibration=True) and the
# frame is still fresh; see _calibration_crop_if_fresh.
_last_calibration = {'bgr': None, 'ts': 0}
CALIBRATION_TTL = 180  # seconds — how stale the stashed calibration frame may be before it's ignored


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_json(path, default_fn):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    data = default_fn()
    save_json(path, data)
    return data

def save_json(path, data):
    tarkovdata.write_json_atomic(path, data, indent=2)     # temp file + os.replace: never half-written


# ---------------------------------------------------------------------------
# Self-update — version comparison
# ---------------------------------------------------------------------------

def _ver_tuple(s):
    """
    Parse a version string ('0.3.1', 'v0.3.1', 'v0.3.0-beta1') into a tuple of
    ints (0, 3, 1). Tolerant of a leading 'v'/'V' and any junk after the
    numeric dot-groups (a trailing '-beta1', build metadata, etc. is simply
    ignored — only the leading numeric run is parsed). Returns None if the
    string has no leading numeric dot-group at all (e.g. '', 'garbage').
    """
    if not s:
        return None
    s = s.strip()
    if s[:1] in ('v', 'V'):
        s = s[1:]
    m = re.match(r'\d+(?:\.\d+)*', s)
    if not m:
        return None
    return tuple(int(x) for x in m.group().split('.'))


def _is_newer(remote, local):
    """
    True iff `remote` version string is strictly newer than `local`.
    Unequal-length tuples (e.g. '0.3' vs '0.3.1') are padded on the right with
    zeros before comparing. False (never raises) when either side fails to
    parse, or on ties/older.
    """
    rt, lt = _ver_tuple(remote), _ver_tuple(local)
    if rt is None or lt is None:
        return False
    n = max(len(rt), len(lt))
    rt = rt + (0,) * (n - len(rt))
    lt = lt + (0,) * (n - len(lt))
    return rt > lt


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

class PriceFetchError(Exception):
    """tarkov.dev did not return item data (outage, rate limit, schema change)."""


def _graphql_prices():
    """Item prices from tarkov.dev's GraphQL API (the fallback source) as a cache dict.
    Not written anywhere; raises PriceFetchError."""
    r = http_requests.post(TARKOV_API, json={'query': PRICE_QUERY}, timeout=30)
    try:
        body = r.json()
    except ValueError:
        raise PriceFetchError(f'tarkov.dev returned HTTP {r.status_code} with no JSON')
    items = (body.get('data') or {}).get('items')
    if not items:
        errs = '; '.join(e.get('message', '?') if isinstance(e, dict) else str(e)
                         for e in body.get('errors') or []) or 'no items'
        raise PriceFetchError(f'tarkov.dev HTTP {r.status_code}: {errs}')
    return {'timestamp': time.time(), 'source': 'graphql', 'items': items}


def parse_sell_rules(data):
    """The RULES_QUERY response's ``data`` as the compact blob sellcalc reads:
    {'flea': {...fleaMarket...}, 'traders': {name: {'currency', 'pay_rates': {level: rate}}}}.
    None when there is nothing usable."""
    data = data or {}
    flea = data.get('fleaMarket') or None
    traders = {}
    for t in data.get('traders') or []:
        name = t.get('name')
        rates = {lv['level']: lv['payRate'] for lv in t.get('levels') or []
                 if lv.get('level') and lv.get('payRate')}
        if name and rates:
            traders[name] = {'currency': (t.get('currency') or {}).get('shortName'),
                             'pay_rates': rates}
    if not flea and not traders:
        return None
    return {'flea': flea, 'traders': traders}


def fetch_sell_rules():
    """Best-effort fetch of the flea fee rates / FiR rule / trader pay rates (GraphQL).
    Never raises: an outage or a schema change leaves the documented constants
    in force (sellcalc)."""
    try:
        r = http_requests.post(TARKOV_API, json={'query': RULES_QUERY}, timeout=20)
        return parse_sell_rules((r.json() or {}).get('data'))
    except Exception as e:
        print(f'[prices] sell rules unavailable, using built-in constants: {e}')
        return None


# Refresh bookkeeping shared by the on-demand paths and the background refresher.
_refresh_lock = threading.RLock()     # one refresh at a time (the refresher, a scan, the Refresh button)
_refresh_state = {                    # in-memory status, merged into /api/prices/status
    'prices': {'attempt': 0, 'error': None},
    'tasks':  {'attempt': 0, 'error': None},
}


def _load_cache(path):
    """A cache file's content, or None if it is missing/unreadable (never creates it)."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _load_meta():
    return tarkovdata.load_meta(META_PATH)


def _save_meta_quietly(meta):
    try:
        tarkovdata.save_meta(META_PATH, meta)
    except OSError as e:
        print(f'[refresh] could not save refresh bookkeeping: {e}')


def _usable_prices(cache):
    return cache if cache and cache.get('items') else None


def fetch_prices_graphql(previous=None):
    """Refresh the price cache from the GraphQL API alone (the fallback source).  Validated
    like the JSON source, written atomically, returns the cache.  Raises PriceFetchError."""
    cache = _graphql_prices()
    previous = previous if previous is not None else _usable_prices(_load_cache(PRICES_PATH))
    try:
        tarkovdata.validate_items(cache['items'], (previous or {}).get('items'))
    except tarkovdata.SourceError as e:
        raise PriceFetchError(f'tarkov.dev GraphQL data refused: {e}')
    rules = fetch_sell_rules() or (_load_cache(PRICES_PATH) or {}).get('rules')
    if rules:
        cache['rules'] = rules
    tarkovdata.write_json_atomic(PRICES_PATH, cache)
    return cache


def refresh_prices(force=False):
    """Bring the price cache up to date.  Source order: json.tarkov.dev (conditional, cheap
    when nothing changed), then the GraphQL API, then - as an error - the cache that is
    already on disk (the callers' stale-cache fallback).  Never replaces a good cache with
    a bad one.  Returns ``{'status', 'source', 'cache', 'new_ids'}``; a new item id on a
    machine that already has a catalog queues the catalog update."""
    with _refresh_lock:
        previous = _usable_prices(_load_cache(PRICES_PATH))
        meta = _load_meta()
        _refresh_state['prices']['attempt'] = meta['attempted']['prices'] = time.time()
        if force:
            meta['etags'] = {}
        try:
            try:
                res = tarkovdata.refresh_prices(PRICES_PATH, meta, previous)
                res['source'] = tarkovdata.SOURCE_NAME
            except Exception as json_err:      # network, bad payload, a converter bug: all fall back
                print(f'[prices] json.tarkov.dev failed ({json_err}); trying the GraphQL API')
                try:
                    cache = fetch_prices_graphql(previous)
                except (PriceFetchError, http_requests.RequestException) as gql_err:
                    raise PriceFetchError(f'json.tarkov.dev: {json_err}; GraphQL: {gql_err}')
                meta['checked']['prices'] = cache['timestamp']
                meta['sources']['prices'] = 'graphql'
                meta['etags'].pop('items', None)      # the cache is no longer the JSON data: never 304 against it
                meta['etags'].pop('items_en', None)
                old_ids = {i['id'] for i in (previous or {}).get('items', ())}
                res = {'status': 'updated', 'source': 'graphql', 'cache': cache,
                       'new_ids': [i['id'] for i in cache['items'] if i['id'] not in old_ids] if old_ids else []}
        except PriceFetchError as e:
            meta['errors']['prices'] = str(e)
            _refresh_state['prices']['error'] = str(e)
            _save_meta_quietly(meta)
            raise
        meta['errors'].pop('prices', None)
        _refresh_state['prices']['error'] = None
        if res['new_ids']:
            meta['catalog_pending'] = sorted(set(meta['catalog_pending']) | set(res['new_ids']))
        _save_meta_quietly(meta)
    if res['new_ids']:
        print(f"[prices] {len(res['new_ids'])} new items")
        _queue_catalog_update()
    return res


def fetch_prices():
    """Refresh the price cache now (see :func:`refresh_prices`) and return it."""
    return refresh_prices()['cache']


def _cache_age_seconds(cache, kind):
    """Seconds since this cache was last confirmed current: written, or re-checked
    (a 304 from json.tarkov.dev rewrites nothing)."""
    ts = (cache or {}).get('timestamp', 0)
    try:
        ts = max(ts, _load_meta()['checked'].get(kind) or 0)
    except Exception:
        pass
    return time.time() - ts


def get_prices():
    """Return cached prices, refreshing if stale.

    The background refresher normally keeps the cache current, so this is cheap.  A failed
    refresh (tarkov.dev outage) falls back to the last good cache, marked ``stale_error`` so
    the UI can say how old the prices are - a temporary API problem must not stop a sell
    scan - and is not retried for RETRY_BACKOFF seconds (a scan never waits on a dead
    network twice in a row).
    """
    cache = _usable_prices(_load_cache(PRICES_PATH))
    if cache and _cache_age_seconds(cache, 'prices') < PRICE_CACHE_TTL:
        return cache
    err = _refresh_state['prices']['error']
    if cache and err and time.time() - _refresh_state['prices']['attempt'] < RETRY_BACKOFF:
        return {**cache, 'stale_error': err}
    try:
        return fetch_prices()
    except (PriceFetchError, http_requests.RequestException) as e:
        if not cache:
            raise
        print(f'[prices] refresh failed, using cached prices: {e}')
        return {**cache, 'stale_error': str(e)}

def build_price_index(cache):
    """Build name/shortname lookup from cache."""
    idx = {}
    for item in cache.get('items', []):
        idx[item['name'].lower()] = item
        idx[item['shortName'].lower()] = item
    return idx


# ---------------------------------------------------------------------------
# Shared scan helpers
# ---------------------------------------------------------------------------

_tesseract_ok = None

def tesseract_available():
    """True if the Tesseract binary is installed and runnable (cached)."""
    global _tesseract_ok
    if _tesseract_ok is None:
        try:
            pytesseract.get_tesseract_version()
            _tesseract_ok = True
        except Exception:
            _tesseract_ok = False
    return _tesseract_ok


def capture_stash_image(settings, require_region=False):
    """
    Grab the configured monitor/region and return (img_pil, img_bgr).
    Raises ScanError when require_region is set but no region is configured.
    """
    region = settings.get('region')
    if require_region and not region:
        raise ScanError('No stash region set. Click "📐 Set Stash Region" first.')
    with mss.mss() as sct:
        monitors = sct.monitors
        if len(monitors) < 2:
            raise ScanError('No monitor available for capture.')
        monitor_idx = settings.get('monitor', 0)
        monitor = monitors[monitor_idx + 1] if monitor_idx + 1 < len(monitors) else monitors[1]
        if region:
            capture_region = {
                'left':   monitor['left'] + region['x'],
                'top':    monitor['top']  + region['y'],
                'width':  region['w'],
                'height': region['h'],
            }
        else:
            capture_region = monitor
        raw = sct.grab(capture_region)
        img = Image.frombytes('RGB', raw.size, raw.bgra, 'raw', 'BGRX')
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img, img_bgr


def _crop_calibration(frame_bgr, region):
    """
    Crop a full-monitor BGR frame to `region` ({'x','y','w','h'}), the same
    coordinate space the region-picker drew its rectangle in.  Returns the
    cropped ndarray, or None if the frame/region is missing or the crop
    would be empty (e.g. the region falls outside the captured frame).
    """
    if frame_bgr is None or not region:
        return None
    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def _calibration_crop_if_fresh(settings):
    """
    Return a cropped BGR frame from the last calibration screenshot, if a
    region is configured and the stashed frame is still within CALIBRATION_TTL
    seconds old. Returns None otherwise (caller falls back to a live grab).
    """
    region = settings.get('region')
    if not region:
        return None
    if time.time() - _last_calibration.get('ts', 0) >= CALIBRATION_TTL:
        return None
    return _crop_calibration(_last_calibration.get('bgr'), region)


def capture_for_scan(settings, from_calibration, require_region, warnings):
    """
    Shared capture step for both scan pipelines (keep-scan and sell-scan).

    When from_calibration is True and the region-picker's stashed full-monitor
    frame is still fresh, crop it to the configured region instead of live-
    grabbing — avoids capturing the helper's own window on top of the game
    right after a region save. Falls back to a normal live grab (and records
    a warning) when the calibration frame is stale or missing.
    """
    if from_calibration:
        img_bgr = _calibration_crop_if_fresh(settings)
        if img_bgr is not None:
            img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            return img, img_bgr
        warnings.append('Calibration frame unavailable or stale — used a live screen grab instead.')
    return capture_stash_image(settings, require_region=require_region)


def map_keep_entries_to_ids(keep_list, price_idx):
    """
    Map every keep-list entry to its tarkov.dev item, conservatively.
    Returns ({tdev_id: entry}, [unmapped_names]).

    Order of trust: exact full-name match > persisted tdev_id > unambiguous
    alias hits (≥4 chars, all agreeing on one item) > tight fuzzy fallback.
    The old first-alias-wins logic let generic aliases like "Beer" or "Water"
    claim the wrong item entirely.
    """
    id_ok = set()
    for it_key, it in price_idx.items():
        id_ok.add(it['id'])

    mapped, unmapped = {}, []
    for cat in keep_list['categories']:
        for entry in cat['items']:
            data = price_idx.get(entry['name'].lower())
            if data is None and entry.get('tdev_id') in id_ok:
                mapped[entry['tdev_id']] = entry
                continue
            if data is None:
                hits = {}
                for alias in entry.get('aliases', []):
                    if len(alias) < 4:
                        continue
                    hit = price_idx.get(alias.lower())
                    if hit:
                        hits[hit['id']] = hit
                if len(hits) == 1:
                    data = next(iter(hits.values()))
            if data is None:
                names = [k for k in price_idx]
                r = rfuzz.extractOne(entry['name'].lower(), names, score_cutoff=93)
                if r:
                    data = price_idx[r[0]]
            if data is not None:
                mapped[data['id']] = entry
            else:
                unmapped.append(entry['name'])
    return mapped, unmapped

# ---------------------------------------------------------------------------
# Item catalog build (the "Build Icon DB" button on the Sell Advisor page)
#
# The identification engine (identify/) matches against a template catalog made from the
# price list plus one tarkov.dev base image per item (every item - ammo, guns, presets and
# containers included - because the engine identifies those too).  These helpers fetch the
# images into data/tmpl_src/; identify.catalog.load_catalog then turns them into
# data/identify_catalog_v2.npz.
# ---------------------------------------------------------------------------

_icon_session = None
def _get_icon_session():
    global _icon_session
    if _icon_session is None:
        s = http_requests.Session()
        adapter = http_requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
        s.mount('https://', adapter)
        s.mount('http://',  adapter)
        _icon_session = s
    return _icon_session


def base_image_path(item_id):
    return os.path.join(TMPL_SRC_DIR, f'{item_id}.png')


def download_base_image(item_id, url):
    """
    Download a tarkov.dev base image (transparent, per-slot resolution) and cache it as a
    BGRA PNG in data/tmpl_src/.  Returns the local path, or None when it could not be fetched.
    """
    path = base_image_path(item_id)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not url:
        return None
    try:
        r = _get_icon_session().get(url, timeout=12)
        if r.status_code != 200:
            return None
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)   # webp -> BGRA
        if img is None or img.size == 0:
            return None
        tmp = path + '.tmp.png'          # write-then-rename: a killed build never leaves a half file
        cv2.imwrite(tmp, img)
        os.replace(tmp, path)
        return path
    except Exception:
        return None


def download_missing_base_images(price_cache, progress_cb=None, workers=24):
    """Fetch the base image of every item that does not have one yet, in parallel.
    Returns (downloaded, failed).  progress_cb(done, total) counts only the missing ones."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [it for it in price_cache.get('items', [])
            if it.get('baseImageLink') and not os.path.exists(base_image_path(it['id']))]
    done = ok = 0
    if progress_cb:
        progress_cb(0, len(todo))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(lambda it: download_base_image(it['id'], it['baseImageLink']), todo):
            done += 1
            ok += bool(path)
            if progress_cb and done % 25 == 0:
                progress_cb(done, len(todo))
    if progress_cb:
        progress_cb(len(todo), len(todo))
    return ok, len(todo) - ok


def catalog_summary():
    """What the Sell Advisor's status line shows: whether the engine's catalog file exists and
    how many tarkov.dev items it covers (None when not built or unreadable)."""
    from identify.config import CATALOG_PATH
    if not os.path.exists(CATALOG_PATH):
        return None
    try:
        with np.load(CATALOG_PATH, allow_pickle=False) as z:     # lazy: only meta_json is read
            return {'items': int(json.loads(str(z['meta_json'])).get('n_api', 0))}
    except Exception as e:
        print(f'[catalog] unreadable: {e}')
        return None


# ---------------------------------------------------------------------------
# Identification: the identify/ package (grid -> segmentation -> template/DINO/OCR matching)
# ---------------------------------------------------------------------------

_v2_engine = [None, None]    # [settings key, Engine]


def scan_with_v2(img_bgr, settings, warnings, scene_out=None, in_raid=None, on_provisional=None, cancel=None):
    """
    Run the identification engine (identify/ package) and adapt its output to what the
    scan routes consume: raw detections as dicts (col,row,W,H,item_id,name,rotated,
    score 0-100,fir,panel,px,py,pw,ph — plus uncertain/count), the panel list as grid dicts
    (cell_w/cell_h/origin_x/origin_y/...), and whether the grid was found at all.
    """
    from identify.config import EngineSettings
    from identify.pipeline import Engine
    if not os.path.exists(PRICES_PATH) or next(os.scandir(TMPL_SRC_DIR), None) is None:
        # clean install: the catalog is built from the price list + tarkov.dev base images
        raise ScanError("The item database isn't built yet. Open the Sell Advisor page and click "
                        "'Build Icon DB' first (needs internet; takes a few minutes the first time).")
    es = EngineSettings.from_settings(settings)
    if tesseract_path_override():
        es.tesseract_cmd = tesseract_path_override()
    key = repr(es)
    if _v2_engine[0] != key:
        _v2_engine[:] = [key, Engine(es)]
    scene = None
    try:                                  # which grids are stash / own gear / loot / windows
        from identify.scene import analyze_scene
        scene = analyze_scene(img_bgr, in_raid=in_raid)
        if scene_out is not None:
            scene_out.update({'scene': scene.scene, 'in_raid': scene.in_raid,
                              'regions': [r.as_dict() for r in scene.regions]})
    except Exception as e:                # never lose the scan over the scene layer
        warnings.append(f'Scene analysis failed ({e}); every grid is treated as stash.')
    def adapt(res):
        panels = [{**p.as_grid_dict(), 'strength': p.strength} for p in res.grid.panels]
        records = []
        for d in res.detections:
            rec = d.to_record()
            reg = res.regions[d.panel] if d.panel < len(res.regions) else None
            if reg is not None:
                rec['role'], rec['side'], rec['region_title'] = reg.role, reg.side, reg.title
            if d.uncertain:
                rec['alternatives'] = [a['name'] for a in (d.evidence.get('alternatives') or ())][:3]
            records.append(rec)
        if scene_out is not None and res.regions:
            scene_out['grids'] = _region_grids(res)
        return records, panels

    cb = None
    if on_provisional is not None:
        def cb(res):                      # the fast first look: (records, panels), items marked provisional
            if res.grid.panels:
                records, panels = adapt(res)
                print(f"[v2] provisional {len(records)} items in {res.timings.get('provisional', 0):.2f}s")
                on_provisional(records, panels)
    res = _v2_engine[1].scan(img_bgr, scene, on_provisional=cb, cancel=cancel)
    warnings.extend(res.warnings)
    records, panels = adapt(res)
    if not panels:
        warnings.append('Stash grid not detected — check the capture region covers '
                        'the stash, or recalibrate.')
        return [], [{'cell_w': 64.0, 'cell_h': 64.0, 'origin_x': 0.0, 'origin_y': 0.0,
                     'x0': 0, 'y0': 0, 'x1': img_bgr.shape[1], 'y1': img_bgr.shape[0]}], True
    print(f"[v2] {len(res.detections)} items in {res.timings.get('total', 0):.2f}s "
          f"({', '.join(f'{k} {v:.2f}' for k, v in res.timings.items() if k != 'total')})")
    return records, panels, False


def _region_grids(res):
    """Per scanned panel: its role and every footprint in it (col, row, w, h, empty) - what the
    raid advice needs to find free room on the player."""
    out = []
    for pi, (panel, reg) in enumerate(zip(res.grid.panels, res.regions)):
        cells = [(it.col, it.row, it.w, it.h, bool(it.empty)) for it in res.items if it.panel == pi]
        out.append({'panel': pi, 'role': reg.role, 'cols': panel.n_cols, 'rows': panel.n_rows,
                    'clip_bottom': bool(panel.clip_bottom), 'cells': cells})
    return out


def tesseract_path_override():
    """pytesseract's command if app.py had to point it at the default Windows install."""
    cmd = pytesseract.pytesseract.tesseract_cmd
    return cmd if cmd and cmd != 'tesseract' else None


def is_unpriced_weapon(item_data, category=None):
    """True for a gun the sell list should skip.

    A gun's value depends on whatever parts are on it, so its identity can't
    price it.  tarkov.dev also tags signal flares (RSP-30) as 'gun', but those
    are fixed special-slot items with a real price, so they are kept.
    """
    types = set((item_data or {}).get('types') or ())
    if 'specialSlot' in types:
        return False
    if types & {'gun', 'preset'}:
        return True
    return item_data is None and category == 'weapon'


def is_dogtag(item_data):
    """True for a PMC dogtag.  Dozens of themed tags share the names "Dogtag BEAR" /
    "Dogtag USEC", so they can't be told apart or priced reliably; the sell list
    skips them.  The Dogtag case is a normal container and is kept."""
    name = (item_data or {}).get('name') or ''
    return name.startswith('Dogtag ') and 'container' not in ((item_data or {}).get('types') or ())


def skip_badge(item_data, category=None):
    """Badge for an item the sell list leaves out ('GUN', 'TAG'), or None to price it."""
    if is_unpriced_weapon(item_data, category):
        return 'GUN'
    if is_dogtag(item_data):
        return 'TAG'
    return None


def build_sell_context(settings, prices):
    """sellcalc context for one scan: settings + the live rules cached with the
    prices + the Intelligence Center level from the hideout progress."""
    intel = 0
    try:
        tasks = get_tasks(allow_fetch=False)
        if tasks:
            progress = load_json(PROGRESS_PATH, default_progress)
            intel = sellcalc.intel_center_level(tasks.get('hideoutStations'),
                                                progress.get('completed_hideout'))
    except Exception as e:
        print(f'[sell] intel center level unavailable: {e}')
    return sellcalc.make_context(settings, (prices or {}).get('rules'), intel)


# ---------------------------------------------------------------------------
# Default data
# ---------------------------------------------------------------------------

# F9 conflicts with EFT's own binds and popular overlay tools, so the default
# is a modifier combo nothing else claims.
DEFAULT_HOTKEY = '<ctrl>+<shift>+s'

# Shared label for the kappa keep-list category — also stamped unconditionally
# onto existing installs by merge_kappa_into_keep_list so they migrate on the
# next wiki sync.
KAPPA_LABEL = 'Collector (Kappa) — hand-ins must be Found In Raid'

def default_settings():
    return {
        'region': None, 'monitor': 0, 'hotkey': DEFAULT_HOTKEY, 'prestige': 3,
        'scan_countdown': 3,   # seconds before the manual Scan button captures (0 = instant)
        # Only kappaRequired tasks count toward the KEEP totals.  Off by default: since 1.0
        # tarkov.dev marks only the Collector's own prerequisite chain (~13 quests) Kappa-required,
        # so "on" would sell the items of almost every open quest.
        'kappa_only_tasks': False,
        'auto_task_progress': True,  # read completed/failed quests from EFT's logs (eftlogs.py)
        'eft_install_dir': None,     # None = find the install via the registry / common paths
        'game_mode': 'auto',         # 'auto' (profile of the latest session) | 'pvp' | 'pve' | 'season'
        'faction': 'auto',           # 'auto' (from faction-only quests in the logs) | 'BEAR' | 'USEC'
        'debug_dumps': True,  # save raw frame + detections of the last few scans under data/debug/
        # Sell advice (see sellcalc.SELL_DEFAULTS for what each means; all are optional in settings.json)
        'flea_requires_fir': None,  # None = follow tarkov.dev's flea rule (live: FiR only); set false when Battlestate lifts it for an event
        'flea_offer_slots': sellcalc.DEFAULT_FLEA_SLOTS,  # simultaneous flea offers (your flea rating decides; shown as 'Offers x/y' in the flea tab)
        'flea_overflow': 'queue',   # flea picks beyond the slots: 'queue' (list when a slot frees) or 'trader' (sell now)
        'flea_min_gain': sellcalc.FLEA_MIN_GAIN,  # roubles an offer must beat the trader by to be worth a slot
        'intel_center_level': None,  # None = read from Tasks & Hideout progress; level 3 = -30% flea fee
        'hideout_management_level': 0,  # skill level 0-51; each level adds 0.3% flea fee discount (with Intel Center 3)
        'skip_traders': ['Ref'],   # Ref pays GP coins, not roubles
        'trader_levels': {},       # e.g. {'Ref': 4} - only Ref's pay rate changes with loyalty level
        'auto_scan': True,  # watch the game passively and scan the stash when it settles (autoscan/)
        'ignore_task_items': False,     # never hold items back for tasks / Kappa: flea vs trader only
        'ignore_hideout_items': False,  # never hold items back for hideout upgrades
        # Open and close with the game (lifecycle.py); changes take effect when settings are saved
        'start_with_windows': True,  # register the tiny watcher at sign-in + keep it running (False: unregister, stop it)
        'follow_tarkov': True,       # open when Tarkov starts, close ~15 s after it exits (False: the watcher opens the app once at sign-in and it stays)
        'show_window_on_game_start': True,  # False: when Tarkov starts the app opens straight to the tray, no window
        'auto_scan_in_raid': False,  # also scan the in-raid inventory (off: only the lobby stash)
        # Live page (identify inventory grids on the fly; the live viewer owns what these do)
        'live_viewer': True,               # identify inventory grids on the fly and show them on the Live page
        'live_in_raid': True,              # also work in raid: what is worth grabbing / what to drop
        'live_top_n': 5,                   # how many 'worth grabbing' items to highlight (1-20)
        'live_min_value_per_slot': 10000,  # roubles: ignore loot worth less than this per slot
    }

def default_keep_list():
    return {
        'categories': [
            {
                'id': 'kappa',
                'label': KAPPA_LABEL,
                'items': [
                    {'id': 'tea',         'name': '42 Signature Blend English Tea', 'aliases': ['42 Sig', 'English Tea'],             'acquired': False},
                    {'id': 'axe',         'name': 'Antique axe',                    'aliases': ['Antique axe'],                       'acquired': False},
                    {'id': 'armband',     'name': 'Armband (Evasion)',               'aliases': ['Armband', 'Evasion'],                'acquired': False},
                    {'id': 'bear_buddy',  'name': 'BEAR Buddy plush toy',            'aliases': ['BEAR Buddy'],                        'acquired': False},
                    {'id': 'drd',         'name': 'DRD body armor',                  'aliases': ['DRD'],                               'acquired': False},
                    {'id': 'phone',       'name': 'Golden 1GPhone smartphone',        'aliases': ['1GPhone', 'Golden phone'],           'acquired': False},
                    {'id': 'loot_lord',   'name': 'Loot Lord plushie',               'aliases': ['Loot Lord'],                         'acquired': False},
                    {'id': 'wz_wallet',   'name': 'WZ Wallet',                        'aliases': ['WZ Wallet'],                         'acquired': False},
                    {'id': 'dumbbell',    'name': 'Mazoni golden dumbbell',           'aliases': ['Mazoni', 'Dumbbell'],                'acquired': False},
                    {'id': 'splint',      'name': 'Tigzresq splint',                  'aliases': ['Tigzresq', 'Splint'],                'acquired': False},
                    {'id': 'firesteel',   'name': 'Old firesteel',                    'aliases': ['Firesteel'],                         'acquired': False},
                    {'id': 'book',        'name': 'Battered antique book',            'aliases': ['Book', 'Battered book'],             'acquired': False},
                    {'id': 'fireklean',   'name': '#FireKlean gun lube',              'aliases': ['FireKlea', 'FireKlean'],             'acquired': False},
                    {'id': 'rooster',     'name': 'Golden rooster figurine',          'aliases': ['Rooster'],                           'acquired': False},
                    {'id': 'badge',       'name': 'Silver Badge',                     'aliases': ['Badge'],                             'acquired': False},
                    {'id': 'beard_oil',   'name': "Deadlyslob's beard oil",           'aliases': ['BeardOil', 'Beard Oil'],             'acquired': False},
                    {'id': 'mayo',        'name': 'Jar of DevilDog mayo',             'aliases': ['Mayo', 'DevilDog'],                  'acquired': False},
                    {'id': 'sprats',      'name': 'Can of sprats',                    'aliases': ['Sprats'],                            'acquired': False},
                    {'id': 'mustache',    'name': 'Fake mustache',                    'aliases': ['Mustache'],                          'acquired': False},
                    {'id': 'kotton',      'name': 'Kotton beanie',                    'aliases': ['Kotton'],                            'acquired': False},
                    {'id': 'raven',       'name': 'Raven figurine',                   'aliases': ['Raven'],                             'acquired': False},
                    {'id': 'pestily',     'name': 'Pestily plague mask',              'aliases': ['Pestily'],                           'acquired': False},
                    {'id': 'shroud',      'name': 'Shroud half-mask',                 'aliases': ['Shroud'],                            'acquired': False},
                    {'id': 'drlupo',      'name': "Can of Dr. Lupo's coffee beans",   'aliases': ["DrLupo's", 'Dr Lupo'],               'acquired': False},
                    {'id': 'veritas',     'name': 'Veritas guitar pick',              'aliases': ['Veritas'],                           'acquired': False},
                    {'id': 'ratcola',     'name': 'Can of RatCola soda',              'aliases': ['RatCola'],                           'acquired': False},
                    {'id': 'smoke',       'name': 'Smoke balaclava',                  'aliases': ['Smoke'],                             'acquired': False},
                    {'id': 'lvndmark',    'name': "LVNDMARK's rat poison",            'aliases': ['LVNDMARK', 'Polson', 'Rat poison'],  'acquired': False},
                    {'id': 'forklift',    'name': 'Missam forklift key',              'aliases': ['Missam'],                            'acquired': False},
                    {'id': 'vhs',         'name': 'Video cassette (Cyborg Killer)',   'aliases': ['VHS', 'Cyborg Killer'],              'acquired': False},
                    {'id': 'bakeezy',     'name': 'BakeEzy cook book',                'aliases': ['BakeEzy'],                           'acquired': False},
                    {'id': 'johnb',       'name': 'JohnB Liquid DNB glasses',         'aliases': ['JohnB'],                             'acquired': False},
                    {'id': 'baddie',      'name': "Baddie's red beard",               'aliases': ['Baddie'],                            'acquired': False},
                    {'id': 'gingy',       'name': 'Gingy keychain',                   'aliases': ['Gingy'],                             'acquired': False},
                    {'id': 'egg',         'name': 'Golden egg',                        'aliases': ['Egg'],                               'acquired': False},
                    {'id': 'pass_',       'name': 'Press pass (NoiceGuy)',             'aliases': ['Pass', 'NoiceGuy'],                  'acquired': False},
                    {'id': 'axel',        'name': 'Axel parrot figurine',              'aliases': ['Axel'],                              'acquired': False},
                    {'id': 'glorious',    'name': 'Glorious E armored mask',           'aliases': ['Glorious'],                          'acquired': False},
                    {'id': 'inseq',       'name': 'Inseq gas pipe wrench',             'aliases': ['Inseq'],                             'acquired': False},
                    {'id': 'viibiin',     'name': 'Viibiin sneaker',                   'aliases': ['Viiblin', 'Viibiin'],                'acquired': False},
                    {'id': 'tamatthi',    'name': 'Tamatthi kunai knife replica',      'aliases': ['Tamatthi'],                          'acquired': False},
                    {'id': 'nut_sack',    'name': 'Nut Sack balaclava',                'aliases': ['Nut Sack', 'NutSack'],               'acquired': False},
                    {'id': 'domontovich', 'name': 'Domontovich ushanka hat',           'aliases': ['Domontovich'],                       'acquired': False},
                ]
            },
            {
                'id': 'tasks',
                'label': 'Task Items (manual)',
                'items': []
            }
        ]
    }


# ---------------------------------------------------------------------------
# Kappa (Collector) list — synced from the wiki
# ---------------------------------------------------------------------------

from html.parser import HTMLParser

class _WikiTableParser(HTMLParser):
    """Collect each top-level <table> as a list of rows of stripped cell text.
    Nested tables are parsed but discarded so their text can't pollute cells."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._stack = []   # one dict per open <table>

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._stack.append({'rows': [], 'row': None, 'cell': None})
        elif self._stack:
            t = self._stack[-1]
            if tag == 'tr':
                t['row'] = []
                t['cell'] = None
            elif tag in ('td', 'th') and t['row'] is not None:
                t['cell'] = []
            elif tag == 'br' and t['cell'] is not None:
                t['cell'].append(' ')

    def handle_endtag(self, tag):
        if not self._stack:
            return
        t = self._stack[-1]
        if tag == 'table':
            done = self._stack.pop()
            if done['cell'] is not None and done['row'] is not None:
                done['row'].append(' '.join(''.join(done['cell']).split()))
            if done['row']:
                done['rows'].append(done['row'])
            if not self._stack:          # nested tables are dropped
                self.tables.append(done['rows'])
        elif tag == 'tr' and t['row'] is not None:
            if t['cell'] is not None:
                t['row'].append(' '.join(''.join(t['cell']).split()))
                t['cell'] = None
            t['rows'].append(t['row'])
            t['row'] = None
        elif tag in ('td', 'th') and t['cell'] is not None:
            t['row'].append(' '.join(''.join(t['cell']).split()))
            t['cell'] = None

    def handle_data(self, data):
        if self._stack and self._stack[-1]['cell'] is not None:
            self._stack[-1]['cell'].append(data)


def _extract_kappa_names(html):
    """Pull the "Item name" column out of the Collector page's item table.
    Locates the column by header text (not position) and right-aligns data
    rows against the header row, since the wiki uses rowspan/colspan cells."""
    p = _WikiTableParser()
    p.feed(html)
    for table in p.tables:
        header_i = col = None
        for i, row in enumerate(table):
            low = [c.strip().lower() for c in row]
            if 'item name' in low:
                header_i, col = i, low.index('item name')
                break
        if header_i is None:
            continue
        header_len = len(table[header_i])
        names = []
        for row in table[header_i + 1:]:
            if len(row) < 2:
                continue
            j = col + (len(row) - header_len)   # right-edge alignment
            if 0 <= j < len(row):
                name = row[j].strip()
                if name:
                    names.append(name)
        if len(names) >= 20:
            return names
    raise ValueError('Collector item table not found (or suspiciously small) — '
                     'wiki layout may have changed')


def fetch_kappa_names(force=False):
    """Return the current Collector item names from the wiki, cached 24 h."""
    if not force and os.path.exists(KAPPA_WIKI_PATH):
        cache = load_json(KAPPA_WIKI_PATH, lambda: None)
        if cache and time.time() - cache.get('timestamp', 0) < KAPPA_WIKI_TTL:
            return cache['names']
    r = http_requests.get(COLLECTOR_WIKI_API, timeout=30, headers={
        'User-Agent': 'TarkovStashHelper/1.0 (github.com/josfire18/tarkov-stash-helper)',
    })
    r.raise_for_status()
    names = _extract_kappa_names(r.json()['parse']['text'])
    save_json(KAPPA_WIKI_PATH, {'timestamp': time.time(), 'names': names})
    return names


# ---------------------------------------------------------------------------
# Prestige requirements — synced from the wiki
# ---------------------------------------------------------------------------

def _parse_prestige_html(html):
    """
    Parse the wiki Prestige page's requirements table into a list of level
    dicts. Factored out of fetch_prestige_requirements so tests can exercise
    the parsing logic on a synthetic HTML fixture without hitting the network.

    Locates the table whose header row contains a cell that casefold-contains
    'prestige level' (not by position — some wiki tables have merged/leading
    cells), then maps each data row's cells to that same row's headers BY
    HEADER TEXT, so a reordered column can't silently scramble fields.

    Cell values (quests/objectives/skills/hideout/items) are kept as display
    strings — the items column mixes rouble figures and item lists, which
    isn't worth structuring further here.

    Raises ValueError if fewer than 5 data rows parse out (layout-drift guard).
    """
    p = _WikiTableParser()
    p.feed(html)

    def _get(rowmap, key_substr):
        for h, v in rowmap.items():
            if key_substr in h.casefold():
                return v
        return ''

    for table in p.tables:
        header_i = None
        for i, row in enumerate(table):
            if any('prestige level' in c.strip().casefold() for c in row):
                header_i = i
                break
        if header_i is None:
            continue
        headers = [c.strip() for c in table[header_i]]
        levels = []
        for row in table[header_i + 1:]:
            if not row or not any(c.strip() for c in row):
                continue
            rowmap = {headers[j]: row[j].strip()
                     for j in range(min(len(headers), len(row)))}
            level_txt = _get(rowmap, 'prestige level')
            m = re.search(r'\d+', level_txt)
            if not m:
                continue
            levels.append({
                'level':      int(m.group()),
                'pmc_level':  _get(rowmap, 'pmc level'),
                'quests':     _get(rowmap, 'quests completed') or _get(rowmap, 'quest'),
                'objectives': _get(rowmap, 'objectives'),
                'skills':     _get(rowmap, 'skills'),
                'hideout':    _get(rowmap, 'hideout'),
                'items':      _get(rowmap, 'items'),
            })
        if len(levels) >= 5:
            return levels
    raise ValueError('Prestige requirements table not found (or suspiciously small) — '
                     'wiki layout may have changed')


def fetch_prestige_requirements(force=False):
    """
    Return the current Prestige requirements from the wiki, cached 7 days.
    Mirrors fetch_kappa_names's caching shape. The cache file is NEVER
    overwritten on a failed fetch/parse — on failure, the stale cache (if any)
    is served instead; only when there's no cache at all does this raise.
    """
    if not force and os.path.exists(PRESTIGE_WIKI_PATH):
        cache = load_json(PRESTIGE_WIKI_PATH, lambda: None)
        if cache and time.time() - cache.get('timestamp', 0) < PRESTIGE_WIKI_TTL:
            return cache['levels']
    try:
        r = http_requests.get(PRESTIGE_WIKI_API, timeout=30, headers={
            'User-Agent': 'TarkovStashHelper/1.0 (github.com/josfire18/tarkov-stash-helper)',
        })
        r.raise_for_status()
        levels = _parse_prestige_html(r.json()['parse']['text'])
        save_json(PRESTIGE_WIKI_PATH, {'timestamp': time.time(), 'levels': levels})
        return levels
    except Exception:
        if os.path.exists(PRESTIGE_WIKI_PATH):
            cache = load_json(PRESTIGE_WIKI_PATH, lambda: None)
            if cache and cache.get('levels'):
                return cache['levels']
        raise


def merge_kappa_into_keep_list(keep_list, names, price_idx=None):
    """
    Merge fresh wiki names into the kappa category, preserving user state.
      - name already present (exact or fuzzy ≥95) → keep the entry (acquired,
        aliases, id survive); fuzzy hits are renamed to the wiki spelling with
        the old name kept as an alias.
      - new wiki name → appended unchecked.
      - wiki-sourced entry (source != 'custom') no longer on the wiki → REMOVED
        from the category outright (previously just flagged 'stale' and kept —
        that left permanently-dead rows accumulating on every sync). User-added
        ('custom') entries are never removed by this pass, matched or not.
      - deletion is only ever reached with a wiki `names` list that already
        cleared fetch_kappa_names's own >=20-name floor, and kappa_sync (the
        only caller in the fetch→merge→save chain) aborts before this runs at
        all if the fetch/parse failed — so a bad/short scrape can't wipe out
        the category.
    Idempotent. Returns {'added', 'removed', 'total'}.
    """
    cat = next((c for c in keep_list['categories'] if c['id'] == 'kappa'), None)
    if cat is None:
        cat = {'id': 'kappa', 'label': KAPPA_LABEL, 'items': []}
        keep_list['categories'].insert(0, cat)
    cat['label'] = KAPPA_LABEL   # migrate existing installs' category label too

    by_name = {it['name'].casefold(): it for it in cat['items']}
    claimed = set()   # entry ids already matched to a wiki name
    added = []

    for n in names:
        entry = by_name.get(n.casefold())
        if entry is None:
            # Fuzzy rescue for wiki renames ('Press pass (NoiceGuy)' →
            # 'Press pass (issued for NoiceGuy)') so check-state survives.
            # Token-set + punctuation stripping scores real renames at 100
            # while unrelated kappa items stay under ~45.
            from rapidfuzz import fuzz as _fuzz, utils as _futils
            cands = {it['name']: it for it in cat['items']
                     if it['id'] not in claimed and it.get('source') != 'custom'}
            if cands:
                hit = rfuzz.extractOne(n, list(cands.keys()),
                                       scorer=_fuzz.token_set_ratio,
                                       processor=_futils.default_process,
                                       score_cutoff=95)
                if hit:
                    entry = cands[hit[0]]
                    old = entry['name']
                    if old.casefold() != n.casefold():
                        entry['name'] = n
                        if old not in entry.get('aliases', []):
                            entry.setdefault('aliases', []).append(old)
        if entry is not None:
            claimed.add(entry['id'])
            entry['source'] = 'wiki'
        else:
            new = {'id': str(uuid.uuid4())[:8], 'name': n, 'aliases': [],
                   'acquired': False, 'source': 'wiki'}
            cat['items'].append(new)
            claimed.add(new['id'])
            added.append(n)

    removed = []
    survivors = []
    for it in cat['items']:
        it.pop('stale', None)   # legacy flag from the old flag-not-delete behaviour
        if it['id'] not in claimed and it.get('source') != 'custom':
            removed.append(it['name'])
            continue
        survivors.append(it)
    cat['items'] = survivors

    # Persist tarkov.dev ids so scans can map entries without alias guessing.
    if price_idx:
        keys = list(price_idx.keys())
        for it in cat['items']:
            if it.get('tdev_id'):
                continue
            data = price_idx.get(it['name'].lower())
            if data is None:
                hit = rfuzz.extractOne(it['name'].lower(), keys, score_cutoff=93)
                if hit:
                    data = price_idx[hit[0]]
            if data:
                it['tdev_id'] = data['id']

    return {'added': added, 'removed': removed, 'total': len(cat['items'])}


def kappa_sync(force=False):
    """Fetch + merge + save. Returns the merge summary. Raises on failure
    (keep_list.json is never touched when the fetch/parse fails — merge only
    ever runs after fetch_kappa_names succeeds). Wiki-sourced entries no
    longer on the wiki are removed (not just flagged) from the kappa
    category; user-added ('custom') entries are never removed."""
    names = fetch_kappa_names(force=force)
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    price_idx = None
    try:
        price_idx = build_price_index(get_prices())
    except Exception as e:
        print(f"[kappa] price index unavailable during sync: {e}")
    summary = merge_kappa_into_keep_list(keep_list, names, price_idx)
    save_json(KEEPLIST_PATH, keep_list)
    print(f"[kappa] synced {len(names)} wiki items — "
          f"{len(summary['added'])} added, {len(summary['removed'])} removed"
          + (f" ({', '.join(summary['removed'])})" if summary['removed'] else ""))
    return summary


# ---------------------------------------------------------------------------
# Task & hideout requirements — from tarkov.dev
# ---------------------------------------------------------------------------

def default_progress():
    return {'completed_tasks': [], 'completed_hideout': [], 'have': {}}


# ---------------------------------------------------------------------------
# Automatic task progress from EFT's own logs (eftlogs.py)
# ---------------------------------------------------------------------------

import eftlogs  # noqa: E402

EFTLOGS_CACHE_PATH = os.path.join(DATA, 'eftlogs_cache.json')
_log_scanner = eftlogs.LogScanner(EFTLOGS_CACHE_PATH)
_log_progress = {'key': None, 'result': None, 'error': None, 'scanned': False}
_log_progress_lock = threading.Lock()


def scan_task_logs(full=False):
    """Bring the log records up to date (full: re-check every folder; otherwise only new folders
    and the live one).  Never raises; the error is kept for the status endpoint."""
    settings = load_json(SETTINGS_PATH, default_settings)
    if not settings.get('auto_task_progress', True):
        return None
    try:
        inst = eftlogs.find_install_dir(settings.get('eft_install_dir') or None)
        if not inst:
            raise FileNotFoundError('Escape from Tarkov install (with a Logs folder) not found')
        res = _log_scanner.scan(inst, full=full or not _log_progress['scanned'])
        _log_progress['scanned'] = True
        _log_progress['error'] = None
        return res
    except Exception as e:
        _log_progress['error'] = str(e)
        print(f'[eftlogs] scan failed: {e}')
        return None


def log_task_progress(cache=None):
    """The quest progress read from the logs for the configured mode/faction, or None when the
    feature is off or nothing could be read.  Cached until the logs or the task data change."""
    settings = load_json(SETTINGS_PATH, default_settings)
    if not settings.get('auto_task_progress', True):
        return None
    if not _log_progress['scanned']:
        scan_task_logs(full=True)
    if cache is None:
        cache = get_tasks(allow_fetch=False)
    tasks = (cache or {}).get('tasks') or None
    key = (_log_scanner.last_scan_at, (cache or {}).get('timestamp'),
           settings.get('game_mode', 'auto'), settings.get('faction', 'auto'))
    with _log_progress_lock:
        if _log_progress['key'] == key and _log_progress['result'] is not None:
            return _log_progress['result']
        try:
            res = _log_scanner.progress(settings.get('game_mode', 'auto'), tasks,
                                        settings.get('faction', 'auto'))
        except Exception as e:
            _log_progress['error'] = str(e)
            return None
        if not res.get('folders'):
            return None
        _log_progress.update(key=key, result=res)
        return res


def effective_progress(cache=None):
    """progress.json with the quest completion the logs prove merged in: completed, failed (a
    failed quest needs nothing any more) and the other faction's quests count as done.  The
    player's clicks on the Tasks page win: ``manual_overrides[task] = 'done'|'open'``.
    Adds ``auto_done`` (ids done because of the logs) and ``log_status``."""
    progress = load_json(PROGRESS_PATH, default_progress)
    manual = set(progress.get('completed_tasks', []))
    overrides = progress.get('manual_overrides') or {}
    lp = log_task_progress(cache)
    auto = set()
    if lp:
        auto = set(lp['completed']) | set(lp['failed']) | set(lp.get('other_faction') or ())
    forced_open = {t for t, v in overrides.items() if v == 'open'}
    forced_done = {t for t, v in overrides.items() if v == 'done'}
    done = ((manual | auto) - forced_open) | forced_done
    out = dict(progress)
    out['completed_tasks'] = sorted(done)
    out['auto_done'] = sorted((auto & done) - forced_done)
    out['log_status'] = None if not lp else {
        'mode': lp['mode'], 'completed': len(lp['completed']), 'failed': len(lp['failed']),
        'active': len(lp['active']), 'faction': lp.get('faction'),
        'last_event_at': lp['last_event_at'], 'reset_at': lp['reset_at'],
        'reset_kind': lp['reset_kind'], 'folders': lp.get('folders')}
    return out


def _graphql_tasks():
    """Task + hideout item requirements from tarkov.dev's GraphQL API (the fallback source) as
    a cache dict.  Not written anywhere; raises RuntimeError."""
    r = http_requests.post(TARKOV_API, json={'query': TASKS_QUERY}, timeout=60)
    try:
        payload = r.json()
    except ValueError:
        raise RuntimeError(f'tarkov.dev returned HTTP {r.status_code} with no JSON')
    if payload.get('errors'):
        raise RuntimeError(f"tarkov.dev tasks query failed: {payload['errors']}")
    data = payload.get('data') or {}
    return {
        'timestamp':       time.time(),
        'source':          'graphql',
        'schema':          tarkovdata.TASKS_SCHEMA,
        'tasks':           data.get('tasks') or [],
        'hideoutStations': data.get('hideoutStations') or [],
    }


def fetch_tasks_graphql(previous=None):
    """Refresh the tasks cache from the GraphQL API alone; validated, written atomically."""
    cache = _graphql_tasks()
    previous = previous if previous is not None else _load_cache(TASKS_CACHE_PATH)
    try:
        tarkovdata.validate_tasks(cache['tasks'], (previous or {}).get('tasks'))
        tarkovdata.validate_stations(cache['hideoutStations'], (previous or {}).get('hideoutStations'))
    except tarkovdata.SourceError as e:
        raise RuntimeError(f'tarkov.dev GraphQL task data refused: {e}')
    tarkovdata.write_json_atomic(TASKS_CACHE_PATH, cache, indent=1)
    return cache


def refresh_tasks(force=False):
    """Bring the tasks/hideout cache up to date: json.tarkov.dev first, then GraphQL; a bad
    payload never replaces the cache on disk.  Returns ``{'status', 'source', 'cache'}``."""
    with _refresh_lock:
        previous = _load_cache(TASKS_CACHE_PATH)
        if previous and not (previous.get('tasks') and previous.get('hideoutStations')):
            previous = None
        meta = _load_meta()
        _refresh_state['tasks']['attempt'] = meta['attempted']['tasks'] = time.time()
        if force:
            meta['etags'] = {k: v for k, v in meta['etags'].items() if k not in
                             ('tasks', 'tasks_en', 'hideout', 'hideout_en')}
        names = tarkovdata.item_names_from_prices(_load_cache(PRICES_PATH))
        try:
            try:
                res = tarkovdata.refresh_tasks(TASKS_CACHE_PATH, meta, previous, names)
                res['source'] = tarkovdata.SOURCE_NAME
            except Exception as json_err:      # network, bad payload, a converter bug: all fall back
                print(f'[tasks] json.tarkov.dev failed ({json_err}); trying the GraphQL API')
                try:
                    cache = fetch_tasks_graphql(previous)
                except (RuntimeError, http_requests.RequestException) as gql_err:
                    raise RuntimeError(f'json.tarkov.dev: {json_err}; GraphQL: {gql_err}')
                meta['checked']['tasks'] = cache['timestamp']
                meta['sources']['tasks'] = 'graphql'
                for k in ('tasks', 'tasks_en', 'hideout', 'hideout_en'):
                    meta['etags'].pop(k, None)
                res = {'status': 'updated', 'source': 'graphql', 'cache': cache}
        except RuntimeError as e:
            meta['errors']['tasks'] = str(e)
            _refresh_state['tasks']['error'] = str(e)
            _save_meta_quietly(meta)
            raise
        meta['errors'].pop('tasks', None)
        _refresh_state['tasks']['error'] = None
        _save_meta_quietly(meta)
        return res


def fetch_tasks():
    """Refresh the task + hideout item requirements now (see :func:`refresh_tasks`)."""
    return refresh_tasks()['cache']


def get_tasks(allow_fetch=True):
    """Cached task data, refreshed when stale (the background refresher normally keeps it
    current; a failed refresh falls back to the cache on disk). With allow_fetch=False,
    returns whatever cache exists (or None) without touching the network - used inside
    scans so a scan can never block on tarkov.dev."""
    cache = _load_cache(TASKS_CACHE_PATH)
    if cache and not (cache.get('tasks') and cache.get('hideoutStations')):
        cache = None
    if cache and (not allow_fetch or _cache_age_seconds(cache, 'tasks') < TASKS_CACHE_TTL):
        return cache
    if not allow_fetch:
        return None
    err = _refresh_state['tasks']['error']
    if cache and err and time.time() - _refresh_state['tasks']['attempt'] < RETRY_BACKOFF:
        return cache
    try:
        return fetch_tasks()
    except (RuntimeError, http_requests.RequestException) as e:
        if not cache:
            raise
        print(f'[tasks] refresh failed, using cached tasks: {e}')
        return cache


# Money hand-ins (e.g. "Compensation for Damage" wants 1M roubles) aren't
# stash items worth tracking — they'd bury real requirements in the aggregate.
CURRENCY_NAMES = {'roubles', 'dollars', 'euros'}

def compute_tasks_view(cache, progress, kappa_only=False, kinds=('giveItem',), split_any_of=False):
    """
    Server-side merged view of what the player still needs.
    Only 'giveItem' objectives count as hand-ins (the Tasks page); the sell
    advisor also passes 'plantItem', whose items are consumed in raid and so
    must not be sold.  Objectives with a null item are skipped (tarkov.dev is
    migrating TaskObjectiveItem.item → items).

    `kappa_only` restricts which TASK objectives contribute to the aggregate
    totals to those on tasks with `kappaRequired` — non-kappa tasks (e.g.
    "Compensation for Damage") still appear in `tasks_out` (the client dims
    them) but no longer inflate the aggregate/KEEP totals. Hideout
    accumulation is untouched regardless, since it's needed for prestige.

    An objective that accepts ANY ONE of several items (tarkov.dev's ``items`` list)
    shows on the Tasks page under its first item, with the rest in ``alternatives``.
    With ``split_any_of`` (the sell advisor) it stays out of ``aggregate`` and is
    returned in ``any_of`` instead - ``{'label', 'count', 'fir', 'items': [{id, name}]}``
    per still-open objective - because each item's own total would be wrong for it.
    """
    done_tasks  = set(progress.get('completed_tasks', []))
    done_levels = set(progress.get('completed_hideout', []))
    have        = progress.get('have', {})

    agg = {}
    def _acc(item, count, fir, src_type, src_name, active):
        rec = agg.setdefault(item['id'], {
            'item_id': item['id'], 'name': item.get('name') or '?',
            'shortName': item.get('shortName') or '',
            'total_needed': 0, 'fir_needed': 0, 'sources': [],
        })
        if active:
            rec['total_needed'] += count
            if fir:
                rec['fir_needed'] += count
            rec['sources'].append({'type': src_type, 'name': src_name,
                                   'count': count, 'fir': bool(fir)})

    any_of = []
    tasks_out = []
    for t in cache.get('tasks', []):
        items = []
        for o in (t.get('objectives') or []):
            if o.get('type') not in kinds:
                continue
            alts = [a for a in tarkovdata.objective_items(o)
                    if (a.get('name') or '').lower() not in CURRENCY_NAMES]
            cnt = o.get('count') or 0
            if not alts or cnt <= 0:
                continue
            it = alts[0]
            fir = bool(o.get('foundInRaid'))
            rec = {'item_id': it['id'], 'name': it.get('name') or '?', 'count': cnt, 'fir': fir}
            if len(alts) > 1:
                rec['alternatives'] = [a.get('name') or '?' for a in alts]
            items.append(rec)
            active = (t['id'] not in done_tasks
                      and (not kappa_only or bool(t.get('kappaRequired'))))
            label = f"{(t.get('trader') or {}).get('name', '?')} — {t['name']}"
            if len(alts) > 1 and split_any_of:
                if active:
                    any_of.append({'label': label, 'count': cnt, 'fir': fir,
                                   'items': [{'id': a['id'], 'name': a.get('name') or '?'} for a in alts]})
                continue
            _acc(it, cnt, fir, 'task', label, active=active)
        if not items:
            continue   # only hand-in tasks are interesting here
        tasks_out.append({
            'id': t['id'], 'name': t['name'],
            'trader': (t.get('trader') or {}).get('name', '?'),
            'minPlayerLevel': t.get('minPlayerLevel') or 0,
            'kappaRequired': bool(t.get('kappaRequired')),
            'completed': t['id'] in done_tasks,
            'items': items,
        })

    stations_out = []
    for s in cache.get('hideoutStations', []):
        levels = []
        for lv in (s.get('levels') or []):
            items = []
            for req in (lv.get('itemRequirements') or []):
                it, cnt = req.get('item'), req.get('count') or 0
                if not it or not it.get('id') or cnt <= 0:
                    continue
                if (it.get('name') or '').lower() in CURRENCY_NAMES:
                    continue
                items.append({'item_id': it['id'], 'name': it.get('name') or '?',
                              'count': cnt})
                _acc(it, cnt, False, 'hideout', f"{s['name']} L{lv['level']}",
                     active=lv['id'] not in done_levels)
            levels.append({'id': lv['id'], 'level': lv['level'],
                           'completed': lv['id'] in done_levels, 'items': items})
        if levels:
            stations_out.append({'id': s['id'], 'name': s['name'], 'levels': levels})

    aggregate = []
    for rec in agg.values():
        if rec['total_needed'] <= 0:
            continue
        rec['have'] = int(have.get(rec['item_id'], 0))
        aggregate.append(rec)
    aggregate.sort(key=lambda r: -(r['total_needed'] - min(r['have'], r['total_needed'])))

    return {
        'aggregate': aggregate,
        'any_of':    any_of,
        'tasks':     tasks_out,
        'stations':  stations_out,
        'cache_age_minutes': round((time.time() - cache.get('timestamp', 0)) / 60, 1),
        'kappa_only': bool(kappa_only),
    }


def get_protected_ids(keep_list, price_idx):
    """The item-keyed half of :func:`get_protected_plan` (the single-item needs)."""
    return get_protected_plan(keep_list, price_idx)[0]


def get_protected_plan(keep_list, price_idx):
    """
    ``(protected, any_of)``.  ``protected`` is
    {tarkov.dev item id: {'reason', 'fir_only', 'need', 'fir_need', 'why'}} for
    everything the player should NOT unconditionally sell: unacquired keep-list
    entries + task/hideout items still short of their required count.  Task
    data is cache-only here — a scan never waits on the network for it.

    ``need`` is how many copies are still short (the scan keeps that many and
    sells the surplus - see sellcalc.allocate_keep), ``fir_need`` how many of
    those must be Found-in-Raid, ``why`` one string per source for the KEEP row.
    Copies on the Tasks page's ``have`` count are treated as already secured
    elsewhere and subtracted.  ``fir_only`` stays the summary the scan used
    before: True when no remaining need accepts a non-FiR copy, so a copy
    confidently read as non-FiR is safe to sell:
      - keep-list KAPPA (Collector) entries — Collector hand-ins require FiR.
      - keep-list manual/task entries — not FiR-gated (fir_only=False), those
        categories accept any copy.
      - task/hideout items — fir_only only when EVERY remaining need across
        all sources is FiR-specific; if even one source accepts non-FiR, a
        non-FiR copy is still needed.
    Task hand-ins and plant-item objectives both count (planted items are
    consumed); completed tasks/hideout levels are excluded by the progress data.

    ``any_of`` lists the open objectives that accept ANY ONE of several items:
    ``{'label', 'count', 'fir', 'items': [{id, name}]}``.  Which copies to keep for them
    depends on what the scan found, so sellcalc.plan_entries allocates them after the
    single-item needs.  ``count`` is already net of ``have`` copies of those items that
    no single-item need claims.
    """
    protected, any_of = {}, []
    settings = load_json(SETTINGS_PATH, default_settings)
    # "Pure money" switches (Settings page): skip whole need sources so every item is routed
    # to the flea or a trader.  Task items include the Collector (Kappa) and manual task lists.
    ignore_tasks = bool(settings.get('ignore_task_items'))
    ignore_hideout = bool(settings.get('ignore_hideout_items'))
    entry_cat = {e['id']: cat
                 for cat in keep_list['categories'] for e in cat['items']}
    mapped, _unmapped = map_keep_entries_to_ids(keep_list, price_idx)
    for tid, entry in mapped.items():
        if not entry.get('acquired'):
            cat = entry_cat.get(entry['id']) or {}
            if ignore_tasks and cat.get('id') in ('kappa', 'tasks'):
                continue
            fir_only = cat.get('id') == 'kappa'
            protected[tid] = {
                'reason': 'On keep list', 'fir_only': fir_only, 'kinds': ['list'],
                'need': 1, 'fir_need': 1 if fir_only else 0,
                'why': [f"{cat.get('label') or 'Keep list'}: 1 needed"
                        + (' (Found in Raid)' if fir_only else '')],
            }
    try:
        cache = get_tasks(allow_fetch=False)
        if cache and (ignore_tasks or ignore_hideout):
            cache = {**cache,
                     'tasks': [] if ignore_tasks else cache.get('tasks', []),
                     'hideoutStations': [] if ignore_hideout else cache.get('hideoutStations', [])}
        if cache:
            progress = effective_progress(cache)
            view = compute_tasks_view(cache, progress,
                                      kappa_only=settings.get('kappa_only_tasks', False),
                                      kinds=('giveItem', 'plantItem'), split_any_of=True)
            spare = {rec['item_id']: max(0, rec['have'] - rec['total_needed'])
                     for rec in view['aggregate']}
            for g in view['any_of']:
                left = g['count']
                for a in g['items']:                       # set-aside copies cover it first
                    i = a['id']
                    avail = spare.get(i, int(progress.get('have', {}).get(i, 0)))
                    use = min(avail, left)
                    spare[i] = avail - use
                    left -= use
                if left:
                    any_of.append({**g, 'count': left})
            for rec in view['aggregate']:
                need, fir_need = sellcalc.remaining_needs(
                    rec['total_needed'], rec['fir_needed'], rec['have'])
                if need <= 0:
                    continue
                srcs = rec['sources']
                first = srcs[0]['name'] if srcs else 'tasks'
                more = f' +{len(srcs) - 1} more' if len(srcs) > 1 else ''
                why = [f"{s['name']} ×{s['count']}" + (' FiR' if s['fir'] else '') for s in srcs]
                have = f", {rec['have']} already set aside" if rec['have'] else ''
                if rec['item_id'] in protected:        # on the keep list too: the needs add up
                    cur = protected[rec['item_id']]
                    cur['need'] += need
                    cur['fir_need'] += fir_need
                    cur['why'] += why
                    cur['kinds'] = sorted(set(cur.get('kinds') or ()) | {x['type'] for x in srcs})
                    cur['fir_only'] = cur['need'] == cur['fir_need']
                    continue
                protected[rec['item_id']] = {
                    'reason': f"Needed: {first} (×{rec['total_needed']}){more}",
                    'fir_only': need == fir_need, 'need': need, 'fir_need': fir_need,
                    'kinds': sorted({x['type'] for x in srcs}),
                    'why': why[:] if not have else why + [have.lstrip(', ')],
                }
    except Exception as e:
        print(f"[tasks] protected-id pass skipped: {e}")
    return protected, any_of


# ---------------------------------------------------------------------------
# Grid detection helpers
# ---------------------------------------------------------------------------


                            # energy is UI chrome (toolbar/header rule), not a grid
                            # line — grid lines sit ≤~2× median even when bold.


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

_badge_font_cache = {}
def _badge_font(size=11):
    if size not in _badge_font_cache:
        from PIL import ImageFont
        for path in ['C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf',
                     'C:/Windows/Fonts/calibrib.ttf', 'C:/Windows/Fonts/segoeui.ttf']:
            try:
                _badge_font_cache[size] = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
        else:
            _badge_font_cache[size] = ImageFont.load_default()
    return _badge_font_cache[size]


def draw_badge(draw, x, y, label, bg=(30, 160, 30, 230)):
    """Draw a small numbered badge at pixel position (x, y)."""
    font = _badge_font(11)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 3
    draw.rectangle([x, y, x + tw + pad * 2 + 1, y + th + pad * 2],
                   fill=bg, outline=(255, 255, 255, 160), width=1)
    draw.text((x + pad, y + pad), label, fill=(255, 255, 255, 255), font=font)


# PIL's ImageDraw takes RGB(A) tuples (the highlight overlay is composited
# onto `img`, a PIL Image, in the scan below — unlike the OpenCV/BGR
# pipeline the rest of this module uses for matching).
# This is the RGB equivalent of the amber/orange (0,165,255) BGR the FiR
# feature spec calls for, so it renders as amber rather than blue on screen.
FIR_AMBER_RGB = (255, 165, 0)


# ---------------------------------------------------------------------------
# Keep-list scan — one pipeline shared by the hotkey and the Scan button
# ---------------------------------------------------------------------------


DEBUG_DIR  = os.path.join(DATA, 'debug')
DEBUG_KEEP = 3   # scan bundles kept (ring buffer)


def _save_debug_bundle(img_bgr, panels, detections, kind):
    """
    Persist the raw frame + detected panels + every raw detection for the
    last few scans under data/debug/scan-<ts>-<kind>/.  This is what turns
    "that scan looked wrong" into an actionable report: the frame doubles as
    an eval image (test_scan.py --prefill data/debug/.../frame.png) and the
    JSON shows exactly what the matcher decided.  Best-effort — a failed
    dump must never break a scan.
    """
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        name = f'scan-{int(time.time() * 1000)}-{kind}'
        d = os.path.join(DEBUG_DIR, name)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, 'frame.png'), img_bgr)
        with open(os.path.join(d, 'scan.json'), 'w', encoding='utf-8') as f:
            json.dump({'kind': kind, 'panels': panels, 'detections': detections},
                      f, indent=1, ensure_ascii=False, default=str)
        old = sorted(p for p in os.listdir(DEBUG_DIR) if p.startswith('scan-'))
        for stale in old[:-DEBUG_KEEP]:
            shutil.rmtree(os.path.join(DEBUG_DIR, stale), ignore_errors=True)
    except Exception as e:
        print(f'[debug] bundle save failed: {e}')


def run_keep_scan(from_calibration=False):
    """
    Capture → identify every item (identify/ engine), annotated for the keep
    list.

    Returns {image, detections, grid, grid_failed, warnings, checklist_matches}.
    Raises ScanError/Exception — callers decide how to surface it.

    `from_calibration=True` reuses the region-picker's just-grabbed full-
    monitor frame (cropped to the configured region) instead of live-
    grabbing — see capture_for_scan.
    """
    _scan_state.update({'running': True, 'phase': 'capture',
                        'done': 0, 'total': 0, 'ts': time.time()})
    try:
        settings   = load_json(SETTINGS_PATH, default_settings)
        keep_list  = load_json(KEEPLIST_PATH, default_keep_list)
        warnings   = []
        # entry id -> keep-list category id, so the FiR check below only ever
        # gates KAPPA (Collector) entries — task/manual entries are unaffected.
        entry_cat  = {e['id']: cat['id']
                      for cat in keep_list['categories'] for e in cat['items']}

        if not tesseract_available():
            warnings.append('Tesseract OCR not installed — label reading disabled '
                            '(winget install UB-Mannheim.TesseractOCR)')

        img, img_bgr = capture_for_scan(settings, from_calibration,
                                        require_region=False, warnings=warnings)

        _scan_state['phase'] = 'identify'
        all_dets, panels, grid_failed = scan_with_v2(img_bgr, settings, warnings)
        grid = panels[0]
        print(f"Grid: {len(panels)} panel(s), "
              f"cell={grid['cell_w']:.2f}×{grid['cell_h']:.2f} "
              f"origin=({grid['origin_x']:.1f},{grid['origin_y']:.1f})")

        detections    = []
        found_entries = {}   # entry_id -> entry, for the confirm-to-check bar

        prices    = get_prices()
        price_idx = build_price_index(prices)
        keepid_to_entry, unmapped = map_keep_entries_to_ids(keep_list, price_idx)
        if unmapped:
            warnings.append(f"{len(unmapped)} keep-list item(s) not in the item "
                            f"catalog: {', '.join(unmapped[:3])}"
                            + ('…' if len(unmapped) > 3 else ''))

        if keepid_to_entry:
            draw = ImageDraw.Draw(img, 'RGBA')
            if settings.get('debug_dumps', True):
                _save_debug_bundle(img_bgr, panels, all_dets, 'keep')
            for d in all_dets:
                entry = keepid_to_entry.get(d['item_id'])
                if not entry:
                    continue
                x, y, w, h = d['px'], d['py'], d['pw'], d['ph']

                # A KAPPA entry whose detection reads confidently non-FiR
                # can't satisfy the Collector hand-in: amber-flag it and keep
                # it off the confirm bar. `fir is None`/True → unchanged.
                not_fir = entry_cat.get(entry['id']) == 'kappa' and d.get('fir') is False
                if not_fir:
                    color, border = FIR_AMBER_RGB + (130,), FIR_AMBER_RGB + (255,)
                else:
                    found_entries[entry['id']] = entry
                    color  = (80, 160, 80, 130) if entry['acquired'] else (180, 50, 50, 150)
                    border = (80, 200, 80, 255) if entry['acquired'] else (220, 60, 60, 255)
                draw.rectangle([x, y, x + w, y + h], fill=color, outline=border, width=2)
                name_suffix = ' (not FIR)' if not_fir else ''
                detections.append({
                    'text': f'{entry["name"]}{name_suffix}', 'matched': entry['name'],
                    'score': d['score'], 'acquired': entry['acquired'],
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'fir': d.get('fir'), 'not_fir': not_fir,
                })

        checklist_matches = [
            {'entry_id': e['id'], 'name': e['name']}
            for e in found_entries.values() if not e.get('acquired')
        ]

        buf = BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        return {'image': encoded, 'detections': detections, 'grid': grid,
                'grid_failed': grid_failed,
                'warnings': warnings, 'checklist_matches': checklist_matches}
    finally:
        _scan_state.update({'running': False, 'phase': None, 'ts': time.time()})


def do_scan():
    """Hotkey-triggered scan: same pipeline as the Scan button, results (or the
    error) parked in _last_scan for the frontend poller."""
    try:
        result = run_keep_scan()
        with _scan_lock:
            _last_scan.update(result, error=None, ts=time.time())
    except Exception as e:
        import traceback
        print(f"[hotkey scan] ERROR:\n{traceback.format_exc()}")
        with _scan_lock:
            _last_scan.update({'image': None, 'detections': [], 'warnings': [],
                               'checklist_matches': [], 'grid_failed': False,
                               'error': str(e), 'ts': time.time()})


# ---------------------------------------------------------------------------
# Global hotkey
# ---------------------------------------------------------------------------

class HotkeyManager:
    """Owns the pynput GlobalHotKeys listener; supports live rebinding."""
    def __init__(self):
        self._listener = None
        self._lock = threading.Lock()
        self.current = None

    @staticmethod
    def validate(hotkey_str):
        """Raises ValueError if pynput can't parse the combo."""
        keyboard.HotKey.parse(hotkey_str)

    def _on_activate(self):
        print(f"Hotkey {self.current} triggered — scanning...")
        if autoscanner.enabled():
            rescan_now()
        else:
            threading.Thread(target=do_scan, daemon=True).start()

    def register(self, hotkey_str):
        self.validate(hotkey_str)
        with self._lock:
            if self._listener is not None:
                try:
                    self._listener.stop()
                except Exception:
                    pass
                self._listener = None
            listener = keyboard.GlobalHotKeys({hotkey_str: self._on_activate})
            listener.daemon = True
            listener.start()
            self._listener = listener
            self.current = hotkey_str
        print(f"Hotkey listener active — press {hotkey_str} in-game to scan")


hotkey_manager = HotkeyManager()


def start_hotkey_listener():
    settings = load_json(SETTINGS_PATH, default_settings)
    hotkey_str = settings.get('hotkey', DEFAULT_HOTKEY)
    # Migrate the old F9 default: EFT and common overlays grab function keys,
    # so plain F9 frequently never reaches this app.
    if hotkey_str == '<f9>':
        hotkey_str = DEFAULT_HOTKEY
        settings['hotkey'] = hotkey_str
        save_json(SETTINGS_PATH, settings)
        print(f"Hotkey migrated F9 → {hotkey_str} (F9 conflicts with EFT itself)")
    try:
        hotkey_manager.register(hotkey_str)
    except Exception as e:
        print(f"Hotkey listener failed: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
@app.route('/live')
def live_page():
    return render_template('live.html')


@app.route('/needs')
def needs_page():
    return render_template('needs.html')


@app.route('/tasks')          # old pages
@app.route('/keep')
def tasks_redirect():
    return redirect('/needs')


@app.route('/sell')
def sell_redirect():
    return redirect('/')

@app.route('/api/last-scan', methods=['GET'])
def last_scan():
    since = float(request.args.get('since', 0))
    with _scan_lock:
        if _last_scan['ts'] > since and (_last_scan['image'] or _last_scan.get('error')):
            return jsonify({'ready': True, **_last_scan})
    return jsonify({'ready': False})

@app.route('/api/scan-status', methods=['GET'])
def scan_status():
    return jsonify(dict(_scan_state))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'tesseract':      tesseract_available(),
        'tesseract_cmd':  pytesseract.pytesseract.tesseract_cmd,
        'catalog_ready':  catalog_summary() is not None,
        'catalog_error':  _index_build_state.get('error'),
        'prices_cached':  os.path.exists(PRICES_PATH),
        'hotkey':         hotkey_manager.current,
        'app_version':    APP_VERSION,
    })

@app.route('/api/hotkey', methods=['POST'])
def set_hotkey():
    hotkey_str = (request.json or {}).get('hotkey', '').strip()
    try:
        hotkey_manager.register(hotkey_str)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Invalid hotkey: {e}'}), 400
    settings = load_json(SETTINGS_PATH, default_settings)
    settings['hotkey'] = hotkey_str
    save_json(SETTINGS_PATH, settings)
    return jsonify({'ok': True, 'hotkey': hotkey_str})

@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    data = request.get_json(silent=True) or {}
    from_calibration = bool(data.get('from_calibration'))
    try:
        return jsonify(run_keep_scan(from_calibration=from_calibration))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[screenshot] ERROR:\n{tb}")
        return jsonify({'error': str(e), 'image': None, 'detections': [],
                        'grid_failed': False,
                        'warnings': [], 'checklist_matches': []})

@app.route('/api/keep-list', methods=['GET'])
def get_keep_list():
    return jsonify(load_json(KEEPLIST_PATH, default_keep_list))

@app.route('/api/keep-list/toggle', methods=['POST'])
def toggle_item():
    data = request.json
    item_id = data.get('id')
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        for item in cat['items']:
            if item['id'] == item_id:
                item['acquired'] = not item['acquired']
                save_json(KEEPLIST_PATH, keep_list)
                return jsonify({'ok': True, 'acquired': item['acquired']})
    return jsonify({'ok': False, 'error': 'Item not found'}), 404

@app.route('/api/keep-list/add', methods=['POST'])
def add_item():
    data = request.json
    cat_id = data.get('category', 'kappa')
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        if cat['id'] == cat_id:
            new_item = {
                'id': str(uuid.uuid4())[:8],
                'name': data['name'],
                'aliases': data.get('aliases', []),
                'acquired': False,
                'source': 'custom',   # wiki sync must never remove user-added items
            }
            if data.get('tdev_id'):
                new_item['tdev_id'] = data['tdev_id']
            if 'task' in data:
                new_item['task'] = data['task']
            if 'count' in data:
                new_item['count'] = data['count']
            cat['items'].append(new_item)
            save_json(KEEPLIST_PATH, keep_list)
            return jsonify({'ok': True, 'item': new_item})
    return jsonify({'ok': False, 'error': 'Category not found'}), 404

@app.route('/api/keep-list/remove/<item_id>', methods=['DELETE'])
def remove_item(item_id):
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        cat['items'] = [i for i in cat['items'] if i['id'] != item_id]
    save_json(KEEPLIST_PATH, keep_list)
    return jsonify({'ok': True})

@app.route('/api/keep-list/acquire', methods=['POST'])
def acquire_items():
    """Batch-mark entries acquired — the scan confirm bar's one-click action."""
    ids = set((request.json or {}).get('ids') or [])
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    updated = 0
    for cat in keep_list['categories']:
        for item in cat['items']:
            if item['id'] in ids and not item['acquired']:
                item['acquired'] = True
                updated += 1
    if updated:
        save_json(KEEPLIST_PATH, keep_list)
    return jsonify({'ok': True, 'updated': updated})

@app.route('/api/kappa/refresh', methods=['POST'])
def kappa_refresh():
    """Force a wiki sync. keep_list.json is untouched when fetch/parse fails."""
    try:
        summary = kappa_sync(force=True)
        return jsonify({'ok': True, **summary})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 502

@app.route('/api/calibration-screenshot', methods=['GET'])
def calibration_screenshot():
    """Capture full screen for region selection — delay lets user tab to game first."""
    delay = int(request.args.get('delay', 0))
    if delay:
        time.sleep(delay)
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        raw = sct.grab(monitor)
        img = Image.frombytes('RGB', raw.size, raw.bgra, 'raw', 'BGRX')
    # Stash the full-monitor frame so a scan triggered right after the region
    # save (from_calibration=True) can crop this instead of live-grabbing.
    _last_calibration['bgr'] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    _last_calibration['ts'] = time.time()
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=75)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': encoded, 'width': img.width, 'height': img.height})

# ---------------------------------------------------------------------------
# Settings API (the Settings page edits every key below; unknown keys are kept as they are)
# ---------------------------------------------------------------------------

class _BadSetting(ValueError):
    """A settings value of the wrong type or outside its choices (the message is shown to the user)."""


def _s_bool(v):
    if isinstance(v, bool):
        return v
    raise _BadSetting('must be true or false')


def _s_bool_or_none(v):
    return None if v is None else _s_bool(v)


def _s_int(lo, hi, nullable=False):
    """A whole number, clamped into [lo, hi] (a number outside the range is corrected, not refused)."""
    def check(v):
        if v is None and nullable:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise _BadSetting('must be a number')
        if isinstance(v, float):
            if v != v or v in (float('inf'), float('-inf')) or not v.is_integer():
                raise _BadSetting('must be a whole number')
            v = int(v)
        return max(lo, min(hi, v))
    return check


def _s_enum(choices):
    """One of ``choices`` (matched case-insensitively, stored in the canonical spelling)."""
    canon = {c.lower(): c for c in choices}
    def check(v):
        if isinstance(v, str) and v.strip().lower() in canon:
            return canon[v.strip().lower()]
        raise _BadSetting('must be one of: ' + ', '.join(choices))
    return check


def _s_text(v):
    """Free text, trimmed; an empty string means 'not set' (None)."""
    if v is None:
        return None
    if not isinstance(v, str) or '\x00' in v or len(v) > 1024:
        raise _BadSetting('must be text')
    return v.strip() or None


def _s_hotkey(v):
    if not isinstance(v, str) or not v.strip():
        raise _BadSetting('must be a key combination such as <ctrl>+<shift>+s')
    v = v.strip()
    try:
        hotkey_manager.validate(v)
    except Exception as e:
        raise _BadSetting(f'is not a valid key combination ({e})')
    return v


def _s_name_list(v):
    if not isinstance(v, list) or len(v) > 64 or not all(isinstance(n, str) for n in v):
        raise _BadSetting('must be a list of names')
    out = []
    for n in (n.strip() for n in v):
        if n and len(n) <= 40 and n not in out:
            out.append(n)
    return out


def _s_trader_levels(v):
    if not isinstance(v, dict) or len(v) > 64:
        raise _BadSetting('must be an object of trader name to loyalty level')
    level = _s_int(1, 4)
    out = {}
    for name, lv in v.items():
        try:
            out[str(name)] = level(lv)
        except _BadSetting:
            raise _BadSetting(f'level for {name} must be a number from 1 to 4')
    return out


def _s_region(v):
    if v is None:
        return None
    if not isinstance(v, dict):
        raise _BadSetting('must be null or {x, y, w, h}')
    out = dict(v)
    for k, lo in (('x', 0), ('y', 0), ('w', 1), ('h', 1)):
        if k not in v:
            raise _BadSetting(f'is missing "{k}"')
        out[k] = _s_int(lo, 100000)(v[k])
    return out


# key -> checker returning the clean value (or raising _BadSetting).  A key that is not listed
# here is not ours to judge (other modules, hand-edited extras) and is stored as given.
_SETTING_RULES = {
    'region': _s_region, 'monitor': _s_int(0, 16), 'hotkey': _s_hotkey, 'prestige': _s_int(0, 6),
    'scan_countdown': _s_int(0, 10), 'kappa_only_tasks': _s_bool, 'debug_dumps': _s_bool,
    'flea_requires_fir': _s_bool_or_none, 'flea_offer_slots': _s_int(0, 100),
    'flea_overflow': _s_enum(('queue', 'trader')), 'flea_min_gain': _s_int(0, 10 ** 9),
    'intel_center_level': _s_int(0, 3, nullable=True), 'hideout_management_level': _s_int(0, 51),
    'skip_traders': _s_name_list, 'trader_levels': _s_trader_levels,
    'auto_scan': _s_bool, 'auto_scan_in_raid': _s_bool, 'auto_scan_exe': _s_text,
    'auto_task_progress': _s_bool, 'eft_install_dir': _s_text,
    'game_mode': _s_enum(('auto', 'pvp', 'pve', 'season')), 'faction': _s_enum(('auto', 'BEAR', 'USEC')),
    'start_with_windows': _s_bool, 'follow_tarkov': _s_bool, 'show_window_on_game_start': _s_bool,
    'ignore_task_items': _s_bool, 'ignore_hideout_items': _s_bool,
    'live_viewer': _s_bool, 'live_in_raid': _s_bool, 'live_top_n': _s_int(1, 20),
    'live_min_value_per_slot': _s_int(0, 10 ** 9),
}

# Modules that must react when a setting changes (lifecycle: the Windows start-up entry, ...)
# append ``fn(old_settings, new_settings)`` here; a failing hook never fails the save.
SETTINGS_CHANGED_HOOKS = []


def _validate_settings(incoming, current):
    """``current`` (the stored settings) updated with ``incoming``.  Returns ``(merged, errors)``;
    ``errors`` maps a key to what is wrong with it and ``merged`` is only good when it is empty.
    Keys that are not in ``incoming`` are left alone (a partial update never drops a setting) and
    a value equal to the stored one is not judged again, so a hand-edited oddity in settings.json
    cannot block every later save."""
    merged = dict(current)
    errors = {}
    for key, value in incoming.items():
        if key in current and type(current[key]) is type(value) and current[key] == value:
            continue
        check = _SETTING_RULES.get(key)
        try:
            merged[key] = check(value) if check else value
        except _BadSetting as e:
            errors[key] = f'{key} {e}'
    return merged, errors


@app.route('/api/settings', methods=['GET'])
def get_settings():
    # lifecycle keys filled in for settings files saved before they existed
    stored = lifecycle.with_defaults(load_json(SETTINGS_PATH, default_settings))
    if request.args.get('full') in ('1', 'true'):     # the Settings page: stored values over the defaults
        defaults = default_settings()
        return jsonify({'settings': {**defaults, **stored}, 'defaults': defaults})
    return jsonify(stored)

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Merge the posted keys into data/settings.json.  Any bad value refuses the whole request
    (400, nothing is written); on success the response carries the settings as saved."""
    incoming = request.get_json(silent=True)
    if not isinstance(incoming, dict):
        return jsonify({'ok': False, 'error': 'Expected a JSON object of settings.'}), 400
    old = load_json(SETTINGS_PATH, default_settings)
    new, errors = _validate_settings(incoming, old)
    if errors:
        return jsonify({'ok': False, 'error': '; '.join(errors.values()), 'errors': errors}), 400
    if new.get('hotkey') != old.get('hotkey') and hotkey_manager.current is not None:
        try:
            hotkey_manager.register(new['hotkey'])      # a changed hotkey takes effect at once
        except Exception as e:
            return jsonify({'ok': False, 'error': f'Invalid hotkey: {e}',
                            'errors': {'hotkey': str(e)}}), 400
    save_json(SETTINGS_PATH, new)
    try:
        autoscanner.poke()                              # Auto-scan on/off and friends: re-evaluate now
    except Exception:
        pass
    try:
        lifecycle.apply_settings(new)   # start_with_windows: register/unregister + start/stop the watcher, now
    except Exception as e:
        print(f'[settings] lifecycle: {e}')
    for hook in list(SETTINGS_CHANGED_HOOKS):
        try:
            hook(old, new)
        except Exception as e:
            print(f'[settings] change hook failed: {e}')
    return jsonify({'ok': True, 'settings': new})

@app.route('/api/reset', methods=['POST'])
def reset_keep_list():
    save_json(KEEPLIST_PATH, default_keep_list())
    # Land on current wiki data when online; the hardcoded defaults are only
    # the offline fallback.
    try:
        kappa_sync()
    except Exception as e:
        print(f"[kappa] post-reset sync skipped: {e}")
    return jsonify({'ok': True})

# ---------------------------------------------------------------------------
# Self-update
# ---------------------------------------------------------------------------

def _fetch_latest_release():
    """Hit the hardcoded GitHub releases/latest endpoint. Raises on any
    network/HTTP/JSON failure — callers must catch."""
    r = http_requests.get(GITHUB_RELEASES_API, timeout=6, headers={
        'User-Agent': 'TarkovStashHelper/1.0 (github.com/josfire18/tarkov-stash-helper)',
    })
    r.raise_for_status()
    return r.json()


def _get_update_status(force=False):
    """
    Shared cache-check + fetch logic behind /api/update-check and
    /api/update-apply. Populates _update_cache (the response the UI sees) and
    _update_state (the download URL, kept server-side) as a side effect.

    Never raises — any network/parse failure is folded into the returned
    dict's 'error' field with update_available False, so the caller always
    gets a 200-shaped result.
    """
    now = time.time()
    if not force and _update_cache['data'] and now - _update_cache['ts'] < UPDATE_CACHE_TTL:
        return _update_cache['data']

    result = {
        'current':           APP_VERSION,
        'latest':            None,
        'update_available':  False,
        'notes_url':         None,
        'can_auto':          False,
        'error':             None,
    }
    try:
        release = _fetch_latest_release()
        tag = release.get('tag_name') or ''
        update_available = _is_newer(tag, APP_VERSION)
        asset = next((a for a in (release.get('assets') or [])
                     if a.get('name') == UPDATE_ASSET_NAME), None)

        result['latest']           = tag
        result['notes_url']        = release.get('html_url')
        result['update_available'] = update_available
        result['can_auto']         = bool(FROZEN and update_available and asset)

        _update_state['tag'] = tag
        if update_available and asset:
            _update_state['download_url'] = asset.get('browser_download_url')
            _update_state['asset_size']   = asset.get('size')
        else:
            _update_state['download_url'] = None
            _update_state['asset_size']   = None
    except Exception as e:
        result['error'] = str(e)

    _update_cache['ts']   = now
    _update_cache['data'] = result
    return result


@app.route('/api/update-check', methods=['GET'])
def update_check():
    force = request.args.get('force') in ('1', 'true', 'True')
    return jsonify(_get_update_status(force=force))


@app.route('/api/update-apply', methods=['POST'])
def update_apply():
    if not FROZEN:
        return jsonify({'error': 'Running from source — update with git pull instead.'}), 400

    status = _get_update_status(force=False)
    if not status.get('update_available') or not _update_state.get('download_url'):
        # Cache may be empty/stale (e.g. app just started) — force one refresh
        # before giving up.
        status = _get_update_status(force=True)
    if not status.get('update_available') or not _update_state.get('download_url'):
        return jsonify({'error': status.get('error') or 'No update available to apply.'}), 400

    download_url = _update_state['download_url']
    asset_size = _update_state.get('asset_size')
    if not download_url.startswith('https://'):
        return jsonify({'error': 'Refusing a non-HTTPS asset URL.'}), 400

    exe_path = sys.executable
    exe_dir = os.path.dirname(exe_path)
    new_path = os.path.join(exe_dir, UPDATE_ASSET_NAME + '.new')

    try:
        with http_requests.get(download_url, stream=True, timeout=120, headers={
            'User-Agent': 'TarkovStashHelper/1.0 (github.com/josfire18/tarkov-stash-helper)',
        }) as r:
            r.raise_for_status()
            with open(new_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        return jsonify({'error': f'Download failed: {e}'}), 502

    downloaded_size = os.path.getsize(new_path) if os.path.exists(new_path) else 0
    if downloaded_size == 0 or (asset_size and downloaded_size != asset_size):
        try:
            os.remove(new_path)
        except Exception:
            pass
        return jsonify({'error': 'Downloaded update file was empty or the wrong size.'}), 502

    # updater.bat: waits for this process to exit (by PID), replaces the exe
    # with the freshly-downloaded one, relaunches it, then deletes itself.
    # Spawned detached below so it survives this process exiting.
    pid = os.getpid()
    bat_path = os.path.join(exe_dir, 'updater.bat')
    bat = (
        '@echo off\r\n'
        ':wait\r\n'
        f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul '
        '&& (timeout /t 1 /nobreak >nul & goto wait)\r\n'
        ':retry\r\n'
        f'del "{exe_path}" >nul 2>&1\r\n'
        f'if exist "{exe_path}" (timeout /t 1 /nobreak >nul & goto retry)\r\n'
        f'move /y "{new_path}" "{exe_path}" >nul\r\n'
        f'start "" "{exe_path}"\r\n'
        'del "%~f0"\r\n'
    )
    try:
        with open(bat_path, 'w', encoding='utf-8') as f:
            f.write(bat)
        subprocess.Popen(
            ['cmd', '/c', bat_path],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True, cwd=exe_dir,
        )
    except Exception as e:
        return jsonify({'error': f'Failed to launch updater: {e}'}), 502

    def _delayed_restart():
        # Give the HTTP response time to flush to the browser before tearing
        # the window/tray down.
        time.sleep(1.5)
        # the watcher runs from this same exe file, and updater.bat cannot replace a running exe
        _shutdown_desktop(stop_watcher=True)
    threading.Thread(target=_delayed_restart, daemon=True).start()

    return jsonify({'ok': True, 'restarting': True})

# ---------------------------------------------------------------------------
# Tasks & hideout page
# ---------------------------------------------------------------------------

@app.route('/api/tasks', methods=['GET'])
def api_tasks():
    try:
        cache = get_tasks()
    except Exception as e:
        return jsonify({'error': str(e)}), 502
    progress = effective_progress(cache)
    settings = load_json(SETTINGS_PATH, default_settings)
    kappa_only = settings.get('kappa_only_tasks', False)
    qp = request.args.get('kappa_only')
    if qp is not None:
        kappa_only = qp not in ('0', 'false', 'False')
    view = compute_tasks_view(cache, progress, kappa_only=kappa_only)
    view['cache_age_minutes'] = round(_cache_age_seconds(cache, 'tasks') / 60, 1)   # since last confirmed current
    view['source'] = cache.get('source') or 'graphql'
    view['task_progress'] = _task_progress_status(progress)
    view['auto_done'] = progress.get('auto_done') or []
    return jsonify(view)

@app.route('/api/tasks/refresh', methods=['POST'])
def api_tasks_refresh():
    try:
        cache = fetch_tasks()
        return jsonify({'ok': True, 'tasks': len(cache['tasks']),
                        'stations': len(cache['hideoutStations'])})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 502

@app.route('/api/tasks/complete', methods=['POST'])
def api_tasks_complete():
    data = request.json or {}
    typ, tid, done = data.get('type'), data.get('id'), bool(data.get('done'))
    if typ not in ('task', 'hideout') or not tid:
        return jsonify({'ok': False, 'error': 'type must be task|hideout, id required'}), 400
    progress = load_json(PROGRESS_PATH, default_progress)
    key = 'completed_tasks' if typ == 'task' else 'completed_hideout'
    ids = set(progress.get(key, []))
    (ids.add if done else ids.discard)(tid)
    progress[key] = sorted(ids)
    if typ == 'task':
        # An explicit click beats what the logs say, both ways (see effective_progress).
        progress.setdefault('manual_overrides', {})[tid] = 'done' if done else 'open'
    save_json(PROGRESS_PATH, progress)
    return jsonify({'ok': True})


def _task_progress_status(progress=None):
    settings = load_json(SETTINGS_PATH, default_settings)
    progress = progress or effective_progress()
    return {'enabled': bool(settings.get('auto_task_progress', True)),
            'log': progress.get('log_status'), 'error': _log_progress['error'],
            'auto_done': len(progress.get('auto_done') or ()),
            'install_dir': _log_scanner.install_dir, 'scanned_at': _log_scanner.last_scan_at}


@app.route('/api/task-progress/status', methods=['GET'])
def api_task_progress_status():
    return jsonify(_task_progress_status())


@app.route('/api/task-progress/rescan', methods=['POST'])
def api_task_progress_rescan():
    res = scan_task_logs(full=True)
    return jsonify({'ok': res is not None, 'scan': res, **_task_progress_status()})

@app.route('/api/tasks/have', methods=['POST'])
def api_tasks_have():
    data = request.json or {}
    item_id = data.get('item_id')
    if not item_id:
        return jsonify({'ok': False, 'error': 'item_id required'}), 400
    try:
        delta = int(data.get('delta') or 0)
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'delta must be an integer'}), 400
    progress = load_json(PROGRESS_PATH, default_progress)
    have = progress.setdefault('have', {})
    have[item_id] = max(0, int(have.get(item_id, 0)) + delta)
    save_json(PROGRESS_PATH, progress)
    return jsonify({'ok': True, 'have': have[item_id]})


@app.route('/api/hideout/level', methods=['POST'])
def api_hideout_level():
    """Set a station's level in one call: every level <= `level` counts as built, the rest not."""
    data = request.get_json(silent=True) or {}
    station_id = data.get('station_id')
    try:
        level = int(data.get('level'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'level must be an integer'}), 400
    try:
        cache = get_tasks()
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 502
    station = next((s for s in cache.get('hideoutStations', []) if s.get('id') == station_id), None)
    if station is None:
        return jsonify({'ok': False, 'error': 'unknown station'}), 404
    progress = load_json(PROGRESS_PATH, default_progress)
    done = set(progress.get('completed_hideout', []))
    for lv in station.get('levels', []):
        (done.add if lv['level'] <= level else done.discard)(lv['id'])
    progress['completed_hideout'] = sorted(done)
    save_json(PROGRESS_PATH, progress)
    return jsonify({'ok': True, 'level': level})


@app.route('/api/items/search', methods=['GET'])
def api_items_search():
    """Autocomplete for the Needs page's "Add item": best name / short-name matches first."""
    q = (request.args.get('q') or '').strip().lower()
    if len(q) < 2:
        return jsonify({'items': []})
    hits = []
    for it in _usable_prices(_load_cache(PRICES_PATH) or {}).get('items', []):
        name, short = it['name'].lower(), (it.get('shortName') or '').lower()
        if q in name or q in short:
            rank = 0 if name.startswith(q) or short.startswith(q) else 1
            hits.append((rank, len(name), it))
    hits.sort(key=lambda h: h[:2])
    return jsonify({'items': [{'id': it['id'], 'name': it['name'], 'shortName': it.get('shortName')}
                              for _, _, it in hits[:12]]})


@app.route('/api/prestige', methods=['GET'])
def api_prestige():
    settings = load_json(SETTINGS_PATH, default_settings)
    try:
        levels = fetch_prestige_requirements()
    except Exception as e:
        return jsonify({'levels': [], 'current': settings.get('prestige', 3),
                        'cache_age_minutes': None, 'error': str(e)})
    cache = load_json(PRESTIGE_WIKI_PATH, lambda: None) if os.path.exists(PRESTIGE_WIKI_PATH) else None
    age = round((time.time() - cache.get('timestamp', 0)) / 60, 1) if cache else None
    return jsonify({'levels': levels, 'current': settings.get('prestige', 3),
                    'cache_age_minutes': age})


@app.route('/api/prestige/advance', methods=['POST'])
def api_prestige_advance():
    """New prestige: bumps settings['prestige'] (capped at 6), resets task/
    hideout progress, and un-acquires every keep-list entry (kappa + manual).
    Region/hotkey/grid/monitor and other settings fields are untouched."""
    settings = load_json(SETTINGS_PATH, default_settings)
    settings['prestige'] = min(6, settings.get('prestige', 0) + 1)
    save_json(SETTINGS_PATH, settings)

    save_json(PROGRESS_PATH, default_progress())

    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    for cat in keep_list['categories']:
        for item in cat['items']:
            item['acquired'] = False
    save_json(KEEPLIST_PATH, keep_list)

    return jsonify({'prestige': settings['prestige']})


# ---------------------------------------------------------------------------
# Sell page
# ---------------------------------------------------------------------------

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

def _cache_status(path, kind, count_key):
    """Status of one cache for /api/prices/status: age since last confirmed current, when it was
    last rewritten, which source produced it, and the last refresh error (None when the latest
    attempt succeeded)."""
    cache = _load_cache(path)
    if not cache or not cache.get(count_key):
        return {'cached': False, 'age_minutes': None, 'count': 0,
                'error': _refresh_state[kind]['error']}
    meta = _load_meta()
    now = time.time()
    return {
        'cached': True,
        'age_minutes': round(_cache_age_seconds(cache, kind) / 60, 1),
        'updated_minutes': round((now - cache.get('timestamp', 0)) / 60, 1),
        'count': len(cache[count_key]),
        'source': cache.get('source') or 'graphql',
        'error': _refresh_state[kind]['error'] or meta['errors'].get(kind),
    }


@app.route('/api/prices/status', methods=['GET'])
def prices_status():
    st = _cache_status(PRICES_PATH, 'prices', 'items')
    st['stale'] = bool(st['cached'] and st['age_minutes'] > 3 * PRICE_REFRESH_INTERVAL / 60)
    st['refresh_minutes'] = PRICE_REFRESH_INTERVAL // 60
    st['tasks'] = _cache_status(TASKS_CACHE_PATH, 'tasks', 'tasks')
    st['catalog_pending'] = len(_load_meta()['catalog_pending'])
    st['catalog_building'] = _index_build_state['running']
    return jsonify(st)

@app.route('/api/prices/refresh', methods=['POST'])
def prices_refresh():
    try:
        res = refresh_prices()
        return jsonify({'ok': True, 'count': len(res['cache']['items']), 'status': res['status'],
                        'source': res['source'], 'new_items': len(res['new_ids'])})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


_index_build_state = {'running': False, 'phase': None, 'done': 0, 'total': 0, 'ts': 0, 'error': None}
_catalog_rerun = threading.Event()     # items changed while a build was running: build once more
_startup_done = threading.Event()      # the engine warm-up at launch finished (builds wait for it)


def _wait_for_idle_scan(timeout=300):
    """Block until no scan is running (a hot-swap of the engine must not pull it out from under one)."""
    t0 = time.time()
    while _scan_state['running'] and time.time() - t0 < timeout:
        time.sleep(1)


def run_icon_db_build(reason='manual'):
    """The Build Icon DB work, synchronously (the caller claimed ``_index_build_state``): download
    the base image of every item that lacks one, rebuild the catalog, wait for any running scan to
    finish, then hot-swap the engine.  New-item ids that were pending are cleared once every image
    arrived.  Repeats once more if items changed while it ran."""
    try:
        if reason != 'manual':
            _startup_done.wait(600)           # never race the launch-time engine warm-up
        while True:
            _catalog_rerun.clear()
            pending = set(_load_meta()['catalog_pending'])
            prices = get_prices()
            def cb(done, total):
                _index_build_state['done']  = done
                _index_build_state['total'] = total
            _index_build_state.update({'phase': 'images', 'done': 0, 'total': 0})
            ok, failed = download_missing_base_images(prices, progress_cb=cb)
            print(f"[catalog] base images: {ok} downloaded, {failed} failed ({reason})")
            _index_build_state['phase'] = 'catalog'    # ~35 s, no progress to report
            from identify.catalog import load_catalog
            load_catalog(force_rebuild=True)
            _index_build_state['phase'] = 'engine'
            _wait_for_idle_scan()
            _warm_v2_engine()
            print('[catalog] built')
            if failed == 0 and pending:
                meta = _load_meta()
                meta['catalog_pending'] = sorted(set(meta['catalog_pending']) - pending)
                _save_meta_quietly(meta)
            if not _catalog_rerun.is_set():
                break
    except Exception as e:
        _index_build_state['error'] = str(e)
        print(f"[catalog] build failed: {e}")
    finally:
        _index_build_state.update({'running': False, 'phase': None})


def start_icon_db_build(reason='manual'):
    """Start :func:`run_icon_db_build` in the background.  False (and a request to build once
    more when the running build finishes) if one is already running."""
    if _index_build_state['running']:
        _catalog_rerun.set()
        return False
    # claim the slot before the thread starts so a double click cannot start two builds
    _index_build_state.update({'running': True, 'phase': 'images', 'done': 0, 'total': 0,
                               'ts': time.time(), 'error': None})
    threading.Thread(target=run_icon_db_build, args=(reason,), daemon=True).start()
    return True


def _has_catalog():
    """True once this install has built its item catalog (so the app should keep it current)."""
    return catalog_summary() is not None


def _queue_catalog_update():
    """New items arrived: fetch their images and rebuild the catalog in the background, but only
    on installs that already have a catalog (a clean install waits for its first Build Icon DB,
    which covers them).  The ids stay in the meta file until a build has covered them."""
    if _has_catalog():
        start_icon_db_build('new items')


@app.route('/api/icons/build-index', methods=['POST'])
def icons_build_index():
    """Build everything the identification engine needs: download the tarkov.dev base image of
    every item that lacks one (data/tmpl_src/), then rebuild the template catalog from them
    (data/identify_catalog_v2.npz) and load it into the running engine."""
    if not start_icon_db_build('manual'):
        return jsonify({'ok': False, 'message': 'Build already in progress'})
    return jsonify({'ok': True, 'message': 'Icon DB build started'})


@app.route('/api/icons/matcher-status', methods=['GET'])
def icons_matcher_status():
    """Status endpoint for the sell page: build progress + whether the catalog is ready."""
    running = _index_build_state['running']
    summary = None if running else catalog_summary()
    return jsonify({
        'running':    running,
        'phase':      _index_build_state.get('phase'),
        'done_count': _index_build_state.get('done', 0),
        'total':      _index_build_state.get('total', 0),
        'error':      _index_build_state.get('error'),
        'ready':      summary is not None,
        'item_count': summary['items'] if summary else 0,
    })


@app.route('/api/sell-scan', methods=['POST'])
def sell_scan():
  try:
    data = request.get_json(silent=True) or {}
    from_calibration = bool(data.get('from_calibration'))
    return _sell_scan_inner(from_calibration=from_calibration)
  except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"[sell_scan] ERROR:\n{tb}")
    return jsonify({'error': str(e), 'traceback': tb, 'image': None, 'results': [], 'grid': None})

def _sell_payload(raw_detections, panels, grid_failed, warnings, scene_info, settings, img, img_bgr,
                  lean, final=True):
    """Everything after identification: prices, protection plan, advice, rows.  Pure of the
    capture; ``final=False`` builds the provisional payload of a progressive scan."""
    warnings = list(warnings)
    grid = panels[0]
    print(f"Grid: {len(panels)} panel(s), "
          f"cell={grid['cell_w']:.2f}×{grid['cell_h']:.2f} "
          f"origin=({grid['origin_x']:.1f},{grid['origin_y']:.1f})")

    prices     = get_prices()
    if prices.get('stale_error'):
        age_h = (time.time() - prices.get('timestamp', 0)) / 3600
        warnings.append(f'Price refresh failed ({prices["stale_error"]}); '
                        f'using prices from {age_h:.0f} h ago')
    price_idx  = build_price_index(prices)
    id_to_item = {it['id']: it for it in prices.get('items', [])}

    # Everything the player shouldn't sell: unacquired keep-list entries +
    # task/hideout items still short of their required count.
    keep_list = load_json(KEEPLIST_PATH, default_keep_list)
    protected, any_of = get_protected_plan(keep_list, price_idx)

    if not tesseract_available():
        warnings.append('Tesseract OCR not installed — name reading disabled '
                        '(winget install UB-Mannheim.TesseractOCR)')

    print(f"[sell_scan] matches: {len(raw_detections)}")
    if settings.get('debug_dumps', True) and not lean and final:    # the live view saves its frames elsewhere
        _save_debug_bundle(img_bgr, panels, raw_detections, 'sell')

    # Sell advice covers the stash and open container windows only.  What is on the player
    # (rig / pockets / backpack / pouch), loot and unclear grids are listed apart ("On you").
    on_you = []
    raid = bool(scene_info.get('in_raid')) or str(scene_info.get('scene') or '').startswith('raid')
    live_advice = None
    if raid:           # grab / drop advice from the raw detections; nothing is sold from a raid view
        try:
            import liveadvice
            live_advice = liveadvice.compute_advice(
                raw_detections, scene_info.get('grids') or [], id_to_item, protected, any_of,
                settings, build_sell_context(settings, prices), scene_info.get('scene'), True)
        except Exception as e:
            print(f'[live] raid advice failed: {e}')
            warnings.append(f'Raid advice failed ({e})')
    if any(d.get('role') for d in raw_detections):
        advice = []
        for d in raw_detections:
            if d.get('role') in (None, 'stash', 'container_window') and not raid:
                advice.append(d)
            else:
                it = id_to_item.get(d['item_id']) or {}
                on_you.append({'matched_name': it.get('name') or d.get('name'), 'item_id': d['item_id'],
                               'count': d.get('count') or 1, 'role': d.get('role') or 'unknown', 'side': d.get('side'),
                               'title': d.get('region_title'), 'uncertain': bool(d.get('uncertain')),
                               **({'provisional': True} if d.get('provisional') else {}),
                               'px': d['px'], 'py': d['py'], 'pw': d['pw'], 'ph': d['ph']})
        raw_detections = advice

    # --- Build every entry, numbered in SELL order ---------------------------
    # The picture goes out untouched: the Live page outlines every tile in its action's
    # colour client-side.  The decisions themselves are sellcalc.plan_entries (pure).
    skipped = []          # guns/dogtags: recognised but never priced
    sellable = []
    for d in sorted(raw_detections, key=lambda r: (r['panel'], r['row'], r['col'])):
        badge = skip_badge(id_to_item.get(d['item_id']), d.get('category'))
        if badge:
            skipped.append({'px': d['px'], 'py': d['py'], 'pw': d['pw'], 'ph': d['ph'],
                            'label': badge, 'matched_name': d.get('name')})
        else:
            sellable.append(d)
    skipped_weapons = len(skipped)

    ctx = build_sell_context(settings, prices)
    results, keep_results = sellcalc.plan_entries(sellable, id_to_item, protected, settings, ctx, any_of)

    alts = {(d['px'], d['py']): d['alternatives'] for d in raw_detections if d.get('alternatives')}
    for k in keep_results:
        if k.get('check') and alts.get((k['px'], k['py'])):
            k['candidates'] = alts[(k['px'], k['py'])]

    results.extend(keep_results)   # KEEP items always at the end

    if lean:           # the live view fetches the picture separately (/api/autoscan/frame)
        encoded = None
    else:
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=88)
        encoded = base64.b64encode(buf.getvalue()).decode()
    return {
        'image':       encoded,
        'image_mime':  'image/jpeg',
        'skipped':     skipped,
        'results':     results,
        'grid':        grid,
        'grid_failed': grid_failed,
        'warnings':    warnings,
        'skipped_weapons': skipped_weapons,
        'scene':       scene_info.get('scene'),
        'regions':     scene_info.get('regions') or [],
        'on_you':      on_you,
        'advice':      live_advice,
    }


def _sell_scan_inner(from_calibration=False, frame_bgr=None, lean=False, publish=None, cancel=None):
    settings = load_json(SETTINGS_PATH, default_settings)

    _scan_state.update({'running': True, 'phase': 'capture',
                        'done': 0, 'total': 0, 'ts': time.time()})
    try:
        warnings = []

        # --- Screenshot (region required for sell scans) ----------------------
        try:
            if frame_bgr is not None:      # full frame captured passively by autoscan.py
                img_bgr = frame_bgr
                img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            else:
                img, img_bgr = capture_for_scan(settings, from_calibration,
                                                require_region=False, warnings=warnings)
        except ScanError as e:
            return jsonify({'image': None, 'results': [], 'grid': None,
                            'grid_failed': False, 'error': str(e)})

        # --- Identification (grid, per-panel origins, every item) -------------
        _scan_state['phase'] = 'identify'
        scene_info = {}
        in_raid = None
        if frame_bgr is not None:          # whole game frame: the lobby menu bar tells raid from lobby
            try:
                from autoscan.detect import menu_chrome
                in_raid = not menu_chrome(img_bgr)
            except Exception:
                pass
        def first_look(records, prov_panels):
            payload = _sell_payload(records, prov_panels, False, warnings, scene_info, settings,
                                    img, img_bgr, lean, final=False)
            payload['provisional'] = True
            publish(payload)

        raw_detections, panels, grid_failed = scan_with_v2(
            img_bgr, settings, warnings, scene_out=scene_info, in_raid=in_raid,
            on_provisional=first_look if publish is not None else None, cancel=cancel)
        payload = _sell_payload(raw_detections, panels, grid_failed, warnings, scene_info, settings,
                                img, img_bgr, lean)
        return jsonify(payload)
    finally:
        _scan_state.update({'running': False, 'phase': None, 'ts': time.time()})


# ---------------------------------------------------------------------------
# Auto-scan (autoscan/): passive capture of the game window, scan on a settled stash
# ---------------------------------------------------------------------------

def _autoscan_scan(frame_bgr, publish=None, cancel=None):
    """Run the normal sell scan on an already-captured full frame; returns its JSON payload.
    ``publish(payload)`` receives the provisional payload (fast first look) before the final one is
    returned; ``cancel()`` stops the pending certification when a newer view arrived."""
    with app.app_context():
        return _sell_scan_inner(frame_bgr=frame_bgr, lean=True, publish=publish, cancel=cancel).get_json()


def _load_settings():
    return load_json(SETTINGS_PATH, default_settings)


import autoscan  # noqa: E402  (needs app + the helpers above)
autoscanner = autoscan.AutoScanner(_autoscan_scan, _load_settings,
                                   busy_fn=lambda: _scan_state['running'], progressive=True,
                                   collect_dir=os.path.join(DATA, 'scenes', 'live'))
app.register_blueprint(autoscan.make_blueprint(
    autoscanner, _load_settings, lambda s: save_json(SETTINGS_PATH, s)))
def rescan_now():
    """Scan the view on screen again, even if it has not changed since the last scan."""
    autoscanner.trigger.reset_view()
    autoscanner.poke()


@app.route('/api/autoscan/rescan', methods=['POST'])
def autoscan_rescan():
    if not autoscanner.enabled():
        return jsonify({'ok': False, 'error': 'Auto-scan is off'}), 409
    rescan_now()
    return jsonify({'ok': True})


# GET /api/lifecycle/status, POST /api/lifecycle/show (a second launch by hand brings this window up)
app.register_blueprint(lifecycle.make_blueprint(_load_settings, lambda: _show_window()))


HOST = '127.0.0.1'
PORT = int(os.environ.get('TSH_PORT', 8877))   # env override: smoke tests next to a running instance
URL = f'http://{HOST}:{PORT}'


def run_server():
    """The Flask app + all /api routes are unchanged — this just serves them
    on localhost instead of the old port-80/custom-hostname setup. The window
    below is the only thing that changed; nothing about scanning or OCR was touched."""
    from waitress import serve
    serve(app, host=HOST, port=PORT, _quiet=True, threads=12)   # SSE streams hold a thread each


def migrate_settings(path=None):
    """One-time fixes to settings saved by older versions.  Returns the list of migrations run."""
    path = path or SETTINGS_PATH
    if not os.path.exists(path):
        return []
    s = load_json(path, default_settings)
    done = set(s.get('migrations') or ())
    ran = []
    if 'kappa_scope_all_tasks' not in done:
        # kappa_only_tasks used to default to on, when ~200 quests were Kappa-required; since 1.0
        # only the Collector's ~13-quest chain is, so "on" sold nearly every open quest's items.
        s['kappa_only_tasks'] = False
        ran.append('kappa_scope_all_tasks')
    if ran:
        s['migrations'] = sorted(done | set(ran))
        save_json(path, s)
    return ran


def _startup_maintenance():
    """One-shot background housekeeping: purge retired CNN model files from
    user machines and freshen the kappa list from the wiki when stale."""
    for fn in ('icon_model.pt', 'icon_model_meta.json', 'icon_index.json'):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[cleanup] removed retired file {fn}")
            except Exception as e:
                print(f"[cleanup] could not remove {fn}: {e}")
    migrate_settings()
    try:
        scan_task_logs(full=True)            # quest progress from the game logs, before the first scan
    except Exception as e:
        print(f"[eftlogs] startup scan skipped: {e}")
    try:
        cache = load_json(KAPPA_WIKI_PATH, lambda: None) if os.path.exists(KAPPA_WIKI_PATH) else None
        if not cache or time.time() - cache.get('timestamp', 0) > KAPPA_WIKI_TTL:
            kappa_sync()
    except Exception as e:
        print(f"[kappa] startup sync skipped (offline?): {e}")
    # Warm the v2 identification engine (loads/builds the template catalog - ~35 s the first
    # time - and the optional DINO model) so the first hotkey scan isn't the one that pays for it.
    try:
        _warm_v2_engine()
    finally:
        _startup_done.set()


def _warm_v2_engine():
    try:
        if os.path.exists(PRICES_PATH):
            from identify.config import EngineSettings
            from identify.pipeline import Engine
            es = EngineSettings.from_settings(load_json(SETTINGS_PATH, default_settings))
            _v2_engine[:] = [repr(es), Engine(es)]
            print('[v2] identification engine ready')
    except Exception as e:
        print(f"[v2] warm-up skipped: {e}")


# ---------------------------------------------------------------------------
# Background refresher: keeps prices, tasks and the item catalog current without being asked
# ---------------------------------------------------------------------------

def _refresh_due(kind, path, interval, meta, now):
    """True when this cache was last confirmed current more than ``interval`` seconds ago (so a
    cache that is already old at startup refreshes immediately) and the last attempt - failed ones
    included - is at least RETRY_BACKOFF (or the interval, if shorter) back."""
    cache = _load_cache(path)
    confirmed = max((cache or {}).get('timestamp', 0), meta['checked'].get(kind) or 0) if cache else 0
    if now - confirmed < interval:
        return False
    attempt = max(meta['attempted'].get(kind) or 0, _refresh_state[kind]['attempt'])
    return now - attempt >= min(RETRY_BACKOFF, interval)


def refresh_cycle(now=None):
    """One pass of the refresher: refresh whatever is due (prices first - the task data takes its
    item names from them), then resume catalog work a previous run left unfinished.  Never
    raises.  Returns the list of what was refreshed."""
    now = time.time() if now is None else now
    ran = []
    for kind, path, interval, fn in (
            ('prices', PRICES_PATH, PRICE_REFRESH_INTERVAL, refresh_prices),
            ('tasks', TASKS_CACHE_PATH, TASKS_REFRESH_INTERVAL, refresh_tasks)):
        try:
            if _refresh_due(kind, path, interval, _load_meta(), now):
                fn()
                ran.append(kind)
        except Exception as e:
            print(f'[refresh] {kind}: {e}')
    try:
        if scan_task_logs():
            ran.append('task_logs')
    except Exception as e:
        print(f'[refresh] task logs: {e}')
    try:
        if _load_meta()['catalog_pending'] and not _index_build_state['running']:
            _queue_catalog_update()
    except Exception as e:
        print(f'[refresh] catalog: {e}')
    return ran


_refresher = {'thread': None, 'stop': threading.Event()}


def _refresher_loop(stop, poll=60):
    while not stop.is_set():
        refresh_cycle()
        stop.wait(poll)


def start_refresher():
    """Start the daemon thread (idempotent).  The first pass runs at once, so a cache that is
    older than its interval is refreshed on startup."""
    t = _refresher['thread']
    if t is not None and t.is_alive():
        return t
    _refresher['stop'].clear()
    t = threading.Thread(target=_refresher_loop, args=(_refresher['stop'],), daemon=True,
                         name='tarkovdev-refresher')
    _refresher['thread'] = t
    t.start()
    return t


# Module-level handle to the running pywebview window / pystray icon / quit
# event, populated by _run_app. Lets the update-apply route's restart thread
# (which has no closure over _run_app's locals) tear the desktop down through
# the exact same path the tray's Quit menu item uses. Stays all-None when run
# from source without the desktop shell (e.g. under pytest).
_desktop = {'window': None, 'tray': None, 'tray_thread': None, 'quitting': None}


def _show_window():
    """Bring the window up (tray 'Open', and a second launch by hand via /api/lifecycle/show)."""
    window = _desktop.get('window')
    if window is None:
        raise RuntimeError('no window yet')
    lifecycle.allow_activation(window)   # a window started with focus=False could not take focus on click
    window.show()


def _shutdown_desktop(stop_watcher=False):
    """
    Tear down the tray + window the same way the tray's Quit action does,
    then exit the process. Shared by on_quit, the post-update-apply
    restart and "Tarkov exited" so all paths behave identically. When
    _desktop was never populated (no pywebview window — running windowless
    from source), this just falls back to os._exit(0).

    The watcher (lifecycle.py) is left running - it opens the app again
    for the next Tarkov launch - unless ``stop_watcher``: the self-update
    must be able to replace the exe the watcher is running from.
    """
    quitting = _desktop.get('quitting')
    if quitting is not None:
        quitting.set()
    _refresher['stop'].set()
    try:
        autoscanner.stop()      # releases the screen duplication
    except Exception:
        pass
    if stop_watcher:
        try:
            lifecycle.stop_watcher()
        except Exception:
            pass
    tray = _desktop.get('tray')
    if tray is not None:
        try:
            tray.stop()
            t = _desktop.get('tray_thread')
            if t is not None:
                t.join(1.0)     # let it delete its notification-area icon, or a ghost icon stays until hovered
        except Exception:
            pass
    window = _desktop.get('window')
    if window is not None:
        try:
            window.destroy()
        except Exception:
            pass
    os._exit(0)


def _run_app():
    """Hosts the local Flask UI in a native window (pywebview) instead of a
    browser tab, so there's no URL for the user to see or navigate to — it
    just looks like a normal desktop app. Closing the window minimizes to
    the tray; Quit from the tray menu actually exits."""
    import inspect
    import webview
    import pystray
    from icon_asset import load_tray_image, set_app_id, ensure_ico, apply_window_icon

    # own AppUserModelID before any window: the taskbar shows our icon, not python.exe's
    print(f"[icon] AppUserModelID set: {set_app_id()}")

    # Single instance, the startup entry + watcher, and "close when Tarkov exits" (lifecycle.py).
    # None = a copy is already running; it was told to show its window and this one just leaves.
    session = lifecycle.begin_app_session(sys.argv[1:], _load_settings, _shutdown_desktop)
    if session is None:
        return

    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=_startup_maintenance, daemon=True).start()
    start_refresher()
    start_hotkey_listener()
    autoscanner.start()

    # Started by the watcher (or to the tray): never take the focus from the game, and with
    # show_window_on_game_start off do not open a window at all.
    opts = lifecycle.window_kwargs(inspect.signature(webview.create_window).parameters,
                                   session['quiet'], session['hidden'])
    window = webview.create_window(
        'Tarkov Stash Helper', URL,
        width=1180, height=860, min_size=(900, 640), **opts,
    )

    quitting = threading.Event()
    _desktop['window'] = window
    _desktop['quitting'] = quitting

    def on_closing():
        if quitting.is_set():
            return True  # allow the real close (Quit was chosen from the tray)
        window.hide()
        return False  # veto the close — minimize to tray instead

    window.events.closing += on_closing
    window.events.loaded += lambda: lifecycle.allow_activation(window)
    ico = ensure_ico(os.path.join(DATA, 'app.ico'))

    def sharpen_icon(*_):        # taskbar / Alt-Tab get a full-size image, not WinForms' single 16 px one
        try:
            if ico:
                apply_window_icon(int(window.native.Handle.ToInt64()), ico)
        except Exception as e:
            print(f"[icon] {e}")
    window.events.shown += sharpen_icon

    def on_open(icon, item):
        _show_window()

    def on_quit(icon, item):
        _shutdown_desktop()      # the watcher stays: it opens the app again at the next Tarkov launch

    def on_quit_stop(icon, item):
        try:
            s = _load_settings()
            s['start_with_windows'] = False
            save_json(SETTINGS_PATH, s)
            lifecycle.apply_settings(s)      # unregisters the startup entry and stops the watcher
        except Exception as e:
            print(f"[lifecycle] could not turn auto-launch off: {e}")
        _shutdown_desktop(stop_watcher=True)

    tray_icon = pystray.Icon(
        'TarkovStashHelper',
        load_tray_image(),
        'Tarkov Stash Helper',
        menu=pystray.Menu(
            pystray.MenuItem('Open Stash Helper', on_open, default=True),
            pystray.MenuItem('Quit', on_quit),
            pystray.MenuItem('Quit and stop auto-launch', on_quit_stop),
        ),
    )

    def tray_setup(icon):
        icon.visible = True
        if session.get('first_registration'):
            try:
                icon.notify('It will now open when Tarkov starts and close when Tarkov exits. '
                            'Right-click this icon > "Quit and stop auto-launch" to turn that off.',
                            'Tarkov Stash Helper')
            except Exception:
                pass

    _desktop['tray'] = tray_icon
    _desktop['tray_thread'] = threading.Thread(target=tray_icon.run, args=(tray_setup,), daemon=True)
    _desktop['tray_thread'].start()

    webview.start(icon=ico)  # blocks; owns the main thread


if __name__ == '__main__':
    if '--selftest' in sys.argv:    # packaged-build smoke test: no window/tray/hotkey (selftest.py)
        import selftest
        sys.exit(selftest.main(sys.modules[__name__], sys.argv[sys.argv.index('--selftest') + 1:]))
    # Packaged windowed builds (PyInstaller --windowed) have no console and
    # sys.stdout is None, so print() would raise — route output to a log file.
    if FROZEN and sys.stdout is None:
        log_path = os.path.join(DATA, 'app.log')
        log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        print(f"Tarkov Stash Helper starting (internal: {URL})")
    _run_app()
