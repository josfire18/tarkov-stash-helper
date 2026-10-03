"""Sell advice economics: flea fee, flea list price, trader choice, flea offer
slots and keep allocation.

Pure functions only - no network, no files, stdlib only - so they are cheap to
unit-test (tests/test_sell_order.py, tests/test_sellcalc.py) and app.py's scan
route stays thin.  app.py re-exports the historical names (``best_trader_price``,
``calc_flea_fee``, ``sell_recommendation`` ...), so ``import app`` keeps working
for test_scan.py's sell-decision metric.

Sources (checked 2026-10-01).  Constants below cite the one they come from;
the numbers CHANGED on 3 Aug 2026, so anything older is stale:

* [PATCH]  Official patch notes 1.1.0.0.46608 (3 Aug 2026), "Economy":
           "Increased the Flea Market fee from 3% to 5%" and "Reduced the price
           traders pay for items sold by players by an average of 20%".
           Quoted in the EFT wiki changelog,
           https://escapefromtarkov.fandom.com/wiki/Changelog
* [WIKI]   EFT wiki "Trading" (rev 2026-09-09), sections Tax / Reputation:
           https://escapefromtarkov.fandom.com/wiki/Trading
           fee formula, Ti = Tr = 0.05, the 1.08 exponent, Intelligence Center
           level 3 = -30% commission (+0.3% per Hideout Management level, 45%
           at level 50), offer slots by flea rating, flea unlocks at PMC level
           15, trader buy-back multipliers (fixed per trader; only Ref varies
           by loyalty level: LL1 0.40, LL2/LL3 0.45, LL4 0.50).
* [API]    the-hideout/tarkov-api, resolvers/itemResolver.mjs ``fleaMarketFee``
           (the reference the fee test transcribes) and datasources/items.mjs
           (the flea entry in ``sellFor`` is ``lastLowPrice``; trader buy
           offers are hard-coded ``minTraderLevel: 1``).  Its flea settings
           (``fleaMarket { sellOfferFeeRate sellRequirementFeeRate
           foundInRaidRequired reputationLevels }``) come from the game's own
           globals, which the app fetches at refresh time (``rules``).
* [DM]     the-hideout/tarkov-data-manager, jobs/update-item-cache.mjs:
           trader sell prices are ``floor(basePrice * payRate of loyalty
           level 1)`` and Ref pays GP coins, so a Ref "priceRUB" is not cash.
"""
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# [PATCH][WIKI] 5 % since 1.1.0.0 (was 3 %).  Overridden at refresh time by the
# game's own value when tarkov.dev answers (rules['flea']['sellOfferFeeRate']).
FLEA_FEE_RATE = 0.05
FLEA_FEE_EXPONENT = 1.08            # [WIKI] / [API]
INTEL_CENTER_DISCOUNT_LEVEL = 3     # [WIKI] Intelligence Center level that unlocks the discount
INTEL_CENTER_DISCOUNT = 0.30        # [WIKI] 30 % off the commission ...
# ... scaled by (1 + 0.01 * Hideout Management level): 45 % at skill level 50. [API]

# Only flea when it beats the trader by at least this much per offer: an offer
# uses one of a handful of slots and sells later than a trader does.
FLEA_MIN_GAIN = 10000

# List price = just under the lowest current offer, never above the 24 h average:
FLEA_UNDERCUT = 0.01                # 1 % under the reference, so we are the cheapest listing
FLEA_LOW_OUTLIER = 0.60             # a lowest offer < 60 % of avg24h is a one-off cheap listing: price from 0.6 x avg
FLEA_HIGH_CAP = 1.10                # a lowest offer > 110 % of avg24h is a thin/inflated market: cap at 1.1 x avg

# [WIKI] offers at flea rating 0.2-6.99 (the common mid-game band), no Unheard
# Edition bonus.  The player's real number is a setting (flea_offer_slots).
DEFAULT_FLEA_SLOTS = 4

# The order of the trader tabs in game, so the sell list walks the traders
# left to right and each trader is visited once.
TRADER_ORDER = ['Prapor', 'Therapist', 'Fence', 'Skier', 'Peacekeeper',
                'Mechanic', 'Ragman', 'Jaeger', 'Ref']

# [WIKI] Ref's buy-back multiplier by loyalty level.  tarkov.dev prices Ref at
# LL1, so a player at a higher level gets payRate[level] / payRate[1] more.
REF_PAY_RATES = {1: 0.40, 2: 0.45, 3: 0.45, 4: 0.50}

# Sell-advice settings (data/settings.json); every key is optional there.
SELL_DEFAULTS = {
    'flea_requires_fir': None,        # None = follow tarkov.dev's fleaMarket.foundInRaidRequired, else True
    'flea_offer_slots': DEFAULT_FLEA_SLOTS,
    'flea_overflow': 'queue',         # flea offers beyond the slot count: 'queue' (list later) | 'trader' (sell now)
    'flea_min_gain': FLEA_MIN_GAIN,
    'intel_center_level': None,       # None = read from the Tasks & Hideout progress
    'hideout_management_level': 0,
    'skip_traders': ['Ref'],          # Ref pays GP coins, not roubles: never recommended unless removed here
    'trader_levels': {},              # e.g. {"Ref": 4}: scales Ref's price (the only trader whose pay rate varies)
}


def sell_setting(settings, key):
    """settings[key], falling back to SELL_DEFAULTS when absent (old settings
    files predate these keys)."""
    value = (settings or {}).get(key)
    return SELL_DEFAULTS[key] if value is None else value


# ---------------------------------------------------------------------------
# Context: settings + live rules, resolved once per scan
# ---------------------------------------------------------------------------

def _rate(value, default):
    """A fee rate as a fraction (tarkov.dev sends 0.05; tolerate 5)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v <= 0:
        return default
    return v / 100.0 if v > 1 else v


def _int(value, default, lo=0, hi=10 ** 9):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def make_context(settings=None, rules=None, intel_level=None):
    """Everything the price functions need, as one dict.

    ``rules`` is the optional ``prices_cache.json['rules']`` blob
    ({'flea': {...fleaMarket...}, 'traders': {name: {'pay_rates': {lvl: rate}}}});
    without it the documented constants apply.  ``intel_level`` is the
    Intelligence Center level read from the hideout progress; the
    ``intel_center_level`` setting overrides it.
    """
    settings = settings or {}
    rules = rules or {}
    flea = rules.get('flea') or {}

    fir = sell_setting(settings, 'flea_requires_fir')
    if fir is None:
        fir = flea.get('foundInRaidRequired')
    if fir is None:
        fir = True      # the live rule (kappaguide 2026-03-15, Steam 1.0 forum thread); Battlestate lifts it for events

    intel = sell_setting(settings, 'intel_center_level')
    if intel is None:
        intel = intel_level or 0

    pay_rates = {'ref': dict(REF_PAY_RATES)}
    for name, info in (rules.get('traders') or {}).items():
        rates = {int(k): float(v) for k, v in (info.get('pay_rates') or {}).items() if v}
        if rates:
            pay_rates[name.lower()] = rates

    return {
        'rules': rules,
        'fee_offer': _rate(flea.get('sellOfferFeeRate'), FLEA_FEE_RATE),
        'fee_requirement': _rate(flea.get('sellRequirementFeeRate'), FLEA_FEE_RATE),
        'intel': _int(intel, 0, 0, 3),
        'hm': _int(sell_setting(settings, 'hideout_management_level'), 0, 0, 51),
        'fir_required': bool(fir),
        'skip': {str(n).lower() for n in sell_setting(settings, 'skip_traders') or ()},
        'trader_levels': {str(k).lower(): _int(v, 1, 1, 4)
                          for k, v in (sell_setting(settings, 'trader_levels') or {}).items()},
        'pay_rates': pay_rates,
        'min_gain': _int(sell_setting(settings, 'flea_min_gain'), FLEA_MIN_GAIN),
        'slots': _int(sell_setting(settings, 'flea_offer_slots'), DEFAULT_FLEA_SLOTS, 0, 100),
        'overflow': 'trader' if sell_setting(settings, 'flea_overflow') == 'trader' else 'queue',
    }


def intel_center_level(stations, completed_level_ids):
    """Highest completed level of the Intelligence Center in the hideout
    progress (0 when none), from tarkov.dev's hideoutStations + the ids the
    Tasks & Hideout page marks completed."""
    done = set(completed_level_ids or ())
    best = 0
    for st in stations or ():
        if (st.get('name') or '').lower() != 'intelligence center':
            continue
        for lv in st.get('levels') or ():
            if lv.get('id') in done:
                best = max(best, lv.get('level') or 0)
    return best


# ---------------------------------------------------------------------------
# Flea fee and list price
# ---------------------------------------------------------------------------

def calc_flea_fee(base_price, listing_price, count=1, intel_level=0, hm_level=0,
                  offer_rate=FLEA_FEE_RATE, requirement_rate=FLEA_FEE_RATE,
                  require_all=False):
    """Flea listing fee in roubles, per the wiki Tax section / tarkov.dev's
    ``fleaMarketFee`` resolver (the test transcribes that resolver):

        VO * Ti * 4^PO * Q  +  VR * Tr * 4^PR * Q

    VO = base price * count / Q, VR = the asking price (per item, or for the
    whole offer with ``require_all``), PO = log10(VO/VR) and PR = log10(VR/VO),
    the one on the smaller side raised to 1.08.  Q is ``count`` unless
    ``require_all``.  Intelligence Center 3+ takes 30 % off, +0.3 % per
    Hideout Management level.  ``listing_price`` is per item unless
    ``require_all``.
    """
    if not base_price or not listing_price or count < 1:
        return 0
    q = 1 if require_all else count
    vo = base_price * (count / q)
    vr = listing_price
    po = math.log10(vo / vr)
    if vr < vo:
        po = po ** FLEA_FEE_EXPONENT
    pr = math.log10(vr / vo)
    if vr >= vo:
        pr = pr ** FLEA_FEE_EXPONENT
    fee = vo * offer_rate * (4 ** po) * q + vr * requirement_rate * (4 ** pr) * q
    if intel_level >= INTEL_CENTER_DISCOUNT_LEVEL:
        fee -= fee * INTEL_CENTER_DISCOUNT * (1 + 0.01 * hm_level)
    return math.floor(fee + 0.5)      # JS Math.round, as the resolver does


def price_420(target, max_drop=0.01):
    """Round ``target`` down to the nearest price ending in 420 - Joey's list
    flourish - but only when that costs at most ``max_drop`` of the price.

    The original version always floored to x420, which can drop a 2,400 price
    to 1,420 (-41 %) and was the only thing between "slightly under the lowest
    offer" and a 20-40 % dump on cheap items.  Bounded, it only ever fires on
    pricier items, where it is a rounding error.
    """
    if target < 1420:
        return target
    rem = target % 1000
    p = (target - rem + 420) if rem >= 420 else (target - rem - 580)
    return p if target - p <= target * max_drop else target


def flea_reference_price(item_data):
    """(price, note): the price to undercut for a fast flea sale, in roubles.

    The lowest CURRENT offer is ``lastLowPrice``; the flea entry of ``sellFor``
    carries the same number, which is all that pre-existing price caches have.
    ``low24hPrice`` is the minimum over the last 24 h - one troll offer that
    already sold drags it far below the live market, and it is what the old
    code used.  It is only a fallback now.  ``avg24hPrice`` is tarkov.dev's
    outlier-robust average, so it bounds the reference:

    * a lowest offer under FLEA_LOW_OUTLIER x avg is a one-off cheap listing
      (bait / mistake): price from that floor instead of dumping at it;
    * one over FLEA_HIGH_CAP x avg is an inflated thin market nobody pays:
      cap it, so the offer actually sells.
    """
    last = item_data.get('lastLowPrice') or 0
    if not last:
        for sf in item_data.get('sellFor') or ():
            if ((sf.get('vendor') or {}).get('name') or '').lower() == 'flea market':
                last = sf.get('priceRUB') or 0
                break
    avg = item_data.get('avg24hPrice') or 0
    ref = last or item_data.get('low24hPrice') or avg
    if not ref:
        return 0, ''
    note = ''
    if avg:
        lo, hi = round(avg * FLEA_LOW_OUTLIER), round(avg * FLEA_HIGH_CAP)
        if ref < lo:
            note = f'lowest offer {ref:,} is far below the 24h average {avg:,}; priced from {lo:,}'
            ref = lo
        elif ref > hi:
            note = f'lowest offer {ref:,} is far above the 24h average {avg:,}; capped at {hi:,}'
            ref = hi
    return int(ref), note


def flea_list_price(reference):
    """List price: FLEA_UNDERCUT under the reference (at least 1 rouble), then
    the bounded x420 ending."""
    if reference <= 0:
        return 0
    price = reference - max(1, round(reference * FLEA_UNDERCUT))
    return max(1, price_420(price))


# ---------------------------------------------------------------------------
# Trader choice
# ---------------------------------------------------------------------------

def trader_pay_factor(name, ctx):
    """How much more than tarkov.dev's LL1-based number this trader pays at
    the player's loyalty level (1.0 unless a level is configured; in practice
    only Ref differs between levels)."""
    ctx = ctx or {}
    key = (name or '').lower()
    level = (ctx.get('trader_levels') or {}).get(key)
    rates = (ctx.get('pay_rates') or {}).get(key)
    if level and rates and rates.get(1) and rates.get(level):
        return rates[level] / rates[1]
    return 1.0


def trader_offers(item_data, ctx=None):
    """Trader buy offers for one unit, best first: [{'name','price','currency'}].
    Skips the flea entry and any trader in the ``skip_traders`` setting."""
    skip = (ctx or {}).get('skip')
    skip = {'ref'} if skip is None else skip
    out = []
    for sf in item_data.get('sellFor') or ():
        vendor = (sf.get('vendor') or {}).get('name') or ''
        if not vendor or vendor.lower() == 'flea market' or vendor.lower() in skip:
            continue
        price = sf.get('priceRUB') or 0
        if price <= 0:
            continue
        out.append({'name': vendor, 'currency': sf.get('currency'),
                    'price': round(price * trader_pay_factor(vendor, ctx))})
    out.sort(key=lambda o: -o['price'])    # stable: equal prices keep tarkov.dev's order
    return out


def best_trader_offer(item_data, ctx=None):
    """The offer to take: the best regular trader; Fence (who buys everything
    at the worst rate, 0.24 of base price [WIKI]) only as a last resort."""
    offers = trader_offers(item_data, ctx)
    regular = [o for o in offers if o['name'].lower() != 'fence']
    pool = regular or offers
    return pool[0] if pool else None


def best_trader_price(item_data, ctx=None):
    """(trader_name, price_in_roubles) for the trader to sell one unit to."""
    offer = best_trader_offer(item_data, ctx)
    return (offer['name'], offer['price']) if offer else (None, 0)


# ---------------------------------------------------------------------------
# Recommendation for one stack
# ---------------------------------------------------------------------------

def flea_block_reason(item_data, fir, settings, rules=None):
    """Why this copy can't be listed on the flea, or None if it can.

    noFlea items are banned outright.  When the flea requires Found-in-Raid
    (the live rule; ``flea_requires_fir`` overrides what tarkov.dev reports,
    for events where Battlestate lifts it), a copy confidently read as non-FiR
    can't be listed.  An indeterminate FiR read (None) is not treated as a
    block - the row tells the player to check.
    """
    if 'noFlea' in (item_data.get('types') or ()):
        return 'banned from flea'
    if make_context(settings, rules)['fir_required'] and fir is False:
        return 'not FiR, cannot list on flea'
    return None


def sell_recommendation(item_data, flea_blocked=None, count=1, ctx=None):
    """Trader, flea and the pick for a stack of ``count`` identical items.

    Prices are per unit; ``gain`` is the whole stack's flea-over-trader margin.
    Flea wins only when it is allowed for this copy, its net after the fee
    beats the best trader by ``min_gain`` for the offer (or no trader buys it).
    """
    ctx = ctx or make_context()
    offer = best_trader_offer(item_data, ctx)
    trader_name = offer['name'] if offer else None
    trader_price = offer['price'] if offer else 0
    count = max(1, count or 1)

    rec = {
        'trader_name':     trader_name,
        'trader_price':    trader_price,
        'trader_currency': offer['currency'] if offer else None,
        'flea_list':       None,
        'flea_net':        None,
        'flea_fee':        None,
        'flea_ref':        None,
        'price_note':      '',
        'gain':            None,
        'recommend':       'trader',
        'reason':          '',
    }

    if flea_blocked:
        rec['reason'] = flea_blocked.capitalize()
        return rec
    ref, note = flea_reference_price(item_data)
    if not ref:
        rec['reason'] = 'No flea data'
        return rec

    base_price = item_data.get('basePrice') or 0
    flea_list = flea_list_price(ref)
    fee = calc_flea_fee(base_price, flea_list, 1, ctx['intel'], ctx['hm'],
                        ctx['fee_offer'], ctx['fee_requirement'])
    flea_net = flea_list - fee
    gain = (flea_net - trader_price) * count
    rec.update(flea_list=flea_list, flea_net=flea_net, flea_fee=fee, gain=gain,
               flea_ref=ref, price_note=note)

    min_gain = ctx['min_gain']
    shown = f'{trader_name} ' if trader_name else ''
    if flea_net <= 0:
        rec['reason'] = f'Flea fee {fee:,} eats the {flea_list:,} listing'
    elif not trader_price:
        rec['recommend'] = 'flea'
        rec['reason'] = 'No trader buys this'
    elif gain >= min_gain:
        rec['recommend'] = 'flea'
        rec['reason'] = f'+{gain:,} over {shown}after fees'
    else:
        rec['reason'] = f'Flea nets {gain:+,} vs {shown}- needs +{min_gain:,} to be worth an offer'
    return rec


# ---------------------------------------------------------------------------
# Ordering and flea offer slots
# ---------------------------------------------------------------------------

def order_for_selling(entries):
    """Sort sell entries into the order the player should sell them.

    Trader sales first, grouped by trader in TRADER_ORDER and, within a
    trader, by stack value (highest first); then the flea offers that fit in
    the player's slots, best margin over the trader first, then the flea queue
    (offers beyond the slots) the same way.  Ties keep their stash reading
    order.
    """
    def trader_rank(name):
        return TRADER_ORDER.index(name) if name in TRADER_ORDER else len(TRADER_ORDER)

    def flea_key(e):
        margin = e.get('gain')
        return (bool(e.get('flea_queue')), -(margin if margin is not None else (e.get('total') or 0)))

    traders = [e for e in entries if e['recommend'] == 'trader']
    flea    = [e for e in entries if e['recommend'] == 'flea']
    traders.sort(key=lambda e: (trader_rank(e['trader_name']), e['trader_name'] or '',
                                -(e.get('total') or 0)))
    flea.sort(key=flea_key)
    return traders + flea


def assign_flea_slots(entries, slots, overflow='queue'):
    """Mark which flea recommendations fit in the player's offer slots.

    One flea offer holds every copy of one item (unstackable copies stack in a
    single offer), so offers are counted per item.  Offers are ranked by how
    much the flea beats the trader by, because that is what a slot buys; the
    best ``slots`` offers get ``flea_slot`` 1..N, the rest ``flea_queue``.
    With overflow='trader' the queue is turned into trader sales right away
    (when a trader buys it).  Mutates and returns ``entries``.
    """
    groups = {}
    for e in entries:
        if e['recommend'] == 'flea':
            groups.setdefault(e.get('item_id') or id(e), []).append(e)
    ranked = sorted(groups.values(),
                    key=lambda g: -sum(e['gain'] if e.get('gain') is not None else (e.get('total') or 0)
                                       for e in g))
    for rank, group in enumerate(ranked):
        for e in group:
            e['flea_slots'] = slots
            e['flea_slot'] = rank + 1 if rank < slots else None
            e['flea_queue'] = rank >= slots
            if rank >= slots and overflow == 'trader' and e.get('trader_price'):
                e['recommend'] = 'trader'
                e['flea_queue'] = False
                e['total'] = e['trader_price'] * e.get('count', 1)
                e['reason'] = f'All {slots} flea offers are taken - sell to {e["trader_name"]} now'
    return entries


# ---------------------------------------------------------------------------
# Keep allocation: keep only what is still needed, sell the surplus
# ---------------------------------------------------------------------------

def remaining_needs(total_needed, fir_needed, have):
    """(need, fir_need): copies still short, and how many of those must be
    Found-in-Raid.  ``have`` copies (counted on the Tasks page) are assumed to
    cover the any-copy needs first, so the FiR part - the stricter one - is
    kept as large as possible (never sell a copy a hand-in may still want)."""
    any_total = max(0, total_needed - fir_needed)
    rem_any = max(0, any_total - have)
    rem_fir = max(0, fir_needed - max(0, have - any_total))
    return rem_any + rem_fir, rem_fir


def keep_need(prot):
    """(need, fir_need) of a protected entry; the legacy keep-list shape
    {'reason', 'fir_only'} means one copy, FiR when fir_only."""
    if 'need' in prot:
        return prot['need'], prot.get('fir_need', 0)
    return 1, 1 if prot.get('fir_only') else 0


def allocate_keep(copies, need, fir_need):
    """How many units of each scanned stack to keep.

    ``copies`` = [(count, fir)] for one item, in stash reading order; ``fir`` is
    True / False / None (None = could not tell, counted as possibly-FiR so it
    is never sold by mistake).  The FiR part of the need is covered by
    FiR-capable copies; the rest preferably by non-FiR copies, which are the
    ones that cannot be listed on the flea anyway.  Returns [kept_units].
    """
    kept = [0] * len(copies)
    fir_left = max(0, fir_need)
    any_left = max(0, need - fir_need)

    def take(i, want):
        n = min(copies[i][0] - kept[i], want)
        kept[i] += n
        return n

    for i, (_, fir) in enumerate(copies):             # FiR need: FiR-capable copies
        if fir is not False and fir_left:
            fir_left -= take(i, fir_left)
    for i, (_, fir) in enumerate(copies):             # any-copy need: non-FiR first
        if fir is False and any_left:
            any_left -= take(i, any_left)
    for i, (_, fir) in enumerate(copies):
        if fir is not False and any_left:
            any_left -= take(i, any_left)
    return kept


def describe_keep(prot, kept, stack):
    """One line saying why units are kept and how many of the stack."""
    need, fir_need = keep_need(prot)
    why = prot.get('why') or [prot.get('reason') or 'On keep list']
    shown = '; '.join(why[:3]) + (f'; +{len(why) - 3} more' if len(why) > 3 else '')
    fir = f', {fir_need} must be Found in Raid' if fir_need else ''
    split = f' - keep {kept} of this stack of {stack}, sell the other {stack - kept}' if kept < stack else ''
    return f'Still need {need}{fir}: {shown}{split}'


def _keep_reason(prot, single_n, groups, kept, stack):
    """The KEEP row's reason: the single-item need and/or each any-of objective it covers."""
    if not groups:
        return describe_keep(prot, kept, stack)
    parts = [describe_keep(prot, single_n, single_n)] if single_n else []
    parts += [f"Keep {n} for {g['label']} ({any_of_label(g)})" for n, g in groups]
    text = '; '.join(parts)
    if kept < stack:
        text += f' - keep {kept} of this stack of {stack}, sell the other {stack - kept}'
    return text


# ---------------------------------------------------------------------------
# The whole plan for one scan
# ---------------------------------------------------------------------------

_GEOMETRY = ('score', 'col', 'row', 'W', 'H', 'rotated', 'fir', 'px', 'py', 'pw', 'ph')


def _geometry(d):
    out = {k: d.get(k) for k in _GEOMETRY}
    out['rotated'] = bool(d.get('rotated', False))
    out['x'], out['y'] = d['px'] + 2, d['py'] + 2
    return out


def any_of_label(group, shown=6):
    """'any of: A, B, C' for an any-of group, long lists cut."""
    names = [i['name'] for i in group['items']]
    more = f', +{len(names) - shown} more' if len(names) > shown else ''
    return f"any of: {', '.join(names[:shown])}{more}"


def allocate_any_of(group, copies):
    """Which copies satisfy one any-of requirement ("hand over ``count`` of A / B / C").

    ``copies`` = [(units, fir, unit_value)] for every scanned stack of ANY item in the set,
    in reading order.  Up to ``group['count']`` units are kept IN TOTAL, taken from the
    cheapest-to-give-up copies first so the valuable ones stay on the sell list; a FiR
    requirement only accepts FiR-capable copies (``fir`` not False), confirmed-FiR before
    unknown on a value tie.  Returns [kept_units] aligned with ``copies``.
    """
    kept = [0] * len(copies)
    left = group['count']
    order = sorted((i for i, (_, fir, _) in enumerate(copies) if not group['fir'] or fir is not False),
                   key=lambda i: (copies[i][2], copies[i][1] is None, copies[i][1] is not False, i))
    for i in order:
        if left <= 0:
            break
        n = min(copies[i][0], left)
        kept[i] = n
        left -= n
    return kept


def plan_entries(detections, id_to_item, protected, settings, ctx, any_of=None):
    """Turn identified stash items into ordered sell rows and keep rows.

    ``detections`` carry item_id, count, fir, uncertain and their pixel rect
    (guns already removed).  Returns (sell_rows, keep_rows): sell rows are
    numbered 1..N in the order to act (traders in tab order, then flea offers
    by margin, then the flea queue); keep rows are numbered 'K'.  A stack that
    is only partly needed becomes a sell row for the surplus plus a keep row
    (``drawn`` False) for the rest.

    ``any_of`` = open objectives that accept any ONE of several items (see
    app.get_protected_plan).  They are allocated after the single-item needs, out of the
    copies those left over, across every scanned stack of every item in the set.
    """
    # An item the engine could not identify with certainty is NEVER sold: selling the wrong
    # thing cannot be undone.  It becomes a KEEP row that asks the player to check it.
    unsure = sorted((d for d in detections if d.get('uncertain') and id_to_item.get(d['item_id'])),
                    key=lambda r: (r.get('panel', 0), r['row'], r['col']))
    dets = sorted((d for d in detections if id_to_item.get(d['item_id']) and not d.get('uncertain')),
                  key=lambda r: (r.get('panel', 0), r['row'], r['col']))

    by_item = {}
    for i, d in enumerate(dets):
        if d['item_id'] in protected:
            by_item.setdefault(d['item_id'], []).append(i)
    kept = {}
    for item_id, idxs in by_item.items():
        need, fir_need = keep_need(protected[item_id])
        alloc = allocate_keep([(dets[i].get('count') or 1, dets[i].get('fir')) for i in idxs],
                              need, fir_need)
        kept.update(zip(idxs, alloc))
    kept_single = dict(kept)

    # any-of requirements: from what the single-item needs left, across the whole set
    group_keeps = {}                                   # det index -> [(units, group)]
    for g in any_of or ():
        ids = {a['id'] for a in g['items']}
        idxs = [i for i, d in enumerate(dets) if d['item_id'] in ids]
        copies = []
        for i in idxs:
            d = dets[i]
            fir = d.get('fir')
            rec = sell_recommendation(
                id_to_item[d['item_id']],
                flea_blocked=flea_block_reason(id_to_item[d['item_id']], fir, settings, ctx.get('rules')),
                count=1, ctx=ctx)
            unit = (rec['flea_net'] if rec['recommend'] == 'flea' else rec['trader_price']) or 0
            copies.append(((d.get('count') or 1) - kept.get(i, 0), fir, unit))
        for i, n in zip(idxs, allocate_any_of(g, copies)):
            if n:
                kept[i] = kept.get(i, 0) + n
                group_keeps.setdefault(i, []).append((n, g))

    sell, keep = [], []
    for i, d in enumerate(dets):
        item = id_to_item[d['item_id']]
        count = d.get('count') or 1
        k = kept.get(i, 0)
        prot = protected.get(d['item_id'])
        base = {'item_id': d['item_id'], 'matched_name': item['name'], **_geometry(d)}

        if k:
            keep.append({**base, 'num': 'K', 'count': k, 'stack': count, 'drawn': k >= count,
                         'recommend': 'keep', 'trader_name': None, 'trader_price': None,
                         'flea_list': None, 'flea_net': None,
                         'reason': _keep_reason(prot, kept_single.get(i, 0), group_keeps.get(i), k, count)})
        if k >= count:
            continue

        left = count - k
        need, fir_need = keep_need(prot) if prot else (0, 0)
        # Only FiR copies can satisfy this item's remaining need and this copy is not one:
        # it is safe to sell, but say why it was not kept.
        non_fir = bool(prot) and need == fir_need and d.get('fir') is False and not k
        rec = sell_recommendation(
            item, flea_blocked=flea_block_reason(item, d.get('fir'), settings, ctx.get('rules')),
            count=left, ctx=ctx)
        if non_fir:
            rec['reason'] += ' - not FIR, cannot be handed in'
        unit = rec['flea_net'] if rec['recommend'] == 'flea' else rec['trader_price']
        sell.append({**base, 'num': None, 'count': left, 'total': (unit or 0) * left,
                     'keep_n': k, 'uncertain': bool(d.get('uncertain')), 'non_fir_note': non_fir,
                     **rec})

    for d in unsure:
        item = id_to_item[d['item_id']]
        count = d.get('count') or 1
        keep.append({'item_id': d['item_id'], 'matched_name': item['name'], **_geometry(d),
                     'num': 'K', 'count': count, 'stack': count, 'drawn': True, 'check': True,
                     'uncertain': True, 'recommend': 'keep', 'trader_name': None,
                     'trader_price': None, 'flea_list': None, 'flea_net': None,
                     **({'provisional': True} if d.get('provisional') else {}),
                     'reason': ('Checking this item...' if d.get('provisional') else
                                f"Not sure this is {item['name']} - check it yourself (never auto-sold)")})

    assign_flea_slots(sell, ctx['slots'], ctx['overflow'])
    sell = order_for_selling(sell)
    for num, row in enumerate(sell, start=1):
        row['num'] = num
    return sell, keep
