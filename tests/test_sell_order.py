"""Sell advice: flea eligibility and the order the player sells in."""
import app


def _item(**kw):
    base = {'types': [], 'basePrice': 10000, 'low24hPrice': 0, 'avg24hPrice': 0, 'sellFor': []}
    base.update(kw)
    return base


def _entry(rec, trader, total):
    return {'recommend': rec, 'trader_name': trader, 'total': total}


def test_noflea_items_are_blocked():
    assert app.flea_block_reason(_item(types=['noFlea']), True, {}) == 'banned from flea'


def test_non_fir_blocked_only_when_rule_on_and_confident():
    on, off = {'flea_requires_fir': True}, {'flea_requires_fir': False}
    assert app.flea_block_reason(_item(), False, on)
    assert app.flea_block_reason(_item(), None, on) is None   # indeterminate never blocks
    assert app.flea_block_reason(_item(), True, on) is None
    assert app.flea_block_reason(_item(), False, off) is None


def test_blocked_item_goes_to_trader_even_when_flea_pays_more():
    it = _item(low24hPrice=90000, sellFor=[{'vendor': {'name': 'Therapist'}, 'priceRUB': 20000}])
    assert app.sell_recommendation(it)['recommend'] == 'flea'
    rec = app.sell_recommendation(it, flea_blocked='not FiR, cannot list on flea')
    assert rec['recommend'] == 'trader' and rec['trader_name'] == 'Therapist'


def test_item_no_trader_buys_goes_to_flea():
    assert app.sell_recommendation(_item(low24hPrice=50000))['recommend'] == 'flea'


def test_order_traders_in_tab_order_then_flea_by_profit():
    out = app.order_for_selling([
        _entry('flea', None, 5), _entry('trader', 'Ref', 1), _entry('trader', 'Prapor', 2),
        _entry('flea', None, 50), _entry('trader', 'Therapist', 9), _entry('trader', 'Prapor', 7),
    ])
    assert [(e['trader_name'], e['total']) for e in out] == [
        ('Prapor', 7), ('Prapor', 2), ('Therapist', 9), ('Ref', 1), (None, 50), (None, 5)]


def test_unknown_trader_sorts_after_known_ones():
    out = app.order_for_selling([_entry('trader', 'Lightkeeper', 99), _entry('trader', 'Jaeger', 1)])
    assert [e['trader_name'] for e in out] == ['Jaeger', 'Lightkeeper']


def test_guns_are_skipped_but_flares_and_knives_are_priced():
    assert app.is_unpriced_weapon({'types': ['gun']}) is True
    assert app.is_unpriced_weapon({'types': ['preset']}) is True
    assert app.is_unpriced_weapon({'types': ['gun', 'specialSlot']}) is False   # RSP-30 flare
    assert app.is_unpriced_weapon({'types': ['knife']}) is False
    assert app.is_unpriced_weapon(None, 'weapon') is True
    assert app.is_unpriced_weapon(None, 'barter') is False


def test_price_outage_falls_back_to_cache(monkeypatch, tmp_path):
    import json, time
    cache = tmp_path / 'prices.json'
    cache.write_text(json.dumps({'timestamp': time.time() - 10 ** 6, 'items': [{'id': 'x'}]}), encoding='utf-8')
    monkeypatch.setattr(app, 'PRICES_PATH', str(cache))

    def down():
        raise app.PriceFetchError('tarkov.dev HTTP 422: GraphQL server unavailable')
    monkeypatch.setattr(app, 'fetch_prices', down)
    out = app.get_prices()
    assert out['items'] == [{'id': 'x'}] and 'unavailable' in out['stale_error']


# ---------------------------------------------------------------------------
# Flea fee (sellcalc.calc_flea_fee) - the formula changed on 3 Aug 2026
# ---------------------------------------------------------------------------
import math

import pytest

import sellcalc


def _api_fee(base, price, count=1, intel=0, hm=0, ti=0.05, tr=0.05, require_all=False):
    """Line-by-line transcription of tarkov-api's `fleaMarketFee` resolver
    (resolvers/itemResolver.mjs, fetched 2026-10-01) - the reference."""
    q = 1 if require_all else count
    vo = base * (count / q)
    vr = price
    po = math.log10(vo / vr)
    if vr < vo:
        po = po ** 1.08
    pr = math.log10(vr / vo)
    if vr >= vo:
        pr = pr ** 1.08
    fee = vo * ti * 4.0 ** po * q + vr * tr * 4.0 ** pr * q
    if intel >= 3:
        discount = 0.3
        discount = discount + discount * hm * 0.01
        fee = fee - fee * discount
    return math.floor(fee + 0.5)     # JS Math.round


def test_fee_at_the_base_price_is_ten_percent_since_patch_1_1_0_0():
    # P = 0 on both sides, so the fee is VO*Ti + VR*Tr = 2 * 5 %.  It was 2 * 3 % before 3 Aug 2026.
    assert app.calc_flea_fee(10000, 10000) == 1000


def test_fee_worked_example_from_the_wiki_formula():
    # base 10,000 listed at 12,000: PO = log10(10/12) (VR >= VO, not powered),
    # PR = log10(12/10) ** 1.08 = 0.06469;  10000*.05*4**-0.07918 + 12000*.05*4**0.06469
    # = 448.0 + 656.3 = 1104
    assert app.calc_flea_fee(10000, 12000) == 1104
    # listed under the base price the 1.08 exponent moves to the offer term: 559.7 + 349.8
    assert app.calc_flea_fee(10000, 8000) == 909


@pytest.mark.parametrize('base,price', [(300, 250), (300, 900), (10000, 7000), (10000, 10000),
                                        (48000, 51000), (48000, 300000), (1200000, 900000)])
@pytest.mark.parametrize('count', [1, 7])
@pytest.mark.parametrize('intel,hm', [(0, 0), (2, 50), (3, 0), (3, 30), (3, 50)])
def test_fee_matches_the_tarkov_dev_resolver(base, price, count, intel, hm):
    assert app.calc_flea_fee(base, price, count, intel, hm) == _api_fee(base, price, count, intel, hm)


def test_fee_require_all_matches_the_resolver():
    # "Require for all items in offer": the asking price is for the whole stack
    assert sellcalc.calc_flea_fee(10000, 60000, count=3, require_all=True) == \
        _api_fee(10000, 60000, count=3, require_all=True)


def test_intel_center_three_takes_thirty_percent_and_hideout_management_adds_to_it():
    full = app.calc_flea_fee(10000, 12000)
    assert app.calc_flea_fee(10000, 12000, intel_level=2) == full             # level 3 is the unlock
    assert app.calc_flea_fee(10000, 12000, intel_level=3) == round(full * 0.70)
    assert app.calc_flea_fee(10000, 12000, intel_level=3, hm_level=50) == round(full * 0.55)   # 45 % off


def test_fee_scales_with_the_number_of_items_and_takes_the_rate_from_the_game():
    assert abs(app.calc_flea_fee(10000, 12000, count=5) - 5 * app.calc_flea_fee(10000, 12000)) <= 3   # rounding only
    old_rate = app.calc_flea_fee(10000, 10000, offer_rate=0.03, requirement_rate=0.03)
    assert old_rate == 600                                                  # the pre-patch 3 % world


def test_fee_degenerate_inputs_are_free_not_errors():
    assert app.calc_flea_fee(0, 5000) == 0 and app.calc_flea_fee(5000, 0) == 0


# ---------------------------------------------------------------------------
# Flea price: what to list at
# ---------------------------------------------------------------------------

def test_price_420_only_snaps_when_it_costs_under_one_percent():
    assert app.price_420(60000) == 59420           # 580 on 60k: a flourish
    assert app.price_420(2400) == 2400             # was 1420 (-41 %)
    assert app.price_420(31111) == 31111
    assert app.price_420(999) == 999


def test_list_price_is_just_under_the_reference_even_for_cheap_items():
    assert sellcalc.flea_list_price(31111) == 30800
    assert sellcalc.flea_list_price(2500) == 2475
    assert sellcalc.flea_list_price(100) == 99
    assert sellcalc.flea_list_price(10) == 9
    assert sellcalc.flea_list_price(1) == 1 and sellcalc.flea_list_price(0) == 0


def test_reference_is_the_current_lowest_offer_not_the_24h_low():
    # Tushonka in the cached data: a past 10k troll offer drags low24h to 28 % of the market
    it = _item(low24hPrice=10165, avg24hPrice=35757, lastLowPrice=31111)
    assert sellcalc.flea_reference_price(it) == (31111, '')


def test_reference_falls_back_to_the_flea_entry_of_sellfor_for_old_price_caches():
    it = _item(low24hPrice=10165, avg24hPrice=35757,
               sellFor=[{'vendor': {'name': 'Flea Market'}, 'priceRUB': 31111}])
    assert sellcalc.flea_reference_price(it)[0] == 31111


def test_reference_guards_a_lowball_outlier_and_an_inflated_floor():
    cheap, note = sellcalc.flea_reference_price(_item(lastLowPrice=9000, avg24hPrice=30000))
    assert cheap == 18000 and 'far below' in note              # 0.6 x avg, not a 70 % dump
    high, note = sellcalc.flea_reference_price(_item(lastLowPrice=60000, avg24hPrice=30000))
    assert high == 33000 and 'far above' in note               # 1.1 x avg
    assert sellcalc.flea_reference_price(_item(lastLowPrice=27000, avg24hPrice=30000)) == (27000, '')


def test_reference_without_current_offer_uses_the_24h_low_then_average():
    assert sellcalc.flea_reference_price(_item(low24hPrice=5000, avg24hPrice=6000))[0] == 5000
    assert sellcalc.flea_reference_price(_item(avg24hPrice=6000))[0] == 6000
    assert sellcalc.flea_reference_price(_item()) == (0, '')


# ---------------------------------------------------------------------------
# Trader choice and rules
# ---------------------------------------------------------------------------

def _sf(**prices):
    return [{'vendor': {'name': n}, 'priceRUB': p} for n, p in prices.items()]


def test_fence_is_only_a_last_resort():
    it = _item(sellFor=_sf(Fence=9000, Therapist=4000))
    assert app.best_trader_price(it) == ('Therapist', 4000)
    assert app.best_trader_price(_item(sellFor=_sf(Fence=9000))) == ('Fence', 9000)
    assert app.best_trader_price(_item()) == (None, 0)


def test_ref_pays_gp_so_it_is_skipped_unless_enabled():
    it = _item(sellFor=_sf(Ref=12000, Mechanic=9000))
    assert app.best_trader_price(it) == ('Mechanic', 9000)
    ctx = sellcalc.make_context({'skip_traders': []})
    assert app.best_trader_price(it, ctx) == ('Ref', 12000)


def test_ref_price_scales_with_the_configured_loyalty_level():
    # tarkov.dev prices Ref at LL1 (0.40); the wiki's LL4 rate is 0.50, so +25 %
    it = _item(sellFor=_sf(Ref=10000))
    ctx = sellcalc.make_context({'skip_traders': [], 'trader_levels': {'Ref': 4}})
    assert app.best_trader_price(it, ctx) == ('Ref', 12500)
    ctx2 = sellcalc.make_context({'skip_traders': [], 'trader_levels': {'Ref': 2}})
    assert app.best_trader_price(it, ctx2) == ('Ref', 11250)
    # the live pay rates from tarkov.dev win over the built-in table
    rules = {'traders': {'Ref': {'pay_rates': {'1': 0.4, '4': 0.6}}}}
    ctx3 = sellcalc.make_context({'skip_traders': [], 'trader_levels': {'Ref': 4}}, rules)
    assert app.best_trader_price(it, ctx3) == ('Ref', 15000)


def test_other_traders_do_not_change_with_loyalty_level():
    it = _item(sellFor=_sf(Prapor=5000))
    ctx = sellcalc.make_context({'trader_levels': {'Prapor': 4}})
    assert app.best_trader_price(it, ctx) == ('Prapor', 5000)


def test_equal_trader_prices_keep_tarkov_dev_order():
    assert app.best_trader_price(_item(sellFor=_sf(Skier=100, Prapor=100)))[0] == 'Skier'


def test_context_reads_live_rates_and_the_fir_rule_from_the_cached_rules():
    rules = {'flea': {'sellOfferFeeRate': 0.07, 'sellRequirementFeeRate': 0.07, 'foundInRaidRequired': False}}
    ctx = sellcalc.make_context({}, rules)
    assert (ctx['fee_offer'], ctx['fee_requirement'], ctx['fir_required']) == (0.07, 0.07, False)
    assert sellcalc.make_context({})['fee_offer'] == 0.05                       # documented fallback
    assert sellcalc.make_context({}, {'flea': {'sellOfferFeeRate': 5}})['fee_offer'] == 0.05   # percent tolerated
    # an explicit setting beats tarkov.dev (events lift the rule before the API notices)
    assert sellcalc.make_context({'flea_requires_fir': True}, rules)['fir_required'] is True


def test_fir_rule_follows_tarkov_dev_unless_the_setting_says_otherwise():
    free = {'flea': {'foundInRaidRequired': False}}
    assert app.flea_block_reason(_item(), False, {}, free) is None
    assert app.flea_block_reason(_item(), False, {'flea_requires_fir': True}, free)
    assert app.flea_block_reason(_item(), False, {}, None)          # no data: the live rule, FiR only


def test_intel_center_level_comes_from_hideout_progress():
    stations = [{'name': 'Intelligence Center', 'levels': [{'id': 'ic-1', 'level': 1},
                                                           {'id': 'ic-2', 'level': 2},
                                                           {'id': 'ic-3', 'level': 3}]},
                {'name': 'Stash', 'levels': [{'id': 's-4', 'level': 4}]}]
    assert sellcalc.intel_center_level(stations, ['ic-1', 'ic-2', 's-4']) == 2
    assert sellcalc.intel_center_level(stations, []) == 0
    ctx = sellcalc.make_context({}, None, intel_level=3)
    assert ctx['intel'] == 3
    assert sellcalc.make_context({'intel_center_level': 0}, None, intel_level=3)['intel'] == 0


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_flea_decision_uses_the_whole_stack_margin_not_one_unit():
    it = _item(basePrice=10000, lastLowPrice=12000, avg24hPrice=12000,
               sellFor=_sf(Prapor=9000))
    one = app.sell_recommendation(it, count=1)
    assert one['recommend'] == 'trader'                       # ~ +1,800 per unit: not worth an offer
    assert one['gain'] < 10000
    stack = app.sell_recommendation(it, count=20)
    assert stack['recommend'] == 'flea' and stack['gain'] == one['gain'] * 20
    assert stack['flea_list'] == 11880 and stack['flea_fee'] == app.calc_flea_fee(10000, 11880)


def test_flea_recommendation_reports_the_fee_and_respects_the_min_gain_setting():
    it = _item(basePrice=10000, lastLowPrice=30000, avg24hPrice=30000, sellFor=_sf(Prapor=9000))
    assert app.sell_recommendation(it)['recommend'] == 'flea'
    strict = sellcalc.make_context({'flea_min_gain': 10 ** 6})
    assert app.sell_recommendation(it, ctx=strict)['recommend'] == 'trader'


def test_intel_center_makes_a_borderline_item_worth_listing():
    it = _item(basePrice=40000, lastLowPrice=60000, avg24hPrice=60000, sellFor=_sf(Prapor=48000))
    plain = app.sell_recommendation(it)
    cheap = app.sell_recommendation(it, ctx=sellcalc.make_context({'intel_center_level': 3,
                                                                   'hideout_management_level': 50}))
    assert cheap['flea_fee'] < plain['flea_fee'] and cheap['gain'] > plain['gain']


def test_a_flea_fee_larger_than_the_listing_never_recommends_flea():
    it = _item(basePrice=10 ** 6, lastLowPrice=100, avg24hPrice=100, sellFor=_sf(Prapor=50))
    rec = app.sell_recommendation(it)
    assert rec['recommend'] == 'trader' and 'fee' in rec['reason']


def test_outlier_note_is_carried_to_the_recommendation():
    it = _item(basePrice=20000, lastLowPrice=9000, avg24hPrice=30000, sellFor=_sf(Prapor=9000))
    rec = app.sell_recommendation(it)
    assert 'far below' in rec['price_note'] and rec['flea_ref'] == 18000


# ---------------------------------------------------------------------------
# Flea offer slots
# ---------------------------------------------------------------------------

def _flea(item_id, gain, count=1, name='Prapor', trader_price=1000):
    return {'recommend': 'flea', 'item_id': item_id, 'gain': gain, 'total': gain + trader_price * count,
            'count': count, 'trader_name': name, 'trader_price': trader_price, 'reason': ''}


def test_slots_go_to_the_biggest_margin_over_the_trader_and_the_rest_queue():
    rows = [_flea('a', 11000), _flea('b', 50000), _flea('c', 20000), _flea('d', 30000)]
    sellcalc.assign_flea_slots(rows, 2)
    by = {r['item_id']: r for r in rows}
    assert (by['b']['flea_slot'], by['d']['flea_slot']) == (1, 2)
    assert by['c']['flea_slot'] is None and by['c']['flea_queue'] and by['a']['flea_queue']
    assert not by['b']['flea_queue'] and by['b']['flea_slots'] == 2
    assert [r['item_id'] for r in app.order_for_selling(rows)] == ['b', 'd', 'c', 'a']


def test_copies_of_one_item_share_a_single_offer():
    rows = [_flea('a', 20000), _flea('a', 20000), _flea('b', 30000), _flea('c', 15000)]
    sellcalc.assign_flea_slots(rows, 2)
    queued = [r['item_id'] for r in rows if r['flea_queue']]
    assert queued == ['c']          # a (two stacks, one offer) and b fit; c waits


def test_overflow_can_sell_to_the_trader_now_instead_of_queueing():
    rows = [_flea('a', 50000), _flea('b', 11000, count=2, trader_price=700)]
    sellcalc.assign_flea_slots(rows, 1, overflow='trader')
    b = rows[1]
    assert b['recommend'] == 'trader' and b['total'] == 1400 and not b['flea_queue']
    assert 'taken' in b['reason']
    out = app.order_for_selling(rows)
    assert [e['recommend'] for e in out] == ['trader', 'flea']


def test_overflow_with_no_trader_stays_on_the_flea_queue():
    rows = [_flea('a', 50000), _flea('b', 11000, trader_price=0)]
    sellcalc.assign_flea_slots(rows, 1, overflow='trader')
    assert rows[1]['recommend'] == 'flea' and rows[1]['flea_queue']


def test_zero_slots_queues_everything():
    rows = [_flea('a', 50000)]
    sellcalc.assign_flea_slots(rows, 0)
    assert rows[0]['flea_queue'] and rows[0]['flea_slot'] is None


# ---------------------------------------------------------------------------
# Keep: only what is still needed
# ---------------------------------------------------------------------------

def test_remaining_needs_subtract_set_aside_copies_from_the_any_copy_part_first():
    assert sellcalc.remaining_needs(5, 0, 0) == (5, 0)
    assert sellcalc.remaining_needs(5, 0, 3) == (2, 0)
    assert sellcalc.remaining_needs(5, 5, 5) == (0, 0)
    assert sellcalc.remaining_needs(5, 2, 4) == (1, 1)        # 3 any-copy needs eat 3 haves, FiR loses 1
    assert sellcalc.remaining_needs(5, 5, 2) == (3, 3)


def test_keep_exactly_what_is_needed_and_sell_the_surplus():
    assert sellcalc.allocate_keep([(7, True)], 5, 0) == [5]
    assert sellcalc.allocate_keep([(3, True), (4, True)], 5, 0) == [3, 2]
    assert sellcalc.allocate_keep([(2, True)], 5, 0) == [2]            # short: keep all, nothing to sell
    assert sellcalc.allocate_keep([(3, True)], 0, 0) == [0]


def test_non_fir_copies_cannot_cover_a_fir_need_and_are_kept_first_for_any_copy_needs():
    assert sellcalc.allocate_keep([(1, False)], 1, 1) == [0]                  # sell it; a FiR copy must be found
    assert sellcalc.allocate_keep([(1, False), (1, True)], 1, 1) == [0, 1]
    # any-copy need: keep the non-FiR one (it cannot go on the flea anyway), sell the FiR one
    assert sellcalc.allocate_keep([(1, True), (1, False)], 1, 0) == [0, 1]
    assert sellcalc.allocate_keep([(1, None)], 1, 1) == [1]                   # unknown FiR is never sold by mistake


def _det(item_id, count=1, fir=True, col=0, row=0):
    return {'item_id': item_id, 'count': count, 'fir': fir, 'col': col, 'row': row, 'panel': 0,
            'px': col * 60, 'py': row * 60, 'pw': 60, 'ph': 60, 'W': 1, 'H': 1, 'score': 90}


def _plan(dets, items, protected, settings=None):
    settings = settings or {}
    return sellcalc.plan_entries(dets, items, protected, settings, sellcalc.make_context(settings))


def test_plan_splits_a_partly_needed_stack_into_a_sell_row_and_a_keep_row():
    items = {'salewa': _item(name='Salewa', sellFor=_sf(Therapist=7000))}
    prot = {'salewa': {'reason': 'Needed: x', 'fir_only': False, 'need': 3, 'fir_need': 0,
                       'why': ['Therapist - Operation Aquarius x2', 'Medstation L2 x1']}}
    sell, keep = _plan([_det('salewa', count=5)], items, prot)
    assert [(r['count'], r['keep_n'], r['total'], r['num']) for r in sell] == [(2, 3, 14000, 1)]
    assert len(keep) == 1 and keep[0]['count'] == 3 and keep[0]['stack'] == 5 and keep[0]['num'] == 'K'
    assert keep[0]['drawn'] is False                           # the sell row marks the cell on the image
    assert 'Still need 3' in keep[0]['reason'] and 'Operation Aquarius' in keep[0]['reason']
    assert 'sell the other 2' in keep[0]['reason']


def test_plan_keeps_a_fully_needed_stack_and_sells_everything_else():
    items = {'a': _item(name='A', sellFor=_sf(Prapor=3000)), 'b': _item(name='B', sellFor=_sf(Prapor=5000))}
    prot = {'a': {'reason': 'On keep list', 'fir_only': True}}              # legacy one-copy shape
    sell, keep = _plan([_det('a', 1, True, 0), _det('b', 2, True, 1)], items, prot)
    assert [r['matched_name'] for r in sell] == ['B'] and sell[0]['total'] == 10000
    assert keep[0]['matched_name'] == 'A' and keep[0]['drawn'] is True


def test_plan_sells_a_non_fir_copy_of_a_fir_only_need_with_the_reason():
    items = {'a': _item(name='A', sellFor=_sf(Prapor=3000))}
    prot = {'a': {'reason': 'On keep list', 'fir_only': True}}
    sell, keep = _plan([_det('a', 1, False)], items, prot)
    assert keep == [] and sell[0]['non_fir_note'] and 'not FIR' in sell[0]['reason']


def test_plan_numbers_rows_in_selling_order_and_marks_the_flea_queue():
    items = {
        't': _item(name='T', sellFor=_sf(Therapist=6000)),
        'f1': _item(name='F1', basePrice=40000, lastLowPrice=90000, avg24hPrice=90000, sellFor=_sf(Prapor=1000)),
        'f2': _item(name='F2', basePrice=40000, lastLowPrice=60000, avg24hPrice=60000, sellFor=_sf(Prapor=1000)),
    }
    dets = [_det('f2', col=0), _det('t', col=1), _det('f1', col=2)]
    sell, _ = _plan(dets, items, {}, {'flea_offer_slots': 1})
    assert [(r['num'], r['matched_name']) for r in sell] == [(1, 'T'), (2, 'F1'), (3, 'F2')]
    assert [r['flea_queue'] for r in sell if r['recommend'] == 'flea'] == [False, True]
    assert sell[1]['flea_slots'] == 1


def test_plan_skips_unknown_items_and_handles_empty_input():
    assert _plan([], {}, {}) == ([], [])
    assert _plan([_det('ghost')], {}, {}) == ([], [])


# ---------------------------------------------------------------------------
# app.py wiring: protected ids from task/hideout data, price refresh rules
# ---------------------------------------------------------------------------

def _tasks_cache():
    def obj(kind, item_id, count, fir=False):
        return {'id': f'{kind}-{item_id}', 'type': kind, 'count': count, 'foundInRaid': fir,
                'item': {'id': item_id, 'name': item_id, 'shortName': item_id}}
    return {'timestamp': 0, 'tasks': [
        {'id': 't1', 'name': 'Gunsmith', 'kappaRequired': True, 'trader': {'name': 'Mechanic'},
         'objectives': [obj('giveItem', 'bolt', 3), obj('giveItem', 'fir_only', 2, True)]},
        {'id': 't2', 'name': 'Planter', 'kappaRequired': True, 'trader': {'name': 'Skier'},
         'objectives': [obj('plantItem', 'camera', 2), obj('findItem', 'found_only', 9)]},
        {'id': 't3', 'name': 'Done Task', 'kappaRequired': True, 'trader': {'name': 'Prapor'},
         'objectives': [obj('giveItem', 'finished', 4)]},
    ], 'hideoutStations': [{'id': 'st', 'name': 'Medstation', 'levels': [
        {'id': 'st-1', 'level': 1, 'itemRequirements': [{'count': 2, 'item': {'id': 'bolt', 'name': 'bolt', 'shortName': 'bolt'}}]}]}]}


def _protect(monkeypatch, tmp_path, progress, keep_list=None, price_idx=None):
    import json
    prog = tmp_path / 'progress.json'
    prog.write_text(json.dumps(progress), encoding='utf-8')
    monkeypatch.setattr(app, 'PROGRESS_PATH', str(prog))
    monkeypatch.setattr(app, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))
    monkeypatch.setattr(app, 'get_tasks', lambda allow_fetch=True: _tasks_cache())
    return app.get_protected_ids(keep_list or {'categories': []}, price_idx or {})


def test_protected_ids_count_what_is_still_needed_across_tasks_hideout_and_planting(monkeypatch, tmp_path):
    p = _protect(monkeypatch, tmp_path, {'completed_tasks': ['t3'], 'completed_hideout': [], 'have': {}})
    assert (p['bolt']['need'], p['bolt']['fir_need'], p['bolt']['fir_only']) == (5, 0, False)   # 3 task + 2 hideout
    assert any('Gunsmith' in w for w in p['bolt']['why']) and any('Medstation' in w for w in p['bolt']['why'])
    assert (p['fir_only']['need'], p['fir_only']['fir_need'], p['fir_only']['fir_only']) == (2, 2, True)
    assert p['camera']['need'] == 2                                  # consumed by planting
    assert 'found_only' not in p                                     # findItem is not handed over
    assert 'finished' not in p                                       # completed task: free to sell


def test_protected_ids_subtract_set_aside_copies_and_drop_finished_hideout_levels(monkeypatch, tmp_path):
    p = _protect(monkeypatch, tmp_path, {'completed_tasks': [], 'completed_hideout': ['st-1'], 'have': {'bolt': 1}})
    assert p['bolt']['need'] == 2                                    # 3 for the task, 1 set aside
    assert any('set aside' in w for w in p['bolt']['why'])
    p = _protect(monkeypatch, tmp_path, {'completed_tasks': ['t1'], 'completed_hideout': ['st-1'], 'have': {}})
    assert 'bolt' not in p and 'fir_only' not in p


def test_protected_ids_add_keep_list_and_task_needs_together(monkeypatch, tmp_path):
    keep = {'categories': [{'id': 'kappa', 'label': 'Collector', 'items': [
        {'id': 'k1', 'name': 'bolt', 'aliases': [], 'acquired': False}]}]}
    idx = {'bolt': {'id': 'bolt', 'name': 'bolt'}}
    p = _protect(monkeypatch, tmp_path, {'completed_tasks': [], 'completed_hideout': [], 'have': {}},
                 keep, idx)['bolt']
    assert (p['need'], p['fir_need'], p['fir_only']) == (6, 1, False)    # 1 FiR Collector + 3 + 2 any-copy


def test_parse_sell_rules_compacts_the_tarkov_dev_response():
    data = {'fleaMarket': {'sellOfferFeeRate': 0.05, 'foundInRaidRequired': True},
            'traders': [{'name': 'Ref', 'currency': {'shortName': 'GP'},
                         'levels': [{'level': 1, 'payRate': 0.4}, {'level': 4, 'payRate': 0.5}]},
                        {'name': 'Nobody', 'currency': None, 'levels': []}]}
    rules = app.parse_sell_rules(data)
    assert rules['flea']['sellOfferFeeRate'] == 0.05
    assert rules['traders'] == {'Ref': {'currency': 'GP', 'pay_rates': {1: 0.4, 4: 0.5}}}
    assert app.parse_sell_rules(None) is None and app.parse_sell_rules({'traders': []}) is None


def test_rules_outage_does_not_stop_a_price_refresh_and_keeps_the_last_good_rules(monkeypatch, tmp_path):
    import json
    cache = tmp_path / 'prices.json'
    cache.write_text(json.dumps({'timestamp': 1, 'items': [], 'rules': {'flea': {'sellOfferFeeRate': 0.05}}}),
                     encoding='utf-8')
    monkeypatch.setattr(app, 'PRICES_PATH', str(cache))

    class Resp:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def post(url, json=None, timeout=None):
        if 'fleaMarket' in json['query']:
            raise app.http_requests.ConnectionError('down')
        return Resp({'data': {'items': [
            {'id': f'x{n}', 'name': 'X', 'shortName': 'X', 'width': 1, 'height': 1,
             'baseImageLink': 'https://img/x', 'sellFor': [{'vendor': {'name': 'Prapor'}, 'priceRUB': 5}]}
            for n in range(1200)]}})      # a complete-looking data set (a handful of items is refused as partial)
    monkeypatch.setattr(app.http_requests, 'post', post)
    out = app.fetch_prices_graphql()
    assert out['items'][0]['id'] == 'x0'
    assert out['rules'] == {'flea': {'sellOfferFeeRate': 0.05}}              # last good rules survive
    assert json.loads(cache.read_text(encoding='utf-8'))['rules'] == out['rules']


def test_dogtags_are_skipped_but_the_dogtag_case_is_priced():
    assert app.skip_badge({'name': 'Dogtag USEC', 'types': ['barter', 'noFlea']}) == 'TAG'
    assert app.skip_badge({'name': 'Dogtag BEAR', 'types': ['barter', 'noFlea']}) == 'TAG'
    assert app.skip_badge({'name': 'Dogtag case', 'types': ['container', 'noFlea']}) is None
    assert app.skip_badge({'name': 'Colt M4A1', 'types': ['gun']}) == 'GUN'
    assert app.skip_badge({'name': 'Salewa', 'types': ['meds']}) is None
