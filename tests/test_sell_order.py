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
