"""Items named on the stash screen are recognised again on the trader sell tab (dimmed or price-tagged)."""
import os

import cv2
import pytest

from identify.seen import SeenItems, signature

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
PITCH = 84.0

# Row 0 of both crops (1440p, the same 10 stash cells): the character screen, and Therapist's sell tab
# a moment later, where the items she buys carry a price tag and the rest are dimmed.
ROW0 = ['Balaclava (Fear)', 'Pack of sugar', 'Can of duck pate', 'SSD drive', 'TerraGroup Labs keycard (Black)',
        'Boreas engine room keycard', 'Boreas crew quarters keycard', 'Compartment C-1 keycard',
        'USEC cottage room key', 'Roubles']
CQ = 6                   # left out of the memory: its look-alike (ER, col 5) must not name it


@pytest.fixture(scope='module')
def imgs():
    a = cv2.imread(os.path.join(FIX, 'seen_stash.png'))
    b = cv2.imread(os.path.join(FIX, 'seen_trader.png'))
    assert a is not None and b is not None
    return a, b


def _sig(img, col, row=0, w=1, h=1):
    return signature(img, (col * PITCH, row * PITCH, w * PITCH, h * PITCH), w, h, PITCH, PITCH)


@pytest.fixture(scope='module')
def memory(imgs):
    stash, _ = imgs
    m = SeenItems()
    for col, name in enumerate(ROW0):
        if col != CQ:
            m.add(_sig(stash, col), 1, 1, {'item_id': name, 'name': name})
    for col, name in ((0, 'Euros'), (2, 'Emergency Water Ration'), (3, 'Emergency Water Ration')):
        m.add(_sig(stash, col, 1), 1, 1, {'item_id': name, 'name': name})
    return m


@pytest.mark.parametrize('col', [c for c in range(len(ROW0)) if c != CQ])
def test_trader_tile_is_named_after_the_stash_picture(imgs, memory, col):
    hit = memory.lookup(_sig(imgs[1], col), 1, 1)
    assert hit is not None, ROW0[col]
    assert hit[0]['name'] == ROW0[col]


def test_look_alike_that_was_not_seen_is_left_alone(imgs, memory):
    assert memory.lookup(_sig(imgs[1], CQ), 1, 1) is None


def test_other_footprints_are_not_compared(imgs, memory):
    assert memory.lookup(_sig(imgs[1], 1, w=2), 2, 1) is None


def test_duplicates_of_one_item_do_not_block_each_other(imgs, memory):
    hit = memory.lookup(_sig(imgs[1], 2, 1), 1, 1)
    assert hit is not None and hit[0]['name'] == 'Emergency Water Ration'
