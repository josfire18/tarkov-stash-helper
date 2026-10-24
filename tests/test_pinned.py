"""Pinned / locked items (diagonal stripes over the cell, locked ones also a padlock) are never scanned."""
import os

import cv2
import pytest

from identify.pinned import is_pinned

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'pinned_stash.png')
PITCH = 69.2
ORIGIN = (2, 2)          # first cell line inside the crop

# (name, col, row, w, h) in the crop of a real stash screenshot (3 rows x 7 columns)
PINNED = [('SP-6 (locked)', 0, 0, 1, 1), ('M67 (locked)', 1, 0, 1, 1), ('Dogtags case', 2, 0, 1, 1),
          ('Key tool', 3, 0, 1, 1), ('MRC', 0, 1, 2, 2)]
FREE = [('CS gas', 4, 0, 1, 1), ('CALOK-B', 2, 1, 1, 1), ('GF MOD3', 3, 1, 1, 1), ('GF MLOK', 4, 1, 1, 1),
        ('Cylinder 12ga', 5, 1, 2, 1), ('RK-3', 2, 2, 1, 1), ('X-17', 3, 2, 1, 1), ('SP-81', 4, 2, 2, 1),
        ('EC', 6, 2, 1, 1)]


@pytest.fixture(scope='module')
def img():
    im = cv2.imread(FIXTURE)
    assert im is not None
    return im


def _rect(col, row, w, h):
    return (ORIGIN[0] + col * PITCH, ORIGIN[1] + row * PITCH, w * PITCH, h * PITCH)


@pytest.mark.parametrize('name,col,row,w,h', PINNED)
def test_pinned_items_are_detected(img, name, col, row, w, h):
    assert is_pinned(img, _rect(col, row, w, h), w, h, PITCH, PITCH), name


@pytest.mark.parametrize('name,col,row,w,h', FREE)
def test_normal_items_are_not_flagged(img, name, col, row, w, h):
    assert not is_pinned(img, _rect(col, row, w, h), w, h, PITCH, PITCH), name


def test_tiny_or_empty_rect_is_not_pinned(img):
    assert not is_pinned(img, (0, 0, 0, 0), 1, 1, PITCH, PITCH)
    assert not is_pinned(img, (0, 0, 10, 10), 1, 1, PITCH, PITCH)
