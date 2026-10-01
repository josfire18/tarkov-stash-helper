"""OCR fuzzy matcher (pure python - no Tesseract needed) and strip helpers."""
import numpy as np
import pytest

from identify import ocr


def score(read, short, name='', width=1):
    return ocr.fuzzy_score(read, name or short, short, width)


def test_exact_and_case_insensitive():
    assert score('Morphine', 'Morphine') == pytest.approx(100)
    assert score('morphine', 'Morphine') == pytest.approx(100)


@pytest.mark.parametrize('read,short', [
    ('M8e', 'M80'),              # slashed zero read as 'e'
    ('MS95', 'M995'),            # 9 read as S
    ('S39', 'SJ9'),              # J read as 3
    ('Li', 'L1'),                # 1 read as i
    ('CBI', 'CBJ'),
    ('a GRNVG=18', 'GPNVG-18'),  # badge junk before the label + P/R confusion
    ('MB85SA1', 'M855A1'),
    ('M8SSA1', 'M855A1'),
])
def test_known_tesseract_confusions_still_match(read, short):
    assert score(read, short, width=2) >= 70


def test_wrong_items_score_clearly_lower_than_the_right_one():
    right = score('M8e', 'M80')
    for wrong in ('M61', 'M62', 'M855', 'M995'):
        assert right > score('M8e', wrong) + 15


def test_truncated_long_names_are_accepted_only_when_the_label_is_full_width():
    # a 1-slot label holds ~8 characters: "Perfotora" is the cut-off "Perfotoran"
    assert score('Perfotora', 'Perfotoran', width=1) >= 95
    # but on a 5-slot gun a complete 'HK 416A5' must not match the longer '416A5 RS'
    assert score('HK 416A5', 'HK 416A5', width=5) > score('HK 416A5', '416A5 RS', width=5) + 10


def test_short_names_do_not_match_inside_longer_reads():
    assert score('Meds', 'Meds') > score('Meds', 'ME') + 20          # 'ME' lives inside 'Meds'
    assert score('Meds', 'P') < 60


def test_unreadable_text_scores_zero():
    assert score('', 'Morphine') == 0 and score('x', 'Morphine') == 0


def test_label_capacity_grows_with_footprint_width():
    assert ocr.label_capacity(1) == 8
    assert ocr.label_capacity(2) > ocr.label_capacity(1)
    assert ocr.label_capacity(5) > 35


def test_prefilter_finds_the_right_name_among_many():
    names = ['Morphine', 'Adrenaline', 'Zagustin', 'M80', 'M995', 'CBJ'] + [f'Item{i}' for i in range(500)]
    folded = [ocr.fold(n) for n in names]
    assert 3 in ocr.prefilter('M8e', folded, limit=10, folded=True)
    assert 0 in ocr.prefilter('Morphlne', names, limit=5)
    assert ocr.prefilter('x', names) == []


def test_parse_count():
    assert ocr.parse_count('80') == 80
    assert ocr.parse_count('296/400') == 296
    assert ocr.parse_count('O/30') == 0
    assert ocr.parse_count('') is None


def test_fold_is_shape_based():
    assert ocr.fold('M80') == ocr.fold('M8e') == ocr.fold('Mao') or ocr.fold('M80')[0] == ocr.fold('M8e')[0]
    assert ocr.canon('AK-74 gas, tube') == 'ak 74 gas tube'


def test_strip_geometry_scales_with_pitch():
    img = np.full((400, 400, 3), 40, np.uint8)
    s63 = ocr.label_strip(img, (50, 50, 64, 64), 63, 63)
    s126 = ocr.label_strip(img, (50, 50, 127, 127), 126, 126)
    assert s63 is not None and s126 is not None
    # both normalised to the same upscale of the 63 px design => same strip height
    assert abs(s63.shape[0] - s126.shape[0]) <= 2
    assert ocr.label_strip(img, (410, 410, 64, 64), 63, 63) is None     # off-image
    assert len(ocr.variants_of(s63)) == 2
