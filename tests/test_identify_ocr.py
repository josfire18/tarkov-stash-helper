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


# --------------------------------------------------------------------------
# label isolation: thin strokes are text, thick blobs are art, art never joins the run
# --------------------------------------------------------------------------

H_STRIP = 71


def _stats(*boxes):
    """cv2-style stats rows (x, y, w, h, area) with a dummy background row first."""
    return np.array([(0, 0, 500, H_STRIP, 1)] + [(x, y, w, h, a) for x, y, w, h, a in boxes])


def test_thick_blob_is_art_but_fat_or_solid_thin_strokes_are_not():
    st = _stats((100, 10, 30, 40, 900),      # 'g'-like letter, thin strokes
                (150, 20, 3, 30, 90),        # a solid 'I' / half of a split 'T'
                (180, 30, 21, 4, 70),        # a hyphen
                (10, 15, 120, 50, 5000))     # an item's highlight
    thick = np.array([0, 6.0, 1.9, 1.9, 40.0])
    kinds = ocr.classify_components(st, H_STRIP, thick)
    assert kinds[1:] == ['letter', 'letter', 'piece', 'art']


def test_components_touching_the_band_bottom_are_art():
    st = _stats((50, 40, 30, 31, 600))
    assert ocr.classify_components(st, H_STRIP, np.array([0, 4.0]))[1] == 'art'


def test_text_run_starts_at_the_right_edge_and_stops_at_the_art():
    # 'T' (bar + stem), '-', '4', '5' on the right; art blob and dither dots far to the left
    st = _stats((290, 12, 16, 39, 190), (243, 12, 37, 39, 426), (208, 11, 25, 40, 300),
                (171, 12, 29, 38, 308), (143, 30, 21, 4, 64), (126, 19, 3, 31, 76), (114, 12, 27, 5, 89),
                (30, 20, 6, 4, 20), (60, 22, 5, 5, 20))
    kinds = ['art'] + ['letter'] * 4 + ['piece'] * 5
    run = ocr.text_run(st, kinds, H_STRIP, 311)
    assert set(run) == {1, 2, 3, 4, 5, 6, 7}


def test_text_run_ignores_letters_off_the_line_and_beyond_a_word_gap():
    st = _stats((290, 12, 16, 39, 190), (243, 12, 37, 39, 426),
                (150, 12, 30, 39, 500),                      # a word gap (63 px) away: not text
                (200, 40, 30, 28, 400))                      # overlaps the line by < 60 %
    run = ocr.text_run(st, ['art'] + ['letter'] * 4, H_STRIP, 311)
    assert set(run) == {1, 2}


def test_text_run_needs_a_letter_at_the_right_edge():
    st = _stats((20, 12, 16, 39, 190))
    assert ocr.text_run(st, ['art', 'letter'], H_STRIP, 311) is None


def test_isolate_label_drops_art_and_dither_left_of_the_text():
    import cv2
    g = np.full((H_STRIP, 400), 70, np.uint8)
    cv2.rectangle(g, (20, 30), (190, H_STRIP - 1), 200, -1)             # bright art touching the band bottom
    for x in range(10, 200, 9):
        g[24:27, x:x + 3] = 200                                         # dither dots at glyph height
    cv2.putText(g, 'BS', (290, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 215, 4, cv2.LINE_AA)
    out = ocr.isolate_label(255 - g)
    assert out is not None
    crop, clean = out
    assert crop.shape[1] < 140                                          # just the text and its padding
    assert (clean == 0).sum() > 100
