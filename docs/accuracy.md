# Identification accuracy

Measured with `python scripts/accuracy_report.py [--no-dino]` on every `data/eval/**/*.png` that has a
`*.truth.full.json` (11 screenshots, 1063 labelled tiles; `w08_1080p_junk` has every row marked
uncertain, so it contributes nothing). Every tile counts as exactly one of these:

* **correct**: the right item and not flagged.
* **wrong**: the wrong item and not flagged. This is the dangerous case.
* **uncertain**: flagged. This counts as a failure too.
* **missed**: the tile was not segmented.

Guns count as correct when they are recognised as a gun, because the sell list skips them. Dogtags count as correct when any dogtag is named.

**Protocol change: clean state per screenshot.** `test_scan.py --score` scans each image twice
(a "warm-up" scan, then the timed scan), and the learned-icon store carries over between images. Two
confirmations of a label teach a cached icon its name, so the warm-up scan graded the timed scan on
what it had just learned. The results then depended on scan order, and the old numbers were inflated.
`accuracy_report.py` snapshots the catalog before any scan and restores it, with an empty learned
store, before every screenshot. Both the baseline and the result below use this protocol.

"full" means stage 1 + DINO + OCR (this dev machine). "lean" means stage 1 + OCR, which is what the packaged exe runs.
It ships without torch.

## Baseline vs now

| screenshot | baseline full (correct/wrong/uncertain/missed) | now full | baseline lean | now lean |
|---|---|---|---|---|
| JunkBox1 (125) | 114 / 0 / 11 / 0 | 113 / 0 / 12 / 0 | | 118 / 0 / 7 / 0 |
| RandomStash (62) | 46 / 0 / 16 / 0 | 47 / 0 / 15 / 0 | | 46 / 0 / 16 / 0 |
| THICC_Plus_Junk2 (88) | 88 / 0 / 0 / 0 | 87 / 0 / 1 / 0 | | 87 / 0 / 1 / 0 |
| w01_1080p_stash (201) | 180 / 0 / 20 / 1 | 180 / 0 / 20 / 1 | | 189 / 0 / 11 / 1 |
| w02_ultrawide (115) | 87 / **2** / 25 / 1 | 109 / 0 / 5 / 1 | | |
| w03_1600p (50) | 48 / 0 / 2 / 0 | 48 / 0 / 2 / 0 | | |
| w04_1440p (95) | 91 / 0 / 4 / 0 | 94 / 0 / 1 / 0 | | |
| w05_1440p_mixed (138) | 136 / 0 / 2 / 0 | 138 / 0 / 0 / 0 | | |
| w07_1080p_firstraid (63) | 59 / 0 / 4 / 0 | 60 / 0 / 3 / 0 | | |
| w09_1440p_mags_guns (126) | 113 / **4** / 9 / 0 | 121 / **1** / 4 / 0 | | 119 / **1** / 6 / 0 |
| **total (1063)** | **962 / 6 / 93 / 2** (90.5 %) | **997 / 1 / 63 / 2** (93.8 %) | **973 / 8 / 80 / 2** (91.5 %) | **1003 / 5 / 53 / 2** (94.4 %) |

The lean per-image rows that are left blank were not broken out. The totals come from the same runs.
Coverage (correct + wrong) / labelled went from 91.1 % to 93.9 % (full) and from 92.3 % to 94.8 % (lean).

**Golden egg / DVD drive.** All 6 labelled Golden egg / DVD drive tiles (JunkBox1, THICC_Plus_Junk2, w01) are
correct and certain in both configurations, both before and after this work. The live confusion the user saw
probably came through the learned-icon store, which the next section closes.

## What changed (and why)

1. **Uncertain items are never sold** (`sellcalc.plan_entries`). An uncertain detection becomes a KEEP row
   drawn with a **CHECK** badge and the reason "Not sure this is X - check it yourself (never auto-sold)".
   Covered by `tests/test_sell_order.py::test_uncertain_detection_is_never_routed_to_sell`.
2. **Footprint is a hard constraint for the label** (`Engine._footprint_ok`). An item's footprint is fixed in
   either orientation. Only guns (any preset size, or bigger when built) and non-magazine weapon mods can be
   bigger than their base icon. Before this, a 1x1 sight's label could name a 1x2 drum magazine.
3. **Label twins via glyph folding** (`ocr.fold_glyph`). Short names that differ only by glyphs that the label
   font draws alike (5/S, 0/O, 1/l/I, 8/B, 2/Z, 6/G) are treated as twins, not as a confident read of one of
   them. Example: `MPX F5` / `MPX FS`. All OCR variants of a label are pooled. Before, the first variant that
   was read won.
4. **Partial reads cannot overrule a literal picture match**: residual <= 5 and DINO >= 0.85, up from 3 / 0.9.
   Truncated labels may also be 2 characters shorter than the label capacity, because punctuation such as the
   "-" in `SCAR-SD` uses label room. This fixed SCAR-SD flash hider -> "Balaclava (Scars)".
5. **Gun vs part with the same label** ("TOZ-106" names the shotgun and its stock). Presets count as the gun,
   both for the footprint and for the picture. When the picture cannot separate them, a calibre printed at the
   bottom left (`ocr.looks_like_caliber`: "20g", "5.45x39", ".366") breaks the tie. The result is still capped
   at uncertain, because item art can look like text: an MP5 upper receiver shows "9x19PARA".
6. **Confidence rules with explicit evidence**:
   * *label-verified* (0.97): an exact read, no differently named item that fits the footprint, and same-label
     twins separated by a clear picture margin (`TWIN_GAP_OK` = 2.0 fused units with DINO, 6.0 on residual
     alone).
   * *unresolved twins*: capped at 0.5, which counts as uncertain.
   * *weapon-sure* (0.95): the label names a gun with no unresolved twin, or the best three different pictures
     are all guns, presets or weapon builds. Guns are skipped by the sell list, so being sure it is a gun is
     enough.
7. **Learning is stricter** (`Engine._learn`). A cached icon is only named when the item can have that
   footprint and its label is not glyph-ambiguous with another item's label. A systematic misread such as
   "MPX FS" on a drum magazine used to "confirm itself" across two scans of one view, and the wrong name then
   stuck permanently.

## Stress test, pair verification, trained model: not done, with evidence

* **Difference-focused pair verification** was built and measured on all 976 real tiles. It compared the truth
  item's template against the best differently named rival, over the pixels where the two templates differ,
  with ±2 px alignment at 63 px/slot. The truth template lost or tied in **131 / 976** tiles. The cause:
  tarkov.dev base images often differ from the game's own render of the item. Examples: Secure Flash drive,
  T-Shaped plug, Pile of meds, Beretta M9A3, ammo packs. Pixel verification against tarkov.dev images is
  therefore **not** a sound hard verifier, and it was left out of the pipeline. It is only reliable against the
  game's own icon-cache renders.
* The **catalog-wide synthetic stress test** (`scripts/confusion_stress.py`) and the **trained discriminative
  model (ONNX)** did not fit in this session's budget. The pair measurement above is also the main risk for
  both. Synthetic cells rendered from tarkov.dev images would measure confusions between *templates*, not
  between real renders and templates. A model trained only on those images would inherit the same domain gap.
  Next step, in order:
  1. Collect real crops by mining the debug bundles (`data/debug/scan-*`) from the user's own scans. Label
     them automatically only when the label is exact and footprint-unique (the label-verified rule).
  2. Train a metric-learning model on game-cache renders plus those crops, with augmentation. Export it to ONNX
     and run it with `cv2.dnn`, which adds no new runtime dependency.
  3. Use its calibrated top-1 / top-2 margin as the picture term. Fit the calibration leave-one-screenshot-out
     (extend `identify/calibrate.py`).
* **Reliability**: the eval set has only 1 (full) / 5 (lean) certain-but-wrong answers out of about 1000, which
  is too few to fit and check a calibration curve. Of the certain answers, 997 / 998 (full) and
  1003 / 1008 (lean) are correct.

## Zero-wrong pass (lean path, this checkout's data)

Measured on this worktree's copy of the eval data with `accuracy_report.py --no-dino`:

| | correct | wrong | uncertain | missed | certified-but-wrong |
|---|---|---|---|---|---|
| before | 923 | **2** | 136 | 2 | 0 |
| after | 924 | **0** | 137 | 2 | 0 |

(The 94.4 % lean figure above came from a richer data set than this checkout has; the two
runs here differ only by the changes below.)

* **w09 #106, TOZ-106 shotgun read as "TOZ-106 stock": now CORRECT.** The partial read "TOZ-186"
  scores equally for the gun, the stock and a pistol grip, and the picture did not separate gun from
  stock; the calibre probe ("20g") did fire but `_label_authority` then refused because another
  *part* ("TOZ-106 custom cut Mosin stock", 90.9) was within 5 points. When the calibre says gun, only
  weapons are rivals (`only_weapons`). The gun is confident (0.95, `verified: weapon+calibre`) only when
  the printed calibre is that gun's own AND no non-weapon item that shares the label names the same
  calibre (`_calibre_names_gun`): an MP5 upper receiver's art reads "9x19PARA" and its name carries 9x19,
  so it can never vouch for the gun.
* **w02 #87, MP5 upper receiver -> MP5 30-round magazine (conf 0.978): now UNCERTAIN ("Check").** The
  exact label "MP5" is printed by the receiver (2x1), the magazine (1x2) and the gun. The picture chose
  the magazine only by turning its template (residual 5.29 vs 6.49 for the receiver, which fits the tile
  unturned) and the font-label anchor was not certain (MP5 20.05 vs "PS" 25.36), so nothing separated
  them. `_unresolved_label_twins` caps such a pick at 0.5 when the chosen item fits only turned, a
  same-label differently named item fits unturned, and the picture lead is below `TWIN_GAP_OK`. Narrower
  variants of the guard (any same-label twin, any residual gap) flagged 4 correct ammo / headset colour
  variants, so it is restricted to the turned-vs-unturned case. Not promoted to correct: the residual
  difference is within noise and no other evidence is left on the lean path.
* **w03 #49 / #50 (MP5 upper receiver -> MP5 gun) and w02 #96 (D3CRX Ranger Green vs Black)**: already
  uncertain (0.5, "printed label is shared by N items") in this checkout, so not wrong; not promoted to
  correct. The D3CRX colour comparison over the opaque pixels (residual 4.77 vs 5.64) was not built in
  the time available.

## What remains (every failure is listed by `accuracy_report.py`)

* **Wrong, full**: w09 #106. A TOZ-106 shotgun was read as "TOZ-186" and the picture chose the stock
  (conf 0.945). The stock and the gun's preset score about equally on the picture, and the partial read never
  reaches the gun check.
* **Wrong, lean only**: MP5 upper receiver -> MP5 30-round magazine (w02) or -> MP5 gun (w03 x2), and D3CRX
  Ranger Green -> Black (colour variant, w02). Without DINO, the residual alone does not separate these
  pictures, and the old logistic calibration still reports them as certain.
* **Uncertain (63 full / 53 lean)**, by cause:
  * Anonymous game renders ("build #N") of non-weapon items whose label read was not exact (T-Shaped plug read
    as "T5Rlug").
  * Items whose tarkov.dev image differs from the game render, with a partial label: Interchange map, Ushanka,
    Physical Bitcoin, UBEY mask, Pack of chlorine.
  * Ammo packs whose label names several calibres ("BP") while the pack pictures differ only slightly.
  * Genuinely identical pictures with a truncated label: **AK-74 vs AKM Hexagon suppressor**, where the label
    shows only "Hexagon" and the residual is 2.47 vs 2.47.
  * Backpacks with no label.
* **Missed**: 2 tiles were not segmented (Ushanka at the viewport edge, Pack of milk at the ultrawide edge).

## Latency

Mean time per scan over the 11 screenshots (RTX 5080 dev machine): full 6.8 s before vs 6.0 s now, lean
4.7 s vs 4.2 s. That is about 10 s on the 2560x1440 mags/guns screen. It is far from the < 2 s goal for a live
viewer. Most of the time goes to Tesseract (4 processes per scan) and to the full-catalog stage 1. Neither was
changed here.

## Shared labels settled by size and picture (2026-10-03, main checkout, lean)

The exact-label anchor capped every tile whose printed label is shared by several items at 0.5
("picture alone decides"), even when only one of them can occupy the tile or the picture clearly
prefers one (M9A3 magazine vs M9A3 side grips, MP5 magazines vs the MP5 gun).
`Engine._twins_settled_by_footprint` keeps the cap only when a same-label twin fits the tile as
drawn AND the picture's lead over it is below the twin gap (6.0 residual-only, 2.0 with DINO).

Strict eval, 1063 tiles, lean: **996 correct / 0 wrong / 65 uncertain / 2 missed** (was 924 / 0 /
137 / 2); certified 490, certified-but-wrong 0.

## Coverage pass: "Check" tiles resolved by evidence (2026-10-12, lean, strict)

Starting point (`accuracy_report.py --no-dino`): 993 correct / 0 wrong / 66 uncertain / 4 missed of 1063,
certified 490, certified-but-wrong 0. After: **1026 correct / 0 wrong / 33 uncertain / 4 missed**
(96.5 %), certified 505, certified-but-wrong 0. No screenshot lost a correct tile at any step. Each step
below was a full report run; the gain is what that step alone added.

| bucket (of the 66) | before | after | what resolves it |
|---|---|---|---|
| exact label shared with a differently *shaped* item (M9A3 mag vs side grips / pistol, MP5 20-round mag vs MP5 gun, VOG-25 vs Khattabka) | 15 | 0 | `_shape_distinct`: the matched templates' silhouettes overlap <= 0.65 (IoU), so half the twin gap (3.0 residual) is enough |
| same, but the tile sits on the panel edge without being clipped (ammo packs, G28, Gatorz glasses, Hawk, Stich carrier) | 6 | 0 | an unclipped tile is measured whole: `_certify` no longer treats "touches the panel border" as "may be cut" |
| settled twins whose calibrated confidence was low (ammo packs whose tarkov.dev art differs from the render: res 11-17) | 3 | 0 | settled twins under an exact game-font label are label-verified (`verified: label+twins`) |
| label strip pixel offset ("Analgin" at 63.09 px/slot) | 1 | 0 | label anchor x shifts reach +-1 px (the strip edge is an integer pixel, the pitch is not); also certified 490 -> 505 tiles |
| look-alike twins (colour variants, loose rounds of two calibres, suppressor vs its dust cover): EXFIL cover, PP gs, NT4, Vityaz | 4 | 0 | `_colour_gap`: mean colour error over the pixels *both* templates draw opaque, rival minus pick >= 12 levels, and the picture must not prefer the rival |
| gun builds: TOZ-106 gun vs stock (exact label), custom-named builds (MP-133 Tactical, KS-23M Drozd; label unreadable) | 4 | 0 | the printed calibre (`20g`, `12g`) is the gun's own and no same-footprint part names it (`_calibre_names_gun`); for an unreadable label also a near-literal gun match (res <= 4) with no part within 3.0 |
| not resolvable on this evidence (see below) | 33 | 33 | |

The remaining 33 (+4 missed), by cause:

* **Wrong item picked, no readable label / no close match (7):** Cordura x2, ZSh helmet, Pilgrim backpack, MP5
  read as MP-153, BP / T-45M1 ammo packs (all twins match equally badly). They are flagged, not wrong.
* **Cut by the scroll edge (4):** ADAR, Hawk, Ushanka, Expeditionary fuel tank. `_certify` skips clipped
  tiles; certifying a bottom-clipped tile (label intact, height unknown) is the next candidate.
* **Look-alike twins the evidence cannot separate (14):** SpecterDR vs FDE (res lead 4.5, colour 7.7), D3CRX
  Ranger Green vs Black (the picture and the colour both lean the *wrong* way, -0.9 / -3.8), PMAG GEN M3 vs
  its "W" variant x3 (colour 4-6), Death Shadow mask (6.3), .338 FMJ among 24 FMJ twins (11.9), MP7 30 vs 40
  round, M62 x2, flechette, Dual Sabot, Secure Flash drive x2.
* **MP5 upper receiver (3):** drawn turned (2x1) or built (1x3), the picture prefers the magazine / gun.
* **Identical pictures (2):** AK-74 vs AKM Hexagon suppressor, truncated label.
* **Label read (3):** "HDD" x2 (the selected tile draws the text dimmer, so the observed-ink check misses the
  "H" and "DD" fits as well) and "Printer paper" (bright art leaks into the observed ink).

Tried and dropped: widening the label ink test to dimmer text (gains 0.72-0.85) certified 14 more tiles but
made 4 previously correct tiles uncertain (Roubles, an FMJ pack). Not shipped.

| screenshot | before (correct / wrong / uncertain / missed) | after |
|---|---|---|
| JunkBox1 (125) | 124 / 0 / 1 / 0 | 124 / 0 / 1 / 0 |
| RandomStash (62) | 56 / 0 / 6 / 0 | 57 / 0 / 5 / 0 |
| THICC_Plus_Junk2 (88) | 86 / 0 / 2 / 0 | 87 / 0 / 1 / 0 |
| w01_1080p_stash (201) | 193 / 0 / 7 / 1 | 195 / 0 / 5 / 1 |
| w02_ultrawide (115) | 97 / 0 / 16 / 2 | 101 / 0 / 12 / 2 |
| w03_1600p (50) | 35 / 0 / 15 / 0 | 48 / 0 / 2 / 0 |
| w04_1440p (95) | 89 / 0 / 6 / 0 | 94 / 0 / 1 / 0 |
| w05_1440p_mixed (138) | 137 / 0 / 1 / 0 | 137 / 0 / 1 / 0 |
| w07_1080p_firstraid (63) | 58 / 0 / 4 / 1 | 59 / 0 / 3 / 1 |
| w09_1440p_mags_guns (126) | 118 / 0 / 8 / 0 | 124 / 0 / 2 / 0 |

Caveats. The thresholds introduced here (`SHAPE_IOU_MAX` 0.65, `SHAPE_GAP_FRAC` 0.5, `COLOUR_GAP_OK` 12,
`WEAPON_LITERAL_RES` 4.0) were read off the ~114 label-twin tiles of this eval set: on it, the wrong-direction
cases have leads of -3.8 to +1.6, and every tile settled by a rule has a margin of at least 3.4 (residual) or
12.0 (colour). That is one stash set, not a calibration curve. All rules apply only under an exact game-font
label whose twin set is closed, never to clipped tiles, and not to `label_twins` (MP5-style) cases. Tests:
`tests/test_identify_coverage.py` (shape / colour twins, calibre-vouched gun builds) and
`tests/test_identify_anchors.py::test_label_with_the_pen_a_pixel_off_is_still_certain`.

## Viewport-cut tiles are certified on their visible part (2026-10-12, lean, strict)

`_certify` used to return early for any tile cut by the scroll edge. It now certifies them on exact
evidence about the *visible* rows only:

* **Bottom-cut tile (label intact):** the closed-set game-font label match runs as usual, but the
  candidate set is every item that can have that footprint *given the visible rows* (width W in either
  orientation, height >= ceil(visible slots - 0.1); weapons / mods that could be built that wide). The
  label strip must lie entirely inside the visible rect. Twins that the footprint cannot separate
  (shared label) are left as they were; nothing is bound for later (`_bind` is skipped).
* **Render anchor on the visible rows** (`CacheIndex.match_clipped`): every footprint of width W at
  least as tall as the visible rows is tried (<= 6 slots), the render cropped to the visible rows
  (top rows for a bottom cut, bottom rows for a top cut, aligned to the rect edge that is real). It
  needs >= 50 % of the item visible, 2x the opaque area of a whole tile, the same exactness threshold
  and the same runner-up rule. Top-cut tiles have no label, so this is their only route.
* A tile cut at both ends is never certified.

Result (`accuracy_report.py --no-dino`): **1027 correct / 0 wrong / 34 uncertain / 2 missed before and
after**, certified 505 -> 510, certified-but-wrong 0, no screenshot changed. The five newly certified
tiles were already correct (Saiga-12K, Salewa x3, Day Pack): they are now label-verified rather than
merely confident. No cut tile was certified by render: the exact render residual on a cut tile is
4.7-35 here (e.g. the correct Salewa render at 4.76 vs the 4.5 bound), so the render route stays dormant
on this set.

The "Cut by the scroll edge (4)" bucket above was mis-described; none of the four is a bottom-cut tile with
a readable label: ADAR is a top-cut custom build (SVD stock, 0.86 slot visible, no exact render exists),
Ushanka is a 2x2 hat whose left column is hidden by another window (width unknown), and Hawk and the
Expeditionary fuel tank in RandomStash are not cut at all (their game-font label does not fit; Hawk reads
"Hawk" by OCR but the strip match fails). Tests: `tests/test_identify_anchors.py::test_cut_tile_*`.

## Label reads on dimmed / busy strips (2026-10-12, lean, strict)

Strict lean eval 1027 -> 1030 correct, 0 wrong, 34 -> 31 Check, 2 missed; certified 510 -> 537, certified-but-wrong 0; no screenshot lost a correct tile.

Causes found per tile (a global "wider ink test" certified more but lost correct tiles, so none of this widens the ink test globally):

* **HDD x2 (w01)**: the selected tile draws the label darker and its 'H' strokes are thin, so the face-colour ink test saw only "DD" and the candidate "DD" (score 8.8) beat "HDD" (10.4). `LabelRenderer.wide_ink` (neutral grey strokes 85-215, dark neighbour, at least 2 rows tall) is now counted as unexplained ink, but only within one em left of a candidate's rendering. "DD" now leaves the H unexplained, "HDD" wins.
* **Printer paper (w01)**: the white item corner across the tile produced "label ink" 20+ px left of the text (unexplained 0.26). Observed ink now only counts when it runs on from the rendering without a gap wider than `GAP_EM` (0.55 em, a word space).
* **Shared labels are not newly flagged**: a label that became certain only under the relaxed read and is printed by several items ("FMJ", 24 ammo) keeps its earlier outcome (`certain(fits, strict=True)` in `_label_anchor`); the relaxed read only certifies a unique item.
* **Hawk (w01) and fuel tank (RandomStash), NOT fixed**: the strip match is exact ("Hawk" 16.7 vs 42.9, "Fuel" 15.1 vs 36.8), but the tile is a viewport-cut taller item (Hawk gunpowder is 1x2, the tile is cut by the panel's bottom edge and segmented as 1x1; the fuel tank is a 2x2 cut by the image's bottom edge, 2x1) and `it.clipped_bottom` / `panel.clip_bottom` is False for both, so the footprint filter drops the true item from the candidate texts. The cause is the panel's bottom-clip detection, not the label read. Both also print a label shared with another item ("Hawk": tomahawk, "Fuel": Metal fuel tank) that the picture would still have to settle.
