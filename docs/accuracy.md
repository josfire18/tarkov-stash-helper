# Identification accuracy

## Round 3: exact anchors ("this icon is for sure this icon")

Certainty no longer comes from a learned probability. A tile is **certified** only when it agrees
*exactly* with something the game itself drew:

* **Anchor 1, exact game render** (`identify/anchors.py`). EFT's icon cache holds the textures the
  game rendered for every item it has shown, and the stash tile draws exactly that texture. The tile
  is compared with each shortlisted render over the render's fully opaque pixels only (so the rarity
  tint and the cell background drop out), with the label band, the bottom band (count, FiR, calibre)
  and the frame excluded. At other UI scales the render is resampled the way the GPU does it
  (bilinear), with a sub-pixel phase search (±2 px coarse 0.5 px, refined to 1/8 px, then reused for
  the whole panel). The score is the mean absolute level error (0 to 255).
* **Anchor 2, exact label in the game's own font** (`identify/fontlabel.py`, `identify/tmpfont.py`).
  The labels are TextMeshPro. The Bender font assets are taken from the installed game's
  `resources.assets` with UnityPy, once, on first run (`data/fonts/`, gitignored, never committed).
  That covers the raw OpenType fonts and the TMP SDF assets: 1024² atlas, 369 glyph records, face
  info, `normalSpacingOffset`, and the material. The game is IL2CPP, so the MonoBehaviour has no
  type tree, and `tmpfont.parse_font_asset` reads the TMP 1.1.0 layout directly. The label uses
  *Bender Outline SDF* (a dark halo all round, `UNDERLAY_ON`). The short name of each candidate is
  rendered by sampling the game's distance field with its glyph rects, bearings and advances. The
  layout is right-aligned, and TMP overflow mode Truncate turns "Powerbank" into "Powerban". The
  size, character spacing, face/halo edges in SDF texels, anti-aliasing ramp and text colour were
  calibrated on real strips (`scripts/calibrate_label.py`, stored in
  `identify/assets/label_params.json`): 11.35 px at 63 px/slot, spacing −6.7 em/100.
  * The material's own shader constants were tried first, with TMP's exact `GetColor` and underlay
    maths, but they did not reproduce the strips: mean ink residual 22 to 38. The label component
    evidently overrides them. The FreeType raster of the extracted OTF reached 10 to 12 but has the
    wrong advances, because TMP adds `normalSpacingOffset`.
  * The calibrated SDF rendering reaches a mean ink residual of about 14.5 (63 px/slot) and 14.9
    (84 px/slot) on held-out strips. It is visually indistinguishable from the real label, but it
    is not pixel-zero: the remaining error is the inpainted background under the anti-aliased halo.
  * Matching is closed-set. The candidates are the stage-1 items, the OCR hits, the render's hint,
    and the printed names nearest to the leading guesses, all footprint-checked. Each is scored by
    its ink residual plus a penalty for observed label ink that it leaves unexplained.
  * The label is certain when the best candidate is exact (total ≤ 24, unexplained ink ≤ 10 %),
    the runner-up is at least 1.6× worse and 8 levels worse, and the game prints that text for
    exactly one item that fits the footprint. A text printed for several items ("MP5", "D3CRX",
    "BP") is settled only by an exact render of one of them. Otherwise the tile is flagged
    *uncertain*, never guessed.
* **Self-reinforcing.** When a tile is label-certified and also exactly matches one render, that
  render is bound to the item permanently (`LearnedNames.bind_certain`; ordinary reads can never
  overwrite it). From then on the render alone certifies, even with the label covered.
* **Decision.** The final answer is certified if either anchor is exact and the two do not
  disagree. A disagreement is flagged uncertain and logged. Otherwise the stage 1/2/3 pipeline
  answers with its own confidence. `extra={'strict_anchors': True}` instead flags every
  uncertified tile, as the design asked. That mode was not measured. With
  `uncertain_below = 0.80`, the default hybrid is what the tables below report.

### Render identities: tarkov.dev association is not proof

A render's item is known only from (a) a binding made by a certain label match, or (b) the
catalog's picture association with a tarkov.dev icon, and (b) is trusted only when the distance is
≤ 1.5 and the runner-up is ≥ 3× further away. The first measurement trusted every association with
the old 1.4× rule, and it certified 3 tiles wrong. The Secure Flash drive's render associates to
*"Flash drive with Mr. Kerman's hash codes"* at distance 4.2 against 15.6 for the next item:
tarkov.dev's art for the right item differs from the render, while the wrong item's art happens to
match it. Two of those 3 tiles were this case. The third was a Marlin rifle vs its "Default" preset,
which is not a real error (presets map to the gun). Picture association is therefore only a hint.

### Measured residuals (`scripts/anchor_measure.py`, `scripts/label_measure.py`)

Anchor 1, aligned residual of every labelled un-clipped tile against its shortlisted renders,
"true" meaning the render the catalog associates with the truth item:

| pitch | tiles | true-render residual min / median / p90 / max | best other-item render min / median / p90 |
|---|---|---|---|
| 63 (1080p, native) | 258 | 0.17 / 4.75 / 12.1 / 22.3 (n = 137) | 4.4 / 19.9 / 41.5 (n = 252) |
| 84 (1440p, bilinear ×4/3) | 778 | 0.17 / 3.51 / 11.3 / 22.4 (n = 312) | **0.53** / 15.5 / 33.5 (n = 759) |

A pixel-exact match measures 0.2 to 2. The tail of the "true" column comes from associations to
a *different* render of the same item (another colour or state), and the screenshots from the web
were taken on other accounts and game versions. Their exact renders are mostly not in this cache:
652 of 1036 tiles have no exact render, against 7 to 17 per screenshot on the user's own
screenshots. The 0.53 "other item" value is the flash-drive pair above: two items, one render. That
is why a render certifies only when no render of a different item is also exact
(`another item renders the same`). Thresholds: exact ≤ 3.0 (native) / 4.5 (resampled),
≥ 250 opaque px at 63 px/slot, runner-up ≥ 2.5× worse.

Anchor 2, adversarial set: truth + the 25 printed names nearest to it among every item that fits
the footprint (988 non-weapon tiles):

| pitch | true text total min / median / p90 / max | best wrong text min / median / p90 |
|---|---|---|
| 63 | 8.8 / 14.3 / 24.3 / 55.3 | 8.7 / 31.6 / 44.9 |
| 84 | 9.0 / 20.0 / 34.3 / 87.7 | 16.8 / 38.2 / 49.0 |

On that set, 570 were certain and right, **0 were certain and wrong**, and 418 were not certain:
art behind the label, truncated names, look-alike candidates.

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
