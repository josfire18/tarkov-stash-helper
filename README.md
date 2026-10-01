# Tarkov Stash Helper

A local desktop tool for Escape from Tarkov that identifies items in your stash
from a screenshot (via icon + OCR matching) and tells you whether to sell to a
trader or the flea market.

Everything runs on your own machine — screen capture, OCR, and item
identification all happen locally. Nothing about your game or your stash is
sent anywhere except public item-price lookups against [tarkov.dev](https://tarkov.dev).

## Using the app (recommended: download the release)

1. Grab the latest `TarkovStashHelper.exe` from the [Releases](../../releases) page.
2. Install **Tesseract OCR** (needed for reading item names off the screen):
   ```
   winget install UB-Mannheim.TesseractOCR
   ```
3. Double-click `TarkovStashHelper.exe`. A window opens — no browser, no URL,
   no console window. Closing the window minimizes it to the system tray;
   right-click the tray icon to reopen or fully quit.
4. First run: build the icon database (button in the app) so it can recognize
   items. This pulls the item catalog + icons from tarkov.dev and reads EFT's
   local icon cache if it can find your game install — it can take a few
   minutes the first time and is cached afterward.
5. Set your capture region/hotkey in Settings, then press the hotkey in-game
   to scan your stash.

Windows may show a SmartScreen warning on first run because the exe isn't
code-signed — click "More info" → "Run anyway". This is a local, open-source
tool; check the source in this repo if you want to verify that yourself.

## Running from source

Requires Python 3.11+.

```
pip install -r requirements.txt
winget install UB-Mannheim.TesseractOCR
python app.py
```

## How it works

Item identification lives in the `identify/` package (the **v2** engine, default). Set
`"identify_engine": "legacy"` in `data/settings.json` to fall back to the old masked-NCC
icon DB that is still in `app.py`.

1. `identify/grid.py` - finds the stash lattice from the 1 px border line EFT draws around
   every item (colour range tolerant of the UI's top gradient, ridge fallbacks for
   JPEG/resampled captures). Returns every panel with its pitch (per axis), all rows/columns
   and viewport-clipped first/last rows.
2. `identify/segment.py` - item footprints = cells joined across edges that have *no* border
   line, so multi-cell items stay whole, identical neighbouring stacks stay separate and
   rotated items need no special case. Empty cells are detected.
3. `identify/catalog.py` - one `.npz` template catalog of **every** item (ammo, guns, presets
   and containers included) from tarkov.dev base images plus EFT's icon cache (re-associated
   to items on every build; unmatched cache renders are kept as anonymous "build" templates).
   `data/identify_catalog_v2.npz` is rebuilt automatically when a source changes.
4. `identify/match.py` + `ocr.py` + `dino.py` - stage 1 masked pixel residual against
   same-footprint templates (overlay bands only down-weighted, background measured from the
   tile), stage 2 DINOv2-small re-rank of the shortlist (optional: needs torch +
   transformers; the exe ships without them), stage 3 OCR of the printed short name
   (batched Tesseract, glyph-confusion-aware fuzzy match) which is authoritative for ammo
   calibres and built weapons.
5. `identify/pipeline.py` - `scan(image) -> list[Detection]` with footprint, rotation, item id,
   calibrated confidence, an `uncertain` flag (a tile the catalog cannot explain is flagged,
   not guessed), per-stage evidence, stack count and Found-in-Raid.

Other pieces: `icon_cache.py` (legacy engine) reads EFT's local icon cache; the UI
(`templates/`) is served locally and hosted in a native window via `pywebview`, with a
`pystray` tray icon.

## Evaluating / labelling

```
python -m pytest tests                                        # unit tests (no network, no game data)
python test_scan.py --score data/eval/stash1.png --engine both  # v2 vs legacy, per-category metrics
python test_scan.py --robustness data/eval/stash1.png          # JPEG / blur / rescale / stretch per stage
python test_scan.py --prefill data/eval/new.png                # draft truth + contact sheets (crop | predicted icon | name)
python test_scan.py --relabel data/eval/new.png fixes.json     # apply the corrections you read off the sheets
python -m identify.calibrate data/eval/new.png                 # refit the confidence calibration
```

Truth files (`<name>.truth.full.json`) label *every* item with its pixel rectangle and mark
items you cannot identify from the crop as `"uncertain": true` (excluded from accuracy, still
counted for segmentation). See `identify/evaltools.py` for the format.

## Known limitations

- Windows only (screen capture region math, hotkey listener, and the default
  Tesseract path are all Windows-specific).
- OCR accuracy depends on screen resolution/scaling — a native-resolution
  capture of the stash region reads noticeably better than a downscaled one.
- v2 needs about 50 px/slot or more (1080p UI scale 80%+) and, for JPEG captures, quality 70+;
  below that the border lines are too degraded to segment reliably.
- Items released after your last "Build Icon DB" / price refresh are not in the catalog; v2
  flags them `uncertain` instead of guessing. Items that share an icon family (dogtags, colour
  variants hidden under attachments) are flagged too.
