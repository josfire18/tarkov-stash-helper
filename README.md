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
5. Open your stash with Tarkov running: Auto-scan (below) scans it by itself. Or set a capture
   region/hotkey in Settings and press the hotkey in-game to scan manually.

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

## Auto-scan

Hands-free mode (default on): the app finds `EscapeFromTarkov.exe`, watches its screen, and when you open an
inventory screen (stash, container window, trader / flea sell screen) and leave it alone for about a second it runs
the normal sell scan on the **whole frame** - no capture region, no hotkey, no screenshots to send. The sell page
shows an **Auto** toggle, a status line ("Waiting for stash" / "Inventory found - waiting for it to settle" /
"Auto: last scan 3 s ago") and refreshes the picture and list by itself when a new result arrives.
Turn it off with the Auto checkbox on the Sell page, or `"auto_scan": false` in `data/settings.json`
(`"auto_scan_exe"` overrides the process name, `"auto_scan_in_raid": true` allows scans of the in-raid inventory).

**How it works** (`autoscan/`):

1. `winapi.py` - process -> main window -> monitor (read-only Win32 queries; the 2560x1440 game window on
   monitor 1 is found even if this app sits on a second monitor). Skips minimised games and frames where this
   app's own window covers the game.
2. `capture.py` - passive capture, chosen after reading the Desktop Duplication, Windows.Graphics.Capture, `dxcam`
   and OBS material: **DXGI Desktop Duplication via `dxcam` is the primary method**, `mss` (GDI) the fallback.
   Desktop Duplication is pull based (one frame per request, only the game's region is copied) and is what OBS
   "Display Capture" uses. On Windows 10 DX11 "exclusive fullscreen" runs through Fullscreen Optimisations (a
   flip-model surface that the DWM still composes), so duplication sees it; Microsoft's docs warn a surface that
   truly bypasses the DWM can come out black, so every frame is black-checked and after 3 black frames (or an
   error) the next backend is tried, with the primary retried every 60 s. Windows.Graphics.Capture was
   rejected: it pushes a callback per composed frame (165/s here) and Microsoft says it is not reliable for
   exclusive fullscreen either. Measured on this machine (2560x1440 primary, RTX 5080): DXGI 3-7 ms per
   grab, GDI ~75 ms.
3. `detect.py` - the cheap detector. It never processes the full frame: line-colour `(84,81,73)` runs are
   taken from every 4th row / column (decimation keeps 1 px lines; an area filter would erase them), fitted with
   a lattice `x0 + k*pitch` on both axes, and a frame counts as an inventory when each axis has 4 lattice lines that
   each run for 2+ cells. A second check looks for the lobby's bottom menu bar (two black hairline rows with lit UI
   between them): an inventory **without** the bar is the in-raid inventory and is never scanned.
4. `trigger.py` - state machine: scan only when two consecutive polls are near-identical (stability), the view
   differs from the last scan (32x32 difference hash, so tooltips and the cursor are ignored but a scroll or tab
   change is not), at least 3 s after the previous scan; scrolling therefore gives one scan per resting position. A
   failed scan is retried only after a view change or 30 s.
5. `service.py` - the below-normal-priority polling thread and the `/api/autoscan/{status,result,toggle}` routes.
   `app.py` only gained the `auto_scan` setting, a `frame_bgr` argument on `_sell_scan_inner`, and start/stop.

**Cost.** Polling is 2 Hz while an inventory is on screen, 1 Hz otherwise, 0.5 Hz after ~30 s of gameplay and every
3 s while the game is not running. The thread runs at `THREAD_PRIORITY_BELOW_NORMAL`, the duplication is released when
the game is gone / the feature is off, and the identify pipeline never runs outside a settled lobby inventory.
Measured (idle loop at 2 Hz, 2560x1440, dev machine): grab 3 ms + detect 7 ms per poll, ~5 % of one CPU core
(about 0.3 % of the whole CPU; 0.5 Hz gameplay polling is a quarter of that), ~0.2 % of the GPU's 3D engine.
Detector per-frame cost: 5 ms on raid frames, 7 ms on inventories (max 15 ms), 720p-1440p.
Frame-time impact on the game itself needs an elevated PresentMon capture and was not measurable without
elevation: run `PresentMon --process_name EscapeFromTarkov.exe` with Auto on and off to check your own setup.

**Detector accuracy** (`python scripts/autoscan_eval.py --frames-dir <extracted frames>`): on 1802 frames sampled from
raid recordings (1440p NVIDIA clips, 1080p captures, Medal clips) there are no false positives on gameplay; the
only frames it flags are real stash / gear screens that were in the recordings. Recall on the 11 labelled
screenshots (+ each crop pasted onto a dark 1440p canvas): 17 of 19, the misses being a stash dimmed by a modal
error dialog and one sparse 1080p screen pasted at the wrong UI scale. The pass also needs the lobby menu bar
to start a scan, so cropped screenshots without it do not count as "in the stash".

**Anti-cheat stance.** Escape from Tarkov is protected by BattlEye. Auto-scan is passive screen capture of the same
kind OBS and Discord perform: it reads pixels the Windows compositor has already produced and queries the window
manager for the game window's rectangle. It does **not** read or write game memory, open the game process,
inject DLLs or hook anything, simulate input, or move/resize/focus the game window.

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
