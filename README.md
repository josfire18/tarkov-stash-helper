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

## Building the exe yourself

```
pip install -r requirements.txt
pip install pyinstaller
build.bat
```

Produces `dist/TarkovStashHelper.exe`. The `data/` folder (icon cache,
settings, price cache) is generated at runtime next to wherever the exe is
run from — it isn't bundled into the build.

## How it works

- `app.py` — Flask backend: screenshot capture (`mss`), stash-grid detection,
  icon identification (masked NCC template matching against a catalog built
  from tarkov.dev + EFT's local icon cache), OCR-based label fusion
  (`pytesseract`) to resolve ambiguous matches, and price/sell-recommendation
  logic.
- `icon_cache.py` — reads EFT's local icon cache and visually associates each
  cached icon with a tarkov.dev item ID.
- The UI (`templates/`) is served locally and hosted in a native window via
  `pywebview` — there's no browser tab or URL involved, it just looks like a
  normal desktop app. A `pystray` tray icon handles minimize/reopen/quit.
- `test_scan.py` — scoring harness for identification accuracy against
  labeled screenshots in `data/eval/`.

## Known limitations

- Windows only (screen capture region math, hotkey listener, and the default
  Tesseract path are all Windows-specific).
- OCR accuracy depends on screen resolution/scaling — a native-resolution
  capture of the stash region reads noticeably better than a downscaled one.
