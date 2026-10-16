# Work queue

Orchestrator → worker. One Opus worker at a time (hard problems), Sonnet workers alongside on
non-overlapping files, orchestrator reviews / tests / merges into `identify-v2`, pushes, then PRs + merges `identify-v2` → `master` after every landed task.
Usage window 06:40–11:40 UTC (2026-10-03), target ≈ 20 %/h, checked every ~25 min.

| # | Task | Worker | Status |
|---|------|--------|--------|
| 1 | Exact-anchor identification: game font extracted from Unity assets, closed-set label render match, pixel-exact game-render match, self-binding, per-tile cache (`feat/exact`) | Opus | landed 9a41427: 478 tiles certified, 0 certified-wrong; lean 1028/5/28/2 (was 1003/5/53/2). WIP (scan-loop wiring of tile cache) on `feat/exact-wip` |
| 2 | Auto-scan saves every settled inventory view (lobby + raid) to `data/scenes/live/` as the scene dataset | orchestrator | done (a07af07) |
| 3 | UI restructure: Live (home) · Needs · Settings. Header clutter gone, no region/build buttons, Needs = quests + hideout (level steppers) + Kappa + pins in one list | Sonnet | landed (feat/ui) |
| 4 | Scene layer `identify/scene.py`: split the screen by landmarks (stash filter toolbar, slot headers, floating windows with title bar + red ✕), grids per region, roles own / loot / stash / container window / picker; picker + occluded cells never scanned | Opus | landed on `feat/scene2` (identify/scene.py; real-frame tests in tests/test_scene.py; raid rules unverified - no in-raid frames yet) |
| 5 | Live pipeline: incremental scan via tile cache, SSE push, in-raid advice (grab by ₽/slot, quest items, drop cheapest own item when full), scene-filtered sell advice | Opus | landed on `feat/live2`: warm re-scan 1.1 s -> 0.4-0.5 s, 2 changed tiles ~1.1 s, new view still 6-9 s (anchors phase = 60 % of it); SSE, `liveadvice.py`, Live page groups, Settings > Live. Raid rules unverified on real raid frames |
| 6 | Hideout levels read from the hideout screen by the auto-scan | Sonnet | after 3 |
| 7 | App icon in the taskbar: own AppUserModelID + window/taskbar icon (shows the Python icon today), exe icon in the spec | orchestrator | done on `feat/fast-first` |
| 8 | Integrate, full test suite, rebuild exe, release | orchestrator | last |
| 9 | Scene layer on every lobby screen (`feat/scene-lobby`, game 1.2.0 frames): stash toolbar found anywhere (flea 'Offer creation' window, trader, receive-all, scav loot transfer); scenes `flea_offer` `flea_handover` `trader` `showcase` `scav_transfer`; roles `incoming` (advised like stash) / `trader_goods` `handover` `offer_window` (never scanned or advised) | Sonnet | done: 38/38 frames labelled right (see tests/test_scene_lobby.py) |

## Findings that shape the work
- Raid recordings on this machine are 1–4 min combat clips: no inventory frames. Real scene data
  comes from the user's own sessions (task 2).
- `identify.grid.detect_grid` merges neighbouring grids that share the lattice phase (gear column +
  item picker + stash became one 21-column panel on a real 1440p frame) → the whole-frame sell scan
  reads rig/backpack/picker items as stash items. Task 4 fixes it.
- tarkov.dev marks only the Collector's 13-quest chain `kappaRequired` since 1.0; hideout progress
  is not in the logs.

## Next (queued 09:40 UTC)
- 9. First look at a new view: DONE on `feat/fast-first` - provisional result (stage 1 + label, items `provisional`, shown pending, never a confident sell/drop) after 1.2-2.1 s, certified final 4.3-7.7 s as a new seq; warm re-scan 0.4 s. A newer view cancels the pending certification. Parallel certification measured SLOWER (GIL/BLAS contention), so it stays sequential.
- 10. Rig / backpack / pocket cells (different border style) scanned so "On you" and in-raid Drop advice see your own items. — landed on `feat/own-grids` (identify/owngrid.py)
- 11. Task 6 (hideout levels from the hideout screen). — next window
- 12. Rebuild exe + GitHub release. — orchestrator, ~11:15 UTC

## Status at 11:00 UTC 2026-10-03 (end of window)
Landed this window: exact anchors (#4), UI Live·Needs·Settings (#5), scene layer (#6), live pipeline + raid advice (#7), progressive scans + taskbar icon (#8), v0.4.0 release (#9), own rig/pockets/backpack grids (#10), lean path 0 confidently-wrong on its eval (this PR).
Next window, in order:
1. Re-run scripts/accuracy_report.py in the MAIN checkout (with learned_icons/fonts) to confirm 0 wrong at the higher coverage; push certification coverage up (the remaining "Check" buckets).
2. Verify scene + raid advice on real raid frames from data/scenes/live/ (collected while playing).
3. Hideout levels read from the hideout screen (task 11).
4. First-look latency: certification is 55–70 % of a cold view.

## Status 2026-10-11 (v0.4.1)
Done: raid state from the game logs (#13); all lobby screens recognised on 38 real 1.2.0 frames, incl. flea offer, trader, receive-all, scav transfer (#14); weak picture match without label never confident; v0.4.1 released.
Strict lean eval: 993 correct / 0 wrong / 66 check / 4 missed; certified 490, certified-but-wrong 0.
Recall fixes on `feat/grid-recall` (38 real 1.2.0 frames): dim trader/flea stash lattice fitted inside the known stash region (`identify/stashfit.py`, dim-aware empty test + edge model) - frame 1 0 -> 120 cells, frames 6-8/26/33/34 60-100 -> 120; scav-transfer backpack found whole (4x5, was 2-3 columns: neutral bright frame lines count in `_grid_extent` on that screen); the 2 eval tiles lost to the empty-slot threshold are found again (gradient-aware empty rule: std < 8, or std < 11 with a plane-fit residual < 4.5).
Next:
1. Real in-raid inventory frames (none yet — all 4 'raid' frames were lobby screens) → verify own vs loot + Grab/Drop.
2. Dim trader-screen stash: detect_grid finds few cells (frames 1, 5-10).
3. Scav-transfer backpack only 2 columns found (owngrid); 2 more missed tiles at the eval stash edges after the empty-slot change.
4. Hideout levels from the hideout screen.
5. The 66 "Check" tiles (coverage).

## Overnight 2026-10-12 (low pace)
Merged: grid recall (#16), coverage (#17, v0.4.2 released), cut-tile certification (#18), label reads (this PR).
Strict lean eval: 1030 / 0 wrong / 31 Check / 2 missed; certified 537, certified-but-wrong 0.
Next: panel bottom-clip detection misses items cut by the panel/image bottom (Hawk gunpowder, fuel tank segmented too short); hideout screen; real raid frames still needed.

Game mode (`feat/game-mode`): the app knows PvP / PvP Season / PvE. Detected from the live application log (`Session mode:` via `eftlogs.RaidState.mode()`, fallback: newest session in the log records); `app.effective_mode()` = the `game_mode` setting or, on auto, the detected one. Prices, tasks, hideout, traders and flea rules come from json.tarkov.dev `regular/` (PvP and Season - there is no seasonal data set) or `pve/`; caches and ETag meta are per data source (`*_pve.json`; regular filenames unchanged), only the active mode is refreshed, the icon catalog stays shared. Header chip + menu, `/api/mode`, `mode` in `/api/autoscan/status`.

Profiles (`feat/profiles`): PvP, PvP Season and PvE are separate characters. `data/profiles/<mode>/` holds `progress.json` (task ticks, hideout levels, have-counts), `keep_list.json` (Collector hand-ins + pins) and `settings.json` (`prestige`, `faction`, `trader_levels`, `intel_center_level`, `hideout_management_level`, `flea_offer_slots`); every other setting stays in `data/settings.json`. The active profile is the effective mode; all access goes through `app.profile_path / load_progress / load_keep_list / _load_settings / _save_settings`. The pre-profile files migrate once into the profile active at that moment (old ones kept as `*.migrated.bak`, log in `data/profiles/migration.log`).

## How it plays (what the app does at each step of a session)
Raids are won on frame time and views change every few seconds, so the app works in the lobby only.

1. **Game starts** - the tray watcher opens the app; it warms the item catalog, the font and the icon-hash index in the
   background (a few seconds when warm). The pill says "Waiting for Tarkov" until the game window exists.
2. **Lobby, stash / gear screen** - when a view stays still for ~0.5 s the whole frame is scanned. A **first look**
   arrives after ~2-3 s: every tile with a name ("Looks like ..."), listed under *Checking...*, outlined dotted grey,
   **never** as sell advice and never in a total ("Checking 55 items..." replaces the headline). Certified tiles then
   turn into real rows as they finish (a progress report every ~2 s); the **final** result (all tiles certified,
   sell / keep / Check groups, totals) lands ~8-10 s after a new view (0.4 s when nothing changed: unchanged tiles
   come from the tile cache). Each new view cancels the pending certification of the old one, so while you scroll you
   see first looks (the picture dims as "not the current screen yet"); stop for ~10 s and you get the final.
3. **Open a container / the item-search window / the backpack in the stash screen** - same flow; the window's grid is
   advised like the stash (items you own), your rig / pockets / backpack are listed under *On you* only.
4. **Trader screen (sell tab)** - the stash is dimmed for items the trader will not buy; dimmed tiles are never named
   confidently (they show as Check), everything the trader buys is final like in the stash.
5. **Flea market: offer creation** - the item picker on the left is scanned like a stash (dimmed = not listable).
6. **After a raid: scav loot transfer ("Receive all")** - the incoming window is read at the true cell pitch
   (scroll bar and cut row do not stretch the lattice) so every incoming item gets a final result next to your stash.
7. **In a raid** - nothing at all: with the game log saying `in_raid` (and *Work in raid* off, the default) the app
   stops capturing, detecting, scanning and collecting frames, releases the screen duplication and only re-reads the
   log tail every 2 s; the pill says "In raid - paused". When the log says lobby again the first inventory view
   is scanned fresh. *Work in raid* (opt-in) restores the old behaviour: grab / drop advice from the loot you open
   (a first look takes seconds and every capture costs frame time, which is why it is off).
8. **Game exits** - the watcher closes the app ~15 s later.

## Parked
- Build pricing: the user opens the weapon inspect/modding window so the attachment icons along the bottom are visible; identify each attachment (plain items -> exact via icon hash) and sum a build value. Parked by the user 2026-10-13.
