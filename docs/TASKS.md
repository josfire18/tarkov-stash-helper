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
- 10. Rig / backpack / pocket cells (different border style) scanned so "On you" and in-raid Drop advice see your own items. — next window
- 11. Task 6 (hideout levels from the hideout screen). — next window
- 12. Rebuild exe + GitHub release. — orchestrator, ~11:15 UTC
