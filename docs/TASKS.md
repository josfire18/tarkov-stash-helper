# Work queue

Orchestrator → worker. One Opus worker at a time (hard problems), Sonnet workers alongside on
non-overlapping files, orchestrator reviews / tests / merges into `identify-v2` and pushes.
Usage window 06:40–11:40 UTC (2026-10-03), target ≈ 20 %/h, checked every ~25 min.

| # | Task | Worker | Status |
|---|------|--------|--------|
| 1 | Exact-anchor identification: game font extracted from Unity assets, closed-set label render match, pixel-exact game-render match, self-binding, per-tile cache (`feat/exact`) | Opus | running |
| 2 | Auto-scan saves every settled inventory view (lobby + raid) to `data/scenes/live/` as the scene dataset | orchestrator | done (a07af07) |
| 3 | UI restructure: Live (home) · Needs · Settings. Header clutter gone, no region/build buttons, Needs = quests + hideout (level steppers) + Kappa + pins in one list | Sonnet | paused (pacing), resumes after 1 |
| 4 | Scene layer `identify/scene.py`: split the screen by landmarks (stash filter toolbar, slot headers, floating windows with title bar + red ✕), grids per region, roles own / loot / stash / container window / picker; picker + occluded cells never scanned | Opus | after 1 |
| 5 | Live pipeline: incremental scan via tile cache, SSE push, in-raid advice (grab by ₽/slot, quest items, drop cheapest own item when full), scene-filtered sell advice | Opus | after 4 |
| 6 | Hideout levels read from the hideout screen by the auto-scan | Sonnet | after 3 |
| 7 | App icon in the taskbar: own AppUserModelID + window/taskbar icon (shows the Python icon today), exe icon in the spec | orchestrator | queued |
| 8 | Integrate, full test suite, rebuild exe, release | orchestrator | last |

## Findings that shape the work
- Raid recordings on this machine are 1–4 min combat clips: no inventory frames. Real scene data
  comes from the user's own sessions (task 2).
- `identify.grid.detect_grid` merges neighbouring grids that share the lattice phase (gear column +
  item picker + stash became one 21-column panel on a real 1440p frame) → the whole-frame sell scan
  reads rig/backpack/picker items as stash items. Task 4 fixes it.
- tarkov.dev marks only the Collector's 13-quest chain `kappaRequired` since 1.0; hideout progress
  is not in the logs.
