# GE flipping project

## Buy limits

- An item with no buy limit in the wiki mapping is treated as having a **buy limit of 1**, unless it has a
  hand-filled limit in `BUY_LIMIT_OVERRIDES` (e.g. Malachi's claw and tooth = 2). Never treat a missing limit as
  unlimited or size orders on volume alone.
- The default lives in `UNKNOWN_BUY_LIMIT` and the overrides in `BUY_LIMIT_OVERRIDES`, in both
  `backtest_flips.py` and `advisor/index.html`; keep them in sync.

## Unflippable items

- Bonds bought on the GE become untradable, so they can never be flipped. They're listed in `UNFLIPPABLE_ITEMS`
  in both `backtest_flips.py` and `advisor/index.html` (keep in sync); `load_market` and `lab/market.py` never load
  them, and the advisor never recommends them.

## Advisor strategy

- `advisor/index.html` runs the strategy lab's soft-rules champion (`lab/policies/soft_rules.py`), not the
  backtester's `recommended` rules. Its constants are that champion's parameters, rounded, copied from the run's
  `champion.json` (the run is named in the comment above them). When a better champion replaces it, copy the new
  values and keep the selection logic in step with the lab: items best score first, each given its quantity while the
  cash lasts. Two advisor-only tweaks (checked in the lab against the champion on validation): items are sized as if
  the stack were cash × 8 ÷ free slots, and an item the cash left can't buy one of is skipped instead of keeping its
  slot empty.
- Items first traded within 30 days (per the wiki's daily history, which starts 2026-07-23) are flagged as new,
  not rejected. Skipped items and the free-slot count are stored in the browser's localStorage.

## Data

- `sync_cache.py` runs in the background at logon as "RS GE Sync" (tray icon; `install_sync_task.ps1` sets it
  up) and keeps `data/` current, so backtests should normally use `--offline` instead of downloading
  themselves. `uv run sync_cache.py --status` shows its progress.
- The running sync executes the live `sync_cache.py` / `backtest_flips.py` code (the exe is a renamed Python,
  not a frozen build), but only picks up changes when restarted: tray > Quit, then the Start menu shortcut.

## Strategy lab (`lab/`)

- A GPU referee (`lab/env.py`) that scores any policy the same way, plus evolution (`lab/evolve.py`) and a live
  dashboard (`uv run python -m lab.serve`, http://127.0.0.1:8765). Module overview: `lab/__init__.py`.
- **Parity is the referee's correctness gate**: with `obs_lag=0`, no haircut, `tax="backtester"` and the
  backtester's fill model, `RulesPolicy` must reproduce `backtest_flips.py` exactly
  (`uv run python -m lab.parity`). Re-run it after any change to `env.py`, `features.py`, `policies/rules.py`, or to
  the backtester's rules (and port those changes to the lab).
- **Holdout discipline**: splits roll forward weekly (`lab/splits.py`). Nothing selects on holdout or forward data;
  read it only through `splits.open_holdout` or `lab.leaderboard`, which log every look to
  `data/lab/holdout_log.jsonl`. Don't tune anything on forward scores.
- Training defaults are deliberately stricter than the backtester: `obs_lag` 300 s, `share:0.5` fills, 3% haircut on
  the end-of-episode dump, RS3 per-item tax rounding.
- Policies see the market only through `Observation` at the check's index; no item IDs or names go into learned
  policies. `uv run pytest lab/tests` includes the look-ahead test.
- `uv run python -m lab.replay` runs a champion (by default the newest soft-rules one, as the advisor runs it,
  from 1.024b with 6 slots) over the whole archive as one account and writes every decision as GE slot events
  (place / fill / collect / cancel / dump) to `data/lab/replays/`. It reads forward data, so it logs a holdout look.
  The dashboard's `ge.html` plays a replay back as the in-game GE screen, backpack and money pouch.
- `lab.evolve --checks 12:00,18:00,00:00` trains for another check-in schedule (local times; default
  `CHECK_TIMES`). The schedule is saved in the run's config, and `lab.replay` and `lab.leaderboard` use it.
- Runs, caches and logs live under `data/lab/` (gitignored).
