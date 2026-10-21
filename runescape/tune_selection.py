"""Tune the item-selection filters in backtest_flips.py with out-of-sample checks.

The cached window is split into FOLDS equal periods (A, B, C, ...), each
simulated independently from a fresh budget so one period's luck can't carry
into the next. Settings are chosen using only the tuning folds (all but the
last), ranked by their *worst* tuning-fold profit so a filter has to help in
every period, not just one. The last fold is held out and only reported, as an
honest estimate of how the chosen settings do on data they weren't picked on.

Usage:
    uv run tune_selection.py --stage oat     # each filter alone, across a range of settings
    uv run tune_selection.py --stage combo   # combinations of the filters that held up alone
    uv run tune_selection.py --stage final   # the study's filter set, neighbours and ablations
    uv run tune_selection.py --stage outbid  # outbid amounts on top of SELECTED
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone

from backtest_flips import (
    BUDGET, DATA_DIR, DAY, STRATEGIES, Simulator, Strategy, cached_end, check_times, contiguous_start, fmt_gp,
    fmt_ts, load_market,
)

FOLDS = 3
BASE = STRATEGIES["baseline"]

_DATA = None  # per-worker (items, data_start, start, end)


def load():
    end = cached_end()
    coverage = contiguous_start(0, end)
    mapping = json.loads((DATA_DIR / "mapping.json").read_text())
    items = load_market(coverage, end, mapping)
    data_start = min(int(item.ts[0]) for item in items)
    return items, data_start, data_start + DAY, end


def fold_bounds(start: int, end: int) -> list[tuple[int, int]]:
    edges = [start + (end - start) * i // FOLDS for i in range(FOLDS + 1)]
    return list(zip(edges, edges[1:]))


def simulate(strategy: Strategy, a: int, b: int) -> dict:
    items, data_start, _, _ = _DATA
    sim = Simulator(items, check_times(a, b), b, BUDGET, strategy, data_start=data_start, quiet=True)
    sim.run()
    peak, drawdown = BUDGET, 0
    for _, equity, _, _ in sim.equity_log:
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    item_profits = [led.profit for led in sim.ledgers.values()]
    return {
        "profit": sim.cash - BUDGET,
        "drawdown": drawdown,
        "fills": sim.fills["buy"],
        "worst_item": min(item_profits, default=0),
        "losing_items": sum(p < 0 for p in item_profits),
        "items": len(item_profits),
    }


def _init() -> None:
    global _DATA
    _DATA = load()


def _job(args: tuple[Strategy, int]) -> tuple[str, int, dict]:
    strategy, fold = args
    _, _, start, end = _DATA
    a, b = fold_bounds(start, end)[fold]
    return strategy.name, fold, simulate(strategy, a, b)


# ---------------------------------------------------------------------------
# Search stages
# ---------------------------------------------------------------------------

def oat_configs() -> list[Strategy]:
    """Each filter on its own, over a range of settings, on top of the unfiltered baseline."""
    configs = [BASE]
    for drift, window in itertools.product((0.05, 0.10, 0.15, 0.20, 0.30), (1, 3, 7)):
        configs.append(replace(BASE, name=f"drift {drift:.0%} / {window}d median", max_drift=drift,
                               drift_window_days=window))
    for ratio in (1.25, 1.5, 2.0, 3.0):
        configs.append(replace(BASE, name=f"outlier ratio {ratio:g}x", max_outlier_ratio=ratio))
    for units in (3, 5, 10):
        configs.append(replace(BASE, name=f"conservative margin, last {units}", selection="conservative",
                               recent_units=units))
    for pct in (0.05, 0.10, 0.15, 0.20, 0.30):
        configs.append(replace(BASE, name=f"spread cap {pct:.0%}", max_margin_pct=pct))
    for n in (3, 6, 12, 24):
        configs.append(replace(BASE, name=f"min {n} trading buckets/side", min_trade_buckets=n))
    for spike in (1.5, 2.0, 3.0, 5.0):
        configs.append(replace(BASE, name=f"volume spike cap {spike:g}x", max_volume_spike=spike))
    return configs


def combo_configs() -> list[Strategy]:
    """Factorial over the filters that held up on their own (see the oat stage)."""
    configs = [BASE]
    for drift, selection, spread, spike in itertools.product(
            (None, 0.10, 0.15, 0.20), ("instant", "conservative"), (None, 0.10, 0.15), (None, 3.0)):
        parts = []
        if drift:
            parts.append(f"drift {drift:.0%}/3d")
        if selection == "conservative":
            parts.append("conservative last-3")
        if spread:
            parts.append(f"spread {spread:.0%}")
        if spike:
            parts.append(f"spike {spike:g}x")
        if not parts:
            continue
        configs.append(replace(BASE, name=" + ".join(parts), max_drift=drift, selection=selection,
                               max_margin_pct=spread, max_volume_spike=spike))
    return configs


# Filter sets suggested by study_selection.py (candidate-level study); confirmed here on whole portfolios.
FILTERED = replace(BASE, name="filtered: age≤1h, margin≥3%, ≥6 buckets, drift≤10%/3d, spike≤2x",
                   max_quote_age_h=1, min_margin_pct=0.03, min_trade_buckets=6, max_drift=0.10,
                   drift_window_days=3, max_volume_spike=2.0)


def final_configs() -> list[Strategy]:
    """The full filter set, its neighbours, and each filter removed in turn (ablation)."""
    return [
        BASE,
        replace(BASE, name="drift≤10%/3d only", max_drift=0.10),
        replace(BASE, name="age≤1h only", max_quote_age_h=1),
        replace(BASE, name="age≤1h + drift≤10%/3d", max_quote_age_h=1, max_drift=0.10),
        FILTERED,
        replace(FILTERED, name="filtered, age≤3h", max_quote_age_h=3),
        replace(FILTERED, name="filtered, drift≤5%", max_drift=0.05),
        replace(FILTERED, name="filtered, drift≤15%", max_drift=0.15),
        replace(FILTERED, name="filtered − age", max_quote_age_h=None),
        replace(FILTERED, name="filtered − margin≥3%", min_margin_pct=None),
        replace(FILTERED, name="filtered − buckets≥6", min_trade_buckets=None),
        replace(FILTERED, name="filtered − drift", max_drift=None),
        replace(FILTERED, name="filtered − spike", max_volume_spike=None),
    ]


def outbid_configs() -> list[Strategy]:
    """Outbid amounts on top of the chosen selection filters (plus the unfiltered baseline for reference)."""
    return [BASE] + [replace(SELECTED, name=f"selected, outbid {fmt_gp(v)}", outbid=v)
                     for v in (1, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000)] + [
        replace(SELECTED, name="selected + conservative ranking", selection="conservative"),
    ]


def cap_configs() -> list[Strategy]:
    """Per-item exposure caps on top of selected + 100k outbid."""
    rec = replace(SELECTED, outbid=100_000)
    return [BASE, replace(rec, name="selected, outbid 100K")] + [
        replace(rec, name=f"selected, outbid 100K, item cap {share:.0%}", max_item_share=share)
        for share in (0.15, 0.20, 0.25, 0.33, 0.50)]


SELECTED = replace(STRATEGIES["selected"], name="selected: age≤1h, ≥6 buckets, drift≤10%/3d, spike≤2x")

def slot_configs() -> list[Strategy]:
    """How buy slots get filled, on top of the recommended strategy."""
    rec = STRATEGIES["recommended"]
    return [BASE] + [replace(rec, name=f"recommended, {policy}", slot_policy=policy)
                     for policy in ("max-value", "one-each-if-short", "one-each", "fill-slots")]


STAGES = {"oat": oat_configs, "combo": combo_configs, "final": final_configs, "outbid": outbid_configs, "cap": cap_configs, "slots": slot_configs}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles/redirects default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=STAGES, default="oat")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    args = parser.parse_args()

    configs = STAGES[args.stage]()
    _, _, start, end = load()
    bounds = fold_bounds(start, end)
    labels = [chr(ord("A") + i) for i in range(FOLDS)]
    print(f"{len(configs)} configs × {FOLDS} folds over {fmt_ts(start)} -> {fmt_ts(end)} "
          f"({(end - start) / DAY:.1f} days, {(end - start) / DAY / FOLDS:.1f} days per fold)", flush=True)

    t0 = time.monotonic()
    results: dict[str, dict[int, dict]] = {c.name: {} for c in configs}
    jobs = [(c, f) for c in configs for f in range(FOLDS)]
    with mp.Pool(args.workers, initializer=_init) as pool:
        for n, (name, fold, metrics) in enumerate(pool.imap_unordered(_job, jobs), 1):
            results[name][fold] = metrics
            if n % 10 == 0 or n == len(jobs):
                print(f"\r  {n}/{len(jobs)} simulations ({time.monotonic() - t0:.0f}s)", end="", flush=True)
    print()

    tune = list(range(FOLDS - 1))
    holdout = FOLDS - 1
    base = results[BASE.name]

    def row_key(name: str) -> int:
        return min(results[name][f]["profit"] for f in tune)

    header = ("| Config | " + " | ".join(f"Fold {labels[f]}" for f in tune) + f" | Worst tuning fold "
              f"| Fold {labels[holdout]} (held out) | Max drawdown | Buys | Losing items | Worst item |")
    lines = [
        f"# Selection tuning: {args.stage}",
        "",
        f"Window {fmt_ts(start)} → {fmt_ts(end)}; folds: "
        + ", ".join(f"{labels[i]} = {fmt_ts(a)} → {fmt_ts(b)}" for i, (a, b) in enumerate(bounds)),
        "",
        f"Each fold starts from a fresh {fmt_gp(BUDGET)}. Sorted by worst tuning-fold profit; "
        f"fold {labels[holdout]} is not used for ranking.",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for name in sorted(results, key=row_key, reverse=True):
        r = results[name]
        cells = [f"{fmt_gp(r[f]['profit'])}" for f in tune]
        lines.append(
            f"| {name} | " + " | ".join(cells)
            + f" | **{fmt_gp(row_key(name))}** | {fmt_gp(r[holdout]['profit'])} "
            f"| {fmt_gp(max(r[f]['drawdown'] for f in r))} | {sum(r[f]['fills'] for f in r)} "
            f"| {sum(r[f]['losing_items'] for f in r)}/{sum(r[f]['items'] for f in r)} "
            f"| {fmt_gp(min(r[f]['worst_item'] for f in r))} |")
    lines.append("")
    lines.append(f"Baseline: " + ", ".join(f"{labels[f]} {fmt_gp(base[f]['profit'])}" for f in range(FOLDS)))

    out = DATA_DIR / "runs" / f"tuning-{args.stage}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
