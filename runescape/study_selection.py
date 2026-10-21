"""Candidate-level study: which selection signals predict flips that don't pay off?

Portfolio backtests are a poor way to pick selection filters: each run makes
only a few hundred fills, so a handful of big trades dominate, and changing one
pick changes cash and slots for everything after it. This study instead treats
every eligible candidate at every check as an independent one-unit flip,
simulated with the same rules as backtest_flips.py:

* buy offer at instasell + 1, live until the next check (the strategy
  re-plans every check); fills on the first 5m bucket whose average instasell
  is at or below it;
* once bought, a sell offer at instabuy - 1, repriced at every check, fills
  on the first bucket whose average instabuy is at or above it;
* anything unsold SELL_DEADLINE_S after the buy is dumped at instasell; 2% tax.

"Eligible" = the baseline universe: buy price 75m-500m, 4h volume GP > 100m,
quotes <= 24h old, positive quoted after-tax margin. Each candidate records
the signals the filters look at, then results are bucketed by each signal.
The key column is *realized / quoted*: of the after-tax margin the quotes
promised on filled flips, what fraction was actually captured. A signal is
useful if realization falls off a cliff past some value; that value is the
filter threshold.

Usage:
    uv run study_selection.py     # writes data/runs/study-<stamp>.md
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np

from backtest_flips import (
    DATA_DIR, DAY, FIVE_MIN, MAX_BUY_PRICE, MAX_QUOTE_AGE_S, MIN_4H_VOLUME_GP, MIN_BUY_PRICE, ItemSeries,
    cached_end, check_times, contiguous_start, fmt_gp, fmt_ts, load_market, sale_tax,
)

SELL_DEADLINE_S = 3 * DAY
RECENT_UNITS = 3


@dataclass
class Event:
    t: int
    item: str
    quoted: int  # after-tax margin per unit the quotes promised
    buy_price: int
    filled: bool
    realized: int  # actual after-tax profit per unit (0 if the buy never filled)
    dumped: bool
    score: float  # min(4h volume, buy limit) × quoted margin, the optimizer's ranking
    signals: dict


def first_at_or_below(item: ItemSeries, side: str, lo: int, hi: int, price: float, below: bool) -> int | None:
    """Start ts of the first bucket in [lo, hi) whose avg `side` price is <= (below) / >= price."""
    stamps, prices, _ = item.quotes[side]
    a, b = np.searchsorted(stamps, lo), np.searchsorted(stamps, hi)
    hits = np.nonzero(prices[a:b] <= price if below else prices[a:b] >= price)[0]
    return int(stamps[a + hits[0]]) if len(hits) else None


def simulate_flip(item: ItemSeries, t: int, next_check: int, checks: list[int], end: int,
                  buy_price: int) -> tuple[bool, int, bool]:
    fill = first_at_or_below(item, "instasell", t, next_check, buy_price, below=True)
    if fill is None:
        return False, 0, False
    deadline = min(fill + SELL_DEADLINE_S, end)
    sell_checks = [c for c in checks if fill < c < deadline]
    for i, c in enumerate(sell_checks):
        quote = item.last_price(c, "instabuy")
        if quote is None:
            continue
        price = quote[0] - 1
        until = sell_checks[i + 1] if i + 1 < len(sell_checks) else deadline
        if first_at_or_below(item, "instabuy", c, until, price, below=False) is not None:
            return True, price - sale_tax(price, 1) - buy_price, False
    quote = item.last_price(deadline, "instasell")
    price = quote[0] if quote else 0
    return True, price - sale_tax(price, 1) - buy_price, True


def signals_for(item: ItemSeries, t: int, data_start: int, sell_quote, buy_quote, quoted: int) -> dict:
    s = {}
    for days in (1, 3, 7):
        drift = 0.0
        for side, quote in (("instasell", sell_quote), ("instabuy", buy_quote)):
            median = item.median_price(t, side, days * DAY)
            if median:
                drift = max(drift, abs(quote[0] / median - 1))
        s[f"drift_{days}d"] = drift
    recent_sell = item.recent_price(t, "instasell", RECENT_UNITS)
    recent_buy = item.recent_price(t, "instabuy", RECENT_UNITS)
    recent_margin = (recent_buy - 1) - (recent_sell + 1) - sale_tax(recent_buy - 1, 1)
    s["outlier_ratio"] = quoted / recent_margin if recent_margin > 0 else math.inf
    s["spread_pct"] = (buy_quote[0] - sell_quote[0]) / sell_quote[0]
    s["trade_buckets"] = min(item.trade_buckets(t, "instasell", DAY), item.trade_buckets(t, "instabuy", DAY))
    prior_days = min(7.0, (t - DAY - data_start) / DAY)
    if prior_days >= 1:
        prior_avg = item.volume(t - DAY - int(prior_days * DAY), t - DAY) / prior_days
        s["volume_spike"] = item.volume(t - DAY, t) / prior_avg if prior_avg else math.inf
    else:
        s["volume_spike"] = math.nan
    s["quote_age_h"] = (t - min(sell_quote[1], buy_quote[1])) / 3600
    s["margin_pct"] = quoted / (sell_quote[0] + 1)
    return s


BINS = {
    "drift_1d": [0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, math.inf],
    "drift_3d": [0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, math.inf],
    "drift_7d": [0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, math.inf],
    "outlier_ratio": [-math.inf, 1.0, 1.25, 1.5, 2.0, 3.0, math.inf, math.inf],
    "spread_pct": [0, 0.05, 0.10, 0.15, 0.20, 0.30, math.inf],
    "trade_buckets": [0, 3, 6, 12, 24, 48, math.inf],
    "volume_spike": [0, 0.5, 1.0, 1.5, 2.0, 3.0, math.inf, math.inf],
    "quote_age_h": [0, 1, 3, 6, 12, 24.01],
    "margin_pct": [0, 0.03, 0.05, 0.10, 0.15, 0.20, math.inf],
}
PCT_SIGNALS = {"drift_1d", "drift_3d", "drift_7d", "spread_pct", "margin_pct"}


def bin_label(name: str, lo: float, hi: float) -> str:
    if lo == hi == math.inf:
        return "∞ (no positive recent margin)" if name == "outlier_ratio" else "∞ (no prior volume)"
    fmt = (lambda v: f"{v:.1%}".replace(".0%", "%")) if name in PCT_SIGNALS else (lambda v: f"{v:g}")
    if lo == -math.inf:
        return f"< {fmt(hi)}"
    if hi == math.inf:
        return f"≥ {fmt(lo)}"
    return f"{fmt(lo)} – {fmt(hi)}"


def stats(group: list[Event]) -> dict | None:
    if not group:
        return None
    filled = [e for e in group if e.filled]
    quoted_filled = sum(e.quoted for e in filled)
    return {
        "n": len(group),
        "fill": len(filled) / len(group),
        "realization": sum(e.realized for e in filled) / quoted_filled if quoted_filled else None,
        "ev": sum(e.realized for e in group) / len(group),
        "ev_pct": sum(e.realized / e.buy_price for e in group) / len(group),
        "total": sum(e.realized for e in group),
    }


def fmt_stats(st: dict | None) -> str:
    if st is None:
        return "| – | – | – | – | – "
    realization = "–" if st["realization"] is None else f"{st['realization']:.0%}"
    return f"| {st['n']} | {st['fill']:.0%} | {realization} | {fmt_gp(st['ev'])} | {st['ev_pct']:.2%} "


STATS_HEADER = "| N | Fill | Realized ÷ quoted | EV | EV ÷ price "


def in_bin(e: Event, name: str, lo: float, hi: float) -> bool:
    v = e.signals[name]
    return v == math.inf if lo == hi == math.inf else lo <= v < hi


def summarize(tune: list[Event], hold: list[Event], name: str) -> list[str]:
    edges = BINS[name]
    lines = [f"### {name}", "",
             "| Bin " + STATS_HEADER.replace(" |", " (tune) |") + STATS_HEADER.replace(" |", " (held out) |") + "|",
             "|---|" + "---|" * 10]
    for lo, hi in zip(edges, edges[1:]):
        a = stats([e for e in tune if in_bin(e, name, lo, hi)])
        b = stats([e for e in hold if in_bin(e, name, lo, hi)])
        if a or b:
            lines.append(f"| {bin_label(name, lo, hi)} " + fmt_stats(a) + fmt_stats(b) + "|")
    lines.append("")
    return lines


def top_pool(events: list[Event], k: int | None) -> list[Event]:
    """The k best-scoring candidates at each check (what the optimizer chooses from); all if k is None."""
    if k is None:
        return events
    by_check: dict[int, list[Event]] = {}
    for e in events:
        by_check.setdefault(e.t, []).append(e)
    pool = []
    for group in by_check.values():
        pool += sorted(group, key=lambda e: e.score, reverse=True)[:k]
    return pool


@dataclass(frozen=True)
class Filter:
    """Keep a candidate if signal <= threshold (keep_below) or >= threshold; threshold None = off."""
    label: str
    signal: str
    keep_below: bool
    threshold: float | None
    unit: str = ""  # "%" formats the threshold as a percentage

    def keep(self, signals: dict) -> bool:
        if self.threshold is None:
            return True
        v = signals[self.signal]
        if isinstance(v, float) and math.isnan(v):
            return True
        return v <= self.threshold if self.keep_below else v >= self.threshold

    def describe(self) -> str:
        if self.threshold is None:
            return f"{self.label}: off"
        if self.threshold >= 1e18:
            return self.label
        value = f"{self.threshold:.1%}".replace(".0%", "%") if self.unit == "%" else f"{self.threshold:g}"
        return f"{self.label} {'≤' if self.keep_below else '≥'} {value}"

    def at(self, threshold: float | None, signal: str | None = None, label: str | None = None) -> Filter:
        return replace(self, threshold=threshold, signal=signal or self.signal, label=label or self.label)


# Applied cumulatively, in this order, by the funnel; the sensitivity table varies one at a time.
QUOTE_AGE = Filter("quote age (h)", "quote_age_h", True, 6)
MIN_MARGIN = Filter("quoted margin / price", "margin_pct", False, 0.03, "%")
TRADE_BUCKETS = Filter("5m buckets with trades per side, 24h", "trade_buckets", False, 6)
RECENT_POSITIVE = Filter("last-3-trades margin positive", "outlier_ratio", True, 1e18)
DRIFT = Filter("drift from 3-day median", "drift_3d", True, 0.10, "%")
VOLUME_SPIKE = Filter("24h volume / prior daily avg", "volume_spike", True, 2.0)
FILTERS = [QUOTE_AGE, MIN_MARGIN, TRADE_BUCKETS, RECENT_POSITIVE, DRIFT, VOLUME_SPIKE]

SENSITIVITY = {
    0: [QUOTE_AGE.at(v) for v in (1, 3, 6, 12, None)],
    1: [MIN_MARGIN.at(v) for v in (0.02, 0.03, 0.05, 0.08, None)],
    2: [TRADE_BUCKETS.at(v) for v in (3, 6, 12, 24, None)],
    3: [RECENT_POSITIVE, RECENT_POSITIVE.at(None)],
    4: [DRIFT.at(v) for v in (0.05, 0.075, 0.10, 0.15, 0.20, None)]
       + [DRIFT.at(0.10, "drift_1d", "drift from 1-day median"), DRIFT.at(0.10, "drift_7d", "drift from 7-day median"),
          DRIFT.at(0.15, "drift_7d", "drift from 7-day median")],
    5: [VOLUME_SPIKE.at(v) for v in (1.5, 2.0, 3.0, None)],
}


def apply_filters(events: list[Event], filters: list[Filter]) -> list[Event]:
    return [e for e in events if all(f.keep(e.signals) for f in filters)]


def funnel(events: list[Event], label: str, k: int | None) -> list[str]:
    scope = "all eligible candidates" if k is None else f"top {k} by score at each check"
    lines = [f"### {label}: {scope}", "",
             "| Step | N | Fill | Realized ÷ quoted | EV | EV ÷ price "
             "| Dropped N | Dropped realized ÷ quoted | Dropped EV | Dropped EV ÷ price |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def dropped_cells(dropped: list[Event]) -> str:
        d = stats(dropped)
        if not d:
            return "| 0 | – | – | – |"
        realization = "–" if d["realization"] is None else f"{d['realization']:.0%}"
        return f"| {d['n']} | {realization} | {fmt_gp(d['ev'])} | {d['ev_pct']:.2%} |"

    kept = events
    pool = top_pool(kept, k)
    lines.append("| baseline universe " + fmt_stats(stats(pool)) + dropped_cells([]))
    for f in FILTERS:
        dropped = [e for e in pool if not f.keep(e.signals)]
        kept = [e for e in kept if f.keep(e.signals)]
        pool = top_pool(kept, k)
        lines.append(f"| + {f.describe()} " + fmt_stats(stats(pool)) + dropped_cells(dropped))
    lines.append("")
    return lines


def sensitivity(tune: list[Event], hold: list[Event]) -> list[str]:
    lines = ["## Threshold sensitivity", "",
             "One filter varied at a time, the others at their defaults (**bold** = default). "
             "Pool metrics are for the top 5 / top 20 candidates by score at each check after filtering.",
             "",
             "| Filter setting | Tune: top-5 EV ÷ price | Tune: top-5 realized ÷ quoted | Tune: top-20 EV ÷ price "
             "| Held out: top-5 EV ÷ price | Held out: top-5 realized ÷ quoted | Held out: top-20 EV ÷ price |",
             "|---|---|---|---|---|---|---|"]

    def cells(events: list[Event], filters: list[Filter]) -> str:
        kept = apply_filters(events, filters)
        top5, top20 = stats(top_pool(kept, 5)), stats(top_pool(kept, 20))
        if not top5:
            return "| – | – | – "
        realization = "–" if top5["realization"] is None else f"{top5['realization']:.0%}"
        return f"| {top5['ev_pct']:.2%} | {realization} | {top20['ev_pct']:.2%} "

    for index, variants in SENSITIVITY.items():
        for variant in variants:
            filters = FILTERS[:index] + [variant] + FILTERS[index + 1:]
            name = variant.describe()
            if variant == FILTERS[index]:
                name = f"**{name}**"
            lines.append(f"| {name} " + cells(tune, filters) + cells(hold, filters) + "|")
    lines.append("")
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    end = cached_end()
    coverage = contiguous_start(0, end)
    mapping = json.loads((DATA_DIR / "mapping.json").read_text())
    items = load_market(coverage, end, mapping)
    data_start = min(int(item.ts[0]) for item in items)
    start = data_start + DAY
    checks = check_times(start, end)
    print(f"Studying {fmt_ts(start)} -> {fmt_ts(end)}: {len(checks)} checks, {len(items)} items", flush=True)

    events: list[Event] = []
    for n, t in enumerate(checks):
        next_check = checks[n + 1] if n + 1 < len(checks) else end
        for item in items:
            sell_quote = item.last_price(t, "instasell")
            buy_quote = item.last_price(t, "instabuy")
            if sell_quote is None or buy_quote is None:
                continue
            buy_price, sell_price = sell_quote[0] + 1, buy_quote[0] - 1
            if not (MIN_BUY_PRICE <= buy_price <= MAX_BUY_PRICE):
                continue
            if min(sell_quote[1], buy_quote[1]) + FIVE_MIN < t - MAX_QUOTE_AGE_S:
                continue
            if item.volume(t - DAY, t) / 6 * buy_price <= MIN_4H_VOLUME_GP:
                continue
            quoted = sell_price - buy_price - sale_tax(sell_price, 1)
            if quoted <= 0:
                continue
            vol4h = item.volume(t - DAY, t) / 6
            max_qty = math.floor(min(vol4h, item.limit) if item.limit else vol4h)
            if max_qty < 1:
                continue
            filled, realized, dumped = simulate_flip(item, t, next_check, checks, end, buy_price)
            events.append(Event(t, item.name, quoted, buy_price, filled, realized, dumped, max_qty * quoted,
                                signals_for(item, t, data_start, sell_quote, buy_quote, quoted)))
        if n % 50 == 0:
            print(f"\r  {n}/{len(checks)} checks, {len(events):,} candidates", end="", flush=True)
    print()

    split = start + (end - start) * 2 // 3
    tune = [e for e in events if e.t < split]
    hold = [e for e in events if e.t >= split]
    filled = [e for e in events if e.filled]
    lines = [
        "# Candidate-level selection study",
        "",
        f"Window {fmt_ts(start)} → {fmt_ts(end)} ({(end - start) / DAY:.1f} days, {len(checks)} checks). "
        f"{len(events):,} eligible candidates, {len(filled):,} filled "
        f"({len(filled) / max(len(events), 1):.0%}); overall realized ÷ quoted on filled flips: "
        f"{sum(e.realized for e in filled) / max(sum(e.quoted for e in filled), 1):.0%}.",
        "",
        f"Tuning period: {fmt_ts(start)} → {fmt_ts(split)} ({len(tune):,} candidates). "
        f"Held out: {fmt_ts(split)} → {fmt_ts(end)} ({len(hold):,} candidates); thresholds are chosen "
        "on the tuning period only.",
        "",
        "Each candidate = one unit flipped with the strategy's rules, independent of budget and slots. "
        "*Realized ÷ quoted*: share of the promised after-tax margin captured on filled flips. "
        "*EV*: mean realized profit per candidate, counting unfilled candidates as 0. "
        "*EV ÷ price*: the same as a share of the buy price (return per gp tied up).",
        "",
        "## Filter funnel",
        "",
    ]
    for k in (None, 20, 5):
        lines += funnel(tune, "Tuning period", k)
        lines += funnel(hold, "Held out", k)
    lines += sensitivity(tune, hold)
    lines += ["## Signals one at a time", ""]
    for name in BINS:
        lines += summarize(tune, hold, name)

    out = DATA_DIR / "runs" / f"study-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps({"t": e.t, "item": e.item, "quoted": e.quoted, "buy_price": e.buy_price,
                                "filled": e.filled, "realized": e.realized, "dumped": e.dumped,
                                **{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                   for k, v in e.signals.items()},
                                "inf_signals": [k for k, v in e.signals.items()
                                                if isinstance(v, float) and math.isinf(v)]}) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {out} (+ per-candidate .jsonl)")


if __name__ == "__main__":
    main()
