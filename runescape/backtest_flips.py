"""Backtest a Grand Exchange flipping strategy against prices.runescape.wiki history.

1. Download: the bulk /5m?timestamp= endpoint returns every item's 5-minute
   trade bucket for one timestamp, and the wiki keeps roughly the last 90 days
   of them. Each bucket is cached under data/5m/<UTC date>.json.gz (only items
   priced >= ARCHIVE_MIN_PRICE, to keep it small), newest first, so the cache
   is always one unbroken window ending now that grows backwards while the
   download runs (`--offline` backtests use whatever that window covers).
   Re-runs only fetch what's new, and days that age out of the API stay on disk.

2. Backtest: replays the cache's whole unbroken window, from its start (after a
   24h warm-up for the trailing volume) to its end, with a 1.5b budget and 8
   GE slots, checking in at fixed local times (CHECK_TIMES, America/Chicago).
   The baseline strategy (see STRATEGIES for the variants):

   * At each check, all buy offers are cancelled (refunding unfilled gold),
     anything bought since the last check is put up for sale at instabuy - 1,
     existing sell offers are repriced to instabuy - 1, and the freed buy
     slots/cash are re-allocated with :func:`choose_offers`.
   * The item universe is re-selected at every check from point-in-time data:
     buy price (instasell + 1) 75m-500m, 4h volume GP (trailing 24h volume /
     6 * buy price) over 100m, a positive after-tax margin, and quotes no
     older than MAX_QUOTE_AGE_S.
   * Buy offers sit at instasell + 1. A buy fills in a 5m bucket when that
     bucket's average instasell price is at or below our offer, up to the
     bucket's instasell volume. Sells mirror this against the instabuy side.
   * Sales pay 2% GE tax. At the end of the window, anything still held is
     dumped at the final instasell price (minus 2%).

   Prices are point-in-time: at a check at time t only buckets that had
   completed by t are visible.

Every step is written to data/runs/<stamp>/journal.md so the run can be audited.

Usage:
    uv run backtest_flips.py                        # sync the 5m archive, then backtest `recommended`
    uv run backtest_flips.py --offline              # backtest on the cached archive only
    uv run backtest_flips.py --strategy outbid-1m   # one named variant
    uv run backtest_flips.py --offline --compare    # every variant, side by side
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

API_BASE = "https://prices.runescape.wiki/api/v2/rs"
USER_AGENT = "rs-ge-backtest/0.1 (personal flipping backtest script; GitHub: Isaiah7579)"
REQUEST_INTERVAL_S = 0.2  # ~5 req/s, same ceiling the web UI uses
PUBLISH_LAG_S = 600  # don't cache buckets fresher than this; they may not be published yet
ARCHIVE_MIN_PRICE = 1_000_000  # archive keeps an item's bucket only if either side traded >= this

MIN_BUY_PRICE = 75_000_000
MAX_BUY_PRICE = 500_000_000
MIN_4H_VOLUME_GP = 100_000_000
LOAD_MIN_PRICE = MIN_BUY_PRICE // 3  # buckets below this can't matter to the strategy
MAX_QUOTE_AGE_S = 24 * 3600
BUDGET = 1_500_000_000
GE_SLOTS = 8
GE_TAX = 0.02
BUY_LIMIT_WINDOW_S = 4 * 3600
# Known buy limits missing from the wiki mapping (new items whose limit isn't documented yet)
BUY_LIMIT_OVERRIDES = {
    64195: 2,  # Malachi's claw
    64191: 2,  # Malachi's tooth
}
UNKNOWN_BUY_LIMIT = 1  # items with no limit in the mapping or the overrides
# Items that can't be resold once bought on the GE, so can never be flipped (never loaded)
UNFLIPPABLE_ITEMS = {
    29492,  # Bond: GE-bought bonds become untradable
}
TOP_N_CANDIDATES = 20
DP_GP_UNIT = 1_000_000  # optimizer budget resolution
NEG_VALUE = -(10 ** 18)

LOCAL_TZ = ZoneInfo("America/Chicago")  # CDT in summer, CST after DST ends
CHECK_TIMES = ("12:00", "14:00", "16:30", "19:30", "21:00", "22:00", "23:00", "01:00", "02:00")

FIVE_MIN = 300
DAY = 86400

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "5m"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class Api:
    def __init__(self, interval_s: float = REQUEST_INTERVAL_S) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.interval_s = interval_s
        self.last_request = 0.0
        self.requests = 0  # HTTP requests sent, retries included

    def get(self, path: str, params: dict | None = None):
        for attempt in range(6):
            wait = self.last_request + self.interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            self.requests += 1
            try:
                resp = self.session.get(f"{API_BASE}/{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"{path} {params} kept failing")


def day_path(day: date, directory: Path = ARCHIVE_DIR) -> Path:
    return directory / f"{day.isoformat()}.json.gz"


def read_day(day: date, directory: Path = ARCHIVE_DIR) -> dict[str, dict[str, list]]:
    """{bucket ts: {item id: [avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume]}}"""
    path = day_path(day, directory)
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


def write_day(day: date, buckets: dict, directory: Path = ARCHIVE_DIR) -> None:
    """Add `buckets` to the day's file, keeping buckets another process (sync_cache.py) wrote meanwhile."""
    directory.mkdir(parents=True, exist_ok=True)
    path = day_path(day, directory)
    merged = {**read_day(day, directory), **buckets}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(json.dumps(merged, separators=(",", ":")).encode()))
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: an --offline run is reading the file right now
            time.sleep(0.5)
    tmp.replace(path)


def compact_bucket(data: dict) -> dict[str, list]:
    out = {}
    for item_id, v in data.items():
        high, low = v.get("avgHighPrice"), v.get("avgLowPrice")
        if max(high or 0, low or 0) < ARCHIVE_MIN_PRICE:
            continue
        out[item_id] = [high, v.get("highPriceVolume") or 0, low, v.get("lowPriceVolume") or 0]
    return out


def utc_day(ts: int) -> date:
    return datetime.fromtimestamp(ts, timezone.utc).date()


def day_start(day: date) -> int:
    return int(datetime.combine(day, dtime(0, 0), timezone.utc).timestamp())


def find_earliest_bucket(api: Api, end: int) -> int:
    """Binary-search the oldest 5m bucket the API still serves (retention is ~90 days)."""
    def has_data(ts: int) -> bool:
        return bool(api.get("5m", {"timestamp": ts})["data"])

    lo, hi = end - 120 * DAY, end - 30 * DAY
    if has_data(lo):
        return lo
    while hi - lo > FIVE_MIN:
        mid = (lo + hi) // 2 // FIVE_MIN * FIVE_MIN
        if has_data(mid):
            hi = mid
        else:
            lo = mid
    return hi


def sync_archive(api: Api, start: int, end: int) -> None:
    """Fetch every 5m bucket in [start, end) that isn't cached yet, newest first.

    Walking backwards keeps the cache contiguous from `end`, so a backtest run
    mid-download sees an ever-longer window instead of a window with holes.
    """
    days = []
    day = utc_day(end - 1)
    while day_start(day) + DAY > start:
        cached = read_day(day)
        lo, hi = max(start, day_start(day)), min(end, day_start(day) + DAY)
        todo = [ts for ts in range(hi - FIVE_MIN, lo - 1, -FIVE_MIN) if str(ts) not in cached]
        if todo:
            days.append((day, cached, todo))
        day -= timedelta(days=1)

    total = sum(len(todo) for _, _, todo in days)
    if not total:
        print("5m archive is up to date.")
        return
    print(f"Fetching {total:,} 5m buckets (~{total * REQUEST_INTERVAL_S / 60:.0f} min at "
          f"{1 / REQUEST_INTERVAL_S:.0f} req/s)...")
    done, t_begin = 0, time.monotonic()
    for day, cached, todo in days:
        try:
            for i, ts in enumerate(todo, 1):
                cached[str(ts)] = compact_bucket(api.get("5m", {"timestamp": ts})["data"])
                done += 1
                if i % 48 == 0:
                    write_day(day, cached)
                if done % 100 == 0 or done == total:
                    rate = done / (time.monotonic() - t_begin)
                    print(f"\r  {done:,}/{total:,} buckets ({day}), ~{(total - done) / rate / 60:.0f} min left ",
                          end="", flush=True)
        finally:
            write_day(day, cached)
        covered = max(start, day_start(day))
        print(f"\n  cache now covers {fmt_ts(covered)} -> {fmt_ts(end)} ({(end - covered) / DAY:.1f} days)",
              flush=True)


def contiguous_start(floor: int, end: int) -> int:
    """Oldest bucket from which every 5m bucket up to `end` is cached (no earlier than `floor`)."""
    day = utc_day(end - 1)
    while day_start(day) + DAY > floor:
        cached = read_day(day)
        lo, hi = max(floor, day_start(day)), min(end, day_start(day) + DAY)
        for ts in range(hi - FIVE_MIN, lo - 1, -FIVE_MIN):
            if str(ts) not in cached:
                return ts + FIVE_MIN
        day -= timedelta(days=1)
    return floor


def cached_end() -> int:
    files = sorted(ARCHIVE_DIR.glob("*.json.gz"))
    if not files:
        raise SystemExit("No cached 5m archive; run without --offline first.")
    buckets = read_day(date.fromisoformat(files[-1].name.split(".")[0]))
    return max(int(ts) for ts in buckets) + FIVE_MIN


# ---------------------------------------------------------------------------
# Market history
# ---------------------------------------------------------------------------

class ItemSeries:
    """One item's 5m buckets, indexed for point-in-time lookups."""

    def __init__(self, item_id: int, meta: dict, rows: list[tuple]):
        rows.sort()
        self.id = item_id
        self.name = meta.get("name", f"Item {item_id}")
        self.limit = BUY_LIMIT_OVERRIDES.get(item_id) or meta.get("limit") or UNKNOWN_BUY_LIMIT
        self.buckets = {r[0]: r for r in rows}  # ts -> (ts, avgHigh, highVol, avgLow, lowVol)
        self.ts = np.array([r[0] for r in rows], dtype=np.int64)
        self.cum_volume = np.concatenate([[0], np.cumsum([r[2] + r[4] for r in rows])])
        self.quotes = {}
        for side, price_col, volume_col in (("instabuy", 1, 2), ("instasell", 3, 4)):
            priced = [(r[0], r[price_col], max(r[volume_col], 1)) for r in rows if r[price_col] is not None]
            self.quotes[side] = (np.array([p[0] for p in priced], dtype=np.int64),
                                 np.array([p[1] for p in priced], dtype=np.float64),
                                 np.array([p[2] for p in priced], dtype=np.int64))

    def last_price(self, t: int, side: str) -> tuple[int, int] | None:
        """(price, bucket ts) of the latest bucket completed by t with trades on `side`."""
        stamps, prices, _ = self.quotes[side]
        k = np.searchsorted(stamps, t - FIVE_MIN, side="right") - 1
        if k < 0:
            return None
        return round(prices[k]), int(stamps[k])

    def recent_price(self, t: int, side: str, units: int) -> int | None:
        """Volume-weighted price of the last `units` units traded on `side` before t.

        The API only has 5m bucket averages, so "the last 3 trades" is
        approximated by walking back bucket by bucket until 3 units are covered.
        """
        stamps, prices, volumes = self.quotes[side]
        k = np.searchsorted(stamps, t - FIVE_MIN, side="right") - 1
        total, value = 0, 0.0
        while k >= 0 and total < units:
            take = min(int(volumes[k]), units - total)
            value += prices[k] * take
            total += take
            k -= 1
        return round(value / total) if total else None

    def trade_buckets(self, t: int, side: str, window: int) -> int:
        """Number of completed 5m buckets in the `window` seconds before t with trades on `side`."""
        stamps = self.quotes[side][0]
        return int(np.searchsorted(stamps, t - FIVE_MIN, side="right") - np.searchsorted(stamps, t - window))

    def median_price(self, t: int, side: str, window: int) -> float | None:
        """Median bucket price on `side` over the `window` seconds before t."""
        stamps, prices, _ = self.quotes[side]
        a = np.searchsorted(stamps, t - window, side="left")
        b = np.searchsorted(stamps, t - FIVE_MIN, side="right")
        return float(np.median(prices[a:b])) if b > a else None

    def volume(self, lo: int, hi: int) -> int:
        """Units traded in buckets that started at/after lo and completed by hi."""
        a = np.searchsorted(self.ts, lo, side="left")
        b = np.searchsorted(self.ts, hi - FIVE_MIN, side="right")
        return int(self.cum_volume[b] - self.cum_volume[a])


def load_market(start: int, end: int, mapping: list[dict]) -> list[ItemSeries]:
    meta_by_id = {m["id"]: m for m in mapping}
    rows: dict[int, list[tuple]] = {}
    day = utc_day(start)
    while day_start(day) < end:
        for ts_str, bucket in read_day(day).items():
            ts = int(ts_str)
            if not start <= ts < end:
                continue
            for id_str, (high, high_vol, low, low_vol) in bucket.items():
                if max(high or 0, low or 0) < LOAD_MIN_PRICE or int(id_str) in UNFLIPPABLE_ITEMS:
                    continue
                rows.setdefault(int(id_str), []).append((ts, high, high_vol, low, low_vol))
        day += timedelta(days=1)
    return [ItemSeries(item_id, meta_by_id.get(item_id, {}), r) for item_id, r in rows.items()]


def check_times(start: int, end: int) -> list[int]:
    """Every CHECK_TIMES local wall-clock time in [start, end)."""
    out = []
    day = datetime.fromtimestamp(start, LOCAL_TZ).date() - timedelta(days=1)
    last = datetime.fromtimestamp(end, LOCAL_TZ).date()
    while day <= last:
        for hhmm in CHECK_TIMES:
            hour, minute = map(int, hhmm.split(":"))
            ts = int(datetime.combine(day, dtime(hour, minute), LOCAL_TZ).timestamp())
            if start <= ts < end:
                out.append(ts)
        day += timedelta(days=1)
    return sorted(out)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    """One variant of the strategy; --compare runs every entry in STRATEGIES."""
    name: str
    description: str
    outbid: int = 1  # buy at instasell + outbid, sell at instabuy - outbid
    selection: str = "instant"  # margin used to rank/optimize: instant | recent | conservative
    pricing: str = "instant"  # what offers are priced off: instant | recent
    recent_units: int = 3  # "recent" = VWAP of the last N units traded on each side
    max_outlier_ratio: float | None = None  # skip if instant margin > ratio × recent margin
    max_drift: float | None = None  # skip if either quote is > drift away from its median price
    drift_window_days: float = 3  # lookback for that median
    max_volume_spike: float | None = None  # skip if 24h volume > x × average daily volume of the prior week
    max_margin_pct: float | None = None  # skip if raw margin / instasell > x
    min_trade_buckets: int | None = None  # skip unless each side traded in >= n 5m buckets over the last 24h
    max_quote_age_h: float | None = None  # skip if either quote's bucket started more than this long ago
    min_margin_pct: float | None = None  # skip if after-tax offer margin / buy price < x
    max_item_share: float | None = None  # cap one item's stock + buy offer at this share of equity
    item_share_min_one: bool = False  # ...but always allow 1 unit (small stacks can't afford 1 within the share)
    # Looser quote-age limits that only fill the cash and slots the strict picks leave over, loosest last
    fallback_quote_age_h: tuple[float, ...] = ()
    # How buy slots are filled: max-value (optimizer's pick) | one-each (1 unit per item) |
    # one-each-if-short (max-value, but 1 unit per item if that leaves slots empty) | fill-slots (use every slot)
    slot_policy: str = "max-value"


STRATEGIES = {s.name: s for s in [
    Strategy("baseline", "offers at instasell+1 / instabuy-1, ranked on the latest bucket's margin"),
    Strategy("outbid-100k", "baseline, outbidding by 100k", outbid=100_000),
    Strategy("outbid-250k", "baseline, outbidding by 250k", outbid=250_000),
    Strategy("outbid-500k", "baseline, outbidding by 500k", outbid=500_000),
    Strategy("outbid-1m", "baseline, outbidding by 1m", outbid=1_000_000),
    Strategy("select-recent", "rank/optimize on the last-3-trades margin; offers still at the latest prices",
             selection="recent"),
    Strategy("select-conservative", "rank/optimize on min(latest margin, last-3-trades margin)",
             selection="conservative"),
    Strategy("price-recent", "rank/optimize and price offers off the last-3-trades VWAP",
             selection="recent", pricing="recent"),
    Strategy("outlier-filter", "baseline, skipping items whose latest margin is > 2x their last-3-trades margin",
             max_outlier_ratio=2.0),
    Strategy("drift-filter", "baseline, skipping items whose latest quote is > 10% off its 3-day median",
             max_drift=0.10),
    # Chosen with study_selection.py + tune_selection.py; see RESEARCH.md for the evidence.
    Strategy("selected", "baseline + selection filters: quotes <= 1h old, >= 6 trading 5m buckets per side in "
             "24h, quotes within 10% of their 3-day median, 24h volume <= 2x the prior daily average",
             max_quote_age_h=1, min_trade_buckets=6, max_drift=0.10, max_volume_spike=2.0),
    Strategy("recommended-1h", "selected, outbidding by 100k, no item over 25% of equity", outbid=100_000,
             max_quote_age_h=1, min_trade_buckets=6, max_drift=0.10, max_volume_spike=2.0, max_item_share=0.25),
    Strategy("recommended", "recommended-1h, but always allowing 1 unit of an item, and filling leftover cash and "
             "slots with items whose quotes are <= 2h, then <= 4h old", outbid=100_000, max_quote_age_h=1,
             min_trade_buckets=6, max_drift=0.10, max_volume_spike=2.0, max_item_share=0.25,
             item_share_min_one=True, fallback_quote_age_h=(2, 4)),
]}


@dataclass
class Candidate:
    item: ItemSeries
    instasell: tuple[int, int]
    instabuy: tuple[int, int]
    buy_price: int
    sell_price: int
    raw_margin: int  # instabuy - instasell
    margin: int  # after-tax profit per unit the optimizer maximizes (depends on Strategy.selection)
    recent_margin: int | None  # after-tax margin at the last-3-trades prices
    four_hr_volume: float
    limit_left: int | None
    max_qty: int
    score: int


def choose_offers(candidates: list[Candidate], cash: int, free_slots: int,
                  exact: bool = False) -> list[tuple[Candidate, int]]:
    """Pick (candidate, quantity) buy offers maximizing sum(qty * after-tax margin).

    Constraints: at most `free_slots` offers, sum(qty * buy_price) <= cash, and
    qty <= min(4h volume, remaining 4h buy limit) per item.

    Best-effort rather than exact: only the top TOP_N_CANDIDATES items by
    score = min(4h volume, buy limit) * after-tax margin are considered,
    then a bounded-knapsack dynamic program with an offer-count dimension is
    solved over them. Unit costs are rounded *up* to DP_GP_UNIT (1m gp), so the
    answer is always affordable but may leave up to ~1% of a slot's value on
    the table. Within those approximations the pick is optimal.
    """
    pool = sorted(candidates, key=lambda c: c.score, reverse=True)[:TOP_N_CANDIDATES]
    budget_units = cash // DP_GP_UNIT
    if free_slots <= 0 or budget_units <= 0 or not pool:
        return []

    # dp[k, b] = best total value using at most k offers and b budget units
    # With exact=True, dp[k, b] is the best value using exactly k offers (unreachable states are NEG), and
    # the answer uses as many offers as any affordable combination allows.
    dp = np.zeros((free_slots + 1, budget_units + 1), dtype=np.int64)
    if exact:
        dp[1:] = NEG_VALUE
    stages = []
    for cand in pool:
        unit_cost = -(-cand.buy_price // DP_GP_UNIT)
        qmax = min(cand.max_qty, budget_units // unit_cost)
        take = None
        if qmax > 0 and cand.margin > 0:
            new = dp.copy()
            take = np.zeros(dp.shape, dtype=np.int32)
            for q in range(1, qmax + 1):
                shift = q * unit_cost
                value = dp[:-1, :budget_units + 1 - shift] + q * cand.margin
                target = new[1:, shift:]
                better = value > target
                target[better] = value[better]
                take[1:, shift:][better] = q
            dp = new
        stages.append((cand, unit_cost, take))

    picks = []
    k, b = free_slots, budget_units
    if exact:
        k = max(k for k in range(free_slots + 1) if dp[k, budget_units] > NEG_VALUE // 2)
    for cand, unit_cost, take in reversed(stages):
        q = int(take[k, b]) if take is not None else 0
        if q:
            picks.append((cand, q))
            k -= 1
            b -= q * unit_cost
    return list(reversed(picks))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def fmt_gp(n: float) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}K"
    return f"{sign}{n:.0f}"


def local(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, LOCAL_TZ)


def fmt_ts(ts: int) -> str:
    return local(ts).strftime("%Y-%m-%d %H:%M %Z")


def fmt_bucket(ts: int) -> str:
    return local(ts).strftime("%m-%d %H:%M")


def fmt_hm(ts: int) -> str:
    return local(ts).strftime("%H:%M")


def sale_tax(price: int, qty: int) -> int:
    return int(price * qty * GE_TAX)


@dataclass
class BuyOffer:
    item: ItemSeries
    price: int
    qty: int
    filled: int = 0


@dataclass
class SellOffer:
    item: ItemSeries
    price: int
    remaining: int


@dataclass
class Ledger:
    bought: int = 0
    spent: int = 0
    sold: int = 0
    sell_net: int = 0
    tax: int = 0
    liquidated: int = 0
    liquidation_net: int = 0

    @property
    def profit(self) -> int:
        return self.sell_net + self.liquidation_net - self.spent


@dataclass
class Simulator:
    items: list[ItemSeries]
    checks: list[int]
    t_end: int
    cash: int
    strategy: Strategy
    data_start: int = 0  # first bucket in the archive (for lookbacks that would run off its start)
    quiet: bool = False  # no journal or progress output (parameter sweeps)
    # How a 5m bucket fills an offer once its average price reaches ours. The API only has bucket averages:
    #   average - every unit in the bucket fills (optimistic: some may have traded on the wrong side of us)
    #   floor   - 1 unit, the most the average proves (an average at/below our bid needs >= 1 trade at/below it)
    fill_model: str = "average"
    journal: list[str] = field(default_factory=list)
    buy_offers: list[BuyOffer] = field(default_factory=list)
    sell_offers: dict[int, SellOffer] = field(default_factory=dict)
    held: dict[int, int] = field(default_factory=dict)  # bought, not yet listed
    purchases: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    ledgers: dict[int, Ledger] = field(default_factory=dict)
    equity_log: list[tuple[int, int, int, int]] = field(default_factory=list)  # (t, equity, buys, sells)
    fills: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.by_id = {item.id: item for item in self.items}

    def fill_cap(self, bucket_volume: int) -> int:
        return 1 if self.fill_model == "floor" else bucket_volume

    def count_units(self, side: str, qty: int, bucket_volume: int) -> None:
        """Tally filled units, and how many of them only the bucket average justifies (beyond the 1 it proves)."""
        self.fills[f"{side}_units"] += qty
        if bucket_volume > 1:
            self.fills[f"{side}_units_from_averages"] += max(0, qty - 1)

    def log(self, line: str = "") -> None:
        if not self.quiet:
            self.journal.append(line)

    def ledger(self, item: ItemSeries) -> Ledger:
        return self.ledgers.setdefault(item.id, Ledger())

    def run(self) -> None:
        current_day = None
        for n, t in enumerate(self.checks):
            if local(t).date() != current_day:
                current_day = local(t).date()
                self.log(f"## {current_day.isoformat()}")
                self.log()
                if n and not self.quiet:
                    print(f"  {current_day}  equity {fmt_gp(self.equity_log[-1][1])}", flush=True)
            self.checkpoint(t, n + 1)
            nxt = self.checks[n + 1] if n + 1 < len(self.checks) else self.t_end
            self.replay_market(t, nxt)
        self.finish()

    def equity(self, t: int) -> int:
        """Cash + gold reserved in buy offers + inventory marked at instasell less tax."""
        value = self.cash + sum(o.price * (o.qty - o.filled) for o in self.buy_offers)
        units = dict(self.held)
        for item_id, offer in self.sell_offers.items():
            units[item_id] = units.get(item_id, 0) + offer.remaining
        for item_id, qty in units.items():
            quote = self.by_id[item_id].last_price(t, "instasell")
            if quote:
                value += quote[0] * qty - sale_tax(quote[0], qty)
        return value

    # -- check ---------------------------------------------------------------

    def checkpoint(self, t: int, number: int) -> None:
        equity = self.equity(t)
        self.equity_log.append((t, equity, self.fills["buy"], self.fills["sell"]))
        self.log(f"### Check {number}: {fmt_ts(t)}")
        self.log()
        self.log(f"Cash **{fmt_gp(self.cash)}** ({self.cash:,} gp); equity (cash + reserved + stock at "
                 f"instasell less tax) **{fmt_gp(equity)}**")
        self.log()

        if self.buy_offers:
            self.log("**Cancel buy offers** (slots re-planned from scratch every check):")
            for offer in self.buy_offers:
                refund = offer.price * (offer.qty - offer.filled)
                self.cash += refund
                self.log(f"- {offer.item.name}: filled {offer.filled}/{offer.qty} @ {offer.price:,}; "
                         f"refund {fmt_gp(refund)}")
            self.buy_offers = []
            self.log()

        if self.held:
            self.log("**List newly bought items for sale:**")
            for item_id, qty in self.held.items():
                item = self.by_id[item_id]
                if item_id in self.sell_offers:
                    self.sell_offers[item_id].remaining += qty
                    self.log(f"- {item.name}: +{qty} added to existing sell offer")
                else:
                    self.sell_offers[item_id] = SellOffer(item, 0, qty)
                    self.log(f"- {item.name}: new sell offer for {qty}")
            self.held = {}
            self.log()

        if self.sell_offers:
            s = self.strategy
            basis = "last-3-trades instabuy" if s.pricing == "recent" else "instabuy"
            self.log(f"**Sell offers (re)priced to {basis} - {s.outbid:,}:**")
            for offer in self.sell_offers.values():
                quote = offer.item.last_price(t, "instabuy")
                if quote is None:
                    self.log(f"- {offer.item.name}: no instabuy history; keeping {offer.price:,}")
                    continue
                if s.pricing == "recent":
                    quote = (offer.item.recent_price(t, "instabuy", s.recent_units), quote[1])
                old = offer.price
                offer.price = quote[0] - s.outbid
                change = "new" if old == 0 else f"{old:,} → {offer.price:,}"
                self.log(f"- {offer.item.name} × {offer.remaining}: {change} "
                         f"({basis} {quote[0]:,}, latest bucket {fmt_bucket(quote[1])})")
            self.log()

        candidates = self.rank_candidates(t, equity)
        free_slots = GE_SLOTS - len(self.sell_offers)
        self.log(f"Slots: {len(self.sell_offers)} selling, **{free_slots} free for buys**; "
                 f"cash available **{fmt_gp(self.cash)}**")
        self.log()
        picks = [(cand, qty, None) for cand, qty in self.plan_offers(candidates, free_slots)]
        # Fallback: what the strict picks leave over goes to items with older (but not stale) quotes
        for age_h in self.strategy.fallback_quote_age_h:
            cash_left = self.cash - sum(cand.buy_price * qty for cand, qty, _ in picks)
            slots_left = free_slots - len(picks)
            if slots_left <= 0 or cash_left < MIN_BUY_PRICE:
                break
            picked = {cand.item.id for cand, _, _ in picks}
            extra = [c for c in self.rank_candidates(t, equity, max_quote_age_h=age_h) if c.item.id not in picked]
            picks += [(cand, qty, age_h) for cand, qty in choose_offers(extra, cash_left, slots_left)]

        if picks:
            self.log("**Place buy offers** (optimizer picks):")
            for cand, qty, fallback_age_h in picks:
                cost = cand.buy_price * qty
                self.cash -= cost
                self.buy_offers.append(BuyOffer(cand.item, cand.buy_price, qty))
                self.log(f"- {cand.item.name}: {qty} × {cand.buy_price:,} = {fmt_gp(cost)} "
                         f"(expected {qty} × {fmt_gp(cand.margin)} after tax = {fmt_gp(qty * cand.margin)})"
                         + (f" [fallback: quotes ≤ {fallback_age_h:g}h]" if fallback_age_h else ""))
            self.log(f"- Cash left after reserving: {fmt_gp(self.cash)}")
        else:
            self.log("**No buy offers placed.**")
        self.log()

    def plan_offers(self, candidates: list[Candidate], free_slots: int) -> list[tuple[Candidate, int]]:
        policy = self.strategy.slot_policy
        one_each = [replace(c, max_qty=1) for c in candidates]
        if policy == "one-each":
            return choose_offers(one_each, self.cash, free_slots)
        if policy == "fill-slots":
            return choose_offers(candidates, self.cash, free_slots, exact=True)
        picks = choose_offers(candidates, self.cash, free_slots)
        if policy == "one-each-if-short" and len(picks) < free_slots and len(candidates) > len(picks):
            spread = choose_offers(one_each, self.cash, free_slots)
            if len(spread) > len(picks):
                picks = spread
        return picks

    def rank_candidates(self, t: int, equity: int, max_quote_age_h: float | None = None) -> list[Candidate]:
        """Eligible candidates, best score first. max_quote_age_h overrides the strategy's (fallback rounds)."""
        s = self.strategy
        if max_quote_age_h is not None:
            s = replace(s, max_quote_age_h=max_quote_age_h)
        candidates = []
        skipped = Counter()
        for item in self.items:
            sell_quote = item.last_price(t, "instasell")
            buy_quote = item.last_price(t, "instabuy")
            if sell_quote is None or buy_quote is None:
                skipped["no price history yet"] += 1
                continue
            recent_sell = item.recent_price(t, "instasell", s.recent_units)
            recent_buy = item.recent_price(t, "instabuy", s.recent_units)
            if s.pricing == "recent":
                buy_price, sell_price = recent_sell + s.outbid, recent_buy - s.outbid
            else:
                buy_price, sell_price = sell_quote[0] + s.outbid, buy_quote[0] - s.outbid
            if not (MIN_BUY_PRICE <= buy_price <= MAX_BUY_PRICE):
                skipped["buy price out of range"] += 1
                continue
            if min(sell_quote[1], buy_quote[1]) + FIVE_MIN < t - MAX_QUOTE_AGE_S:
                skipped["quote older than 24h"] += 1
                continue
            vol4h = item.volume(t - DAY, t) / 6
            if vol4h * buy_price <= MIN_4H_VOLUME_GP:
                skipped["4h volume GP ≤ 100M"] += 1
                continue
            offer_margin = sell_price - buy_price - sale_tax(sell_price, 1)
            recent_sell_price = recent_buy - s.outbid
            recent_margin = recent_sell_price - (recent_sell + s.outbid) - sale_tax(recent_sell_price, 1)
            margin = {"instant": offer_margin, "recent": recent_margin,
                      "conservative": min(offer_margin, recent_margin)}[s.selection]
            if margin <= 0:
                skipped["no profit after tax"] += 1
                continue
            if s.max_quote_age_h is not None and t - min(sell_quote[1], buy_quote[1]) > s.max_quote_age_h * 3600:
                skipped[f"quote older than {s.max_quote_age_h:g}h"] += 1
                continue
            if s.min_margin_pct is not None and offer_margin < s.min_margin_pct * buy_price:
                skipped[f"after-tax margin < {s.min_margin_pct:.0%} of price"] += 1
                continue
            if s.max_outlier_ratio is not None and offer_margin > s.max_outlier_ratio * recent_margin:
                skipped[f"outlier: latest margin > {s.max_outlier_ratio:g}x last-{s.recent_units} margin"] += 1
                continue
            if s.max_drift is not None:
                drifted = False
                for side, quote in (("instasell", sell_quote), ("instabuy", buy_quote)):
                    median = item.median_price(t, side, int(s.drift_window_days * DAY))
                    drifted |= median is not None and abs(quote[0] / median - 1) > s.max_drift
                if drifted:
                    skipped[f"drift: quote > {s.max_drift:.0%} off {s.drift_window_days:g}-day median"] += 1
                    continue
            if s.max_margin_pct is not None and buy_quote[0] - sell_quote[0] > s.max_margin_pct * sell_quote[0]:
                skipped[f"spread > {s.max_margin_pct:.0%} of price"] += 1
                continue
            if s.min_trade_buckets is not None and min(item.trade_buckets(t, "instasell", DAY),
                                                       item.trade_buckets(t, "instabuy", DAY)) < s.min_trade_buckets:
                skipped[f"< {s.min_trade_buckets} trading 5m buckets per side in 24h"] += 1
                continue
            if s.max_volume_spike is not None:
                prior_days = min(7.0, (t - DAY - self.data_start) / DAY)
                if prior_days >= 1:
                    prior_avg = item.volume(t - DAY - int(prior_days * DAY), t - DAY) / prior_days
                    if vol4h * 6 > s.max_volume_spike * prior_avg:
                        skipped[f"volume spike: 24h > {s.max_volume_spike:g}x prior daily avg"] += 1
                        continue
            limit_left = None
            if item.limit:
                recent = self.purchases.get(item.id, [])
                recent[:] = [(ts, q) for ts, q in recent if ts > t - BUY_LIMIT_WINDOW_S]
                limit_left = max(0, item.limit - sum(q for _, q in recent))
            max_qty = math.floor(vol4h if limit_left is None else min(vol4h, limit_left))
            if s.max_item_share is not None:
                held = self.sell_offers[item.id].remaining if item.id in self.sell_offers else 0
                share_qty = math.floor(s.max_item_share * equity / buy_price)
                if s.item_share_min_one:
                    share_qty = max(share_qty, 1)
                max_qty = min(max_qty, share_qty - held)
            if max_qty < 1:
                skipped["cap < 1 unit (volume or buy limit)"] += 1
                continue
            candidates.append(Candidate(item, sell_quote, buy_quote, buy_price, sell_price,
                                        buy_quote[0] - sell_quote[0], margin, recent_margin, vol4h,
                                        limit_left, max_qty, max_qty * margin))

        candidates.sort(key=lambda c: c.score, reverse=True)
        self.log(f"<details><summary>Candidate ranking"
                 f"{f' (quotes ≤ {s.max_quote_age_h:g}h)' if s.max_quote_age_h is not None else ''}: "
                 f"{len(candidates)} eligible, top "
                 f"{min(len(candidates), TOP_N_CANDIDATES)} go to the optimizer</summary>")
        self.log()
        self.log("| # | Item | Instasell (bucket) | Instabuy (bucket) | Raw margin | Ranked margin "
                 "| Last-3 margin | Buy @ | Sell @ | 4h vol | Limit left | Max qty | Score |")
        self.log("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, c in enumerate(candidates[:TOP_N_CANDIDATES], 1):
            limit_txt = "∞" if c.limit_left is None else str(c.limit_left)
            self.log(f"| {i} | {c.item.name} | {c.instasell[0]:,} ({fmt_bucket(c.instasell[1])}) "
                     f"| {c.instabuy[0]:,} ({fmt_bucket(c.instabuy[1])}) | {fmt_gp(c.raw_margin)} "
                     f"| {fmt_gp(c.margin)} | {fmt_gp(c.recent_margin)} | {c.buy_price:,} | {c.sell_price:,} "
                     f"| {c.four_hr_volume:.1f} | {limit_txt} | {c.max_qty} | {fmt_gp(c.score)} |")
        self.log()
        self.log("Skipped: " + (", ".join(f"{reason} ({n})" for reason, n in skipped.most_common()) or "none"))
        self.log()
        self.log("</details>")
        self.log()
        return candidates

    # -- market replay -------------------------------------------------------

    def replay_market(self, start: int, end: int) -> None:
        self.log(f"#### Market {fmt_hm(start)}–{fmt_hm(end)}")
        self.log()
        traces: dict[str, list[str]] = {}
        events = []
        finished_sells = set()
        for ts in range(start, end, FIVE_MIN):
            for offer in self.buy_offers:
                if offer.filled >= offer.qty:
                    continue
                bucket = offer.item.buckets.get(ts)
                if bucket is None or not bucket[4]:
                    continue
                avg_low, low_vol = bucket[3], bucket[4]
                matched = avg_low <= offer.price
                qty = min(offer.qty - offer.filled, self.fill_cap(low_vol)) if matched else 0
                self.count_units("buy", qty, low_vol)
                traces.setdefault(f"BUY {offer.item.name} @ {offer.price:,}", []).append(
                    f"| {fmt_hm(ts)} | instasell avg {avg_low:,.0f} × {low_vol} | {'≤ offer ✔' if matched else '✘'} "
                    f"| {qty} |")
                if qty:
                    offer.filled += qty
                    self.fills["buy"] += 1
                    self.held[offer.item.id] = self.held.get(offer.item.id, 0) + qty
                    self.purchases.setdefault(offer.item.id, []).append((ts, qty))
                    led = self.ledger(offer.item)
                    led.bought += qty
                    led.spent += qty * offer.price
                    events.append(f"- {fmt_hm(ts)} **BUY FILL** {offer.item.name}: {qty} @ {offer.price:,} "
                                  f"({offer.filled}/{offer.qty} filled)")
            for offer in self.sell_offers.values():
                if offer.remaining <= 0:
                    continue
                bucket = offer.item.buckets.get(ts)
                if bucket is None or not bucket[2]:
                    continue
                avg_high, high_vol = bucket[1], bucket[2]
                matched = avg_high >= offer.price
                qty = min(offer.remaining, self.fill_cap(high_vol)) if matched else 0
                self.count_units("sell", qty, high_vol)
                traces.setdefault(f"SELL {offer.item.name} @ {offer.price:,}", []).append(
                    f"| {fmt_hm(ts)} | instabuy avg {avg_high:,.0f} × {high_vol} | {'≥ offer ✔' if matched else '✘'} "
                    f"| {qty} |")
                if qty:
                    gross = qty * offer.price
                    tax = sale_tax(offer.price, qty)
                    self.cash += gross - tax
                    offer.remaining -= qty
                    self.fills["sell"] += 1
                    led = self.ledger(offer.item)
                    led.sold += qty
                    led.sell_net += gross - tax
                    led.tax += tax
                    events.append(f"- {fmt_hm(ts)} **SELL FILL** {offer.item.name}: {qty} @ {offer.price:,} "
                                  f"= {fmt_gp(gross)} - {fmt_gp(tax)} tax → cash {fmt_gp(self.cash)}")
                    if offer.remaining == 0:
                        finished_sells.add(offer.item.id)
        for item_id in finished_sells:
            del self.sell_offers[item_id]

        self.log("\n".join(events) if events else "_No fills._")
        self.log()
        if traces:
            self.log("<details><summary>Every 5m bucket where an offered item traded on our side</summary>")
            self.log()
            for label, lines in traces.items():
                self.log(f"**{label}**")
                self.log()
                self.log("| Bucket | Market | Match | Qty |")
                self.log("|---|---|---|---|")
                self.log("\n".join(lines))
                self.log()
            self.log("</details>")
            self.log()

    # -- end of window -------------------------------------------------------

    def finish(self) -> None:
        t = self.t_end
        self.log(f"## End of window {fmt_ts(t)}")
        self.log()
        for offer in self.buy_offers:
            refund = offer.price * (offer.qty - offer.filled)
            self.cash += refund
            self.log(f"- Cancel buy {offer.item.name}: refund {fmt_gp(refund)}")
        self.buy_offers = []

        to_dump: dict[int, int] = dict(self.held)
        for item_id, offer in self.sell_offers.items():
            to_dump[item_id] = to_dump.get(item_id, 0) + offer.remaining
        self.held, self.sell_offers = {}, {}
        for item_id, qty in to_dump.items():
            if qty <= 0:
                continue
            item = self.by_id[item_id]
            quote = item.last_price(t, "instasell")
            price = quote[0] if quote else 0
            gross = price * qty
            tax = sale_tax(price, qty)
            self.cash += gross - tax
            led = self.ledger(item)
            led.liquidated += qty
            led.liquidation_net += gross - tax
            led.tax += tax
            src = f"bucket {fmt_bucket(quote[1])}" if quote else "no price!"
            self.log(f"- **Dump** {item.name}: {qty} × {price:,} (instasell, {src}) = {fmt_gp(gross)} "
                     f"- {fmt_gp(tax)} tax")
        if not to_dump:
            self.log("- Nothing left to dump.")
        self.log()
        self.equity_log.append((t, self.cash, self.fills["buy"], self.fills["sell"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_strategy(strategy: Strategy, items: list[ItemSeries], checks: list[int], start: int, end: int,
                 budget: int, run_dir: Path, data_start: int, fill_model: str) -> tuple[Simulator, dict]:
    sim = Simulator(items, checks, end, budget, strategy, data_start=data_start, fill_model=fill_model)
    sim.log(f"# GE flipping backtest journal: {strategy.name}")
    sim.log()
    sim.log(f"- Strategy: **{strategy.name}**: {strategy.description}")
    sim.log(f"- Window: {fmt_ts(start)} → {fmt_ts(end)} ({(end - start) / DAY:.1f} days, {len(checks)} checks)")
    sim.log(f"- Check times ({LOCAL_TZ.key}): {', '.join(CHECK_TIMES)}")
    sim.log(f"- Budget {fmt_gp(budget)}, {GE_SLOTS} slots, {GE_TAX:.0%} tax on sales, "
            f"buy limit window {BUY_LIMIT_WINDOW_S // 3600}h")
    sim.log(f"- Universe (re-picked each check): buy {fmt_gp(MIN_BUY_PRICE)}–{fmt_gp(MAX_BUY_PRICE)}, "
            f"4h volume GP > {fmt_gp(MIN_4H_VOLUME_GP)}, positive after-tax margin, quotes ≤ 24h old")
    sim.log("- Fill rule: buy fills in a 5m bucket if that bucket's avg instasell ≤ our price; sell fills if avg "
            "instabuy ≥ our price. " + ("Up to the bucket's whole volume (fill model `average`)." if fill_model ==
            "average" else "1 unit per bucket, the most the average proves (fill model `floor`)."))
    sim.log("- All times are local; bucket times are bucket start (MM-DD HH:MM).")
    sim.log()

    sim.run()

    profit = sim.cash - budget
    sim.log("## Daily equity")
    sim.log()
    sim.log("| Day | Equity at last check | Change | Buy fills | Sell fills |")
    sim.log("|---|---|---|---|---|")
    daily: dict[date, tuple[int, int, int]] = {}
    for t, equity, buys, sells in sim.equity_log:
        daily[local(t).date()] = (equity, buys, sells)
    prev = (budget, 0, 0)
    worst_day = 0
    for day, (equity, buys, sells) in daily.items():
        sim.log(f"| {day} | {fmt_gp(equity)} | {fmt_gp(equity - prev[0])} | {buys - prev[1]} | {sells - prev[2]} |")
        worst_day = min(worst_day, equity - prev[0])
        prev = (equity, buys, sells)
    sim.log()

    sim.log("## Summary by item")
    sim.log()
    sim.log("| Item | Bought | Spent | Sold | Sell net | Dumped | Dump net | Tax | Profit |")
    sim.log("|---|---|---|---|---|---|---|---|---|")
    for item_id, led in sorted(sim.ledgers.items(), key=lambda kv: kv[1].profit, reverse=True):
        sim.log(f"| {sim.by_id[item_id].name} | {led.bought} | {fmt_gp(led.spent)} | {led.sold} "
                f"| {fmt_gp(led.sell_net)} | {led.liquidated} | {fmt_gp(led.liquidation_net)} | {fmt_gp(led.tax)} "
                f"| {fmt_gp(led.profit)} |")
    sim.log()
    sim.log(f"- Starting cash: {budget:,} gp")
    sim.log(f"- Final cash: {sim.cash:,} gp")
    sim.log(f"- **Profit: {profit:,} gp ({fmt_gp(profit)}, {profit / budget:.2%})**")
    ledger_total = sum(led.profit for led in sim.ledgers.values())
    sim.log(f"- Cross-check (sum of item profits): {ledger_total:,} gp "
            f"{'✔ matches' if ledger_total == profit else '✘ MISMATCH'}")
    (run_dir / f"journal-{strategy.name}.md").write_text("\n".join(sim.journal), encoding="utf-8")

    midpoint = start + (end - start) // 2
    equity_mid = budget
    peak, max_drawdown = budget, 0
    for t, equity, _, _ in sim.equity_log:
        if t <= midpoint:
            equity_mid = equity
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    metrics = {
        "profit": profit,
        "first_half": equity_mid - budget,
        "second_half": sim.cash - equity_mid,
        "max_drawdown": max_drawdown,
        "worst_day": worst_day,
        "buy_fills": sim.fills["buy"],
        "sell_fills": sim.fills["sell"],
        "items": len(sim.ledgers),
        "units": sim.fills["buy_units"] + sim.fills["sell_units"],
        "units_from_averages": sim.fills["buy_units_from_averages"] + sim.fills["sell_units_from_averages"],
    }
    return sim, metrics


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles/redirects default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--offline", action="store_true", help="use the cached 5m archive, don't download")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recommended")
    parser.add_argument("--compare", action="store_true", help="run every strategy variant and compare them")
    parser.add_argument("--fill-model", choices=("average", "floor"), default="average",
                        help="units a bucket fills once its average reaches our price: all of them, or 1")
    args = parser.parse_args()

    mapping_path = DATA_DIR / "mapping.json"
    if args.offline:
        end = cached_end()
    else:
        api = Api()
        end = (int(time.time()) - PUBLISH_LAG_S - FIVE_MIN) // FIVE_MIN * FIVE_MIN + FIVE_MIN
        earliest = find_earliest_bucket(api, end)
        print(f"API serves 5m buckets from {fmt_ts(earliest)}")
        sync_archive(api, earliest, end)
        DATA_DIR.mkdir(exist_ok=True)
        mapping_path.write_text(json.dumps(api.get("mapping")))
    mapping = json.loads(mapping_path.read_text())

    coverage = contiguous_start(0, end)
    print(f"Loading archive (contiguous from {fmt_ts(coverage)})...")
    items = load_market(coverage, end, mapping)
    if not items:
        raise SystemExit("No items in the archive above the price floor.")
    data_start = min(int(item.ts[0]) for item in items)
    start = data_start + DAY  # 24h warm-up for trailing volume
    checks = check_times(start, end)
    if not checks:
        raise SystemExit("Window too short for any check times.")
    print(f"Simulating {fmt_ts(start)} -> {fmt_ts(end)}: {(end - start) / DAY:.1f} days, {len(checks)} checks, "
          f"{len(items)} items")

    run_dir = DATA_DIR / "runs" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    names = list(STRATEGIES) if args.compare else [args.strategy]
    results = []
    for name in names:
        t_run = time.monotonic()
        sim, metrics = run_strategy(STRATEGIES[name], items, checks, start, end, args.budget, run_dir,
                                    data_start, args.fill_model)
        results.append((STRATEGIES[name], metrics))
        print(f"  {name:22} profit {fmt_gp(metrics['profit']):>10} ({metrics['profit'] / args.budget:7.2%})"
              f"  [{time.monotonic() - t_run:.0f}s]", flush=True)

    if not args.compare:
        print()
        print(f"{'Item':40} {'Bought':>6} {'Sold':>5} {'Dumped':>6} {'Profit':>12}")
        for item_id, led in sorted(sim.ledgers.items(), key=lambda kv: kv[1].profit, reverse=True):
            print(f"{sim.by_id[item_id].name[:40]:40} {led.bought:>6} {led.sold:>5} {led.liquidated:>6} "
                  f"{fmt_gp(led.profit):>12}")
        print(f"Journal: {run_dir / f'journal-{args.strategy}.md'}")
        return

    base = results[0][1]["profit"]
    lines = [
        f"# Strategy comparison: {fmt_ts(start)} → {fmt_ts(end)} ({(end - start) / DAY:.1f} days, "
        f"{len(checks)} checks, budget {fmt_gp(args.budget)}, fill model `{args.fill_model}`)",
        "",
        "*Units from averages*: share of filled units (buys + sells) that only the bucket average justifies, "
        "beyond the 1 unit per bucket it proves.",
        "",
        "| Strategy | Profit | ROI | vs baseline | 1st half | 2nd half | Max drawdown | Worst day "
        "| Buy fills | Sell fills | Units from averages | Items | Description |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy, m in results:
        lines.append(f"| {strategy.name} | {fmt_gp(m['profit'])} | {m['profit'] / args.budget:.2%} "
                     f"| {fmt_gp(m['profit'] - base)} | {fmt_gp(m['first_half'])} | {fmt_gp(m['second_half'])} "
                     f"| {fmt_gp(m['max_drawdown'])} | {fmt_gp(m['worst_day'])} | {m['buy_fills']} "
                     f"| {m['sell_fills']} | {m['units_from_averages'] / max(m['units'], 1):.0%} "
                     f"| {m['items']} | {strategy.description} |")
    (run_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\nJournals and compare.md: {run_dir}")


if __name__ == "__main__":
    main()
