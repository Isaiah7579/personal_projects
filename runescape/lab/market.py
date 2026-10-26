"""The cached 5m archive as dense [bucket, item] arrays.

Bucket i starts at t0 + 300 * i and covers the archive's contiguous window. Items are in the same order
backtest_flips.load_market produces (first appearance), so ties in the backtester's stable sorts break the same
way here. The backtester's own ItemSeries objects are built alongside, for features and parity checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from backtest_flips import (ARCHIVE_MIN_PRICE, DATA_DIR, FIVE_MIN, UNFLIPPABLE_ITEMS, ItemSeries, cached_end,
                            contiguous_start, day_start, read_day, utc_day)

LAB_DIR = DATA_DIR / "lab"


@dataclass
class Market:
    t0: int  # start of bucket 0
    end: int  # end of the last bucket
    data_start: int  # first bucket any item traded in (the backtester's lookback floor)
    item_ids: np.ndarray  # [N]
    names: list[str]
    limits: np.ndarray  # [N] 4h buy limits, with the backtester's overrides and default
    # [T, N] bucket averages and volumes; price NaN where that side didn't trade
    high: np.ndarray  # instabuy average
    high_vol: np.ndarray
    low: np.ndarray  # instasell average
    low_vol: np.ndarray
    items: list[ItemSeries] = field(repr=False)

    @property
    def T(self) -> int:
        return self.high.shape[0]

    @property
    def N(self) -> int:
        return self.high.shape[1]

    def bucket(self, ts: int) -> int:
        return (ts - self.t0) // FIVE_MIN

    def last_quotes(self) -> tuple[np.ndarray, np.ndarray]:
        """[T, N] int64 last instasell / instabuy price (rounded like ItemSeries.last_price) of buckets <= i, or 0."""
        out = []
        for prices in (self.low, self.high):
            idx = np.where(np.isnan(prices), 0, np.arange(self.T)[:, None])
            np.maximum.accumulate(idx, axis=0, out=idx)
            filled = prices[idx, np.arange(self.N)]
            filled[np.isnan(filled)] = 0
            out.append(np.rint(filled).astype(np.int64))
        return out[0], out[1]


def load(min_bucket_price: int = ARCHIVE_MIN_PRICE, end: int | None = None, start: int | None = None) -> Market:
    """Load the contiguous archive ending at `end` (default: the cache's end), from `start` (default: as far back
    as it's contiguous).

    min_bucket_price drops an item's bucket unless either side traded at or above it, like load_market's
    LOAD_MIN_PRICE (pass that for parity with the backtester; the archive itself already drops < 1m). Like
    load_market, it skips UNFLIPPABLE_ITEMS.
    """
    end = end or cached_end()
    start = start or contiguous_start(0, end)
    mapping = json.loads((DATA_DIR / "mapping.json").read_text())
    meta_by_id = {m["id"]: m for m in mapping}
    rows: dict[int, list[tuple]] = {}
    day = utc_day(start)
    while day_start(day) < end:
        for ts_str, bucket in read_day(day).items():
            ts = int(ts_str)
            if not start <= ts < end:
                continue
            for id_str, (high, high_vol, low, low_vol) in bucket.items():
                if max(high or 0, low or 0) < min_bucket_price or int(id_str) in UNFLIPPABLE_ITEMS:
                    continue
                rows.setdefault(int(id_str), []).append((ts, high, high_vol, low, low_vol))
        day += timedelta(days=1)
    items = [ItemSeries(item_id, meta_by_id.get(item_id, {}), r) for item_id, r in rows.items()]

    T, N = (end - start) // FIVE_MIN, len(items)
    high = np.full((T, N), np.nan)
    low = np.full((T, N), np.nan)
    high_vol = np.zeros((T, N), dtype=np.int64)
    low_vol = np.zeros((T, N), dtype=np.int64)
    for n, item in enumerate(items):
        for ts, h, hv, lo, lv in item.buckets.values():
            i = (ts - start) // FIVE_MIN
            if h is not None:
                high[i, n] = h
            if lo is not None:
                low[i, n] = lo
            high_vol[i, n], low_vol[i, n] = hv, lv
    return Market(t0=start, end=end, data_start=min(int(item.ts[0]) for item in items),
                  item_ids=np.array([item.id for item in items]), names=[item.name for item in items],
                  limits=np.array([item.limit for item in items], dtype=np.int64),
                  high=high, high_vol=high_vol, low=low, low_vol=low_vol, items=items)
