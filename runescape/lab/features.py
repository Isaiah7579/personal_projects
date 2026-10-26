"""Point-in-time features at every check time: what a policy is allowed to see.

A check at time t sees only buckets that completed by t - obs_lag. The raw fields use the backtester's exact
definitions (ItemSeries.last_price, recent_price, trade_buckets, median_price, volume), vectorized over checks
per item; tests/test_features.py checks them against the ItemSeries methods.
"""

from __future__ import annotations

import math

import numpy as np

from backtest_flips import DAY, FIVE_MIN, GE_TAX, LOCAL_TZ, check_times
from lab.market import LAB_DIR, Market

VERSION = 1
RECENT_UNITS = 3
DRIFT_WINDOW = 3 * DAY
RETURN_HORIZONS = {"ret_1h": 3600, "ret_4h": 4 * 3600, "ret_24h": DAY}
MISSING_TS = -1

INT_FIELDS = ("is_px", "is_ts", "ib_px", "ib_ts", "is_r3", "ib_r3", "is_tb", "ib_tb", "vol24")
FLOAT_FIELDS = ("is_med", "ib_med", "prior_avg", "volat", *RETURN_HORIZONS)


def _last(stamps: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the latest quote whose bucket completed by t (-1 if none), as ItemSeries.last_price."""
    return np.searchsorted(stamps, t - FIVE_MIN, side="right") - 1


def _side(item, side: str, t: np.ndarray) -> dict[str, np.ndarray]:
    stamps, prices, volumes = item.quotes[side]
    C = len(t)
    if not len(stamps):  # never traded on this side
        nan = np.full(C, np.nan)
        return {"px": np.zeros(C, dtype=np.int64), "ts": np.full(C, MISSING_TS, dtype=np.int64),
                "r3": np.zeros(C, dtype=np.int64), "tb": np.zeros(C, dtype=np.int64), "med": nan, "volat": nan,
                **{name: nan for name in RETURN_HORIZONS}}
    k = _last(stamps, t)
    has = k >= 0
    px = np.zeros(C, dtype=np.int64)
    ts = np.full(C, MISSING_TS, dtype=np.int64)
    px[has] = np.rint(prices[k[has]]).astype(np.int64)
    ts[has] = stamps[k[has]]

    # recent_price: walk back bucket by bucket until RECENT_UNITS are covered, in the same float order
    value = np.zeros(C)
    total = np.zeros(C, dtype=np.int64)
    j = k.copy()
    for _ in range(RECENT_UNITS):
        live = (j >= 0) & (total < RECENT_UNITS)
        take = np.where(live, np.minimum(volumes[np.maximum(j, 0)], RECENT_UNITS - total), 0)
        value = np.where(live, value + prices[np.maximum(j, 0)] * take, value)
        total += take
        j -= 1
    r3 = np.zeros(C, dtype=np.int64)
    r3[total > 0] = np.rint(value[total > 0] / total[total > 0]).astype(np.int64)

    tb = np.searchsorted(stamps, t - FIVE_MIN, side="right") - np.searchsorted(stamps, t - DAY)

    med = np.full(C, np.nan)
    a = np.searchsorted(stamps, t - DRIFT_WINDOW, side="left")
    b = np.searchsorted(stamps, t - FIVE_MIN, side="right")
    for c in np.nonzero(b > a)[0]:
        med[c] = np.median(prices[a[c]:b[c]])

    # 24h volatility: std of log quote prices in the window
    logp = np.log(prices) if len(prices) else prices
    s1 = np.concatenate([[0.0], np.cumsum(logp)])
    s2 = np.concatenate([[0.0], np.cumsum(logp * logp)])
    a24 = np.searchsorted(stamps, t - DAY, side="left")
    n = b - a24
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (s1[b] - s1[a24]) / n
        var = np.maximum((s2[b] - s2[a24]) / n - mean * mean, 0)
    volat = np.where(n >= 2, np.sqrt(var), np.nan)

    rets = {}
    for name, h in RETURN_HORIZONS.items():
        kp = _last(stamps, t - h)
        ok = has & (kp >= 0)
        r = np.full(C, np.nan)
        r[ok] = np.log(prices[k[ok]] / prices[kp[ok]])
        rets[name] = r
    return {"px": px, "ts": ts, "r3": r3, "tb": tb, "med": med, "volat": volat, **rets}


def _mean_of_sides(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average of the two sides' values, or whichever one exists."""
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))


def _volume(item, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """ItemSeries.volume, vectorized: units in buckets that started at/after lo and completed by hi."""
    a = np.searchsorted(item.ts, lo, side="left")
    b = np.searchsorted(item.ts, hi - FIVE_MIN, side="right")
    return item.cum_volume[b] - item.cum_volume[a]


def compute(market: Market, checks: np.ndarray, obs_lag: int) -> dict[str, np.ndarray]:
    """Raw [C, N] features at each check (seen at check - obs_lag)."""
    t = checks - obs_lag
    C, N = len(checks), market.N
    out = {f: np.zeros((C, N), dtype=np.int64) for f in INT_FIELDS}
    out.update({f: np.full((C, N), np.nan) for f in FLOAT_FIELDS})
    prior_days = np.minimum(7.0, (t - DAY - market.data_start) / DAY)
    span = (prior_days * DAY).astype(np.int64)
    for n, item in enumerate(market.items):
        sides = {"is": _side(item, "instasell", t), "ib": _side(item, "instabuy", t)}
        for p, s in sides.items():
            for key in ("px", "ts", "r3", "tb", "med"):
                out[f"{p}_{key}"][:, n] = s[key]
        for name in ("volat", *RETURN_HORIZONS):
            out[name][:, n] = _mean_of_sides(sides["is"][name], sides["ib"][name])
        out["vol24"][:, n] = _volume(item, t - DAY, t)
        prior = _volume(item, t - DAY - span, t - DAY) / np.where(prior_days > 0, prior_days, 1)
        out["prior_avg"][:, n] = np.where(prior_days >= 1, prior, np.nan)
    return out


class Features:
    """Raw features at every check time of the market's window, cached on disk."""

    def __init__(self, market: Market, obs_lag: int, cache: bool = True):
        self.market = market
        self.obs_lag = obs_lag
        self.checks = np.array(check_times(market.t0 + DAY, market.end), dtype=np.int64)
        min_price = int(min(np.nanmin(market.high), np.nanmin(market.low)))
        path = LAB_DIR / "cache" / (f"features-v{VERSION}-{market.t0}-{market.end}-n{market.N}-p{min_price}"
                                    f"-lag{obs_lag}.npz")
        if cache and path.exists():
            with np.load(path) as z:
                self.raw = {k: z[k] for k in z.files}
        else:
            self.raw = compute(market, self.checks, obs_lag)
        if cache and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            for old in path.parent.glob(f"features-v{VERSION}-*-p{min_price}-lag{obs_lag}.npz"):
                old.unlink(missing_ok=True)  # an older cache end; keep one per price floor and lag
            np.savez(path, **self.raw)
        self.check_index = {int(t): c for c, t in enumerate(self.checks)}

    def universe(self, k: int) -> np.ndarray:
        """[C, k] item indices learned policies may buy at each check, -1 padded: the k items with the most 24h
        GP volume among those with both quotes <= 24h old. Uses only what the check can see."""
        r = self.raw
        t = self.checks[:, None]
        fresh = (r["is_px"] > 0) & (r["ib_px"] > 0) & (np.minimum(r["is_ts"], r["ib_ts"]) >= t - DAY)
        gp = np.where(fresh, r["vol24"] * r["is_px"].astype(float), -1.0)
        order = np.argsort(-gp, axis=1, kind="stable")[:, :k]
        return np.where(np.take_along_axis(gp, order, axis=1) > 0, order, -1)

    def inputs(self) -> np.ndarray:
        """[C, N, F] normalized market inputs for learned policies (no item identity), NaN-free."""
        r = self.raw
        is_px, ib_px = r["is_px"].astype(float), r["ib_px"].astype(float)
        has = (is_px > 0) & (ib_px > 0)
        safe_is = np.where(has, is_px, 1.0)
        t = self.checks[:, None].astype(float)
        age_is = np.where(has, (t - r["is_ts"]) / 3600, 24.0)
        age_ib = np.where(has, (t - r["ib_ts"]) / 3600, 24.0)
        tax = np.floor(ib_px * GE_TAX)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols = [
                np.clip(np.log10(safe_is) - 8, -3, 3),
                np.clip((ib_px - is_px - tax) / safe_is, -0.2, 0.2) * 10,
                np.log1p(np.clip(age_is, 0, 48)),
                np.log1p(np.clip(age_ib, 0, 48)),
                np.clip(is_px / r["is_med"] - 1, -0.5, 0.5) * 5,
                np.clip(ib_px / r["ib_med"] - 1, -0.5, 0.5) * 5,
                np.log1p(r["is_tb"]) / math.log(289),
                np.log1p(r["ib_tb"]) / math.log(289),
                np.log1p(r["vol24"]) / 5,
                np.clip(np.log((r["vol24"] + 1) / (r["prior_avg"] + 1)), -3, 3),
                np.clip(r["ret_1h"] * 20, -3, 3),
                np.clip(r["ret_4h"] * 10, -3, 3),
                np.clip(r["ret_24h"] * 5, -3, 3),
                np.clip(r["volat"] * 10, 0, 3),
                np.clip((r["ib_r3"] - r["is_r3"]) / safe_is, -0.2, 0.2) * 10,
                np.broadcast_to(np.log1p(self.market.limits) / 3, is_px.shape),
                np.clip(np.log10(r["vol24"] * safe_is + 1) - 9, -3, 3),
            ]
        x = np.stack(cols, axis=-1)
        x[~has] = 0
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def time_inputs(self) -> np.ndarray:
        """[C, 4] time of day and day of week (local), as sin/cos."""
        from datetime import datetime
        out = []
        for t in self.checks:
            d = datetime.fromtimestamp(int(t), LOCAL_TZ)
            h = (d.hour + d.minute / 60) / 24 * 2 * math.pi
            w = d.weekday() / 7 * 2 * math.pi
            out.append([math.sin(h), math.cos(h), math.sin(w), math.cos(w)])
        return np.array(out, dtype=np.float32)


MARKET_INPUTS = ["log_price", "net_spread", "age_is", "age_ib", "drift_is", "drift_ib", "tb_is", "tb_ib", "vol24",
                 "vol_spike", "ret_1h", "ret_4h", "ret_24h", "volat", "recent_spread", "limit", "vol24_gp"]
