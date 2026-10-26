"""Features match the backtester's ItemSeries methods, and can't see the future."""

import numpy as np

from backtest_flips import DAY
from lab import market as M
from lab.features import Features


def test_features_match_item_series(market, features):
    rng = np.random.default_rng(1)
    lag = features.obs_lag
    for _ in range(2000):
        c, n = rng.integers(len(features.checks)), rng.integers(market.N)
        item, t = market.items[n], int(features.checks[c]) - lag
        for side, p in (("instasell", "is"), ("instabuy", "ib")):
            quote = item.last_price(t, side)
            assert (features.raw[f"{p}_px"][c, n], features.raw[f"{p}_ts"][c, n]) == (quote if quote else (0, -1))
            assert features.raw[f"{p}_r3"][c, n] == (item.recent_price(t, side, 3) or 0)
            assert features.raw[f"{p}_tb"][c, n] == item.trade_buckets(t, side, DAY)
            median = item.median_price(t, side, 3 * DAY)
            got = features.raw[f"{p}_med"][c, n]
            assert (median is None and np.isnan(got)) or median == got
        assert features.raw["vol24"][c, n] == item.volume(t - DAY, t)


def test_no_lookahead(market, features):
    """Cutting off everything after a point changes no feature at any check before it."""
    cut = int(features.checks[len(features.checks) // 2]) + 1800
    early = M.load(end=cut // 300 * 300, start=market.t0)
    f_early = Features(early, obs_lag=features.obs_lag, cache=False)
    shared = len(f_early.checks)
    assert np.array_equal(f_early.checks, features.checks[:shared])
    cols = [market.item_ids.tolist().index(i) for i in early.item_ids]  # items seen before the cut
    for name, arr in f_early.raw.items():
        np.testing.assert_array_equal(arr, features.raw[name][:shared][:, cols], err_msg=name)
    np.testing.assert_array_equal(f_early.inputs(), features.inputs()[:shared][:, cols])
