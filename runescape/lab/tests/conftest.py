import pytest
import torch

from backtest_flips import DAY, LOAD_MIN_PRICE
from lab import env as E
from lab import market as M
from lab.features import Features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def market():
    return M.load()


@pytest.fixture(scope="session")
def features(market):
    return Features(market, obs_lag=300)


@pytest.fixture(scope="session")
def mkt(market, features):
    return E.MarketOnDevice(market, features, DEVICE)


@pytest.fixture(scope="session")
def parity_mkt():
    """The backtester's view (its 25m bucket floor), lag 0."""
    market = M.load(min_bucket_price=LOAD_MIN_PRICE)
    return E.MarketOnDevice(market, Features(market, obs_lag=0), DEVICE)


@pytest.fixture(scope="session")
def short_window(market):
    """The last 3 days, for quick simulations."""
    return market.end - 3 * DAY, market.end
