"""The referee: legal books, conserved money, exact parity with the backtester."""

import numpy as np
import pytest
import torch

from backtest_flips import BUDGET, STRATEGIES, Simulator
from lab import env as E
from lab.policies.base import empty
from lab.policies.neural import Neural
from lab.policies.rules import RulesPolicy
from lab.policies.soft_rules import SoftRules
from lab.tests.conftest import DEVICE


def test_legalize_invariants():
    g = torch.Generator(device="cpu").manual_seed(0)
    B, N, M, S = 64, 50, 30, 8
    for _ in range(20):
        inv = torch.randint(0, 5, (B, N), generator=g)
        limit = torch.randint(0, 10, (B, N), generator=g)
        cash = torch.randint(0, 10 ** 9, (B,), generator=g)
        item = torch.stack([torch.randperm(N, generator=g)[:M] for _ in range(B)])
        item[torch.rand(B, M, generator=g) < 0.2] = -1
        side = torch.randint(0, 2, (B, M), generator=g)
        price = torch.randint(-10, 10 ** 8, (B, M), generator=g)
        qty = torch.randint(-2, 20, (B, M), generator=g)
        prio = torch.randn(B, M, generator=g)
        book, reserve = E.legalize(E.Proposals(item, side, price, qty, prio), cash, inv, limit, S)
        live = book.item >= 0
        assert book.item.shape == (B, S)
        assert (live.sum(1) <= S).all()
        assert (book.qty[live] > 0).all() and (book.price[live] > 0).all()
        it = book.item.clamp(min=0)
        sells, buys = live & (book.side == E.SELL), live & (book.side == E.BUY)
        assert (book.qty[sells] <= inv.gather(1, it)[sells]).all()
        assert (book.qty[buys] <= limit.gather(1, it)[buys]).all()
        cost = torch.where(buys, book.price * book.qty, 0).sum(1)
        assert torch.equal(cost, reserve) and (reserve <= cash).all()
        # kept offers are the highest-priority valid ones
        assert (book.priority[:, :-1] >= book.priority[:, 1:]).all()


class Idle:
    name = "idle"

    def act(self, obs):
        return empty(obs.cash.shape[0], 1, obs.cash.device)


def test_idle_policy_keeps_its_cash(mkt, short_window):
    episodes = E.Episodes.from_windows(mkt.features, [short_window])
    cash = torch.tensor([1_800_000_000], device=DEVICE)
    res = E.run(mkt, Idle(), episodes, cash, torch.zeros(1, dtype=torch.int64, device=DEVICE), E.EnvConfig(), 300)
    assert int(res.final_cash[0]) == 1_800_000_000
    assert (res.equity[0, :int(episodes.n_checks[0]) + 1] == 1_800_000_000).all()


@pytest.mark.parametrize("make", [lambda: (RulesPolicy(STRATEGIES["recommended"]), None),
                                  lambda: (SoftRules(), SoftRules.init_params()),
                                  lambda: (Neural(), Neural.init_params())])
def test_profit_equals_item_ledgers(mkt, short_window, make):
    """Final cash - start = sum over items of (sales + dump - spent), with per-item tax and a haircut."""
    policy, params = make()
    episodes = E.Episodes.from_windows(mkt.features, [short_window])
    stacks = torch.tensor([1_500_000_000, 6_000_000_000], device=DEVICE)
    ep = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    if params is not None:
        policy = policy.bind(params.to(DEVICE)[None], 2)
    res = E.run(mkt, policy, episodes, stacks, ep, E.EnvConfig(), 300, track=True)
    led = res.ledger
    profit = (led["sell_net"] + led["dump_net"] - led["spent"]).sum(1)
    assert torch.equal(res.final_cash - stacks, profit)
    assert (led["bought"].sum(1) == led["sold"].sum(1) + led["dumped"].sum(1)).all()


def test_parity_with_backtester(parity_mkt, short_window):
    """Short-window version of lab.parity (the full run: uv run python -m lab.parity)."""
    market = parity_mkt.market
    start, end = short_window
    checks = [int(t) for t in parity_mkt.features.checks if start <= t < end]
    for name in ("baseline", "recommended"):
        for fill_model in ("average", "floor"):
            sim = Simulator(market.items, checks, end, BUDGET, STRATEGIES[name], data_start=market.data_start,
                            quiet=True, fill_model=fill_model)
            sim.run()
            res = E.run(parity_mkt, RulesPolicy(STRATEGIES[name]),
                        E.Episodes.from_windows(parity_mkt.features, [(start, end)]),
                        torch.tensor([BUDGET], device=DEVICE), torch.zeros(1, dtype=torch.int64, device=DEVICE),
                        E.EnvConfig(fill_model=fill_model, haircut=0.0, tax="backtester"), obs_lag=0)
            assert int(res.final_cash[0]) == sim.cash, (name, fill_model)
            assert res.equity[0, :len(checks)].tolist() == [e for _, e, _, _ in sim.equity_log[:len(checks)]]
            assert (int(res.buy_fills[0]), int(res.sell_fills[0])) == (sim.fills["buy"], sim.fills["sell"])
