"""The recommended strategy's filters as continuous parameters, vectorized on the GPU, for evolution to tune.

Each parameter is unconstrained in the search space and mapped onto a sensible range (PARAMS). Allocation is
greedy by score = max units x after-tax margin (the referee's legalize trims to slots and cash in that order)
instead of the backtester's knapsack, and offers are priced a fraction of the spread inside the quotes instead of
a fixed outbid. Only items in the check's universe can be bought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from backtest_flips import DAY
from lab.policies.base import BUY, SELL, Observation, Proposals


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool
    init: float  # in the parameter's own units


PARAMS = [
    Param("max_quote_age_h", 0.25, 24, True, 1.0),
    Param("min_trade_buckets", 0, 48, False, 6),
    Param("max_drift", 0.01, 0.5, True, 0.10),
    Param("max_volume_spike", 1.0, 10, True, 2.0),
    Param("buy_outbid_spread", 0.0, 0.45, False, 0.03),  # buy at instasell + x * spread
    Param("sell_outbid_spread", 0.0, 0.45, False, 0.03),  # sell at instabuy - x * spread
    Param("max_item_share", 0.05, 1.0, True, 0.25),
    Param("min_margin_pct", 0.0, 0.05, False, 0.001),
    Param("min_price", 1e6, 5e10, True, 75e6),
    Param("max_price_ratio", 1.0, 1000, True, 500 / 75),  # max price = min price * this
    Param("min_4h_volume_gp", 1e6, 1e11, True, 100e6),
]


def to_values(params: torch.Tensor) -> dict[str, torch.Tensor]:
    """[P, d] unconstrained -> {name: [P]} in each parameter's units (sigmoid onto its range)."""
    out = {}
    for k, p in enumerate(PARAMS):
        u = torch.sigmoid(params[:, k])
        out[p.name] = p.lo * (p.hi / p.lo) ** u if p.log else p.lo + (p.hi - p.lo) * u
    return out


def from_values(values: dict[str, float]) -> torch.Tensor:
    """Inverse of to_values for one parameter set -> [d]."""
    raw = []
    for p in PARAMS:
        v = values[p.name]
        u = math.log(v / p.lo) / math.log(p.hi / p.lo) if p.log else (v - p.lo) / (p.hi - p.lo)
        u = min(max(u, 1e-4), 1 - 1e-4)
        raw.append(math.log(u / (1 - u)))
    return torch.tensor(raw)


def gather_items(table: torch.Tensor, c_e: torch.Tensor, ep: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """table [C, N] -> [B, k] values at each account's check for items [B, k] (item -1 -> table[..., 0])."""
    return table[c_e[ep][:, None], items.clamp(min=0)]


def spread_prices(raw: dict, c_e, ep, items, buy_x, sell_x):
    sq = gather_items(raw["is_px"], c_e, ep, items)
    bq = gather_items(raw["ib_px"], c_e, ep, items)
    spread = (bq - sq).clamp(min=0).double()
    buy = sq + (buy_x[:, None] * spread).floor().long() + 1
    sell = bq - (sell_x[:, None] * spread).floor().long() - 1
    return sq, bq, buy, sell


class SoftRules:
    """A batch of soft-rule variants; account b uses variant b // E."""
    name = "soft-rules"
    dim = len(PARAMS)

    def __init__(self, params: torch.Tensor | None = None, E: int = 1):
        self.params = params
        self.E = E

    @staticmethod
    def init_params() -> torch.Tensor:
        return from_values({p.name: p.init for p in PARAMS})

    def bind(self, params: torch.Tensor, E: int) -> SoftRules:
        return SoftRules(params, E)

    def describe(self, params: torch.Tensor) -> dict[str, float]:
        return {k: float(v[0]) for k, v in to_values(params[None]).items()}

    def act(self, obs: Observation) -> Proposals:
        raw, B = obs.mkt.raw, obs.cash.shape[0]
        variant = torch.arange(B, device=obs.cash.device) // self.E
        v = {k: x[variant] for k, x in to_values(self.params).items()}  # [B] each
        t = obs.t[obs.ep][:, None]

        # sells: everything held, priced inside the spread (keep the old price if there's no quote)
        held = obs.held
        _, bq_h, _, sell_h = spread_prices(raw, obs.c, obs.ep, held, v["buy_outbid_spread"], v["sell_outbid_spread"])
        sell_h = torch.where(bq_h > 0, sell_h, obs.listed_price.gather(1, held.clamp(min=0)))
        sells = Proposals(held, torch.full_like(held, SELL), sell_h,
                          obs.inv.gather(1, held.clamp(min=0)) * (held >= 0),
                          torch.full(held.shape, 1e15, device=held.device))

        # buys: universe items passing the filters, best score first
        items = obs.mkt.universe[obs.c][obs.ep]  # [B, K]
        g = lambda name: gather_items(raw[name], obs.c, obs.ep, items)
        sq, bq, buy, sell = spread_prices(raw, obs.c, obs.ep, items, v["buy_outbid_spread"],
                                          v["sell_outbid_spread"])
        oldest = torch.minimum(g("is_ts"), g("ib_ts"))
        margin = sell - buy - sell * 2 // 100
        vol4h = g("vol24").double() / 6
        ok = (items >= 0) & (oldest >= 0) & (margin > 0)
        ok &= (t - oldest) <= (v["max_quote_age_h"] * 3600)[:, None]
        ok &= torch.minimum(g("is_tb"), g("ib_tb")) >= v["min_trade_buckets"][:, None]
        for quote, med in ((sq, g("is_med")), (bq, g("ib_med"))):
            ok &= torch.isnan(med) | ((quote / med - 1).abs() <= v["max_drift"][:, None])
        prior = g("prior_avg")
        ok &= torch.isnan(prior) | (vol4h * 6 <= v["max_volume_spike"][:, None] * prior)
        ok &= margin >= v["min_margin_pct"][:, None] * buy
        min_price = v["min_price"][:, None]
        ok &= (buy >= min_price) & (buy <= min_price * v["max_price_ratio"][:, None])
        ok &= vol4h * buy >= v["min_4h_volume_gp"][:, None]
        inv = obs.inv.gather(1, items.clamp(min=0))
        share_qty = (v["max_item_share"][:, None] * obs.equity[:, None] / buy.clamp(min=1)).floor().long()
        max_qty = torch.minimum(vol4h.floor().long(), share_qty.clamp(min=1) - inv)
        ok &= max_qty >= 1
        score = (max_qty * margin).double()
        buys = Proposals(torch.where(ok, items, -1), torch.full_like(items, BUY), buy, max_qty.clamp(min=0),
                         torch.where(ok, score, -1.0).float())
        return Proposals.cat(sells, buys)
