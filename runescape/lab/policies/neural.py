"""A small neural policy: one MLP, shared by every item, scoring each item from what it looks like right now.

Per account and item it sees the item's market inputs (features.MARKET_INPUTS: no identity), the account's position
in it, the account's cash share / size / slots held and the time of day and week, and outputs:

    0 buy priority   (buy if > 0; higher first)
    1 buy price      instasell x (1 + 3% tanh(.))
    2 sell price     instabuy  x (1 + 3% tanh(.))
    3 order size     sigmoid(.) x 50% of equity
    4 list for sale  (list everything held if > 0, otherwise keep holding)

It reads the check's universe (for buys) plus the account's held items (for sells). The parameters of P variants
are stacked as [P, dim] and evaluated in one batched matmul per layer; account b uses variant b // E.
"""

from __future__ import annotations

import math

import torch

from lab.features import MARKET_INPUTS
from lab.policies.base import BUY, SELL, Observation, Proposals

HIDDEN = 32
ACCOUNT_INPUTS = ["inv_share", "limit_left_share", "has_inv", "price_vs_equity"]
GLOBAL_INPUTS = ["cash_share", "log_equity", "items_held", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
F_IN = len(MARKET_INPUTS) + len(ACCOUNT_INPUTS) + len(GLOBAL_INPUTS)
OUTPUTS = 5
PRICE_RANGE = 0.03
MAX_SIZE = 0.5
SHAPES = [("w1", (F_IN, HIDDEN)), ("b1", (HIDDEN,)), ("w2", (HIDDEN, HIDDEN)), ("b2", (HIDDEN,)),
          ("w3", (HIDDEN, OUTPUTS)), ("b3", (OUTPUTS,))]


def unpack(params: torch.Tensor) -> dict[str, torch.Tensor]:
    out, k = {}, 0
    for name, shape in SHAPES:
        n = math.prod(shape)
        out[name] = params[:, k:k + n].reshape(params.shape[0], *shape)
        k += n
    return out


class Neural:
    name = "neural"
    dim = sum(math.prod(shape) for _, shape in SHAPES)

    def __init__(self, params: torch.Tensor | None = None, E: int = 1):
        self.params = params
        self.E = E

    @staticmethod
    def init_params(seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        parts = []
        for name, shape in SHAPES:
            if name.startswith("w"):
                parts.append(torch.randn(shape, generator=g).flatten() / math.sqrt(shape[0]))
            elif name == "b3":  # start out listing everything and buying small
                parts.append(torch.tensor([0.0, 0.0, 0.0, -2.0, 3.0]))
            else:
                parts.append(torch.zeros(math.prod(shape)))
        return torch.cat(parts)

    def bind(self, params: torch.Tensor, E: int) -> Neural:
        return Neural(params, E)

    def describe(self, params: torch.Tensor) -> dict[str, float]:
        return {"params": float(params.numel()), "norm": float(params.norm())}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, R, F_IN] -> [B, R, OUTPUTS], variant b // E for account b."""
        B, R, _ = x.shape
        P = self.params.shape[0]
        w = unpack(self.params)
        h = x.reshape(P, (B // P) * R, F_IN)
        h = torch.tanh(torch.baddbmm(w["b1"][:, None], h, w["w1"]))
        h = torch.tanh(torch.baddbmm(w["b2"][:, None], h, w["w2"]))
        return torch.baddbmm(w["b3"][:, None], h, w["w3"]).reshape(B, R, OUTPUTS)

    def act(self, obs: Observation) -> Proposals:
        mkt, B = obs.mkt, obs.cash.shape[0]
        uni = mkt.universe[obs.c][obs.ep]  # [B, K]
        items = torch.cat([uni, obs.held], dim=1)  # [B, R]: universe rows (buys), then held rows (sells)
        K = uni.shape[1]
        it = items.clamp(min=0)
        c_b = obs.c[obs.ep][:, None]
        market_x = mkt.x[c_b, it]  # [B, R, F]
        sq = mkt.raw["is_px"][c_b, it]
        bq = mkt.raw["ib_px"][c_b, it]
        equity = obs.equity.clamp(min=1).double()[:, None]
        inv = obs.inv.gather(1, it)
        limit = mkt.limits[it]
        account_x = torch.stack([
            (inv * sq).double() / equity,
            obs.limit_left.gather(1, it).double() / limit.clamp(min=1),
            (inv > 0).double(),
            torch.log10(sq.clamp(min=1).double() / equity).clamp(-6, 2) / 3,
        ], dim=-1).float()
        n_held = (obs.held >= 0).sum(1, keepdim=True)
        glob = torch.cat([
            (obs.cash.double()[:, None] / equity).float(),
            (torch.log10(equity).float() - 9.5),
            n_held.float() / 8,
            mkt.time_x[obs.c][obs.ep],
        ], dim=1)
        x = torch.cat([market_x, account_x, glob[:, None, :].expand(-1, items.shape[1], -1)], dim=-1)
        x = torch.where((items >= 0)[..., None], x, 0.0)
        out = self.forward(x)

        o_buy, o_held = out[:, :K], out[:, K:]
        buy_px = (sq[:, :K].double() * (1 + PRICE_RANGE * torch.tanh(o_buy[..., 1].double()))).round().long()
        size = torch.sigmoid(o_buy[..., 3].double()) * MAX_SIZE * equity
        buy_qty = (size / buy_px.clamp(min=1)).floor().long()
        wants = (uni >= 0) & (o_buy[..., 0] > 0) & (sq[:, :K] > 0)
        buys = Proposals(torch.where(wants, uni, -1), torch.full_like(uni, BUY), buy_px, buy_qty,
                         o_buy[..., 0])

        held = obs.held
        sell_px = (bq[:, K:].double() * (1 + PRICE_RANGE * torch.tanh(o_held[..., 2].double()))).round().long()
        sell_px = torch.where(bq[:, K:] > 0, sell_px, obs.listed_price.gather(1, held.clamp(min=0)))
        listing = (held >= 0) & (o_held[..., 4] > 0)
        sells = Proposals(torch.where(listing, held, -1), torch.full_like(held, SELL), sell_px,
                          obs.inv.gather(1, held.clamp(min=0)), 1e6 + o_held[..., 4])
        return Proposals.cat(sells, buys)
