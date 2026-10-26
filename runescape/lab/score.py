"""Report cards: score any policy on a split the same way.

Every day-start check in the split that leaves room for a full episode starts one (so episodes overlap by design:
more, steadier samples), and each is run from a fresh stack at each of STACKS. The headline is the mean log growth
per episode (so +0.05 ≈ +5% a week); the worst episode and CVaR25 (mean of the worst quarter) say how bad the bad
weeks are.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lab import env as E
from lab.splits import EPISODE, Split, episode_starts

STACKS = (1_500_000_000, 1_800_000_000, 3_000_000_000, 6_000_000_000)


def cvar(x: torch.Tensor, q: float = 0.25, dim: int = -1) -> torch.Tensor:
    """Mean of the worst ceil(q * n) values along dim."""
    k = max(1, math.ceil(q * x.shape[dim]))
    return x.topk(k, dim=dim, largest=False).values.mean(dim)


def episodes_for(mkt: E.MarketOnDevice, split: Split, length: int = EPISODE) -> tuple[E.Episodes, list[int]]:
    starts = episode_starts(split, mkt.features.checks, length)
    if not starts:
        raise SystemExit(f"No room for a {length / 86400:.0f}-day episode in {split.describe()}")
    return E.Episodes.from_windows(mkt.features, [(s, s + length) for s in starts]), starts


def evaluate(mkt: E.MarketOnDevice, policy, split: Split, cfg: E.EnvConfig, obs_lag: int,
             stacks: tuple[int, ...] = STACKS, params: torch.Tensor | None = None, track: bool = False) -> dict:
    """Score one policy (or one parameter vector of a parametric policy) on every episode of `split`."""
    episodes, starts = episodes_for(mkt, split)
    n_ep, dev = episodes.E, mkt.low.device
    # accounts: stack-major, so account b plays episode b % n_ep with stack b // n_ep
    ep = torch.arange(n_ep * len(stacks), device=dev) % n_ep
    cash = torch.tensor(np.repeat(stacks, n_ep), device=dev)
    if params is not None:
        # one variant: every account uses row 0, so E = all accounts
        policy = policy.bind(params.to(dev)[None], len(ep))
    res = E.run(mkt, policy, episodes, cash, ep, cfg, obs_lag=obs_lag, track=track)
    g = res.log_growth.view(len(stacks), n_ep)  # [stack, episode]
    eq = res.equity.double() / res.start_cash.double()[:, None]
    n = torch.tensor(episodes.n_checks, device=dev)[ep]
    drawdowns = []
    for b in range(len(ep)):
        curve = eq[b, :int(n[b]) + 1]
        drawdowns.append(float((1 - curve / curve.cummax(0).values).max()))
    card = {
        "score": float(g.mean()),
        "worst": float(g.min()),
        "cvar25": float(cvar(g.flatten())),
        "by_stack": {f"{s / 1e9:g}b": float(g[k].mean()) for k, s in enumerate(stacks)},
        "weekly_return": float(torch.expm1(g).mean()),
        "max_drawdown": max(drawdowns),
        "buy_fills": float(res.buy_fills.double().mean()),
        "sell_fills": float(res.sell_fills.double().mean()),
        "slot_use": float(res.slot_use.mean()),
        "episodes": n_ep,
        "starts": starts,
        "growth": g.cpu().tolist(),
    }
    if track:
        card["result"] = res
    card["curves"] = {f"{s / 1e9:g}b": [eq[k * n_ep + e, :int(episodes.n_checks[e]) + 1].cpu().tolist()
                                         for e in range(n_ep)] for k, s in enumerate(stacks)}
    return card
