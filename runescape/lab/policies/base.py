"""The policy interface every strategy implements.

A policy maps an Observation (what can be seen at a check, see lab.env.Observation) to Proposals: up to M offers
per account (item, side, price, qty, priority). The referee's `legalize` keeps the best `slots` legal ones, so a
policy never has to get the bookkeeping exactly right, and rule sets, neural nets and anything else are scored the
same way.

Parametric policies (the ones evolution can tune) also have `dim` parameters and build a batch of variants with
`with_params(params [P, dim])`; account b then uses variant b // E of the E episodes (b % E).
"""

from __future__ import annotations

from typing import Protocol

import torch

from lab.env import BUY, SELL, Observation, Proposals

__all__ = ["BUY", "SELL", "Observation", "Policy", "Proposals", "empty"]


class Policy(Protocol):
    name: str

    def act(self, obs: Observation) -> Proposals: ...


def empty(B: int, M: int, device) -> Proposals:
    z = torch.zeros(B, M, dtype=torch.int64, device=device)
    return Proposals(z - 1, z.clone(), z.clone(), z.clone(), torch.zeros(B, M, device=device))
