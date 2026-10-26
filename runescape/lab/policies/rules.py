"""The backtester's hand-written strategies (backtest_flips.Strategy) as a lab policy.

A vectorized port of Simulator.rank_candidates / plan_offers / checkpoint's offer logic onto the observation,
reusing backtest_flips.choose_offers for the knapsack. Same arithmetic in the same order, so with obs_lag 0 and the
backtester's tax and fill rules it reproduces backtest_flips exactly (lab/parity.py checks that). Runs on the CPU.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from backtest_flips import (DAY, FIVE_MIN, GE_TAX, MAX_BUY_PRICE, MAX_QUOTE_AGE_S, MIN_4H_VOLUME_GP, MIN_BUY_PRICE,
                            Strategy, choose_offers)
from lab.policies.base import BUY, SELL, Observation, Proposals


def sale_tax_1(price: np.ndarray) -> np.ndarray:
    """backtest_flips.sale_tax(price, 1), elementwise."""
    return np.trunc(price * 1 * GE_TAX).astype(np.int64)


class RulesPolicy:
    def __init__(self, strategy: Strategy):
        assert strategy.drift_window_days == 3 and strategy.recent_units == 3, "features hold 3-day/3-unit values"
        self.strategy = strategy
        self.name = f"rules:{strategy.name}"

    def candidates(self, raw: dict, t: int, equity: int, inv: np.ndarray, limit_left: np.ndarray,
                   data_start: int, max_quote_age_h: float | None) -> list[SimpleNamespace]:
        """Simulator.rank_candidates: eligible items, best score first."""
        s = self.strategy
        age_h = s.max_quote_age_h if max_quote_age_h is None else max_quote_age_h
        sq, bq = raw["is_px"], raw["ib_px"]
        sts, bts = raw["is_ts"], raw["ib_ts"]
        rs, rb = raw["is_r3"], raw["ib_r3"]
        ok = (sts >= 0) & (bts >= 0)
        if s.pricing == "recent":
            buy, sell = rs + s.outbid, rb - s.outbid
        else:
            buy, sell = sq + s.outbid, bq - s.outbid
        ok &= (MIN_BUY_PRICE <= buy) & (buy <= MAX_BUY_PRICE)
        oldest = np.minimum(sts, bts)
        ok &= ~(oldest + FIVE_MIN < t - MAX_QUOTE_AGE_S)
        vol4h = raw["vol24"] / 6
        ok &= ~(vol4h * buy <= MIN_4H_VOLUME_GP)
        offer_margin = sell - buy - sale_tax_1(sell)
        recent_sell_price = rb - s.outbid
        recent_margin = recent_sell_price - (rs + s.outbid) - sale_tax_1(recent_sell_price)
        margin = {"instant": offer_margin, "recent": recent_margin,
                  "conservative": np.minimum(offer_margin, recent_margin)}[s.selection]
        ok &= margin > 0
        if age_h is not None:
            ok &= ~(t - oldest > age_h * 3600)
        with np.errstate(invalid="ignore", divide="ignore"):
            if s.min_margin_pct is not None:
                ok &= ~(offer_margin < s.min_margin_pct * buy)
            if s.max_outlier_ratio is not None:
                ok &= ~(offer_margin > s.max_outlier_ratio * recent_margin)
            if s.max_drift is not None:
                drifted = np.zeros_like(ok)
                for quote, median in ((sq, raw["is_med"]), (bq, raw["ib_med"])):
                    drifted |= ~np.isnan(median) & (np.abs(quote / median - 1) > s.max_drift)
                ok &= ~drifted
            if s.max_margin_pct is not None:
                ok &= ~(bq - sq > s.max_margin_pct * sq)
            if s.min_trade_buckets is not None:
                ok &= ~(np.minimum(raw["is_tb"], raw["ib_tb"]) < s.min_trade_buckets)
            if s.max_volume_spike is not None:
                prior_days = min(7.0, (t - DAY - data_start) / DAY)
                if prior_days >= 1:
                    ok &= ~(vol4h * 6 > s.max_volume_spike * raw["prior_avg"])
        out = []
        for n in np.nonzero(ok)[0]:
            max_qty = math.floor(min(vol4h[n], int(limit_left[n])))
            if s.max_item_share is not None:
                share_qty = math.floor(s.max_item_share * equity / int(buy[n]))
                if s.item_share_min_one:
                    share_qty = max(share_qty, 1)
                max_qty = min(max_qty, share_qty - int(inv[n]))
            if max_qty < 1:
                continue
            out.append(SimpleNamespace(idx=int(n), buy_price=int(buy[n]), margin=int(margin[n]), max_qty=max_qty,
                                       score=max_qty * int(margin[n])))
        out.sort(key=lambda cand: cand.score, reverse=True)
        return out

    def plan(self, cands: list, cash: int, free_slots: int) -> list:
        """Simulator.plan_offers."""
        policy = self.strategy.slot_policy
        one_each = [SimpleNamespace(**{**vars(c), "max_qty": 1}) for c in cands]
        if policy == "one-each":
            return choose_offers(one_each, cash, free_slots)
        if policy == "fill-slots":
            return choose_offers(cands, cash, free_slots, exact=True)
        picks = choose_offers(cands, cash, free_slots)
        if policy == "one-each-if-short" and len(picks) < free_slots and len(cands) > len(picks):
            spread = choose_offers(one_each, cash, free_slots)
            if len(spread) > len(picks):
                picks = spread
        return picks

    def act(self, obs: Observation) -> Proposals:
        s = self.strategy
        raw_all = obs.mkt.features.raw
        data_start = obs.mkt.market.data_start
        c, t, ep = obs.c.cpu().numpy(), obs.t.cpu().numpy(), obs.ep.cpu().numpy()
        active = obs.active.cpu().numpy()
        cash, equity = obs.cash.cpu().numpy(), obs.equity.cpu().numpy()
        inv, limit_left, listed = obs.inv.cpu().numpy(), obs.limit_left.cpu().numpy(), obs.listed_price.cpu().numpy()
        rows = []
        for b in range(len(cash)):
            if not active[b]:
                rows.append([])
                continue
            e = ep[b]
            raw = {k: v[c[e]] for k, v in raw_all.items()}
            offers = []
            held = np.nonzero(inv[b] > 0)[0]
            for n in held:  # list everything held at instabuy - outbid (keep the old price if there's no quote)
                if raw["ib_ts"][n] >= 0:
                    price = (raw["ib_r3"][n] if s.pricing == "recent" else raw["ib_px"][n]) - s.outbid
                else:
                    price = listed[b, n]
                offers.append((int(n), SELL, int(price), int(inv[b, n])))
            free_slots = obs.slots - len(held)
            cands = self.candidates(raw, int(t[e]), int(equity[b]), inv[b], limit_left[b], data_start, None)
            picks = self.plan(cands, int(cash[b]), free_slots)
            for age_h in s.fallback_quote_age_h:
                cash_left = int(cash[b]) - sum(p.buy_price * q for p, q in picks)
                slots_left = free_slots - len(picks)
                if slots_left <= 0 or cash_left < MIN_BUY_PRICE:
                    break
                picked = {p.idx for p, _ in picks}
                extra = [x for x in self.candidates(raw, int(t[e]), int(equity[b]), inv[b], limit_left[b],
                                                    data_start, age_h) if x.idx not in picked]
                picks += choose_offers(extra, cash_left, slots_left)
            offers += [(p.idx, BUY, p.buy_price, q) for p, q in picks]
            rows.append(offers)
        M = max(1, max(len(r) for r in rows))
        B = len(rows)
        arr = np.zeros((4, B, M), dtype=np.int64)
        arr[0] = -1
        prio = np.full((B, M), -np.inf, dtype=np.float32)
        for b, offers in enumerate(rows):
            for k, offer in enumerate(offers):
                arr[:, b, k] = offer
                prio[b, k] = M - k  # sells first, then the picks in order; all fit in the slots
        dev = obs.cash.device
        item, side, price, qty = (torch.tensor(a, device=dev) for a in arr)
        return Proposals(item, side, price, qty, torch.tensor(prio, device=dev))
