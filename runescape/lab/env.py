"""The referee: a batched GPU simulator of GE accounts, with the backtester's rules.

B independent accounts trade in lockstep, each in one of E episodes (an episode = a run of check times and an end
time; accounts in the same episode see the same market). At each check, per account:

1. Equity = cash + gold reserved in buy offers + stock at the last instasell less tax (logged; lag-0 prices).
2. Buy offers are cancelled and refunded; units bought since the last check join the inventory; sell offers are
   withdrawn (their units stay in the inventory). The policy then proposes a new book, and `legalize` keeps the
   best 8 legal offers.
3. Until the next check, each 5m bucket fills buys whose price is >= the bucket's instasell average and sells whose
   price is <= its instabuy average, up to the fill model's cap. Units bought can't be sold before the next check.

At the episode's end, buys are cancelled and everything held is dumped at the last instasell (less a haircut and
tax). This mirrors backtest_flips.Simulator.checkpoint / replay_market / finish; parity.py checks it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from backtest_flips import BUY_LIMIT_WINDOW_S, FIVE_MIN, GE_SLOTS, GE_TAX
from lab.features import Features
from lab.market import Market

BUY, SELL = 0, 1
LIMIT_BUCKETS = BUY_LIMIT_WINDOW_S // FIVE_MIN  # 48


@dataclass
class EnvConfig:
    fill_model: str = "share:0.5"  # average (whole bucket volume) | floor (1 unit) | share:x (max(1, x*volume))
    haircut: float = 0.03  # taken off the end-of-episode dump price
    tax: str = "per_item"  # per_item (RS3: each item's 2% rounded down) | backtester (floor of 2% of
    # the total, on everything: backtest_flips.sale_tax)
    slots: int = GE_SLOTS
    held_slots: int = 16  # how many held items an observation lists per account

    def fill_cap(self, volume: torch.Tensor) -> torch.Tensor:
        if self.fill_model == "average":
            return volume
        if self.fill_model == "floor":
            return torch.ones_like(volume)
        share = float(self.fill_model.split(":")[1])
        return torch.clamp((volume * share).floor().long(), min=1)

    def tax_of(self, price: torch.Tensor, qty: torch.Tensor) -> torch.Tensor:
        """Tax on selling qty at price."""
        if self.tax == "backtester":
            return (price.double() * qty.double() * GE_TAX).floor().long()
        return qty * (price * 2 // 100)


@dataclass
class Episodes:
    """E episodes: the checks each one trades at (indices into Features.checks) and its end time."""
    first_check: np.ndarray  # [E]
    n_checks: np.ndarray  # [E]
    end: np.ndarray  # [E] end timestamps (bucket boundaries)

    @classmethod
    def from_windows(cls, features: Features, windows: list[tuple[int, int]]) -> Episodes:
        firsts, counts, ends = [], [], []
        for start, end in windows:
            idx = np.nonzero((features.checks >= start) & (features.checks < end))[0]
            if not len(idx):
                raise ValueError(f"no checks in window {start}-{end}")
            firsts.append(idx[0])
            counts.append(len(idx))
            ends.append(end)
        return cls(np.array(firsts), np.array(counts), np.array(ends, dtype=np.int64))

    @property
    def E(self) -> int:
        return len(self.first_check)


class MarketOnDevice:
    """The market and its features as tensors, shared by every run on one device."""

    def __init__(self, market: Market, features: Features, device: str = "cuda", universe_k: int = 256):
        self.market, self.features, self.device = market, features, device
        dev = torch.device(device)
        self.t0, self.T, self.N = market.t0, market.T, market.N
        self.low = torch.tensor(market.low, device=dev)
        self.high = torch.tensor(market.high, device=dev)
        self.low_vol = torch.tensor(market.low_vol, device=dev, dtype=torch.int64)
        self.high_vol = torch.tensor(market.high_vol, device=dev, dtype=torch.int64)
        last_is, last_ib = market.last_quotes()
        self.last_is = torch.tensor(last_is, device=dev)
        self.last_ib = torch.tensor(last_ib, device=dev)
        self.limits = torch.tensor(market.limits, device=dev)
        self.checks = torch.tensor(features.checks, device=dev)
        self.raw = {k: torch.tensor(v, device=dev) for k, v in features.raw.items()}
        self.x = torch.tensor(features.inputs(), device=dev)  # [C, N, F]
        self.time_x = torch.tensor(features.time_inputs(), device=dev)  # [C, 4]
        self.universe_all = torch.tensor(features.universe(universe_k), device=dev)  # [C, K]
        self.universe = self.universe_all
        # a fixed half/half split of the items by a hash of their ID, for --item-split
        self.item_group = torch.tensor((market.item_ids * 2654435761 % 2 ** 32) >> 31, device=dev)

    def use_items(self, group: int | None) -> None:
        """Let learned policies buy only items of one half of the item split (None: every item)."""
        if group is None:
            self.universe = self.universe_all
            return
        u = self.universe_all
        keep = (u >= 0) & (self.item_group[u.clamp(min=0)] == group)
        # keep each check's universe sorted by GP volume, with the other half's slots padded out at the end
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
        self.universe = torch.where(keep, u, -1).gather(1, order)

    def bucket(self, ts: torch.Tensor) -> torch.Tensor:
        return (ts - self.t0) // FIVE_MIN


@dataclass
class Observation:
    """What a policy sees at a check. Market data is indexed by check (`c`); accounts map to episodes by `ep`."""
    mkt: MarketOnDevice  # policies may only use .raw / .x / .time_x / .universe at index c, and .limits
    c: torch.Tensor  # [E] check index into the features
    t: torch.Tensor  # [E] check time
    ep: torch.Tensor  # [B] episode of each account
    active: torch.Tensor  # [B] accounts still trading
    cash: torch.Tensor  # [B] after buy offers were refunded
    equity: torch.Tensor  # [B] marked with the prices the check can see
    inv: torch.Tensor  # [B, N] units held (sellable now)
    limit_left: torch.Tensor  # [B, N] remaining 4h buy limit
    listed_price: torch.Tensor  # [B, N] price of the previous sell offer for the item (0 if none)
    held: torch.Tensor  # [B, H] items held, most units first, -1 padded
    slots: int


@dataclass
class Proposals:
    """Up to M offers per account. (item, side) pairs must be unique per account; item -1 = no offer."""
    item: torch.Tensor  # [B, M] long
    side: torch.Tensor  # [B, M] BUY / SELL
    price: torch.Tensor  # [B, M] int64
    qty: torch.Tensor  # [B, M] int64
    priority: torch.Tensor  # [B, M] float, higher first

    @staticmethod
    def cat(*parts: Proposals) -> Proposals:
        return Proposals(*(torch.cat([getattr(p, f) for p in parts], dim=1)
                           for f in ("item", "side", "price", "qty", "priority")))


def legalize(p: Proposals, cash: torch.Tensor, inv: torch.Tensor, limit_left: torch.Tensor,
             slots: int) -> tuple[Proposals, torch.Tensor]:
    """Turn any proposals into a legal book: sells only what's held, buys within the 4h limit, positive prices,
    the `slots` highest-priority offers, and buys trimmed to the cash available in priority order.
    Returns the book as [B, slots] and the cash it reserves."""
    B = cash.shape[0]
    if p.item.shape[1] < slots:  # pad so there are always `slots` entries to pick from
        pad = slots - p.item.shape[1]
        z = torch.zeros(B, pad, dtype=torch.int64, device=cash.device)
        p = Proposals.cat(p, Proposals(z - 1, z, z, z, torch.zeros(B, pad, device=cash.device)))
    it = p.item.clamp(min=0)
    qty = torch.where(p.side == SELL, torch.minimum(p.qty, inv.gather(1, it)),
                      torch.minimum(p.qty, limit_left.gather(1, it)))
    valid = (p.item >= 0) & (p.price > 0) & (qty > 0)
    prio = torch.where(valid, p.priority.float(), torch.tensor(-torch.inf, device=cash.device))
    top_prio, idx = prio.topk(slots, dim=1)
    keep = torch.isfinite(top_prio)
    item = torch.where(keep, p.item.gather(1, idx), -1)
    side = p.side.gather(1, idx)
    price = p.price.gather(1, idx)
    qty = torch.where(keep, qty.gather(1, idx), 0)
    left = cash.clone()
    for s in range(slots):  # cash, in priority order
        is_buy = keep[:, s] & (side[:, s] == BUY)
        afford = torch.where(is_buy, torch.minimum(qty[:, s], left // price[:, s].clamp(min=1)), qty[:, s])
        qty[:, s] = afford
        left -= torch.where(is_buy, afford * price[:, s], 0)
    item = torch.where(qty > 0, item, -1)
    return Proposals(item, side, price, qty, top_prio), cash - left


@dataclass
class Result:
    final_cash: torch.Tensor  # [B]
    start_cash: torch.Tensor  # [B]
    equity: torch.Tensor  # [B, R] at each check (lag-0 marks), then final cash at index n_checks
    ep: torch.Tensor  # [B]
    buy_fills: torch.Tensor  # [B] (offer, bucket) fill events, like Simulator.fills["buy"]
    sell_fills: torch.Tensor
    trades: list[tuple] = field(default_factory=list)  # (account, bucket ts, item, side, qty, price) if tracked
    ledger: dict[str, torch.Tensor] = field(default_factory=dict)  # [B, N] per-item totals if tracked
    slot_use: torch.Tensor | None = None  # [B] mean offers placed per check

    @property
    def log_growth(self) -> torch.Tensor:
        return torch.log(self.final_cash.double().clamp(min=1) / self.start_cash.double())


def run(mkt: MarketOnDevice, policy, episodes: Episodes, cash: torch.Tensor, ep: torch.Tensor,
        cfg: EnvConfig, obs_lag: int = 0, track: bool = False) -> Result:
    """Simulate accounts with starting `cash` [B], account b trading in episode ep[b]."""
    dev = cash.device
    B, N, S = cash.shape[0], mkt.N, cfg.slots
    ar = torch.arange(B, device=dev)
    first = torch.tensor(episodes.first_check, device=dev)
    n_checks = torch.tensor(episodes.n_checks, device=dev)
    ends = torch.tensor(episodes.end, device=dev)
    R = int(n_checks.max())

    start_cash = cash.clone()
    cash = cash.clone()
    inv = torch.zeros(B, N, dtype=torch.int64, device=dev)
    pending = torch.zeros_like(inv)
    listed = torch.zeros_like(inv)
    s_item = torch.full((B, S), -1, dtype=torch.int64, device=dev)
    s_side = torch.zeros_like(s_item)
    s_price = torch.zeros_like(s_item)
    s_qty = torch.zeros_like(s_item)
    s_filled = torch.zeros_like(s_item)
    ring_item = torch.zeros(B, LIMIT_BUCKETS, S, dtype=torch.int64, device=dev)
    ring_qty = torch.zeros_like(ring_item)
    ring_bucket = torch.full((B, LIMIT_BUCKETS), -10 ** 9, dtype=torch.int64, device=dev)
    equity_log = torch.zeros(B, R + 1, dtype=torch.int64, device=dev)
    buy_fills = torch.zeros(B, dtype=torch.int64, device=dev)
    sell_fills = torch.zeros_like(buy_fills)
    offers_placed = torch.zeros_like(buy_fills)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    final_cash = torch.zeros_like(cash)
    trades, ledger = [], {}
    if track:
        ledger = {k: torch.zeros(B, N, dtype=torch.int64, device=dev)
                  for k in ("bought", "spent", "sold", "sell_net", "tax", "dumped", "dump_net")}

    def marked_value(units: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
        """Units [B, N] at the last instasell of bucket_idx [B] (and before), less tax."""
        held_units, held_items = units.topk(min(64, N), dim=1)  # only held items have value; dense if > 64
        if bool((held_units[:, -1] > 0).any()):
            held_units, held_items = units, torch.arange(N, device=dev).expand(B, -1)
        price = mkt.last_is[bucket_idx.clamp(min=0)[:, None], held_items]
        return (held_units * price - cfg.tax_of(price, held_units)).sum(1)

    for r in range(R):
        active = (r < n_checks[ep]) & ~done
        c_e = (first + r).clamp(max=len(mkt.checks) - 1)
        t_e = mkt.checks[c_e]
        t_b = t_e[ep]
        i_b = mkt.bucket(t_b)

        # 1. equity at the check
        reserved = torch.where(s_side == BUY, s_price * (s_qty - s_filled), 0).sum(1)
        units = inv + pending
        equity = cash + reserved + marked_value(units, i_b - 1)
        equity_log[:, r] = torch.where(active, equity, equity_log[:, r])
        obs_equity = cash + reserved + marked_value(units, mkt.bucket(t_b - obs_lag) - 1) if obs_lag else equity

        # 2. cancel buys, take in bought units, withdraw sells
        cash = cash + torch.where(active, reserved, 0)
        inv = torch.where(active[:, None], inv + pending, inv)
        pending = torch.where(active[:, None], 0, pending)
        listed = torch.where(inv > 0, listed, 0)
        recent = ring_bucket > (i_b - LIMIT_BUCKETS)[:, None]  # bucket ts > t - 4h
        used = torch.zeros_like(inv).scatter_add_(1, ring_item.view(B, -1),
                                                  (ring_qty * recent[:, :, None]).view(B, -1))
        limit_left = (mkt.limits[None, :] - used).clamp(min=0)
        held_count, held = inv.topk(min(cfg.held_slots, N), dim=1)
        held = torch.where(held_count > 0, held, -1)

        obs = Observation(mkt, c_e, t_e, ep, active, cash, obs_equity, inv, limit_left, listed, held, S)
        book, reserve = legalize(policy.act(obs), cash, inv, limit_left, S)
        live = active[:, None] & (book.item >= 0)
        s_item = torch.where(live, book.item, -1)
        s_side, s_price = book.side, book.price
        s_qty = torch.where(live, book.qty, 0)
        s_filled = torch.zeros_like(s_qty)
        cash = cash - torch.where(active, reserve, 0)
        offers_placed += live.sum(1)
        is_sell = live & (s_side == SELL)
        for k in range(S):  # one slot at a time: padded slots all point at item 0
            col = s_item[:, k].clamp(min=0)
            listed[ar, col] = torch.where(is_sell[:, k], s_price[:, k], listed[ar, col])

        # 3. replay buckets until the next check (or the episode's end)
        last = r + 1 >= n_checks
        nxt = torch.where(last, ends, mkt.checks[(c_e + 1).clamp(max=len(mkt.checks) - 1)])
        length = (nxt - t_e) // FIVE_MIN  # [E]
        L = int(length[(r < n_checks)].max()) if bool((r < n_checks).any()) else 0
        len_b = length[ep]
        it = s_item.clamp(min=0)
        buy_slot = (s_item >= 0) & (s_side == BUY)
        sell_slot = (s_item >= 0) & (s_side == SELL)
        for s in range(L):
            stepping = active & (s < len_b)
            bi = (i_b + s).clamp(max=mkt.T - 1)
            bidx = bi[:, None].expand(-1, S)
            lv = mkt.low_vol[bidx, it]
            hv = mkt.high_vol[bidx, it]
            buy = buy_slot & stepping[:, None] & (lv > 0) & (mkt.low[bidx, it] <= s_price) & (s_filled < s_qty)
            q_buy = torch.where(buy, torch.minimum(s_qty - s_filled, cfg.fill_cap(lv)), 0)
            sell = sell_slot & stepping[:, None] & (hv > 0) & (mkt.high[bidx, it] >= s_price) & (s_qty > 0)
            q_sell = torch.where(sell, torch.minimum(s_qty, cfg.fill_cap(hv)), 0)
            s_filled += q_buy
            pending.scatter_add_(1, it, q_buy)
            gross = q_sell * s_price
            tax = cfg.tax_of(s_price, q_sell)
            s_qty -= q_sell
            cash += (gross - tax).sum(1)
            inv.scatter_add_(1, it, -q_sell)
            buy_fills += (q_buy > 0).sum(1)
            sell_fills += (q_sell > 0).sum(1)
            pos = bi % LIMIT_BUCKETS
            ring_item[ar, pos] = torch.where(stepping[:, None], it, ring_item[ar, pos])
            ring_qty[ar, pos] = torch.where(stepping[:, None], q_buy, ring_qty[ar, pos])
            ring_bucket[ar, pos] = torch.where(stepping, bi, ring_bucket[ar, pos])
            if track:
                ledger["bought"].scatter_add_(1, it, q_buy)
                ledger["spent"].scatter_add_(1, it, q_buy * s_price)
                ledger["sold"].scatter_add_(1, it, q_sell)
                ledger["sell_net"].scatter_add_(1, it, gross - tax)
                ledger["tax"].scatter_add_(1, it, tax)
                if bool((q_buy > 0).any() or (q_sell > 0).any()):
                    ts = mkt.t0 + bi * FIVE_MIN
                    for q, side_id in ((q_buy, BUY), (q_sell, SELL)):
                        for b, k in (q > 0).nonzero().tolist():
                            trades.append((b, int(ts[b]), int(s_item[b, k]), side_id, int(q[b, k]),
                                           int(s_price[b, k])))

        # 4. episodes ending after this interval: cancel buys, dump everything
        ending = active & last[ep]
        if bool(ending.any()):
            refund = torch.where(buy_slot, s_price * (s_qty - s_filled), 0).sum(1)
            units = inv + pending
            i_end = mkt.bucket(ends[ep]) - 1
            price = mkt.last_is[i_end.clamp(min=0)]
            price = (price.double() * (1 - cfg.haircut)).floor().long() if cfg.haircut else price
            dump_tax = cfg.tax_of(price, units)
            proceeds = (units * price - dump_tax).sum(1)
            cash = cash + torch.where(ending, refund + proceeds, 0)
            if track:
                e = ending[:, None]
                ledger["dumped"] += torch.where(e, units, 0)
                ledger["dump_net"] += torch.where(e, units * price - dump_tax, 0)
                ledger["tax"] += torch.where(e, dump_tax, 0)
            inv = torch.where(ending[:, None], 0, inv)
            pending = torch.where(ending[:, None], 0, pending)
            s_item = torch.where(ending[:, None], -1, s_item)
            final_cash = torch.where(ending, cash, final_cash)
            equity_log[ar, n_checks[ep]] = torch.where(ending, cash, equity_log[ar, n_checks[ep]])
            done |= ending

    return Result(final_cash, start_cash, equity_log, ep, buy_fills, sell_fills, trades, ledger,
                  offers_placed.double() / n_checks[ep].double())
