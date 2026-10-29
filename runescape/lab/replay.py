"""Replay a champion over the whole archive as one account and log every decision, for a GE-screen viewer.

One episode runs from the first check after the feature warm-up to the end of the data, from one starting stack,
with the referee's usual rules (lab.env). The log is written as GE events, in time order:

- place: an offer goes into a GE slot at a check (with what the policy saw: quotes, their ages, and for buys the
  price it means to sell at and the margin after tax)
- fill: some of an offer traded in a 5m bucket (t is the bucket's start)
- collect: a finished offer is collected at the next check, freeing its slot
- cancel: an unfinished offer is aborted at a check (a buy's unspent gold comes back; a sell's units go back to the
  inventory, usually to be relisted at a new price)
- dump: at the end, everything still held is sold at the last instasell less the haircut and tax

The referee cancels every buy and withdraws every sell at each check and the policy proposes a fresh book; an offer
proposed again with the same item, side, price and remaining quantity is logged as staying in its slot instead.
Cash is rebuilt from the events and checked against the referee's final cash.

Most of the archive is the champion's training and validation data, so only the part after its validation weeks
(meta.segments) is an honest test. That part is forward data, so each replay is logged as a holdout look.

    uv run python -m lab.replay                               # newest soft-rules champion, run like the advisor
    uv run python -m lab.replay --stack 1.8e9 --slots 8
Output: data/lab/replays/<stamp>-<run>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

import torch

from backtest_flips import DAY, fmt_gp, fmt_ts
from lab import env as E
from lab import market as M
from lab import splits
from lab.evolve import POLICIES, RUNS_DIR, check_schedule, write_json
from lab.features import Features
from lab.market import LAB_DIR
from lab.policies.soft_rules import SoftRules, to_values

REPLAYS_DIR = LAB_DIR / "replays"
AS_THE_ADVISOR = {"soft-rules": "soft-rules-advisor"}  # how the advisor page runs each champion


def newest_run(policy: str) -> str:
    for d in sorted(RUNS_DIR.glob("*"), reverse=True):
        if (d / "champion.pt").exists() and torch.load(d / "champion.pt")["policy"] == policy:
            return d.name
    raise SystemExit(f"No {policy} run with a champion yet.")


class Log:
    """Turns the referee's books and fills into GE slot events, keeping the account's cash and stock alongside."""

    def __init__(self, mkt: E.MarketOnDevice, cfg: E.EnvConfig, cash: int, values: dict[str, float] | None):
        self.mkt, self.cfg, self.values = mkt, cfg, values
        self.cash = cash
        self.slots: list[dict | None] = [None] * cfg.slots
        self.inv: dict[int, int] = {}  # units held, bought ones included
        self.cost: dict[int, int] = {}  # gp paid for the units held
        self.events: list[dict] = []
        self.checks: list[dict] = []
        self.traded: set[int] = set()
        self.next_id = 0

    def tax(self, price: int, qty: int) -> int:
        return int(self.cfg.tax_of(torch.tensor(price), torch.tensor(qty)))

    def emit(self, t: int, kind: str, slot: int, offer: dict, **extra) -> None:
        self.events.append({"t": t, "type": kind, "slot": slot, "offer": offer["id"], "item": offer["item"],
                            "side": "buy" if offer["side"] == E.BUY else "sell", "price": offer["price"],
                            "qty": offer["qty"], "filled": offer["filled"], **extra})

    def context(self, c: int, t: int, item: int, side: int, price: int) -> dict:
        raw = self.mkt.features.raw
        sq, bq = int(raw["is_px"][c, item]), int(raw["ib_px"][c, item])
        ctx = {"instasell": sq, "instabuy": bq,
               "instasell_age_s": t - int(raw["is_ts"][c, item]) if raw["is_ts"][c, item] >= 0 else None,
               "instabuy_age_s": t - int(raw["ib_ts"][c, item]) if raw["ib_ts"][c, item] >= 0 else None}
        if side == E.BUY and self.values and bq > 0:
            sell = bq - math.floor(self.values["sell_outbid_spread"] * max(bq - sq, 0)) - 1
            ctx |= {"planned_sell": sell, "margin": sell - price - sell * 2 // 100}
        elif side == E.SELL and self.inv.get(item):
            ctx["avg_cost"] = self.cost[item] // self.inv[item]
        return ctx

    def check(self, t: int, c: int, equity: int, book: list[tuple]) -> None:
        """book: the referee's offers at this check, as (env slot, item, side, price, qty)."""
        wanted = {(item, side, price, qty): k for k, item, side, price, qty in book}
        for g, offer in enumerate(self.slots):  # offers placed at the last check
            if offer is None:
                continue
            left = offer["qty"] - offer["filled"]
            key = (offer["item"], offer["side"], offer["price"], left)
            if left and key in wanted:  # proposed again as it stands: it stays in its slot
                offer["env_slot"] = wanted.pop(key)
                continue
            if left == 0:
                self.emit(t, "collect", g, offer)
            elif offer["side"] == E.BUY:
                refund = offer["price"] * left
                self.cash += refund
                self.emit(t, "cancel", g, offer, refund=refund)
            else:
                self.emit(t, "cancel", g, offer)
            self.slots[g] = None
        for (item, side, price, qty), k in sorted(wanted.items(), key=lambda kv: kv[1]):
            g = self.slots.index(None)
            offer = {"id": self.next_id, "item": item, "side": side, "price": price, "qty": qty, "filled": 0,
                     "env_slot": k}
            self.next_id += 1
            self.slots[g] = offer
            self.traded.add(item)
            if side == E.BUY:
                self.cash -= price * qty
            self.emit(t, "place", g, offer, **self.context(c, t, item, side, price))
        self.checks.append({"t": t, "equity": equity, "cash": self.cash,
                            "held": {str(n): q for n, q in sorted(self.inv.items()) if q},
                            "slots": [None if o is None else o["id"] for o in self.slots]})

    def fill(self, ts: int, env_slot: int, item: int, side: int, qty: int, price: int) -> None:
        g, offer = next((g, o) for g, o in enumerate(self.slots) if o and o["env_slot"] == env_slot)
        assert (offer["item"], offer["side"], offer["price"]) == (item, side, price)
        offer["filled"] += qty
        extra = {"fill_qty": qty, "done": offer["filled"] == offer["qty"]}
        if side == E.BUY:
            self.inv[item] = self.inv.get(item, 0) + qty
            self.cost[item] = self.cost.get(item, 0) + qty * price
        else:
            tax = self.tax(price, qty)
            avg = self.cost[item] // self.inv[item]
            self.cost[item] -= avg * qty
            self.inv[item] -= qty
            self.cash += qty * price - tax
            extra |= {"tax": tax, "net": qty * price - tax, "profit": qty * (price - avg) - tax}
        self.emit(ts, "fill", g, offer, **extra)

    def finish(self, t: int) -> None:
        for g, offer in enumerate(self.slots):
            if offer is None:
                continue
            left = offer["qty"] - offer["filled"]
            if offer["side"] == E.BUY and left:
                self.cash += offer["price"] * left
                self.emit(t, "cancel", g, offer, refund=offer["price"] * left)
            else:
                self.emit(t, "collect" if not left else "cancel", g, offer)
            self.slots[g] = None
        i_end = self.mkt.market.bucket(t) - 1
        for item, units in sorted(self.inv.items()):
            if not units:
                continue
            price = int(self.mkt.last_is[i_end, item])
            if self.cfg.haircut:
                price = math.floor(price * (1 - self.cfg.haircut))
            tax = self.tax(price, units)
            net = units * price - tax
            self.cash += net
            self.events.append({"t": t, "type": "dump", "item": item, "qty": units, "price": price, "tax": tax,
                                "net": net, "profit": net - self.cost[item]})
            self.inv[item] = 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", help="run directory name (default: the newest soft-rules run with a champion)")
    parser.add_argument("--policy", choices=POLICIES, help="policy to run the champion's parameters with (default: "
                                                           "as the advisor runs it)")
    parser.add_argument("--stack", type=float, default=1.024e9, help="starting cash")
    parser.add_argument("--slots", type=int, default=6, help="free GE slots")
    parser.add_argument("--checks", help="check-in times, local, comma-separated (default: the run's)")
    parser.add_argument("--fill-model", help="default: the run's")
    parser.add_argument("--haircut", type=float, help="taken off the final dump price (default: the run's)")
    parser.add_argument("--reason", default="decision log for the GE viewer", help="goes in the holdout log")
    args = parser.parse_args()

    run_dir = RUNS_DIR / (args.run or newest_run("soft-rules"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    champion = torch.load(run_dir / "champion.pt")
    policy_name = args.policy or AS_THE_ADVISOR.get(champion["policy"], champion["policy"])
    if issubclass(POLICIES[policy_name], SoftRules) != issubclass(POLICIES[champion["policy"]], SoftRules):
        raise SystemExit(f"{policy_name} can't use a {champion['policy']} champion's parameters")
    a = config["args"]
    lag = a["obs_lag"]
    cfg = E.EnvConfig(fill_model=args.fill_model or a["fill_model"],
                      haircut=a["haircut"] if args.haircut is None else args.haircut, slots=args.slots)

    market = M.load()
    check_at = check_schedule({"checks": args.checks} if args.checks else a)
    mkt = E.MarketOnDevice(market, Features(market, lag, check_at=check_at), "cuda")
    start = int(next(t for t in mkt.features.checks if t >= market.t0 + splits.WARMUP))
    end = market.end
    episodes = E.Episodes.from_windows(mkt.features, [(start, end)])
    forward_start = config["splits"]["validation"]["end"]
    if end > forward_start:
        splits.HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with splits.HOLDOUT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": int(time.time()), "start": forward_start, "end": end,
                                "what": f"replay of {run_dir.name}", "reason": args.reason}) + "\n")
    dev = mkt.low.device
    stack = int(args.stack)
    policy = POLICIES[policy_name]().bind(champion["params"].to(dev)[None], 1)
    print(f"{run_dir.name} gen {champion.get('gen', '?')} as {policy_name}: {fmt_gp(stack)}, {cfg.slots} slots, "
          f"{cfg.fill_model}, {fmt_ts(start)} → {fmt_ts(end)} ({(end - start) / DAY:.1f} days, "
          f"{episodes.n_checks[0]} checks)", flush=True)
    t_run = time.monotonic()
    res = E.run(mkt, policy, episodes, torch.tensor([stack], device=dev), torch.zeros(1, dtype=torch.int64, device=dev),
                cfg, obs_lag=lag, track=True)
    print(f"simulated in {time.monotonic() - t_run:.0f}s", flush=True)

    values = None
    if issubclass(POLICIES[policy_name], SoftRules):
        values = {k: float(v[0]) for k, v in to_values(champion["params"][None]).items()}
    log = Log(mkt, cfg, stack, values)
    first = int(episodes.first_check[0])
    books: dict[int, list[tuple]] = {}
    for _, t, k, item, side, price, qty in res.books:
        books.setdefault(t, []).append((k, item, side, price, qty))
    fills = sorted(res.trades, key=lambda tr: (tr[1], tr[6]))
    f = 0
    for r in range(int(episodes.n_checks[0])):
        c = first + r
        t = int(mkt.features.checks[c])
        log.check(t, c, int(res.equity[0, r]), books.get(t, []))
        nxt = int(mkt.features.checks[c + 1]) if r + 1 < episodes.n_checks[0] else end
        while f < len(fills) and fills[f][1] < nxt:
            _, ts, item, side, qty, price, k = fills[f]
            log.fill(ts, k, item, side, qty, price)
            f += 1
    log.finish(end)
    final = int(res.final_cash[0])
    assert log.cash == final, f"cash rebuilt from the events {log.cash} != referee's {final}"

    # equity where the champion's training and validation ended, to tell the honest part of the run apart
    sp = config["splits"]
    eq_at = lambda ts: next((ch["equity"] for ch in log.checks if ch["t"] >= ts), final)
    segments = []
    for name, lo, hi in (("training", start, sp["train"]["end"]),
                         ("validation", sp["validation"]["start"], sp["validation"]["end"]),
                         ("forward", sp["validation"]["end"], end)):
        lo, hi = max(lo, start), min(hi, end)
        if hi <= lo:
            continue
        g = math.log((final if hi >= end else eq_at(hi)) / eq_at(lo))
        segments.append({"name": name, "start": lo, "end": hi, "growth": g,
                         "weekly": math.expm1(g * 7 * DAY / (hi - lo))})
    items = {str(n): {"id": int(market.item_ids[n]), "name": market.names[n], "limit": int(market.limits[n])}
             for n in sorted(log.traded)}
    out = {
        "meta": {"run": run_dir.name, "gen": champion.get("gen"), "policy": policy_name, "params": values,
                 "stack": stack, "slots": cfg.slots, "checks": list(check_at), "fill_model": cfg.fill_model, "haircut": cfg.haircut,
                 "obs_lag": lag, "start": start, "end": end, "final_cash": final, "splits": sp,
                 "segments": segments, "created": int(time.time()),
                 "notes": "Times are UTC epoch seconds; a fill's t is the start of its 5m bucket. Item keys index "
                          "`items`. Prices and cash are whole gp."},
        "items": items,
        "checks": log.checks,
        "events": log.events,
    }
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = REPLAYS_DIR / f"{stamp}-{run_dir.name}.json"
    write_json(path, out)

    kinds = {}
    for ev in log.events:
        kinds[ev["type"]] = kinds.get(ev["type"], 0) + 1
    print(f"{fmt_gp(stack)} → {fmt_gp(final)}; {len(items)} items traded; "
          + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    for s in segments:
        print(f"  {s['name']:10} {fmt_ts(s['start'])} → {fmt_ts(s['end'])}  {s['weekly'] * 100:+6.1f}%/wk")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
