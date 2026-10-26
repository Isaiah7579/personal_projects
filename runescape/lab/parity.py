"""Check the lab's referee against backtest_flips: same strategy, same window, same answer.

Runs a backtest_flips strategy through backtest_flips.Simulator and through lab.env with RulesPolicy (obs_lag 0,
no haircut, the backtester's tax rounding and fill model), then compares final cash, the equity at every check,
fill counts and per-item profit. Any difference means the referee (or a feature) is wrong.

    uv run python -m lab.parity                                  # baseline + recommended, average + floor
    uv run python -m lab.parity --strategy recommended --fill-model floor --days 20
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from backtest_flips import BUDGET, DAY, LOAD_MIN_PRICE, STRATEGIES, Simulator, fmt_gp, fmt_ts
from lab import env as E
from lab import market as M
from lab.features import Features
from lab.policies.rules import RulesPolicy


def compare(mkt: E.MarketOnDevice, strategy_name: str, fill_model: str, start: int, end: int) -> bool:
    market = mkt.market
    strategy = STRATEGIES[strategy_name]
    checks = [int(t) for t in mkt.features.checks if start <= t < end]
    t_run = time.monotonic()
    sim = Simulator(market.items, checks, end, BUDGET, strategy, data_start=market.data_start, quiet=True,
                    fill_model=fill_model)
    sim.run()
    t_sim = time.monotonic() - t_run

    t_run = time.monotonic()
    episodes = E.Episodes.from_windows(mkt.features, [(start, end)])
    cfg = E.EnvConfig(fill_model=fill_model, haircut=0.0, tax="backtester")
    dev = mkt.low.device
    res = E.run(mkt, RulesPolicy(strategy), episodes, torch.tensor([BUDGET], device=dev),
                torch.zeros(1, dtype=torch.int64, device=dev), cfg, obs_lag=0, track=True)
    t_env = time.monotonic() - t_run

    ours = res.equity[0, :len(checks)].tolist()
    theirs = [equity for _, equity, _, _ in sim.equity_log[:len(checks)]]
    first_diff = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
    led = {k: v[0].cpu().numpy() for k, v in res.ledger.items()}
    profit_ours = led["sell_net"] + led["dump_net"] - led["spent"]
    item_diffs = []
    for n, item in enumerate(market.items):
        theirs_p = sim.ledgers[item.id].profit if item.id in sim.ledgers else 0
        if int(profit_ours[n]) != theirs_p:
            item_diffs.append((item.name, theirs_p, int(profit_ours[n])))
    checks_ok = {
        "final cash": (sim.cash, int(res.final_cash[0])),
        "buy fills": (sim.fills["buy"], int(res.buy_fills[0])),
        "sell fills": (sim.fills["sell"], int(res.sell_fills[0])),
    }
    ok = first_diff is None and not item_diffs and all(a == b for a, b in checks_ok.values())
    print(f"{'PASS' if ok else 'FAIL'}  {strategy_name:12} {fill_model:8} profit {fmt_gp(sim.cash - BUDGET):>9} "
          f"(backtester {t_sim:.0f}s, lab {t_env:.0f}s)")
    for label, (a, b) in checks_ok.items():
        if a != b:
            print(f"      {label}: backtester {a:,}  lab {b:,}")
    if first_diff is not None:
        print(f"      equity differs from check {first_diff} ({fmt_ts(checks[first_diff])}): "
              f"backtester {theirs[first_diff]:,}  lab {ours[first_diff]:,}")
    for name, a, b in item_diffs[:10]:
        print(f"      item {name}: backtester {fmt_gp(a)}  lab {fmt_gp(b)}")
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--fill-model", action="append", choices=("average", "floor"))
    parser.add_argument("--days", type=float, help="only the last N days (default: the whole cache)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    market = M.load(min_bucket_price=LOAD_MIN_PRICE)
    features = Features(market, obs_lag=0)
    mkt = E.MarketOnDevice(market, features, args.device)
    end = market.end
    start = market.data_start + DAY if args.days is None else max(market.data_start + DAY, int(end - args.days * DAY))
    print(f"Window {fmt_ts(start)} → {fmt_ts(end)} ({(end - start) / DAY:.1f} days), {market.N} items")
    results = [compare(mkt, s, f, start, end) for s in (args.strategy or ["baseline", "recommended"])
               for f in (args.fill_model or ["average", "floor"])]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
