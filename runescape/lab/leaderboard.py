"""Forward scores: every run's champion on the data that arrived after its validation weeks.

That data played no part in training or in picking the champion, and it grows as the sync collects more, so
re-running this later gives each old champion a longer honest test. The hand-written rules are scored on the same
weeks alongside. Each run's result goes to its scores.json (the dashboard's Forward columns), and every evaluation
is logged as a holdout look (lab.splits.open_holdout's log), because it is.

    uv run python -m lab.leaderboard --reason "checking the first neural run"
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from backtest_flips import DAY, STRATEGIES, fmt_ts
from lab import env as E
from lab import market as M
from lab import score, splits
from lab.evolve import POLICIES, RUNS_DIR, write_json
from lab.features import Features
from lab.policies.rules import RulesPolicy


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reason", required=True, help="why you're looking (goes in the holdout log)")
    parser.add_argument("--run", action="append", help="only these runs (default: every run with a champion)")
    args = parser.parse_args()

    runs = [d for d in sorted(RUNS_DIR.glob("*")) if (d / "champion.pt").exists() and (not args.run or d.name in args.run)]
    if not runs:
        raise SystemExit("No runs with a champion yet.")
    market = M.load()
    devices: dict[int, E.MarketOnDevice] = {}
    for d in runs:
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        a = config["args"]
        lag = a["obs_lag"]
        if lag not in devices:
            devices[lag] = E.MarketOnDevice(market, Features(market, lag), "cuda")
        mkt = devices[lag]
        forward = splits.Split("forward", config["splits"]["validation"]["end"], market.end)
        if forward.end - forward.start < splits.EPISODE:
            print(f"{d.name}: only {(forward.end - forward.start) / DAY:.1f} days after its validation; skipped")
            continue
        splits.HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with splits.HOLDOUT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": int(time.time()), "start": forward.start, "end": forward.end,
                                "what": f"forward score of {d.name}", "reason": args.reason}) + "\n")
        cfg = E.EnvConfig(fill_model=a["fill_model"], haircut=a["haircut"])
        champion = torch.load(d / "champion.pt")
        card = score.evaluate(mkt, POLICIES[champion["policy"]](), forward, cfg, lag, params=champion["params"])
        rules = score.evaluate(mkt, RulesPolicy(STRATEGIES["recommended"]), forward, cfg, lag)
        result = {"score": card["score"], "worst": card["worst"], "cvar25": card["cvar25"],
                  "by_stack": card["by_stack"], "rules": rules["score"], "rules_worst": rules["worst"],
                  "start": forward.start, "end": forward.end, "days": (forward.end - forward.start) / DAY,
                  "episodes": card["episodes"], "evaluated": int(time.time()), "reason": args.reason}
        write_json(d / "scores.json", {"forward": result})
        print(f"{d.name}: {forward.describe()}\n  champion {card['score']:+.4f} (worst {card['worst']:+.4f})   "
              f"rules {rules['score']:+.4f} (worst {rules['worst']:+.4f})")
    print(f"Holdout looks so far: {splits.holdout_looks()} (latest data {fmt_ts(market.end)})")


if __name__ == "__main__":
    main()
