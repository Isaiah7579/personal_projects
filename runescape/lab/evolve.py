"""Evolve a policy with evolution strategies, scored by the referee; writes a run the dashboard shows live.

Each generation: P = 2h + 1 variants (the current parameters θ plus h mirrored pairs θ ± σε) each trade the same
E random week-long training episodes, at stacks drawn log-uniformly from 1.5b-6b. A variant's fitness is the CVaR25
of its log growth over those episodes (the mean of its worst quarter: it has to hold up every week, not just
average well). θ moves along the rank-weighted noise (OpenAI-ES) with Adam. Every --eval-every generations, θ is
scored on the validation split; the best so far is the champion. The holdout is never touched here.

    uv run python -m lab.evolve --policy soft-rules --gens 200
    uv run python -m lab.evolve --policy neural --gens 2000 --pairs 512
Output: data/lab/runs/<stamp>-<policy>/ (config.json, gens.jsonl, evals/, champion.*, summary.json)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from backtest_flips import STRATEGIES
from lab import env as E
from lab import market as M
from lab import score, splits
from lab.features import Features
from lab.market import LAB_DIR
from lab.policies.neural import Neural
from lab.policies.rules import RulesPolicy
from lab.policies.soft_rules import SoftRules

POLICIES = {"soft-rules": SoftRules, "neural": Neural}
RUNS_DIR = LAB_DIR / "runs"
REPLAY_STACK = 1_800_000_000
REPLAY_ITEMS = 8


def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(x)
    ranks[x.argsort()] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    return ranks / (len(x) - 1) - 0.5


def write_json(path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def replay(mkt: E.MarketOnDevice, policy, params, split: splits.Split, cfg: E.EnvConfig, obs_lag: int) -> dict:
    """The champion's trades in the validation episode it did median-best in, at 1.8b, with the price history of
    the items it traded most, for the dashboard's trade replay."""
    card = score.evaluate(mkt, policy, split, cfg, obs_lag, stacks=(REPLAY_STACK,), params=params, track=True)
    res = card.pop("result")
    g = card["growth"][0]
    e = int(np.argsort(g)[len(g) // 2])
    start = card["starts"][e]
    end = start + splits.EPISODE
    trades = [tr for tr in res.trades if tr[0] == e]
    volume = {}
    for _, _, item, _, qty, price in trades:
        volume[item] = volume.get(item, 0) + qty * price
    top = sorted(volume, key=volume.get, reverse=True)[:REPLAY_ITEMS]
    lo, hi = mkt.market.bucket(start), mkt.market.bucket(end)
    items = []
    for n in top:
        step = 6  # 30-minute points
        series = {}
        for key, arr in (("instasell", mkt.market.low), ("instabuy", mkt.market.high)):
            seg = arr[lo:hi, n]
            pts = []
            for k in range(0, len(seg), step):
                chunk = seg[k:k + step]
                ok = ~np.isnan(chunk)
                pts.append(None if not ok.any() else float(chunk[ok].mean()))
            series[key] = pts
        items.append({"name": mkt.market.names[n], "t0": start, "step": step * 300, **series,
                      "fills": [[ts, side, qty, price] for _, ts, item, side, qty, price in trades if item == n]})
    return {"episode_start": start, "growth": g[e], "equity": card["curves"][f"{REPLAY_STACK / 1e9:g}b"][e],
            "checks": [int(t) for t in mkt.features.checks if start <= t < end], "items": items,
            "price_tiers": price_tiers(trades)}


def price_tiers(trades) -> list[int]:
    """Gp spent on buys by log10 price, in quarter-decade bins from 1m (6.0) to 100b (11.0)."""
    bins = [0] * 20
    for _, _, _, side, qty, price in trades:
        if side == E.BUY and price > 0:
            bins[min(19, max(0, int((math.log10(price) - 6) * 4)))] += qty * price
    return bins


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", choices=POLICIES, default="soft-rules")
    parser.add_argument("--gens", type=int, default=300)
    parser.add_argument("--pairs", type=int, default=256, help="mirrored noise pairs per generation (P = 2h + 1)")
    parser.add_argument("--episodes", type=int, default=8, help="training episodes per generation")
    parser.add_argument("--sigma", type=float, default=None, help="noise scale (default: 0.3 soft-rules, 0.05 neural)")
    parser.add_argument("--sigma-decay", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=None, help="Adam step (default: 0.05 soft-rules, 0.01 neural)")
    parser.add_argument("--weight-decay", type=float, default=None, help="L2 pull toward 0 (default: 0.001 "
                                                                          "neural, 0 soft-rules)")
    parser.add_argument("--fill-model", default="share:0.5")
    parser.add_argument("--haircut", type=float, default=0.03)
    parser.add_argument("--obs-lag", type=int, default=300)
    parser.add_argument("--stack-min", type=float, default=1.5e9)
    parser.add_argument("--stack-max", type=float, default=6e9)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0, help="stop after this many evals without a new "
                                                                    "champion (0 = never)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="", help="label for the run")
    parser.add_argument("--item-split", action="store_true",
                        help="train (and pick champions) on half the items; also report validation on the other half")
    args = parser.parse_args()
    sigma = args.sigma or (0.3 if args.policy == "soft-rules" else 0.05)
    lr = args.lr or (0.05 if args.policy == "soft-rules" else 0.01)
    weight_decay = args.weight_decay if args.weight_decay is not None else (0.0 if args.policy == "soft-rules"
                                                                             else 0.001)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    market = M.load()
    features = Features(market, args.obs_lag)
    mkt = E.MarketOnDevice(market, features, dev)
    train_items = 0 if args.item_split else None
    mkt.use_items(train_items)
    split = splits.make(market.t0, market.end)
    starts = splits.episode_starts(split.train, features.checks)
    cfg = E.EnvConfig(fill_model=args.fill_model, haircut=args.haircut)
    family = POLICIES[args.policy]
    policy = family()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}-{args.policy}{'-' + args.name if args.name else ''}"
    (run_dir / "evals").mkdir(parents=True)
    config = {"policy": args.policy, "args": vars(args), "sigma": sigma, "lr": lr, "dim": family.dim,
              "splits": split.to_json(), "market": {"t0": market.t0, "end": market.end, "items": market.N},
              "train_starts": len(starts), "started": int(time.time()),
              "start_values": policy.describe(family.init_params()) if args.policy == "soft-rules" else None}
    write_json(run_dir / "config.json", config)
    print(f"Run {run_dir.name}: {args.policy} ({family.dim} parameters), {2 * args.pairs + 1} variants × "
          f"{args.episodes} episodes per generation")
    for s in (split.train, split.validation):
        print("  " + s.describe())

    print("Scoring the rules baseline (recommended) on validation...", flush=True)
    baseline = score.evaluate(mkt, RulesPolicy(STRATEGIES["recommended"]), split.validation, cfg, args.obs_lag)
    baseline.pop("starts")
    write_json(run_dir / "baseline.json", baseline)
    print(f"  recommended: {baseline['score']:+.4f} (worst {baseline['worst']:+.4f})")

    theta = family.init_params().to(dev)
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    best, best_gen, evals_since_best = -math.inf, None, 0
    log = (run_dir / "gens.jsonl").open("a", encoding="utf-8")
    t_start = time.monotonic()
    for gen in range(args.gens):
        t_gen = time.monotonic()
        windows = rng.choice(len(starts), size=min(args.episodes, len(starts)), replace=False)
        episodes = E.Episodes.from_windows(features, [(starts[w], starts[w] + splits.EPISODE) for w in windows])
        n_ep = episodes.E
        stacks = np.exp(rng.uniform(math.log(args.stack_min), math.log(args.stack_max), n_ep)).astype(np.int64)
        noise = torch.randn(args.pairs, family.dim, device=dev)
        params = torch.cat([theta[None], theta + sigma * noise, theta - sigma * noise])
        P = params.shape[0]
        ep = torch.arange(P * n_ep, device=dev) % n_ep
        cash = torch.tensor(stacks, device=dev)[ep]
        res = E.run(mkt, policy.bind(params, n_ep), episodes, cash, ep, cfg, obs_lag=args.obs_lag)
        growth = res.log_growth.view(P, n_ep).float()
        fitness = score.cvar(growth, 0.25, dim=1)

        ranks = centered_ranks(fitness[1:])
        grad = ((ranks[:args.pairs] - ranks[args.pairs:])[:, None] * noise).sum(0) / (2 * args.pairs * sigma)
        grad -= weight_decay * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        theta = theta + lr * (m / (1 - 0.9 ** (gen + 1))) / ((v / (1 - 0.999 ** (gen + 1))).sqrt() + 1e-8)
        sigma *= args.sigma_decay

        q = torch.quantile(fitness, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev)).tolist()
        row = {"gen": gen, "time": round(time.monotonic() - t_start, 1), "sec": round(time.monotonic() - t_gen, 2),
               "fit_q": q, "fit_max": float(fitness.max()), "center": float(fitness[0]),
               "center_mean": float(growth[0].mean()), "pop_mean": float(growth.mean()), "sigma": sigma,
               "norm": float(theta.norm()), "slot_use": float(res.slot_use.view(P, n_ep)[0].mean()),
               "fills": float((res.buy_fills + res.sell_fills).view(P, n_ep)[0].double().mean())}
        if args.policy == "soft-rules":
            row["values"] = policy.describe(theta)

        if gen % args.eval_every == 0 or gen == args.gens - 1:
            card = score.evaluate(mkt, policy, split.validation, cfg, args.obs_lag, params=theta)
            card.pop("starts")
            row["val"] = {k: card[k] for k in ("score", "worst", "cvar25", "by_stack", "max_drawdown")}
            if args.item_split:  # the same weeks, buying only items it never trained on
                mkt.use_items(1)
                unseen = score.evaluate(mkt, policy, split.validation, cfg, args.obs_lag, params=theta)
                mkt.use_items(train_items)
                row["val_unseen"] = {k: unseen[k] for k in ("score", "worst", "cvar25")}
            write_json(run_dir / "evals" / f"gen_{gen:05d}.json",
                       {"gen": gen, "score": card["score"], "curves": card["curves"][f"{REPLAY_STACK / 1e9:g}b"]})
            if card["score"] > best:
                best, best_gen, evals_since_best = card["score"], gen, 0
                torch.save({"policy": args.policy, "params": theta.cpu(), "gen": gen}, run_dir / "champion.pt")
                rep = replay(mkt, policy, theta, split.validation, cfg, args.obs_lag)
                write_json(run_dir / "champion.json", {"gen": gen, "card": card, "replay": rep,
                                                       "values": policy.describe(theta)})
                with (run_dir / "champions.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"gen": gen, "score": card["score"], "price_tiers": rep["price_tiers"],
                                        "values": policy.describe(theta)}) + "\n")
                row["champion"] = True
            else:
                evals_since_best += 1
            write_json(run_dir / "summary.json", {
                "run": run_dir.name, "policy": args.policy, "name": args.name, "gens": gen + 1,
                "best_val": best, "best_gen": best_gen, "baseline_val": baseline["score"],
                "trained_through": split.train.end, "validation_end": split.validation.end,
                "updated": int(time.time())})
        log.write(json.dumps(row) + "\n")
        log.flush()
        val_txt = f"  val {row['val']['score']:+.4f}{' ★' if row.get('champion') else ''}" if "val" in row else ""
        print(f"gen {gen:4d}  fit p50 {q[2]:+.4f}  max {row['fit_max']:+.4f}  θ {row['center']:+.4f}"
              f"{val_txt}  [{row['sec']:.1f}s]", flush=True)
        if args.patience and evals_since_best >= args.patience:
            print(f"No new champion in {args.patience} evaluations; stopping.")
            break
    print(f"Best validation {best:+.4f} at gen {best_gen} (recommended: {baseline['score']:+.4f}). Run: {run_dir}")


if __name__ == "__main__":
    main()
