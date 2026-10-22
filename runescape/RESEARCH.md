# GE flipping research notes

Status as of 2026-10-21; quote-age fallback, cap floor and buy-limit overrides added 2026-10-22. Data: ~31 days of 5-minute GE history (2026-09-19 → 2026-10-21) out of the ~90 the
wiki keeps; the rest is still downloading into `data/5m/`. Everything below should be re-checked on the full
window (see *Re-running*).

## Recommended strategy (`--strategy recommended`, the default)

Unchanged from the original spec: 1.5b budget, 8 slots, buy at instasell + outbid, sell at instabuy − outbid,
re-plan every check (12:00, 14:00, 16:30, 19:30, 21:00, 22:00, 23:00, 01:00, 02:00 CDT), knapsack optimizer
over the top 20 by `min(4h volume, buy limit) × after-tax margin`, 2% tax, dump at instasell at the end.

**Item selection** — an item is only eligible at a check if all of these hold:

| Rule | Why (what it catches) |
|---|---|
| Both quotes (last instasell and instabuy trades) are **≤ 1 hour old**; cash and slots left over after those picks go to items **≤ 2h**, then **≤ 4h** old | Stale quotes promise margins that no longer exist; the strongest single predictor of a flip under-delivering. But 1h alone often leaves cash idle, and the fallback (2026-10-22, see *Evidence* 4) beat strict 1h at every stack size. |
| **≥ 6** five-minute buckets with trades on **each** side in the last 24h | Thinly traded items have unreliable prices and are easy to move. |
| Both quotes within **10% of their 3-day median** price | Pumps, dumps and fat-finger trades. This is the main market-manipulation guard (e.g. Pink pumpkin pumped from ~35m to 100–150m on Oct 12; the unfiltered strategy bought at 100m and sold at 34.5m). |
| Last 24h volume **≤ 2×** the prior days' average daily volume | Volume bursts that accompany pumps, and that also inflate the 4h-volume estimate the optimizer sizes orders with. |

**Execution**

| Rule | Why |
|---|---|
| **Outbid by 100k** on both buy and sell offers | 50k–500k all beat 1gp in every tuning period; 100k had the best worst period. ≥ 1m starts eating the margin. |
| **No single item over 25% of equity** (stock + open buy offer), **but always at least 1 unit** | Concentration risk: uncapped, one item (Malachi's claw, which had no published buy limit) built up 21 units and lost 352m. 20–50% all work; 25% is mid-plateau. The 1-unit floor matters below ~300m, where 25% is less than the cheapest eligible item (75m) and the cap alone blocks every trade. |
| **Buy limit 2** for Malachi's claw and tooth (`BUY_LIMIT_OVERRIDES`) | Released 2026-10-12; the wiki mapping has no limit for them, which the code treats as unlimited. |

Rejected after testing: ranking on last-3-trades margin (hurt), pricing offers off recent averages (worst
variant), "latest margin > 2× recent margin" outlier rule and "recent margin must be positive" (no effect once
the rules above are in place), conservative min(latest, recent) ranking (mixed: helped tuning, hurt held-out),
minimum 3% margin (the two methods disagree; see open questions), and forcing more offers to fill all 8 slots
(`--stage slots` on 45 days: the optimizer's own pick beat "1 unit per item if slots are left empty", "always 1
unit per item" and "use every slot" in every period, including the held-out one).

## Evidence

Two independent methods, both with a held-out period that played no part in choosing settings.

1. **Candidate-level study** (`study_selection.py`): every eligible item at every check (~20k) is flipped as one
   unit under the strategy's rules, independent of budget/slots. Measures what share of the *quoted* after-tax
   margin is actually *realized*, for the top-5 / top-20 candidates the optimizer would pick from.
   - Unfiltered, the top-5 picks realize **≈ −2% (tuning) / −1% (held out)** of their quoted margin — the
     optimizer chases the biggest quoted margins, which are disproportionately stale or manipulated.
   - With the selection rules: **≈ 40–46% / 35–40%**, and expected return per gp invested goes from ~0.2% to
     ~0.7–1.0% per candidate in both periods.
   - Threshold sensitivity (one rule varied, others fixed): quote age ≤ 1h best in both periods; drift 5–10% a
     plateau, ≥ 15% clearly worse; volume spike 1.5–2× fine, off is worse.
2. **Portfolio folds** (`tune_selection.py`): the window split into 3 periods of ~10 days, each from a fresh
   1.5b; settings ranked by their *worst* tuning period, the last period held out.

   | Config (latest run) | Period A | Period B | Period C (held out) | Max drawdown |
   |---|---|---|---|---|
   | baseline | −96m | 303m | 220m | 377m |
   | selected (rules above, 1gp)† | 662m | 561m | 309m | 114m |
   | + outbid 100k | 921m | 695m | 501m | 107m |
   | + item cap 25% (now `recommended-1h`) | 886m | 829m | 645m | 95m |

   † from the outbid run, on ~29.5 days of cache (the others are ~31 days), so its periods are slightly shifted.

   Ablation (each selection rule removed in turn) lowered the worst period every time except the 3% margin rule.
3. **Whole window** (31 days, compounding): baseline +384m (26%); selected +2.03b (135%); recommended
   (now `recommended-1h`) **+2.72b (181%)**, max drawdown 100m, no losing day, both halves ≈ +1.34b.
4. **Quote-age fallback and cap floor by stack size** (2026-10-22, 69 days in 3 folds of ~23 days, each from a
   fresh stack; ad-hoc script, not in the repo). Mean profit per fold, worst fold in brackets, fill model
   `floor` (1 unit per 5m bucket); `average` ranked them the same way:

   | Config | 250m | 500m | 1b | 2b |
   |---|---|---|---|---|
   | 1h, cap 25% (`recommended-1h`) | 0 (0) | 684m (579m) | 1.27b (1.09b) | 1.65b (1.47b) |
   | 1h, cap 25% but ≥ 1 unit | 562m (432m) | 790m (745m) | 1.27b (1.09b) | 1.65b (1.47b) |
   | 1h, cap 50%, ≥ 1 unit | 578m (464m) | 851m (692m) | 1.22b (1.04b) | 1.52b (1.23b) |
   | 1h, no cap | 423m (334m) | 726m (589m) | 1.20b (1.04b) | 1.44b (1.15b) |
   | 2h, cap 25%, ≥ 1 unit | 604m (535m) | 1.08b (656m) | 1.46b (1.05b) | 1.95b (1.43b) |
   | 4h, cap 25%, ≥ 1 unit | 545m (447m) | 866m (627m) | 1.25b (956m) | 1.57b (1.17b) |
   | 24h, cap 25%, ≥ 1 unit | 406m (200m) | 696m (417m) | 1.17b (889m) | 1.39b (1.05b) |
   | 1h, leftovers to 2h then 4h, cap 25%, ≥ 1 unit (**recommended**) | 601m (471m) | 883m (703m) | 1.37b (1.18b) | 1.74b (1.47b) |

   The 1-unit floor is free at ≥ 1b and essential below. Looser caps help a little at small stacks and hurt
   at large ones. A flat 2h limit has the highest mean but a worse worst fold than 1h at 1b and 2b; 4h and 24h
   are worse than 2h. The fallback beat strict 1h in 11 of 12 folds and cut idle cash to 3–8%, so it's the
   recommended rule. Recheck 2h vs fallback on more data.

Reports: `data/runs/study-*.md`, `data/runs/tuning-*.md`, `data/runs/<stamp>/compare.md` + per-strategy journals.

## Read this before trusting the numbers

The *relative* ranking of strategies is the defensible part. The *absolute* returns are an upper bound:

- **Fill model is optimistic.** A buy fills whenever a 5m bucket's *average* instasell is at or below our price,
  up to that bucket's volume, as if we were always first in the queue and nobody else was flipping. Real fills
  will be fewer, and our own offers would move these thin markets.
- **Bucket averages, not trades.** A bucket average can sit below our price even if no single trade did, and
  vice versa.
- **31 days, ~700 fills.** Folds are ~10 days each; a few large trades still move them.
- **Bonds are ~30% of profit.** Very liquid, so plausible, but check tax treatment (below) and run without them.

## Open questions for later research

1. **Validate the fill model against reality.** Paper-trade or log real GE offers with the recommended rules and
   compare fill rates / realized margins with the backtest's per-flip numbers (realized ÷ quoted ≈ 35–45%).
   This is the biggest unknown.
2. **Re-run everything on the full ~89 days** once the download finishes; confirm the thresholds still sit on
   plateaus and the held-out period still agrees.
3. **Minimum margin %.** The candidate study says a 3% floor helps the top-20 pool; the portfolio ablation says
   removing it helps slightly. Resolve on more data.
4. **Trade-count threshold** (6 vs 12 buckets) is noisy between periods.
5. **Outbid as % of price** (e.g. 0.05–0.1%) instead of flat 100k, since items range 75m–500m. The model can't
   reward queue priority, the main real benefit of outbidding, so real gains may differ from modelled ones.
6. **Unknown buy limits.** Malachi's claw and tooth are now hard-coded to 2 (`BUY_LIMIT_OVERRIDES`). Future
   items released without a documented limit will again be treated as unlimited; consider a conservative
   default (e.g. 2–10) instead of volume-only sizing. New items also tend to slide in price after release,
   which the drift filter doesn't catch; consider skipping items tradeable for under ~2–4 weeks.
7. **GE tax rules.** Confirm the 2% rate, any per-trade cap, and whether bonds (or other items) are exempt.
8. **Overnight gap.** Buy offers sit 10h between the 02:00 and 12:00 checks. Test cancelling buys at 02:00, or a
   tighter overnight filter.
9. **Sell-side risk.** Sells are repriced to instabuy − outbid each check with no stop-loss or max holding time;
   test a max hold (e.g. 24–48h) and whether chasing the price down helps.
10. **Expected-profit ranking.** Instead of hard filters, rank by quoted margin × historical realization rate for
    similar candidates (by quote age, drift, liquidity). Needs care to avoid overfitting.
11. **More manipulation signals.** Deviation from the official GE guide price (historical guide prices would be
    needed, e.g. from the Weird Gloop exchange API), spread volatility, repeated same-size trades.
12. **Check-time schedule.** Compare against hourly checks to see what the fixed CDT schedule costs.
13. **Buy-limit semantics.** The model uses a rolling 4h window; RS3 counts 4h from the first purchase.

## Re-running

```
uv run backtest_flips.py                    # finish/update the 5m download, then backtest `recommended`
uv run backtest_flips.py --offline --compare
uv run study_selection.py                   # candidate-level study + threshold sensitivity
uv run tune_selection.py --stage final      # selection rules, neighbours and ablations by period
uv run tune_selection.py --stage outbid     # outbid sweep on top of `selected`
uv run tune_selection.py --stage cap        # per-item exposure cap
```
