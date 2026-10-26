"""Rolling, week-based train / validation / holdout splits.

Weeks start Monday 00:00 UTC (RS3 updates land on Mondays). Every run recomputes the boundaries from the data it
has, so as the sync collects more, everything shifts forward: the holdout is always the newest data and older
holdout weeks become validation, then training. Each run records the boundaries it used.

The holdout is only handed out by open_holdout(), which logs every look to data/lab/holdout_log.jsonl. Nothing
automated selects on it; the dashboard shows how often it has been looked at.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from backtest_flips import DAY, fmt_ts
from lab.market import LAB_DIR

WEEK = 7 * DAY
WARMUP = 7 * DAY  # features look back up to a week (prior daily volume)
EPISODE = 7 * DAY
HOLDOUT_LOG = LAB_DIR / "holdout_log.jsonl"


@dataclass(frozen=True)
class Split:
    name: str
    start: int  # episodes trade only in [start, end)
    end: int

    def describe(self) -> str:
        return f"{self.name}: {fmt_ts(self.start)} → {fmt_ts(self.end)} ({(self.end - self.start) / DAY:.1f} days)"


@dataclass(frozen=True)
class Splits:
    train: Split
    validation: Split
    _holdout: Split

    def to_json(self) -> dict:
        return {"train": asdict(self.train), "validation": asdict(self.validation), "holdout": asdict(self._holdout)}


def week_start(ts: int) -> int:
    """Monday 00:00 UTC on or before ts."""
    d = datetime.fromtimestamp(ts, timezone.utc)
    midnight = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    return midnight - d.weekday() * DAY


def make(t0: int, end: int, validation_weeks: int = 2, holdout_weeks: int = 2) -> Splits:
    """Holdout: the last `holdout_weeks` complete weeks plus the current partial week. Validation: the weeks
    before that. Training: everything earlier, after the feature warm-up."""
    hold_start = week_start(end) - holdout_weeks * WEEK
    val_start = hold_start - validation_weeks * WEEK
    train_start = t0 + WARMUP
    if val_start - train_start < EPISODE:
        raise SystemExit(f"Not enough data for a training episode before validation starts ({fmt_ts(val_start)}).")
    return Splits(Split("train", train_start, val_start), Split("validation", val_start, hold_start),
                  Split("holdout", hold_start, end))


def open_holdout(splits: Splits, reason: str, what: str) -> Split:
    """The holdout split; every call is logged."""
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": int(time.time()), "start": splits._holdout.start, "end": splits._holdout.end,
                            "what": what, "reason": reason}) + "\n")
    return splits._holdout


def holdout_looks() -> int:
    return sum(1 for _ in HOLDOUT_LOG.open(encoding="utf-8")) if HOLDOUT_LOG.exists() else 0


def episode_starts(split: Split, checks: np.ndarray, length: int = EPISODE) -> list[int]:
    """Check times that can start an episode in `split`: each day's first check (the first after the overnight
    gap, 12:00 with the default CHECK_TIMES), for episodes ending by split.end."""
    starts, prev = [], None
    for t in map(int, checks):
        if (prev is None or t - prev >= 6 * 3600) and split.start <= t and t + length <= split.end:
            starts.append(t)
        prev = t
    return starts
