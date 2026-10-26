"""Splits: chronological, week-aligned, and training never reaches validation."""

import json

from backtest_flips import DAY
from lab import splits as S


def test_splits_are_ordered_and_week_aligned(market):
    sp = S.make(market.t0, market.end)
    assert sp.train.start < sp.train.end == sp.validation.start < sp.validation.end == sp._holdout.start
    assert sp._holdout.end == market.end
    for t in (sp.validation.start, sp._holdout.start):
        assert S.week_start(t) == t  # Monday 00:00 UTC
    assert sp._holdout.end - sp._holdout.start >= 14 * DAY


def test_training_episodes_end_before_validation(market, features):
    sp = S.make(market.t0, market.end)
    starts = S.episode_starts(sp.train, features.checks)
    assert starts and all(sp.train.start <= s and s + S.EPISODE <= sp.validation.start for s in starts)


def test_holdout_access_is_logged(market, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HOLDOUT_LOG", tmp_path / "log.jsonl")
    sp = S.make(market.t0, market.end)
    assert S.holdout_looks() == 0
    h = S.open_holdout(sp, reason="test", what="nothing")
    assert h == sp._holdout and S.holdout_looks() == 1
    assert json.loads((tmp_path / "log.jsonl").read_text())["reason"] == "test"
