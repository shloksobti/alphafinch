"""Fitness (training period only) and the sealed exam (holdout period, PASS/FAIL only).

Fitness rewards risk-adjusted return that is *consistent* across years and penalises heavy
trading and bloated code. The holdout is never used for fitness or shown to the AI.

The exam: a champion's holdout ALPHA t-statistic (return not explained by its exposure to the
equal-weight market) must clear a bar that accounts for every exam
attempt in the run (t > t^{-1}(alpha / K) for an exam budget K). Because each attempt reveals
only one bit, this bar is valid no matter how adaptively the population evolved
("deflate by bits, not trials").
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .engine import Stats, alpha_stats, portfolio_returns, normalise, stats as compute_stats
from . import sandbox


def fitness(s: Stats, code: str) -> float:
    if s.n_days < 252:
        return -9.0
    yearly = np.array(list(s.yearly.values()))
    hit = float((yearly > 0).mean()) if len(yearly) else 0.0
    consistency = 0.5 * (hit - 0.5)                         # +/-0.25 for always up / always down
    trading = 0.02 * max(0.0, s.turnover - 12.0)            # > ~monthly full rebalancing costs extra
    bloat = 0.0005 * max(0, len(code) - 1500)               # Occam: long code must earn its keep
    lev = 0.5 * max(0.0, 0.2 - s.exposure)                  # mostly-cash strategies are not alpha
    # half Sharpe (is it a good investment?) + half appraisal ratio (does it beat the market?)
    return float(0.5 * s.sharpe + 0.5 * s.appraisal + consistency - trading - bloat - lev)


class SealedExam:
    """Holds the holdout. Only answers PASS/FAIL, at most `budget` times."""

    def __init__(self, prices_full: pd.DataFrame, holdout_start, budget: int = 10, alpha: float = 0.05):
        self._px = prices_full
        self._start = pd.Timestamp(holdout_start)
        self.budget, self.alpha = budget, alpha
        n_hold = int((prices_full.index >= self._start).sum())
        self.bar = float(sstats.t.isf(alpha / budget, df=max(n_hold - 1, 1)))
        self.attempts: list[dict] = []

    @property
    def left(self) -> int:
        return self.budget - len(self.attempts)

    def sit(self, sid: str, code: str) -> str:
        if self.left <= 0:
            return "EXHAUSTED"
        try:
            w = sandbox.run(code, self._px)           # causal code: run on full history, score holdout rows
            r, to, held = portfolio_returns(w, self._px)
            mkt = self._px.pct_change().mean(axis=1).fillna(0)
            mask = r.index >= self._start
            t = alpha_stats(r[mask], mkt[mask])[3]
            verdict = "PASS" if t > self.bar else "FAIL"
        except Exception:
            verdict, t = "FAIL", float("nan")
        self.attempts.append({"id": sid, "verdict": verdict, "_t": t})
        return verdict

    def reveal(self, sid: str, code: str) -> dict:
        """Full holdout statistics. Call only AFTER evolution has finished."""
        w = sandbox.run(code, self._px)
        r, to, held = portfolio_returns(w, self._px)
        mask = r.index >= self._start
        s = compute_stats(r[mask], to[mask], held[mask], self._px[mask])
        return {"sharpe": s.sharpe, "cagr": s.cagr, "max_dd": s.max_dd, "t": s.t, "alpha": s.alpha,
                "alpha_t": s.alpha_t, "beta": s.beta,
                "equity": (1 + r[mask]).cumprod()}
