"""Fitness (training period only), the sealed exam, and the graded verdict.

Fitness rewards an edge that holds up in EVERY era of the training period, not one lucky stretch:
each of four eras is scored (half Sharpe, half appraisal ratio = alpha per unit of active risk),
and fitness blends the median era with the worst era. Penalties for heavy trading, bloated code
and mostly-cash portfolios. The holdout is never used for fitness or shown to the AI.

The sealed exam answers PASS/FAIL only, at most `budget` times. To pass, a strategy's holdout
alpha t-statistic must exceed t^{-1}(alpha / budget). Because each attempt reveals one bit, this
bar is valid however adaptively the population evolved ("deflate by bits, not trials").
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .engine import Stats, alpha_stats, stats as compute_stats


def fitness(s: Stats, code: str) -> float:
    if s.n_days < 252:
        return -9.0
    if s.eras:
        scores = np.array([e["score"] for e in s.eras])
        core = 0.5 * float(np.median(scores)) + 0.5 * float(scores.min())
    else:
        core = 0.5 * s.sharpe + 0.5 * s.appraisal
    trading = 0.02 * max(0.0, s.turnover - 12.0)            # > ~monthly full rebalancing costs extra
    bloat = 0.0006 * max(0, len(code) - 1500)               # Occam: long code must earn its keep
    lev = 0.5 * max(0.0, 0.2 - s.exposure)                  # mostly-cash strategies are not alpha
    return float(core - trading - bloat - lev)


def verdict_grade(alpha_t: float, bar: float) -> str:
    """Graded reading of a holdout result (shown only after a run has finished)."""
    if alpha_t > bar:
        return "PASS"
    if alpha_t > 1.0:
        return "PROMISING"
    return "FAIL"


def years_needed(appraisal: float, bar: float) -> float | None:
    """Years of out-of-sample data needed for an edge of this size to clear the bar."""
    if appraisal <= 0.05:
        return None
    return (bar / appraisal) ** 2


class SealedExam:
    """Holds the holdout. Only answers PASS/FAIL, at most `budget` times."""

    def __init__(self, lab, budget: int = 10, alpha: float = 0.05):
        self.lab = lab
        self._start = lab.hold
        self.budget, self.alpha = budget, alpha
        n_hold = int((lab.full.index >= self._start).sum())
        self.bar = float(sstats.t.isf(alpha / budget, df=max(n_hold - 1, 1)))
        self.attempts: list[dict] = []

    @property
    def left(self) -> int:
        return self.budget - len(self.attempts)

    def _holdout(self, code: str):
        res = self.lab.run(code, split="full", check_leaks=False)   # code is causal: run on all history
        mask = res.returns.index >= self._start
        return res, mask

    def sit(self, sid: str, code: str) -> str:
        if self.left <= 0:
            return "EXHAUSTED"
        try:
            res, mask = self._holdout(code)
            t = alpha_stats(res.returns[mask], self.lab.mkt["full"][mask])[3]
            verdict = "PASS" if t > self.bar else "FAIL"
        except Exception:
            verdict, t = "FAIL", float("nan")
        self.attempts.append({"id": sid, "verdict": verdict, "_t": t})
        return verdict

    def reveal(self, sid: str, code: str) -> dict:
        """Full holdout statistics. Call only AFTER evolution has finished."""
        res, mask = self._holdout(code)
        s = compute_stats(res.returns[mask], res.turnover[mask], res.gross[mask], self.lab.mkt["full"][mask],
                          eras=False)
        mkt = self.lab.mkt["full"][mask]
        mkt_eq = (1 + mkt).cumprod()
        return {"sharpe": s.sharpe, "cagr": s.cagr, "max_dd": s.max_dd, "t": s.t, "alpha": s.alpha,
                "alpha_t": s.alpha_t, "appraisal": s.appraisal, "beta": s.beta,
                "grade": verdict_grade(s.alpha_t, self.bar), "years_needed": years_needed(s.appraisal, self.bar),
                "equity": (1 + res.returns[mask]).cumprod(), "market_equity": mkt_eq,
                "market_cagr": float(mkt_eq.iloc[-1] ** (252 / max(len(mkt_eq) - 1, 1)) - 1)}
