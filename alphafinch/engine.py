"""Backtest engine: no look-ahead by construction, realistic costs, and a leak detector.

Convention: `strategy(prices)` returns target weights W (dates x assets) using information
up to and including each row's close. The engine holds W[t] from close t to close t+1, so
W is lagged one day before it meets returns. Gross exposure is capped at 1 (|w| sums to <= 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import sandbox

COST_BPS = 5.0          # per unit of turnover, one way
ANN = 252


def normalise(w: pd.DataFrame) -> pd.DataFrame:
    gross = w.abs().sum(axis=1)
    scale = np.where(gross > 1, 1 / gross.replace(0, 1), 1.0)
    return w.mul(scale, axis=0)


def portfolio_returns(w: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = COST_BPS):
    rets = prices.pct_change().fillna(0.0)
    held = normalise(w).shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    r = (held * rets).sum(axis=1) - turnover * cost_bps / 1e4
    return r, turnover, held


@dataclass
class Stats:
    sharpe: float
    cagr: float
    vol: float
    max_dd: float
    turnover: float          # average annual one-way turnover
    exposure: float          # average gross exposure
    beta: float              # beta to the equal-weight market
    alpha: float = 0.0       # annualised return not explained by market exposure
    appraisal: float = 0.0   # alpha / residual volatility (annualised): risk-adjusted alpha
    alpha_t: float = 0.0     # t-statistic of alpha
    yearly: dict = field(default_factory=dict)
    n_days: int = 0

    @property
    def t(self) -> float:
        return self.sharpe * math.sqrt(self.n_days / ANN)


def alpha_stats(r: pd.Series, mkt: pd.Series):
    """OLS of strategy returns on the equal-weight market: (beta, annual alpha, appraisal, alpha t)."""
    x, y = mkt.values, r.values
    if len(y) < 30 or x.var() == 0:
        return 0.0, 0.0, 0.0, 0.0
    beta = float(np.cov(y, x)[0, 1] / x.var())
    a = float(y.mean() - beta * x.mean())
    resid = y - a - beta * x
    sd = float(resid.std(ddof=2))
    if sd * math.sqrt(ANN) < 0.01:          # < 1%/yr tracking error: no active bet, no alpha to rate
        return beta, a * ANN, 0.0, 0.0
    se = sd * math.sqrt(1 / len(y) + x.mean() ** 2 / (((x - x.mean()) ** 2).sum()))
    return beta, a * ANN, a / sd * math.sqrt(ANN), a / se


def stats(r: pd.Series, turnover: pd.Series, held: pd.DataFrame, prices: pd.DataFrame) -> Stats:
    r = r.iloc[1:]
    sd = r.std()
    sharpe = float(r.mean() / sd * math.sqrt(ANN)) if sd > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    years = max(len(r) / ANN, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if len(eq) and eq.iloc[-1] > 0 else -1.0
    mkt = prices.pct_change().mean(axis=1).reindex(r.index).fillna(0)
    beta, alpha, appraisal, alpha_t = alpha_stats(r, mkt)
    yearly = {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}
    return Stats(sharpe, cagr, float(sd * math.sqrt(ANN)), dd, float(turnover.mean() * ANN),
                 float(held.abs().sum(axis=1).mean()), beta, alpha, appraisal, alpha_t, yearly, len(r))


CUTS = (0.55, 0.85)


def leak_check(prices: pd.DataFrame, w_full: pd.DataFrame, w_cuts: list) -> None:
    """Weights computed on truncated histories must equal the full-history weights on every
    row before the cut. If they differ, the strategy used future data (shift(-k), centred
    windows, full-sample normalisation, ...)."""
    n = len(prices)
    for frac, w_cut in zip(CUTS, w_cuts):
        cut = int(n * frac)
        a = normalise(w_full.iloc[:cut]).values
        b = normalise(w_cut.iloc[:cut]).values
        if not np.allclose(a, b, atol=1e-6, equal_nan=True):
            bad = np.argwhere(~np.isclose(a, b, atol=1e-6))[0][0]
            raise sandbox.StrategyError(
                f"look-ahead detected: weights on {prices.index[bad].date()} change when later data is removed")


def backtest(code: str, prices: pd.DataFrame, check_leaks: bool = True):
    frames = [prices] + ([prices.iloc[:int(len(prices) * f)] for f in CUTS] if check_leaks else [])
    ws = sandbox.run_many(code, frames)
    if check_leaks:
        leak_check(prices, ws[0], ws[1:])
    r, to, held = portfolio_returns(ws[0], prices)
    return r, stats(r, to, held, prices)
