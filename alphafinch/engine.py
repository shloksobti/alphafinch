"""Performance statistics from a strategy's daily returns (computed in the main process)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 252
N_ERAS = 4


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
    appraisal: float = 0.0   # alpha / residual volatility, annualised
    alpha_t: float = 0.0     # t-statistic of alpha
    yearly: dict = field(default_factory=dict)
    n_days: int = 0
    eras: list = field(default_factory=list)   # per-era {start, end, sharpe, appraisal, score}

    @property
    def t(self) -> float:
        return self.sharpe * math.sqrt(self.n_days / ANN)


def alpha_stats(r: pd.Series, mkt: pd.Series):
    """OLS of strategy returns on the equal-weight market: (beta, annual alpha, appraisal, alpha t)."""
    x, y = mkt.values.astype(float), r.values.astype(float)
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


def _sharpe(r):
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(ANN)) if sd > 0 else 0.0


def stats(r: pd.Series, turnover: pd.Series, gross: pd.Series, mkt: pd.Series, eras: bool = True) -> Stats:
    r, mkt = r.iloc[1:], mkt.reindex(r.index).fillna(0).iloc[1:]
    turnover, gross = turnover.iloc[1:], gross.iloc[1:]
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    years = max(len(r) / ANN, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if len(eq) and eq.iloc[-1] > 0 else -1.0
    beta, alpha, appraisal, alpha_t = alpha_stats(r, mkt)
    yearly = {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}
    era_list = []
    if eras and len(r) > N_ERAS * 126:
        bounds = np.linspace(0, len(r), N_ERAS + 1).astype(int)
        for a_, b_ in zip(bounds[:-1], bounds[1:]):
            cr, cm = r.iloc[a_:b_], mkt.iloc[a_:b_]
            sh, ap = _sharpe(cr), alpha_stats(cr, cm)[2]
            era_list.append({"start": str(cr.index[0].date()), "end": str(cr.index[-1].date()),
                             "sharpe": sh, "appraisal": ap, "score": 0.5 * sh + 0.5 * ap})
    return Stats(_sharpe(r), cagr, float(r.std() * math.sqrt(ANN)), dd, float(turnover.mean() * ANN),
                 float(gross.mean()), beta, alpha, appraisal, alpha_t, yearly, len(r), era_list)
