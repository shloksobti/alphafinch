"""`tk`: a small quant toolkit available inside every strategy (no import needed).

Every function is causal: its value on row t depends only on rows up to t, so strategies built
from it pass the look-ahead detector. Cross-sectional functions work row by row (one date at a
time, across assets); time-series functions use trailing windows.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

ANN = 252


# ---------------------------------------------------------------- cross-sectional (per date)
def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across assets on each date, centred on 0 (from -0.5 to +0.5)."""
    r = x.rank(axis=1)
    return (r - 1).div((r.count(axis=1) - 1).replace(0, np.nan), axis=0) - 0.5


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Standardise across assets on each date."""
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def winsorize(x: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Clip each date's cross-sectional z-scores at +-z (tames outliers)."""
    m, s = x.mean(axis=1), x.std(axis=1)
    return x.clip(m - z * s, m + z * s, axis=0)


def demean(x: pd.DataFrame) -> pd.DataFrame:
    """Subtract each date's cross-sectional mean (a dollar-neutral tilt)."""
    return x.sub(x.mean(axis=1), axis=0)


def neutralize(x: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Subtract each group's mean on each date, e.g. neutralize(score, data.sector) removes
    sector bets so only within-sector differences remain."""
    if groups is None:
        return demean(x)
    g = groups.reindex(x.columns)
    return x - x.T.groupby(g).transform("mean").T


# ---------------------------------------------------------------- scores -> weights
def long_short(score: pd.DataFrame, q: float = 0.2) -> pd.DataFrame:
    """Long the top `q` fraction, short the bottom `q`, equal weight, gross exposure 1."""
    r = score.rank(axis=1, pct=True)
    long, short = (r > 1 - q).astype(float), (r <= q).astype(float)
    long = long.div(long.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    short = short.div(short.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    return (long - short).fillna(0.0)


def long_only(score: pd.DataFrame, q: float = 0.2) -> pd.DataFrame:
    """Equal weight in the top `q` fraction of assets by score."""
    w = (score.rank(axis=1, pct=True) > 1 - q).astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def proportional(score: pd.DataFrame) -> pd.DataFrame:
    """Weights proportional to a (signed) score, gross exposure 1."""
    return score.div(score.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def cap(w: pd.DataFrame, max_weight: float = 0.05) -> pd.DataFrame:
    """Limit any single position to +-max_weight."""
    return w.clip(-max_weight, max_weight)


def rebalance(w: pd.DataFrame, every: str = "M") -> pd.DataFrame:
    """Only trade on the first trading day of each period ('W', 'M' or 'Q'); hold in between."""
    key = pd.Series(w.index.to_period(every).astype(str), index=w.index)
    first = (key != key.shift(1)).values
    out = w.copy()
    out.loc[~first] = np.nan
    return out.ffill().fillna(0.0)


def vol_target(w: pd.DataFrame, prices: pd.DataFrame, target: float = 0.10, window: int = 63,
               max_leverage: float = 1.0) -> pd.DataFrame:
    """Scale the whole portfolio so its trailing volatility is near `target` (annualised)."""
    port = (w.shift(1) * prices.pct_change()).sum(axis=1)
    vol = port.rolling(window, min_periods=window // 2).std() * np.sqrt(ANN)
    scale = (target / vol.replace(0, np.nan)).clip(upper=max_leverage).fillna(0.0)
    return w.mul(scale, axis=0)


# ---------------------------------------------------------------- time series (trailing)
def returns(prices: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return prices.pct_change(n)


def market(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight market daily return."""
    return prices.pct_change().mean(axis=1)


def rolling_beta(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Each asset's trailing beta to the equal-weight market."""
    r, m = prices.pct_change(), market(prices)
    cov = r.rolling(window, min_periods=window // 2).cov(m)
    return cov.div(m.rolling(window, min_periods=window // 2).var(), axis=0)


def residual(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Daily returns with each asset's trailing market beta removed (idiosyncratic returns)."""
    r, m = prices.pct_change(), market(prices)
    return r - rolling_beta(prices, window).shift(1).mul(m, axis=0)


def ts_zscore(x: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """How unusual today's value is versus the asset's own trailing history."""
    m = x.rolling(window, min_periods=window // 2)
    return (x - m.mean()) / m.std().replace(0, np.nan)


def smooth(x, halflife: float = 5):
    """Exponentially weighted average (reduces turnover of a noisy signal)."""
    return x.ewm(halflife=halflife, min_periods=1).mean()


def drawdown(prices):
    """Distance below the running peak (0 at a high, negative below it)."""
    return prices / prices.cummax() - 1


TK = SimpleNamespace(**{f.__name__: f for f in [
    rank, zscore, winsorize, demean, neutralize, long_short, long_only, proportional, cap, rebalance,
    vol_target, returns, market, rolling_beta, residual, ts_zscore, smooth, drawdown]})

DOC = """- `tk`, a toolkit of causal helpers (no import needed):
  cross-section per date: tk.rank(x), tk.zscore(x), tk.winsorize(x, z=3), tk.demean(x), tk.neutralize(x, data.sector)
  scores to weights: tk.long_short(score, q=0.2) (dollar-neutral, gross 1), tk.long_only(score, q=0.2), tk.proportional(score), tk.cap(w, 0.05)
  portfolio: tk.rebalance(w, every='M'|'W'|'Q'), tk.vol_target(w, prices, target=0.10, window=63)
  time series: tk.returns(prices, n), tk.market(prices), tk.rolling_beta(prices, 252), tk.residual(prices, 252) (beta-removed daily returns), tk.ts_zscore(x, 252), tk.smooth(x, halflife), tk.drawdown(prices)"""
