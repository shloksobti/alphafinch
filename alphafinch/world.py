"""The world exam: test frozen strategies on stock markets they have never seen.

One market's three sealed years rarely hold enough evidence to prove a realistic edge. Many
markets over many years can. The world exam runs each strategy, unchanged, on every market over
the same exam window and pools the evidence:

  1. In each market, the strategy's daily return is regressed on that market's equal-weight
     benchmark over the window. Its active return = return - beta x benchmark.
  2. Each calendar day, active returns are averaged over the markets open that day.
  3. The pooled t-statistic of that daily series uses Newey-West standard errors, which allow for
     autocorrelation and for markets moving together (they share days).
  4. For robustness the verdict uses the LOWER of that t and the Stouffer combination of the
     per-market t-statistics (sum / sqrt(markets)). Each can be fooled in a different way
     (Newey-West by slow common crashes, Stouffer by markets moving together); a strategy has to
     clear both. Calibrated with placebo strategies in scripts/world_placebo.py.

The bar is t^{-1}(alpha / K), K = strategies tested (plus any earlier looks at these markets
and years). Strategies that cannot run in a market (e.g. they need US-only fundamentals) are
excluded from that market and the exclusion is reported.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st

from . import data
from .engine import ANN, alpha_stats
from .fitness import verdict_grade, years_needed
from .lab import Lab
from .sandbox import StrategyError
from .universe import WORLD


def newey_west_t(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Mean of x and its t-statistic with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return float(x.mean()) if n else 0.0, 0.0
    d = x - x.mean()
    lrv = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    se = math.sqrt(max(lrv, 1e-30) / n)
    return float(x.mean()), float(x.mean() / se)


def market_run(lab: Lab, code: str, start: pd.Timestamp) -> dict:
    res = lab.run(code, "full", check_leaks=True)
    m = res.returns.index >= start
    r, mkt = res.returns[m], lab.mkt["full"][m]
    beta, alpha, appraisal, t = alpha_stats(r, mkt)
    return {"active": r - beta * mkt, "alpha": alpha, "alpha_t": t, "beta": beta, "appraisal": appraisal,
            "days": int(m.sum()), "return": float((1 + r).prod() ** (ANN / max(m.sum(), 1)) - 1),
            "market_return": float((1 + mkt).prod() ** (ANN / max(m.sum(), 1)) - 1)}


def exam(strategies: dict[str, str], markets=None, start="2017-10-02", end=None, alpha=0.05,
         prior_looks: int = 0, lags: int = 10, workers: int = 4, progress=None, cost_bps: float = 5.0) -> list[dict]:
    """Run every strategy on every market; return one pooled verdict per strategy."""
    """`markets` is a list of market names, or a dict {name: Panel} (offline use, tests)."""
    markets = markets or list(WORLD)
    panels = markets if isinstance(markets, dict) else {m: None for m in markets}
    start = pd.Timestamp(start)
    K = len(strategies) + prior_looks
    per = {name: {} for name in strategies}
    for mk, panel in panels.items():
        panel = panel if panel is not None else data.load(mk)
        if end:
            panel = panel.before(pd.Timestamp(end) + pd.Timedelta(days=1))
        if (panel.index >= start).sum() < 60 or (panel.index < start).sum() < 260:
            for name in strategies:
                per[name][mk] = {"error": "not enough data around the exam window"}
            continue
        with Lab(panel, start, workers=workers, timeout=900, cost_bps=cost_bps) as lab:
            for name, code in strategies.items():
                if progress:
                    progress(name, mk)
                try:
                    per[name][mk] = market_run(lab, code, start)
                except StrategyError as e:
                    per[name][mk] = {"error": str(e).splitlines()[0][:120]}
    out = []
    for name in strategies:
        ok = {mk: v for mk, v in per[name].items() if "error" not in v}
        if not ok:
            out.append({"name": name, "markets": per[name], "grade": "NOT RUN", "K": K})
            continue
        pooled = pd.concat({mk: v["active"] for mk, v in ok.items()}, axis=1).mean(axis=1, skipna=True).dropna()
        mean, t_nw = newey_west_t(pooled.values, lags)
        t_st = sum(v["alpha_t"] for v in ok.values()) / math.sqrt(len(ok))
        t = min(t_nw, t_st)
        if pooled.std() * math.sqrt(ANN) < 0.01:          # no active bet anywhere: nothing to test
            mean, t, t_nw, t_st = 0.0, 0.0, 0.0, 0.0
        bar = float(st.t.isf(alpha / K, max(len(pooled) - 1, 1)))
        sd = float(pooled.std())
        appraisal = mean / sd * math.sqrt(ANN) if sd > 0 else 0.0
        out.append({"name": name, "markets": per[name], "pooled_alpha": mean * ANN, "pooled_t": t,
                    "t_newey_west": t_nw, "t_stouffer": t_st,
                    "appraisal": appraisal, "bar": bar, "K": K, "days": len(pooled),
                    "positive_markets": sum(v["alpha"] > 0 for v in ok.values()), "n_markets": len(ok),
                    "grade": verdict_grade(t, bar), "years_needed": years_needed(appraisal, bar),
                    "pooled": pooled})
    return out
