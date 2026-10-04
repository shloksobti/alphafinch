import numpy as np, pandas as pd

def _f0726(prices, data):
    """Hedged Calm Residual: hold calm, uptrending, low-beta stocks that also show strong idiosyncratic (beta-stripped) momentum, with a hedge sized to realised book beta. Hypothesis: market-driven trends fade in weak eras while stock-specific persistent drift is under-reacted to, so ranking on residual momentum keeps alpha when pure market-beta tilts stop working."""
    VOL_WIN, MOM_WIN, SKIP, TOP_FRAC = 111, 250, 16, 0.4
    VIX_LO, VIX_HI, MIN_EXPO, REBAL = 15.6549, 35.0, 0.25, 10
    BETA_WIN, HEDGE, RES_W = 250, 0.7, 1.0
    ret = prices.pct_change()
    mkt = ret.mean(axis=1)
    mp = BETA_WIN // 2
    vol = ret.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    mom = prices.shift(SKIP) / prices.shift(MOM_WIN) - 1.0
    cov = ret.rolling(BETA_WIN, min_periods=mp).cov(mkt)
    beta = cov.div(mkt.rolling(BETA_WIN, min_periods=mp).var(), axis=0)
    # residual returns: strip beta-explained part, then cumulate over momentum window skipping recent month
    resid = ret - beta.shift(1).mul(mkt, axis=0)
    rmom = resid.rolling(MOM_WIN - SKIP, min_periods=(MOM_WIN - SKIP) // 2).sum().shift(SKIP)
    rmom = rmom / vol.replace(0, np.nan)
    score = (vol.rank(axis=1, pct=True, ascending=False)
             + 0.5 * mom.rank(axis=1, pct=True)
             + RES_W * rmom.rank(axis=1, pct=True)
             + 0.5 * beta.rank(axis=1, pct=True, ascending=False))
    score = score.where(vol.notna() & mom.notna() & beta.notna() & rmom.notna())
    rk = score.rank(axis=1, pct=True)
    sel = (rk >= 1 - TOP_FRAC).astype(float)
    w = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    on = pd.Series(np.arange(len(w)) % REBAL == 0, index=w.index)
    w = w.where(on, np.nan).ffill().fillna(0.0)
    bk = (w * beta.fillna(1.0)).sum(axis=1).clip(0, 1.5)
    vix = data.macro["vix"].reindex(prices.index).ffill().rolling(5, min_periods=1).mean()
    expo = (1.0 - (vix - VIX_LO) / (VIX_HI - VIX_LO) * (1.0 - MIN_EXPO)).clip(MIN_EXPO, 1.0).fillna(1.0)
    w = w.mul(expo, axis=0)
    avail = prices.notna().astype(float)
    ew = avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hedge = (bk * expo * HEDGE).rolling(REBAL, min_periods=1).mean()
    return w - ew.mul(hedge, axis=0)


import numpy as np, pandas as pd

def _f0506(prices, data):
    """Steady Ascent: each month hold the top quarter of assets by a blend of volatility-adjusted 12-1 momentum and low recent volatility, weighted by inverse volatility. Hypothesis: investors underreact to information that arrives as a smooth, quiet drift and overpay for lottery-like volatile winners, so calm winners keep drifting while avoiding the momentum crashes of noisy high-beta names."""
    LOOKBACK, SKIP, VOL_WIN, TOP, MOM_WEIGHT = 252, 21, 46, 0.2702, 0.5055
    rets = prices.pct_change(fill_method=None)
    # Smooth (risk-adjusted) momentum: 12-1 return per unit of volatility over the same window
    mom = prices.shift(SKIP) / prices.shift(LOOKBACK) - 1
    long_vol = rets.rolling(LOOKBACK - SKIP, min_periods=(LOOKBACK - SKIP) // 2).std().shift(SKIP)
    smooth = mom / long_vol.replace(0, np.nan)
    # Recent calmness
    vol = rets.rolling(VOL_WIN, min_periods=VOL_WIN).std().replace(0, np.nan)
    r_mom = smooth.rank(axis=1, pct=True)
    r_calm = (-vol).rank(axis=1, pct=True)
    score = MOM_WEIGHT * r_mom + (1 - MOM_WEIGHT) * r_calm   # NaN if either leg missing
    sel = score.rank(axis=1, pct=True) >= 1 - TOP
    w = (1.0 / vol).where(sel, 0.0).fillna(0.0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # Rebalance only on the first trading day of each month to limit turnover
    month = pd.Series(prices.index.month, index=prices.index)
    first_day = (month != month.shift(1)).values
    w.loc[~first_day] = np.nan
    return w.ffill().fillna(0.0)


def strategy(prices, data):
    """The Team: a diversified team of 2 evolved strategies (Hedged Calm Residual v31, Steady Ascent v16)."""
    out = prices * 0.0
    out = out + 0.7399 * _f0726(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.2601 * _f0506(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return out
