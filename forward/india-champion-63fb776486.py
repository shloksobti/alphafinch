import numpy as np, pandas as pd

def strategy(prices, data):
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
