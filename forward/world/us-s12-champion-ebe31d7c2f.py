import numpy as np, pandas as pd

def strategy(prices, data):
    """Quiet Sector Tether: within each sector, own stocks with low market correlation (volatility-purged) and short high co-movers, inverse-vol sized. Hypothesis: high co-movers carry crowded systematic risk and are overpriced by investors who seek market exposure, while idiosyncratic low-correlation names are neglected, so the spread earns alpha beyond beta."""
    CORR_WINDOW = 126
    VOL_WINDOW = 82
    Q = 0.2
    REBAL = 'M'
    SMOOTH_HL = 10

    rets = prices.pct_change()
    mkt = rets.mean(axis=1)

    # Rolling correlation with the equal-weight market
    corr = rets.rolling(CORR_WINDOW, min_periods=CORR_WINDOW).corr(mkt)
    vol = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    # Purge volatility: cross-sectional residual of corr on log vol
    lv = np.log(vol.replace(0, np.nan))
    zc = tk.zscore(corr)
    zv = tk.zscore(lv)
    cov = (zc * zv).mean(axis=1)
    var = (zv * zv).mean(axis=1)
    b = cov / var.replace(0, np.nan)
    resid = zc.sub(zv.mul(b, axis=0))

    # Low correlation = high score, neutral within sector
    score = -resid
    score = tk.neutralize(tk.winsorize(score), data.sector)
    score = tk.smooth(score, SMOOTH_HL)

    w = tk.long_short(score, q=Q)

    # Inverse-vol sizing, keep each leg balanced to gross 1
    inv = 1.0 / vol.replace(0, np.nan)
    w = (w * inv).fillna(0.0)
    longs = w.clip(lower=0)
    shorts = w.clip(upper=0)
    ls = longs.sum(axis=1).replace(0, np.nan)
    ss = shorts.abs().sum(axis=1).replace(0, np.nan)
    w = 0.5 * longs.div(ls, axis=0) + 0.5 * shorts.div(ss, axis=0)
    w = w.fillna(0.0)

    w = tk.cap(w, 0.03)
    return tk.rebalance(w, every=REBAL)
