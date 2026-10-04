import numpy as np, pandas as pd

def _f0733(prices, data):
    """Calm Quality Shield: hold profitable cheap stocks tilted within sectors, hedged partly by an equal-weight short, and scaled down smoothly when VIX is high and rising. Hypothesis: the market-timing signal protects against panics, while the quality-value tilt supplies stock-selection alpha that plain market exposure lacks."""
    VIX_LO, VIX_HI, WINDOW, HEDGE, SMOOTH = 24.5931, 34.0, 10, 0.5, 5
    ey = data.fund["earnings_yield"].reindex(prices.index).ffill()
    btm = data.fund["book_to_market"].reindex(prices.index).ffill()
    roe = data.fund["roe"].reindex(prices.index).ffill()
    sec = data.sector.reindex(prices.columns)

    def sector_rank(x):
        r = x.rank(axis=1, pct=True)
        grp = r.T.groupby(sec.values).transform("mean").T
        return r - grp

    value = sector_rank(ey) + sector_rank(btm)
    score = value.where(roe > 0)
    score = score.sub(score.mean(axis=1), axis=0)
    long = score.clip(lower=0)
    long = long.div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    n = prices.shape[1]
    ew = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    book = long - HEDGE * ew

    vix = data.macro["vix"].reindex(prices.index).ffill()
    level = ((VIX_HI - vix) / (VIX_HI - VIX_LO)).clip(0, 1)
    rising = (vix > vix.rolling(WINDOW, min_periods=WINDOW).mean()).astype(float)
    scale = level.where(rising > 0, 1.0 - 0.5 * (1 - level))
    scale = scale.rolling(SMOOTH, min_periods=1).mean().fillna(1.0)
    w = book.mul(scale, axis=0).fillna(0)
    return w.rolling(SMOOTH, min_periods=1).mean()


import numpy as np, pandas as pd

def _f0012(prices):
    """Momentum Crown: at each month start, hold the top quarter of assets by 12-1 month return."""
    LOOKBACK, SKIP, TOP = 252, 21, 0.25
    mom = prices.shift(SKIP) / prices.shift(LOOKBACK) - 1
    rank = mom.rank(axis=1, pct=True)
    w = (rank >= 1 - TOP).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    month = pd.Series(prices.index.month, index=prices.index)
    first_day = (month != month.shift(1)).values      # rebalance on the first trading day of each month
    w.loc[~first_day] = np.nan
    return w.ffill().fillna(0.0)


def strategy(prices, data):
    """The Team: a diversified team of 2 evolved strategies (Calm Quality Shield v14, Momentum Crown)."""
    out = prices * 0.0
    out = out + 0.7071 * _f0733(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.2929 * _f0012(prices).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return out
