"""Seed population: classic, simple ideas the islands start from."""

SEEDS = {
    "Trend Rider": '''
import numpy as np, pandas as pd

def strategy(prices):
    """Trend Rider: hold assets whose price is above their 200-day average."""
    LOOKBACK = 200
    above = prices > prices.rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    w = above.astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
''',
    "Momentum Crown": '''
import numpy as np, pandas as pd

def strategy(prices):
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
''',
    "Snapback": '''
import numpy as np, pandas as pd

def strategy(prices):
    """Snapback: buy last week's biggest losers, short the winners (short-term reversal)."""
    LOOKBACK = 5
    r = prices.pct_change(LOOKBACK)
    z = r.sub(r.mean(axis=1), axis=0)
    w = -z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)
''',
    "Calm Seeker": '''
import numpy as np, pandas as pd

def strategy(prices):
    """Calm Seeker: overweight the least volatile assets (low-volatility anomaly)."""
    WINDOW = 63
    vol = prices.pct_change().rolling(WINDOW, min_periods=WINDOW).std()
    inv = 1.0 / vol
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
''',
    "Storm Shelter": '''
import numpy as np, pandas as pd

def strategy(prices):
    """Storm Shelter: equal weight, but step aside when market volatility spikes."""
    WINDOW, CALM = 21, 0.20
    mkt = prices.pct_change().mean(axis=1)
    vol = mkt.rolling(WINDOW, min_periods=WINDOW).std() * np.sqrt(252)
    on = (vol < CALM).astype(float)
    w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    return w.mul(on, axis=0).fillna(0.0)
''',
    "Breakout Hunter": '''
import numpy as np, pandas as pd

def strategy(prices):
    """Breakout Hunter: buy assets making 55-day highs, exit at 20-day lows."""
    ENTER, EXIT = 55, 20
    hi = prices.rolling(ENTER, min_periods=ENTER).max()
    lo = prices.rolling(EXIT, min_periods=EXIT).min()
    sig = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    sig[prices >= hi] = 1.0
    sig[prices <= lo] = 0.0
    w = sig.ffill().fillna(0.0)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
''',
}

IMMIGRANT_THEMES = [
    "calendar seasonality (turn-of-month, day-of-week, month-of-year)",
    "volatility targeting and risk parity",
    "cross-sectional mean reversion after extreme moves",
    "dual momentum: absolute plus relative strength",
    "drawdown-based regime filters",
    "correlation-aware diversification",
    "trend strength measured by regression slope or R-squared",
    "pairs of assets that move together and diverge",
    "skewness or tail-risk based asset selection",
    "breadth: the share of assets in an uptrend as a market signal",
]
