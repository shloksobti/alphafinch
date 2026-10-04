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


SEEDS.update({
    "Sector Relay": '''
import numpy as np, pandas as pd

def strategy(prices, data):
    """Sector Relay: own the stocks of the three sectors with the best 6-month momentum. Hypothesis: sector trends persist as capital rotates slowly between industries."""
    LOOKBACK, TOP = 126, 3
    mom = prices / prices.shift(LOOKBACK) - 1
    sector = data.sector.reindex(prices.columns)
    sec_mom = mom.T.groupby(sector.values).mean().T
    rank = sec_mom.rank(axis=1, ascending=False)
    winners = (rank <= TOP).astype(float)
    w = winners.reindex(columns=sector.values).set_axis(prices.columns, axis=1).fillna(0.0)
    month = pd.Series(prices.index.month, index=prices.index)
    w.loc[(month == month.shift(1)).values] = np.nan
    w = w.ffill().fillna(0.0)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
''',
    "Quiet Money": '''
import numpy as np, pandas as pd

def strategy(prices, data):
    """Quiet Money: prefer stocks whose trading volume has dried up relative to their own history. Hypothesis: neglected stocks are underpriced because attention is scarce."""
    SHORT, LONG, TOP = 21, 252, 0.2
    if data.volume is None:
        return prices * 0.0
    dv = (data.volume * prices).rolling(SHORT, min_periods=SHORT).mean()
    base = (data.volume * prices).rolling(LONG, min_periods=LONG).mean()
    neglect = -(dv / base)
    rank = neglect.rank(axis=1, pct=True)
    w = (rank >= 1 - TOP).astype(float)
    month = pd.Series(prices.index.month, index=prices.index)
    w.loc[(month == month.shift(1)).values] = np.nan
    w = w.ffill().fillna(0.0)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
''',
    "Fear Gauge": '''
import numpy as np, pandas as pd

def strategy(prices, data):
    """Fear Gauge: hold the market unless the volatility index is high and rising. Hypothesis: panics cluster, so stepping aside when fear is accelerating avoids the worst drawdowns."""
    HIGH, WINDOW = 25.0, 10
    ew = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    if data.macro is None or "vix" not in data.macro:
        return ew
    vix = data.macro["vix"].reindex(prices.index).ffill()
    danger = (vix > HIGH) & (vix > vix.rolling(WINDOW, min_periods=WINDOW).mean())
    return ew.mul((~danger).astype(float), axis=0)
''',
    "Cheap Quality": '''
import numpy as np, pandas as pd

def strategy(prices, data):
    """Cheap Quality: own profitable companies with high earnings yields, rebalanced monthly. Hypothesis: investors overpay for glamour and underpay for boring, profitable firms."""
    TOP = 0.2
    if "earnings_yield" not in data.fund:
        return prices * 0.0
    ey = data.fund["earnings_yield"]
    roe = data.fund.get("roe", ey * 0 + 1)
    score = ey.rank(axis=1, pct=True) + roe.rank(axis=1, pct=True)
    w = (score.rank(axis=1, pct=True) >= 1 - TOP).astype(float).where(score.notna(), 0.0)
    month = pd.Series(prices.index.month, index=prices.index)
    w.loc[(month == month.shift(1)).values] = np.nan
    w = w.ffill().fillna(0.0)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
''',
})

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
    "trading volume and attention: crowding, neglect, volume shocks",
    "sector-neutral selection: compare stocks only with their own sector",
    "macro regimes: interest rates, the yield curve, credit spreads, currency moves",
    "the volatility index as a fear and greed signal",
    "price gaps and intraday range from open/high/low",
    "fundamentals: value, profitability, growth and their interaction with momentum",
]

FUTURES_THEMES = [
    "time-series momentum: each future's own trend over several horizons, sized by volatility",
    "cross-asset signals: what bonds, the dollar or commodities say about equities, and vice versa",
    "carry proxies: assets whose excess returns persist (e.g. currencies, curve shape from the 2y/10y/30y bonds)",
    "risk parity and volatility targeting across asset classes",
    "commodity seasonality (month-of-year) and mean reversion after extreme moves",
    "crisis alpha: positioning that profits when the volatility index spikes",
    "relative value within an asset class (one bond, currency or commodity against its peers)",
    "macro regimes: the yield curve, credit spreads and the dollar as switches between asset classes",
]
