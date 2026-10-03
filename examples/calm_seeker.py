import numpy as np, pandas as pd


def strategy(prices):
    """Calm Seeker: overweight the least volatile assets (low-volatility anomaly)."""
    WINDOW = 63
    vol = prices.pct_change().rolling(WINDOW, min_periods=WINDOW).std()
    inv = 1.0 / vol
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
