"""Market data: free sources, no API keys, cached locally.

Markets
  us        ~30 large US stocks + SPY, daily closes (Yahoo Finance chart API)
  india     ~25 NIFTY large caps, daily closes (Yahoo Finance, .NS tickers)
  crypto    top coins vs USDT, daily closes (Binance public API)
  industries 49 US industry portfolios since 1926 (Ken French Data Library)
  synthetic  a regime-switching simulated market (offline; for demos and tests)
Custom tickers: --tickers AAPL,MSFT,... (Yahoo symbols).
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path.home() / ".alphafinch" / "data"
UA = {"User-Agent": "Mozilla/5.0 (alphafinch; +https://github.com/shloksobti/alphafinch)"}

UNIVERSES = {
    "us": ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO",
           "PEP", "WMT", "HD", "UNH", "V", "MA", "DIS", "INTC", "CSCO", "ORCL", "PFE", "MRK", "CVX",
           "BA", "CAT", "MCD", "NKE", "IBM", "T"],
    "india": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
              "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
              "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
              "NESTLEIND.NS", "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS",
              "ONGC.NS"],
    "crypto": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "LTCUSDT",
               "TRXUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT", "XLMUSDT", "ETCUSDT"],
}


def _cached(name: str, fetch, max_age_days: float = 1.0) -> pd.Series | pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.csv"
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400:
        return pd.read_csv(f, index_col=0, parse_dates=True)
    try:
        obj = fetch()
    except Exception:
        obj = None
    if obj is None or len(obj) == 0:
        return pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None
    obj.to_frame() .to_csv(f) if isinstance(obj, pd.Series) else obj.to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)


def _yahoo(symbol: str) -> pd.Series | None:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=0&period2={int(time.time())}"
           "&interval=1d&events=split,div")
    r = requests.get(url, headers=UA, timeout=20)
    if r.status_code != 200:
        return None
    res = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
    s = pd.Series(adj, index=ts, name=symbol, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def _binance(symbol: str) -> pd.Series | None:
    out, start = [], 1500000000000  # 2017-07
    while True:
        r = requests.get("https://api.binance.com/api/v3/klines", timeout=20,
                         params={"symbol": symbol, "interval": "1d", "startTime": start, "limit": 1000})
        if r.status_code != 200:
            return None
        rows = r.json()
        if not rows:
            break
        out += rows
        start = rows[-1][0] + 86400000
        if len(rows) < 1000:
            break
    if not out:
        return None
    idx = pd.to_datetime([x[0] for x in out], unit="ms")
    return pd.Series([float(x[4]) for x in out], index=idx, name=symbol)


def _french_industries() -> pd.DataFrame:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers=UA, timeout=60).content))
    text = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(text) if l.strip().startswith(",Agric"))
    cols = [c.strip() for c in text[start].split(",")[1:]]
    rows = []
    for l in text[start + 1:]:
        p = l.split(",")
        if not p[0].strip().isdigit():
            break
        rows.append([p[0].strip()] + p[1:])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    r = df.set_index("date").astype(float).replace([-99.99, -999.0], np.nan) / 100
    return (1 + r.fillna(0)).cumprod()  # total-return index


def synthetic(n_assets: int = 20, years: int = 25, seed: int = 0) -> pd.DataFrame:
    """Regime-switching market with mild momentum and reversal effects (offline demo data)."""
    rng = np.random.default_rng(seed)
    n = years * 252
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() > 1 / 250 else 1 - regime[t - 1]
    mkt = np.where(regime == 0, 0.0005, -0.0004) + np.where(regime == 0, 0.009, 0.018) * rng.standard_normal(n)
    beta = rng.uniform(0.6, 1.4, n_assets)
    eps = 0.012 * rng.standard_normal((n, n_assets))
    drift = np.zeros((n, n_assets))
    for t in range(60, n):  # weak 3-month momentum, 1-week reversal
        drift[t] = 0.02 * eps[t - 60:t - 5].mean(0) - 0.05 * eps[t - 5:t].mean(0)
    r = mkt[:, None] * beta + eps + drift
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=[f"SYN{i:02d}" for i in range(n_assets)])


DEFAULT_START = {"us": "2008-04-01", "india": "2008-01-01", "crypto": "2020-10-01"}


def load(market: str = "us", tickers: list[str] | None = None, start: str | None = None,
         progress=None) -> pd.DataFrame:
    """Daily close (total return where available), rows = dates, columns = assets."""
    if market == "synthetic":
        return synthetic()
    if market == "industries":
        df = _cached("french_49_industries", _french_industries, max_age_days=30)
        df = df.loc["1970":] if start is None else df.loc[start:]
        return df.dropna(axis=1)
    syms = tickers or UNIVERSES[market]
    src = _binance if market == "crypto" else _yahoo
    series = []
    for i, s in enumerate(syms):
        if progress:
            progress(i, len(syms), s)
        df = _cached(f"{market}_{s}", lambda s=s: src(s))
        if df is not None and len(df) > 500:
            series.append(df.iloc[:, 0].rename(s))
        time.sleep(0.15)
    if not series:
        raise RuntimeError("could not download any data (offline?). Try --market synthetic")
    px = pd.concat(series, axis=1).sort_index()
    start = start or DEFAULT_START.get(market)
    if start:
        px = px.loc[start:]
    # keep assets that cover (almost) the whole window; forward-fill short gaps only
    px = px.dropna(axis=1, thresh=int(0.95 * len(px))).ffill(limit=5)
    px = px.dropna(how="any")
    if market != "crypto":
        px = px[px.index.dayofweek < 5]
    return px
