"""Market data panels: free sources, no API keys, cached in ~/.alphafinch.

A Panel holds everything a strategy may look at, aligned on trading days:
  close, open, high, low, volume   DataFrames (dates x assets); close is split/dividend adjusted
  sector                           Series (asset -> sector)
  macro                            DataFrame (dates x series): rates, volatility index, index level,
                                   commodities, FX. Published-with-a-lag series are lagged.
  fund                             dict of DataFrames (dates x assets), point-in-time fundamentals
                                   (US only, from SEC filings: each value appears the day after
                                   the filing that disclosed it)

Markets: us (S&P 500), india (NIFTY 200), us30, crypto, industries (Ken French), synthetic, and the
world markets uk, europe, japan, hongkong, australia, canada, korea (see universe.WORLD).
"""
from __future__ import annotations

import io
import pickle
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import universe as uni

CACHE = Path.home() / ".alphafinch"
UA = {"User-Agent": "Mozilla/5.0 (alphafinch research tool; https://github.com/shloksobti/alphafinch)"}
import os


def _sec_headers():
    """The SEC requires a User-Agent naming the requester with an email address
    (https://www.sec.gov/os/accessing-edgar-data). Set ALPHAFINCH_SEC_CONTACT="Name email@domain"."""
    contact = os.environ.get("ALPHAFINCH_SEC_CONTACT")
    return {"User-Agent": f"alphafinch {contact}"} if contact else None
DEFAULT_START = {"us": "2010-01-01", "india": "2010-01-01", "us30": "2008-04-01", "crypto": "2020-10-01",
                 "industries": "1970-01-01", **{m: "2010-01-01" for m in uni.WORLD}}
WORLD_INDEX = {"uk": "^FTSE", "europe": "^STOXX50E", "japan": "^N225", "hongkong": "^HSI", "australia": "^AXJO",
               "canada": "^GSPTSE", "korea": "^KS11"}

MACRO = {
    "us": {"yahoo": {"vix": "^VIX", "index": "^GSPC", "oil": "CL=F", "gold": "GC=F"},
           "fred": {"rate_10y": ("DGS10", 1), "rate_3m": ("DGS3MO", 1), "curve_10y_2y": ("T10Y2Y", 1),
                    "credit_spread": ("BAMLH0A0HYM2", 1), "dollar_index": ("DTWEXBGS", 1)}},
    "india": {"yahoo": {"vix": "^INDIAVIX", "index": "^NSEI", "oil": "CL=F", "gold": "GC=F"},
              "fred": {"rate_10y": ("INDIRLTLT01STM", 45), "usd_inr": ("DEXINUS", 1), "us_rate_10y": ("DGS10", 1)}},
    "us30": {"yahoo": {"vix": "^VIX", "index": "^GSPC"}, "fred": {"rate_10y": ("DGS10", 1)}},
    "crypto": {"yahoo": {"vix": "^VIX", "index": "^GSPC"}, "fred": {"rate_10y": ("DGS10", 1)}},
}
for _m, _ix in WORLD_INDEX.items():         # world markets: local index; the US VIX as the global fear gauge
    MACRO[_m] = {"yahoo": {"vix": "^VIX", "index": _ix, "oil": "CL=F", "gold": "GC=F"},
                 "fred": {"us_rate_10y": ("DGS10", 1)}}


@dataclass
class Panel:
    close: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    sector: pd.Series | None = None
    macro: pd.DataFrame | None = None
    fund: dict = field(default_factory=dict)

    @property
    def index(self):
        return self.close.index

    @property
    def columns(self):
        return self.close.columns

    @property
    def shape(self):
        return self.close.shape

    def _map(self, f):
        g = lambda x: None if x is None else f(x)
        return Panel(f(self.close), g(self.open), g(self.high), g(self.low), g(self.volume), self.sector,
                     g(self.macro), {k: f(v) for k, v in self.fund.items()})

    def head(self, n: int) -> "Panel":
        return self._map(lambda x: x.iloc[:n])

    def before(self, ts) -> "Panel":
        return self._map(lambda x: x[x.index < ts])

    def describe(self) -> str:
        """What a strategy can see (used in AI prompts)."""
        n = self.shape[1]
        lines = [f"- data.close / data.open / data.high / data.low: prices (dates x {n} assets)"
                 if self.open is not None else f"- data.close: prices (dates x {n} assets)"]
        if self.volume is not None:
            lines.append("- data.volume: shares traded per day (dates x assets)")
        if self.sector is not None and self.sector.nunique() > 1:
            lines.append(f"- data.sector: pandas Series mapping asset -> sector ({self.sector.nunique()} sectors)")
        if self.macro is not None and len(self.macro.columns):
            lines.append("- data.macro: DataFrame (dates x series): " + ", ".join(self.macro.columns))
        if self.fund:
            lines.append("- data.fund: dict of point-in-time fundamentals (dates x assets), each value known from "
                         "the day after its SEC filing: " + ", ".join(self.fund))
        return "\n".join(lines)

    def astype32(self) -> "Panel":
        return self._map(lambda x: x.astype("float32"))


# ------------------------------------------------------------------------------------- caching
def _cache_df(name: str, fetch, max_age_days: float = 1.0) -> pd.DataFrame | None:
    f = CACHE / "series" / f"{name}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400:
        return pd.read_csv(f, index_col=0, parse_dates=True)
    try:
        df = fetch()
    except Exception:
        df = None
    if df is None or len(df) == 0:
        return pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None
    df.to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)


# ------------------------------------------------------------------------------------- sources
def _yahoo_ohlcv(symbol: str) -> pd.DataFrame | None:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=0&period2={int(time.time())}"
           "&interval=1d&events=split,div")
    r = None
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(2 + 4 * attempt)
            continue
        break
    if r is None or r.status_code != 200:
        return None
    res = r.json()["chart"]["result"][0]
    if "timestamp" not in res:
        return None
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(res["timestamp"], unit="s").normalize(), dtype=float)
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df["raw_close"] = df["close"]
    if adj is not None:
        ratio = pd.Series(adj, index=df.index, dtype=float) / df["close"]
        for k in ("open", "high", "low", "close"):
            df[k] = df[k] * ratio
    return df[~df.index.duplicated(keep="last")].dropna(subset=["close"])


def _binance_ohlcv(symbol: str) -> pd.DataFrame | None:
    out, start = [], 1500000000000
    while True:
        r = requests.get("https://api.binance.com/api/v3/klines", timeout=30,
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
    df = pd.DataFrame([[float(x[i]) for i in (1, 2, 3, 4, 5)] for x in out], index=idx,
                      columns=["open", "high", "low", "close", "volume"])
    df["raw_close"] = df["close"]
    return df


def _fred(series_id: str) -> pd.DataFrame | None:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.set_index("date").dropna()


SEC_CONCEPTS = {
    "net_income": ["NetIncomeLoss"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "equity": ["StockholdersEquity"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
}


def _sec_fundamentals(cik: int) -> pd.DataFrame | None:
    """Annual (10-K) values, each dated by its FILING date (point-in-time)."""
    headers = _sec_headers()
    if headers is None:
        return None
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=headers, timeout=60)
    if r.status_code != 200:
        return None
    facts = r.json().get("facts", {})
    gaap, dei = facts.get("us-gaap", {}), facts.get("dei", {})
    cols = {}
    for name, tags in SEC_CONCEPTS.items():
        rows = []
        for tag in tags:
            for items in gaap.get(tag, {}).get("units", {}).values():
                for it in items:
                    if it.get("form") in ("10-K", "10-K/A") and it.get("fp") == "FY" and "filed" in it:
                        rows.append((it["filed"], it["end"], it["val"]))
            if rows:
                break
        if rows:
            d = pd.DataFrame(rows, columns=["filed", "end", "val"])
            d["filed"], d["end"] = pd.to_datetime(d["filed"]), pd.to_datetime(d["end"])
            d = d[(d["filed"] - d["end"]).dt.days.between(0, 400)]
            d = d.sort_values(["filed", "end"]).groupby("filed").last()   # latest period in each filing
            cols[name] = d["val"]
    shares = [(it["filed"], it["val"]) for it in
              dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", []) if "filed" in it]
    if shares:
        s = pd.DataFrame(shares, columns=["filed", "val"])
        s["filed"] = pd.to_datetime(s["filed"])
        cols["shares"] = s.groupby("filed")["val"].last()
    return pd.DataFrame(cols).sort_index() if cols else None


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
    return (1 + r.fillna(0)).cumprod()


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
    for t in range(60, n):
        drift[t] = 0.02 * eps[t - 60:t - 5].mean(0) - 0.05 * eps[t - 5:t].mean(0)
    r = mkt[:, None] * beta + eps + drift
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"SYN{i:02d}" for i in range(n_assets)])


# ------------------------------------------------------------------------------------- panels
def _fund_panel(univ: pd.DataFrame, close: pd.DataFrame, raw: pd.DataFrame, progress=None) -> dict:
    ciks = dict(zip(univ["symbol"], univ["cik"]))
    syms = [s for s in close.columns if ciks.get(s)]

    def one(s):
        time.sleep(0.15)
        return s, _cache_df(f"sec_{ciks[s]}", lambda: _sec_fundamentals(int(ciks[s])), max_age_days=30)

    out = {}
    with ThreadPoolExecutor(5) as ex:                     # SEC allows 10 requests/second
        for i, (s, df) in enumerate(ex.map(one, syms)):
            if progress:
                progress(i, len(syms), f"SEC filings {s}")
            if df is not None:
                out[s] = df
    idx = close.index

    def field_(name):
        m = {}
        for s, df in out.items():
            if name in df:
                ser = df[name].dropna()
                ser.index = ser.index + pd.Timedelta(days=1)          # known the day after filing
                ser = ser.groupby(level=0).last()
                m[s] = ser.reindex(idx.union(ser.index)).ffill().reindex(idx)
        return pd.DataFrame(m).reindex(columns=close.columns)

    ni, rev, eq, eps, sh = (field_(k) for k in ("net_income", "revenue", "equity", "eps", "shares"))
    mcap = sh * raw
    fund = {"market_cap": mcap, "earnings_yield": eps / raw, "book_to_market": eq / mcap,
            "roe": ni / eq.where(eq > 0), "sales_growth": rev / rev.shift(252) - 1}
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in fund.items()}


def _macro(market: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    spec = MACRO.get(market, {})
    cols = {}
    for name, sym in spec.get("yahoo", {}).items():
        df = _cache_df(f"yahoo_{sym}", lambda sym=sym: _yahoo_ohlcv(sym))
        if df is not None:
            cols[name] = df["close"]
    for name, (sid, lag_days) in spec.get("fred", {}).items():
        df = _cache_df(f"fred_{sid}", lambda sid=sid: _fred(sid))
        if df is not None:
            s = df.iloc[:, 0].copy()
            s.index = s.index + pd.Timedelta(days=lag_days)           # publication lag: no look-ahead
            cols[name] = s
    if not cols:
        return pd.DataFrame(index=idx)
    m = pd.DataFrame({k: v.groupby(level=0).last() for k, v in cols.items()})
    return m.reindex(m.index.union(idx)).ffill().reindex(idx)


def build(market: str, tickers=None, start=None, progress=None) -> Panel:
    start = start or DEFAULT_START.get(market)
    if market == "synthetic":
        px = synthetic()
        return Panel(px, sector=pd.Series("Synthetic", index=px.columns))
    if market == "industries":
        px = _cache_df("french_49_industries", _french_industries, max_age_days=30)
        px = px.loc[start:].dropna(axis=1)
        return Panel(px, sector=pd.Series(list(px.columns), index=px.columns))
    univ = uni.get(market) if not tickers else pd.DataFrame({"symbol": tickers, "sector": "Unknown", "cik": 0})
    src = _binance_ohlcv if market == "crypto" else _yahoo_ohlcv
    frames = {}

    def one(s):
        return s, _cache_df(f"{market}_{s}", lambda: src(s))

    with ThreadPoolExecutor(4) as ex:
        for i, (s, df) in enumerate(ex.map(one, univ["symbol"])):
            if progress:
                progress(i, len(univ), s)
            if df is not None and len(df) > 500:
                frames[s] = df
    if not frames:
        raise RuntimeError("could not download any data (offline?). Try --market synthetic")
    close = pd.DataFrame({s: d["close"] for s, d in frames.items()}).sort_index()
    if start:
        close = close.loc[start:]
    if market != "crypto":
        close = close[close.index.dayofweek < 5]
    close = close.dropna(axis=1, thresh=int(0.95 * len(close))).ffill(limit=5).dropna(how="any")
    keep, idx = close.columns, close.index
    pick = lambda k: pd.DataFrame({s: frames[s][k] for s in keep}).reindex(idx).ffill(limit=5)
    sector = univ.set_index("symbol")["sector"].reindex(keep).fillna("Unknown")
    panel = Panel(close, pick("open"), pick("high"), pick("low"), pick("volume").fillna(0), sector,
                  _macro(market, idx))
    if market == "us" and not tickers and "cik" in univ and _sec_headers() is not None:
        panel.fund = _fund_panel(univ, close, pick("raw_close"), progress)
    return panel


def load(market: str = "us", tickers=None, start=None, progress=None, refresh_days: float = 1.0) -> Panel:
    """Cached full panel for a market (rebuilt at most once a day)."""
    fund_tag = "_fund" if (market == "us" and _sec_headers() is not None) else ""
    key = f"{market}_{'-'.join(tickers) if tickers else 'default'}_{start or DEFAULT_START.get(market)}{fund_tag}"
    f = CACHE / "panels" / f"{key}.pkl"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists() and time.time() - f.stat().st_mtime < refresh_days * 86400:
        return pickle.loads(f.read_bytes())
    p = build(market, tickers, start, progress).astype32()
    f.write_bytes(pickle.dumps(p))
    return p
