"""Universes: which assets a market contains, plus sector labels (and SEC CIKs for the US)."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path.home() / ".alphafinch" / "universes"
SNAPSHOTS = Path(__file__).parent / "universes"       # shipped member lists (scripts/build_world_universes.py)
WORLD = {"uk": "FTSE 100", "europe": "Eurozone large caps (DAX, CAC 40, IBEX 35, FTSE MIB, AEX)",
         "japan": "Nikkei 225", "hongkong": "Hang Seng", "australia": "S&P/ASX 200", "canada": "S&P/TSX 60",
         "korea": "KOSPI 200"}
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NIFTY_URL = "https://archives.nseindia.com/content/indices/ind_nifty{n}list.csv"
UA = {"User-Agent": "Mozilla/5.0 (alphafinch; https://github.com/shloksobti/alphafinch)"}

US30 = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "PEP", "WMT",
        "HD", "UNH", "V", "MA", "DIS", "INTC", "CSCO", "ORCL", "PFE", "MRK", "CVX", "BA", "CAT", "MCD", "NKE",
        "IBM", "T"]
CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT",
          "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT", "XLMUSDT", "ETCUSDT"]


def _fetch_csv(url: str, name: str, max_age_days: float = 30) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if f.exists() and time.time() - f.stat().st_mtime < max_age_days * 86400:
        return pd.read_csv(f)
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(f, index=False)
        return df
    except Exception:
        if f.exists():
            return pd.read_csv(f)
        raise


def sp500() -> pd.DataFrame:
    """columns: symbol (Yahoo), sector, cik"""
    df = _fetch_csv(SP500_URL, "sp500.csv")
    return pd.DataFrame({"symbol": df["Symbol"].str.replace(".", "-", regex=False),
                         "sector": df["GICS Sector"], "cik": df["CIK"].astype(int)})


def nifty(n: int = 200) -> pd.DataFrame:
    """columns: symbol (Yahoo .NS), sector"""
    df = _fetch_csv(NIFTY_URL.format(n=n), f"nifty{n}.csv")
    df.columns = [c.strip() for c in df.columns]
    return pd.DataFrame({"symbol": df["Symbol"].str.strip() + ".NS", "sector": df["Industry"].str.strip()})


def india_fno() -> list[str]:
    """Stocks with single-stock futures on NSE, as Yahoo symbols. A snapshot of NSE's
    fo_mktlots.csv (today's list, so not point-in-time), shipped with the package."""
    return pd.read_csv(SNAPSHOTS / "india_fno.csv")["symbol"].tolist()


# Futures, represented by funds that hold and roll real futures (or the underlying, for equity indices,
# bonds and currencies). Free continuous futures prices splice contracts without adjusting for the
# roll, which creates fake jumps; these funds' returns include the roll correctly. data.build turns
# them into excess returns (minus the T-bill rate), which is what a futures position earns.
FUTURES = {
    "equity": {"SPY": "S&P 500 (ES)", "QQQ": "Nasdaq 100 (NQ)", "IWM": "Russell 2000 (RTY)",
               "EFA": "MSCI EAFE", "EEM": "MSCI Emerging Markets", "EWJ": "Japan (Nikkei/TOPIX)",
               "FEZ": "Euro Stoxx 50", "EWU": "FTSE 100", "FXI": "China large caps", "INDA": "India (NIFTY)",
               "EWZ": "Brazil (Ibovespa)", "EWY": "Korea (KOSPI)"},
    "rates": {"SHY": "US 2-year (ZT)", "IEI": "US 5-year (ZF)", "IEF": "US 10-year (ZN)", "TLT": "US 30-year (ZB)",
              "BWX": "International government bonds", "TIP": "US inflation-linked bonds"},
    "fx": {"FXE": "Euro (6E)", "FXY": "Japanese yen (6J)", "FXB": "British pound (6B)", "FXA": "Australian dollar (6A)",
           "FXC": "Canadian dollar (6C)", "FXF": "Swiss franc (6S)", "UUP": "US dollar index (DX)"},
    "energy": {"USO": "WTI crude oil (CL)", "BNO": "Brent crude oil", "UNG": "Natural gas (NG)", "UGA": "Gasoline (RB)"},
    "metals": {"GLD": "Gold (GC)", "SLV": "Silver (SI)", "CPER": "Copper (HG)", "PPLT": "Platinum (PL)",
               "PALL": "Palladium (PA)"},
    "agriculture": {"CORN": "Corn (ZC)", "WEAT": "Wheat (ZW)", "SOYB": "Soybeans (ZS)", "CANE": "Sugar (SB)",
                    "DBA": "Agriculture basket"},
}


def futures() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "sector": cls, "name": nm} for cls, d in FUTURES.items() for s, nm in d.items()])


def india_futures() -> pd.DataFrame:
    """NSE futures: every stock with single-stock futures, plus NIFTY and BANKNIFTY index futures
    (represented by index funds that include dividends)."""
    sectors = nifty(500).set_index("symbol")["sector"]
    fno = india_fno()
    rows = [{"symbol": s, "sector": sectors.get(s, "Unknown")} for s in fno]
    rows += [{"symbol": "NIFTYBEES.NS", "sector": "Index"}, {"symbol": "BANKBEES.NS", "sector": "Index"}]
    return pd.DataFrame(rows)


def get(market: str) -> pd.DataFrame:
    if market == "us":
        return sp500()
    if market == "india":
        return nifty(200)
    if market == "us30":
        return pd.DataFrame({"symbol": US30, "sector": "Unknown", "cik": 0})
    if market == "crypto":
        return pd.DataFrame({"symbol": CRYPTO, "sector": "Crypto"})
    if market == "futures":
        return futures()
    if market == "india-futures":
        return india_futures()
    if market in WORLD:
        return pd.read_csv(SNAPSHOTS / f"{market}.csv", dtype=str)
    raise ValueError(market)
