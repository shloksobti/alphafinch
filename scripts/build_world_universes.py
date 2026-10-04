"""Snapshot index members for the world markets into alphafinch/universes/*.csv.

Sources: Wikipedia constituent tables and Nikkei's component page. Run occasionally; the package
ships the snapshot so the tool never scrapes at runtime. Needs lxml (dev only).
"""
import io
import re
from pathlib import Path

import pandas as pd
import requests
from lxml import html as lh

H = {"User-Agent": "alphafinch-research/0.1 (index constituents; https://github.com/shloksobti/alphafinch)"}
OUT = Path(__file__).resolve().parents[1] / "alphafinch" / "universes"

# coarse sectors, so differently-labelled European tables share one taxonomy
COARSE = [("Financials", r"bank|financ|insur|asset|exchange|capital|invest|real estate|reit|propert"),
          ("Energy", r"oil|gas|energy|petrol"),
          ("Utilities", r"utilit|electric|water|power"),
          ("Health Care", r"health|pharma|medic|biotech|life science"),
          ("Technology", r"tech|software|semiconductor|computer|information|electronic|internet"),
          ("Communication", r"telecom|communic|media|entertain"),
          ("Materials", r"chemic|material|steel|metal|mining|paper|construction material|basic"),
          ("Industrials", r"industr|aero|defen|construct|engineer|machin|transport|logist|airline|automat|"
                          r"business serv|electrical|build"),
          ("Consumer Discretionary", r"auto|car|apparel|luxury|retail|consumer (cyclical|discretionary|services)|"
                                     r"hotel|travel|tour|leisure|textile|cloth|media"),
          ("Consumer Staples", r"food|beverage|drink|staple|tobacco|personal|household|consumer goods")]


def coarse(s: str) -> str:
    s = str(s).lower()
    for name, pat in COARSE:
        if re.search(pat, s):
            return name
    return "Other"


def wiki(page: str) -> pd.DataFrame:
    text = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers=H, timeout=30).text
    ts = [t for t in pd.read_html(io.StringIO(text)) if len(t) >= 20 and not any("Year" in str(c) for c in t.columns)]
    return ts[0]


def col(df, *keys):
    for c in df.columns:
        if any(k.lower() in str(c).lower() for k in keys):
            return df[c]
    raise KeyError(keys)


def yahoo_dash(sym: str) -> str:
    return str(sym).strip().replace(".", "-").replace(" ", "-")


def uk():
    t = wiki("FTSE_100_Index")
    return pd.DataFrame({"symbol": col(t, "Ticker").map(lambda s: yahoo_dash(s).rstrip("-") + ".L"),
                         "sector": col(t, "sector").map(coarse)})


def europe():
    rows = []
    for page, sec in [("DAX", "Sector"), ("CAC_40", "Sector"), ("IBEX_35", "Sector"), ("FTSE_MIB", "Sector"),
                      ("AEX_index", "Sector")]:
        t = wiki(page)
        rows.append(pd.DataFrame({"symbol": col(t, "Ticker").astype(str).str.strip(), "sector": col(t, sec).map(coarse)}))
    df = pd.concat(rows).drop_duplicates("symbol")
    return df[df.symbol.str.contains(r"\.(?:DE|PA|MC|MI|AS)$")]


def hongkong():
    t = wiki("Hang_Seng_Index")
    code = col(t, "Ticker").astype(str).str.extract(r"(\d+)")[0].astype(int)
    return pd.DataFrame({"symbol": code.map(lambda c: f"{c:04d}.HK"), "sector": col(t, "Sub-index")})


def australia():
    t = wiki("S&P/ASX_200")
    return pd.DataFrame({"symbol": col(t, "Code").astype(str).str.strip() + ".AX", "sector": col(t, "Sector")})


def canada():
    t = wiki("S&P/TSX_60")
    return pd.DataFrame({"symbol": col(t, "Symbol").map(lambda s: yahoo_dash(s) + ".TO"), "sector": col(t, "Sector")})


def korea():
    t = wiki("KOSPI_200")
    return pd.DataFrame({"symbol": col(t, "Symbol").astype(str).str.zfill(6) + ".KS", "sector": col(t, "Sector")})


def japan():
    text = requests.get("https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225", headers=H, timeout=30).text
    doc = lh.fromstring(text)
    rows = []
    for table in doc.iter("table"):
        head = table.getprevious()
        while head is not None and not (head.text_content() or "").strip():
            head = head.getprevious()
        sector = (head.text_content().strip() if head is not None else "Unknown")[:60]
        df = pd.read_html(io.StringIO(lh.tostring(table, encoding="unicode")))[0]
        if "Code" in df.columns:
            rows += [{"symbol": f"{str(c).strip()}.T", "sector": sector} for c in df["Code"]]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("uk", uk), ("europe", europe), ("hongkong", hongkong), ("australia", australia),
                     ("canada", canada), ("korea", korea), ("japan", japan)]:
        df = fn().dropna().drop_duplicates("symbol")
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:10s} {len(df):4d} members, {df.sector.nunique():2d} sectors, e.g. {', '.join(df.symbol[:4])}")
