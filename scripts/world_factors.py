"""Is a world-exam alpha just known factors? Monthly regression on Fama-French developed-ex-US
5 factors + momentum (Ken French Data Library)."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from alphafinch import data
from alphafinch.engine import ANN, alpha_stats
from alphafinch.lab import Lab
from alphafinch.universe import WORLD

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def french(name: str) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(requests.get(BASE + name, timeout=60).content))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(",") or "Mkt-RF" in l or "WML" in l)
    rows = []
    for l in lines[i + 1:]:
        p = [x.strip() for x in l.split(",")]
        if not p[0].isdigit() or len(p[0]) != 6:
            if rows:
                break
            continue
        rows.append(p)
    cols = [c.strip() for c in lines[i].split(",")][1:]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.PeriodIndex([r[0] for r in rows], freq="M"),
                      columns=cols).astype(float) / 100
    return df


if __name__ == "__main__":
    code = Path(sys.argv[1]).read_text()
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2017-10-02")
    act = {}
    for mk in WORLD:
        with Lab(data.load(mk), start, workers=4, timeout=900) as lab:
            res = lab.run(code, "full", check_leaks=False)
            m = res.returns.index >= start
            r, mkt = res.returns[m], lab.mkt["full"][m]
            act[mk] = (r - alpha_stats(r, mkt)[0] * mkt)
    daily = pd.concat(act, axis=1).mean(axis=1).dropna()
    monthly = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    ff = french("Developed_ex_US_5_Factors_CSV.zip").join(french("Developed_ex_US_Mom_Factor_CSV.zip"), how="inner")
    df = pd.concat([monthly.rename("y"), ff], axis=1, join="inner").dropna()
    y = df["y"].values
    names = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML"] if c in df]
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in names])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    # Newey-West (3 lags) standard errors for the regression
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, 4):
        w = 1 - L / 4
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    print(f"months {n}: {df.index[0]} -> {df.index[-1]}")
    print(f"raw monthly mean {y.mean() * 12:+.2%}/yr, t {y.mean() / (y.std(ddof=1) / np.sqrt(n)):+.2f}")
    for nm, c, s in zip(["alpha (annual)"] + names, coef, se):
        scale = 12 if nm.startswith("alpha") else 1
        print(f"  {nm:15s} {c * scale:+.4f}  t {c / s:+.2f}")
    print(f"  R^2 {1 - resid.var() / y.var():.2f}")
