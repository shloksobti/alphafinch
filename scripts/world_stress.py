"""Stress tests for a world-exam result: costs, causal beta hedge, legs, and years."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alphafinch import data, world
from alphafinch.engine import ANN, alpha_stats
from alphafinch.lab import Lab
from alphafinch.universe import WORLD


def pooled_t(series: dict) -> tuple[float, float]:
    p = pd.concat(series, axis=1).mean(axis=1).dropna()
    m, t = world.newey_west_t(p.values, 10)
    return m * ANN, t


if __name__ == "__main__":
    code = Path(sys.argv[1]).read_text()
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2017-10-02")
    long_code = code.replace("    return tk.rebalance(w, every=REBAL)", "    return tk.rebalance(w.clip(lower=0) * 2, every=REBAL)")
    short_code = code.replace("    return tk.rebalance(w, every=REBAL)", "    return tk.rebalance(w.clip(upper=0) * 2, every=REBAL)")
    assert long_code != code and short_code != code
    by_cost = {c: {} for c in (5, 15, 30, 50)}
    hedged, legs, turn = {}, {"long": {}, "short": {}}, {}
    for mk in WORLD:
        panel = data.load(mk)
        for c in by_cost:
            with Lab(panel, start, workers=4, timeout=900, cost_bps=c) as lab:
                res = lab.run(code, "full", check_leaks=False)
                msk = res.returns.index >= start
                r, mkt = res.returns[msk], lab.mkt["full"][msk]
                beta = alpha_stats(r, mkt)[0]
                by_cost[c][mk] = r - beta * mkt
                if c == 5:
                    turn[mk] = float(res.turnover[msk].mean() * ANN)
                    # causal hedge: trailing 252-day beta known before each day
                    rf, mf = res.returns, lab.mkt["full"]
                    b = (rf.rolling(252, min_periods=126).cov(mf) / mf.rolling(252, min_periods=126).var()).shift(1)
                    hedged[mk] = (rf - b * mf)[msk]
                    for leg, lc in (("long", long_code), ("short", short_code)):
                        lr = lab.run(lc, "full", check_leaks=False).returns[msk]
                        legs[leg][mk] = lr - (mkt if leg == "long" else -mkt)   # vs the benchmark it trades against
    print("turnover x/yr:", {k: round(v, 1) for k, v in turn.items()})
    for c, s in by_cost.items():
        a, t = pooled_t(s)
        print(f"costs {c:2d} bps per unit turnover: pooled alpha {a:+.2%}/yr  NW t {t:+.2f}")
    a, t = pooled_t(hedged)
    print(f"causal rolling-beta hedge (5 bps): pooled active {a:+.2%}/yr  NW t {t:+.2f}")
    for leg in legs:
        a, t = pooled_t(legs[leg])
        print(f"{leg} leg vs benchmark: {a:+.2%}/yr  NW t {t:+.2f}")
    p = pd.concat(by_cost[5], axis=1).mean(axis=1).dropna()
    print("by year:", {int(y): f"{g.mean() * ANN:+.1%}" for y, g in p.groupby(p.index.year)})
