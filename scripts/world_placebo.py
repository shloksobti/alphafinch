"""Calibrate the world exam with placebo strategies (random long/short books redrawn monthly).

Without costs, placebo t-statistics should be centred on 0 with a spread of about 1, and about 5%
should exceed 1.645. With costs, placebos should lose roughly their trading costs.
"""
import json
import sys

import numpy as np

from alphafinch import world

PLACEBO = '''import numpy as np, pandas as pd

def strategy(prices):
    """Placebo {seed}: a random long/short book, redrawn each month."""
    SEED = {seed}
    month = prices.index.year * 12 + prices.index.month
    first = np.r_[True, month[1:] != month[:-1]]
    draws = np.random.default_rng(SEED).random((int(first.sum()), prices.shape[1]))   # one row per month, in order
    rank = pd.DataFrame(draws[np.cumsum(first) - 1], index=prices.index, columns=prices.columns).rank(axis=1, pct=True)
    w = (rank > 0.8).astype(float) - (rank <= 0.2).astype(float)
    return w.div(w.abs().sum(axis=1), axis=0)
'''

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cost = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res = world.exam({f"placebo{i}": PLACEBO.replace("{seed}", str(i)) for i in range(first, first + n)}, alpha=0.05 * n,
                     cost_bps=cost)
    bad = [r for r in res if "pooled_t" not in r]
    if bad:
        print(bad[0]["markets"])
    t = np.array([r["pooled_t"] for r in res if "pooled_t" in r])
    per = np.array([v["alpha_t"] for r in res for v in r["markets"].values() if "error" not in v])
    alpha = np.array([r["pooled_alpha"] for r in res if "pooled_t" in r])
    ok = [r for r in res if "pooled_t" in r]
    for lags in (0, 10, 30, 60):
        tl = np.array([world.newey_west_t(r["pooled"].values, lags)[1] for r in ok])
        print(f"lags {lags:3d}: sd {tl.std(ddof=1):.3f}  share > 1.645 {np.mean(tl > 1.645):.3f}")
    st = np.array([sum(v["alpha_t"] for v in r["markets"].values() if "error" not in v) / np.sqrt(r["n_markets"])
                   for r in ok])
    print(f"stouffer: sd {st.std(ddof=1):.3f}  share > 1.645 {np.mean(st > 1.645):.3f}")
    print(f"verdict t (min): share > 1.645 {np.mean(t > 1.645):.3f}  > 2.326 {np.mean(t > 2.326):.3f}  max {t.max():.2f}")
    print(json.dumps({"n": len(t), "cost_bps": cost, "pooled_t_mean": t.mean(), "pooled_t_sd": t.std(ddof=1),
                      "share_above_1.645": float((t > 1.645).mean()), "max_t": t.max(),
                      "pooled_alpha_mean": alpha.mean(),
                      "per_market_t_mean": per.mean(), "per_market_t_sd": per.std(ddof=1)}, indent=1))
