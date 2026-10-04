"""Download and cache the world market panels (prices only; no strategy is run)."""
from alphafinch import data
from alphafinch.universe import WORLD

if __name__ == "__main__":
    for m in WORLD:
        p = data.load(m)
        print(f"{m:10s} {p.shape[1]:4d} assets  {p.index[0].date()} -> {p.index[-1].date()}  "
              f"{p.sector.nunique():2d} sectors  macro {list(p.macro.columns)}", flush=True)
