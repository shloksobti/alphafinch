"""Graded sealed-exam statistics for every exam attempt of a run (docs/preregistration-v2.md)."""
import json
import sys
from pathlib import Path

import pandas as pd

from alphafinch import data, fitness
from alphafinch.lab import Lab

if __name__ == "__main__":
    run = Path(sys.argv[1])
    meta = json.loads((run / "meta.json").read_text())
    pop = {p["id"]: p for p in json.loads((run / "population.json").read_text())}
    panel = data.load(meta["market"])
    with Lab(panel, pd.Timestamp(meta["holdout_start"]), workers=2, timeout=600) as lab:
        ex = fitness.SealedExam(lab, budget=meta["exam_budget"], alpha=meta["alpha"])
        for a in meta["exam"]:
            p = pop[a["id"]]
            r = ex.reveal(p["id"], p["code"])
            print(f"{p['name'][:42]:42s} gen {p['gen']:2d}  train fit {p['fitness']:+.2f}  sealed alpha {r['alpha']:+.1%}  "
                  f"t {r['alpha_t']:+.2f}  beta {r['beta']:.2f}  CAGR {r['cagr']:+.1%} vs EW {r['market_cagr']:+.1%}  "
                  f"{r['grade']}  (data needed {r['years_needed']:.0f}y)" if r['years_needed'] < 1e4 else f"{r['grade']}")
