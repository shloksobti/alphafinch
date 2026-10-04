"""Stand-in-holdout stats of the champion and team of runs made with --end (docs/search-ablation.md)."""
import json
import sys
from pathlib import Path

import pandas as pd

from alphafinch import data, engine
from alphafinch.lab import Lab

if __name__ == "__main__":
    for run in map(Path, sys.argv[1:]):
        meta = json.loads((run / "meta.json").read_text())
        assert meta.get("end"), f"{run} was not made with --end"
        pop = {p["id"]: p for p in json.loads((run / "population.json").read_text())}
        panel = data.load(meta["market"]).before(pd.Timestamp(meta["end"]) + pd.Timedelta(days=1))
        hold = pd.Timestamp(meta["holdout_start"])
        with Lab(panel, hold, workers=2, timeout=600) as lab:
            for role in ("champion_id", "team_id"):
                p = pop.get(meta.get(role))
                if p is None:
                    continue
                res = lab.run(p["code"], "full", check_leaks=False)
                m = res.returns.index >= hold
                beta, alpha, appraisal, t = engine.alpha_stats(res.returns[m], lab.mkt["full"][m])
                print(json.dumps({"run": run.name, "market": meta["market"], "search": meta["search"],
                                  "seed": meta.get("seed"), "role": role[:-3], "name": p["name"],
                                  "train_fitness": p["fitness"], "alpha": alpha, "alpha_t": t, "beta": beta}))
