"""Forward test: freeze strategies today, judge them on data that does not exist yet.

A sealed holdout can be contaminated (earlier looks, language models that read about those
years). Data from after the freeze date cannot be. `freeze` copies a run's champion and team
into forward/ with the date, git commit, market and the exact asset list. `score` downloads
fresh data and tests each frozen strategy on the days after its freeze date only, using the
same statistic as the sealed exam: the t-stat of daily alpha against the equal-weight
benchmark of the frozen assets, after costs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
from scipy import stats as st

from . import data
from .engine import alpha_stats
from .fitness import verdict_grade
from .lab import Lab

ALPHA = 0.025     # per frozen strategy, one-sided


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def freeze(run_dir: str | Path, out: str | Path = "forward") -> list[dict]:
    run, out = Path(run_dir), Path(out)
    meta = json.loads((run / "meta.json").read_text())
    pop = json.loads((run / "population.json").read_text())
    market = meta["market"]
    panel = data.load(market)
    picks = [("champion", (run / "champion.py").read_text())]
    teams = [p for p in pop if p["op"] == "team" and p["fitness"] is not None]
    if teams:
        picks.append(("team", max(teams, key=lambda p: p["gen"])["code"]))
    out.mkdir(parents=True, exist_ok=True)
    reg_f = out / "registry.json"
    reg = json.loads(reg_f.read_text()) if reg_f.exists() else []
    added = []
    for role, code in picks:
        h = hashlib.sha256(code.encode()).hexdigest()[:10]
        if any(r["sha"] == h and r["market"] == market for r in reg):
            continue
        name = f"{market}-{role}-{h}"
        (out / f"{name}.py").write_text(code)
        entry = {"name": name, "file": f"{name}.py", "sha": h, "market": market, "role": role,
                 "source_run": str(run), "frozen_through": str(panel.index[-1].date()),
                 "assets": [str(c) for c in panel.columns], "git": _git_hash()}
        reg.append(entry)
        added.append(entry)
    reg_f.write_text(json.dumps(reg, indent=1))
    return added


def _subset(p: data.Panel, cols) -> data.Panel:
    cols = [c for c in cols if c in p.columns]
    f = lambda x: None if x is None else x[cols]
    return data.Panel(p.close[cols], f(p.open), f(p.high), f(p.low), f(p.volume),
                      None if p.sector is None else p.sector.reindex(cols), p.macro,
                      {k: v.reindex(columns=cols) for k, v in p.fund.items()})


def score(out: str | Path = "forward", refresh: bool = True) -> list[dict]:
    out = Path(out)
    reg = json.loads((out / "registry.json").read_text())
    panels, rows = {}, []
    for e in reg:
        if e["market"] not in panels:
            panels[e["market"]] = data.load(e["market"], refresh_days=0 if refresh else 1)
        panel = _subset(panels[e["market"]], e["assets"])
        cut = pd.Timestamp(e["frozen_through"])
        n = int((panel.index > cut).sum())
        row = {"name": e["name"], "market": e["market"], "frozen_through": e["frozen_through"], "days": n,
               "missing_assets": len(e["assets"]) - panel.shape[1]}
        if n < 20:
            rows.append({**row, "grade": "WAITING"})
            continue
        with Lab(panel, panel.index[panel.index > cut][0], workers=1, timeout=600) as lab:
            res = lab.run((out / e["file"]).read_text(), "full", check_leaks=False)
            mask = res.returns.index > cut
            beta, alpha, appraisal, t = alpha_stats(res.returns[mask], lab.mkt["full"][mask])
        bar = float(st.t.isf(ALPHA, n - 1))
        r, m = res.returns[mask], lab.mkt["full"][mask]
        rows.append({**row, "return": float((1 + r).prod() - 1), "market_return": float((1 + m).prod() - 1),
                     "alpha": alpha, "alpha_t": t, "bar": bar, "grade": verdict_grade(t, bar)})
    return rows
