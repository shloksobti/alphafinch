"""The morning report: a single self-contained HTML file plus machine-readable outputs."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import pandas as pd

from .sandbox import StrategyError
from .ui import OP_ICON

CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e4e3df;--card:#ffffff;--acc:#2a78d6;--ok:#008300;--warn:#c98500}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#33322f;--card:#222220;--acc:#3987e5;--ok:#3fb950;--warn:#e3b341}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
.sub{color:var(--mut);margin:0 0 24px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto;font-size:13px}
.pass{color:var(--ok);font-weight:600}.warn{color:var(--acc);font-weight:600}.fail{color:var(--warn);font-weight:600}img{max-width:100%;border-radius:8px;border:1px solid var(--line)}
.tree{font-size:14px}.tree li{margin:4px 0}.note{color:var(--mut);font-size:13px}
"""


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _chart(curves: dict, title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for (k, s), c in zip(curves.items(), colors):
        ax.plot(s.index, s.values, label=k, color=c, lw=1.6)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_yscale("log")
    out = _png(fig)
    plt.close(fig)
    return out


def _team_html(evo) -> str:
    t = evo.team
    if t is None:
        return ""
    members = "".join(f"<li>{html.escape(evo.all[m].name)} <span class=note>fitness {evo.all[m].fitness:+.2f}</span></li>"
                      for m in t.parents if m in evo.all)
    return (f"<h2>The team</h2><p>{len(t.parents)} diverse survivors held together, weighted by inverse volatility "
            f"and chosen on training data only. Training fitness {t.fitness:+.2f}, Sharpe {t.stats.sharpe:.2f}, "
            f"alpha {t.stats.alpha:+.1%}/yr. Exam: <b>{t.exam or 'not taken'}</b>.</p><ul class=tree>{members}</ul>")


def _notebook_html(evo) -> str:
    ok = [e for e in evo.notebook if e["fitness"] is not None]
    if not ok:
        return ""
    best = sorted(ok, key=lambda e: -e["fitness"])[:12]
    rows = "".join(f"<tr><td>{html.escape(e['name'])}</td><td>{html.escape(e['idea'])}"
                   f"<br><span class=note>{html.escape(e['hypothesis'])}</span></td>"
                   f"<td class=n>{e['fitness']:+.2f}</td><td>{html.escape(e['why'])}</td></tr>" for e in best)
    return (f"<h2>Lab notebook: the AI's best ideas</h2><p class=note>{len(evo.notebook)} AI-bred ideas were recorded; "
            f"the strongest are below with the AI's own hypothesis.</p><table><tr><th>Strategy</th><th>Idea and hypothesis"
            f"</th><th class=n>Fitness</th><th>Weakness</th></tr>{rows}</table>")


def _tree_html(evo, ind, depth=0, seen=None) -> str:
    seen = seen or set()
    if ind is None or ind.id in seen or depth > 8:
        return ""
    seen.add(ind.id)
    kids = "".join(_tree_html(evo, evo.all.get(p), depth + 1, seen) for p in ind.parents)
    label = (f"{OP_ICON.get(ind.op, '•')} <b>{html.escape(ind.name)}</b> "
             f"<span class=note>{ind.op}, generation {ind.gen}, fitness {ind.fitness:+.2f}</span>")
    return f"<li>{label}{'<ul>' + kids + '</ul>' if kids else ''}</li>"


GRADE = {"PASS": ("✅ PASS", "pass"), "PROMISING": ("🤔 PROMISING", "warn"), "FAIL": ("❌ FAIL", "fail")}


def write(evo, out_dir: Path, market_label: str, holdout_start) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    champ = evo.champion()
    pop = sorted([p for isl in evo.islands for p in isl], key=lambda p: -p.fitness)
    uniq, names = [], set()
    for p in pop:
        if p.code not in names:
            names.add(p.code)
            uniq.append(p)
    top = uniq[:10]

    # training equity curves of the top 3 vs equal-weight buy & hold
    curves = {}
    for p in ([evo.team] if evo.team is not None else []) + top[:3]:
        if p.returns is not None:
            curves[p.name] = (1 + p.returns).cumprod()
    curves["Buy & hold (equal weight)"] = (1 + evo.lab.mkt["train"]).cumprod()
    train_png = _chart(curves, "Training period (what evolution could see)")

    # exam results; holdout stats revealed only now that evolution is over
    exam_rows, hold_curves = [], {}
    for a in evo.exam.attempts:
        ind = evo.all[a["id"]]
        rev = evo.exam.reveal(ind.id, ind.code)
        exam_rows.append((ind, a["verdict"], rev))
        hold_curves[f"{ind.name} ({a['verdict']})"] = rev["equity"]
    if hold_curves:
        hold_curves["Buy & hold (equal weight)"] = exam_rows[-1][2]["market_equity"]
    hold_png = _chart(dict(list(hold_curves.items())[-5:]), "Sealed holdout (revealed after evolution ended)") if hold_curves else None

    s = champ.stats
    passed = [r for r in exam_rows if r[1] == "PASS"]
    verdict = (f"<span class=pass>{len(passed)} strateg{'y' if len(passed) == 1 else 'ies'} passed the sealed exam.</span>"
               if passed else "<span class=fail>No strategy passed the sealed exam.</span> That is the honest answer "
               "when no edge survives out of sample.")
    rows = "".join(
        f"<tr><td>{OP_ICON.get(p.op, '•')} {html.escape(p.name)}</td><td>{p.op}</td><td class=n>{p.gen}</td>"
        f"<td class=n>{p.fitness:+.2f}</td><td class=n>{p.stats.sharpe:.2f}</td><td class=n>{p.stats.alpha:+.1%}</td><td class=n>{p.stats.cagr:+.1%}</td>"
        f"<td class=n>{p.stats.max_dd:.0%}</td><td class=n>{p.stats.turnover:.1f}x</td></tr>" for p in top)
    def _yn(r):
        y = r["years_needed"]
        return "never (no edge)" if y is None else (f"{y:.0f} years" if y < 200 else "200+ years")
    erows = "".join(
        f"<tr><td>{html.escape(i.name)}</td><td class={GRADE[r['grade']][1]}>{GRADE[r['grade']][0]}</td>"
        f"<td class=n>{r['alpha']:+.1%}</td><td class=n>{r['alpha_t']:.2f}</td><td class=n>{r['cagr']:+.1%}</td>"
        f"<td class=n>{r['market_cagr']:+.1%}</td><td class=n>{_yn(r)}</td></tr>"
        for i, v, r in exam_rows)
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AlphaFinch report</title><style>{CSS}</style></head><body><main>
<h1>🐦 AlphaFinch morning report</h1>
<p class=sub>{html.escape(market_label)} · {len(evo.all)} strategies bred over {evo.gen} generations on {len(evo.islands)} islands · {evo.llm_calls} AI calls · training data to {(pd.Timestamp(holdout_start) - pd.Timedelta(days=1)).date()}, sealed holdout from {pd.Timestamp(holdout_start).date()}</p>
<p>{verdict}</p>
<div class=cards>
<div class=card><span>Champion</span><b>{html.escape(champ.name)}</b></div>
<div class=card><span>Training Sharpe</span><b>{s.sharpe:.2f}</b></div>
<div class=card><span>Training alpha / yr</span><b>{s.alpha:+.1%}</b></div>
<div class=card><span>Max drawdown</span><b>{s.max_dd:.0%}</b></div>
<div class=card><span>Exam bar (t-stat)</span><b>{evo.exam.bar:.2f}</b></div>
</div>
<h2>Sealed exam</h2>
<p class=note>Champions only ever learned PASS or FAIL. To pass, a strategy's holdout <b>alpha</b> (return beyond what its exposure to the equal-weight market explains) needed a t-statistic above a bar that accounts for every exam attempt in the run ({evo.exam.budget} allowed), so a pass cannot be explained by repeated tries. Full holdout numbers are shown here only because evolution has ended. Do not use them to keep tuning.</p>
<table><tr><th>Strategy</th><th>Verdict</th><th class=n>Sealed alpha/yr</th><th class=n>Alpha t-stat</th><th class=n>Return/yr</th><th class=n>Buy-everything/yr</th><th class=n>Data needed to prove it</th></tr>{erows or '<tr><td colspan=7>No exam attempts.</td></tr>'}</table>
<p class=note>✅ PASS: alpha t-statistic above the bar. 🤔 PROMISING: positive alpha with t above 1, real-looking but unproven. ❌ FAIL: no evidence of an edge. "Data needed" estimates how many years of out-of-sample data an edge of this size would need to clear the bar.</p>
{f'<p><img src="{hold_png}" alt="Holdout equity curves"></p>' if hold_png else ''}
<h2>Leaderboard (training period)</h2>
<table><tr><th>Strategy</th><th>Born by</th><th class=n>Gen</th><th class=n>Fitness</th><th class=n>Sharpe</th><th class=n>Alpha/yr</th><th class=n>CAGR</th><th class=n>Max DD</th><th class=n>Turnover</th></tr>{rows}</table>
{f'<p><img src="{train_png}" alt="Training equity curves"></p>' if train_png else ''}
{_team_html(evo)}
{_notebook_html(evo)}
<h2>Family tree of the champion</h2>
<ul class=tree>{_tree_html(evo, champ)}</ul>
<h2>Champion code</h2>
<pre>{html.escape(champ.code)}</pre>
<p class=note>Research software, not investment advice. Backtests ignore taxes, slippage beyond 5 bps per unit turnover, borrow costs and capacity.</p>
</main></body></html>"""
    meta = {"market_label": market_label, "holdout_start": str(pd.Timestamp(holdout_start).date()),
            "train_start": str(evo.lab.train.index[0].date()), "generations": evo.gen, "islands": len(evo.islands),
            "provider": evo.llm.name if evo.llm else "none", "model": getattr(evo.llm, "model", None),
            "exam_budget": evo.exam.budget, "exam_bar": evo.exam.bar,
            "exam": [{"id": a["id"], "verdict": a["verdict"]} for a in evo.exam.attempts]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    path = out_dir / "report.html"
    path.write_text(doc)
    (out_dir / "champion.py").write_text(champ.code)
    if evo.team is not None:
        (out_dir / "team.py").write_text(evo.team.code)
    (out_dir / "population.json").write_text(json.dumps([
        {"id": p.id, "name": p.name, "op": p.op, "parents": list(p.parents), "gen": p.gen, "island": p.island,
         "fitness": None if p.stats is None else p.fitness, "raw_fitness": p.raw_fitness, "robust": p.robust,
         "error": p.error, "exam": p.exam, "code": p.code,
         "train": None if p.stats is None else {"sharpe": p.stats.sharpe, "cagr": p.stats.cagr, "max_dd": p.stats.max_dd,
                                                "turnover": p.stats.turnover, "beta": p.stats.beta,
                                                "alpha": p.stats.alpha, "eras": p.stats.eras}}
        for p in evo.all.values()], indent=1))
    return path
