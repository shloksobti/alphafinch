"""Append the pre-registered runs' full exam results to docs/preregistration.md."""
import json

import pandas as pd

from alphafinch import data, fitness

RUNS = [("runs/prereg/20261004-003032", "industries", "A (industries)"),
        ("runs/prereg/20261004-003920", "india", "B (India)")]

if __name__ == "__main__":
    rows, summ = [], []
    for d, market, label in RUNS:
        P = json.load(open(f"{d}/population.json"))
        meta = json.load(open(f"{d}/meta.json"))
        px = data.load(market)
        ex = fitness.SealedExam(px, pd.Timestamp(meta["holdout_start"]), budget=5, alpha=0.025)
        by = {p["id"]: p for p in P}
        for e in meta["exam"]:
            p = by[e["id"]]
            r = ex.reveal(p["id"], p["code"])
            rows.append(f"| {label} | {p['name']} | {p['op']} | {p['gen']} | {e['verdict']} | "
                        f"{r['alpha']:+.1%} | {r['alpha_t']:.2f} | {r['cagr']:+.1%} |")
        n_ai = sum(1 for p in P if p["op"] in ("mutate", "crossover", "immigrant"))
        summ.append(f"- Run {label}: {len(P)} strategies bred ({n_ai} by the AI); exam bar {meta['exam_bar']:.2f}; "
                    f"{len(meta['exam'])} exam attempts, 0 passed.")
    txt = ("\n\n## Results (added after both runs finished, 2026-10-04)\n\n"
           "**No strategy passed in either run.** Every exam attempt is listed; sealed-period numbers were "
           "computed only after each run ended.\n\n"
           "| Run | Champion examined | Born by | Gen | Verdict | Sealed alpha/yr | Alpha t | Sealed return/yr |\n"
           "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n" + "\n".join(summ) +
           "\n\nRun B made 3 of its 5 allowed attempts: a new champion must clearly beat the last one examined "
           "before it may sit the exam.\n")
    open("docs/preregistration.md", "a").write(txt)
    print(txt)
