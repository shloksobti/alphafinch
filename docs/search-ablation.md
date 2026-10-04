# Does search v2 beat v1? (decided before running, 2026-10-04)

The real sealed years (Oct 2023 – Oct 2026) are **not used**. Every run here ignores all data after
2023-09-29 (`--end 2023-09-29`). Its last 3 years (Oct 2020 – Sep 2023) act as a stand-in holdout.
v2 additionally holds out Oct 2017 – Sep 2020 as its validation window, so it breeds on less data.

- v1: the original search (half-Sharpe fitness, eras only, no validation, original prompt).
- v2: alpha fitness, random-halves check, validation-chosen champion, strategy toolkit, richer prompt.

Design: markets {US, India} × seeds {11, 12} × {v1, v2} = 8 runs, each 25 generations × 4 islands,
Claude Code with `--strong-model opus`, the same seed strategies.

Metric: stand-in-holdout alpha t of the final champion and of the team (computed for both, whether
or not they sat the exam), from `scripts/ablation_results.py`.

Decision rule: adopt v2 if its mean alpha t over the 8 (champion, team) results per arm is higher
than v1's. With 4 runs per arm this detects only a large difference; a small gap counts as "no
evidence either way".

## Results (appended after all 8 runs; nothing above was changed)

Stand-in holdout Oct 2020 – Sep 2023. Teams appear only where one formed. Raw rows in
`runs/ablation/results.jsonl` (local), reproduced with `scripts/ablation_results.py`.

| Market | Seed | Role | v1: alpha/yr, t (beta) | v2: alpha/yr, t (beta) |
|---|---|---|---|---|
| India | 11 | champion | −1.5%, −0.35 (0.33) | +2.7%, +1.10 (0.00) |
| India | 11 | team | −0.6%, −0.24 (0.31) | +1.0%, +0.62 (0.15) |
| India | 12 | champion | −0.5%, −0.14 (0.41) | +2.3%, +1.82 (0.00) |
| India | 12 | team | none | +1.8%, +0.97 (0.15) |
| US | 11 | champion | −2.7%, −0.77 (0.75) | +0.0%, +0.03 (0.02) |
| US | 12 | champion | −2.0%, −0.53 (0.67) | −0.4%, −0.16 (−0.17) |
| US | 12 | team | none | +0.5%, +0.23 (0.00) |

Mean alpha t: **v1 −0.41 (0 of 5 positive), v2 +0.66 (6 of 7 positive)**. By the pre-set rule,
v2 is adopted. v2 was better in all 4 matched champion pairs (sign-test p = 1/16). The gain is
large in India and small in the US, where v2 mostly reached zero alpha rather than a clear edge.
v2's champions are close to market-neutral; v1's kept beta 0.3–0.75.
