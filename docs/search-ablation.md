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
