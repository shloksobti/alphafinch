# Pre-registration v2: the upgraded search on US and Indian stocks

Written and committed on 2026-10-04, **before** either run started. Runs A and B
(`preregistration.md`) both failed. This round tests the upgraded system: richer data, robust
fitness across four eras, a lab notebook, a strong model for crossovers and new ideas, teams,
and a ban on hard-coded asset and sector names.

## Runs

| | Run C (US) | Run D (India) |
|---|---|---|
| Universe | S&P 500 members with ≥ 95% history: 426 stocks | NIFTY 200 members with ≥ 95% history: 139 stocks |
| Extra data | volume, OHLC, 11 sectors, 9 macro series, SEC fundamentals (point-in-time, +1 day after filing) | volume, OHLC, 18 sectors, 7 macro series |
| Training | 2010-08-06 → 2023-09-29 | 2010-10-27 → 2023-09-29 |
| Sealed holdout | 2023-10-02 → 2026-10-02 (3 years) | 2023-10-03 → 2026-10-01 (3 years) |
| Search | 25 generations × 4 islands × 8, Claude Code (default model) + Opus for crossovers and immigrants | same |
| Exam budget (this run) | 5 | 5 |

```
ALPHAFINCH_SEC_CONTACT="<name> <email>" alphafinch evolve --market us --holdout-years 3 \
  --exam-budget 5 --alpha 0.00625 --generations 25 --provider claude-code --strong-model opus \
  --seed 2 --out runs/prereg2
alphafinch evolve --market india --holdout-years 3 --exam-budget 5 --alpha 0.0125 \
  --generations 25 --provider claude-code --strong-model opus --seed 2 --out runs/prereg2
```

## The bar, counting earlier looks

These sealed periods have been looked at before, during development. Every earlier exam
attempt counts against the budget, so the bar is set as if all of them were part of this test.

| | Earlier attempts on this holdout | This run | Total K | Per-attempt α (0.025 / K) | Bar on alpha t |
|---|---|---|---|---|---|
| US | 15 (10 recorded + up to 5 unrecorded, rounded up) | 5 | 20 | 0.00125 | **3.03** |
| India | 5 (2 dev runs + 3 from run B, which overlapped) | 5 | 10 | 0.0025 | **2.82** |

`--alpha` is the family rate over the 5 new attempts (5 × per-attempt α). Each market gets
α = 0.025, so 0.05 in total across both.

The test statistic is the t-stat of daily alpha against the equal-weight benchmark of the same
universe, over the sealed holdout, after 5 bps costs.

## Graded verdict (decided now)

- **PASS:** alpha t ≥ the bar above. Something real was found.
- **PROMISING:** 1 < t < bar. Report the years of data a pass would need, (bar / appraisal)².
- **FAIL:** t ≤ 1.

We report every attempt for both markets, whatever the outcome.

## Known threats, disclosed up front

1. **AI hindsight.** The language models were trained on text up to 2025–2026, which overlaps
   the sealed holdout. They may "know" which stocks, sectors or styles did well. Mitigations:
   hard-coded tickers and sector names are rejected by the sandbox, and the AI only ever sees
   training-period results. Even so, a style-level leak (for example "AI-related momentum
   works") is still possible. A pass here is therefore **not** proof on its own.
2. **Survivorship.** The universes are today's index members. Within the universe, this
   affects the strategy and the benchmark alike, but it is not a clean historical universe.
3. **Earlier looks.** Counted in the bar above, with uncertainty about the unrecorded US runs
   resolved conservatively.

## Forward test (the gold standard)

Whatever happens, the final champion and team from each run will be frozen (code + git hash)
in `forward/` and scored on data after 2026-10-02, which no model has seen. The forward
verdict uses the same statistic with K = 1 per frozen strategy, first checked at 6 months and
again at 12 months.
