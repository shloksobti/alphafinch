# Pre-registered runs: can AI-evolved strategies pass the sealed exam?

**Committed before either run started** (see this file's git commit time). Whatever the results,
both runs are reported in full. No other runs on these holdouts will be made or reported.

## Question
Can strategies bred by an AI (Claude Sonnet through Claude Code) beat their market in years
they never saw, by more than luck can explain?

## Runs

| | Run A | Run B |
|---|---|---|
| Market | `industries`: 49 US industry portfolios (Ken French, CRSP 2026-08 vintage) | `india`: 25 NSE large caps (Yahoo Finance, downloaded 2026-10-04) |
| Training period | 1970-01-02 → 2016-08-30 | 2008-01-01 → 2021-09-30 |
| Sealed period | 2016-08-31 → 2026-08-31 (10 years) | 2021-10-01 → 2026-10-01 (5 years) |
| Exam attempts | 5 | 5 |
| Exam α | 0.025 (so the two runs together have a 5% false-PASS rate) | 0.025 |
| Exam bar | alpha t-statistic > 2.58 | alpha t-statistic > 2.58 |

Exact commands (AlphaFinch 0.1.0 at this commit):

```bash
alphafinch evolve --market industries --holdout-years 10 --exam-budget 5 --alpha 0.025 \
  --generations 25 --provider claude-code --seed 1 --out runs/prereg
alphafinch evolve --market india --holdout-years 5 --exam-budget 5 --alpha 0.025 \
  --generations 25 --provider claude-code --seed 1 --out runs/prereg
```

All other settings are defaults (4 islands of 8, 4 offspring per island per generation,
migration every 4 generations, Claude Sonnet via `claude -p`).

## Decision rule
- A strategy **passes** if its alpha t-statistic over the sealed period, measured against the
  equal-weight benchmark of the same universe, exceeds the bar. The exam returns only PASS/FAIL
  during the run.
- Any PASS is reported as a pre-registered result. If both runs end with no PASS, we report that.
- Sealed-period statistics are inspected only after each run has finished.


## Results (added after both runs finished, 2026-10-04)

**No strategy passed in either run.** Every exam attempt is listed; sealed-period numbers were computed only after each run ended.

| Run | Champion examined | Born by | Gen | Verdict | Sealed alpha/yr | Alpha t | Sealed return/yr |
|---|---|---|---|---|---|---|---|
| A (industries) | Trend Crown | crossover | 1 | FAIL | -0.2% | -0.05 | +4.3% |
| A (industries) | Residual Drift | mutate | 3 | FAIL | +0.1% | 0.08 | +5.8% |
| A (industries) | Hedged Momentum Core | crossover | 17 | FAIL | -0.1% | -0.06 | +4.2% |
| A (industries) | Residual Drift v23 | tweak | 21 | FAIL | +0.0% | 0.04 | +3.1% |
| A (industries) | Guarded Residual Tide v4 | tweak | 25 | FAIL | +1.5% | 0.91 | +4.7% |
| B (India) | Calm Momentum | crossover | 1 | FAIL | -2.6% | -0.69 | +6.4% |
| B (India) | Steady Crown v3 | tweak | 2 | FAIL | +2.2% | 0.68 | +10.4% |
| B (India) | Smooth Crown | migrant | 12 | FAIL | -2.0% | -0.55 | +5.2% |

- Run A (industries): 448 strategies bred (286 by the AI); exam bar 2.58; 5 exam attempts, 0 passed.
- Run B (India): 448 strategies bred (286 by the AI); exam bar 2.58; 3 exam attempts, 0 passed.

Run B made 3 of its 5 allowed attempts: a new champion must clearly beat the last one examined before it may sit the exam.
