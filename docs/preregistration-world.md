# Pre-registration: the world exam (run E)

Written and committed on 2026-10-04, **before** any candidate strategy was run on any of these
markets. Only placebo strategies (random portfolios, `scripts/world_placebo.py`) have touched
this data, to calibrate the test.

## Question

Do the strategies bred by search v2 earn alpha in stock markets they have never seen?

## Candidates (K = 5), frozen in `forward/world/`

Every champion and team from the four search-v2 comparison runs (`docs/search-ablation.md`),
chosen by role, not by result. Each was bred only on data before 2017-09-29 and has never seen
a non-US, non-Indian stock.

| File | Strategy | Bred on |
|---|---|---|
| `india-s11-champion-4fc2f68fea.py` | Tethered Twin Snapback | NIFTY 200, 2010–2017 |
| `india-s11-team-275643cc57.py` | The Team | NIFTY 200, 2010–2017 |
| `india-s12-champion-436043f9aa.py` | Quiet Intraday Relay v2 | NIFTY 200, 2010–2017 |
| `india-s12-team-6d7925a18b.py` | The Team | NIFTY 200, 2010–2017 |
| `us-s12-champion-ebe31d7c2f.py` | Quiet Sector Tether v2 | S&P 500, 2010–2017 |

Excluded by rule, before any result: two US strategies that need SEC fundamentals
(`data.fund`), which only exist for the US.

## Markets and window

UK (FTSE 100), Eurozone (DAX, CAC 40, IBEX 35, FTSE MIB, AEX), Japan (Nikkei 225), Hong Kong
(Hang Seng), Australia (S&P/ASX 200), Canada (S&P/TSX 60) and Korea (KOSPI 200): 814 stocks with
95% price history since 2010, from today's index members (`alphafinch/universes/`). In these
markets `data.macro["vix"]` is the US VIX.

**Primary window: 2017-10-02 → 2026-10-02** (about 9 years × 7 markets).
Secondary (reported, not used for the verdict): 2020-10-01 → 2026-10-02, which is after the
validation window the champions were chosen on.

## Test

```
alphafinch world-exam forward/world/*.py --start 2017-10-02 --alpha 0.05 --json runs/world/primary.json
alphafinch world-exam forward/world/*.py --start 2020-10-01 --alpha 0.05 --json runs/world/secondary.json
```

- In each market: alpha against that market's equal-weight benchmark, after 5 bps costs.
- Pooled: the lower of (a) the Newey–West t of the daily cross-market average of active returns
  and (b) the Stouffer combination of the per-market alpha t's.
- Bar: t^{-1}(0.05 / 5) ≈ 2.33. **PASS** if pooled t > 2.33; **PROMISING** if 1 < t ≤ 2.33;
  **FAIL** otherwise.

## Calibration (done before this registration, placebos only)

200 random long/short placebos without costs: Newey–West t spread 1.00, Stouffer 0.94; the verdict
statistic (the lower of the two) exceeded 1.645 for 3.5% and the 2.33 bar for 0 of 200. With
costs, placebos lost 0.9%/yr and none came close to passing.

## Known threats, disclosed up front

1. **AI hindsight.** The strategies' code was written by language models that have read about
   2017–2025 markets. No asset or sector names are allowed, and the AI only saw Indian or US
   training results before 2017, but a style-level leak is possible.
2. **Time overlap with selection.** The champions were chosen on Indian (or US) data from
   2017-10 to 2020-09, which overlaps the primary window in time, though not in market. Global
   factor returns are correlated, so the secondary window (2020-10 onward) is reported as a
   cleaner check.
3. **Survivorship.** Today's index members. Alpha is measured against the same list.
4. **Currencies.** Returns are in local currency; each market's benchmark is in the same
   currency, so alpha is currency-neutral.

All five results are reported whatever they show.

## Results (appended 2026-10-04, after the run; nothing above was changed)

**Primary window (2017-10-02 → 2026-10-02), bar 2.33: 1 of 5 passed.**

| Strategy | Pooled alpha/yr | Verdict t (NW / Stouffer) | Markets positive | Grade |
|---|---|---|---|---|
| **Quiet Sector Tether v2** (US-bred) | **+2.9%** | **3.40** (3.40 / 4.30) | **7 / 7** | **PASS** |
| The Team (India s12) | +2.3% | 2.14 (2.14 / 3.70) | 6 / 7 | PROMISING |
| Quiet Intraday Relay v2 (India s12) | +0.9% | 1.03 (1.03 / 1.89) | 5 / 7 | PROMISING |
| The Team (India s11) | −0.6% | −0.79 | 4 / 7 | FAIL |
| Tethered Twin Snapback (India s11) | −2.6% | −3.09 | 0 / 7 | FAIL |

Per-market alpha of the passing strategy: UK +1.8%, Eurozone +3.0%, Japan +1.4%, Hong Kong +2.7%,
Australia +4.6%, Canada +4.4%, Korea +1.6% a year. Its beta is slightly negative (−0.09 to −0.24).
Notably, it had **failed** its stand-in test in the US (2020–23, t −0.16): it was not chosen for
looking good.

**Secondary window (2020-10-01 → 2026-10-02):** Quiet Sector Tether v2 +2.2%/yr, t 2.17,
positive in 7 / 7 markets: PROMISING, just under the bar. The four others FAIL.

### What the passing strategy does

Within each sector, it buys the stocks that move least with the market (correlation purged of
volatility) and shorts those that move most, sized by inverse volatility and rebalanced monthly.
This closely resembles **"betting against correlation"** (Asness, Frazzini, Gormsen and Pedersen,
*Journal of Financial Economics*, 2020), whose evidence ends before this window. The AI arrived
at it from US data before 2017; the world exam is an out-of-sample test of that idea in seven
other markets.

### Stress tests (run after the verdict, `scripts/world_stress.py`, `scripts/world_factors.py`)

| Test | Pooled alpha/yr | t |
|---|---|---|
| As registered (5 bps per unit turnover) | +2.9% | 3.40 |
| 15 bps | +2.6% | 2.94 |
| 30 bps | +2.0% | 2.26 |
| 50 bps | +1.2% | 1.36 |
| Beta hedged with a trailing (causal) 252-day beta | +2.8% | 3.22 |
| Monthly, after the Fama–French developed-ex-US 5 factors + momentum | +2.5% | 2.64 (R² 0.13) |

Turnover is 2–5× a year. The alpha comes mostly from the short leg (high co-movers lagged the
benchmark by 3.2%/yr). By calendar year: positive in 8 of 10 (2022 −1.5%, 2026 to date −0.4%).

### Caveats

- **Shorting.** Borrow fees (roughly 0.25–1%/yr on the short half) are not modelled. Korea banned
  most short selling from March 2020 to May 2021 and all of it from November 2023 to March 2025,
  so the Korean leg was not fully implementable then.
- **Costs.** The edge survives 15–30 bps per unit of turnover but not 50; UK and Hong Kong stamp
  duties sit inside that range for large caps.
- **Known idea.** This is most likely an AI rediscovery of a published anomaly, confirmed out of
  sample, not a new one.
- **Hindsight and survivorship** as disclosed above.
