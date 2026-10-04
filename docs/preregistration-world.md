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
