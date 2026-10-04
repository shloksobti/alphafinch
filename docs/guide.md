# AlphaFinch guide

Everything AlphaFinch does, in plain language. Run `alphafinch` with no arguments for a one-screen
overview, and `alphafinch <command> -h` for every option of one command.

- [Install](#install)
- [The two choices: market and rules](#the-two-choices-market-and-rules)
- [Commands](#commands)
- [Markets](#markets)
- [Trading rules](#trading-rules)
- [AI providers](#ai-providers)
- [Writing a strategy](#writing-a-strategy)
- [Reading the results](#reading-the-results)
- [The exams](#the-exams)
- [Data notes](#data-notes)

## Install

```bash
pip install "alphafinch[all] @ git+https://github.com/shloksobti/alphafinch"
alphafinch demo          # offline, no AI, no network: checks everything works
```

Python 3.10 or newer. Data is downloaded for free on first use and cached in `~/.alphafinch`.

## The two choices: market and rules

Every command that touches data asks two things:

| | Decides | Example |
|---|---|---|
| **market** | *what* can be traded | `india` = 139 NIFTY 200 stocks, `india-futures` = NSE futures |
| **rules** | *how* it can be traded | `--long-only` = buy only, no shorting |

The market is a plain word after the command. Rules are flags. If you give no rules, strategies may
go long or short (the default), and futures markets automatically get futures rules.

```bash
alphafinch evolve india                       # Indian stocks, long or short
alphafinch evolve india --long-only           # Indian stocks, buy only
alphafinch evolve india-futures               # NSE futures: shorting and leverage allowed
alphafinch evolve us --market-neutral --max-position 3%
```

## Commands

| Command | What it does |
|---|---|
| `alphafinch demo` | A one-minute offline run on a simulated market. No AI, no network. |
| `alphafinch evolve [market]` | The main event: an AI breeds strategies for a few minutes to hours, then the champion sits the sealed exam. Writes a report. |
| `alphafinch backtest FILE [market]` | Test one strategy file on the training years (the sealed years are never touched). |
| `alphafinch holdings FILE [market]` | The portfolio a strategy wants after the latest close: every position, its weight, long or short. |
| `alphafinch world-exam FILE...` | Test strategies, unchanged, on 7 stock markets and pool the evidence. |
| `alphafinch forward freeze runs/<run>` | Freeze a run's champion and team today… |
| `alphafinch forward score` | …and, months later, judge them only on data that didn't exist when they were frozen. |
| `alphafinch replay runs/<run>` | Re-watch a finished run as a short animated story. |
| `alphafinch markets` | List every market. |

### Common `evolve` options

| Option | Default | Meaning |
|---|---|---|
| `--generations 20` | 20 | How long to breed. 25 × 4 islands takes roughly 20–40 minutes with Claude Code. |
| `--provider auto` | auto | Which AI writes strategies (see [AI providers](#ai-providers)). |
| `--strong-model opus` | none | Use a bigger model for crossovers and brand-new ideas. |
| `--exam-budget 10` | 10 | How many times the sealed exam may be sat in this run. |
| `--holdout-years 3` | 3 | How many recent years are sealed away. |
| `--end 2023-09-29` | none | Pretend the data stops on this date, for research on past periods that leaves recent years untouched. |

## Markets

| Market | What's in it | Since |
|---|---|---|
| `us` | S&P 500 stocks (about 430 with full history), sectors, US macro data, SEC fundamentals | 2010 |
| `india` | NIFTY 200 stocks (about 140 with full history), sectors, Indian macro data | 2010 |
| `india-futures` | NSE futures: NIFTY and BANKNIFTY index futures plus every F&O stock (about 145) | 2012 |
| `futures` | 39 global futures: stock indices, government bonds, currencies, energy, metals, agriculture | 2012 |
| `uk` `europe` `japan` `hongkong` `australia` `canada` `korea` | FTSE 100, Eurozone large caps, Nikkei 225, Hang Seng, ASX 200, TSX 60, KOSPI 200 | 2010 |
| `us30` | 30 US mega-caps | 2008 |
| `crypto` | 15 large coins against USDT (Binance) | 2020 |
| `industries` | 49 US industry portfolios (Ken French), free of survivorship bias | 1970 |
| `synthetic` | A simulated market, offline | — |

Use your own list with `--tickers RELIANCE.NS,TCS.NS,INFY.NS` (any Yahoo symbols).

## Trading rules

Rules are enforced by the engine. Whatever a strategy asks for, its positions are adjusted to fit
the rules every day before anything is simulated, so a strategy can't break them and still look
good. The AI is told the rules too, so it designs within them.

| Flag | Rule |
|---|---|
| *(none)* | Long or short, total exposure up to 1× capital |
| `--long-only` | Buy only: no shorting and no leverage. The cash / spot market, and the realistic choice for most retail investors (in India, cash-market shorts can't be held overnight) |
| `--market-neutral` | Longs and shorts roughly equal (net exposure within ±10%), with a 0.5%/yr fee on short positions |
| `--max-position 5%` | No single position above 5% of capital (combines with any of the above) |
| `--leverage 2` | Total exposure (longs + shorts) up to 2× capital |

Futures markets (`futures`, `india-futures`) use futures rules automatically: shorting is as easy
as buying, and total exposure may reach 3× (margin). Add `--long-only` to forbid shorts there too.

## AI providers

| `--provider` | What you need |
|---|---|
| `claude-code` | [Claude Code](https://claude.com/claude-code) installed; no API key |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY`; pick a model with `--model` |
| `ollama` | a local model running at `localhost:11434` |
| `compatible` | any OpenAI-compatible server: `--base-url … --model …` |
| `none` | nothing: parameter tweaks and blends only, fully offline |

`auto` (the default) uses the first one it finds.

## Writing a strategy

A strategy is one Python function that returns how much of your capital to hold in each asset, every
day.

```python
def strategy(prices, data):
    """Sector Spread: sector-neutral 6-month momentum. Hypothesis: news diffuses slowly within industries."""
    score = tk.neutralize(tk.zscore(prices.pct_change(126)), data.sector)
    return tk.rebalance(tk.long_short(score, q=0.2), every="M")
```

- **Input.** `prices` is a table of daily closing prices (dates × assets). `data` also has `open`,
  `high`, `low`, `volume`, `sector`, `macro` (VIX, index, rates, oil, gold…) and, for `us`, `fund`
  (point-in-time fundamentals).
- **Output.** A table of weights with the same shape: `0.02` means 2% of capital long, `-0.02` means
  2% short. Row *t* is held from the close of day *t* to the close of day *t+1*.
- **No peeking.** Row *t* may only use information up to day *t*. Every strategy is re-run on cut-off
  histories, and any that changes its past decisions when future data is removed is rejected.
- **No hindsight by name.** Strategies may not hard-code tickers or sector names (on the futures
  market, asset classes like `"rates"` are fine).
- **Allowed imports:** `numpy`, `pandas`, `math`. The toolkit `tk` is available without importing.

The toolkit (`tk`):

| Function | Does |
|---|---|
| `tk.rank(x)`, `tk.zscore(x)`, `tk.winsorize(x)`, `tk.demean(x)` | Compare assets with each other on each day |
| `tk.neutralize(x, data.sector)` | Remove sector bets, keeping only within-sector differences |
| `tk.long_short(score, q=0.2)` | Long the top 20%, short the bottom 20%, equal weight |
| `tk.long_only(score, q=0.2)` | Hold the top 20% |
| `tk.proportional(score)`, `tk.cap(w, 0.05)` | Weights in proportion to a score; cap positions |
| `tk.rebalance(w, every="M")` | Only trade on the first day of each week, month or quarter |
| `tk.vol_target(w, prices, target=0.10)` | Scale the portfolio to about 10% annual volatility |
| `tk.rolling_beta(prices)`, `tk.residual(prices)` | Each asset's market sensitivity; returns with the market removed |
| `tk.ts_zscore(x)`, `tk.smooth(x, halflife)`, `tk.drawdown(prices)` | Unusualness vs own history; smoothing; distance from the peak |

```bash
alphafinch backtest my_strategy.py india
alphafinch holdings my_strategy.py india --long-only
```

## Reading the results

Each `evolve` run writes a folder `runs/<timestamp>/`:

| File | Contents |
|---|---|
| `report.html` | The morning report: champion, exam verdict, equity curves, team, lab notebook, family tree |
| `champion.py` | The champion's code, ready for `backtest`, `holdings` or `world-exam` |
| `team.py` | A diversified team of survivors, if one formed |
| `population.json` | Every strategy bred, with its scores |

The terms that matter:

- **Alpha:** return beyond what simply being exposed to the market explains. Owning the market earns none.
- **t (t-statistic):** how sure we can be that the alpha isn't luck. About 2 is "probably real" for a
  single test. The bar rises with the number of attempts.
- **PASS / PROMISING / FAIL:** PASS cleared the bar. PROMISING has t above 1 but below the bar (the
  report says how many years of data a pass would need). FAIL shows no evidence of an edge.
- **Training numbers** are what evolution optimised. Never read them as expected returns.

## The exams

1. **Choosing the champion.** The last three training years are held back from breeding. The top
   finalists are scored once on them, and the best becomes the champion. The AI never sees those scores.
2. **The sealed exam.** The most recent three years, never seen by evolution or the AI. Each attempt
   reveals only PASS or FAIL, and the bar accounts for every attempt.
3. **The world exam.** A frozen strategy runs unchanged on 7 other stock markets. The pooled verdict
   uses the lower of two statistics, and was calibrated with 200 random placebo strategies (none passed).
4. **The forward test.** Freeze today, score in six months on data no one has seen.

Don't re-run until something passes: that turns the exam back into training data. If you've already
looked at a period, count it (`--alpha`, `--prior-looks`) or test on new markets and new data.

## Data notes

- **Survivorship.** Stock universes are today's index members, so absolute returns look better than
  they were in real time. Alpha is measured against the same list, which limits the bias but doesn't
  remove it. `industries` is free of it.
- **Futures.** Free continuous futures prices fake big gains or losses at every contract roll. AlphaFinch
  builds `futures` from funds that hold and roll the real contracts, and `india-futures` from each
  stock's total return. Both are converted to excess returns over the short-term interest rate, which
  is what a futures position earns. The F&O list is today's, not historical.
- **Bad data.** A move of 80% or more that fully reverses within 5 days is treated as a broken split
  adjustment and repaired. Each repair is printed.
- **Fundamentals (US).** From SEC filings, available from the day after each 10-K. The SEC requires a
  contact: `export ALPHAFINCH_SEC_CONTACT="Your Name you@example.com"`.
- **Costs.** 5 bps per unit of turnover by default. Borrow fees apply to shorts under `--market-neutral`.

Research and educational software, not investment advice.
