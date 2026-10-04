---
name: alphafinch
description: Evolve, backtest and honestly evaluate trading strategies with AlphaFinch. Use when the user wants to discover, breed, improve or test trading strategies on stocks (US, India, UK, Europe, Japan and more), futures (global or NSE), crypto or industry portfolios, wants to know what a strategy holds today, or asks whether a backtest is overfit.
---

# AlphaFinch in Claude Code

AlphaFinch evolves trading strategies with an AI (islands, mutation, crossover) and scores the
winners on a sealed holdout that they never see.

## Run an evolution
Use the Claude Code provider so the user's existing subscription does the breeding:

```bash
pip install "alphafinch[all]"
alphafinch evolve india --provider claude-code --generations 20
```

The market is a plain word (`alphafinch markets` lists them: `us`, `india`, `india-futures`,
`futures`, `uk`, `europe`, `japan`, … `crypto`, `industries`, `synthetic`). Rules are flags:
`--long-only` (no shorting), `--market-neutral`, `--max-position 5%`, `--leverage 2`. Long runs are
fine to leave running; the live view shows progress. Full reference: `docs/guide.md`.

## Read the results
The run prints a report path (`runs/<timestamp>/report.html`), plus `champion.py` and
`population.json`. When summarising for the user:
- Lead with the sealed-exam verdicts. PASS means the strategy's holdout alpha cleared a bar that
  already accounts for every exam attempt. FAIL is the common, honest outcome.
- Training-period numbers are what evolution optimised. Never present them as expected returns.
- Never re-run evolution in a loop "until something passes". That turns the holdout back into a
  training set; tell the user a fresh holdout (later data, another market) is needed instead.

## Backtest one strategy, see its holdings, test it abroad
Write `def strategy(prices, data) -> weights DataFrame` (numpy/pandas/math and the `tk` toolkit,
no look-ahead, no hard-coded tickers), then:

```bash
alphafinch backtest my_strategy.py india
alphafinch holdings my_strategy.py india      # what it wants to hold after the latest close
alphafinch world-exam my_strategy.py          # 7 stock markets it has never seen
```

This is research software, not investment advice.
