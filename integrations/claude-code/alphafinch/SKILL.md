---
name: alphafinch
description: Evolve, backtest and honestly evaluate trading strategies with AlphaFinch. Use when the user wants to discover, breed, improve or test trading strategies on US stocks, Indian stocks (NSE), crypto or industry portfolios, or asks whether a backtest is overfit.
---

# AlphaFinch in Claude Code

AlphaFinch evolves trading strategies with an AI (islands, mutation, crossover) and scores the
winners on a sealed holdout that they never see.

## Run an evolution
Use the Claude Code provider so the user's existing subscription does the breeding:

```bash
pip install "alphafinch[all]"
alphafinch evolve --market us --provider claude-code --generations 20
```

Markets: `us`, `india`, `crypto`, `industries` (survivorship-free, since 1970), `synthetic` (offline),
or `--tickers AAPL,MSFT,...`. Long runs are fine to leave running; the live view shows progress.

## Read the results
The run prints a report path (`runs/<timestamp>/report.html`), plus `champion.py` and
`population.json`. When summarising for the user:
- Lead with the sealed-exam verdicts. PASS means the strategy's holdout alpha cleared a bar that
  already accounts for every exam attempt. FAIL is the common, honest outcome.
- Training-period numbers are what evolution optimised. Never present them as expected returns.
- Never re-run evolution in a loop "until something passes". That turns the holdout back into a
  training set; tell the user a fresh holdout (later data, another market) is needed instead.

## Backtest one strategy
Write `def strategy(prices) -> weights DataFrame` (numpy/pandas only, no look-ahead), then:

```bash
alphafinch backtest my_strategy.py --market india
```

This is research software, not investment advice.
