"""Prompts for the genetic operators. The AI only ever sees training-period results."""
from __future__ import annotations

import re

SYSTEM = """You are a quantitative researcher writing trading strategies as small, readable Python functions.

Contract:
- Define exactly one function `strategy(prices)` where `prices` is a pandas DataFrame of daily closing prices (rows = dates ascending, columns = assets).
- Return a DataFrame of target portfolio weights with the same index and columns. Positive = long, negative = short. Gross exposure is capped at 1 by the engine (weights are scaled down if the absolute values sum above 1).
- The weight on row t may use only information up to and including row t. The engine applies it from the close of t to the close of t+1. Never use future data: no shift(-k), no centred windows, no statistics computed over the whole sample. Look-ahead is detected automatically and the strategy is discarded.
- Allowed imports: numpy, pandas, math only. No file, network or system access.
- Keep it short (under 40 lines) and vectorised. Put tunable numbers in UPPER_CASE constants at the top of the function.
- Start the function with a one-line docstring: "<Name>: <one-sentence idea>". The name is 2-3 evocative words.
- Trading costs of 5 bps per unit of turnover are charged, so avoid needless churn.
- Strategies are judged on Sharpe ratio AND on alpha: return beyond what their exposure to the equal-weight market explains. Simply owning the market earns no alpha.

Reply with a single ```python code block and nothing else."""


def report_card(name: str, s) -> str:
    years = ", ".join(f"{y}: {r:+.0%}" for y, r in list(s.yearly.items())[-12:])
    return (f"{name}: Sharpe {s.sharpe:.2f}, CAGR {s.cagr:+.1%}, volatility {s.vol:.1%}, "
            f"max drawdown {s.max_dd:.0%}, turnover {s.turnover:.1f}x/yr, gross exposure {s.exposure:.2f}, "
            f"beta to market {s.beta:.2f}, alpha {s.alpha:+.1%}/yr (appraisal ratio {s.appraisal:.2f}).\nRecent years: {years}")


def mutate(code: str, card: str, market: str) -> str:
    return f"""Market: {market}.
Here is a strategy and its TRAINING-period report card:

```python
{code.strip()}
```

{card}

Write an improved offspring. Make one or two focused, well-motivated changes that address its weaknesses (e.g. a regime filter, better exits, volatility scaling, a different ranking signal, smarter rebalancing). Do not just change numbers. Give it a new name."""


def crossover(code_a: str, card_a: str, code_b: str, card_b: str, market: str) -> str:
    return f"""Market: {market}.
Two successful parent strategies (TRAINING-period results):

Parent A:
```python
{code_a.strip()}
```
{card_a}

Parent B:
```python
{code_b.strip()}
```
{card_b}

Write a child strategy that combines the best idea of each parent into one coherent strategy (not just an average of their weights). Give it a new name."""


def immigrant(theme: str, market: str) -> str:
    return f"""Market: {market}.
Invent a new strategy from scratch around this theme: {theme}.
It should be different from textbook momentum. Give it a name."""


def extract_code(text: str) -> str | None:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m[0]
    return text if "def strategy" in text else None


def strategy_name(code: str, fallback: str = "Unnamed") -> str:
    m = re.search(r'def strategy\(.*?\):\s*\n\s*(?:"""|\'\'\')\s*([^:\n]{2,40}):', code)
    return m.group(1).strip() if m else fallback
