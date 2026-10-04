"""Prompts for the genetic operators. The AI only ever sees training-period results."""
from __future__ import annotations

import re

_SYSTEM = """You are a quantitative researcher. You write trading strategies as small, readable Python functions, and you think like a scientist: every strategy starts from an economic hypothesis about WHY it should earn returns that the market does not already reward.

Contract:
- Define exactly one function `strategy(prices, data)`. `prices` is a pandas DataFrame of daily closing prices (rows = dates ascending, columns = assets) and equals `data.close`. `data` also provides:
{data}
- Return a DataFrame of target portfolio weights with the same index and columns as `prices`. Positive = long, negative = short. Gross exposure is capped at 1 by the engine (weights are scaled down if their absolute values sum above 1).
- The weight on row t may use only information up to and including row t. The engine applies it from the close of t to the close of t+1. Never use future data: no shift(-k), no centred windows, no statistics computed over the whole sample. Look-ahead is detected automatically and the strategy is discarded.
- Allowed imports: numpy, pandas, math only. No file, network or system access. Missing values are common (new listings, missing fundamentals): handle NaN explicitly.
- Keep it short (under 45 lines) and vectorised. Put tunable numbers in UPPER_CASE constants at the top of the function.
- The docstring must be one line: "<Name>: <one-sentence idea>. Hypothesis: <why this should earn alpha>." The name is 2-3 evocative words.
- Trading costs of 5 bps per unit of turnover are charged, so avoid needless churn.
- Strategies are judged on Sharpe ratio AND alpha (return beyond what exposure to the equal-weight market explains), separately in four eras of history; the WORST era matters as much as the typical one. Simply owning the market earns no alpha. An edge that exists only at one exact parameter value is penalised.

Reply with a single ```python code block and nothing else."""


def system(data_description: str) -> str:
    return _SYSTEM.format(data=data_description)


def report_card(name: str, s, robust: float | None = None) -> str:
    eras = "; ".join(f"{e['start'][:4]}-{e['end'][:4]}: Sharpe {e['sharpe']:.2f}, appraisal {e['appraisal']:.2f}"
                     for e in s.eras)
    rob = f" Nearby parameter settings scored {robust:+.2f} on average." if robust is not None else ""
    return (f"{name}: Sharpe {s.sharpe:.2f}, CAGR {s.cagr:+.1%}, volatility {s.vol:.1%}, max drawdown {s.max_dd:.0%}, "
            f"turnover {s.turnover:.1f}x/yr, gross exposure {s.exposure:.2f}, beta {s.beta:.2f}, "
            f"alpha {s.alpha:+.1%}/yr (appraisal ratio {s.appraisal:.2f}).\nBy era: {eras}.{rob}")


def weakness(s, robust: float | None, raw: float | None) -> str:
    if s is None:
        return "failed"
    issues = []
    if s.eras:
        worst = min(s.eras, key=lambda e: e["score"])
        if worst["score"] < 0:
            issues.append(f"lost in {worst['start'][:4]}-{worst['end'][:4]}")
    if s.appraisal < 0.2:
        issues.append("little alpha beyond market exposure")
    if s.turnover > 20:
        issues.append(f"trades too much ({s.turnover:.0f}x/yr)")
    if robust is not None and raw is not None and robust < raw - 0.2:
        issues.append("fragile to small parameter changes")
    return ", ".join(issues) or "solid"


def notebook(entries: list[dict], n_best: int = 8, n_recent: int = 8) -> str:
    if not entries:
        return ""
    ok = [e for e in entries if e["fitness"] is not None]
    best = sorted(ok, key=lambda e: -e["fitness"])[:n_best]
    recent = [e for e in entries[-40:] if e not in best][-n_recent:]
    fmt = lambda e: (f"- {e['name']} ({e['op']}): {e['idea']}"
                     + (f" | fitness {e['fitness']:+.2f}" if e["fitness"] is not None else "") + f" | {e['why']}")
    out = "LAB NOTEBOOK (ideas already tried; do not repeat them, build on what worked and avoid what failed):\n"
    out += "Best so far:\n" + "\n".join(fmt(e) for e in best)
    if recent:
        out += "\nRecent attempts:\n" + "\n".join(fmt(e) for e in recent)
    return out + "\n\n"


def mutate(code: str, card: str, market: str, nb: str = "") -> str:
    return f"""Market: {market}.
{nb}Here is a strategy and its TRAINING-period report card:

```python
{code.strip()}
```

{card}

Write an improved offspring. Diagnose its main weakness from the report card (a losing era, low alpha, too much trading, fragility) and make one or two focused, well-motivated changes that address it. You may use any field of `data`. Do not just change numbers. Give it a new name and hypothesis."""


def crossover(code_a: str, card_a: str, code_b: str, card_b: str, market: str, nb: str = "") -> str:
    return f"""Market: {market}.
{nb}Two successful parent strategies (TRAINING-period results):

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

Write a child that combines the best idea of each parent into one coherent strategy with a single clear hypothesis (not just an average of their weights). Give it a new name."""


def immigrant(theme: str, market: str, nb: str = "") -> str:
    return f"""Market: {market}.
{nb}Invent a NEW strategy from scratch around this theme: {theme}.
Start from an economic hypothesis (who is on the other side of the trade, and why would they keep losing?). Use whichever fields of `data` best test it. It must be different from everything in the lab notebook. Give it a name."""


def extract_code(text: str) -> str | None:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m[0]
    return text if "def strategy" in text else None


_DOC = re.compile(r'def strategy\(.*?\):\s*\n\s*(?:"""|\'\'\')\s*(.*?)(?:"""|\'\'\')', re.S)


def strategy_name(code: str, fallback: str = "Unnamed") -> str:
    m = _DOC.search(code)
    if m and ":" in m.group(1):
        name = m.group(1).split(":", 1)[0].strip()
        if 2 <= len(name) <= 40:
            return name
    return fallback


def idea_and_hypothesis(code: str) -> tuple[str, str]:
    m = _DOC.search(code)
    doc = " ".join(m.group(1).split()) if m else ""
    body = doc.split(":", 1)[1].strip() if ":" in doc else doc
    if "Hypothesis:" in body:
        idea, hyp = body.split("Hypothesis:", 1)
        return idea.strip()[:160], hyp.strip()[:200]
    return body[:160], ""
