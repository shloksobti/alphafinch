<div align="center">

# 🐦 AlphaFinch

### Evolve trading strategies with AI while you sleep.<br>Then make them pass an exam they can't cheat.

<img src="docs/demo.gif" alt="AlphaFinch evolving strategies in the terminal" width="900">

`pip install alphafinch` · works with **Claude Code**, **Anthropic**, **OpenAI**, **Ollama**, or **no AI at all**

</div>

---

AlphaFinch is an [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style lab for markets. An AI writes trading strategies as small Python functions, backtests them and breeds the fittest. Generation after generation it mutates them, crosses them with each other, and lets the population evolve across islands.

Every AI trading demo has the same problem: **evolution is the most powerful overfitting machine ever built.** Give it enough generations and it will "discover" a spectacular strategy in pure noise. So AlphaFinch locks the most recent years of data in a **sealed exam**. Evolution never sees it, the AI never sees it, and champions only ever learn **PASS** or **FAIL**. If something passes, it beat a bar that already accounts for every attempt.

## Try it in 60 seconds

```bash
pip install "alphafinch[all]"
alphafinch demo                       # offline: synthetic market, no AI, no network
alphafinch evolve --market us         # real US stocks; AI provider auto-detected
alphafinch evolve --market india      # NSE large caps
alphafinch evolve --market crypto     # top coins vs USDT
```

Then open `runs/<timestamp>/report.html` for the morning report: winners, sealed-exam verdicts, equity curves and the champion's family tree.

Replay any finished run as a short cinematic story (the GIF above is one, from a real run):

```bash
alphafinch replay runs/<timestamp>
```

## How strategies breed

```
   🏝 Island 1          🏝 Island 2          🏝 Island 3          🏝 Island 4
  ┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐
  │ population │ ─✈️─▶ │ population │ ─✈️─▶ │ population │ ─✈️─▶ │ population │ ─✈️─┐
  └────────────┘       └────────────┘       └────────────┘       └────────────┘     │
        ▲                                                                           │
        └──────────────────── champions migrate every few generations ──────────────┘
```

Each generation, on every island:

| Operator | Who does it | What happens |
|---|---|---|
| 🧬 **Mutate** | AI | Reads a parent's code and its report card ("bleeds in sideways markets, trades too much") and improves one thing |
| 💞 **Crossover** | AI | Combines the best idea of two successful parents into one coherent child |
| 🛶 **Immigrant** | AI | Invents a brand-new strategy around a fresh theme (seasonality, breadth, tail risk…) |
| 🔧 **Tweak** | no AI | Nudges one numeric constant (lookback 20 → 23): fast fine-tuning |
| 🎨 **Blend** | no AI | The child holds a mix of both parents' portfolios |

**Survival** uses training data only. Fitness is half Sharpe ratio and half *alpha* (return beyond what market exposure explains), plus a bonus for consistency across years and penalties for heavy trading and bloated code.

**Diversity** is protected in two ways:
- Each island keeps a **niche map** (slow vs fast traders, market-neutral vs market-hugging), and the best strategy in every niche survives, so the population can't collapse into 100 copies of one idea.
- Champions **migrate** between islands. That's Darwin's Galápagos insight, and the same diversity trick AlphaEvolve uses.

## The sealed exam

```
├──────────────── training data: evolution sees this ────────────────┤🔒 sealed: last 3 years ┤
```

- The last `--holdout-years` (default 3; 1.5 for crypto) are locked away.
- A generation's champion may sit the exam only when it clearly beats the last one examined.
- The exam answers **PASS / FAIL**, nothing else, with a fixed budget of attempts (default 10).
- To pass, the strategy's **holdout alpha** must have a t-statistic above `t⁻¹(α / budget)`. Each attempt reveals only one bit, so this bar stays valid however adaptively the population evolved. The theory is in [*Deflate by Bits, Not Trials*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458).

Most runs end with **no strategy passing**. That's the honest answer when nothing in the data survives out of sample, and it's exactly what other tools hide.

## Bring any AI (or your Claude Code)

| `--provider` | Setup | Notes |
|---|---|---|
| `claude-code` | have [Claude Code](https://claude.com/claude-code) installed | no API key; uses your Claude subscription via `claude -p` |
| `anthropic` | `ANTHROPIC_API_KEY=...` | default model `claude-opus-5-5` |
| `openai` | `OPENAI_API_KEY=...` | choose with `--model` |
| `ollama` | a local model at `localhost:11434` | free and private; default `qwen2.5-coder:14b` |
| `compatible` | `--base-url ... --model ...` | any OpenAI-compatible server |
| `none` | nothing | tweaks and blends only, fully offline |

`--provider auto` (the default) picks the first one it finds. Inside Claude Code you can also just ask: *"evolve trading strategies on Indian stocks with alphafinch"*. A skill is included in [`integrations/claude-code`](integrations/claude-code).

## Markets

| `--market` | Universe | Free source, no key |
|---|---|---|
| `us` | 29 US large caps + SPY, since 2008 | Yahoo Finance |
| `india` | 25 NIFTY large caps, since 2008 | Yahoo Finance (`.NS`) |
| `crypto` | 15 top coins vs USDT, since late 2020 | Binance public API |
| `industries` | 49 US industry portfolios, since 1970 | Ken French Data Library |
| `synthetic` | regime-switching simulated market | offline |

Or bring your own: `--tickers AAPL,MSFT,TSLA,...` (any Yahoo symbols). Data is cached in `~/.alphafinch`.

> ⚠️ **Survivorship bias.** The `us` and `india` lists are *today's* large caps, so absolute returns look better than they would have in real time. Alpha is measured against the same list, which limits the damage. For serious research, use `industries`.

## Safety: the AI writes code, so it runs in a sandbox

- **Static checks:** only `numpy`, `pandas` and `math`; no file, network, process or dunder access.
- **Process isolation:** each strategy runs in a separate process with restricted builtins, a CPU limit and a timeout.
- **Look-ahead detector:** every strategy is re-run on truncated histories. If its past weights change when future data is removed (`shift(-1)`, centred windows, full-sample normalisation…), it's discarded as "peeked at the future" 💀.
- **Realistic execution:** costs of 5 bps per unit of turnover, and weights act from the next close.

## Write your own strategy

```python
import numpy as np, pandas as pd

def strategy(prices):
    """Calm Seeker: overweight the least volatile assets."""
    WINDOW = 63
    vol = prices.pct_change().rolling(WINDOW, min_periods=WINDOW).std()
    inv = 1.0 / vol
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
```

```bash
alphafinch backtest my_strategy.py --market us
```

## A real run

US large caps, 12 generations, 4 islands, bred by Claude Sonnet through Claude Code (`--provider claude-code`):

- **228 strategies** were bred: 137 by the AI (84 mutations, 35 crossovers, 18 immigrants) and the rest by tweaks, blends, seeds and migration. One was discarded by the sandbox.
- The AI's best inventions included *Calm Crown* (volatility-adjusted momentum with market-volatility targeting), *Calm Momentum Shield* and *Residual Calm Drift*.
- The champion was *Storm Shelter × Momentum Crown v6*, a blend fine-tuned over six generations.

| Champion | Training (2008–2023) | 🔒 Sealed exam (Oct 2023 – Oct 2026) |
|---|---|---|
| Alpha per year | **+7.2%** | **+0.6%** (t = 0.17, bar 2.58) → **FAIL** |
| Return per year | | +20.9% |
| Equal-weight buy & hold | | +20.7% |

The champion made plenty of money in the sealed years: almost exactly as much as buying all 30 stocks. A normal backtest would call it a winner. The alpha exam shows the edge it learned in training didn't survive. That's what AlphaFinch is for.

## FAQ

**Will this make me money?** Probably not, and AlphaFinch is built to tell you so. Most evolved strategies fail the sealed exam. Treat a PASS as a lead worth investigating, not a trading signal.

**Why not just backtest on all the data?** Because with enough tries you'll always find something that worked by luck. The sealed exam is the only part of the run that can't be gamed.

**Can I run it longer?** Yes, try `--generations 100` overnight. More generations make better *training* strategies, but the exam bar doesn't move, so the verdict stays honest.

**Can I reuse the holdout after a run?** Don't. Once you've seen the results, re-running until something passes turns the holdout back into training data. Use later data or another market.

## Disclaimer

Research and educational software, not investment advice. Backtests ignore taxes, borrow costs, capacity limits and slippage beyond the modelled costs.

## License

MIT · built by [Shlok Sobti](https://github.com/shloksobti)
