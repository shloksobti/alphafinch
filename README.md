<div align="center">

# 🐦 AlphaFinch

### AI evolves trading strategies while you sleep.<br>Then they sit an exam they can't cheat.

<img src="docs/demo.gif" alt="AlphaFinch: install, evolve strategies, and the sealed exam" width="900">

Works with **Claude Code**, **Anthropic**, **OpenAI**, **Ollama**, any OpenAI-compatible server, or **no AI at all**.

</div>

---

AlphaFinch is an [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style lab for markets. An AI writes trading strategies as short Python functions, backtests them, and breeds the fittest: mutating them, crossing them and letting populations evolve on separate islands.

The catch with every AI trading demo: **evolution is the best overfitting machine ever built.** Run it long enough and it will "discover" a brilliant strategy in pure noise. AlphaFinch is built around that problem. The most recent years are locked in a **sealed exam** that neither evolution nor the AI ever sees, and the bar to pass accounts for every attempt.

**Most runs end with nothing passing.** That's the honest answer, and the point.

## What it found

We pre-registered every test before running it ([`docs/`](docs)) and report every result.

- **One market, three sealed years: 0 of 10 passed** (S&P 500 and NIFTY 200). Training scores rose while out-of-sample alpha fell: overfitting, caught in the act.
- **So we rebuilt the search** to score alpha, choose champions on years breeding never saw, and demand an edge that holds across random halves of the stocks. On a stand-in exam it beat the original search in 4 of 4 matched runs.
- **Then the world exam:** five strategies, bred only on data before 2017, were frozen and tested unchanged on **seven stock markets they had never seen**, from October 2017 to October 2026.

| Strategy | Alpha / year | t (bar 2.33) | Markets with positive alpha | |
|---|---|---|---|---|
| **Quiet Sector Tether v2** | **+2.9%** | **3.40** | **7 of 7** | ✅ PASS |
| The Team (India-bred) | +2.3% | 2.14 | 6 of 7 | 🟡 PROMISING |
| Quiet Intraday Relay v2 | +0.9% | 1.03 | 5 of 7 | 🟡 PROMISING |
| Two others | negative | | | ❌ FAIL |

*Quiet Sector Tether* buys, within each sector, the stocks that move least with the market and shorts those that move most. It is essentially **"betting against correlation"**, an anomaly published by AQR researchers in 2020: the AI rediscovered it from pre-2017 US data, and it held up in the UK, the Eurozone, Japan, Hong Kong, Australia, Canada and Korea. It survives higher trading costs (t 2.94 at 15 bps), a causal beta hedge (t 3.22) and the six standard Fama–French factors (alpha +2.5%/yr, t 2.64).

The caveats are in the [full write-up](docs/preregistration-world.md): it is weaker in 2020–26 alone (t 2.17), short-borrowing fees aren't modelled, and Korea banned short selling for parts of the period. It's frozen in [`forward/`](forward) to be judged again on data that doesn't exist yet.

## Quick start

```bash
pip install "alphafinch[all] @ git+https://github.com/shloksobti/alphafinch"

alphafinch demo                                # offline: synthetic market, no AI, no network (~1 min)
alphafinch evolve --market us                  # S&P 500; AI provider auto-detected
alphafinch evolve --market india               # NIFTY 200
alphafinch evolve --market futures             # 39 futures: equities, rates, FX, energy, metals, agriculture
alphafinch world-exam runs/<run>/champion.py   # test a champion on 7 markets it has never seen
alphafinch holdings runs/<run>/champion.py --market india   # what it wants to hold today
alphafinch replay runs/<run>                   # re-watch a finished run as a short story
```

Each run writes `runs/<timestamp>/report.html`: the champion, its family tree, the lab notebook, equity curves and the exam verdict.

## How it works

```
 breed (training years) ─▶ choose (validation years) ─▶ 🔒 sealed exam ─▶ 🌍 world exam ─▶ ⏳ forward test
```

**Breeding.** Four islands, each with a population of strategies. Every generation the AI **mutates** a parent using its report card, **crosses** two parents into one idea, or invents an **immigrant** from a fresh hypothesis. No-AI operators **tweak** a constant or **blend** two portfolios. Champions migrate between islands, and each island keeps the best strategy in every niche (fast or slow, market-neutral or market-hugging), so the population can't collapse onto one idea.

**Fitness rewards a real edge, not a lucky one:**
- **Alpha, not returns.** Each of four training eras is scored on its appraisal ratio: return beyond market exposure, per unit of risk. The *worst* era counts as much as the typical one.
- **Broad, not narrow.** The portfolio is re-scored on random halves of the stocks, and the worst half counts.
- **Stable, not knife-edge.** Parameters are nudged and neighbours re-scored. Penalties for heavy trading and bloated code.

**The AI works like a researcher.** Every strategy starts with a written hypothesis. A lab notebook of every idea tried, and how it fared, goes into each prompt. A toolkit (`tk`) makes sector-neutral long/short books, residual returns and volatility targeting one-liners. Use `--strong-model` to give crossovers and new ideas to a bigger model.

**Choosing the champion.** The last three training years are held back from breeding. The top ten finalists and a diversified team are scored once on those years, and the best becomes the champion. The AI never sees those scores.

**The sealed exam.** The most recent three years. Each attempt reveals only PASS or FAIL, and the bar is t⁻¹(α / attempts), valid however adaptively the search ran ([the theory](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458)). Results are graded **PASS**, **PROMISING** (t > 1, with the years of data a pass would need) or **FAIL**.

**The world exam.** Three years of one market can rarely prove a realistic edge. So a frozen strategy runs unchanged on seven other markets, and the evidence is pooled. The verdict uses the lower of two pooled t-statistics (Newey–West and Stouffer), so a strategy has to convince both. Calibrated with placebos: 0 of 200 random strategies passed.

**The forward test.** `alphafinch forward freeze runs/<run>` today, `alphafinch forward score` in six months. No model has seen tomorrow's data.

## Trading rules (mandates)

Real investors have constraints. A mandate is enforced by the engine on every strategy's weights, so a strategy can never break the rules and still look good. The AI is told the rules too, so it designs within them.

| `--mandate` | Rules |
|---|---|
| `long-only` | Cash / spot market: no shorting, no leverage. Realistic for most retail investors, e.g. in India, where cash-market shorts can't be held overnight |
| `long-short` | Shorts allowed, with a 0.5%/yr borrow fee on short positions |
| `market-neutral` | Long/short with net market exposure held within ±10% |
| `derivatives` | Shorts only where single-stock futures exist (India: the NSE F&O list), up to 2× gross, 3%/yr financing above 1× |
| `futures` | Default for `--market futures`: shorting is free, up to 3× gross (margin) |

Fine-tune with `--max-gross`, `--max-weight 0.05` and `--borrow-bps`. Mandates work with `evolve`, `backtest`, `holdings` and `world-exam`.

## Futures

`--market futures` gives the AI 39 futures across equity indices, government bonds, currencies, energy, metals and agriculture, so it can test hypotheses like trend-following, crisis alpha, carry or cross-asset signals.

Free continuous futures prices splice contracts without adjusting for the roll, which creates fake jumps: on Yahoo's natural-gas series a rolled position "earned" +20% a year when it really lost 12%. So AlphaFinch builds each future from a fund that holds and rolls the real contracts, converted to excess returns over T-bills, which is what a futures position earns.

## Safety and honesty

- **Sandbox:** AI-written code may import only `numpy`, `pandas` and `math`, with no file, network or dunder access. It runs in separate processes with restricted builtins, a CPU limit and a timeout.
- **Look-ahead detector:** every strategy is re-run on truncated histories. If past weights change when future data is removed, it's discarded.
- **No hindsight by name:** code that hard-codes a ticker or sector is rejected, so the AI can't simply pick stocks it knows did well.
- **Costs:** 5 bps per unit of turnover. Weights act from the next close.
- **Survivorship:** universes are *today's* index members. Alpha is measured against the same list, which limits the bias but doesn't remove it.

## Bring any AI

| `--provider` | Setup | Notes |
|---|---|---|
| `claude-code` | [Claude Code](https://claude.com/claude-code) installed | no API key; runs `claude -p` |
| `anthropic` | `ANTHROPIC_API_KEY` | default `claude-opus-5-5` |
| `openai` | `OPENAI_API_KEY` | pick with `--model` |
| `ollama` | a local model at `localhost:11434` | free and private |
| `compatible` | `--base-url … --model …` | any OpenAI-compatible server |
| `none` | nothing | tweaks and blends only, offline |

`--provider auto` (the default) uses the first one it finds. A Claude Code skill is included in [`integrations/claude-code`](integrations/claude-code).

## Markets and data

All free, no keys:

| `--market` | Universe |
|---|---|
| `us` | S&P 500 since 2010, with SEC fundamentals (point-in-time, the day after each 10-K) |
| `india` | NIFTY 200 since 2010 |
| `uk` `europe` `japan` `hongkong` `australia` `canada` `korea` | FTSE 100, Eurozone large caps, Nikkei 225, Hang Seng, ASX 200, TSX 60, KOSPI 200 |
| `us30` `crypto` `industries` `synthetic` | 30 US mega-caps, 15 coins, 49 US industries since 1970, simulated |

Strategies see `prices` plus `data.open/high/low/volume`, `data.sector`, `data.macro` (VIX, index, oil, gold, rates and more) and, for the US, `data.fund` (market cap, earnings yield, book-to-market, ROE, sales growth).

US fundamentals need a contact email, because the SEC asks every client for one: `export ALPHAFINCH_SEC_CONTACT="Your Name you@example.com"`.

## Write your own

```python
def strategy(prices, data):
    """Sector Spread: sector-neutral 6-month momentum. Hypothesis: news diffuses slowly within industries."""
    score = tk.neutralize(tk.zscore(prices.pct_change(126)), data.sector)
    return tk.rebalance(tk.long_short(score, q=0.2), every="M")
```

```bash
alphafinch backtest my_strategy.py --market us
alphafinch world-exam my_strategy.py
```

## FAQ

**Will this make me money?** Probably not, and AlphaFinch is built to tell you so. A PASS is a lead worth researching, not a trading signal.

**Why not just backtest on all the data?** With enough tries, something always worked by luck. Only data the search never touched can tell luck from skill.

**Can I re-run until something passes?** You can, but then the exam means nothing. Count your earlier looks (`--alpha`, `--prior-looks`) or test on new markets and new data.

## About

Built by [Shlok Sobti](https://github.com/shloksobti) at [Invsify](https://invsify.com), a SEBI-registered investment advisory in India. AlphaFinch is an independent open-source research project: nothing in this repository is investment advice or a recommendation from Invsify.

Research and educational software. Backtests ignore taxes, borrow costs, capacity limits and slippage beyond the modelled costs.

MIT License.
