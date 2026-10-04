<div align="center">

# 🐦 AlphaFinch

### AI evolves trading strategies while you sleep.<br>Then they sit an exam they can't cheat.

<p>
<a href="https://pypi.org/project/alphafinch/"><img src="https://img.shields.io/pypi/v/alphafinch?logo=pypi&logoColor=white&color=3775A9" alt="PyPI"></a>
<a href="https://pypi.org/project/alphafinch/"><img src="https://img.shields.io/pypi/dm/alphafinch?label=downloads&color=3775A9" alt="Downloads"></a>
<a href="https://github.com/shloksobti/alphafinch/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/shloksobti/alphafinch/tests.yml?branch=main&label=tests&logo=github" alt="tests"></a>
<a href="https://github.com/shloksobti/alphafinch"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
<a href="https://github.com/shloksobti/alphafinch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
<br>
<a href="https://github.com/shloksobti/alphafinch/blob/main/docs/preregistration-world.md"><img src="https://img.shields.io/badge/world%20exam-PASS%207%2F7%20markets-brightgreen" alt="World exam: PASS in 7 of 7 markets"></a>
<a href="https://github.com/shloksobti/alphafinch/tree/main/docs"><img src="https://img.shields.io/badge/results-pre--registered-blueviolet" alt="Pre-registered results"></a>
<a href="https://github.com/shloksobti/alphafinch/blob/main/docs/guide.md#markets"><img src="https://img.shields.io/badge/markets-15%20(stocks%20%C2%B7%20futures%20%C2%B7%20crypto)-orange" alt="15 markets"></a>
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458"><img src="https://img.shields.io/badge/paper-SSRN%207557458-b31b1b" alt="Paper on SSRN"></a>
<img src="https://img.shields.io/badge/data-free%2C%20no%20API%20keys-informational" alt="Free data">
</p>

<img src="https://raw.githubusercontent.com/shloksobti/alphafinch/main/docs/demo.gif" alt="AlphaFinch: install, evolve strategies, and the sealed exam" width="900">

Works with **Claude Code**, **Anthropic**, **OpenAI**, **Ollama**, any OpenAI-compatible server, or **no AI at all**.

</div>

---

AlphaFinch is an [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style lab for markets. An AI writes trading strategies as short Python functions, backtests them, and breeds the fittest: mutating them, crossing them and letting populations evolve on separate islands.

The catch with every AI trading demo: **evolution is the best overfitting machine ever built.** Run it long enough and it will "discover" a brilliant strategy in pure noise. AlphaFinch is built around that problem. The most recent years are locked in a **sealed exam** that neither evolution nor the AI ever sees, and the bar to pass accounts for every attempt.

**Most runs end with nothing passing.** That's the honest answer, and the point.

## What it found

We pre-registered every test before running it ([`docs/`](docs)) and report every result.

- **One market, three sealed years: 0 of 10 passed** on the S&P 500 and NIFTY 200 ([details](docs/preregistration-v2.md); earlier runs on industries and India: 0 of 8, [details](docs/preregistration.md)). Training scores rose while out-of-sample alpha fell: overfitting, caught in the act.
- **So we rebuilt the search** to score alpha, choose champions on years breeding never saw, and demand an edge that holds across random halves of the stocks. On a stand-in exam it beat the original search in 4 of 4 matched runs ([details](docs/search-ablation.md)).
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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shloksobti/alphafinch/blob/main/examples/quickstart.ipynb) **Try it in your browser first:** the quickstart notebook runs the demo, backtests a strategy on Indian stocks, shows its holdings and reproduces the world-exam result, with nothing to install.

Needs Python 3.10+. No API keys or data subscriptions: market data is free and downloaded on first use. Prefer the latest code? `pip install "alphafinch[all] @ git+https://github.com/shloksobti/alphafinch"`.

```bash
pip install "alphafinch[all]"

alphafinch demo                                # offline: synthetic market, no AI, no network (~1 min)
alphafinch evolve india                        # NIFTY 200 stocks; AI provider auto-detected
alphafinch evolve india --long-only            # same, but no shorting (cash market)
alphafinch evolve india-futures                # NSE futures: NIFTY, BANKNIFTY, every F&O stock
alphafinch evolve futures                      # 39 global futures: indices, bonds, FX, commodities
alphafinch holdings runs/<run>/champion.py india   # what the champion wants to hold today
alphafinch world-exam runs/<run>/champion.py   # test it on 7 markets it has never seen
alphafinch replay runs/<run>                   # re-watch a finished run as a short story
```

Two choices shape every run: the **market** (a plain word, e.g. `india`) and the **rules** (flags, e.g. `--long-only`). Type `alphafinch` alone for an overview, `alphafinch evolve -h` for every option, or read the **[guide](docs/guide.md)**.

**What a run takes.** The default run (20 generations × 4 islands) makes a few hundred AI calls and takes roughly 20–40 minutes. The first run on a market also downloads its data (a few minutes, then cached). With `claude-code` it uses your existing subscription; with an API key you pay your provider's usual rates.

**What you get** in `runs/<timestamp>/`:

| File | Contents |
|---|---|
| `report.html` | The morning report: champion, exam verdict, equity curves, the team, the lab notebook, the family tree |
| `champion.py` | The champion's code, ready for `backtest`, `holdings` or `world-exam` |
| `team.py` | A diversified team of survivors, when one forms |
| `population.json` | Every strategy bred, with its scores |

`alphafinch replay runs/<timestamp>` re-tells any finished run as a short story:

<img src="https://raw.githubusercontent.com/shloksobti/alphafinch/main/docs/replay.gif" alt="alphafinch replay: the AI writing a strategy, a new champion, and the sealed exam" width="820">

## How it works

```
 breed ─▶ choose ─▶ 🔒 sealed exam ─▶ 🌍 world exam ─▶ ⏳ forward test
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

## Trading rules

Two choices shape every run: the **market** (*what* can be traded) and the **rules** (*how*). Rules are enforced by the engine on every strategy's positions, so a strategy can't break them and still look good, and the AI is told them up front.

| Flag | Rule |
|---|---|
| *(none)* | Long or short, up to 1× capital |
| `--long-only` | Buy only, no leverage: the cash / spot market |
| `--market-neutral` | Longs and shorts roughly equal, with a borrow fee on shorts |
| `--max-position 5%` | No single position above 5% |
| `--leverage 2` | Total exposure up to 2× capital |

Futures markets get futures rules automatically: shorting is as easy as buying, up to 3× exposure.

## Futures

- `futures`: 39 global futures across stock indices, government bonds, currencies, energy, metals and agriculture, for hypotheses like trend-following, crisis alpha, carry or cross-asset signals.
- `india-futures`: NSE futures, NIFTY and BANKNIFTY plus every F&O stock.

Free continuous futures prices fake big gains or losses at every contract roll: on Yahoo's natural-gas series a rolled position "earned" +20% a year when it really lost 12%. So AlphaFinch builds futures from funds that hold and roll the real contracts (and Indian stock futures from each stock's total return), converted to excess returns over the short-term interest rate, which is what a futures position earns.

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

| Market | Universe |
|---|---|
| `us` | S&P 500 since 2010, with SEC fundamentals (point-in-time, the day after each 10-K) |
| `india` | NIFTY 200 since 2010 |
| `india-futures` | NSE futures: NIFTY, BANKNIFTY and every F&O stock, since 2012 |
| `futures` | 39 global futures since 2012 |
| `uk` `europe` `japan` `hongkong` `australia` `canada` `korea` | FTSE 100, Eurozone large caps, Nikkei 225, Hang Seng, ASX 200, TSX 60, KOSPI 200 |
| `us30` `crypto` `industries` `synthetic` | 30 US mega-caps, 15 coins, 49 US industries since 1970, simulated |

Or use your own list: `--tickers RELIANCE.NS,TCS.NS,INFY.NS` (any Yahoo symbols).

Strategies see `prices` plus `data.open/high/low/volume`, `data.sector`, `data.macro` (VIX, index, oil, gold, rates and more) and, for the US, `data.fund` (market cap, earnings yield, book-to-market, ROE, sales growth).

US fundamentals need a contact email, because the SEC asks every client for one: `export ALPHAFINCH_SEC_CONTACT="Your Name you@example.com"`.

## Write your own

```python
def strategy(prices, data):
    """Sector Spread: sector-neutral 6-month momentum. Hypothesis: news diffuses slowly within industries."""
    score = tk.neutralize(tk.zscore(prices.pct_change(126)), data.sector)
    return tk.rebalance(tk.long_short(score, q=0.2), every="M")
```

A strategy returns, for every day, the fraction of capital to hold in each asset (negative means short). It may use `numpy`, `pandas`, `math` and the built-in toolkit `tk`, must never use future data, and may not name tickers. The [guide](docs/guide.md#writing-a-strategy) lists every field and toolkit function.

```bash
alphafinch backtest my_strategy.py us      # training years only; the sealed years stay sealed
alphafinch holdings my_strategy.py us      # what it wants to hold after the latest close
alphafinch world-exam my_strategy.py       # 7 stock markets it has never seen
```

## FAQ

**Will this make me money?** Probably not, and AlphaFinch is built to tell you so. A PASS is a lead worth researching, not a trading signal.

**Why not just backtest on all the data?** With enough tries, something always worked by luck. Only data the search never touched can tell luck from skill.

**Can I re-run until something passes?** You can, but then the exam means nothing. Count your earlier looks (`--alpha`, `--prior-looks`) or test on new markets and new data.

**Does it tell me what to buy?** `alphafinch holdings` shows the positions a strategy wants today. That's the output of a research tool, not a recommendation: check the exam verdict and the caveats first.

**Can I use my own data?** Any Yahoo symbols with `--tickers`. Other sources can be added in `alphafinch/data.py`, which returns a simple `Panel` of aligned tables.

## Development

```bash
git clone https://github.com/shloksobti/alphafinch && cd alphafinch
pip install -e ".[all,dev]"
pytest -q                     # about 40 tests, offline, under a minute
```

Issues and pull requests are welcome: new markets, data sources, toolkit functions and exams especially.

## Citation

The sealed exam's bar comes from:

> Shlok Sobti, *Deflate by Bits, Not Trials*, SSRN 7557458 (2026). [papers.ssrn.com/abstract=7557458](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458)

## About

Built by [Shlok Sobti](https://github.com/shloksobti) at [Invsify](https://invsify.com), a SEBI-registered investment advisory in India. AlphaFinch is an independent open-source research project: nothing in this repository is investment advice or a recommendation from Invsify.

Research and educational software. Backtests ignore taxes, capacity limits and slippage beyond the modelled costs; borrow fees are charged only under `--market-neutral`.

MIT License.
