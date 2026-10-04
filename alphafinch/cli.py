"""alphafinch command line.

  alphafinch demo                       offline demo, no AI and no network (about 1 minute)
  alphafinch evolve --market us         evolve strategies (provider auto-detected)
  alphafinch evolve --market india      Indian stocks (NIFTY 200)
  alphafinch backtest my_strategy.py    backtest one strategy file on the training period
  alphafinch replay runs/<timestamp>    re-animate a finished run as a short story
  alphafinch forward freeze runs/<ts>   freeze a run's champion and team for a forward test
  alphafinch forward score              judge frozen strategies on data after their freeze date
  alphafinch world-exam a.py b.py       test frozen strategies on 7 stock markets they've never seen
  alphafinch markets                    list built-in markets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console

from . import data, engine, fitness, llm, report
from .evolve import Config, Evolution
from .lab import Lab
from .sandbox import StrategyError
from .universe import WORLD
from .ui import LiveView

console = Console()
MARKET_LABEL = {"us": "US stocks (S&P 500)", "india": "Indian stocks (NIFTY 200)", "us30": "30 US mega-caps",
                "crypto": "Crypto vs USDT", "industries": "49 US industry portfolios (Ken French)",
                "synthetic": "Synthetic market (offline)",
                **WORLD}


def _load(args) -> data.Panel:
    if getattr(args, "sec_contact", None):
        os.environ["ALPHAFINCH_SEC_CONTACT"] = args.sec_contact
    tickers = [t.strip() for t in args.tickers.split(",")] if getattr(args, "tickers", None) else None
    with console.status(f"[cyan]Loading {MARKET_LABEL.get(args.market, args.market)}…") as st:
        panel = data.load(args.market, tickers, getattr(args, "start", None),
                          progress=lambda i, n, s: st.update(f"[cyan]Downloading {s} ({i + 1}/{n})…"))
    if getattr(args, "end", None):              # pretend the data stops here (research on past periods)
        panel = panel.before(pd.Timestamp(args.end) + pd.Timedelta(days=1))
    extras = []
    if panel.volume is not None:
        extras.append("volume")
    if panel.sector is not None and panel.sector.nunique() > 1:
        extras.append(f"{panel.sector.nunique()} sectors")
    if panel.macro is not None and len(panel.macro.columns):
        extras.append(f"{len(panel.macro.columns)} macro series")
    if panel.fund:
        extras.append("SEC fundamentals")
    console.print(f"[green]✓[/] {panel.shape[1]} assets, {panel.index[0].date()} → {panel.index[-1].date()}"
                  + (f" · {', '.join(extras)}" if extras else ""))
    if args.market == "us" and not panel.fund and not tickers:
        console.print("[dim]  Tip: set ALPHAFINCH_SEC_CONTACT=\"Your Name you@email.com\" to add SEC fundamentals "
                      "(the SEC requires a contact email).[/]")
    return panel


def _holdout_start(panel: data.Panel, holdout_years: float | None, market: str = "us"):
    holdout_years = holdout_years or (1.5 if market == "crypto" else 3.0)
    start = panel.index[-1] - pd.DateOffset(years=holdout_years)
    return panel.index[panel.index.searchsorted(start)]


def cmd_evolve(args, demo=False):
    panel = _load(args)
    hold_start = _holdout_start(panel, args.holdout_years, args.market)
    if (panel.index < hold_start).sum() < 3 * 252:
        console.print("[red]Not enough training history (need 3+ years before the holdout).[/]")
        return 1
    provider_name = "none" if demo else (llm.auto() if args.provider == "auto" else args.provider)
    try:
        provider = llm.get(provider_name, args.model, args.base_url, args.api_key)
        strong = llm.get(provider_name, args.strong_model, args.base_url, args.api_key) \
            if (provider and getattr(args, "strong_model", None)) else None
    except (llm.LLMError, ImportError) as e:
        console.print(f"[red]Could not start provider '{provider_name}': {e}[/]")
        return 1
    if provider is None and not demo:
        console.print("[yellow]No AI provider found. Running offline (parameter tweaks and blends only).\n"
                      "Set ANTHROPIC_API_KEY / OPENAI_API_KEY, install Claude Code, or run Ollama for AI breeding.[/]")
    label = MARKET_LABEL.get(args.market, args.market)
    search = getattr(args, "search", "v2")
    val_years = getattr(args, "validation_years", None)
    val_years = (0 if search == "v1" else 3.0) if val_years is None else val_years
    val_start = None
    if val_years > 0:
        val_start = panel.index[panel.index.searchsorted(hold_start - pd.DateOffset(years=val_years))]
        if (panel.index < val_start).sum() < 3 * 252:
            console.print("[red]Not enough history for a validation window; use --validation-years 0.[/]")
            return 1
    with Lab(panel, hold_start, workers=args.lab_workers, validation_start=val_start) as lab:
        exam = fitness.SealedExam(lab, budget=args.exam_budget, alpha=getattr(args, "alpha", 0.05))
        cfg = Config(islands=args.islands, island_size=args.island_size, offspring=args.offspring,
                     generations=args.generations, workers=args.workers, seed=args.seed,
                     team_size=0 if getattr(args, "no_team", False) else 5, search=search)
        evo = Evolution(lab, exam, provider, label, cfg, strong=strong)
        view = LiveView(evo, label)
        evo.on_event = view
        with view:
            champ = evo.run()
        if champ is not None and champ.exam in ("PASS", "FAIL") and not getattr(args, "no_reveal", False):
            from rich.live import Live
            from .cinema import exam_reveal
            star = evo.team if (evo.team is not None and evo.team.exam == "PASS") else champ
            rev = exam.reveal(star.id, star.code)
            span = f"{hold_start.strftime('%b %Y')} – {panel.index[-1].strftime('%b %Y')}"
            console.print()
            with Live(console=console, auto_refresh=False) as live:
                exam_reveal(live, star.name, span, rev["market_equity"].values, rev, star.exam, exam.bar,
                            min(console.width, 118), time.sleep)
        out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
        with console.status("[cyan]Writing the morning report…"):
            path = report.write(evo, out, label, hold_start)
            meta = json.loads((out / "meta.json").read_text())
            meta.update(market=args.market, tickers=getattr(args, "tickers", None), start=getattr(args, "start", None),
                        end=getattr(args, "end", None), seed=args.seed, alpha=getattr(args, "alpha", 0.05), search=search,
                        validation_start=None if val_start is None else str(val_start.date()),
                        champion_id=champ.id if champ else None, team_id=evo.team.id if evo.team else None)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
    passed = [a for a in exam.attempts if a["verdict"] == "PASS"]
    console.print()
    console.print(f"[bold]Champion:[/] {champ.name}  (training fitness {champ.fitness:+.2f})")
    if evo.team is not None:
        console.print(f"[bold]Team:[/] {len(evo.team.parents)} strategies  (training fitness {evo.team.fitness:+.2f}, "
                      f"exam {evo.team.exam or 'not taken'})")
    console.print(("[bold green]" if passed else "[bold yellow]") +
                  f"{len(passed)} of {len(exam.attempts)} exam attempts passed the sealed holdout.[/]")
    console.print(f"[bold]Report:[/] {path}")
    console.print(f"[dim]Champion code: {out / 'champion.py'}[/]")
    return 0


def cmd_backtest(args):
    panel = _load(args)
    hold = _holdout_start(panel, args.holdout_years, args.market)
    code = Path(args.file).read_text()
    with Lab(panel, hold, workers=1) as lab:
        try:
            res = lab.run(code, "train")
        except StrategyError as e:
            console.print(f"[red]Rejected:[/] {e}")
            return 1
        s = engine.stats(res.returns, res.turnover, res.gross, lab.mkt["train"])
    console.print(f"Training period {lab.train.index[0].date()} → {lab.train.index[-1].date()}")
    console.print(f"Sharpe {s.sharpe:.2f} · alpha {s.alpha:+.1%}/yr · CAGR {s.cagr:+.1%} · vol {s.vol:.1%} · "
                  f"max drawdown {s.max_dd:.0%} · turnover {s.turnover:.1f}x/yr · beta {s.beta:.2f} · "
                  f"fitness {fitness.fitness(s, code):+.2f}")
    for e in s.eras:
        console.print(f"  {e['start'][:4]}–{e['end'][:4]}: Sharpe {e['sharpe']:+.2f}, appraisal {e['appraisal']:+.2f}")
    console.print("[dim]The holdout is not touched by `backtest`. Use `evolve` to sit the sealed exam.[/]")
    return 0


def cmd_replay(args):
    from .cinema import Cinema
    run = Path(args.run)
    if not (run / "population.json").exists():
        console.print(f"[red]{run} is not an AlphaFinch run directory[/]")
        return 1
    meta_f = run / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if "holdout_start" not in meta or args.market:          # older runs: rebuild settings from flags
        market = args.market or meta.get("market", "us")
        panel = data.load(market)
        hs = _holdout_start(panel, args.holdout_years, market)
        meta.update(market=market, holdout_start=str(hs.date()),
                    market_label=MARKET_LABEL.get(market, market), islands=meta.get("islands", 4))
        meta_f.write_text(json.dumps(meta, indent=1))
    Cinema(run, speed=args.speed, width=args.width).run()
    return 0


def cmd_forward(args):
    from . import forward
    if args.sec_contact:
        os.environ["ALPHAFINCH_SEC_CONTACT"] = args.sec_contact
    if args.action == "freeze":
        if not args.run:
            console.print("[red]usage: alphafinch forward freeze runs/<timestamp>[/]")
            return 1
        added = forward.freeze(args.run, args.dir)
        for e in added:
            console.print(f"[green]✓[/] froze {e['name']} (judged on data after {e['frozen_through']})")
        if not added:
            console.print("[dim]Nothing new to freeze.[/]")
        return 0
    rows = forward.score(args.dir)
    for r in rows:
        if r["grade"] == "WAITING":
            console.print(f"[dim]{r['name']}: {r['days']} new trading days since {r['frozen_through']}, waiting[/]")
            continue
        colour = {"PASS": "green", "PROMISING": "#9a6700", "FAIL": "red"}[r["grade"]]
        console.print(f"[bold]{r['name']}[/] · {r['days']} days after {r['frozen_through']} · "
                      f"return {r['return']:+.1%} vs market {r['market_return']:+.1%} · alpha {r['alpha']:+.1%}/yr · "
                      f"t {r['alpha_t']:+.2f} (bar {r['bar']:.2f}) · [{colour}]{r['grade']}[/]")
    return 0


def cmd_world_exam(args):
    from . import world
    strategies = {}
    for f in args.files:
        code = Path(f).read_text()
        strategies[Path(f).stem] = code
    markets = [m.strip() for m in args.markets.split(",")] if args.markets != "all" else None
    console.print(f"[bold]World exam[/] · {len(strategies)} strategies · "
                  f"{len(markets or WORLD)} markets · from {args.start}"
                  + (f" · {args.prior_looks} earlier looks counted" if args.prior_looks else ""))
    with console.status("[cyan]Running…") as stt:
        res = world.exam(strategies, markets, args.start, args.end, args.alpha, args.prior_looks,
                         workers=args.lab_workers, progress=lambda n, m: stt.update(f"[cyan]{n} on {m}…"))
    colours = {"PASS": "green", "PROMISING": "#9a6700", "FAIL": "red", "NOT RUN": "dim"}
    for r in res:
        console.print()
        console.print(f"[bold]{r['name']}[/]")
        for mk, v in r["markets"].items():
            if "error" in v:
                console.print(f"  {mk:10s} [dim]not run: {v['error']}[/]")
            else:
                console.print(f"  {mk:10s} alpha {v['alpha']:+6.1%}/yr  t {v['alpha_t']:+5.2f}  beta {v['beta']:+.2f}  "
                              f"return {v['return']:+6.1%} vs market {v['market_return']:+6.1%}")
        if r["grade"] == "NOT RUN":
            console.print("  [dim]could not run in any market[/]")
            continue
        console.print(f"  [bold]pooled[/]     alpha {r['pooled_alpha']:+6.1%}/yr  t {r['pooled_t']:+5.2f}  "
                      f"(bar {r['bar']:.2f}, K = {r['K']})  positive in {r['positive_markets']}/{r['n_markets']} markets  "
                      f"[{colours[r['grade']]}]{r['grade']}[/]")
    if args.json:
        Path(args.json).write_text(json.dumps([{k: v for k, v in r.items() if k != "pooled"} | {
            "markets": {mk: {k: x for k, x in v.items() if k != "active"} for mk, v in r["markets"].items()}}
            for r in res], indent=1, default=float))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="alphafinch", description="Evolve trading strategies with AI, honestly.")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--market", default="us", choices=list(MARKET_LABEL))
        sp.add_argument("--tickers", help="comma-separated Yahoo symbols (overrides the market's universe)")
        sp.add_argument("--start", help="first date to use, e.g. 2012-01-01")
        sp.add_argument("--holdout-years", type=float, default=None,
                        help="years sealed away for the exam (default 3; 1.5 for crypto)")
        sp.add_argument("--sec-contact", help='"Name email@domain" sent to the SEC to fetch US fundamentals')
        sp.add_argument("--end", help="ignore all data after this date, e.g. 2023-10-01 (research on past periods)")

    ev = sub.add_parser("evolve", help="evolve strategies")
    common(ev)
    ev.add_argument("--provider", default="auto",
                    choices=["auto", "anthropic", "openai", "ollama", "compatible", "claude-code", "none"])
    ev.add_argument("--model")
    ev.add_argument("--strong-model", help="model for crossovers and new ideas, e.g. opus")
    ev.add_argument("--base-url")
    ev.add_argument("--api-key")
    ev.add_argument("--generations", type=int, default=20)
    ev.add_argument("--islands", type=int, default=4)
    ev.add_argument("--island-size", type=int, default=8)
    ev.add_argument("--offspring", type=int, default=4, help="children per island per generation")
    ev.add_argument("--exam-budget", type=int, default=10)
    ev.add_argument("--alpha", type=float, default=0.05, help="false-certification rate of the sealed exam")
    ev.add_argument("--workers", type=int, default=6, help="parallel AI requests")
    ev.add_argument("--lab-workers", type=int, default=4, help="parallel backtest processes")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default="runs")
    ev.add_argument("--no-team", action="store_true", help="skip building a team of survivors at the end")
    ev.add_argument("--no-reveal", action="store_true", help="skip the animated exam reveal at the end")
    ev.add_argument("--search", default="v2", choices=["v2", "v1"],
                    help="v2 (default): alpha fitness, random-halves check, validation, toolkit. v1: the original search")
    ev.add_argument("--validation-years", type=float, default=None,
                    help="years before the holdout used to choose the champion, never for breeding (default 3; 0 = off)")

    dm = sub.add_parser("demo", help="offline demo on a synthetic market (no AI, no network)")
    dm.add_argument("--generations", type=int, default=12)
    dm.add_argument("--out", default="runs")

    bt = sub.add_parser("backtest", help="backtest one strategy file")
    bt.add_argument("file")
    common(bt)

    rp = sub.add_parser("replay", help="re-animate a finished run as a short cinematic story")
    rp.add_argument("run", help="run directory, e.g. runs/20261003-231730")
    rp.add_argument("--speed", type=float, default=1.0)
    rp.add_argument("--width", type=int, default=118)
    rp.add_argument("--market", help="only for runs saved before meta.json existed")
    rp.add_argument("--holdout-years", type=float, default=None)

    fw = sub.add_parser("forward", help="forward test: freeze strategies now, judge them on future data")
    fw.add_argument("action", choices=["freeze", "score"])
    fw.add_argument("run", nargs="?", help="run directory to freeze")
    fw.add_argument("--dir", default="forward", help="where frozen strategies live")
    fw.add_argument("--sec-contact", help='"Name email@domain" sent to the SEC to fetch US fundamentals')

    we = sub.add_parser("world-exam", help="test frozen strategies on many markets they have never seen")
    we.add_argument("files", nargs="+", help="strategy .py files")
    we.add_argument("--markets", default="all", help=f"comma-separated, default all: {','.join(WORLD)}")
    we.add_argument("--start", default="2017-10-02", help="first day of the exam window")
    we.add_argument("--end", help="last day of the exam window (default: latest data)")
    we.add_argument("--alpha", type=float, default=0.05)
    we.add_argument("--prior-looks", type=int, default=0,
                    help="strategies tested on these markets and years before (raises the bar)")
    we.add_argument("--lab-workers", type=int, default=4)
    we.add_argument("--json", help="also write results to this file")

    sub.add_parser("markets", help="list built-in markets")

    args = p.parse_args(argv)
    if args.cmd == "evolve":
        return cmd_evolve(args)
    if args.cmd == "demo":
        ns = argparse.Namespace(market="synthetic", tickers=None, start=None, holdout_years=5.0, provider="none",
                                model=None, strong_model=None, base_url=None, api_key=None,
                                generations=args.generations, islands=4, island_size=8, offspring=4, exam_budget=10,
                                workers=6, lab_workers=4, seed=0, out=args.out, sec_contact=None,
                                validation_years=0)
        return cmd_evolve(ns, demo=True)
    if args.cmd == "backtest":
        return cmd_backtest(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    if args.cmd == "forward":
        return cmd_forward(args)
    if args.cmd == "world-exam":
        return cmd_world_exam(args)
    if args.cmd == "markets":
        for k, v in MARKET_LABEL.items():
            console.print(f"[bold]{k:11s}[/] {v}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
