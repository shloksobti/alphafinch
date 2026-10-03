"""alphafinch command line.

  alphafinch demo                       offline demo, no AI and no network (about 1 minute)
  alphafinch evolve --market us         evolve strategies (provider auto-detected)
  alphafinch backtest my_strategy.py    backtest one strategy file on the training period
  alphafinch markets                    list built-in markets
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console

from . import data, engine, fitness, llm, report, sandbox
from .evolve import Config, Evolution
from .ui import LiveView

console = Console()
MARKET_LABEL = {"us": "US large-cap stocks", "india": "Indian large-cap stocks (NSE)", "crypto": "Crypto vs USDT",
                "industries": "49 US industry portfolios (Ken French)", "synthetic": "Synthetic market (offline)"}


def _load(args) -> pd.DataFrame:
    tickers = [t.strip() for t in args.tickers.split(",")] if getattr(args, "tickers", None) else None
    with console.status(f"[cyan]Loading {MARKET_LABEL.get(args.market, args.market)}…") as st:
        px = data.load(args.market, tickers, getattr(args, "start", None),
                       progress=lambda i, n, s: st.update(f"[cyan]Downloading {s} ({i + 1}/{n})…"))
    console.print(f"[green]✓[/] {px.shape[1]} assets, {px.index[0].date()} → {px.index[-1].date()}")
    return px


def _split(px: pd.DataFrame, holdout_years: float | None, market: str = "us"):
    holdout_years = holdout_years or (1.5 if market == "crypto" else 3.0)
    start = px.index[-1] - pd.DateOffset(years=holdout_years)
    start = px.index[px.index.searchsorted(start)]
    return px[px.index < start], start


def cmd_evolve(args, demo=False):
    px = _load(args)
    train, hold_start = _split(px, args.holdout_years, args.market)
    if len(train) < 3 * 252:
        console.print("[red]Not enough training history (need 3+ years before the holdout).[/]")
        return 1
    provider_name = "none" if demo else (llm.auto() if args.provider == "auto" else args.provider)
    try:
        provider = llm.get(provider_name, args.model, args.base_url, args.api_key)
    except (llm.LLMError, ImportError) as e:
        console.print(f"[red]Could not start provider '{provider_name}': {e}[/]")
        return 1
    if provider is None and not demo:
        console.print("[yellow]No AI provider found. Running offline (parameter tweaks and blends only).\n"
                      "Set ANTHROPIC_API_KEY / OPENAI_API_KEY, install Claude Code, or run Ollama for AI breeding.[/]")
    exam = fitness.SealedExam(px, hold_start, budget=args.exam_budget)
    cfg = Config(islands=args.islands, island_size=args.island_size, offspring=args.offspring,
                 generations=args.generations, workers=args.workers, seed=args.seed)
    label = MARKET_LABEL.get(args.market, args.market)
    evo = Evolution(train, exam, provider, label, cfg)
    view = LiveView(evo, label)
    evo.on_event = view
    with view:
        champ = evo.run()
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    with console.status("[cyan]Writing the morning report…"):
        path = report.write(evo, out, label, train, hold_start)
        import json
        meta = json.loads((out / "meta.json").read_text())
        meta.update(market=args.market, tickers=getattr(args, "tickers", None), start=getattr(args, "start", None))
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    passed = [a for a in exam.attempts if a["verdict"] == "PASS"]
    console.print()
    console.print(f"[bold]Champion:[/] {champ.name}  (training fitness {champ.fitness:+.2f})")
    console.print(("[bold green]" if passed else "[bold yellow]") +
                  f"{len(passed)} of {len(exam.attempts)} exam attempts passed the sealed holdout.[/]")
    console.print(f"[bold]Report:[/] {path}")
    console.print(f"[dim]Champion code: {out / 'champion.py'}[/]")
    return 0


def cmd_backtest(args):
    px = _load(args)
    train, _ = _split(px, args.holdout_years, args.market)
    code = Path(args.file).read_text()
    try:
        _, s = engine.backtest(code, train)
    except sandbox.StrategyError as e:
        console.print(f"[red]Rejected:[/] {e}")
        return 1
    console.print(f"Training period {train.index[0].date()} → {train.index[-1].date()}")
    console.print(f"Sharpe {s.sharpe:.2f} · CAGR {s.cagr:+.1%} · vol {s.vol:.1%} · max drawdown {s.max_dd:.0%} · "
                  f"turnover {s.turnover:.1f}x/yr · beta {s.beta:.2f} · fitness {fitness.fitness(s, code):+.2f}")
    console.print("[dim]The holdout is not touched by `backtest`. Use `evolve` to sit the sealed exam.[/]")
    return 0


def cmd_replay(args):
    import json
    from .cinema import Cinema
    run = Path(args.run)
    if not (run / "population.json").exists():
        console.print(f"[red]{run} is not an AlphaFinch run directory[/]")
        return 1
    meta_f = run / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if "holdout_start" not in meta or args.market:          # older runs: rebuild settings from flags
        market = args.market or meta.get("market", "us")
        px = data.load(market)
        _, hs = _split(px, args.holdout_years, market)
        meta.update(market=market, holdout_start=str(hs.date()),
                    market_label=MARKET_LABEL.get(market, market), islands=meta.get("islands", 4))
        meta_f.write_text(json.dumps(meta, indent=1))
    Cinema(run, speed=args.speed, width=args.width).run()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="alphafinch", description="Evolve trading strategies with AI, honestly.")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--market", default="us", choices=["us", "india", "crypto", "industries", "synthetic"])
        sp.add_argument("--tickers", help="comma-separated Yahoo symbols (overrides the market's universe)")
        sp.add_argument("--start", help="first date to use, e.g. 2005-01-01")
        sp.add_argument("--holdout-years", type=float, default=None,
                        help="years sealed away for the exam (default 3; 1.5 for crypto)")

    ev = sub.add_parser("evolve", help="evolve strategies")
    common(ev)
    ev.add_argument("--provider", default="auto",
                    choices=["auto", "anthropic", "openai", "ollama", "compatible", "claude-code", "none"])
    ev.add_argument("--model")
    ev.add_argument("--base-url")
    ev.add_argument("--api-key")
    ev.add_argument("--generations", type=int, default=20)
    ev.add_argument("--islands", type=int, default=4)
    ev.add_argument("--island-size", type=int, default=8)
    ev.add_argument("--offspring", type=int, default=4, help="children per island per generation")
    ev.add_argument("--exam-budget", type=int, default=10)
    ev.add_argument("--workers", type=int, default=6)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default="runs")

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

    sub.add_parser("markets", help="list built-in markets")

    args = p.parse_args(argv)
    if args.cmd == "evolve":
        return cmd_evolve(args)
    if args.cmd == "demo":
        ns = argparse.Namespace(market="synthetic", tickers=None, start=None, holdout_years=5.0, provider="none",
                                model=None, base_url=None, api_key=None, generations=args.generations, islands=4,
                                island_size=8, offspring=4, exam_budget=10, workers=6, seed=0, out=args.out)
        return cmd_evolve(ns, demo=True)
    if args.cmd == "backtest":
        return cmd_backtest(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    if args.cmd == "markets":
        for k, v in MARKET_LABEL.items():
            console.print(f"[bold]{k:11s}[/] {v}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
