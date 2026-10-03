"""`alphafinch replay`: re-animate a finished run as a short cinematic story.

Every frame is reconstructed from the run's saved population (code, lineage, operators) and
from re-running the strategies on the same data, so nothing is invented. Scenes:
  1. title card
  2. births: the AI writing new strategies (changed lines highlighted) while their training
     equity curves draw against their parent and the market; new champions are crowned
  3. the sealed exam: the hidden years are unsealed and the verdict revealed
  4. end card
"""
from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import data, engine, fitness, sandbox
from .braille import chart
from .ui import OP_ICON

OP_VERB = {"mutate": "mutating", "crossover": "crossing", "immigrant": "inventing", "tweak": "tuning",
           "blend": "blending", "seed": "planting"}


class Cinema:
    def __init__(self, run_dir: Path, speed: float = 1.0, width: int = 118):
        self.dir = Path(run_dir)
        self.speed = speed
        self.pop = json.loads((self.dir / "population.json").read_text())
        self.by_id = {p["id"]: p for p in self.pop}
        meta_f = self.dir / "meta.json"
        self.meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        self.console = Console(width=width)
        self.width = width

    # helpers ---------------------------------------------------------------------------------
    def _sleep(self, s):
        time.sleep(s / self.speed)

    def _equity(self, code, px):
        r, _ = engine.backtest(code, px, check_leaks=False)
        return (1 + r).cumprod().values

    def _code_panel(self, ind, parent_code, shown_lines, title_extra=""):
        lines = ind["code"].rstrip().splitlines()
        new = set()
        if parent_code:
            pl = [l.strip() for l in parent_code.splitlines()]
            new = {i for i, l in enumerate(lines) if l.strip() and l.strip() not in pl}
        else:
            new = set(range(len(lines)))
        body = Text()
        for i, l in enumerate(lines[:shown_lines][-16:]):
            idx = i + max(0, shown_lines - 16)
            mark, style = ("+ ", "bold green") if idx in new else ("  ", "grey70")
            body.append(mark, style="green" if idx in new else "grey35")
            body.append(l[:60] + "\n", style=style)
        if shown_lines < len(lines):
            body.append("▌", style="bold white blink")
        verb = OP_VERB.get(ind["op"], ind["op"])
        parents = " + ".join(self.by_id[p]["name"] for p in ind["parents"] if p in self.by_id) or "a new idea"
        return Panel(body, title=f"{OP_ICON.get(ind['op'], '•')}  the AI is {verb}: [bold]{ind['name']}[/]{title_extra}",
                     subtitle=f"[grey62]from {parents}[/]", border_style="magenta", height=20, width=68)

    def _chart_panel(self, curves, upto, title, legend):
        c = chart(curves, width=self.width - 72, height=15, upto=upto)
        leg = Text()
        for name, colour in legend:
            leg.append("━━ ", style=colour)
            leg.append(name + "   ", style="white")
        return Panel(Group(c, Text(""), leg), title=title, border_style="cyan", height=20, width=self.width - 68)

    def _header(self, gen, bred, champ_name, champ_fit):
        t = Text.assemble(("🐦 AlphaFinch  ", "bold magenta"),
                          (f"{self.meta.get('market_label', 'market')}  ", "bold"),
                          (f"generation {gen}  ", "cyan"), (f"strategies bred {bred}  ", "white"),
                          ("👑 ", ""), (f"{champ_name} ", "bold green"), (f"{champ_fit:+.2f}", "green"))
        return Panel(t, border_style="magenta", width=self.width)

    # scenes ----------------------------------------------------------------------------------
    def _prepare(self):
        market = self.meta.get("market", "us")
        tickers = self.meta.get("tickers")
        px = data.load(market, tickers.split(",") if tickers else None, self.meta.get("start"))
        hold = pd.Timestamp(self.meta["holdout_start"])
        train = px[px.index < hold]
        self.mkt_train = (1 + train.pct_change().fillna(0).mean(axis=1)).cumprod().values
        ordered = sorted([p for p in self.pop if p["fitness"] is not None], key=lambda p: p["id"])
        self.seeds = [p for p in ordered if p["op"] == "seed"]
        self.children = [p for p in ordered if p["op"] not in ("seed", "migrant")]
        self.stillborn = [p for p in self.pop if p["error"]]
        best, rec = max(s["fitness"] for s in self.seeds), []
        ai_ops = ("mutate", "crossover", "immigrant")
        for c in self.children:                     # every new record holder
            if c["fitness"] > best + 1e-9:
                best = c["fitness"]
                rec.append(c["id"])
        ai_rec = [r for r in rec if self.by_id[r]["op"] in ai_ops]
        self.featured = (ai_rec or rec)[-3:]
        self.champ = max(ordered, key=lambda p: p["fitness"])
        self.eq = {}
        for cid in self.featured:
            c = self.by_id[cid]
            par = self.by_id.get(c["parents"][0]) if c["parents"] else None
            self.eq[cid] = self._equity(c["code"], train)
            if par:
                self.eq[par["id"]] = self._equity(par["code"], train)
        self.exam = fitness.SealedExam(px, hold, budget=self.meta.get("exam_budget", 10))
        self.rev = self.exam.reveal(self.champ["id"], self.champ["code"])
        hold_px = px[px.index >= hold]
        self.mkt_hold = (1 + hold_px.pct_change().fillna(0).mean(axis=1)).cumprod().values
        self.span = f"{hold.strftime('%b %Y')} – {px.index[-1].strftime('%b %Y')}"
        v = next((e["verdict"] for e in self.meta.get("exam", []) if e["id"] == self.champ["id"]), None)
        self.verdict = v or ("PASS" if self.rev["alpha_t"] > self.exam.bar else "FAIL")

    def run(self):
        with self.console.status("[magenta]⏪ rewinding the run…"):
            self._prepare()
        islands = self.meta.get("islands", 4)
        model = self.meta.get("model") or self.meta.get("provider") or "AI"
        model = {"sonnet": "Claude Sonnet", "opus": "Claude Opus"}.get(model, model)
        feed: list[Text] = []

        def frame(*parts):
            return Group(*parts)

        with Live(console=self.console, auto_refresh=False, screen=False, transient=False) as live:
            # 1. title ---------------------------------------------------------------------------
            title = Group(Align.center(Text("🐦  AlphaFinch", style="bold magenta")),
                          Align.center(Text("AI breeds trading strategies. A sealed exam decides if they're real.")),
                          Text(""),
                          Align.center(Text(f"{self.meta.get('market_label', '')}  ·  {islands} islands  ·  bred by {model}",
                                            style="grey62")))
            live.update(Panel(title, border_style="magenta", width=self.width, padding=(1, 2)), refresh=True)
            self._sleep(2.0)

            # 2. births --------------------------------------------------------------------------
            cur = max(self.seeds, key=lambda p: p["fitness"])
            bred = 0
            for c in self.children:
                bred += 1
                line = Text.assemble((f" {OP_ICON.get(c['op'], '•')} ", ""), (f"{c['name']:<30}", "white"),
                                     (f"{c['op']:<10}", "grey50"), (f"fitness {c['fitness']:+.2f}", "grey62"))
                if c["id"] not in self.featured:
                    feed.append(line)
                    live.update(frame(self._header(c["gen"], bred, cur["name"], cur["fitness"]),
                                      Panel(Group(*feed[-12:]), title="🐣 strategies being born", border_style="grey35",
                                            width=self.width, height=14)), refresh=True)
                    self._sleep(0.09)
                    continue
                par = self.by_id.get(c["parents"][0]) if c["parents"] else None
                curves = [(self.mkt_train, "#6c7086")] + ([(self.eq[par["id"]], "#89b4fa")] if par else []) + [(self.eq[c["id"]], "bold #a6e3a1")]
                legend = [("market", "#6c7086")] + ([("parent", "#89b4fa")] if par else []) + [(c["name"][:22], "bold #a6e3a1")]
                n_lines = len(c["code"].splitlines())
                steps = 30
                row = None
                for k in range(1, steps + 1):
                    shown = int(n_lines * min(1, k / (steps * 0.55)))
                    upto = min(1.0, max(0.03, (k - steps * 0.3) / (steps * 0.6)))
                    row = Table.grid()
                    row.add_row(self._code_panel(c, par["code"] if par else None, shown),
                                self._chart_panel(curves, upto, "📈 growth of $1 (training years)", legend))
                    live.update(frame(self._header(c["gen"], bred, cur["name"], cur["fitness"]), row), refresh=True)
                    self._sleep(0.06)
                beat = f"  beats {par['name']} ({par['fitness']:+.2f})" if par else ""
                crown = Text.assemble(("  👑 NEW CHAMPION  ", "bold #1e1e2e on #f9e2af"), ("  ", ""), (c["name"], "bold #f9e2af"),
                                      (f"   fitness {c['fitness']:+.2f}{beat}", "#f9e2af"))
                cur = c
                live.update(frame(self._header(c["gen"], bred, cur["name"], cur["fitness"]), row, crown), refresh=True)
                self._sleep(1.8)
                feed.append(Text.assemble((" 👑 ", ""), (f"{c['name']:<30}", "bold #f9e2af"), (f"{c['op']:<10}", "grey50"),
                                          (f"fitness {c['fitness']:+.2f}", "#f9e2af")))

            # 3. the sealed exam -------------------------------------------------------------------
            for k in range(0, 21):
                bar = "█" * k + "░" * (20 - k)
                live.update(Panel(Group(Align.center(Text("🔒  THE SEALED EXAM", style="bold #f9e2af")),
                                        Align.center(Text(f"{self.champ['name']} has never seen {self.span}", style="grey62")),
                                        Text(""), Align.center(Text(f"unsealing  {bar}", style="#f9e2af"))),
                                  border_style="#f9e2af", width=self.width, padding=(1, 2)), refresh=True)
                self._sleep(0.05)
            curves = [(self.mkt_hold, "#6c7086"), (self.rev["equity"].values, "bold #a6e3a1")]
            leg = Text()
            for name, col in [("buy everything (equal weight)", "#6c7086"), (self.champ["name"], "bold #a6e3a1")]:
                leg.append("━━ ", style=col)
                leg.append(name + "    ")
            panel = None
            for k in range(1, 36):
                panel = Panel(Group(chart(curves, width=self.width - 6, height=12, upto=k / 35), Text(""), leg),
                              title=f"🔒 the sealed years, {self.span}: growth of $1", border_style="#f9e2af", width=self.width)
                live.update(panel, refresh=True)
                self._sleep(0.05)
            passed = self.verdict == "PASS"
            stamp = Text.assemble(("  ✅ PASS  " if passed else "  ❌ FAIL  ",
                                   "bold #ffffff on #1a7f37" if passed else "bold #ffffff on #cf222e"),
                                  (f"   sealed-years alpha {self.rev['alpha']:+.1%}/yr  ·  t = {self.rev['alpha_t']:.2f}  ·  bar {self.exam.bar:.2f}",
                                   "white"))
            moral = Text("  It beat the market by more than luck can explain." if passed else
                         f"  It made {self.rev['cagr']:+.0%} a year. So did buying everything. Its edge was luck.",
                         style="bold green" if passed else "bold #f9e2af")
            live.update(frame(panel, stamp, moral), refresh=True)
            self._sleep(3.5)

            # 4. end card ------------------------------------------------------------------------
            live.update(Panel(Group(Align.center(Text("🐦  AlphaFinch", style="bold magenta")),
                                    Align.center(Text("evolve strategies while you sleep · an exam they can't cheat")),
                                    Text(""), Align.center(Text("pip install alphafinch", style="bold cyan"))),
                              border_style="magenta", width=self.width, padding=(1, 2)), refresh=True)
            self._sleep(2.5)
