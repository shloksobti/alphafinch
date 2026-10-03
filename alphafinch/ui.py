"""Live terminal view of an evolution run (Rich)."""
from __future__ import annotations

import time
from collections import deque

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

OP_ICON = {"seed": "🌱", "mutate": "🧬", "crossover": "💞", "immigrant": "🛶", "tweak": "🔧",
           "blend": "🎨", "migrant": "✈️ "}
SPARK = "▁▂▃▄▅▆▇█"
NICHE_SPEED = ["slow", "medium", "fast"]
NICHE_EXPO = ["neutral", "partial", "market"]


def spark(xs, width=40):
    xs = list(xs)[-width:]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return SPARK[3] * len(xs)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7)] for x in xs)


class LiveView:
    def __init__(self, evo, title: str):
        self.evo, self.title = evo, title
        self.log = deque(maxlen=12)
        self.best_hist: list[float] = []
        self.t0 = time.time()
        self.live = Live(self.render(), refresh_per_second=8, screen=False, transient=False)

    # event hook -----------------------------------------------------------------------------
    def __call__(self, kind, **kw):
        e = self.evo
        if kind == "born":
            c = kw["child"]
            parents = " + ".join(p.name for p in kw.get("parents", [])) or "the wild"
            arrow = "[green]▲[/]" if c.fitness > max((p.fitness for p in kw.get("parents", [])), default=-9) else "[dim]·[/]"
            self.log.append(f"{OP_ICON.get(c.op, '•')} {arrow} [bold]{c.name}[/] [dim]({c.op} of {parents})[/] fit {c.fitness:+.2f}")
        elif kind == "stillborn":
            c = kw["child"]
            why = "peeked at the future" if "look-ahead" in (c.error or "") else (c.error or "")[:50]
            self.log.append(f"💀 [red]{c.name}[/] [dim]did not survive: {why}[/]")
        elif kind == "migration":
            self.log.append("✈️  [cyan]Champions migrate between islands[/]")
        elif kind == "exam":
            i = kw["ind"]
            colour = "green" if i.exam == "PASS" else "yellow"
            self.log.append(f"📝 [bold {colour}]{i.name} sat the sealed exam: {i.exam}[/] "
                            f"[dim]({e.exam.left} attempts left)[/]")
        elif kind == "llm_error":
            self.log.append(f"[red]AI error:[/] [dim]{kw['error'][:70]}[/]")
        elif kind in ("generation", "seeded"):
            b = kw.get("best")
            if b:
                self.best_hist.append(b.fitness)
        self.live.update(self.render())

    # rendering ------------------------------------------------------------------------------
    def _header(self):
        e = self.evo
        el = int(time.time() - self.t0)
        prov = e.llm.name + (f" · {e.llm.model}" if e.llm and e.llm.model else "") if e.llm else "no AI (offline)"
        t = Text.assemble(("🐦 AlphaFinch ", "bold magenta"), (f"  {self.title}  ", "bold"),
                          (f"generation {e.gen}/{e.cfg.generations}  ", "cyan"),
                          (f"strategies bred {len(e.all)}  ", ""),
                          (f"exam {e.exam.budget - e.exam.left}/{e.exam.budget}  ", "yellow"),
                          (f"{el // 60}m{el % 60:02d}s  ", "dim"), (prov, "dim"))
        return Panel(t, border_style="magenta")

    def _islands(self):
        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in self.evo.islands:
            grid.add_column(ratio=1)
        cells = []
        for i, isl in enumerate(self.evo.islands):
            t = Table(show_header=False, box=None, pad_edge=False, expand=True)
            t.add_column(ratio=3, no_wrap=True, overflow="ellipsis")
            t.add_column(justify="right")
            for p in sorted(isl, key=lambda p: -p.fitness)[:6]:
                t.add_row(f"{OP_ICON.get(p.op, '•')} {p.name}", f"{p.fitness:+.2f}")
            cells.append(Panel(t, title=f"🏝  Island {i + 1}", border_style="blue"))
        grid.add_row(*cells)
        return grid

    def _champion(self):
        c = self.evo.champion()
        if not c:
            return Panel("…", title="Champion")
        s = c.stats
        lin = self.evo.lineage(c, 6)
        tree = "  ←  ".join(f"{OP_ICON.get(x.op, '•')} {x.name}" for x in lin)
        body = Text.from_markup(
            f"[bold green]{c.name}[/]   fitness [bold]{c.fitness:+.2f}[/]   "
            f"train Sharpe {s.sharpe:.2f} · alpha {s.alpha:+.1%}/yr · CAGR {s.cagr:+.1%} · max drawdown {s.max_dd:.0%} · "
            f"{NICHE_SPEED[c.niche[0]]} trader · {NICHE_EXPO[c.niche[1]]} exposure\n"
            f"[dim]lineage:[/] {tree}\n"
            f"[dim]best fitness by generation:[/] [green]{spark(self.best_hist)}[/]")
        return Panel(body, title="👑 Champion (training data only)", border_style="green")

    def _log(self):
        return Panel(Group(*[Text.from_markup(x) for x in self.log]) if self.log else Text("warming up…"),
                     title="📜 Evolution log", border_style="white")

    def render(self):
        return Group(self._header(), self._islands(), self._champion(), self._log())

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *a):
        self.live.update(self.render())
        return self.live.__exit__(*a)
