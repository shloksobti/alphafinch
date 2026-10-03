"""The evolutionary loop.

Population structure (the Galapagos model)
  * N islands evolve independently; every `migrate_every` generations each island's champion
    migrates to the next island.
  * Each island keeps a niche map over (trading speed x market exposure). The best strategy in
    every occupied niche always survives, so the population cannot collapse onto one idea
    (the MAP-Elites idea that AlphaEvolve also uses). Remaining slots go to the fittest.

Operators
  mutate     AI rewrites one parent, guided by its training report card
  crossover  AI combines the best ideas of two parents into one child
  immigrant  AI invents a new strategy around a fresh theme
  tweak      no AI: nudge one numeric constant (fine-tuning)
  blend      no AI: child holds a mix of both parents' portfolios

Selection uses training data only. Champions sit the sealed exam (holdout, PASS/FAIL only).
"""
from __future__ import annotations

import ast
import re
import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import engine, fitness as fit, prompts, sandbox
from .llm import LLMError, Provider
from .seeds import IMMIGRANT_THEMES, SEEDS

_ids = itertools.count(1)


@dataclass
class Individual:
    id: str
    name: str
    code: str
    op: str
    parents: tuple = ()
    gen: int = 0
    island: int = 0
    fitness: float = float("-inf")
    stats: object = None
    error: str | None = None
    exam: str | None = None

    @property
    def niche(self) -> tuple:
        s = self.stats
        speed = 0 if s.turnover < 4 else (1 if s.turnover < 15 else 2)
        expo = 0 if abs(s.beta) < 0.3 else (1 if abs(s.beta) < 0.7 else 2)
        return speed, expo


def new_id() -> str:
    return f"f{next(_ids):04d}"


_VER = re.compile(r"(\s+(v\d+|#\d+|[IVX]{1,5}))+$")


def base_name(name: str) -> str:
    """Strip version suffixes: 'Storm Shelter v7' -> 'Storm Shelter'."""
    return _VER.sub("", name).strip()


def family(name: str) -> str:
    """The first parent's family: 'Calm Seeker x Storm Shelter v3' -> 'Calm Seeker'."""
    return base_name(base_name(name).split(" x ")[0])


# ---------------------------------------------------------------------------- no-AI operators
def tweak(code: str, rng: random.Random) -> str | None:
    """Multiply one UPPER_CASE numeric constant by a random factor in [0.7, 1.4]."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]
            tup = [t for t in node.targets if isinstance(t, ast.Tuple)]
            if names and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                targets.append(node.value)
            for t in tup:
                if all(isinstance(e, ast.Name) and e.id.isupper() for e in t.elts) and isinstance(node.value, ast.Tuple):
                    targets += [v for v in node.value.elts if isinstance(v, ast.Constant)
                                and isinstance(v.value, (int, float)) and not isinstance(v.value, bool)]
    if not targets:
        return None
    c = rng.choice(targets)
    old = c.value
    f = rng.uniform(0.7, 1.4)
    new = max(1, int(round(old * f))) if isinstance(old, int) else round(old * f, 4)
    if new == old:
        new = old + 1 if isinstance(old, int) else round(old * 1.1, 4)
    lines = code.splitlines(keepends=True)
    ln = lines[c.lineno - 1]
    lines[c.lineno - 1] = ln[:c.col_offset] + repr(new) + ln[c.end_col_offset:]
    return "".join(lines)


def blend(a: "Individual", b: "Individual", rng: random.Random) -> str:
    mix = round(rng.uniform(0.3, 0.7), 2)
    fa, fb = f"_{a.id}", f"_{b.id}"
    ca = a.code.replace("def strategy(", f"def {fa}(", 1)
    cb = b.code.replace("def strategy(", f"def {fb}(", 1)
    return f'''{ca.strip()}


{cb.strip()}


def strategy(prices):
    """{family(a.name)} x {family(b.name)}: a blend of both parents' portfolios."""
    MIX = {mix}
    wa = {fa}(prices).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    wb = {fb}(prices).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return MIX * wa + (1 - MIX) * wb
'''


# ---------------------------------------------------------------------------- evolution
@dataclass
class Config:
    islands: int = 4
    island_size: int = 8
    offspring: int = 4               # children per island per generation
    generations: int = 20
    migrate_every: int = 4
    workers: int = 6
    exam_margin: float = 0.05        # a new champion must beat the last examined one by this much
    seed: int = 0


class Evolution:
    def __init__(self, train: pd.DataFrame, exam: fit.SealedExam, provider: Provider | None,
                 market: str, cfg: Config, on_event=None):
        self.train, self.exam, self.llm, self.market, self.cfg = train, exam, provider, market, cfg
        self.rng = random.Random(cfg.seed)
        self.islands: list[list[Individual]] = [[] for _ in range(cfg.islands)]
        self.all: dict[str, Individual] = {}
        self.gen = 0
        self.on_event = on_event or (lambda kind, **kw: None)
        self.examined_best = float("-inf")
        self._versions: dict[str, int] = {}
        self.llm_calls = 0
        self.failures = 0

    # -- evaluation -------------------------------------------------------------------------
    def evaluate(self, ind: Individual) -> Individual:
        try:
            _, s = engine.backtest(ind.code, self.train)
            ind.stats = s
            ind.fitness = fit.fitness(s, ind.code)
        except sandbox.StrategyError as e:
            ind.error = str(e).splitlines()[0][:160]
        except Exception as e:  # defensive: never let one child crash the run
            ind.error = f"{type(e).__name__}: {e}"[:160]
        return ind

    def _ask(self, prompt: str) -> str | None:
        self.llm_calls += 1
        try:
            return prompts.extract_code(self.llm.complete(prompts.SYSTEM, prompt))
        except LLMError as e:
            self.on_event("llm_error", error=str(e))
            return None

    # -- reproduction -----------------------------------------------------------------------
    def _version(self, name: str) -> str:
        b = base_name(name)
        self._versions[b] = self._versions.get(b, 1) + 1
        return f"{b} v{self._versions[b]}"

    def _pick(self, island: list[Individual], k: int = 3) -> Individual:
        pool = self.rng.sample(island, min(k, len(island)))
        return max(pool, key=lambda i: i.fitness)

    def _make_child(self, isl: int, op: str) -> Individual | None:
        pop = self.islands[isl]
        a = self._pick(pop)
        if op == "tweak":
            code, parents, name = tweak(a.code, self.rng), (a.id,), self._version(a.name)
        elif op == "blend":
            others = [p for p in pop if family(p.name) != family(a.name)]
            b = self._pick(others or [p for p in pop if p.id != a.id] or pop)
            code, parents = blend(a, b, self.rng), (a.id, b.id)
            name = f"{family(a.name)} x {family(b.name)}"
        elif op == "mutate":
            code = self._ask(prompts.mutate(a.code, prompts.report_card(a.name, a.stats), self.market))
            parents, name = (a.id,), None
        elif op == "crossover":
            b = self._pick([p for p in pop if p.id != a.id] or pop)
            code = self._ask(prompts.crossover(a.code, prompts.report_card(a.name, a.stats),
                                               b.code, prompts.report_card(b.name, b.stats), self.market))
            parents, name = (a.id, b.id), None
        else:  # immigrant
            code = self._ask(prompts.immigrant(self.rng.choice(IMMIGRANT_THEMES), self.market))
            parents, name = (), None
        if not code:
            return None
        name = name or prompts.strategy_name(code, fallback=f"Finch {next(_ids)}")
        child = Individual(new_id(), name, code, op, parents, self.gen, isl)
        return self.evaluate(child)

    def _ops(self) -> list[str]:
        if self.llm is None:
            return self.rng.choices(["tweak", "blend"], weights=[0.65, 0.35], k=self.cfg.offspring)
        ops = self.rng.choices(["mutate", "crossover", "tweak", "blend"], weights=[0.45, 0.25, 0.2, 0.1],
                               k=self.cfg.offspring)
        if self.rng.random() < 0.35:
            ops[-1] = "immigrant"
        return ops

    # -- survival ---------------------------------------------------------------------------
    def _survive(self, pop: list[Individual]) -> list[Individual]:
        alive, seen = [], set()
        for p in sorted(pop, key=lambda p: -p.fitness):   # identical code survives only once
            key = " ".join(p.code.split())
            if p.error is None and np.isfinite(p.fitness) and key not in seen:
                seen.add(key)
                alive.append(p)
        best_in_niche: dict[tuple, Individual] = {}
        for p in alive:
            if p.niche not in best_in_niche or p.fitness > best_in_niche[p.niche].fitness:
                best_in_niche[p.niche] = p
        keep = sorted(best_in_niche.values(), key=lambda p: -p.fitness)[: self.cfg.island_size]
        rest = sorted([p for p in alive if p not in keep], key=lambda p: -p.fitness)
        return keep + rest[: self.cfg.island_size - len(keep)]

    # -- main loop --------------------------------------------------------------------------
    def seed_population(self):
        seeds = [Individual(new_id(), name, code.strip() + "\n", "seed") for name, code in SEEDS.items()]
        with ThreadPoolExecutor(self.cfg.workers) as ex:
            seeds = list(ex.map(self.evaluate, seeds))
        for i in range(self.cfg.islands):
            for s in seeds:
                c = Individual(new_id(), s.name, s.code, "seed", (), 0, i, s.fitness, s.stats, s.error)
                self.islands[i].append(c)
                self.all[c.id] = c
            self.islands[i] = self._survive(self.islands[i])
        self.on_event("seeded", best=self.champion())

    def champion(self) -> Individual | None:
        alive = [p for isl in self.islands for p in isl]
        return max(alive, key=lambda p: p.fitness) if alive else None

    def step(self):
        self.gen += 1
        jobs = [(i, op) for i in range(self.cfg.islands) for op in self._ops()]
        with ThreadPoolExecutor(self.cfg.workers) as ex:
            children = list(ex.map(lambda j: self._make_child(*j), jobs))
        for c in children:
            if c is None:
                self.failures += 1
                continue
            self.all[c.id] = c
            if c.error:
                self.failures += 1
                self.on_event("stillborn", child=c)
                continue
            self.islands[c.island].append(c)
            self.on_event("born", child=c, parents=[self.all[p] for p in c.parents if p in self.all])
        for i in range(self.cfg.islands):
            before = {p.id for p in self.islands[i]}
            self.islands[i] = self._survive(self.islands[i])
            died = before - {p.id for p in self.islands[i]}
            self.on_event("selection", island=i, died=len(died))
        if self.cfg.migrate_every and self.gen % self.cfg.migrate_every == 0:
            for i in range(self.cfg.islands):
                champ = max(self.islands[i], key=lambda p: p.fitness)
                j = (i + 1) % self.cfg.islands
                m = Individual(new_id(), champ.name, champ.code, "migrant", (champ.id,), self.gen, j,
                               champ.fitness, champ.stats)
                self.all[m.id] = m
                self.islands[j] = self._survive(self.islands[j] + [m])
            self.on_event("migration")
        champ = self.champion()
        if champ and champ.exam is None and champ.fitness > self.examined_best + self.cfg.exam_margin \
                and self.exam.left > 1:  # keep one attempt for the final champion
            self._sit(champ)
        self.on_event("generation", gen=self.gen, best=champ)

    def _sit(self, ind: Individual):
        ind.exam = self.exam.sit(ind.id, ind.code)
        self.examined_best = max(self.examined_best, ind.fitness)
        self.on_event("exam", ind=ind)

    def run(self):
        t0 = time.time()
        self.seed_population()
        for _ in range(self.cfg.generations):
            self.step()
        champ = self.champion()
        if champ and champ.exam is None and self.exam.left > 0:
            self._sit(champ)
        self.on_event("done", seconds=time.time() - t0)
        return champ

    def lineage(self, ind: Individual, depth: int = 12) -> list[Individual]:
        out, cur = [], ind
        while cur and len(out) < depth:
            out.append(cur)
            cur = self.all.get(cur.parents[0]) if cur.parents else None
        return out
