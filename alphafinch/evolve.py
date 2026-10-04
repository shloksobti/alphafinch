"""The evolutionary loop.

Population structure (the Galapagos model)
  * N islands evolve independently; every `migrate_every` generations each island's champion
    migrates to the next island.
  * Each island keeps a niche map over (trading speed x market exposure). The best strategy in
    every occupied niche always survives, so the population cannot collapse onto one idea.

Operators
  mutate     AI rewrites one parent, guided by its report card and the lab notebook
  crossover  AI combines the best ideas of two parents (uses the strong model if configured)
  immigrant  AI invents a new hypothesis-driven strategy (uses the strong model if configured)
  tweak      no AI: nudge one numeric constant
  blend      no AI: child holds a mix of both parents' portfolios

Robustness: a promising child is re-tested with two nearby parameter settings; if its edge
only exists at one exact setting, its fitness is marked down.
Lab notebook: every AI-bred idea, its hypothesis and what went wrong is remembered and shown
to the AI, so it stops re-inventing the same strategy.
Team: at the end, a diverse team of survivors is chosen on training data and sits the exam.
Validation (when the Lab has a validation window): breeding never sees the last years of the
training period. At the end, the top finalists by training fitness (and the team) are scored once
on that window, and the best of them becomes the champion that sits the exam. Validation scores
are never shown to the AI, so the window stays out of the search.
"""
from __future__ import annotations

import ast
import itertools
import random
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import engine, fitness as fit, prompts
from .lab import Lab
from .llm import LLMError, Provider
from .sandbox import StrategyError
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
    raw_fitness: float | None = None     # before the robustness check
    robust: float | None = None          # mean fitness of nearby parameter settings
    returns: object = None               # training daily returns (kept for team building)
    val: float | None = None             # appraisal ratio on the validation window (finalists only)

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


def arity(code: str) -> int:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == "strategy":
            return len(node.args.args)
    return 1


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
            if names and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)) \
                    and not isinstance(node.value.value, bool):
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


def combine(members: list, weights: list, name: str, idea: str) -> str:
    """Code for a strategy holding a weighted mix of several strategies' portfolios."""
    parts, calls = [], []
    for m, w in zip(members, weights):
        fn = f"_{m.id}"
        parts.append(m.code.replace("def strategy(", f"def {fn}(", 1).strip())
        call = f"{fn}(prices, data)" if arity(m.code) >= 2 else f"{fn}(prices)"
        calls.append(f"    out = out + {w:.4f} * {call}.reindex(index=prices.index, columns=prices.columns).fillna(0.0)")
    body = "\n".join(calls)
    return "\n\n\n".join(parts) + f'''


def strategy(prices, data):
    """{name}: {idea}"""
    out = prices * 0.0
{body}
    return out
'''


def blend(a: "Individual", b: "Individual", rng: random.Random) -> str:
    mix = round(rng.uniform(0.3, 0.7), 2)
    return combine([a, b], [mix, 1 - mix], f"{family(a.name)} x {family(b.name)}",
                   "a blend of both parents' portfolios.")


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
    robust_check: bool = True
    team_size: int = 5
    seed: int = 0
    search: str = "v2"               # "v1" = the original search (for comparisons)
    finalists: int = 10              # candidates scored on the validation window


class Evolution:
    def __init__(self, lab: Lab, exam: fit.SealedExam, provider: Provider | None, market: str, cfg: Config,
                 on_event=None, strong: Provider | None = None):
        self.lab, self.exam, self.llm, self.strong, self.market, self.cfg = lab, exam, provider, strong, market, cfg
        self.data_desc = lab.train.describe()
        self.rng = random.Random(cfg.seed)
        self.islands: list[list[Individual]] = [[] for _ in range(cfg.islands)]
        self.all: dict[str, Individual] = {}
        self.gen = 0
        self.on_event = on_event or (lambda kind, **kw: None)
        self.examined_best = float("-inf")
        self._versions: dict[str, int] = {}
        self.llm_calls = 0
        self.failures = 0
        self.notebook: list[dict] = []
        self.team: Individual | None = None
        self.final: Individual | None = None

    # -- evaluation -------------------------------------------------------------------------
    def _score(self, code: str, bloat: bool = True):
        res = self.lab.run(code, "train")
        v2 = self.cfg.search == "v2"
        s = engine.stats(res.returns, res.turnover, res.gross, self.lab.mkt["train"],
                         halves=res.halves if v2 else None, search=self.cfg.search)
        return s, fit.fitness(s, code if bloat else "", self.cfg.search), res.returns

    def evaluate(self, ind: Individual) -> Individual:
        try:
            ind.stats, ind.fitness, ind.returns = self._score(ind.code, bloat=ind.op != "team")
            ind.raw_fitness = ind.fitness
        except StrategyError as e:
            ind.error = str(e).splitlines()[0][:160]
        except Exception as e:  # defensive: never let one child crash the run
            ind.error = f"{type(e).__name__}: {e}"[:160]
        return ind

    def robustness(self, ind: Individual):
        """Re-test with two nearby parameter settings; mark down knife-edge strategies."""
        rng = random.Random(zlib.crc32(ind.code.encode()))      # deterministic across processes
        fits = []
        for _ in range(2):
            code = tweak(ind.code, rng)
            if not code:
                return
            try:
                fits.append(self._score(code)[1])
            except Exception:
                fits.append(-1.0)
        ind.robust = float(np.mean(fits))
        ind.fitness = min(ind.raw_fitness, 0.5 * ind.raw_fitness + 0.5 * ind.robust)

    def _note(self, ind: Individual):
        if ind.op not in ("mutate", "crossover", "immigrant"):
            return
        idea, hyp = prompts.idea_and_hypothesis(ind.code)
        if ind.error:
            why = "peeked at the future" if "look-ahead" in ind.error else f"crashed: {ind.error[:60]}"
        else:
            why = prompts.weakness(ind.stats, ind.robust, ind.raw_fitness)
        self.notebook.append({"name": ind.name, "idea": idea, "hypothesis": hyp, "op": ind.op,
                              "fitness": None if ind.error else ind.fitness, "why": why})

    def _ask(self, prompt: str, strong: bool = False) -> str | None:
        self.llm_calls += 1
        llm = self.strong if (strong and self.strong) else self.llm
        try:
            return prompts.extract_code(llm.complete(prompts.system(self.data_desc, self.cfg.search), prompt))
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
        nb = prompts.notebook(self.notebook)
        if op == "tweak":
            code, parents, name = tweak(a.code, self.rng), (a.id,), self._version(a.name)
        elif op == "blend":
            others = [p for p in pop if family(p.name) != family(a.name)]
            b = self._pick(others or [p for p in pop if p.id != a.id] or pop)
            code, parents = blend(a, b, self.rng), (a.id, b.id)
            name = f"{family(a.name)} x {family(b.name)}"
        elif op == "mutate":
            code = self._ask(prompts.mutate(a.code, prompts.report_card(a.name, a.stats, a.robust), self.market, nb))
            parents, name = (a.id,), None
        elif op == "crossover":
            b = self._pick([p for p in pop if p.id != a.id] or pop)
            code = self._ask(prompts.crossover(a.code, prompts.report_card(a.name, a.stats, a.robust),
                                               b.code, prompts.report_card(b.name, b.stats, b.robust),
                                               self.market, nb), strong=True)
            parents, name = (a.id, b.id), None
        else:  # immigrant
            code = self._ask(prompts.immigrant(self.rng.choice(IMMIGRANT_THEMES), self.market, nb), strong=True)
            parents, name = (), None
        if not code:
            return None
        name = name or prompts.strategy_name(code, fallback=f"Finch {next(_ids)}")
        return self.evaluate(Individual(new_id(), name, code, op, parents, self.gen, isl))

    def _ops(self) -> list[str]:
        if self.llm is None:
            return self.rng.choices(["tweak", "blend"], weights=[0.65, 0.35], k=self.cfg.offspring)
        ops = self.rng.choices(["mutate", "crossover", "tweak", "blend"], weights=[0.45, 0.25, 0.2, 0.1],
                               k=self.cfg.offspring)
        if self.rng.random() < 0.4:
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
                if s.error:
                    continue
                c = Individual(new_id(), s.name, s.code, "seed", (), 0, i, s.fitness, s.stats, None,
                               raw_fitness=s.raw_fitness, returns=s.returns)
                self.islands[i].append(c)
                self.all[c.id] = c
            self.islands[i] = self._survive(self.islands[i])
        self.on_event("seeded", best=self.champion())

    def champion(self) -> Individual | None:
        if self.final is not None:
            return self.final
        alive = [p for isl in self.islands for p in isl]
        return max(alive, key=lambda p: p.fitness) if alive else None

    def _contender(self, c: Individual) -> bool:
        isl = sorted((p.fitness for p in self.islands[c.island]), reverse=True)
        return len(isl) < 3 or c.raw_fitness > isl[min(2, len(isl) - 1)]

    def step(self):
        self.gen += 1
        jobs = [(i, op) for i in range(self.cfg.islands) for op in self._ops()]
        with ThreadPoolExecutor(self.cfg.workers) as ex:
            children = list(ex.map(lambda j: self._make_child(*j), jobs))
        if self.cfg.robust_check:
            todo = [c for c in children if c and not c.error and self._contender(c)]
            with ThreadPoolExecutor(self.cfg.workers) as ex:
                list(ex.map(self.robustness, todo))
        for c in children:
            if c is None:
                self.failures += 1
                continue
            self.all[c.id] = c
            self._note(c)
            if c.error:
                self.failures += 1
                self.on_event("stillborn", child=c)
                continue
            self.islands[c.island].append(c)
            self.on_event("born", child=c, parents=[self.all[p] for p in c.parents if p in self.all])
        for i in range(self.cfg.islands):
            self.islands[i] = self._survive(self.islands[i])
        if self.cfg.migrate_every and self.gen % self.cfg.migrate_every == 0:
            for i in range(self.cfg.islands):
                champ = max(self.islands[i], key=lambda p: p.fitness)
                j = (i + 1) % self.cfg.islands
                m = Individual(new_id(), champ.name, champ.code, "migrant", (champ.id,), self.gen, j,
                               champ.fitness, champ.stats, raw_fitness=champ.raw_fitness, robust=champ.robust,
                               returns=champ.returns)
                self.all[m.id] = m
                self.islands[j] = self._survive(self.islands[j] + [m])
            self.on_event("migration")
        champ = self.champion()
        reserve = 2 if self.cfg.team_size > 1 else 1        # final champion + team
        if champ and self.lab.val is None and champ.exam is None and champ.fitness > self.examined_best + self.cfg.exam_margin \
                and self.exam.left > reserve:
            self._sit(champ)
        self.on_event("generation", gen=self.gen, best=champ)

    def _sit(self, ind: Individual):
        ind.exam = self.exam.sit(ind.id, ind.code)
        self.examined_best = max(self.examined_best, ind.fitness)
        self.on_event("exam", ind=ind)

    def build_team(self) -> Individual | None:
        """Pick up to `team_size` strong, mutually diverse strategies (training data only) and
        hold them together, weighted by inverse volatility."""
        pool, seen = [], set()
        for p in sorted([p for isl in self.islands for p in isl], key=lambda p: -p.fitness):
            if p.returns is not None and p.code not in seen:
                seen.add(p.code)
                pool.append(p)
        if not pool:
            return None
        floor = max(0.0, 0.5 * pool[0].fitness)          # only good strategies may join
        pool = [p for p in pool if p.fitness >= floor]
        team = []
        for p in pool[:30]:
            if all(abs(np.corrcoef(p.returns.values[1:], q.returns.values[1:])[0, 1]) < 0.7 for q in team):
                team.append(p)
            if len(team) == self.cfg.team_size:
                break
        if len(team) < 2:
            return None
        inv = np.array([1 / max(q.returns.std(), 1e-6) for q in team])
        w = list(inv / inv.sum())
        names = ", ".join(q.name for q in team)
        code = combine(team, w, "The Team", f"a diversified team of {len(team)} evolved strategies ({names}).")
        ind = self.evaluate(Individual(new_id(), "The Team", code, "team", tuple(q.id for q in team), self.gen, 0))
        if ind.error:
            return None
        self.all[ind.id] = ind
        return ind

    def validate(self, ind: Individual) -> float:
        """Appraisal ratio on the validation window (data breeding never saw)."""
        try:
            res = self.lab.run(ind.code, "dev", check_leaks=False)
            mask = res.returns.index >= self.lab.val
            ind.val = float(engine.alpha_stats(res.returns[mask], self.lab.mkt["dev"][mask])[2])
        except Exception:
            ind.val = float("-inf")
        return ind.val

    def finalists(self) -> list[Individual]:
        out, seen = [], set()
        for p in sorted([p for isl in self.islands for p in isl], key=lambda p: -p.fitness):
            key = " ".join(p.code.split())
            if key not in seen:
                seen.add(key)
                out.append(p)
            if len(out) == self.cfg.finalists:
                break
        return out

    def run(self):
        t0 = time.time()
        self.seed_population()
        for _ in range(self.cfg.generations):
            self.step()
        if self.cfg.team_size > 1:
            self.team = self.build_team()
        if self.lab.val is not None:                      # choose the champion on unseen years
            fins = self.finalists()
            with ThreadPoolExecutor(self.cfg.workers) as ex:
                list(ex.map(self.validate, fins + ([self.team] if self.team else [])))
            self.final = max(fins, key=lambda p: p.val)
            self.on_event("validated", finalists=fins, champion=self.final)
        champ = self.champion()
        if champ and champ.exam is None and self.exam.left > 0:
            self._sit(champ)
        if self.team and self.exam.left > 0 and champ:
            ok = (self.team.val >= champ.val - 0.1) if self.lab.val is not None \
                else (self.team.fitness >= champ.fitness - 0.05)
            if ok:
                self._sit(self.team)
        self.on_event("done", seconds=time.time() - t0)
        return champ

    def lineage(self, ind: Individual, depth: int = 12) -> list[Individual]:
        out, cur = [], ind
        while cur and len(out) < depth:
            out.append(cur)
            cur = self.all.get(cur.parents[0]) if cur.parents else None
        return out
