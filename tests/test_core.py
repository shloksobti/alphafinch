import random

import numpy as np
import pandas as pd
import pytest

from alphafinch import data, engine, evolve, fitness, prompts, world
from alphafinch.lab import Lab
from alphafinch.sandbox import StrategyError
from alphafinch.seeds import SEEDS

HOLD = pd.Timestamp("2006-01-02")


def _panel():
    px = data.synthetic(n_assets=8, years=8, seed=1)
    rng = np.random.default_rng(2)
    vol = pd.DataFrame(1e6 * rng.lognormal(0, 0.5, px.shape), index=px.index, columns=px.columns)
    macro = pd.DataFrame({"vix": 20.0}, index=px.index)
    sector = pd.Series(["Banks", "Energy"] * 4, index=px.columns)
    return data.Panel(px, px, px, px, vol, sector, macro).astype32()


@pytest.fixture(scope="module")
def lab():
    with Lab(_panel(), HOLD, workers=2, timeout=20) as lb:
        yield lb


def _stats(lab, code):
    res = lab.run(code, "train")
    return engine.stats(res.returns, res.turnover, res.gross, lab.mkt["train"])


@pytest.mark.parametrize("name", list(SEEDS))
def test_seeds_are_valid_and_causal(lab, name):
    s = _stats(lab, SEEDS[name])
    assert s.n_days > 1000 and np.isfinite(s.sharpe) and len(s.eras) == 4


def test_training_split_never_contains_holdout(lab):
    assert lab.train.index.max() < HOLD and lab.full.index.max() > HOLD
    code = "def strategy(prices):\n    assert prices.index.max() < __HOLD__\n    return prices * 0\n"
    with pytest.raises(StrategyError):          # dunder names are rejected before running
        lab.run(code)


@pytest.mark.parametrize("code", [
    "import pandas as pd\ndef strategy(prices):\n    return (prices.shift(-1) > prices).astype(float)\n",
    "def strategy(prices):\n    return -(prices - prices.mean()) / prices.std()\n",
    "def strategy(prices):\n    return (prices.rolling(21, center=True).mean() > prices).astype(float)\n",
    "def strategy(prices, data):\n    v = data.volume\n    return (v.shift(-2) > v).astype(float)\n",
])
def test_lookahead_is_detected(lab, code):
    with pytest.raises(StrategyError, match="look-ahead"):
        lab.run(code)


@pytest.mark.parametrize("code", [
    "import os\ndef strategy(prices):\n    return os.listdir('.')\n",
    "def strategy(prices):\n    return open('/etc/passwd').read()\n",
    "def strategy(prices):\n    return prices.__class__\n",
    "import subprocess\ndef strategy(prices):\n    return prices\n",
    "def strategy(prices):\n    prices.to_csv('/tmp/x.csv')\n    return prices\n",
    "def strategy(prices):\n    return eval('1')\n",
])
def test_sandbox_blocks_unsafe_code(lab, code):
    with pytest.raises(StrategyError):
        lab.run(code)


def test_sandbox_times_out_and_recovers(lab):
    with pytest.raises(StrategyError, match="timed out"):
        lab.run("def strategy(prices):\n    while True:\n        pass\n")
    assert np.isfinite(_stats(lab, SEEDS["Calm Seeker"]).sharpe)   # pool restarted and still works


def test_data_fields_reach_strategies(lab):
    code = ("import pandas as pd\ndef strategy(prices, data):\n"
            "    assert data.sector.nunique() == 2 and 'vix' in data.macro\n"
            "    return (data.volume > 0).astype(float)\n")
    assert _stats(lab, code).exposure > 0.9


def test_equal_weight_has_no_alpha(lab):
    code = ("import pandas as pd\ndef strategy(prices):\n"
            "    return pd.DataFrame(1/prices.shape[1], index=prices.index, columns=prices.columns)\n")
    s = _stats(lab, code)
    assert abs(s.beta - 1) < 0.01 and abs(s.alpha) < 0.002 and s.appraisal == 0.0


def test_fitness_punishes_a_losing_era():
    good = engine.Stats(1, .1, .1, -.1, 2, 1, 1, 0, 0, 0, {}, 2000,
                        [{"score": 0.5}, {"score": 0.5}, {"score": 0.5}, {"score": 0.5}])
    spiky = engine.Stats(1, .1, .1, -.1, 2, 1, 1, 0, 0, 0, {}, 2000,
                         [{"score": 1.8}, {"score": 0.5}, {"score": 0.4}, {"score": -0.6}])
    assert fitness.fitness(good, "x") > fitness.fitness(spiky, "x")


def test_tweak_changes_exactly_one_constant():
    code = SEEDS["Momentum Crown"]
    out = evolve.tweak(code, random.Random(3))
    diff = [(a, b) for a, b in zip(code.splitlines(), out.splitlines()) if a != b]
    assert out != code and len(diff) == 1


def test_blend_of_mixed_arity_parents_is_valid(lab):
    a = evolve.Individual("fA", "Trend Rider", SEEDS["Trend Rider"], "seed")
    b = evolve.Individual("fB", "Fear Gauge", SEEDS["Fear Gauge"], "seed")
    assert np.isfinite(_stats(lab, evolve.blend(a, b, random.Random(0))).sharpe)


def test_exam_reveals_only_pass_fail_and_respects_budget(lab):
    ex = fitness.SealedExam(lab, budget=2)
    assert ex.sit("x", SEEDS["Calm Seeker"]) in ("PASS", "FAIL")
    assert ex.sit("y", SEEDS["Trend Rider"]) in ("PASS", "FAIL")
    assert ex.sit("z", SEEDS["Snapback"]) == "EXHAUSTED"
    r = ex.reveal("x", SEEDS["Calm Seeker"])
    assert r["grade"] in ("PASS", "PROMISING", "FAIL") and "market_equity" in r


def test_names_hypotheses_and_notebook():
    code = 'def strategy(prices):\n    """Night Owl: trades at night. Hypothesis: insomniacs overpay."""\n    return prices*0\n'
    assert prompts.strategy_name(code) == "Night Owl"
    assert prompts.idea_and_hypothesis(code) == ("trades at night.", "insomniacs overpay.")
    nb = prompts.notebook([{"name": "Night Owl", "op": "mutate", "idea": "x", "hypothesis": "y",
                            "fitness": 0.4, "why": "solid"}])
    assert "Night Owl" in nb and "LAB NOTEBOOK" in nb
    assert evolve.base_name("Night Owl v12") == "Night Owl"
    assert evolve.family("Night Owl x Calm Seeker v3") == "Night Owl"


def test_offline_evolution_runs_with_team(lab):
    ex = fitness.SealedExam(lab, budget=4)
    e = evolve.Evolution(lab, ex, None, "synthetic",
                         evolve.Config(islands=2, island_size=5, offspring=2, generations=2, workers=2, search="v1"))
    champ = e.run()
    assert champ is not None and np.isfinite(champ.fitness)
    team = e.build_team()
    best = max(p.fitness for isl in e.islands for p in isl)
    eligible = [p for isl in e.islands for p in isl if p.fitness >= max(0.0, 0.5 * best)]
    if team is None:                               # only if too few strong survivors to form one
        assert len({p.code for p in eligible}) < 2 or best <= 0
    else:
        assert len(team.parents) >= 2 and np.isfinite(team.fitness)


def test_hard_coded_asset_and_sector_names_are_rejected(lab):
    for code in ["def strategy(prices):\n    w = prices * 0\n    w['SYN03'] = 1.0\n    return w\n",
                 "def strategy(prices, data):\n    return (data.sector == 'Banks').astype(float) + prices * 0\n"]:
        with pytest.raises(StrategyError, match="hard-coded"):
            lab.run(code)


def test_toolkit_strategies_are_causal_and_get_halves(lab):
    code = ('def strategy(prices, data):\n'
            '    """Sector Spread: sector-neutral momentum. Hypothesis: slow diffusion."""\n'
            '    score = tk.neutralize(tk.zscore(prices.pct_change(60)), data.sector)\n'
            '    return tk.rebalance(tk.vol_target(tk.long_short(score, 0.25), prices), "M")\n')
    res = lab.run(code)
    assert res.halves is not None and len(res.halves) == 4
    s = engine.stats(res.returns, res.turnover, res.gross, lab.mkt["train"], halves=res.halves)
    assert len(s.halves) == 4 and abs(s.beta) < 0.5


def test_v2_fitness_ignores_beta_and_punishes_a_narrow_edge():
    eras = [{"score": 0.5}] * 4
    broad = engine.Stats(1, .1, .1, -.1, 2, 1, 1, 0, 0.5, 0, {}, 2000, eras, [0.5, 0.4, 0.6, 0.5])
    narrow = engine.Stats(1, .1, .1, -.1, 2, 1, 1, 0, 0.5, 0, {}, 2000, eras, [1.0, -0.2, 0.9, 0.1])
    assert fitness.fitness(broad, "x") > fitness.fitness(narrow, "x")
    assert fitness.fitness(broad, "x", "v1") == fitness.fitness(narrow, "x", "v1")


def test_validation_chooses_the_champion_and_breeding_never_sees_it():
    val = pd.Timestamp("2004-01-02")
    with Lab(_panel(), HOLD, workers=2, timeout=20, validation_start=val) as lb:
        assert lb.train.index.max() < val <= lb.dev.index.max() < HOLD
        ex = fitness.SealedExam(lb, budget=3)
        e = evolve.Evolution(lb, ex, None, "synthetic",
                             evolve.Config(islands=2, island_size=5, offspring=2, generations=2, workers=2))
        champ = e.run()
        assert champ is e.final and champ.val is not None
        assert champ.val == max(p.val for p in e.finalists())
        assert all(a["id"] in (champ.id, getattr(e.team, "id", None)) for a in ex.attempts)   # no mid-run exams


def test_v1_search_still_runs(lab):
    ex = fitness.SealedExam(lab, budget=3)
    e = evolve.Evolution(lab, ex, None, "synthetic",
                         evolve.Config(islands=2, island_size=5, offspring=2, generations=2, workers=2, search="v1"))
    assert np.isfinite(e.run().fitness)
    assert "tk." not in prompts.system("x", "v1") and "tk.neutralize" in prompts.system("x")


def _world_panels(n=3):
    out = {}
    for i in range(n):
        px = data.synthetic(n_assets=12, years=8, seed=10 + i)
        rng = np.random.default_rng(i)
        vol = pd.DataFrame(1e6 * rng.lognormal(0, 0.5, px.shape), index=px.index, columns=px.columns)
        out[f"m{i}"] = data.Panel(px, px, px, px, vol, pd.Series(["Banks", "Energy", "Tech"] * 4, index=px.columns),
                                  pd.DataFrame({"vix": 20.0}, index=px.index)).astype32()
    return out


def test_newey_west_t_is_calibrated_on_noise():
    rng = np.random.default_rng(0)
    ts = [world.newey_west_t(rng.normal(0, 0.01, 1500))[1] for _ in range(200)]
    assert abs(np.mean(ts)) < 0.25 and 0.8 < np.std(ts) < 1.25


def test_world_exam_placebos_fail_and_errors_are_reported():
    from alphafinch import world
    eq = ("import pandas as pd\ndef strategy(prices):\n"
          "    return pd.DataFrame(1/prices.shape[1], index=prices.index, columns=prices.columns)\n")
    fund = "def strategy(prices, data):\n    return data.fund['earnings_yield'] * 0 + prices * 0\n"
    rnd = ("import numpy as np, pandas as pd\ndef strategy(prices):\n"
           "    h = (np.arange(prices.shape[1])[None, :] * 7 + np.arange(len(prices))[:, None] // 21) % 5\n"
           "    return pd.DataFrame((h == 0) * 1.0 - (h == 1) * 1.0, index=prices.index, columns=prices.columns) / 4\n")
    res = {r["name"]: r for r in world.exam({"eq": eq, "fund": fund, "rnd": rnd}, _world_panels(),
                                             start="2003-01-02", workers=2)}
    assert res["eq"]["pooled_t"] == 0.0 and res["eq"]["grade"] == "FAIL"
    assert res["fund"]["grade"] == "NOT RUN" and all("error" in v for v in res["fund"]["markets"].values())
    assert res["rnd"]["n_markets"] == 3 and res["rnd"]["K"] == 3
    assert res["rnd"]["bar"] > 2.0 and res["rnd"]["grade"] != "PASS"


def test_mandate_projection_enforces_every_rule():
    from alphafinch import mandate as md
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.normal(0, 0.3, (50, 10)))
    lo = md.project(w, md.get("long-only"))
    assert (lo.values >= 0).all() and (lo.abs().sum(axis=1) <= 1 + 1e-9).all()
    mn = md.project(w, md.get("market-neutral"))
    assert (mn.sum(axis=1).abs() <= 0.1 + 1e-9).all() and (mn.abs().sum(axis=1) <= 1 + 1e-9).all()
    mask = np.array([True] * 3 + [False] * 7)
    fo = md.project(w, md.get("derivatives"), mask)
    assert (fo.values[:, ~mask] >= 0).all() and (fo.abs().sum(axis=1) <= 2 + 1e-9).all()
    cap = md.project(w, md.get("long-short", max_weight=0.05))
    assert cap.abs().values.max() <= 0.05 + 1e-12


def test_mandate_is_enforced_in_the_lab_and_charges_borrow():
    from alphafinch import mandate as md
    short_all = ("import pandas as pd\ndef strategy(prices):\n"
                 "    return pd.DataFrame(-1/prices.shape[1], index=prices.index, columns=prices.columns)\n")
    with Lab(_panel(), HOLD, workers=1, timeout=20, mandate=md.get("long-only")) as lb:
        r = lb.run(short_all)
        assert r.gross.abs().max() == 0 and r.returns.abs().max() == 0       # shorts are simply not allowed
    with Lab(_panel(), HOLD, workers=1, timeout=20) as free, \
            Lab(_panel(), HOLD, workers=1, timeout=20, mandate=md.get("long-short", borrow_bps=500)) as fee:
        diff = (free.run(short_all).returns - fee.run(short_all).returns).iloc[5:]
        assert np.allclose(diff, 0.05 / 252, rtol=0.05)                       # 500 bps a year on a full short
    assert "NO SHORTING" in md.get("long-only").describe()


def test_holdings_respect_the_mandate():
    from alphafinch import mandate as md
    with Lab(_panel(), None, workers=1, timeout=20, mandate=md.get("long-only")) as lb:
        w = lb.weights(SEEDS["Snapback"], last=3)          # a long/short seed
        assert len(w) == 3 and (w.values >= 0).all() and (w.abs().sum(axis=1) <= 1 + 1e-9).all()


def test_broken_split_adjustments_are_repaired_but_real_crashes_are_not():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"glitch": 100.0, "crash": 100.0}, index=idx)
    px.iloc[10:13, 0] = 10.0                        # -90% then back: a bad split adjustment
    px.iloc[10:, 1] = 40.0                          # a real -60% crash that stays down
    fixed, fixes = data.repair_splits(px)
    assert np.allclose(fixed["glitch"], 100.0) and [f[0] for f in fixes] == ["glitch"]
    assert (fixed["crash"] == px["crash"]).all()
