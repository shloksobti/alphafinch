import random

import numpy as np
import pandas as pd
import pytest

from alphafinch import data, engine, evolve, fitness, prompts, sandbox
from alphafinch.seeds import SEEDS

PX = data.synthetic(n_assets=8, years=8, seed=1)
TRAIN = PX.loc[:"2005"]


@pytest.mark.parametrize("name", list(SEEDS))
def test_seeds_are_valid_and_causal(name):
    _, s = engine.backtest(SEEDS[name], TRAIN)
    assert s.n_days > 1000 and np.isfinite(s.sharpe)


@pytest.mark.parametrize("code", [
    "import pandas as pd\ndef strategy(prices):\n    return (prices.shift(-1) > prices).astype(float)\n",
    "def strategy(prices):\n    return -(prices - prices.mean()) / prices.std()\n",
    "def strategy(prices):\n    return (prices.rolling(21, center=True).mean() > prices).astype(float)\n",
])
def test_lookahead_is_detected(code):
    with pytest.raises(sandbox.StrategyError, match="look-ahead"):
        engine.backtest(code, TRAIN)


@pytest.mark.parametrize("code", [
    "import os\ndef strategy(prices):\n    return os.listdir('.')\n",
    "def strategy(prices):\n    return open('/etc/passwd').read()\n",
    "def strategy(prices):\n    return prices.__class__\n",
    "import subprocess\ndef strategy(prices):\n    return prices\n",
    "def strategy(prices):\n    prices.to_csv('/tmp/x.csv')\n    return prices\n",
    "def strategy(prices):\n    return eval('1')\n",
])
def test_sandbox_blocks_unsafe_code(code):
    with pytest.raises(sandbox.StrategyError):
        engine.backtest(code, TRAIN)


def test_sandbox_times_out():
    code = "def strategy(prices):\n    while True:\n        pass\n"
    with pytest.raises(sandbox.StrategyError, match="timed out"):
        sandbox.run(code, TRAIN, timeout=3)


def test_engine_lags_weights_one_day():
    # a strategy that is long only on days the asset rose TODAY must not earn today's return
    code = "def strategy(prices):\n    return (prices.pct_change() > 0).astype(float)\n"
    w = sandbox.run(code, TRAIN)
    r, _, held = engine.portfolio_returns(w, TRAIN)
    assert held.iloc[0].abs().sum() == 0  # nothing held on day 0
    assert (held.shift(-1).iloc[:-1].values == engine.normalise(w).iloc[:-1].values).all()


def test_equal_weight_has_no_alpha():
    code = ("import pandas as pd\ndef strategy(prices):\n"
            "    return pd.DataFrame(1/prices.shape[1], index=prices.index, columns=prices.columns)\n")
    _, s = engine.backtest(code, TRAIN)
    assert abs(s.beta - 1) < 0.01 and abs(s.alpha) < 0.002 and s.appraisal == 0.0


def test_tweak_changes_exactly_one_constant():
    code = SEEDS["Momentum Crown"]
    out = evolve.tweak(code, random.Random(3))
    assert out != code and "def strategy" in out
    diff = [(a, b) for a, b in zip(code.splitlines(), out.splitlines()) if a != b]
    assert len(diff) == 1


def test_blend_child_is_valid():
    a = evolve.Individual("fA", "Trend Rider", SEEDS["Trend Rider"], "seed")
    b = evolve.Individual("fB", "Calm Seeker", SEEDS["Calm Seeker"], "seed")
    code = evolve.blend(a, b, random.Random(0))
    _, s = engine.backtest(code, TRAIN)
    assert np.isfinite(s.sharpe)


def test_exam_reveals_only_pass_fail_and_respects_budget():
    ex = fitness.SealedExam(PX, "2006-01-01", budget=2)
    v1 = ex.sit("x", SEEDS["Calm Seeker"])
    v2 = ex.sit("y", SEEDS["Trend Rider"])
    assert v1 in ("PASS", "FAIL") and v2 in ("PASS", "FAIL")
    assert ex.sit("z", SEEDS["Snapback"]) == "EXHAUSTED"


def test_names_and_code_extraction():
    txt = "Here you go\n```python\ndef strategy(prices):\n    \"\"\"Night Owl: trades at night.\"\"\"\n    return prices*0\n```"
    code = prompts.extract_code(txt)
    assert prompts.strategy_name(code) == "Night Owl"
    assert evolve.base_name("Night Owl v12") == "Night Owl"
    assert evolve.family("Night Owl x Calm Seeker v3") == "Night Owl"


def test_offline_evolution_runs():
    ex = fitness.SealedExam(PX, "2006-01-01", budget=4)
    e = evolve.Evolution(TRAIN, ex, None, "synthetic",
                         evolve.Config(islands=2, island_size=4, offspring=2, generations=2, workers=2))
    champ = e.run()
    assert champ is not None and np.isfinite(champ.fitness)
    assert len(ex.attempts) >= 1
