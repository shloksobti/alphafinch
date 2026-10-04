"""The Lab: a pool of sandboxed worker processes that run strategies against market data.

Each worker loads the market panel ONCE (from a pickle written by the main process). A task
sends only strategy code; the worker executes it with restricted builtins, checks it for
look-ahead, simulates the portfolio and returns only daily return/turnover/exposure arrays.
During evolution strategies see the training panel only. The sealed holdout is used solely by
`exam_returns`, whose results never flow back to the AI.

Strategy contract:
    def strategy(prices):            # prices = data.close
    def strategy(prices, data):      # data.close/open/high/low/volume/sector/macro/fund
Both return target weights (dates x assets); row t is held from close t to close t+1.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .sandbox import ALLOWED_IMPORTS, SAFE_BUILTINS, StrategyError, check

COST_BPS = 5.0
LEAK_CUTS = (0.55, 0.85)
_W: dict = {}


# ------------------------------------------------------------------------------- worker side
def _init(path: str, hold_start):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (24 * 3600, 24 * 3600))
    except Exception:
        pass
    with open(path, "rb") as f:
        full = pickle.load(f)
    _W["full"] = full
    _W["train"] = full.before(pd.Timestamp(hold_start)) if hold_start is not None else full


def _namespace(p) -> SimpleNamespace:
    cp = lambda x: None if x is None else x.copy()
    return SimpleNamespace(close=p.close.copy(), open=cp(p.open), high=cp(p.high), low=cp(p.low),
                           volume=cp(p.volume), sector=None if p.sector is None else p.sector.copy(),
                           macro=cp(p.macro), fund={k: v.copy() for k, v in p.fund.items()})


def _imp(name, *a, **k):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(name)
    return __import__(name, *a, **k)


def _weights(code: str, p) -> pd.DataFrame:
    import math as _m
    g = {"__builtins__": dict(SAFE_BUILTINS, __import__=_imp), "np": np, "pd": pd, "math": _m}
    exec(compile(code, "<strategy>", "exec"), g)
    f = g["strategy"]
    nargs = f.__code__.co_argcount
    ns = _namespace(p)
    w = f(ns.close, ns) if nargs >= 2 else f(ns.close)
    if not isinstance(w, pd.DataFrame):
        raise StrategyError("strategy must return a DataFrame of weights")
    w = w.reindex(index=p.close.index, columns=p.close.columns)
    return w.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalise(w: pd.DataFrame) -> pd.DataFrame:
    gross = w.abs().sum(axis=1)
    return w.mul(np.where(gross > 1, 1 / gross.replace(0, 1), 1.0), axis=0)


def _simulate(w: pd.DataFrame, close: pd.DataFrame, cost_bps: float):
    rets = close.astype("float64").pct_change().fillna(0.0)
    held = _normalise(w).shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    r = (held * rets).sum(axis=1) - turnover * cost_bps / 1e4
    return r.values, turnover.values, held.abs().sum(axis=1).values


def _task(code: str, split: str, check_leaks: bool, cost_bps: float):
    try:
        p = _W[split]
        w = _weights(code, p)
        if check_leaks:
            n = len(p.close)
            for frac in LEAK_CUTS:
                cut = int(n * frac)
                wc = _weights(code, p.head(cut))
                a, b = _normalise(w.iloc[:cut]).values, _normalise(wc).values
                if not np.allclose(a, b, atol=1e-6, equal_nan=True):
                    bad = np.argwhere(~np.isclose(a, b, atol=1e-6))[0][0]
                    return {"error": f"look-ahead detected: weights on {p.close.index[bad].date()} "
                                     "change when later data is removed"}
        r, to, gross = _simulate(w, p.close, cost_bps)
        return {"r": r.astype("float32"), "to": to.astype("float32"), "gross": gross.astype("float32")}
    except StrategyError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300] + "\n" + traceback.format_exc(limit=2)[-300:]}


# ------------------------------------------------------------------------------- main side
@dataclass
class Result:
    returns: pd.Series
    turnover: pd.Series
    gross: pd.Series


class Lab:
    def __init__(self, panel, holdout_start=None, workers: int = 4, timeout: float = 180.0, cost_bps=COST_BPS):
        self.full = panel
        self.hold = pd.Timestamp(holdout_start) if holdout_start is not None else None
        self.train = panel.before(self.hold) if self.hold is not None else panel
        self.workers, self.timeout, self.cost_bps = workers, timeout, cost_bps
        fd, self._path = tempfile.mkstemp(prefix="alphafinch_", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(panel, f)
        self._pool = None
        self._start()
        mk = lambda p: p.close.astype("float64").pct_change().fillna(0.0).mean(axis=1)
        self.mkt = {"train": mk(self.train), "full": mk(self.full)}

    def _start(self):
        self._pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                         initargs=(self._path, self.hold), max_tasks_per_child=60)

    def _restart(self):
        pool, self._pool = self._pool, None
        for proc in list(getattr(pool, "_processes", {}).values()):
            try:
                proc.kill()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)
        self._start()

    def run(self, code: str, split: str = "train", check_leaks: bool = True) -> Result:
        check(code)
        fut = self._pool.submit(_task, code, split, check_leaks, self.cost_bps)
        try:
            out = fut.result(timeout=self.timeout)
        except FutTimeout:
            self._restart()
            raise StrategyError(f"timed out after {self.timeout:.0f}s") from None
        except Exception as e:  # broken pool (e.g. a worker crashed): restart and report
            self._restart()
            raise StrategyError(f"worker crashed: {type(e).__name__}") from None
        if "error" in out:
            raise StrategyError(out["error"])
        idx = (self.train if split == "train" else self.full).close.index
        return Result(pd.Series(out["r"], idx, dtype="float64"), pd.Series(out["to"], idx, dtype="float64"),
                      pd.Series(out["gross"], idx, dtype="float64"))

    def close(self):
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
