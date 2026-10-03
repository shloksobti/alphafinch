"""Run untrusted, AI-written strategy code safely.

Two layers:
1. Static check (AST): only numpy / pandas / math imports, no dunder access, no
   open/exec/eval/__import__/compile/globals, no attribute access to os/sys/subprocess.
2. Execution in a separate process with a restricted builtins dict, a CPU-time limit and a
   wall-clock timeout. The process receives prices and returns weights; it has no other I/O.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import traceback

import numpy as np
import pandas as pd

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "help", "memoryview",
                "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "builtins"}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
                           "isinstance", "len", "list", "map", "max", "min", "pow", "range", "reversed",
                           "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception",
                           "ValueError", "ZeroDivisionError", "KeyError", "IndexError", "print"]}


class StrategyError(Exception):
    pass


def check(code: str) -> None:
    """Raise StrategyError if the code is not acceptable."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise StrategyError(f"syntax error: {e}") from None
    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise StrategyError(f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise StrategyError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise StrategyError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise StrategyError("dunder attribute access is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in {"to_csv", "to_pickle", "read_csv",
                                                                 "read_pickle", "to_parquet", "read_parquet",
                                                                 "to_excel", "read_excel", "to_sql", "read_sql",
                                                                 "save", "load", "tofile", "fromfile"}:
            raise StrategyError(f"file I/O ('{node.attr}') is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "strategy":
            has_fn = True
    if not has_fn:
        raise StrategyError("code must define `def strategy(prices):`")


def _worker(code: str, frames: list, q) -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    except Exception:
        pass
    try:
        import math

        def _imp(name, *a, **k):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ImportError(name)
            return __import__(name, *a, **k)

        outs = []
        for prices in frames:
            g = {"__builtins__": dict(SAFE_BUILTINS, __import__=_imp), "np": np, "pd": pd, "math": math}
            exec(compile(code, "<strategy>", "exec"), g)
            w = g["strategy"](prices.copy())
            if not isinstance(w, pd.DataFrame):
                raise StrategyError("strategy must return a DataFrame of weights")
            outs.append(w)
        q.put(("ok", outs))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)[-600:]}"))


_CTX = mp.get_context("spawn")


def run_many(code: str, frames: list, timeout: float = 90.0) -> list:
    """Execute `strategy(prices)` on each frame inside ONE sandboxed subprocess."""
    check(code)
    q = _CTX.Queue()
    p = _CTX.Process(target=_worker, args=(code, frames, q), daemon=True)
    p.start()
    try:
        status, payload = q.get(timeout=timeout)
    except Exception:
        p.kill()
        raise StrategyError(f"timed out after {timeout:.0f}s") from None
    finally:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
    if status == "err":
        raise StrategyError(payload)
    out = []
    for w, prices in zip(payload, frames):
        w = w.reindex(index=prices.index, columns=prices.columns)
        out.append(w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))
    return out


def run(code: str, prices: pd.DataFrame, timeout: float = 90.0) -> pd.DataFrame:
    return run_many(code, [prices], timeout)[0]
