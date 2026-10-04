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


def check_names(code: str, banned: set[str]) -> None:
    """Strategies must choose assets from data, not by name: a hard-coded ticker or sector is a
    route for an AI's memory of what happened later (hindsight) to leak into a backtest."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if v in banned:
                raise StrategyError(f"hard-coded asset or sector name '{v}' is not allowed: select assets from data")
