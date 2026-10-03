"""Terminal line charts.

`chart` draws solid lines with box-drawing characters (asciichart style), which render
crisply in any monospace font. Several series share one y-axis; later series draw on top.
"""
from __future__ import annotations

import numpy as np
from rich.text import Text


def _resample(y: np.ndarray, n: int) -> np.ndarray:
    """Average within each column (smooths daily noise instead of sampling it)."""
    if len(y) <= n:
        idx = np.linspace(0, len(y) - 1, n)
        return np.interp(idx, np.arange(len(y)), y)
    edges = np.linspace(0, len(y), n + 1).astype(int)
    return np.array([y[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def chart(series: list[tuple[np.ndarray, str]], width: int = 56, height: int = 12,
          upto: float = 1.0, log: bool = True) -> Text:
    """Render series (values, rich style) as solid lines. `upto` in (0, 1] draws only the
    first fraction of each series, for animation."""
    ys = []
    for v, _ in series:
        v = np.asarray(v, float)
        ys.append(np.log(v) if log else v)
    cols = max(2, width)
    rs = [_resample(y, cols) for y in ys]
    lo = min(float(r.min()) for r in rs)
    hi = max(float(r.max()) for r in rs)
    span = hi - lo if hi > lo else 1.0
    grid = [[" "] * cols for _ in range(height)]
    style = [[""] * cols for _ in range(height)]
    last = max(2, int(cols * upto))
    LIGHT = dict(h="─", up_from="╯", up_to="╭", down_from="╮", down_to="╰", v="│")
    HEAVY = dict(h="━", up_from="┛", up_to="┏", down_from="┓", down_to="┗", v="┃")
    for r, (_, st) in zip(rs, series):
        g = HEAVY if "bold" in st else LIGHT
        rows = np.round((1 - (r - lo) / span) * (height - 1)).astype(int)
        for x in range(last):
            y0 = rows[x]
            if x == 0:
                grid[y0][x], style[y0][x] = g["h"], st
                continue
            yp = rows[x - 1]
            if y0 == yp:
                grid[y0][x], style[y0][x] = g["h"], st
            elif y0 < yp:                       # going up (smaller row index)
                grid[yp][x], style[yp][x] = g["up_from"], st
                grid[y0][x], style[y0][x] = g["up_to"], st
                for k in range(y0 + 1, yp):
                    grid[k][x], style[k][x] = g["v"], st
            else:                               # going down
                grid[yp][x], style[yp][x] = g["down_from"], st
                grid[y0][x], style[y0][x] = g["down_to"], st
                for k in range(yp + 1, y0):
                    grid[k][x], style[k][x] = g["v"], st
    out = Text()
    for r in range(height):
        for c in range(cols):
            out.append(grid[r][c], style=style[r][c])
        if r < height - 1:
            out.append("\n")
    return out
