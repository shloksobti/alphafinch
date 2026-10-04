"""Mandates: the trading rules a strategy must live within, enforced by the engine.

Whatever a strategy asks for, its weights are projected onto the mandate each day before they
are simulated, so a strategy can never break the rules and still look good. The AI is told the
rules too, so it designs within them instead of being clipped.

Presets
  long-only       cash / spot market: no shorting, no leverage (the default for most investors)
  long-short      shorts allowed on any asset, borrow fee charged on short positions
  market-neutral  long-short with net market exposure held within +-10%
  futures         for the futures market: shorting is free and gross exposure may reach 3x (margin);
                  returns are already net of the cash rate, so no financing charge
  derivatives     shorts only where single-stock futures exist (India: the F&O list; elsewhere
                  any asset), gross exposure up to 2x with a financing charge above 1x
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

ANN = 252


@dataclass(frozen=True)
class Mandate:
    name: str = "long-short"
    shorts: str = "all"              # "none", "all", or "fno" (only assets in `shortable`)
    max_gross: float = 1.0           # sum of |weights|; above 1 means leverage
    min_net: float = -1.0            # bounds on sum of weights (net market exposure)
    max_net: float = 1.0
    max_weight: float = 1.0          # cap on any single position (absolute)
    borrow_bps: float = 0.0          # annual fee on short notional
    financing_bps: float = 0.0       # annual fee on gross exposure above 1
    shortable: tuple = field(default_factory=tuple)   # assets that may be shorted when shorts="fno"

    def describe(self) -> str:
        """Plain-language rules for the AI prompt."""
        out = []
        if self.shorts == "none":
            out.append("NO SHORTING: weights must be >= 0 (negative weights are set to zero by the engine)")
        elif self.shorts == "fno":
            out.append(f"Shorting is allowed only in the {len(self.shortable)} assets that have single-stock "
                       "futures; listed in data.shortable (a boolean Series). Other negative weights are set to zero")
        else:
            out.append("Shorting is allowed")
        if self.borrow_bps:
            out.append(f"short positions pay a borrow fee of {self.borrow_bps:.0f} bps a year")
        out.append(f"gross exposure (sum of |weights|) is capped at {self.max_gross:g}"
                   + (f"; exposure above 1 pays {self.financing_bps:.0f} bps a year financing" if self.max_gross > 1 else ""))
        if self.min_net > -1 or self.max_net < 1:
            out.append(f"net exposure (sum of weights) is kept between {self.min_net:+g} and {self.max_net:+g}")
        if self.max_weight < 1:
            out.append(f"no single position may exceed {self.max_weight:.0%} of capital")
        return "; ".join(out) + "."

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "long-only": Mandate("long-only", shorts="none", min_net=0.0),
    "long-short": Mandate("long-short", shorts="all", borrow_bps=50),
    "market-neutral": Mandate("market-neutral", shorts="all", min_net=-0.1, max_net=0.1, borrow_bps=50),
    "futures": Mandate("futures", shorts="all", max_gross=3.0, min_net=-3.0, max_net=3.0),
    "derivatives": Mandate("derivatives", shorts="fno", max_gross=2.0, min_net=-2.0, max_net=2.0,
                           borrow_bps=50, financing_bps=300),
}


def get(name: str | None, **overrides) -> Mandate:
    m = PRESETS[name] if name else Mandate("unconstrained")
    over = {k: v for k, v in overrides.items() if v is not None}
    return replace(m, **over) if over else m


def for_market(name: str | None, market: str, **overrides) -> Mandate | None:
    """Resolve a preset for a market. 'derivatives' shorts only F&O stocks in India; elsewhere
    any stock (via borrow or single-stock futures/CFDs)."""
    if name is None and market == "futures":
        name = "futures"
    if name is None and not any(v is not None for v in overrides.values()):
        return None
    m = get(name, **overrides)
    if m.shorts == "fno":
        if market == "india":
            from .universe import india_fno
            m = replace(m, shortable=tuple(india_fno()))
        else:
            m = replace(m, shorts="all")
    return m


def project(w: pd.DataFrame, m: Mandate | None, short_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Map requested weights onto the mandate, one date at a time (so it stays causal)."""
    if m is None:
        gross = w.abs().sum(axis=1)
        return w.mul(np.where(gross > 1, 1 / gross.replace(0, 1), 1.0), axis=0)
    a = w.values.astype("float64").copy()
    if m.shorts == "none":
        a = np.clip(a, 0, None)
    elif m.shorts == "fno" and short_mask is not None:
        a[:, ~short_mask] = np.clip(a[:, ~short_mask], 0, None)
    a = np.clip(a, -m.max_weight, m.max_weight)
    gross = np.abs(a).sum(axis=1)
    a *= np.where(gross > m.max_gross, m.max_gross / np.where(gross > 0, gross, 1), 1.0)[:, None]
    longs, shorts = np.clip(a, 0, None), np.clip(a, None, 0)
    L, S = longs.sum(axis=1), -shorts.sum(axis=1)
    net = L - S
    # too long: shrink longs until net = max_net; too short: shrink shorts until net = min_net
    hi = net > m.max_net + 1e-12
    k = np.where(hi & (L > 0), (S + m.max_net) / np.where(L > 0, L, 1), 1.0).clip(0, 1)
    longs *= k[:, None]
    lo = net < m.min_net - 1e-12
    k = np.where(lo & (S > 0), (L - m.min_net) / np.where(S > 0, S, 1), 1.0).clip(0, 1)
    shorts *= k[:, None]
    return pd.DataFrame(longs + shorts, index=w.index, columns=w.columns)


def carry_costs(held: np.ndarray, m: Mandate | None) -> np.ndarray:
    """Daily borrow and financing charges per asset (same shape as `held`)."""
    if m is None or (not m.borrow_bps and not m.financing_bps):
        return np.zeros_like(held)
    c = np.clip(-held, 0, None) * m.borrow_bps / 1e4 / ANN
    if m.financing_bps:
        gross = np.abs(held).sum(axis=1, keepdims=True)
        excess = np.clip(gross - 1, 0, None)
        share = np.abs(held) / np.where(gross > 0, gross, 1)
        c = c + excess * share * m.financing_bps / 1e4 / ANN
    return c
