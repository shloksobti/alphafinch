import numpy as np, pandas as pd

def _f0623(prices, data):
    """Calm Braid Keel: hold sector-relative leaders on braided 12-1 risk-adjusted momentum, hedged by a beta-matched short, with exposure scaled down when cross-sectional momentum dispersion or market stress is high. Hypothesis: momentum underreaction premium is real but crashes in turbulent regimes (the weak last era), so de-risking when market volatility is elevated relative to its own history keeps the drift while cutting regime losses."""
    LOOKBACKS = (250, 280, 310)
    SKIPS = (15, 18, 21, 24)
    VOL_WIN = 126
    SECTOR_WIN = 163
    SECTOR_W = 0.5
    REV_WIN = 5
    REV_W = 0.15
    TOP = 0.25
    CALM_Q = 0.4
    CALM_W = 0.3
    HEDGE = 0.6
    CAP = 0.04
    STRESS_WIN = 252
    STRESS_FLOOR = 0.5
    SMOOTH_HL = 10

    sector = data.sector.reindex(prices.columns)
    rets = prices.pct_change(fill_method=None)
    vol = rets.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().replace(0, np.nan)
    ann_vol = vol * np.sqrt(252)

    parts = []
    for lb in LOOKBACKS:
        for sk in SKIPS:
            m = (prices.shift(sk) / prices.shift(lb) - 1) / ann_vol
            parts.append(tk.zscore(tk.winsorize(m)))
    mom = sum(parts) / len(parts)
    mom_rel = tk.zscore(tk.neutralize(mom, sector))

    r_sec = prices / prices.shift(SECTOR_WIN) - 1
    sec_trend = tk.zscore(r_sec - tk.neutralize(r_sec, sector))
    rev = tk.zscore(tk.winsorize(prices / prices.shift(REV_WIN) - 1))

    score = mom_rel + SECTOR_W * sec_trend.fillna(0) - REV_W * rev.fillna(0)
    valid = prices.notna() & vol.notna() & score.notna()
    score = score.where(valid)

    inv_vol = (1.0 / vol).where(valid)
    lead = (score.rank(axis=1, pct=True) >= 1 - TOP).astype(float) * inv_vol
    lead = lead.div(lead.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    calm = (vol.where(valid).rank(axis=1, pct=True) <= CALM_Q).astype(float) * inv_vol
    calm = calm.div(calm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    longs = tk.cap((1 - CALM_W) * lead + CALM_W * calm, CAP).fillna(0.0)

    beta = tk.rolling_beta(prices, 252).clip(-1, 3).fillna(1.0)
    book_beta = (longs * beta).sum(axis=1).clip(0, 1.5)
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    hedge = valid.astype(float).div(n_valid, axis=0).mul(HEDGE * book_beta, axis=0).fillna(0.0)

    w = longs - hedge
    gross = w.abs().sum(axis=1).clip(lower=1.0)
    w = w.div(gross, axis=0)

    # Regime throttle: market realised vol vs its own trailing history (causal)
    mkt = tk.market(prices)
    mret = mkt.pct_change(fill_method=None) if isinstance(mkt, pd.Series) else mkt
    mvol = mret.rolling(21, min_periods=10).std()
    stress = tk.ts_zscore(mvol, STRESS_WIN).fillna(0.0)
    scale = (1.0 - 0.25 * stress.clip(0, 3)).clip(STRESS_FLOOR, 1.0)
    scale = scale.ewm(halflife=SMOOTH_HL).mean()
    w = w.mul(scale, axis=0)

    w = tk.rebalance(w, every='M')
    return w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)


def _f0469(prices, data):
    """Tethered Twin Snapback: fade short-horizon divergences between each stock and its leave-one-out sector peer basket, scaled by how tightly the two co-move and damped when the gap opened on heavy volume. Hypothesis: quiet gaps between tightly bonded stocks come from impatient flows (index, redemption and retail orders) that pay for immediacy, so supplying that liquidity earns the convergence, while heavy-volume gaps carry news and are left alone."""
    WINDOWS = (5, 10, 21)      # braid of divergence horizons (days)
    VOL_WIN = 63               # window for spread volatility
    CORR_WIN = 126             # window for stock-vs-peer co-movement
    MIN_CORR = 0.2             # below this the "pair" is not a pair
    VOLUME_WIN = 63            # baseline for abnormal volume
    SMOOTH_HL = 3              # signal smoothing halflife (days)
    Q = 0.3                    # fraction held on each side
    CAP = 0.03                 # max weight per name

    rets = prices.pct_change(fill_method=None)
    valid = rets.notna()
    r0 = rets.fillna(0.0)
    sector = data.sector.reindex(prices.columns).fillna("unclassified")

    # Leave-one-out peer basket: mean return of the other stocks in the sector
    sec_sum = r0.T.groupby(sector).transform("sum").T
    sec_cnt = valid.astype(float).T.groupby(sector).transform("sum").T
    n_peers = sec_cnt - valid.astype(float)
    peers = (sec_sum - r0) / n_peers.where(n_peers >= 2)

    # Spread between a stock and its twin basket, in units of its own volatility
    spread = (rets - peers).where(valid)
    spread_vol = spread.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    spread_vol = spread_vol.where(spread_vol > 0)
    div = 0.0
    for w in WINDOWS:
        div = div + spread.rolling(w, min_periods=w).sum() / (spread_vol * np.sqrt(w))
    div = div / len(WINDOWS)

    # Bond strength: only stocks that really move with their peers are "pairs"
    corr = rets.rolling(CORR_WIN, min_periods=CORR_WIN // 2).corr(peers)
    bond = ((corr - MIN_CORR) / (1.0 - MIN_CORR)).clip(0.0, 1.0)

    # Quiet gate: divergence on abnormal volume is likely information, so damp it
    vol = data.volume.reindex_like(prices).where(lambda v: v > 0)
    base = vol.rolling(VOLUME_WIN, min_periods=VOLUME_WIN // 2).median()
    abnormal = (vol.rolling(WINDOWS[0], min_periods=2).mean() / base - 1.0).clip(0.0, 3.0)
    quiet = (1.0 / (1.0 + abnormal)).fillna(1.0)

    # Fade the divergence: long the laggard twin, short the one that ran ahead
    score = -tk.winsorize(tk.zscore(div), z=3) * bond * quiet
    score = score.where(prices.notna())
    score = tk.neutralize(score, data.sector)
    score = tk.smooth(score, SMOOTH_HL).where(prices.notna())

    w = tk.long_short(score, q=Q)
    w = tk.cap(w, CAP)
    w = tk.rebalance(w, every="W")
    return w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)


def strategy(prices, data):
    """The Team: a diversified team of 2 evolved strategies (Calm Braid Keel, Tethered Twin Snapback)."""
    out = prices * 0.0
    out = out + 0.4212 * _f0623(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.5788 * _f0469(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return out
