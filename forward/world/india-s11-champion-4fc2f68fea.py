def strategy(prices, data):
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
