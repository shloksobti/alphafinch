import numpy as np, pandas as pd

def _f0631(prices, data):
    """Quiet Intraday Relay: sector-neutral book fading quiet-volume residual moves, with extra weight on the intraday (open-to-close) part of the move, sized by VIX stress and anchored by smooth low-volatility residual momentum. Hypothesis: intraday price pressure is driven by order-flow and liquidity demand that reverts, while overnight gaps carry information that persists, so fading the intraday component isolates the overshoot and makes the reversal edge less dependent on parameter choices."""
    LOOKBACKS = (5, 10, 21)
    VOL_WINDOW = 63
    VOLU_SHORT, VOLU_LONG = 10, 120
    NEWS_TILT = 0.5
    MOM_LONG, MOM_SKIP, MOM_MIN = 252, 21, 150
    CALM_WINDOW, CALM_WEIGHT = 126, 0.4
    MOM_WEIGHT = 0.35
    VIX_WINDOW = 302
    REV_SLOPE = 0.25
    BASE, SLOPE, FLOOR = 0.6, 0.2, 0.3
    QUANTILE, BROAD_MIX = 0.3, 0.5
    HALFLIFE, MAX_W = 3, 0.03
    INTRA_WEIGHT, INTRA_LBS = 0.3, (3, 5, 10)

    live = prices.notna()
    sector = data.sector.reindex(prices.columns)
    resid = tk.residual(prices, 252).where(live)
    vol = resid.rolling(VOL_WINDOW, min_periods=20).std().replace(0, np.nan)

    def z(x):
        x = x.replace([np.inf, -np.inf], np.nan).where(live)
        return tk.winsorize(tk.zscore(tk.neutralize(x, sector)), z=3)

    raw = None
    for lb in LOOKBACKS:
        cum = resid.rolling(lb, min_periods=max(2, lb // 2)).sum()
        zi = z(-(cum / (vol * np.sqrt(lb))))
        raw = zi if raw is None else raw + zi
    raw = raw / len(LOOKBACKS)

    # Intraday component: market-demeaned open-to-close return, normalised by its own noise
    intra = np.log(data.close / data.open).replace([np.inf, -np.inf], np.nan).where(live)
    intra = intra.sub(intra.mean(axis=1), axis=0)
    ivol = intra.rolling(VOL_WINDOW, min_periods=20).std().replace(0, np.nan)
    ri = None
    for lb in INTRA_LBS:
        c = intra.rolling(lb, min_periods=max(2, lb // 2)).sum() / (ivol * np.sqrt(lb))
        zi = z(-c).fillna(0.0)
        ri = zi if ri is None else ri + zi
    ri = z(ri / len(INTRA_LBS)).fillna(0.0)
    raw = (1 - INTRA_WEIGHT) * raw.fillna(0.0) + INTRA_WEIGHT * ri

    v = data.volume.reindex(index=prices.index, columns=prices.columns).astype(float)
    v = v.where(v > 0)
    abn = np.log(v.rolling(VOLU_SHORT, min_periods=5).mean() /
                 v.rolling(VOLU_LONG, min_periods=40).mean()).replace([np.inf, -np.inf], np.nan)
    abn_z = tk.winsorize(tk.zscore(abn), z=3).fillna(0.0)
    damp = (1 - NEWS_TILT * abn_z.clip(0, 2) / 2).clip(0.2, 1.0)
    rev = z(raw * damp)

    r_skip = resid.shift(MOM_SKIP)
    win = MOM_LONG - MOM_SKIP
    mom = r_skip.rolling(win, min_periods=MOM_MIN).mean() / \
        r_skip.rolling(win, min_periods=MOM_MIN).std().replace(0, np.nan)
    rets = prices.pct_change(fill_method=None)
    tvol = rets.rolling(CALM_WINDOW, min_periods=CALM_WINDOW // 2).std().replace(0, np.nan)
    calm = -np.log(tvol)
    slow = z(z(mom).fillna(0.0) + CALM_WEIGHT * z(calm).fillna(0.0))

    vix = data.macro['vix'].reindex(prices.index).ffill()
    mu = vix.rolling(VIX_WINDOW, min_periods=60).mean()
    sd = vix.rolling(VIX_WINDOW, min_periods=60).std().replace(0, np.nan)
    fear = ((vix - mu) / sd).clip(-3, 3).ewm(halflife=5, min_periods=1).mean().fillna(0.0)
    rev_share = ((1 - MOM_WEIGHT) * (1 + REV_SLOPE * fear)).clip(0.3, 0.9)

    score = rev.fillna(0.0).mul(rev_share, axis=0) + slow.fillna(0.0).mul(1 - rev_share, axis=0)
    score = score.where(live)
    score = tk.neutralize(tk.winsorize(tk.zscore(score), z=3), sector)

    w_q = tk.long_short(score, q=QUANTILE).fillna(0.0)
    s = score.fillna(0.0)
    w_p = s.div(s.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = (1 - BROAD_MIX) * w_q + BROAD_MIX * w_p
    w = tk.smooth(w, HALFLIFE).fillna(0.0)

    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)
    w = tk.cap(w, MAX_W).fillna(0.0)
    intensity = (BASE + SLOPE * fear).clip(FLOOR, 1.0).fillna(BASE)
    w = w.mul(intensity, axis=0)

    w = w.where(live, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)


import numpy as np, pandas as pd

def _f0516(prices, data):
    """Calm Tide Rebound: sector-neutral long-short blending smooth calm residual momentum with a volume-discounted residual reversal leg that is sized up in high-VIX regimes and ignores stale overnight gaps. Hypothesis: short-term reversal is liquidity-provision compensation that pays most when market stress is elevated, while slow momentum stalls in weak eras, so a stress-scaled reversal leg lifts the weakest era's alpha."""
    LOOKBACK, SKIP, VOL_WINDOW = 252, 21, 126
    MIN_OBS, LOWVOL_WEIGHT, HALFLIFE, MAX_W = 182, 0.5, 5, 0.03
    SECTOR_WEIGHT, SECTOR_WINDOW = 0.5, 165
    REV_WINDOW, REV_WEIGHT = 5, 0.8011
    STRESS_WINDOW, STRESS_SLOPE = 252, 0.3

    resid = tk.residual(prices, 252)
    valid = prices.notna() & resid.notna()
    resid = resid.where(valid)

    r_skip = resid.shift(SKIP)
    win = LOOKBACK - SKIP
    mean_r = r_skip.rolling(win, min_periods=MIN_OBS).mean()
    std_r = r_skip.rolling(win, min_periods=MIN_OBS).std()
    mom = mean_r / std_r.replace(0, np.nan)

    rets = prices.pct_change(fill_method=None)
    vol = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().replace(0, np.nan)
    calm = -np.log(vol)

    rev_raw = resid.rolling(REV_WINDOW, min_periods=REV_WINDOW).sum()
    rev_raw = rev_raw / (resid.rolling(63, min_periods=40).std() * np.sqrt(REV_WINDOW)).replace(0, np.nan)
    vol_ratio = data.volume.rolling(REV_WINDOW, min_periods=3).mean() / \
        data.volume.rolling(63, min_periods=40).mean().replace(0, np.nan)
    rev = -rev_raw / (1 + vol_ratio.clip(0, 4).fillna(1))

    def z(x):
        return tk.zscore(tk.winsorize(tk.neutralize(x, data.sector), z=3))

    mom_z, calm_z, rev_z = z(mom), z(calm), z(rev)
    stock_score = mom_z + LOWVOL_WEIGHT * calm_z
    stock_score = stock_score.where(mom_z.notna() & calm_z.notna())
    stock_score = tk.neutralize(stock_score, data.sector)

    r6 = resid.rolling(SECTOR_WINDOW, min_periods=SECTOR_WINDOW // 2).sum().shift(SKIP // 3)
    sec = data.sector.reindex(prices.columns)
    sector_mean = r6.T.groupby(sec).transform("mean").T
    sector_z = tk.zscore(sector_mean)

    # Stress regime: VIX relative to its own trailing history (causal)
    vix = data.macro["vix"].reindex(prices.index).ffill()
    stress = tk.ts_zscore(vix, STRESS_WINDOW).clip(-2, 2).fillna(0.0)
    rev_scale = (1 + STRESS_SLOPE * stress).clip(0.4, 1.6)

    slow = tk.smooth(stock_score + SECTOR_WEIGHT * sector_z, HALFLIFE)
    fast = tk.smooth(rev_z.fillna(0.0), 3)
    score = slow + REV_WEIGHT * fast.mul(rev_scale, axis=0)
    score = score.where(stock_score.notna())

    rk = tk.rank(score)
    centred = tk.demean(rk)
    w = centred.div(centred.abs().sum(axis=1).replace(0, np.nan), axis=0)
    w = tk.cap(w.fillna(0.0), MAX_W)
    w = tk.rebalance(w, every='W')
    w = w.reindex(index=prices.index, columns=prices.columns)
    return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)


import numpy as np, pandas as pd

def _f0061(prices, data):
    """Twin Current: hold stocks that lead their own sector on risk-adjusted 12-1 momentum, tilted toward sectors with the strongest 6-month trend. Hypothesis: investors under-react both to slow industry-level capital rotation and to firm-specific news, so momentum split into a sector leg and a within-sector leg captures two separate drifts with less noise than raw momentum."""
    LOOKBACK, SKIP = 252, 21        # stock momentum window (12-1)
    SEC_LOOKBACK = 126              # sector trend window
    VOL_WINDOW = 126                # risk adjustment window
    W_STOCK, W_SECTOR = 0.6, 0.4    # blend of within-sector and sector legs
    TOP_Q = 0.30                    # fraction of universe held
    MAX_W = 0.04                    # per-name cap
    MIN_HISTORY = 200               # observations needed before a stock is eligible

    sector = data.sector.reindex(prices.columns)
    rets = prices.pct_change(fill_method=None)

    # Leg 1: risk-adjusted 12-1 stock momentum, measured relative to sector peers
    vol = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std() * np.sqrt(252)
    mom = prices.shift(SKIP) / prices.shift(LOOKBACK) - 1
    stock_score = tk.winsorize(tk.zscore(mom / vol.replace(0, np.nan)), z=3)
    within = tk.zscore(tk.neutralize(stock_score, sector))

    # Leg 2: sector trend, the average 6-month return of the sector's members
    mom6 = prices / prices.shift(SEC_LOOKBACK) - 1
    mom6 = tk.winsorize(tk.zscore(mom6), z=3)
    sec_mean = mom6.T.groupby(sector.values).transform('mean').T
    sec_mean = sec_mean.reindex(index=prices.index, columns=prices.columns)
    sec_score = tk.zscore(sec_mean)

    # Combine: a stock must be a leader within a leading sector to score highest
    score = W_STOCK * within.fillna(0.0) + W_SECTOR * sec_score.fillna(0.0)
    eligible = prices.notna().rolling(LOOKBACK, min_periods=1).sum() >= MIN_HISTORY
    score = score.where(eligible & stock_score.notna() & prices.notna())

    w = tk.long_only(score, q=TOP_Q)
    w = tk.cap(w.fillna(0.0), MAX_W)
    w = tk.rebalance(w, every='M')
    w = w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return w


import numpy as np, pandas as pd

def _f0494(prices, data):
    """Leader Tilt Current: long-biased book that holds calm, steady sector leaders on blended residual and risk-adjusted price 12-1 momentum inside trending sectors, partly funded by a smaller short in the mirror-image laggards. Hypothesis: investors under-react to slow incremental firm and industry news, and this drift is cleaner in steadily rising leaders than in volatile losers that snap back, so the winner side deserves more capital than the loser side."""
    LOOKBACK, SKIP = 252, 21          # 12-1 momentum window
    SEC_LOOKBACK = 126                # sector trend window
    VOL_WIN = 126                     # risk / calmness window
    W_RES, W_PRICE, W_STEADY, W_CALM = 1.0, 0.6, 0.4, 0.4
    W_STOCK, W_SECTOR = 0.65, 0.35    # within-sector leg vs sector leg
    LONG_GROSS, SHORT_GROSS = 0.65, 0.35
    Q, MAX_W = 0.3, 0.04
    MIN_OBS = 200

    close = prices.where(prices > 0)
    sector = data.sector.reindex(prices.columns)
    rets = close.pct_change(fill_method=None)
    enough = rets.notna().rolling(LOOKBACK, min_periods=1).sum() >= MIN_OBS

    def z(x):
        x = x.replace([np.inf, -np.inf], np.nan).where(enough & close.notna())
        return tk.winsorize(tk.zscore(x), z=3)

    # Firm leg (parent A): beta-removed 12-1 momentum per unit of residual risk, plus steadiness
    resid = tk.residual(close, LOOKBACK)
    span, minp = LOOKBACK - SKIP, MIN_OBS - SKIP
    r_sum = resid.rolling(span, min_periods=minp).sum().shift(SKIP)
    r_vol = resid.rolling(span, min_periods=minp).std().shift(SKIP)
    z_res = z(r_sum / (r_vol * np.sqrt(span)).replace(0, np.nan))
    steady = (resid > 0).where(resid.notna()).rolling(span, min_periods=minp).mean().shift(SKIP)

    # Firm leg (parent B): risk-adjusted raw 12-1 price momentum
    vol = rets.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    z_price = z((close.shift(SKIP) / close.shift(LOOKBACK) - 1) / (vol * np.sqrt(252)).replace(0, np.nan))

    stock = (W_RES * z_res + W_PRICE * z_price.fillna(0.0)
             + W_STEADY * z(steady).fillna(0.0) + W_CALM * z(-vol).fillna(0.0))
    stock = stock.where(z_res.notna())
    within = tk.zscore(tk.neutralize(stock, sector))

    # Sector leg: average of members' residual drift (A) and raw 6-month trend (B)
    s_min = SEC_LOOKBACK // 2
    s_sum = resid.rolling(SEC_LOOKBACK, min_periods=s_min).sum()
    s_vol = resid.rolling(SEC_LOOKBACK, min_periods=s_min).std() * np.sqrt(SEC_LOOKBACK)
    sec_raw = 0.5 * z(s_sum / s_vol.replace(0, np.nan)) + 0.5 * z(close / close.shift(SEC_LOOKBACK) - 1)
    sec_mean = sec_raw - tk.neutralize(sec_raw, sector)
    sec_score = tk.winsorize(tk.zscore(sec_mean), z=3)

    score = W_STOCK * within + W_SECTOR * sec_score.fillna(0.0)
    score = score.where(z_res.notna() & within.notna())

    # Asymmetric book: more capital to leaders, smaller short in laggards
    longs = tk.cap(tk.long_only(score, q=Q).fillna(0.0), MAX_W)
    shorts = tk.cap(tk.long_only(-score, q=Q).fillna(0.0), MAX_W)
    w = LONG_GROSS * longs - SHORT_GROSS * shorts
    w = tk.rebalance(w, every='M')
    w = w.reindex(index=prices.index, columns=prices.columns)
    w = w.where(close.notna(), 0.0)
    return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def strategy(prices, data):
    """The Team: a diversified team of 4 evolved strategies (Quiet Intraday Relay v2, Calm Tide Rebound v8, Twin Current, Leader Tilt Current)."""
    out = prices * 0.0
    out = out + 0.4584 * _f0631(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.2226 * _f0516(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.0830 * _f0061(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    out = out + 0.2359 * _f0494(prices, data).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return out
