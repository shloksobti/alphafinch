import numpy as np, pandas as pd

def strategy(prices, data):
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
