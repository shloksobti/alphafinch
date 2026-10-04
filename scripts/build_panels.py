import sys, time
from alphafinch import data
if __name__ == "__main__":
    for m in sys.argv[1:]:
        t0 = time.time(); last = [0]
        def prog(i, n, s):
            if i - last[0] >= 50 or i == n - 1:
                last[0] = i; print(f"  {m}: {i+1}/{n} {s}", flush=True)
        p = data.load(m, progress=prog)
        print(m, p.shape, p.index[0].date(), '->', p.index[-1].date(), f'{time.time()-t0:.0f}s')
        print('  macro:', list(p.macro.columns) if p.macro is not None else None)
        print('  fund:', {k: f'{v.notna().mean().mean():.0%} filled' for k, v in p.fund.items()})
        print('  sectors:', p.sector.nunique())
