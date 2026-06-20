import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes

# The combiner measured corr of monthly BOOK RETURNS (long/short portfolio P&L).
# This check measured corr of the FEATURE RANKS (cross-sectional signal values).
# These are DIFFERENT. A feature can be rank-decorrelated from momentum's RANKS
# but its long/short BOOK can still co-move with momentum's book (both load on
# the same risk factor in returns). Reconcile:

m = _merge_outcomes(_load_panel(Path('data/alpha_panel')), Path('accuracy.db'), 5)
mom = 'return_20d__cs_rank'
gp  = 'fund_gp_assets__cs_rank'

# 1. feature-rank corr (what check_new_axes measured)
sub = m[[gp, mom]].dropna()
print('feature-RANK corr gp vs mom:', round(sub[gp].rank().corr(sub[mom].rank()), 3))

# 2. but do they pick the SAME STOCKS at the extremes? (top decile overlap)
#    books trade the extremes, so extreme-overlap drives book correlation
def top_decile_overlap(df, a, b):
    ov = []
    for d, g in df.groupby('date'):
        if len(g) < 20: continue
        ta = set(g.nlargest(max(2,len(g)//10), a)['ticker'])
        tb = set(g.nlargest(max(2,len(g)//10), b)['ticker'])
        if ta: ov.append(len(ta & tb)/len(ta))
    return np.mean(ov) if ov else np.nan
m2 = m.dropna(subset=[gp, mom])
print('top-decile overlap gp vs mom:', round(top_decile_overlap(m2, gp, mom), 3),
      '(0.1=random, >0.3=they pick same names)')

# 3. the REAL question for combiner value: book-return corr.
#    proxy: daily long-short return of each signal, then corr.
def ls_daily_ret(df, sigcol):
    out = {}
    for d, g in df.groupby('date'):
        if len(g) < 20: continue
        k = max(2, len(g)//10)
        longs = g.nlargest(k, sigcol)['actual_return'].mean()
        shorts = g.nsmallest(k, sigcol)['actual_return'].mean()
        out[d] = longs - shorts
    return pd.Series(out)
r_gp = ls_daily_ret(m2, gp); r_mom = ls_daily_ret(m2, mom)
both = pd.concat([r_gp, r_mom], axis=1).dropna()
print('BOOK-return corr gp vs mom (the combiner-relevant number):',
      round(both.iloc[:,0].corr(both.iloc[:,1]), 3))
