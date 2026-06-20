import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes

# Build LONG-SHORT (market-neutral) daily books for mom/gp/op/ep on the panel,
# then test whether combining them beats LS-momentum-alone. This is the test the
# combiner SHOULD have run (its books were long-only -> market-beta-dominated).
m = _merge_outcomes(_load_panel(Path('data/alpha_panel')), Path('accuracy.db'), 5)
sigs = {'mom':'return_20d__cs_rank','gp':'fund_gp_assets__cs_rank',
        'op':'fund_op_equity__cs_rank','ep':'fund_ep__cs_rank',
        'short':'short_pct_float__cs_rank','pc':'pc_ratio_snap__cs_rank'}

def ls_daily(df, col, flip=False):
    out = {}
    for d, g in df.groupby('date'):
        g2 = g.dropna(subset=[col])
        if len(g2) < 20: continue
        k = max(2, len(g2)//5)  # quintile
        s = g2.sort_values(col)
        longs = (s.tail(k) if not flip else s.head(k))['actual_return'].mean()
        shorts = (s.head(k) if not flip else s.tail(k))['actual_return'].mean()
        out[d] = longs - shorts
    return pd.Series(out).sort_index()

# short_pct and pc may need sign flip (short interest = bearish -> flip)
books = {}
for name, col in sigs.items():
    if col not in m.columns: continue
    flip = name in ('short',)  # high short = underperform -> long low-short
    books[name] = ls_daily(m, col, flip=flip)

B = pd.DataFrame(books).dropna()
print(f'LS market-neutral books, {len(B)} days, {B.shape[1]} books')
print('\n=== LS BOOK correlation (market-neutral, the HONEST signal corr) ===')
print(B.corr().round(2))

def sharpe(r): 
    return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0

print('\n=== Sharpe of each LS book + combinations ===')
for c in B.columns:
    print(f'  {c:6s} LS Sharpe: {sharpe(B[c]):+.2f}')
# equal-weight combos
core = [c for c in ['mom','gp','op','ep'] if c in B.columns]
print(f'  EW(mom,gp,op,ep):     {sharpe(B[core].mean(axis=1)):+.2f}')
allcols = list(B.columns)
print(f'  EW(all incl short,pc): {sharpe(B[allcols].mean(axis=1)):+.2f}')
print(f'  mom-alone benchmark:   {sharpe(B["mom"]):+.2f}')
