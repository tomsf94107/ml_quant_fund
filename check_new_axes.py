import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes

m = _merge_outcomes(_load_panel(Path('data/alpha_panel')), Path('accuracy.db'), 5)
mom = 'return_20d__cs_rank'
cands = {
  'insider_60d': 'insider_60d__cs_rank',
  'insider_21d': 'insider_21d__cs_rank',
  'short_pct_float': 'short_pct_float__cs_rank',
  'pc_ratio_snap': 'pc_ratio_snap__cs_rank',
  'eps_surprise': 'eps_surprise__cs_rank',
  'days_to_earnings': 'days_to_earnings__cs_rank',
  'dark_pool_ratio': 'dark_pool_ratio__cs_rank',
  'fund_gp_assets': 'fund_gp_assets__cs_rank',
}
print(f'{"axis":22s} {"nonnull":9s} {"corr_to_mom":12s} verdict')
for name, col in cands.items():
    if col not in m.columns:
        print(f'{name:22s} -         -            NOT IN PANEL')
        continue
    sub = m[[col, mom]].dropna()
    if len(sub) < 500:
        print(f'{name:22s} {m[col].notna().sum():<9d} thin-data')
        continue
    corr = sub[col].rank().corr(sub[mom].rank())
    verdict = 'DECORRELATED-axis' if abs(corr) < 0.3 else 'correlated-skip'
    print(f'{name:22s} {m[col].notna().sum():<9d} {corr:+.3f}       {verdict}')
