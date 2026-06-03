"""
Purged-WF test of bucket_cap on the LONG-ONLY TOP-DECILE momentum book
(mirrors live momentum_shadow.py: decile, cap=3), comparing:
  (1) no cap         — raw top decile
  (2) cap=3 fragmented — current live taxonomy
  (3) cap=3 consolidated — AI-hardware buckets merged (fake-diversification fix)
Same purged folds / embargo / cost as momentum_purged_wf.py. Long-only object
(top-decile mean fwd return, net of cost), NOT the long-short quintile spread.
Run: PYTHONPATH=. python3 analysis/momentum_cap_wf.py
"""
import pandas as pd, numpy as np, csv
from pathlib import Path
from analysis.walk_forward import purged_kfold_indices, EMBARGO_DAYS, N_FOLDS
from analysis.momentum_purged_wf import build_close_panel, momentum, COST_BPS

ROOT = Path(__file__).resolve().parent.parent
DECILE = 0.10
FWD = 20
STEP = 20
CAP = 3

# ── taxonomy: fragmented (live) vs consolidated (merge AI-hardware + dup buckets) ──
def load_bucket_map(consolidate=False):
    meta = pd.read_csv(ROOT / "tickers_metadata.csv")
    bmap = dict(zip(meta["ticker"].str.upper(), meta["bucket"].fillna("UNK")))
    if not consolidate:
        return bmap
    MERGE = {
        # AI-hardware complex -> one bucket (the ~6-effective-bet cluster)
        "Core Silicon":"AI_HW","Custom Silicon":"AI_HW","Memory":"AI_HW",
        "Networking":"AI_HW","Server Hardware":"AI_HW","Neoclouds":"AI_HW",
        "Semiconductor Equipment":"AI_HW","Semi Equipment":"AI_HW",
        "Hyperscaler":"AI_HW","Physical AI":"AI_HW",
        # power variants
        "Power/Cooling":"Power","Power/Industrial":"Power",
        # consumer variants
        "Consumer Tech":"Consumer","E-commerce":"Consumer",
    }
    return {t: MERGE.get(b, b) for t, b in bmap.items()}

def top_decile(score_row, k, bmap=None, cap=None):
    ranked = score_row.dropna().sort_values(ascending=False)
    if cap is None or bmap is None:
        return list(ranked.head(k).index)
    top, per = [], {}
    for t in ranked.index:
        b = bmap.get(t, "UNK")
        if per.get(b, 0) < cap:
            top.append(t); per[b] = per.get(b, 0) + 1
        if len(top) >= k:
            break
    return top

def fold_longonly(sig, ret_fwd, fold_dates, step_dates, bmap=None, cap=None):
    use = [d for d in step_dates if d in fold_dates]
    rets = []
    for d in use:
        row = sig.loc[d].dropna()
        if len(row) < 10:
            continue
        k = max(1, int(len(row) * DECILE))
        picks = top_decile(row, k, bmap, cap)
        fr = ret_fwd.loc[d]
        vals = [fr.get(t, np.nan) for t in picks]
        m = np.nanmean(vals)
        if not np.isnan(m):
            rets.append(m)
    if len(rets) < 2:
        return None
    sa = np.array(rets); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    pers = 252/20
    net = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(net),3), len(rets), round(float(sa.mean())*100,3)

def run_config(panel, kind, label, bmap=None, cap=None):
    sig = momentum(panel, kind)
    ret_fwd = (panel.shift(-FWD)/panel - 1.0).clip(-0.40, 0.40)
    valid = panel.index[252:len(panel)-FWD]
    step_dates = list(valid[::STEP])
    daily = pd.Series(list(valid))
    nets = []
    for fi, (tr, te) in enumerate(purged_kfold_indices(daily, n_folds=N_FOLDS, embargo=EMBARGO_DAYS)):
        fd = set(daily.iloc[te].tolist())
        r = fold_longonly(sig, ret_fwd, fd, step_dates, bmap, cap)
        if r: nets.append(r[0])
    if nets:
        return round(float(np.mean(nets)),3), len(nets), [n for n in nets]
    return None, 0, []

if __name__ == "__main__":
    import sys
    print("Loading universe + building close panel...")
    meta = pd.read_csv(ROOT / "tickers_metadata.csv")
    tickers = meta["ticker"].str.upper().tolist()
    panel = build_close_panel(tickers, "2018-01-01")
    print(f"panel: {panel.shape[1]} tickers x {panel.shape[0]} days\n")

    frag = load_bucket_map(consolidate=False)
    cons = load_bucket_map(consolidate=True)

    for kind in ("mom_6_1", "mom_12_1"):
        print(f"=== {kind} : long-only top-decile, purged WF ({N_FOLDS} folds) ===")
        configs = [
            ("no cap",            None, None),
            ("cap=3 fragmented",  frag, CAP),
            ("cap=3 consolidated",cons, CAP),
        ]
        print(f"  {'config':<22}{'mean net Sh':>12}{'folds':>7}   per-fold")
        print("  " + "-"*60)
        for label, bmap, cap in configs:
            m, nf, perfold = run_config(panel, kind, label, bmap, cap)
            pf = " ".join(f"{x:+.2f}" for x in perfold)
            print(f"  {label:<22}{m:>+12.3f}{nf:>7}   [{pf}]")
        print()
