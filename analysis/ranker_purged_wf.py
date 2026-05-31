"""
analysis/ranker_purged_wf.py — HONEST validation of the lambdarank ranker.

The saved GLOBAL_ranker was fit on ALL 2020-present data with NO split, NO embargo,
NO OOS test (the '--validate' path was never built; the '+1.56pp/5d' claim is
in-sample). Scoring it on its training window gave a fake +5.6 Sharpe.

This does it correctly per the research (Lopez de Prado purged CV; "full re-fitting
at each fold, strict information discipline"): for each purged time fold, RETRAIN a
fresh lambdarank on TRAIN-only rows (strictly past, with embargo), then score the
embargoed TEST fold and measure net-of-cost cross-sectional quintile spread. The
model never sees its test data. Reuses purged_kfold_indices (true walk-forward) and
the exact lambdarank config from train_global_ranker.py.

GATE: pooled net Sh > 0.3 AND positive in most folds. Expect a BELIEVABLE number
(somewhere near momentum's +1.0), NOT +5.6 — if it's still huge, suspect leak.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
from analysis.walk_forward import purged_kfold_indices, EMBARGO_DAYS, N_FOLDS

COST_BPS = 10.0
LAMBDARANK = dict(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                  num_leaves=31, max_depth=-1, reg_alpha=0.1, reg_lambda=0.1,
                  verbose=-1, random_state=42)


def build_pooled(tickers, horizon, start):
    """Pooled panel: per ticker, features + forward return, tagged by date."""
    from features.builder import build_feature_dataframe
    from models.classifier import FEATURE_COLUMNS
    rows = []
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            df = build_feature_dataframe(tk, start_date=start)
            if df.empty or len(df) < 300:
                continue
            df = df.copy()
            df["fwd"] = df["close"].shift(-horizon) / df["close"] - 1.0
            df["_ticker"] = tk
            df["_date"] = pd.to_datetime(df["date"]) if "date" in df else pd.to_datetime(df.index)
            rows.append(df[FEATURE_COLUMNS + ["fwd", "_ticker", "_date"]])
        except Exception as e:
            print(f"  skip {tk}: {type(e).__name__}", flush=True)
    pooled = pd.concat(rows, ignore_index=True).dropna(subset=["fwd"])
    return pooled, FEATURE_COLUMNS


def relevance(g):
    """Per-date quintile relevance 0-4 from forward return (ranker label)."""
    pct = g["fwd"].rank(method="dense", pct=True)
    return pd.cut(pct, bins=[-.01,.2,.4,.6,.8,1.01], labels=[0,1,2,3,4]).astype(int)


def fit_fold(train_df, feat):
    train_df = train_df.sort_values("_date")
    train_df = train_df.groupby("_date").filter(lambda x: len(x) >= 2)
    if train_df.empty:
        return None
    train_df = train_df.copy()
    train_df["rel"] = train_df.groupby("_date", group_keys=False).apply(relevance)
    groups = train_df.groupby("_date").size().tolist()
    m = lgb.LGBMRanker(**LAMBDARANK)
    m.fit(train_df[feat].values, train_df["rel"].values, group=groups)
    return m


def score_fold(model, test_df, feat, winsor=0.40):
    """Net-of-cost quintile spread on the test fold, rebalanced per date."""
    spreads = []
    for d, g in test_df.groupby("_date"):
        if len(g) < 10:
            continue
        sc = model.predict(g[feat].values)
        fr = g["fwd"].clip(-winsor, winsor).values
        order = np.argsort(sc)
        k = max(1, len(sc)//5)
        sr = np.nanmean(fr[order[:k]])    # low score = short
        lr = np.nanmean(fr[order[-k:]])   # high score = long
        if not (np.isnan(lr) or np.isnan(sr)):
            spreads.append(lr - sr)
    if len(spreads) < 3:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    # test dates within a fold are consecutive trading days -> ~daily rebal in-fold
    pers = 252
    net = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== RANKER HONEST PURGED-WF (h={args.horizon}d, retrain per fold) ===")
    pooled, feat = build_pooled(tickers, args.horizon, args.start)
    print(f"  pooled: {len(pooled)} rows, {pooled['_date'].nunique()} dates, {len(feat)} feats\n")
    # purged folds over the unique dates
    dates = pd.Series(sorted(pooled["_date"].unique()))
    print(f"  {'fold':<6}{'test window':<26}{'gross':>8}{'net Sh':>9}{'n':>5}{'mean%':>9}")
    print("  " + "-"*64)
    fold_nets = []
    for fi,(tr_idx, te_idx) in enumerate(purged_kfold_indices(dates, n_folds=N_FOLDS, embargo=EMBARGO_DAYS)):
        tr_dates = set(dates.iloc[tr_idx]); te_dates = set(dates.iloc[te_idx])
        tr_df = pooled[pooled["_date"].isin(tr_dates)]
        te_df = pooled[pooled["_date"].isin(te_dates)]
        t0=time.time()
        model = fit_fold(tr_df, feat)
        if model is None:
            continue
        r = score_fold(model, te_df, feat)
        if r:
            g,nt,n,mp = r
            lo,hi = min(te_dates), max(te_dates)
            print(f"  {fi:<6}{str(lo.date())+'..'+str(hi.date()):<26}{g:>+8.2f}{nt:>+9.3f}{n:>5d}{mp:>+9.4f}  ({time.time()-t0:.0f}s)")
            fold_nets.append(nt)
    if fold_nets:
        pos = sum(1 for x in fold_nets if x>0)
        print(f"\n  POOLED: mean fold net Sh {np.mean(fold_nets):+.3f}, positive {pos}/{len(fold_nets)}")
        gate = np.mean(fold_nets)>0.3 and pos>=max(1,int(0.8*len(fold_nets)))
        print(f"  GATE: {'PASS' if gate else 'FAIL'}")
    print("\n  This is the HONEST number (model never sees its test data). If still")
    print("  huge (>3), suspect remaining leak in features. Believable ~ momentum's +1.0.")


if __name__ == "__main__":
    main()
