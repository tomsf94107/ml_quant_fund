"""
analysis/phase0_band_sweep.py — Phase 0 turnover-reduction test.

Phase 0 showed inverted reversal survives GROSS (149 +0.29, 578 +0.54) but daily
decile-flipping turnover (69%, 87%) kills NET (149 +0.10, 578 -0.37).

This sweeps a NO-TRADE BAND (Garleanu-Pedersen 2013 "inaction band"): a name ENTERS
the inverted long book at the bottom-Xpct of pred, but only EXITS when it drifts back
past a WIDER exit pct. Names hovering near the boundary stay held instead of churning.

Keeps the DAILY h=5 signal (the edge lives at h=5; weekly is DEAD per horizon sweep).
The question: is there a band width where cost-saved > edge-lost-to-staleness?

Reuses the exact build_pooled_pit + purged-WF path from eval_global_pit for
apples-to-apples comparison with the Phase 0 numbers.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.eval_global_pit import build_pooled_pit
from analysis.walk_forward import purged_kfold_indices, detect_feature_cols
from models.classifier import XGB_PARAMS
from xgboost import XGBClassifier

COST_BPS = 10.0
DECILE_ENTRY = 0.10  # enter at bottom/top 10%


def pooled_oos_preds(pooled, feat_cols, n_folds, embargo):
    """Run purged-WF, return df with date/pred/ret/ticker for OOS folds."""
    X = pooled[feat_cols].values
    y = pooled["actual_up"].astype(int).values
    ret = pooled["actual_return"].values.astype(float)
    dates = pooled["prediction_date"]
    tk = pooled["ticker"].values
    rows_d, rows_p, rows_r, rows_t = [], [], [], []
    for tr, te in purged_kfold_indices(dates, n_folds=n_folds, embargo=embargo):
        if len(np.unique(y[tr])) < 2:
            continue
        m = XGBClassifier(**dict(XGB_PARAMS))
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows_d.extend(dates.iloc[te].tolist()); rows_p.extend(p.tolist())
        rows_r.extend(ret[te].tolist());        rows_t.extend(tk[te].tolist())
    return pd.DataFrame({"date": rows_d, "pred": rows_p, "ret": rows_r, "ticker": rows_t})


def sweep_band(ic_df, exit_pct):
    """
    Inverted book: LONG the model's bottom decile, SHORT its top.
    Enter long at pred <= entry quantile; exit only when pred rises above exit quantile.
    exit_pct=None means no band (flip every day = original Phase 0).
    Returns (gross_sharpe, net_sharpe, turnover, n_dates).
    """
    long_held, short_held = set(), set()
    spreads, turnovers = [], []
    for d in sorted(ic_df["date"].unique()):
        g = ic_df[ic_df["date"] == d]
        if len(g) < 10:
            continue
        g = g.sort_values("pred")
        n = len(g)
        ranks = {t: i / n for i, t in enumerate(g["ticker"])}  # 0=lowest pred
        rets  = dict(zip(g["ticker"], g["ret"]))
        if exit_pct is None:
            k = max(1, n // 10)
            new_long  = set(g.head(k)["ticker"])   # lowest pred -> long (inverted)
            new_short = set(g.tail(k)["ticker"])   # highest pred -> short
        else:
            # keep held names until they cross the exit band; add new ones past entry
            new_long  = {t for t in long_held  if t in ranks and ranks[t] <= exit_pct}
            new_short = {t for t in short_held if t in ranks and ranks[t] >= 1 - exit_pct}
            new_long  |= {t for t, r in ranks.items() if r <= DECILE_ENTRY}
            new_short |= {t for t, r in ranks.items() if r >= 1 - DECILE_ENTRY}
        # turnover vs prior held book (both legs)
        if long_held or short_held:
            lt = 1.0 - len(new_long & long_held) / max(1, len(new_long))
            st = 1.0 - len(new_short & short_held) / max(1, len(new_short))
            turnovers.append((lt + st) / 2.0)
        # realized inverted spread of the HELD book: long-leg ret - short-leg ret
        lr = np.mean([rets[t] for t in new_long  if t in rets]) if new_long  else 0.0
        sr = np.mean([rets[t] for t in new_short if t in rets]) if new_short else 0.0
        spreads.append(lr - sr)
        long_held, short_held = new_long, new_short
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    turn = float(np.mean(turnovers)) if turnovers else 1.0
    cost = turn * (COST_BPS / 1e4) * 2.0
    gross = (sa.mean() / sd * np.sqrt(50)) if sd > 0 else float("nan")
    net   = ((sa.mean() - cost) / sd * np.sqrt(50)) if sd > 0 else float("nan")
    return round(float(gross), 3), round(float(net), 3), round(turn, 3), len(sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--tickers-file", default="tickers.txt")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in (ROOT / args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== Phase 0 band sweep: {len(tickers)} tickers, h={args.horizon} ===", flush=True)
    pooled = build_pooled_pit(tickers, args.start_date, args.horizon)
    feat = [c for c in detect_feature_cols(pooled)]
    ic_df = pooled_oos_preds(pooled, feat, args.folds, args.embargo)
    print(f"  pooled OOS preds: {len(ic_df)} rows\n")
    print(f"  {'band (exit pct)':<18}{'turnover':>10}{'gross Sh':>10}{'net Sh':>10}{'n_dates':>9}")
    print("  " + "-" * 56)
    for label, ex in [("none (daily flip)", None), ("exit@15%", 0.15),
                      ("exit@20%", 0.20), ("exit@25%", 0.25), ("exit@30%", 0.30), ("exit@35%", 0.35), ("exit@40%", 0.40), ("exit@50%", 0.50)]:
        r = sweep_band(ic_df, ex)
        if r:
            g, nt, tu, nd = r
            print(f"  {label:<18}{tu*100:>9.1f}%{g:>+10.3f}{nt:>+10.3f}{nd:>9d}")
    print("\n  READ: a band where net Sh > 0 (ideally toward +0.5) = turnover SOLVABLE,")
    print("  reversion is tradeable -> build Phase 2a PCA residual reversal.")
    print("  If net stays <=0 at all bands -> signal decays too fast to hold -> reconsider.")


if __name__ == "__main__":
    main()
