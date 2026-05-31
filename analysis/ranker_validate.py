"""
analysis/ranker_validate.py — validate the TRAINED lambdarank ranker through the
SAME honest gate that killed 4 signals and passed momentum.

The ranker (models/saved/GLOBAL_ranker_{h}d.joblib) is the architecturally-correct,
research-backed model (learning-to-rank for cross-sectional momentum — Poh/Lim/
Zohren/Roberts show ~3x Sharpe vs sorting a classifier). It's wired into the
generator but LOGGED-ONLY: the BUY decision still uses the broken per-ticker prob.
Its claimed "+1.56pp/5d Q5-Q1" was never validated through our harness.

This loads the trained ranker, scores the full universe each rebalance day using
the SAME build_feature_dataframe the ranker was trained on (no hand-rolled feature
drift), ranks into quintiles, longs top / shorts bottom, and measures net-of-cost
spread over NON-overlapping windows + per-regime. GATE: net Sh > 0.3 AND positive
in current regime.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import joblib

COST_BPS = 10.0


def score_universe(tickers, horizon, start):
    """Build features per ticker via the SAME builder the ranker trained on, score
    each row with the ranker, and assemble a date x ticker panel of (score, fwd_ret)."""
    from features.builder import build_feature_dataframe
    rk = joblib.load(ROOT / "models" / "saved" / f"GLOBAL_ranker_{horizon}d.joblib")
    scores, fwds = {}, {}
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            df = build_feature_dataframe(tk, start_date=start)
            if df.empty or len(df) < 260:
                continue
            df = df.copy()
            fwd = (df["close"].shift(-horizon) / df["close"] - 1.0)
            # subset to EXACTLY the ranker's feature_cols (numeric only) — passing
            # the full df includes date/ticker object cols which LightGBM rejects
            Xdf = df.reindex(columns=rk.feature_cols).fillna(0.0).astype("float32")
            sc = rk.predict_proba(Xdf)           # ranker score per row
            sc = np.asarray(sc)
            if sc.ndim == 2:                      # predict_proba may return (n,2)
                sc = sc[:, 1]
            s = pd.Series(sc, index=pd.to_datetime(df["date"]) if "date" in df else df.index)
            f = pd.Series(fwd.values, index=s.index)
            scores[tk] = s; fwds[tk] = f
        except Exception as e:
            print(f"  skip {tk}: {type(e).__name__}", flush=True)
    score_panel = pd.DataFrame(scores).sort_index()
    fwd_panel   = pd.DataFrame(fwds).reindex_like(score_panel)
    return score_panel, fwd_panel


def backtest(score_panel, fwd_panel, horizon, winsor=0.40):
    step = max(horizon, 5)  # non-overlapping-ish: rebalance every `horizon` days
    idx = score_panel.index
    dates = idx[::step]
    spreads = []
    for d in dates:
        row = score_panel.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = fwd_panel.loc[d].clip(-winsor, winsor)
        order = row.sort_values()
        k = max(1, len(order)//5)
        lr = np.nanmean([fr.get(t, np.nan) for t in order.tail(k).index])  # top score = long
        sr = np.nanmean([fr.get(t, np.nan) for t in order.head(k).index])
        if not (np.isnan(lr) or np.isnan(sr)):
            spreads.append(lr - sr)
    if len(spreads) < 3:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    pers = 252/step
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2022-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== RANKER VALIDATION (trained {args.horizon}d model): {len(tickers)} names ===")
    sp, fp = score_universe(tickers, args.horizon, args.start)
    print(f"  scored panel: {sp.shape[0]} days x {sp.shape[1]} names\n")
    r = backtest(sp, fp, args.horizon)
    if r:
        g, nt, nd, mp = r
        print(f"  FULL ({args.start}->now): gross {g:+.2f}  net Sh {nt:+.3f}  n={nd}  mean% {mp:+.3f}")
        print(f"  GATE net>0.3: {'PASS' if nt>0.3 else 'FAIL'}")
    # per-regime
    print("\n  --- per-regime ---")
    for rl, rs, re_ in [("2022 bear","2022-01-01","2022-12-31"),
                        ("2023-24","2023-01-01","2024-12-31"),
                        ("2025-26 (current)","2025-01-01","2026-12-31")]:
        sub_s = sp.loc[(sp.index>=rs)&(sp.index<=re_)]
        sub_f = fp.loc[(fp.index>=rs)&(fp.index<=re_)]
        if len(sub_s) < 60:
            print(f"  {rl:<20} (short)"); continue
        rr = backtest(sub_s, sub_f, args.horizon)
        if rr: print(f"  {rl:<20} net Sh {rr[1]:+.3f}  n={rr[2]}  mean% {rr[3]:+.3f}")
    print("\n  If 5d ranker FAILS gate -> the '+1.56pp' claim was unvalidated; train 20d.")
    print("  If PASS + positive current -> rewire BUY decision to ranker top-quintile.")


if __name__ == "__main__":
    main()
