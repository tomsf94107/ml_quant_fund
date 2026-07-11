"""
IPCA out-of-sample test (Hunt #6, Jun 1 2026) — the decisive test. In-sample R2=0.126
means nothing (IPCA fits in-sample by design). This measures OOS rank-IC under
expanding-window walk-forward + decorrelation from momentum (2nd-return-signal gate).

Uses the package's predict(mean_factor=True) — the authors' OOS method (in-sample
factor mean as expected future factor, the principled fix for unknown f_{t+1}).

GATE: pooled OOS rank-IC >0.02 AND t>2 AND |corr to momentum|<0.3 -> 2nd return signal
-> unlocks C1. OOS IC ~0 despite in-sample 0.126 -> small-T overfit, IPCA dead here.
OOS IC>0 but corr>0.3 -> just re-extracts momentum, adds nothing.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research.ipca_feasibility import build_char_panel, to_monthly, standardize_xs, CHARS


def run_oos(mp, chars, n_factors=4, min_train_months=48, embargo=1):
    from ipca import InstrumentedPCA
    months = sorted(mp["date"].unique())
    ics = []; preds_by_date = {}
    for k in range(min_train_months, len(months) - embargo):
        train_months = months[:k]; test_month = months[k + embargo]
        tr = mp[mp["date"].isin(train_months)].set_index(["ticker", "date"]).sort_index()
        te = mp[mp["date"] == test_month]
        if len(te) < 20:
            continue
        Xtr, ytr = tr[chars], tr["fwd_ret"]
        te_idx = te.set_index(["ticker", "date"]).sort_index()
        Xte = te_idx[chars]
        try:
            reg = InstrumentedPCA(n_factors=n_factors, intercept=False, max_iter=1000)
            reg.fit(X=Xtr, y=ytr)
            # predictOOS: all obs same date t, predicts t+1. Returns array aligned to Xte rows.
            pred = reg.predictOOS(X=Xte, y=te_idx["fwd_ret"], mean_factor=True)
            tk = Xte.index.get_level_values("ticker")
            pser = pd.Series(np.asarray(pred).ravel(), index=tk)
            realized = pd.Series(te_idx["fwd_ret"].values, index=tk)
            both = pd.concat([pser.rename("p"), realized.rename("r")], axis=1).dropna()
            if len(both) < 20:
                continue
            ic, _ = spearmanr(both["p"], both["r"])
            if not np.isnan(ic):
                ics.append((test_month, ic)); preds_by_date[test_month] = both["p"]
        except Exception:
            continue
    return ics, preds_by_date


def momentum_by_date(tickers, start="2018-01-01"):
    from features.builder import _download
    from analysis.momentum_purged_wf import momentum
    closes = {}
    for t in tickers:
        try:
            d = _download(t, start, None)
            closes[t] = pd.Series(d["close"].values, index=pd.to_datetime(d["date"].values))
        except Exception:
            pass
    panel = pd.DataFrame(closes).sort_index()
    return momentum(panel, "mom_6_1")


def main():
    tickers = [t.strip().upper() for t in open(ROOT / "tickers.txt")
               if t.strip() and not t.startswith("#")]
    print(f"Building panel ({len(tickers)} tickers)...")
    panel = build_char_panel(tickers)
    chars = [c for c in CHARS if c in panel.columns]
    mp = to_monthly(panel); mp = standardize_xs(mp, chars)
    print(f"  {len(mp)} rows, {mp['date'].nunique()} months\n")
    print("OOS purged-WF (expanding, 48m min train, 1m embargo)...")
    ics, preds = run_oos(mp, chars)
    if not ics:
        print("  no OOS folds — abort"); return
    ic_vals = np.array([v for _, v in ics])
    print(f"\n  OOS months scored: {len(ic_vals)}")
    print(f"  rank-IC: mean {ic_vals.mean():+.4f}  median {np.median(ic_vals):+.4f}  %>0: {(ic_vals>0).mean()*100:.0f}%")
    print(f"  IC t-stat: {ic_vals.mean()/ic_vals.std()*np.sqrt(len(ic_vals)):+.2f}")
    mom = momentum_by_date(tickers)
    corrs = []
    for d, _ in ics:
        if d not in preds: continue
        mrows = mom.loc[:d]
        if mrows.empty: continue
        mr = mrows.iloc[-1].dropna()
        both = pd.concat([preds[d].rename("ipca"), mr.rename("mom")], axis=1).dropna()
        if len(both) > 20:
            c, _ = spearmanr(both["ipca"], both["mom"])
            if not np.isnan(c): corrs.append(c)
    if corrs:
        print(f"\n  corr(IPCA pred, momentum) over {len(corrs)} months: mean {np.mean(corrs):+.3f}  median {np.median(corrs):+.3f}")
    print("\n" + "="*70)
    print("GATE: OOS IC >0.02 AND t>2 AND |corr mom|<0.3 -> 2nd RETURN signal, unlocks C1")
    print("  IC ~0 -> small-T overfit, IPCA dead here.  IC>0 but corr>0.3 -> just momentum.")
    print("="*70)


if __name__ == "__main__":
    main()
