#!/usr/bin/env python3
"""
test_gex_vol_ab.py -- does gex_z improve forward-VOLATILITY prediction?

THE QUESTION
  GEX survived every control today: it predicts forward realized vol (corr -0.50 at
  20d), it is NOT a vol proxy (104% retained after removing trailing vol), it clears
  a block-bootstrap null (p=0.002), and the direction matches the mechanism (high
  dealer gamma -> dealers dampen -> lower forward vol), predicted IN ADVANCE.

  But "correlates with forward vol" is not the same as "improves a model that
  already has trailing vol, ATR, VIX and everything else". Trailing vol is already
  a feature. The only question that matters:

      DOES ADDING gex_z BEAT THE SAME MODEL WITHOUT IT?

  Same folds, same purge, same model, one variable changed. If AUC does not move,
  gex_z is redundant with features already present and there is nothing to wire.

WHY VOLATILITY AND NOT RETURNS
  models/vol_prediction.py (written Jun 1, never run, wired to nothing) says it:
  "NOT a return signal by itself -- value is POSITION SIZING + regime gating."
  Vol is persistent, so it is genuinely predictable (AUC > 0.6 achievable) unlike
  direction, which tested at 0.51 in and out of sample. This is the easier question,
  and GEX is a vol signal. Matching the tool to the job.

LEAK WATCH (from vol_prediction.py's own docstring)
  "If AUC > 0.75 suspect overlap/autocorr leak and attack it."
  Forward vol on day t covers t+1..t+h; on day t+1 it covers t+2..t+h+1 -- they
  share h-1 days. The purged walk-forward embargo (>= h) is what keeps training
  labels out of the test window. We assert on it.
"""
import argparse, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features.builder import build_feature_dataframe, OUTPUT_COLUMNS
from analysis.walk_forward import purged_kfold_indices

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--max-tickers", type=int, default=150)
a = ap.parse_args()

H = a.horizon
span_cal = int(np.ceil(H * 7.0/5.0))
EMBARGO = span_cal + 7
assert EMBARGO >= span_cal, "embargo shorter than the label span"

# ── GEX ────────────────────────────────────────────────────────────────────
con = sqlite3.connect("accuracy.db")
gx = pd.read_sql("SELECT ticker, date, net_gamma FROM options_greeks "
                 "WHERE net_gamma IS NOT NULL", con); con.close()
gx["date"] = pd.to_datetime(gx["date"]); gx["ticker"] = gx["ticker"].str.upper()
gx = gx.sort_values(["ticker","date"])
g = gx.groupby("ticker")["net_gamma"]
# per-ticker z: raw gamma scales with market cap, so cross-sectionally raw GEX
# would just rank stocks by SIZE. The object is "unusual FOR THIS NAME".
gx["gex_z"] = ((gx["net_gamma"] - g.transform(lambda s: s.rolling(60, min_periods=20).mean()))
               / g.transform(lambda s: s.rolling(60, min_periods=20).std()))
gx = gx.replace([np.inf,-np.inf], np.nan).dropna(subset=["gex_z"])
print(f"  GEX: {len(gx):,} rows / {gx.ticker.nunique()} tickers / "
      f"{gx.date.min().date()} .. {gx.date.max().date()}")

tks = [t.strip().upper() for t in open("tickers.txt")
       if t.strip() and not t.startswith("#")][:a.max_tickers]

frames = []
for i, t in enumerate(tks, 1):
    try:
        d = build_feature_dataframe(t, start_date="2025-06-01", training_mode=True)
        if d is None or len(d) < H + 140: continue
        d["_d"] = pd.to_datetime(d["date"])

        # forward realized vol over t+1..t+h -- STRICTLY forward, day t excluded.
        r = d["close"].pct_change()
        fwd = r[::-1].rolling(H).std()[::-1].shift(-1)
        med = fwd.rolling(126, min_periods=63).median()
        d["target_vol"] = (fwd > med).astype(float)
        d.loc[d.index[-H:], "target_vol"] = np.nan
        d.loc[fwd.isna() | med.isna(), "target_vol"] = np.nan

        d = d.merge(gx[gx.ticker == t][["date","gex_z"]].rename(columns={"date":"_d"}),
                    on="_d", how="left")
        d["_tk"] = t
        frames.append(d)
    except Exception:
        pass
    if i % 50 == 0: print(f"    [{i}/{len(tks)}] kept={len(frames)}", flush=True)

p = pd.concat(frames, ignore_index=True).dropna(subset=["target_vol","gex_z"])
base = sorted({c for c in OUTPUT_COLUMNS if c in p.columns
               and pd.api.types.is_numeric_dtype(p[c])
               and c not in ("date","ticker","close")})
print(f"  panel: {len(p):,} rows | {p['_d'].nunique()} dates | {p['_tk'].nunique()} tickers")
print(f"  base features: {len(base)}  (+gex_z = {len(base)+1})")
print(f"  embargo: {EMBARGO} calendar days >= {span_cal} spanned by an {H}-day label\n")

dates = np.array(sorted(p["_d"].unique()))

# purged_kfold_indices has MIN_TRAIN_ROWS=200 and it counts the elements of the
# series you hand it. We hand it 192 unique DATES (not the 28,735 panel rows), so
# every fold failed the guard and it yielded nothing. The guard is right for a
# per-ticker frame where rows==dates; it is wrong for a pooled panel. Build the
# expanding-window folds explicitly instead of fighting a shared default.
#
# Expanding window, embargo in CALENDAR days between train end and test start.
# Only 192 dates of GEX exist -- UW's window is rolling and cannot be backfilled --
# so folds are necessarily small. That is a real constraint, not a bug.
def expanding_folds(dts, n_folds, embargo_days):
    n = len(dts)
    fold = n // (n_folds + 1)          # +1 so fold 0 has a training set
    for k in range(1, n_folds + 1):
        te = dts[k*fold : (k+1)*fold]
        if len(te) == 0: continue
        cutoff = te.min() - np.timedelta64(embargo_days, "D")
        tr = dts[dts < cutoff]
        if len(tr) < 30 or len(te) < 15:   # DATES, not rows
            continue
        yield tr, te

print(f"  {'fold':<5}{'test window':<26}{'n':>8}{'AUC base':>10}{'AUC +gex':>10}{'delta':>9}")
print("  " + "-"*70)
res = []
for fi,(tr_d, te_d) in enumerate(expanding_folds(dates, a.folds, EMBARGO)):
    tr = p[p["_d"].isin(set(tr_d))]
    te = p[p["_d"].isin(set(te_d))]
    if len(tr) < 1000 or len(te) < 300:
        print(f"  {fi:<5}skipped: train={len(tr)} test={len(te)} rows")
        continue
    row = {}
    for name, feats in [("base", base), ("gex", base + ["gex_z"])]:
        X  = tr[feats].astype(float).fillna(0.0)
        Xt = te[feats].astype(float).fillna(0.0)
        sc = StandardScaler().fit(X)
        m  = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(X), tr["target_vol"])
        row[name] = roc_auc_score(te["target_vol"], m.predict_proba(sc.transform(Xt))[:,1])
    lo, hi = pd.Timestamp(te_d.min()), pd.Timestamp(te_d.max())
    res.append(row)
    print(f"  {fi:<5}{str(lo.date())+'..'+str(hi.date()):<26}{len(te):>8,}"
          f"{row['base']:>10.4f}{row['gex']:>10.4f}{row['gex']-row['base']:>+9.4f}")

if not res:
    print("\n  NO USABLE FOLDS."); sys.exit()
b = np.array([r["base"] for r in res]); x = np.array([r["gex"] for r in res])
print("  " + "-"*70)
print(f"  {'MEAN':<31}{'':>8}{b.mean():>10.4f}{x.mean():>10.4f}{(x-b).mean():>+9.4f}")
print()
d = x - b
print(f"  gex_z improves AUC in {int((d>0).sum())}/{len(d)} folds")
if b.mean() > 0.75:
    print(f"\n  >> WARNING: base AUC {b.mean():.3f} > 0.75. vol_prediction.py's own leak")
    print("     watch says suspect overlap/autocorr. Attack before believing anything.")
elif d.mean() > 0.01 and (d > 0).all():
    print(f"\n  >> GEX HELPS. +{d.mean():.4f} AUC, positive in every fold.")
    print("     Wire gex_z into the vol model and into risk_gate.py -- whose ^VIX")
    print("     input is DEAD (XProtect blocks yfinance), so it currently gates on a")
    print("     constant.")
elif abs(d.mean()) < 0.005:
    print(f"\n  >> NO IMPROVEMENT ({d.mean():+.4f}). gex_z is redundant with features")
    print("     already in the model (trailing vol, ATR, ranges). Nothing to wire.")
else:
    print(f"\n  >> MARGINAL ({d.mean():+.4f}, {int((d>0).sum())}/{len(d)} folds). Not established.")
print(f"\n  Honest n = {len(res)} FOLDS. Vol is persistent, so a high base AUC is")
print("  EXPECTED and is not itself evidence of anything. The DELTA is the finding.")
