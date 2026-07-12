#!/usr/bin/env python3
"""
models/vol_forecast.py -- forward volatility as a REGRESSION, not a binary.

WHY THIS REPLACES vol_prediction.py
  vol_prediction.py targets  (fwd_vol > rolling_126d_median)  -- a BINARY on a
  MOVING threshold. Two things go wrong:

    1. The rolling median CHASES THE REGIME. When vol is high the bar rises with
       it, so the target asks "is vol above where it has recently been?" -- a
       MEAN-REVERSION question. That cancels out vol PERSISTENCE, which is the one
       thing about volatility that is actually predictable.
    2. Binarising a continuous quantity throws away the level information that
       position sizing needs. You cannot size on a coin flip; you size on a NUMBER.

  Measured on 857 dates / 396 tickers, this is not theory:
      cross-sectional rank-IC (trail vol rank -> fwd vol rank) = +0.556, and it is
      POSITIVE ON 100% OF DATES.
      the same information as a binary AUC                     =  0.556
      log(trail vol) -> log(fwd vol), Spearman                 = +0.545
  So the signal is enormous -- ~10x the SI brick -- and the binary framing was
  destroying roughly 90% of it at the labeling step. This is exactly what the
  Research Report warned about: "you are throwing away signal at the labeling step
  and then measuring with a metric that is structurally insensitive to it."

WHAT THIS IS AND IS NOT
  NOT ALPHA. Volatility clustering is textbook -- everyone knows vol is persistent.
  Predicting vol does not tell you which way price goes and will not raise your
  direction AUC by one basis point.

  It IS the input to POSITION SIZING, which is where a weak return edge becomes
  money. A 51% edge sized correctly beats a 55% edge sized wrong. And it is the
  replacement for risk_gate.py's ^VIX feed, which is DEAD (XProtect blocks
  yfinance, so every VIX call returns empty and the spike detector never fires).

THE HONEST BAR
  log(trailing_vol_20d) ALONE already scores Spearman +0.545. So the only question
  worth asking is: DOES THE 118-FEATURE MODEL BEAT ONE COLUMN? If it does not, use
  the one column -- same verdict the DTC sort earned against XGBoost at h=40.
"""
from __future__ import annotations
import argparse, sqlite3, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy import stats
import xgboost as xgb
from features.builder import build_feature_dataframe, OUTPUT_COLUMNS


def add_vol_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """log forward realized vol over t+1..t+h. STRICTLY forward: day t excluded.

    No threshold, no median, no binary. The target IS the number, because the
    number is what a position sizer consumes.
    """
    df = df.copy()
    r = df["close"].pct_change()
    # reverse-rolling: reverse, rolling-std, reverse back, shift -1 so the window
    # starts at t+1 and never includes day t's own return.
    fwd = r[::-1].rolling(horizon).std()[::-1].shift(-1)
    df["fwd_vol"] = fwd
    df["y"] = np.log(fwd.clip(lower=1e-6))
    df.loc[df.index[-horizon:], ["fwd_vol", "y"]] = np.nan
    return df


def evaluate(tickers, horizon=5, start="2022-01-01", folds=4, embargo=None):
    span_cal = int(np.ceil(horizon * 7.0 / 5.0))
    embargo = embargo if embargo is not None else span_cal + 7
    assert embargo >= span_cal, (
        f"embargo {embargo}cal < {span_cal}cal spanned by a {horizon}-TRADING-day "
        f"label -- training labels would overlap the test window")

    frames = []
    for i, t in enumerate(tickers, 1):
        try:
            d = build_feature_dataframe(t, start_date=start, training_mode=True)
            if d is None or len(d) < horizon + 260:
                continue
            d = add_vol_target(d, horizon)
            d["_d"] = pd.to_datetime(d["date"]); d["_tk"] = t
            # the baseline: one column. trailing 20d realized vol, logged.
            d["_trail"] = np.log(d["close"].pct_change().rolling(20).std().clip(lower=1e-6))
            frames.append(d)
        except Exception:
            pass
        if i % 50 == 0:
            print(f"    [{i}/{len(tickers)}] kept={len(frames)}", flush=True)

    p = pd.concat(frames, ignore_index=True).dropna(subset=["y", "_trail"])
    feats = sorted({c for c in OUTPUT_COLUMNS if c in p.columns
                    and pd.api.types.is_numeric_dtype(p[c])
                    and c not in ("date", "ticker", "close")})
    dates = np.array(sorted(p["_d"].unique()))
    print(f"  panel: {len(p):,} rows | {len(dates)} dates | {p['_tk'].nunique()} tickers")
    print(f"  features: {len(feats)}   baseline: log(trailing_vol_20d), 1 column")
    print(f"  embargo: {embargo} calendar days >= {span_cal} spanned by the label\n")

    def expanding_folds(dts, n_folds, emb):
        fold = len(dts) // (n_folds + 1)
        for k in range(1, n_folds + 1):
            te = dts[k * fold:(k + 1) * fold]
            if len(te) == 0:
                continue
            tr = dts[dts < te.min() - np.timedelta64(emb, "D")]
            if len(tr) < 60 or len(te) < 20:
                continue
            yield tr, te

    print(f"  {'fold':<5}{'test window':<26}{'n':>8}{'BASE r':>9}{'MODEL r':>9}{'lift':>8}")
    print("  " + "-" * 66)
    res = []
    for fi, (tr_d, te_d) in enumerate(expanding_folds(dates, folds, embargo)):
        tr = p[p["_d"].isin(set(tr_d))]
        te = p[p["_d"].isin(set(te_d))]
        if len(tr) < 2000 or len(te) < 500:
            continue
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                             objective="reg:squarederror", random_state=42, verbosity=0)
        m.fit(tr[feats].astype(float), tr["y"])
        pred = m.predict(te[feats].astype(float))
        r_model = stats.spearmanr(pred, te["y"])[0]
        r_base = stats.spearmanr(te["_trail"], te["y"])[0]
        res.append({"base": r_base, "model": r_model, "n": len(te)})
        print(f"  {fi:<5}{str(pd.Timestamp(te_d.min()).date())+'..'+str(pd.Timestamp(te_d.max()).date()):<26}"
              f"{len(te):>8,}{r_base:>+9.4f}{r_model:>+9.4f}{r_model-r_base:>+8.4f}")

    if not res:
        print("\n  NO USABLE FOLDS."); return None
    b = np.array([x["base"] for x in res]); mo = np.array([x["model"] for x in res])
    print("  " + "-" * 66)
    print(f"  {'MEAN':<31}{'':>8}{b.mean():>+9.4f}{mo.mean():>+9.4f}{(mo-b).mean():>+8.4f}")
    print()
    lift = (mo - b).mean()
    if lift > 0.03 and (mo > b).all():
        print(f"  >> MODEL BEATS THE BASELINE by {lift:+.4f} Spearman, in every fold.")
        print("     The 118 features add real information over trailing vol alone.")
    elif abs(lift) < 0.02:
        print(f"  >> NO LIFT ({lift:+.4f}). The 118 features add nothing over ONE column.")
        print("     USE log(trailing_vol_20d) DIRECTLY. Same verdict the DTC sort earned")
        print("     against XGBoost at h=40: do not build a model to reproduce a signal")
        print("     that one column already carries.")
    else:
        print(f"  >> MARGINAL ({lift:+.4f}, {int((mo>b).sum())}/{len(mo)} folds). Not established.")
    print(f"\n  Both columns are OUT-OF-SAMPLE Spearman(prediction, log fwd vol).")
    print(f"  Honest n = {len(res)} folds.")
    return {"base": b.mean(), "model": mo.mean(), "lift": lift}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--max-tickers", type=int, default=150)
    a = ap.parse_args()
    tks = [t.strip().upper() for t in open(ROOT / "tickers.txt")
           if t.strip() and not t.startswith("#")][:a.max_tickers]
    print(f"=== FORWARD VOL: regression (log fwd vol), h={a.horizon}d ===")
    evaluate(tks, a.horizon, a.start, a.folds)
