#!/usr/bin/env python3
"""
si_ml_vs_sort_wf40.py -- Does XGBoost@h=40 beat a one-line sort on days_to_cover?

THE QUESTION
  The SI brick is REAL but SLOW. Horizon sweep (validate_si_v2, per-date IC + NW):
      h=1   IC -0.0177  NW-t -1.97   NOT significant
      h=3   IC -0.0235  NW-t -2.68   (fluke -- non-monotonic with h=1 and h=5)
      h=5   IC -0.0194  NW-t -1.97   NOT significant
      h=10  IC -0.0334  NW-t -3.90   REAL
      h=20  IC -0.0368  NW-t -3.59   REAL
      h=40  IC -0.0534  NW-t -4.76   REAL, strongest  <-- the brick lives here

  The production model (h=1/3/5) cannot use it. The brick is traded separately
  (si_live_ledger.csv, long low-DTC, 40d hold). Open question: does ML at the
  brick's OWN horizon add anything over the raw sort?

    BASELINE : rank by days_to_cover, 40d hold -> IC -0.053, IR -0.62, 74% sign
    TEST     : XGBoost @ h=40, all builder features + PIT days_to_cover
    GATE     : beat |IC| 0.053 by a real margin -> ML adds something.

THE TWO TRAPS
  TRAP 1 -- OVERLAP. A 40-trading-day label means two rows one day apart share 39
  of their 40 forward days: labels ~97% identical. Without an embargo >= the label
  span, the model sees its test answers. Effective n collapses from ~20,000
  stock-days to ~500. THIS IS THE BUG BEHIND PEAD's t=-20 AND THE RANKER'S FAKE
  +5.6 SHARPE.
    Subtlety: purged_kfold_indices embargoes in CALENDAR days, but target_40d
    spans 40 TRADING days ~= 56 calendar. An embargo of 40 would LOOK right and
    still leak ~16 trading days. We ASSERT on this rather than trust it.

  TRAP 2 -- LOOK-AHEAD ON THE SIGNAL. settlement_date is when the position was
  MEASURED; FINRA DISSEMINATES ~8 BD later. Keying on settlement_date is 8 BD of
  look-ahead. si_dissemination_lag_test measured the edge survives the real lag
  (IC -0.053 -> -0.044, 92% retained), so we enter at settlement + 8 BD.

WHAT IT DOES
  Standalone. Touches NOTHING in production -- not builder.py, not
  add_forecast_targets (20 modules call it), not walk_forward_batched. Answer the
  question first; wire it only if the answer says to.
"""
import argparse, os, sqlite3, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from analysis.walk_forward import purged_kfold_indices
from features.builder import build_feature_dataframe, OUTPUT_COLUMNS
import xgboost as xgb

# Production XGB params (depth 2, 30 trees, reg_lambda 10). A deeper model would
# be a different experiment -- and would overfit ~500 effective observations.
XGB_PARAMS = dict(
    n_estimators=30, max_depth=2, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, verbosity=0,
)
DISSEMINATION_LAG_BD = 8
CLIP_DTC = 50.0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_si_pit():
    con = sqlite3.connect("short_interest.db")
    si = pd.read_sql("SELECT ticker, settlement_date, days_to_cover FROM short_interest "
                     "WHERE days_to_cover IS NOT NULL", con)
    con.close()
    si["settlement_date"] = pd.to_datetime(si["settlement_date"])
    si = si[si["days_to_cover"] <= CLIP_DTC]
    si["knowable_at"] = si["settlement_date"] + pd.tseries.offsets.BDay(DISSEMINATION_LAG_BD)
    si["ticker"] = si["ticker"].str.upper()
    return si[["ticker","knowable_at","days_to_cover"]].sort_values("knowable_at")

def attach_dtc(df, ticker, si):
    """As-of join: on each date, the most recent SI reading already PUBLISHED.

    build_feature_dataframe returns a RangeIndex with the date in a `date` COLUMN
    holding python datetime.date objects -- not a DatetimeIndex. merge_asof needs
    a real datetime64 key on both sides, so convert explicitly. (Assuming the
    index was dates cost a whole run: every merge raised MergeError and the panel
    silently kept only 8 of 100 tickers.)
    """
    df = df.copy()
    df["_d"] = pd.to_datetime(df["date"])
    s = si[si["ticker"] == ticker]
    if s.empty:
        df["days_to_cover"] = np.nan
        return df
    out = pd.merge_asof(
        df.sort_values("_d"),
        s[["knowable_at", "days_to_cover"]].sort_values("knowable_at"),
        left_on="_d", right_on="knowable_at", direction="backward")
    return out.drop(columns=["knowable_at"], errors="ignore")

def build_panel(tickers, horizon, start):
    si = load_si_pit()
    log(f"  SI: {len(si):,} PIT rows, {si.ticker.nunique()} tickers "
        f"(entry = settlement + {DISSEMINATION_LAG_BD} BD)")
    frames, skipped = [], 0
    for i, t in enumerate(tickers, 1):
        try:
            d = build_feature_dataframe(t, start_date=start, training_mode=True)
            if d is None or len(d) < horizon + 60:
                skipped += 1; continue
            d = attach_dtc(d, t, si)          # sorts by _d
            fwd = d["close"].shift(-horizon) / d["close"] - 1.0
            fwd.iloc[-horizon:] = np.nan
            d["fwd_ret"] = fwd; d["_tk"] = t; d["_date"] = d["_d"]
            frames.append(d)
        except Exception as e:
            skipped += 1
            if i <= 5: log(f"    {t}: {type(e).__name__} {str(e)[:50]}")
        if i % 50 == 0:
            log(f"    [{i}/{len(tickers)}] kept={len(frames)} skipped={skipped}")
    if not frames:
        log("  NO DATA -- abort."); sys.exit(1)
    p = pd.concat(frames, ignore_index=True)
    p = p.dropna(subset=["fwd_ret","days_to_cover"])
    p["target"] = (p["fwd_ret"] > 0).astype(int)
    # OUTPUT_COLUMNS contains `date` and `ticker` -- identifiers, not features.
    # Passing them to XGBoost raised "float() argument must be ... not
    # 'datetime.date'". Take only numeric columns, and ASSERT rather than trust:
    # a silent non-numeric would either crash here or (worse) get label-encoded
    # into a leak.
    cand = [c for c in OUTPUT_COLUMNS if c in p.columns] + ["days_to_cover"]
    drop = {"date", "ticker", "_d", "_tk", "_date", "fwd_ret", "target", "close"}
    feats = sorted({c for c in cand if c not in drop
                    and pd.api.types.is_numeric_dtype(p[c])})
    bad = [c for c in feats if not pd.api.types.is_numeric_dtype(p[c])]
    assert not bad, f"non-numeric features slipped through: {bad}"
    dropped = sorted(set(cand) - set(feats))
    if dropped:
        log(f"  dropped {len(dropped)} non-numeric/identifier cols: {dropped[:6]}")
    log(f"  panel: {len(p):,} rows | {p['_date'].nunique()} dates | "
        f"{p['_tk'].nunique()} tickers | {len(feats)} features")
    return p, feats

def per_date_ic(df, signal_col, ret_col="fwd_ret", min_names=20):
    """Spearman IC WITHIN each date -> a DATE-KEYED Series.

    Returns a Series indexed by date, NOT a bare array. A depth-2 XGBoost emits
    coarse probabilities, so on many dates it has <5 distinct values across ~380
    stocks and the date is skipped: ML landed on 313 dates while the raw sort
    landed on 646. Comparing them by ARRAY POSITION (ml[:n] - so[:n]) subtracts
    DIFFERENT DATES from each other and the bootstrap CI is meaningless. Keying by
    date forces an explicit inner join before any lift is computed.
    """
    out = {}
    for d, g in df.groupby("_date"):
        if len(g) < min_names:
            continue
        if g[signal_col].nunique() < 5:
            continue
        ic = g[signal_col].corr(g[ret_col], method="spearman")
        if pd.notna(ic):
            out[d] = ic
    return pd.Series(out, dtype=float).sort_index()


def strip_beta(df, signal_col, beta_col="beta_60d"):
    """Cross-sectionally residualise a signal against beta, WITHIN each date.

    WHY. The first run gave ML IC +0.20 at h=40 -- 4x anything the literature or
    this fund's own research says is achievable, and the fund's own May finding
    was that "the edge is directional (sector beta), not relative". A model handed
    beta_60d, momentum and volatility, asked to predict 40-day direction, learns
    "high-beta names rise more". In a bull tape that yields a large WITHIN-DATE
    correlation between prob_up and forward return which is pure market beta, not
    stock selection.

    The null control CANNOT catch this: shuffling returns within a date destroys
    the beta exposure too, so the null passes while the "signal" is still beta.

    So: regress the signal on beta per date, keep the residual, re-measure IC.
    What survives is what the model knows BEYOND beta. That is the alpha.
    """
    if beta_col not in df.columns:
        return None
    out = df.copy()
    def _resid(g):
        x = g[beta_col].astype(float).values
        y = g[signal_col].astype(float).values
        if len(g) < 20 or np.nanstd(x) < 1e-9:
            return pd.Series(y, index=g.index)
        x = np.nan_to_num(x, nan=float(np.nanmean(x)))
        b = np.polyfit(x, y, 1)
        return pd.Series(y - np.polyval(b, x), index=g.index)
    out["_resid"] = out.groupby("_date", group_keys=False).apply(_resid)
    return out

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return np.nan
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n-1)+1):
        var += 2.0 * (1.0 - k/(lag+1.0)) * ((e[k:] @ e[:-k]) / n)
    return x.mean()/np.sqrt(var/n) if var > 0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=40, help="TRADING days")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--embargo", type=int, default=None, help="CALENDAR days")
    a = ap.parse_args()

    # THE ASSERTION THAT MAKES THIS TEST HONEST.
    # purged_kfold_indices embargoes in CALENDAR days; the label spans `horizon`
    # TRADING days ~= horizon*7/5 calendar. If embargo < label span, training
    # labels reach INTO the test window and the model sees its own answers. That
    # is invisible in the output -- you just get a great number.
    span_cal = int(np.ceil(a.horizon * 7.0/5.0))
    embargo = a.embargo if a.embargo is not None else span_cal + 14
    assert embargo >= span_cal, (
        f"EMBARGO TOO SHORT: {embargo} calendar days < {span_cal} spanned by a "
        f"{a.horizon}-TRADING-day label. This is the PEAD t=-20 bug.")

    tickers = [t.strip().upper() for t in Path(a.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print("="*78)
    print(f"  ML vs RAW DTC SORT  --  h={a.horizon} trading days")
    print("="*78)
    log(f"  embargo  : {embargo} calendar days (label spans ~{span_cal}) -- ASSERTED")
    log(f"  folds    : {a.folds} (expanding window, train strictly in the past)")
    log(f"  baseline : raw days_to_cover sort -> IC -0.053, IR -0.62")
    log(f"  tickers  : {len(tickers)}")
    print()
    p, feats = build_panel(tickers, a.horizon, a.start)
    dates = pd.Series(sorted(p["_date"].unique()))
    nw_lag = max(1, a.horizon // 5)
    print()
    print(f"  {'fold':<5}{'test window':<26}{'n':>7}{'ML IC':>9}{'SORT IC':>10}"
          f"{'BETA IC':>9}{'ML-noBeta':>11}")
    print("  " + "-"*78)
    ml_all, sort_all, null_all, beta_all, mlstrip_all = [], [], [], [], []
    for fi,(tr_i,te_i) in enumerate(purged_kfold_indices(dates, n_folds=a.folds, embargo=embargo)):
        tr = p[p["_date"].isin(set(dates.iloc[tr_i]))]
        te = p[p["_date"].isin(set(dates.iloc[te_i]))]
        if len(tr) < 500 or len(te) < 200: continue
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(tr[feats].astype(float), tr["target"])
        te = te.copy()
        te["_ml"]   = m.predict_proba(te[feats].astype(float))[:,1]
        te["_sort"] = -te["days_to_cover"]     # low DTC = bullish (validated sign)
        ic_ml = per_date_ic(te, "_ml")
        ic_so = per_date_ic(te, "_sort")

        # BETA BASELINE: what does beta ALONE score on these same dates? If it is
        # ~= the ML IC, the model's "edge" is market beta, not stock selection.
        te["_beta"] = te["beta_60d"] if "beta_60d" in te.columns else np.nan
        ic_be = (per_date_ic(te, "_beta") if te["_beta"].notna().any()
                 else pd.Series(dtype=float))

        # BETA-STRIPPED ML: residual of prob_up after removing beta, per date.
        tb = strip_beta(te, "_ml")
        ic_ms = per_date_ic(tb, "_resid") if tb is not None else pd.Series(dtype=float)

        tn = te.copy()
        tn["fwd_ret"] = tn.groupby("_date")["fwd_ret"].transform(
            lambda s: np.random.permutation(s.values))
        ic_nu = per_date_ic(tn, "_ml")

        if len(ic_ml)==0 or len(ic_so)==0: continue
        ml_all.append(ic_ml); sort_all.append(ic_so); null_all.append(ic_nu)
        beta_all.append(ic_be); mlstrip_all.append(ic_ms)
        td = set(dates.iloc[te_i]); lo,hi = min(td), max(td)
        print(f"  {fi:<5}{str(lo.date())+'..'+str(hi.date()):<26}{len(te):>7,}"
              f"{ic_ml.mean():>+9.4f}{ic_so.mean():>+10.4f}"
              f"{(ic_be.mean() if len(ic_be) else np.nan):>+9.4f}"
              f"{(ic_ms.mean() if len(ic_ms) else np.nan):>+11.4f}")
    if not ml_all:
        print("\n  NO USABLE FOLDS."); return

    ml = pd.concat(ml_all).sort_index()
    so = pd.concat(sort_all).sort_index()
    nu = pd.concat(null_all).sort_index()
    be = pd.concat(beta_all).sort_index() if any(len(x) for x in beta_all) else pd.Series(dtype=float)
    ms = pd.concat(mlstrip_all).sort_index() if any(len(x) for x in mlstrip_all) else pd.Series(dtype=float)

    print(); print("="*78); print("  VERDICT"); print("="*78)
    print(f"  {'':26s}{'mean IC':>10}{'IC IR':>9}{'NW-t':>9}{'+sign':>8}{'dates':>7}")
    rows = [("ML (xgb h=%d)" % a.horizon, ml), ("SORT (raw DTC)", so)]
    if len(be): rows.append(("BETA ALONE (beta_60d)", be))
    if len(ms): rows.append(("ML minus BETA (alpha)", ms))
    rows.append(("NULL (shuffled)", nu))
    for nm, v in rows:
        ir = v.mean()/v.std() if v.std() > 0 else np.nan
        print(f"  {nm:26s}{v.mean():>+10.4f}{ir:>+9.3f}{nw_t(v.values, nw_lag):>+9.2f}"
              f"{100*(v>0).mean():>7.0f}%{len(v):>7d}")

    # DATE-PAIRED LIFT. The first version did ml[:n] - so[:n], subtracting
    # DIFFERENT DATES by array position. An inner join is the only honest pairing.
    j = ml.index.intersection(so.index)
    print()
    print(f"  paired on {len(j)} common dates (ML had {len(ml)}, SORT had {len(so)})")
    if len(j) < 20:
        print("  >> TOO FEW PAIRED DATES. Inconclusive."); return
    d = (ml.loc[j] - so.loc[j]).values
    boot = np.array([np.random.choice(d, len(d), replace=True).mean() for _ in range(2000)])
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
    print(f"  LIFT (ML - SORT) : {d.mean():+.4f}   95% CI [{lo_ci:+.4f}, {hi_ci:+.4f}]")
    print()

    if abs(nu.mean()) > 0.01:
        print("  >> NULL CONTROL FAILED (shuffled IC = %+.4f). Pipeline broken." % nu.mean())
        return

    beta_mean  = be.mean() if len(be) else float("nan")
    alpha_mean = ms.mean() if len(ms) else float("nan")

    if len(be) and len(ms):
        if abs(alpha_mean) < 0.02:
            print("  >> IT WAS BETA. Beta alone scores IC %+.4f; ML scores %+.4f; but after"
                  % (beta_mean, ml.mean()))
            print("     stripping beta cross-sectionally the ML residual collapses to %+.4f."
                  % alpha_mean)
            print("     The model is not picking stocks -- it is loading on market beta and")
            print("     being paid for it in a rising tape. That is not alpha; it inverts in")
            print("     a drawdown. Trade the SORT (IC %+.4f, IR %+.3f): stock-specific by"
                  % (so.mean(), so.mean()/so.std() if so.std() > 0 else float("nan")))
            print("     construction, and already running in si_live_ledger.csv.")
            return
        print("  >> ML retains IC %+.4f AFTER beta-stripping (beta alone: %+.4f)."
              % (alpha_mean, beta_mean))
        print("     Some of the edge is genuinely cross-sectional.")

    if lo_ci > 0:
        print("  >> ML BEATS THE SORT on paired dates. Lift CI excludes zero.")
    elif hi_ci < 0:
        print("  >> THE SORT BEATS ML.")
    else:
        print("  >> NO DIFFERENCE (CI spans zero). ML adds nothing over the raw sort.")
    print()
    print(f"  Honest n = {len(j)} PAIRED DATES (not {len(p):,} stock-date rows).")
    print(f"  Embargo {embargo} calendar days >= {span_cal} spanned by the label.")


if __name__ == "__main__":
    main()
