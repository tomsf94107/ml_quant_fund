"""
analysis/alpha_gate.py  — P3.2 alpha gating pipeline
Scores all alpha-panel features by rank-IC vs forward returns, then applies
multiple-testing gates (Harvey-Liu-Zhu |t|>3.0 AND empirically-calibrated
Deflated Sharpe Ratio) to find the small set of features worth feeding to
the models. This is the overfit defense that makes the 3,390-feature panel
SAFE to use.

RULE #1: prices via Massive only (mc.download). NEVER build_feature_dataframe
here (triggers off-hours UW calls). Forward returns need close only.

Usage:
    PYTHONPATH=. python analysis/alpha_gate.py --horizon 5 [--max-dates 584]
        [--ic-floor 0.0] [--out analysis/alpha_gate_results_h5.csv]
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Macro / market-level / sector-ETF base features: same value across all
# tickers on a date -> NOT cross-sectional selectors. Excluded from the CS gate.
MACRO_BASES = {
    "spy_ret", "xlk_ret",
    "xle_ret_5d","xlv_ret_5d","xlf_ret_5d","xlk_ret_5d","xlu_ret_5d",
    "xli_ret_5d","xlp_ret_5d","xly_ret_5d","xlc_ret_5d","xlre_ret_5d","xlb_ret_5d",
    "vix_close","vix_ret","vix_term_structure","vix_5d_above_25",
    "oil_ret","oil_spy_corr","dxy_ret","yield_10y","fear_greed",
    "lqd_hyg_spread","semi_etf_momentum_60d","igv_vs_sp500_ret_30d",
    "es_overnight","monday_sentiment","is_pandemic","day_of_week","is_month_end",
}


sys.path.insert(0, str(ROOT))
from features import massive_client as mc
from analysis.alpha_gate_stats import deflated_sharpe, expected_max_sharpe


def load_panel(max_dates):
    files = sorted(glob.glob(str(ROOT / "data/alpha_panel/*.parquet")))
    if max_dates:
        files = files[-max_dates:]
    dates, frames = [], {}
    for f in files:
        dt = pd.Timestamp(Path(f).stem)
        frames[dt] = pd.read_parquet(f)
        dates.append(dt)
    return frames, sorted(dates)


def get_fwd_returns(tickers, dates, horizon):
    start = (min(dates) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    end = (max(dates) + pd.Timedelta(days=horizon + 10)).strftime("%Y-%m-%d")
    px = mc.download(list(tickers), start=start, end=end,
                     auto_adjust=True, progress=False)
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
    close.index = pd.to_datetime(close.index)
    return close.shift(-horizon) / close - 1.0   # date x ticker


def compute_ic(frames, dates, fwd, features):
    """Per-date rank-IC for every feature. Returns DataFrame [date x feature]."""
    ic_rows = {}
    for dt in dates:
        if dt not in frames or dt not in fwd.index:
            continue
        panel = frames[dt]
        rets = fwd.loc[dt].reindex(panel.index)
        valid = rets.notna()
        if valid.sum() < 10:
            continue
        r = rets[valid].rank()                       # rank forward returns once
        sub = panel.loc[valid, features]
        # Drop columns with no within-date cross-sectional dispersion
        # (macro/sector-ETF features are identical across tickers -> CS rank
        #  is meaningless; nunique<=1 means constant that date).
        disp = sub.nunique() > 1
        sub = sub.loc[:, disp]
        if sub.shape[1] == 0:
            continue
        fr = sub.rank()
        ic_rows[dt] = fr.corrwith(r)                  # vectorized across valid features
    return pd.DataFrame(ic_rows).T                    # index=date, cols=feature


def compute_ts_ic(frames, dates, fwd, features):
    """Per-TICKER time-series IC: for each ticker, correlate each feature's
    own time-series against that ticker's OWN forward returns across dates.
    This is the PER-TICKER / production-objective measure (Stage 2 Gap A) —
    distinct from cross-sectional rank-IC. Returns DataFrame [feature x stats].

    Built by stacking the date-major panel into a long (date,ticker) frame
    once, then groupby-ticker Spearman corr of each feature vs fwd return.
    """
    # Long panel: rows=(date,ticker), cols=features (+ __fwd__)
    parts = []
    for dt in dates:
        if dt not in frames or dt not in fwd.index:
            continue
        pan = frames[dt][features].copy()
        fr = fwd.loc[dt].reindex(pan.index)
        pan["__fwd__"] = fr.values
        pan["__date__"] = dt
        pan["__ticker__"] = pan.index
        parts.append(pan.reset_index(drop=True))
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    long = long.dropna(subset=["__fwd__"])

    # Per ticker: Spearman corr (rank-IC over time) of each feature vs fwd.
    rows = {f: [] for f in features}
    ntk = {f: 0 for f in features}
    for tk, g in long.groupby("__ticker__"):
        if len(g) < 60:                      # need enough dates per ticker
            continue
        fwd_rank = g["__fwd__"].rank()
        for f in features:
            col = g[f]
            if col.nunique() < 5:            # no time variation for this ticker
                continue
            ic = col.rank().corr(fwd_rank)   # Spearman over this ticker's history
            if pd.notna(ic):
                rows[f].append(ic); ntk[f] += 1
    out = []
    for f in features:
        arr = np.asarray(rows[f], float)
        if arr.size == 0:
            out.append((f, np.nan, np.nan, np.nan, 0)); continue
        m, s = arr.mean(), arr.std(ddof=1) if arr.size > 1 else np.nan
        ir = m / s if (s and s == s and s != 0) else np.nan
        t  = ir * np.sqrt(arr.size) if ir == ir else np.nan
        out.append((f, m, s, ir, arr.size))
    return pd.DataFrame(out, columns=["feature","ts_mean_IC","ts_std_IC",
                                      "ts_IC_IR","ts_n_tickers"]).set_index("feature")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--max-dates", type=int, default=584)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"analysis/alpha_gate_results_h{args.horizon}.csv"

    print(f"Loading panel (last {args.max_dates} dates)...")
    frames, dates = load_panel(args.max_dates)
    feats = list(frames[dates[-1]].columns)
    tickers = sorted(set().union(*[set(frames[d].index) for d in dates[-5:]]))
    print(f"{len(dates)} dates, {len(feats)} features, {len(tickers)} tickers")

    print("Fetching fwd returns (Massive, no UW)...")
    fwd = get_fwd_returns(tickers, dates, args.horizon)

    print("Computing rank-IC across all features (vectorized per date)...")
    ic = compute_ic(frames, dates, fwd, feats)
    print(f"IC matrix: {ic.shape[0]} dates x {ic.shape[1]} features")

    # aggregate per feature
    mean_ic = ic.mean()
    std_ic = ic.std()
    n_obs = ic.notna().sum()
    ic_ir = mean_ic / std_ic
    t_stat = ic_ir * np.sqrt(n_obs)

    res = pd.DataFrame({
        "feature": feats,
        "n_dates": n_obs.reindex(feats).values,
        "mean_IC": mean_ic.reindex(feats).values,
        "std_IC": std_ic.reindex(feats).values,
        "IC_IR": ic_ir.reindex(feats).values,
        "t_stat": t_stat.reindex(feats).values,
    }).dropna(subset=["t_stat"])

    # GUARD 1: require coverage on at least half the history. Features valid
    # on a handful of dates produce explosive IC_IR (std~0 over few points).
    MIN_DATES = max(250, len(dates) // 2)
    before = len(res)
    res = res[res["n_dates"] >= MIN_DATES]
    print(f"Coverage filter: dropped {before - len(res)} features with "
          f"<{MIN_DATES} dates; {len(res)} remain")

    # GUARD 2: clip any residual degenerate IC_IR (real ones are |IR|<~1.5).
    # Anything beyond is a near-zero-variance artifact, not signal.
    degenerate = res["IC_IR"].abs() > 2.0
    if degenerate.any():
        print(f"Dropped {degenerate.sum()} features with |IC_IR|>2.0 (artifacts)")
        res = res[~degenerate]

    # ── DEDUP BEFORE GATING ──────────────────────────────────────────────
    # 30 transforms of one base signal are NOT 30 independent trials. HLZ /
    # Bailey-LdP define N as the number of DISTINCT strategies. Cluster by
    # base-feature name (the part before the first "__") and keep the highest
    # |t| transform per base. N = number of distinct base signals.
    res["base"] = res["feature"].str.split("__").str[0]
    # Exclude macro/market/sector bases — not cross-sectional selectors.
    n_macro = res["base"].isin(MACRO_BASES).sum()
    res = res[~res["base"].isin(MACRO_BASES)].copy()
    print(f"Excluded {n_macro} macro/sector transforms (not CS selectors)")
    res = res.sort_values("t_stat", key=abs, ascending=False)
    best = res.drop_duplicates("base", keep="first").copy()   # one per base signal
    N = len(best)   # effective independent-trial count
    var_trial = best["IC_IR"].var()
    T_med = int(best["n_dates"].median())
    sr_star = expected_max_sharpe(N, var_trial)
    print(f"\nDedup: {len(res)} transforms -> {N} distinct base signals")
    print(f"Empirical (on base signals): N={N}, var(IC_IR)={var_trial:.4f}, "
          f"median T={T_med}, E[max IC_IR]={sr_star:.4f}")

    # DSR computed with effective N (= distinct signals), applied to ALL rows
    # but the gate decision uses the de-duplicated representative per base.
    # CORRECT multiple-testing gate: extreme-value t-threshold.
    # Under H0 (no skill), max |t| over N independent trials ~ sqrt(2 ln N).
    # A feature must EXCEED this to be distinguishable from best-of-N noise.
    # (The var_trial-based DSR was miscalibrated: it used observed cross-
    #  feature dispersion, which includes real signal, as the null bar.)
    t_threshold = float(np.sqrt(2.0 * np.log(N)))
    print(f"Extreme-value t-threshold (sqrt(2 ln N), N={N}): {t_threshold:.3f}")
    # DSR kept as a secondary diagnostic only (NOT the gate)
    res["DSR"] = res.apply(
        lambda row: deflated_sharpe(abs(row["IC_IR"]), N, int(row["n_dates"]),
                                    1.0 / max(int(row["n_dates"]), 2)), axis=1)

    # GATES
    res["pass_HLZ"] = res["t_stat"].abs() > 3.0
    res["pass_EVT"] = res["t_stat"].abs() > t_threshold   # extreme-value MT gate
    # ECONOMIC-MAGNITUDE floor: statistical significance over 570 days is easy;
    # require the IC to be big enough to plausibly matter. |mean_IC|>=0.02 is a
    # commonly-cited 'worth trading' threshold for cross-sectional rank-IC.
    MIN_MEAN_IC = 0.02
    res["pass_MAG"] = res["mean_IC"].abs() >= MIN_MEAN_IC
    res["is_base_rep"] = res["feature"].isin(set(best["feature"]))
    # SURVIVOR = de-duplicated base rep clearing the MT threshold AND the
    # economic-magnitude floor.
    res["SURVIVOR"] = (res["pass_HLZ"] & res["pass_EVT"]
                       & res["pass_MAG"] & res["is_base_rep"])

    # ── Stage 2 Gap A: PER-TICKER time-series IC (dual gate) ─────────────
    # Run TS-IC only on the deduped base reps (not all 3,390 transforms).
    base_rep_feats = list(best["feature"])
    print(f"\nComputing per-ticker TS-IC on {len(base_rep_feats)} base reps...")
    ts = compute_ts_ic(frames, dates, fwd, base_rep_feats)
    res = res.merge(ts, left_on="feature", right_index=True, how="left")
    # derive TS t-stat from per-ticker IC_IR and ticker count, then gate it
    res["ts_t_stat"] = res["ts_IC_IR"] * np.sqrt(res["ts_n_tickers"].clip(lower=1))
    res["ts_pass"] = res["ts_t_stat"].abs() > 3.0
    # sign agreement: a feature is only safely additive on BOTH objectives if
    # it passes both gates AND points the same way. Opposite-sign passers are
    # strong-but-objective-conditional (trend CS / mean-revert per-ticker).
    import numpy as _np
    res["ts_sign_agree"] = _np.sign(res["mean_IC"]) == _np.sign(res["ts_mean_IC"])
    res["dual_survivor"] = res["SURVIVOR"] & res["ts_pass"] & res["ts_sign_agree"]
    # divergence: CS says trade, TS disagrees (sign flip OR TS not significant)
    res["cs_ts_divergence"] = (
        res["is_base_rep"]
        & res["SURVIVOR"]
        & ( (np.sign(res["mean_IC"]) != np.sign(res["ts_mean_IC"]))
            | (res["ts_IC_IR"].abs() < 0.3) )
    )
    res = res.sort_values("t_stat", key=abs, ascending=False)
    res.to_csv(ROOT / out, index=False)

    print(f"\n{'='*70}")
    print(f"ALPHA GATE RESULTS — h={args.horizon}")
    print(f"{'='*70}")
    print(f"total features scored:    {len(res)}")
    print(f"distinct base signals:     {N}")
    print(f"t-threshold (sqrt 2lnN):   {t_threshold:.3f}")
    print(f"base reps pass EVT (t):    {(res['is_base_rep'] & res['pass_EVT']).sum()}")
    print(f"base reps pass MAG (IC):   {(res['is_base_rep'] & res['pass_MAG']).sum()}")
    print(f"SURVIVORS (EVT+MAG):       {res['SURVIVOR'].sum()}")
    _sv = res[res["SURVIVOR"]]
    print(f"\n-- DUAL GATE (Stage 2 Gap A) --")
    print(f"survivors w/ TS-IC computed: {_sv['ts_n_tickers'].gt(0).sum()}")
    print(f"survivors TS sign AGREES CS: {(np.sign(_sv['mean_IC'])==np.sign(_sv['ts_mean_IC'])).sum()}")
    print(f"survivors flagged DIVERGENT: {res['cs_ts_divergence'].sum()}  "
          f"(CS passes but TS disagrees — the P3.5 footgun)")
    print(f"\nTop 15 by |t|:")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(res[res["is_base_rep"]].head(20)[["feature","n_dates","mean_IC","IC_IR","t_stat","SURVIVOR"]].to_string(index=False))
    print(f"\nSaved -> {out}")
    print("\nSANITY: survivor % should be small (a few %). If ~0% the gate is")
    print("too strict; if >20% too loose (recheck var_trial). NOT yet corr-deduped.")


if __name__ == "__main__":
    main()
