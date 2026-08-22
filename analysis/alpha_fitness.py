"""
analysis/alpha_fitness.py — score the panel alphas + decay tracking.

Modes:
  (default) cross-sectional fitness, universe-wide, per alpha. WorldQuant frame.
  --by-ticker  time-series fitness PER TICKER (noisy; research view; small n each).
Filters (cross-sectional mode):
  - market-wide alphas (constant across tickers per day: dxy/spy/vix/sector ETFs)
    are FLAGGED is_market_wide=1 — they tilt the book, they don't rank stocks.
  - t-stat gate: rank-IC t-stat reported; survivors need |t|>3 (Harvey-Liu-Zhu
    multiple-testing bar) given thousands of candidates, NOT just IC>0.02.
Decay: writes dated snapshots (append) -> --decay reports fitness slope over time.

  python -m analysis.alpha_fitness --panel-dir data/alpha_panel --db accuracy.db --horizon 1 --write
  python -m analysis.alpha_fitness --by-ticker --horizon 1
  python -m analysis.alpha_fitness --decay
"""
from __future__ import annotations
import argparse, sqlite3, glob, os, warnings
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

TRADING_DAYS = 252
MIN_OBS = 30
TURNOVER_FLOOR = 0.125
_COST = float(os.environ.get("ML_QUANT_COST_BPS", "10.0")) / 10_000.0

# Prefixes of base features that are MARKET-WIDE (same value across all tickers
# on a given day): they cannot rank the cross-section. Flagged, not scored as
# stock-pickers. Derived from the feature names seen in the panel.
# Market-wide / drop classification is DATA-DRIVEN (analysis/detect_mw.py),
# not a guessed prefix list. An alpha is market_wide if its values are constant
# across tickers per date (<=2 distinct, measured over sampled dates); drop if
# near-all-NaN; else per_ticker. Verified by detect_mw._selftest against ground
# truth. Replaces the old hand-typed _MARKET_PREFIXES (caught ~190 of 1415).
from analysis.detect_mw import classify_bases, classify_alpha

_BASE_CLASSES = None

def _get_base_classes():
    global _BASE_CLASSES
    if _BASE_CLASSES is None:
        _BASE_CLASSES = classify_bases()  # base-level, verified by detect_mw._selftest
    return _BASE_CLASSES


def _load_panel(panel_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(panel_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquets in {panel_dir}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df.index.name = "ticker"
        df = df.reset_index()
        df["date"] = pd.Timestamp(Path(f).stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _merge_outcomes(panel: pd.DataFrame, db_path: Path, horizon: int) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    out = pd.read_sql(
        "SELECT ticker, prediction_date AS date, actual_return "
        "FROM outcomes WHERE horizon=?", con, params=(horizon,))
    con.close()
    out["date"] = pd.to_datetime(out["date"])
    m = panel.merge(out, on=["ticker", "date"], how="inner")
    if m.empty:
        raise SystemExit("no overlap panel<->outcomes — check horizon/dates")
    return m



def _decile_mono(a, r, dates, n_dec: int = 10):
    """Spearman(decile index, decile mean return), computed PER DATE then averaged.

    Separates tradeable IC from mid-book IC. Equal-weight per date: pooling
    stock-date rows overweights dates with more names and has twice produced a
    spurious result in this system (SI brick t=-20; decile spread +0.96% vs the
    correct -0.10%). Returns None when there are too few usable dates.
    """
    try:
        df = pd.DataFrame({"a": np.asarray(a, dtype=float),
                           "r": np.asarray(r, dtype=float),
                           "d": np.asarray(dates)}).dropna()
        if df.empty:
            return None
        # TIE GUARD (2026-08-22). Deciles are only meaningful if the signal can
        # actually separate names. is_squeeze_setup has an 86.5% modal share, so
        # a stable sort orders 8 of 10 deciles by ORIGINAL ROW POSITION
        # (alphabetical by ticker after _load_panel) -- not by signal. It scored
        # mono +0.94 on what is effectively a 2-point ladder, and appeared at the
        # top of the survivor list across every transform. ma5_above_ma20 is 62%.
        # Binary/flag features cannot be decile-ranked; say so rather than
        # returning a confident number computed off ties.
        _modal = df["a"].value_counts(normalize=True).iloc[0]
        if _modal > 0.30:
            return None
        sums = np.zeros(n_dec)
        cnts = np.zeros(n_dec)
        used = 0
        for _d, g in df.groupby("d", sort=False):
            n = len(g)
            if n < n_dec * 3:            # need ~3 names per decile to be meaningful
                continue
            g = g.sort_values("a", kind="mergesort")
            idx = np.minimum((np.arange(n) * n_dec) // n, n_dec - 1)
            rv = g["r"].to_numpy()
            for k in range(n_dec):
                m = idx == k
                if m.any():
                    sums[k] += rv[m].mean()
                    cnts[k] += 1
            used += 1
        if used < 10 or (cnts == 0).any():
            return None
        means = sums / cnts
        rk_i = pd.Series(np.arange(1, n_dec + 1)).rank()
        rk_m = pd.Series(means).rank()
        sd_i, sd_m = rk_i.std(), rk_m.std()
        if not sd_i or not sd_m:
            return None
        return round(float(np.corrcoef(rk_i, rk_m)[0, 1]), 4)
    except Exception as _e:
        # was a bare except returning None -- indistinguishable from 'ran
        # and found nothing'. That is the exact failure this column exists
        # to expose, reintroduced inside the fix for it. 2026-08-22.
        import os as _os
        if _os.environ.get('ML_QUANT_MONO_DEBUG'):
            raise
        print(f'  [warn] _decile_mono failed: {type(_e).__name__}: {_e}')
        return None

def _score_one(a: pd.Series, r: pd.Series, dates: pd.Series) -> dict | None:
    """Daily-IC IR method (Grinold): compute cross-sectional rank-IC PER DATE,
    then t-stat the time series of daily ICs. This treats each DAY as one obs
    (~n_days, independent-ish) instead of every (ticker,date) row (autocorrelated,
    massively inflates t). ic_t = mean(IC)/std(IC) * sqrt(n_days)."""
    msk = (~a.isna()) & (~r.isna())
    n = int(msk.sum())
    if n < MIN_OBS:
        return None
    a, r, d = a[msk], r[msk], dates[msk]
    if a.std() == 0 or a.nunique() < 3:
        return None
    # per-date cross-sectional rank-IC
    tmp = pd.DataFrame({"a": a, "r": r, "d": d})
    daily = tmp.groupby("d").apply(
        lambda g: g["a"].rank().corr(g["r"].rank()) if len(g) >= 5 and g["a"].nunique() >= 3 else np.nan
    ).dropna()
    if len(daily) < 5:
        return None
    ic = daily.mean()
    ic_sd = daily.std()
    n_days = len(daily)
    ic_t = (ic / ic_sd * np.sqrt(n_days)) if ic_sd and ic_sd > 0 else 0.0
    if np.isnan(ic) or np.isnan(ic_t):
        return None
    # fitness from pooled position/return (cost-aware), unchanged
    pos = ((a - a.mean()) / a.std()).clip(-1, 1)
    rp = pos * r
    gross = rp.mean() * TRADING_DAYS
    turn = pos.diff().abs().mean()
    turn = max(turn if not np.isnan(turn) else 1.0, TURNOVER_FLOOR)
    net = gross - turn * _COST * TRADING_DAYS
    vol = rp.std() * np.sqrt(TRADING_DAYS)
    sharpe = net / vol if vol and vol > 0 else 0.0
    fit = np.sqrt(abs(net) / max(turn, TURNOVER_FLOOR)) * sharpe
    # DECILE MONOTONICITY (added 2026-08-21): rank_ic scores the FULL
    # cross-section, sharpe scores the EXTREMES. When they disagree in sign this
    # is the metric that says which. Diagnostic only -- no existing field moves.
    mono = _decile_mono(a, r, d)
    return {"n_obs": n, "n_days": n_days, "rank_ic": round(ic, 5),
            "ic_t": round(ic_t, 2), "sharpe": round(sharpe, 4),
            "turnover": round(turn, 4), "fitness": round(fit, 4),
            "mono": mono}


def score_cross_sectional(panel_dir: Path, db_path: Path, horizon: int) -> pd.DataFrame:
    panel = _load_panel(panel_dir)
    m = _merge_outcomes(panel, db_path, horizon)
    base_classes = _get_base_classes()
    cols = [c for c in panel.columns if c not in ("ticker", "date")]
    rows = []
    for c in cols:
        res = _score_one(m[c], m["actual_return"], m["date"])
        if res:
            res["alpha"] = c
            res["horizon"] = horizon
            res["is_market_wide"] = int(classify_alpha(c, base_classes) == "market_wide")
            rows.append(res)
    df = pd.DataFrame(rows)
    return df.sort_values("fitness", ascending=False)


def score_by_ticker(panel_dir: Path, db_path: Path, horizon: int) -> pd.DataFrame:
    """Time-series fitness PER TICKER: for each ticker, which alphas predict ITS
    own forward return over ITS history. Noisy (small n/ticker) — research view."""
    panel = _load_panel(panel_dir)
    m = _merge_outcomes(panel, db_path, horizon)
    cols = [c for c in panel.columns if c not in ("ticker", "date")]
    rows = []
    for tk, g in m.groupby("ticker"):
        if len(g) < MIN_OBS:
            continue
        for c in cols:
            a, r = g[c], g["actual_return"]
            msk = (~a.isna()) & (~r.isna())
            if msk.sum() < MIN_OBS or a[msk].std() == 0:
                continue
            ic = a[msk].rank().corr(r[msk].rank())
            if np.isnan(ic):
                continue
            n = int(msk.sum())
            t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic * ic, 1e-9))
            rows.append({"ticker": tk, "alpha": c, "horizon": horizon,
                         "n_obs": n, "rank_ic": round(ic, 5), "ic_t": round(t, 2)})
    df = pd.DataFrame(rows)
    return df.sort_values("ic_t", ascending=False)


def write_snapshot(df: pd.DataFrame, db_path: Path, table: str):
    df = df.copy()
    df["scored_date"] = date.today().isoformat()
    con = sqlite3.connect(db_path)
    df.to_sql(table, con, if_exists="append", index=False)
    con.close()


def report_deduped(db_path: Path, horizon: int = None, table: str = "alpha_fitness",
                   scored_date: str = None, market_wide: int = 0, min_ic_t: float = 0.0):
    """Read-side dedup for ranking readability. Collapses all transforms of the
    same BASE feature (base__transform) to one row = the transform with the
    highest |ic_t|. Snapshot features (pc_ratio_snap etc.) generate ~17 near-
    identical transforms that otherwise flood the top of the ranking with copies
    of one signal. This is a REPORT lens only — it does NOT alter the panel,
    the model inputs, or the stored per-transform fitness. Fix Jul 1 2026.

    NOTE: name-based collapse assumes same-base transforms are redundant. True
    for snapshot/slow features; for genuinely varying bases, different transforms
    may carry distinct signal — so use this for reading rankings, not for culling
    features from the model."""
    con = sqlite3.connect(db_path)
    try:
        q = f"SELECT * FROM {table} WHERE 1=1"
        params = []
        if scored_date is None:
            scored_date = con.execute(f"SELECT MAX(scored_date) FROM {table}").fetchone()[0]
        q += " AND scored_date=?"; params.append(scored_date)
        if horizon is not None:
            q += " AND horizon=?"; params.append(horizon)
        if market_wide is not None:
            q += " AND is_market_wide=?"; params.append(market_wide)
        df = pd.read_sql(q, con, params=params)
    finally:
        con.close()
    if df.empty:
        return df
    df = df[df["ic_t"].abs() >= min_ic_t].copy()
    if df.empty:
        return df
    df["base"] = df["alpha"].str.split("__").str[0]
    df["abs_ic_t"] = df["ic_t"].abs()
    # keep, per (base, horizon), the single transform with max |ic_t|
    idx = df.groupby(["base", "horizon"])["abs_ic_t"].idxmax()
    out = df.loc[idx].drop(columns=["abs_ic_t"]).sort_values("ic_t", key=lambda x: x.abs(), ascending=False)
    return out.reset_index(drop=True)


def report_decay(db_path: Path, table: str = "alpha_fitness"):
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", con)
    except Exception:
        con.close(); raise SystemExit(f"no {table} table yet")
    con.close()
    dates = sorted(df["scored_date"].unique())
    print(f"{table} snapshots: {len(dates)} dates {dates}")
    if len(dates) < 2:
        print("DECAY: need >=2 snapshots. Have 1. Re-run scoring later.")
        return
    piv = df.pivot_table(index="alpha", columns="scored_date", values="fitness")
    d = (piv[dates[-1]] - piv[dates[0]]).rename("change").sort_values()
    print(f"\nDECAY {dates[0]} -> {dates[-1]}:  decayed {(d<0).sum()}  improved {(d>0).sum()}")
    print(f"worst 5:\n{d.head().to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="data/alpha_panel")
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--by-ticker", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--decay", action="store_true")
    a = ap.parse_args()
    if a.decay:
        report_decay(Path(a.db)); raise SystemExit

    if a.by_ticker:
        df = score_by_ticker(Path(a.panel_dir), Path(a.db), a.horizon)
        print(f"per-ticker: {len(df)} (ticker,alpha) pairs scored (horizon {a.horizon}d)")
        strong = df[df["ic_t"].abs() > 3]
        print(f"  |t|>3 (HLZ bar): {len(strong)} pairs across {strong['ticker'].nunique()} tickers")
        print(f"\ntop 15 per-ticker by |t|:\n{df.head(15).to_string(index=False)}")
        if a.write:
            write_snapshot(df, Path(a.db), "alpha_fitness_by_ticker")
            print(f"\nappended -> alpha_fitness_by_ticker ({date.today().isoformat()})")
    else:
        df = score_cross_sectional(Path(a.panel_dir), Path(a.db), a.horizon)
        stock = df[df["is_market_wide"] == 0].copy()
        sig = stock[stock["ic_t"].abs() > 3].copy()
        sig["base"] = sig["alpha"].str.split("__").str[0]
        n_bases = sig["base"].nunique()
        print(f"cross-sectional: {len(df)} alphas scored (horizon {a.horizon}d)")
        print(f"  market-wide (flagged): {(df['is_market_wide']==1).sum()}   stock-picking: {len(stock)}")
        print(f"  survivors |t|>3: {len(sig)} transform-copies = {n_bases} DISTINCT base features")
        print(f"  (transform-copies inflate the count; distinct bases is the real signal count)")
        # rank by |t| (significance), NOT fitness — fitness buries real signals (e.g. negative-fitness pc_ratio)
        sig_sorted = sig.reindex(sig["ic_t"].abs().sort_values(ascending=False).index)
        print(f"\nDISTINCT survivor bases by best |t| (one row per base):")
        best = sig_sorted.drop_duplicates("base")
        print(best.head(20)[['base','rank_ic','ic_t','sharpe','n_obs']].to_string(index=False))
        if a.write:
            write_snapshot(df, Path(a.db), "alpha_fitness")
            print(f"\nappended -> alpha_fitness ({date.today().isoformat()})")
