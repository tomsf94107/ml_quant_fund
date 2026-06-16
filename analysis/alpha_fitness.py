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
_MARKET_PREFIXES = ("dxy", "spy", "vix", "oil", "xl", "igv", "tnx", "move",
                    "fear", "greed", "regime", "market", "sp500", "risk_prev")


def _is_market_wide(alpha_name: str) -> int:
    base = alpha_name.split("__")[0].lower()
    return int(any(base.startswith(p) for p in _MARKET_PREFIXES))


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
    return {"n_obs": n, "n_days": n_days, "rank_ic": round(ic, 5),
            "ic_t": round(ic_t, 2), "sharpe": round(sharpe, 4),
            "turnover": round(turn, 4), "fitness": round(fit, 4)}


def score_cross_sectional(panel_dir: Path, db_path: Path, horizon: int) -> pd.DataFrame:
    panel = _load_panel(panel_dir)
    m = _merge_outcomes(panel, db_path, horizon)
    cols = [c for c in panel.columns if c not in ("ticker", "date")]
    rows = []
    for c in cols:
        res = _score_one(m[c], m["actual_return"], m["date"])
        if res:
            res["alpha"] = c
            res["horizon"] = horizon
            res["is_market_wide"] = _is_market_wide(c)
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
        stock = df[df["is_market_wide"] == 0]
        print(f"cross-sectional: {len(df)} alphas scored (horizon {a.horizon}d)")
        print(f"  market-wide (flagged, not stock-pickers): {(df['is_market_wide']==1).sum()}")
        print(f"  STOCK-PICKING alphas: {len(stock)}")
        print(f"    of those, fitness>0:        {(stock['fitness']>0).sum()}")
        print(f"    of those, |t|>3 (HLZ bar):  {(stock['ic_t'].abs()>3).sum()}  <- the defensible survivors")
        print(f"\ntop 12 STOCK-PICKING alphas (market-wide excluded), by fitness:")
        print(stock.head(12)[['alpha','rank_ic','ic_t','sharpe','fitness','n_obs']].to_string(index=False))
        if a.write:
            write_snapshot(df, Path(a.db), "alpha_fitness")
            print(f"\nappended -> alpha_fitness ({date.today().isoformat()})")
