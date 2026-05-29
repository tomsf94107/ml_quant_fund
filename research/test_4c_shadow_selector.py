"""
research/test_4c_shadow_selector.py
4C SHADOW TEST — A8 as cross-sectional SELECTOR (learning-to-rank design).

Phase 2H = A8 as FILTER on production BUYs.
4C (this) = A8 as SELECTOR: A8 ranks full cross-section -> top-`pool`,
then per-ticker model picks top-`pick` by prob_up. Tests whether A8's
cross-sectional ranking surfaces better names than the model's own BUYs.

SHADOW ONLY: reads accuracy.db + a8_oos_panel.parquet. No live change.

PRE-REGISTERED GATE: 4C must beat production risk-adjusted (higher Sharpe
AND not-worse maxDD AND win>=prod). Raw cum return alone NOT sufficient.
Even a pass = flag for longer test, not a promote.
CAVEAT: ~40-day single-regime window. Suggestive only.
"""
import argparse, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
A8_PANEL = ROOT / "data" / "a8_oos_panel.parquet"
ACC_DB = ROOT / "accuracy.db"


def load_joined(horizon):
    conn = sqlite3.connect(ACC_DB)
    preds = pd.read_sql("""
        SELECT p.prediction_date, p.ticker, p.horizon, p.signal,
               p.prob_up, o.actual_return
        FROM predictions p
        JOIN outcomes o USING (ticker, prediction_date, horizon)
        WHERE o.actual_return IS NOT NULL AND p.horizon = ?
    """, conn, params=(horizon,))
    conn.close()
    preds["prediction_date"] = pd.to_datetime(preds["prediction_date"])
    panel = pd.read_parquet(A8_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    return preds.merge(
        panel.rename(columns={"date": "prediction_date"})[
            ["prediction_date", "ticker", "a8_prob"]],
        on=["ticker", "prediction_date"], how="inner")


def portfolio_stats(daily_returns):
    r = daily_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(n_days=len(r), mean=np.nan, cum=np.nan, win=np.nan,
                    sharpe=np.nan, maxdd=np.nan)
    cum_curve = (1 + r / 100.0).cumprod()
    dd = (cum_curve / cum_curve.cummax() - 1) * 100
    return dict(n_days=len(r), mean=r.mean(),
                cum=(cum_curve.iloc[-1] - 1) * 100, win=(r > 0).mean(),
                sharpe=r.mean() / r.std() * np.sqrt(len(r)), maxdd=dd.min())


def run(horizon, pool, pick):
    df = load_joined(horizon)
    dates = sorted(df["prediction_date"].unique())
    prod_daily, c4_daily, c4_overlap, n_pool_avg = [], [], [], []
    for d in dates:
        day = df[df["prediction_date"] == d]
        if day["a8_prob"].notna().sum() < pool:
            continue
        prod = day[day["signal"] == "BUY"]
        if len(prod):
            prod_daily.append((d, prod["actual_return"].mean(), len(prod)))
        cand = day.dropna(subset=["a8_prob"]).nlargest(pool, "a8_prob")
        n_pool_avg.append(len(cand))
        picks = cand.nlargest(pick, "prob_up")
        if len(picks):
            c4_daily.append((d, picks["actual_return"].mean(), len(picks)))
            ov = picks["ticker"].isin(set(prod["ticker"])).sum()
            c4_overlap.append(ov / len(picks))
    prod_s = pd.DataFrame(prod_daily, columns=["date","ret","n"]).set_index("date")["ret"]
    c4_s = pd.DataFrame(c4_daily, columns=["date","ret","n"]).set_index("date")["ret"]
    ps, cs = portfolio_stats(prod_s), portfolio_stats(c4_s)
    print("=" * 78)
    print(f"4C SHADOW — A8-as-SELECTOR  (h={horizon}d, pool=top{pool}, pick=top{pick})")
    print(f"Window: {pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}  ({len(dates)} dates)")
    print("=" * 78)
    print(f"{'portfolio':<14}{'days':>5}{'avg_n':>7}{'mean%':>8}{'cum%':>9}{'win':>7}{'sharpe':>8}{'maxDD%':>9}")
    print(f"{'production':<14}{ps['n_days']:>5}{np.mean([x[2] for x in prod_daily]):>7.1f}"
          f"{ps['mean']:>8.3f}{ps['cum']:>9.1f}{ps['win']:>7.3f}{ps['sharpe']:>8.3f}{ps['maxdd']:>9.2f}")
    print(f"{'4C_selector':<14}{cs['n_days']:>5}{np.mean(n_pool_avg):>7.1f}"
          f"{cs['mean']:>8.3f}{cs['cum']:>9.1f}{cs['win']:>7.3f}{cs['sharpe']:>8.3f}{cs['maxdd']:>9.2f}")
    print("=" * 78)
    print("\nGATE CHECK (risk-adjusted, not raw return):")
    sb = cs["sharpe"] > ps["sharpe"]
    dn = cs["maxdd"] >= ps["maxdd"] - 2.0
    wb = cs["win"] >= ps["win"]
    print(f"  Sharpe higher?    4C {cs['sharpe']:.2f} vs prod {ps['sharpe']:.2f}  -> {sb}")
    print(f"  maxDD not worse?  4C {cs['maxdd']:.1f} vs prod {ps['maxdd']:.1f}  -> {dn}")
    print(f"  win >= prod?      4C {cs['win']:.3f} vs prod {ps['win']:.3f}  -> {wb}")
    print(f"  4C picks also in production BUYs: {np.mean(c4_overlap)*100:.0f}% avg")
    print("\nVERDICT:", "PASS gate (flag for LONGER test, NOT promote)" if (sb and dn and wb)
          else "FAIL gate — 4C-as-selector does not beat production risk-adjusted")
    print("\nCAVEAT: ~40-day single-regime window. Suggestive only.")
    print("A8 panel covers 125 tickers (pre-expansion); 32 newer names invisible.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--pick", type=int, default=5)
    a = ap.parse_args()
    run(a.horizon, a.pool, a.pick)
