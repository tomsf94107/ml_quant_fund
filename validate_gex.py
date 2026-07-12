#!/usr/bin/env python3
"""
validate_gex.py -- is dealer gamma exposure a real brick? Same gauntlet as SI.

WHAT IS BEING TESTED
  GEX = net dealer gamma. The mechanism is not a statistical pattern, it is a
  FORCED TRADE:
    GEX > 0  dealers are long gamma -> they SELL rallies, BUY dips -> they DAMPEN
             moves -> expect mean reversion and lower realized vol.
    GEX < 0  dealers are short gamma -> they BUY rallies, SELL dips -> they
             AMPLIFY moves -> expect momentum and higher realized vol.
  This is a DIFFERENT AXIS from everything else in the system: it is not what
  prices did, it is what market makers must do next.

THE GAUNTLET (every control that caught a fake result in this fund)
  1. PER-DATE IC, not pooled stock-dates. Pooling ~400 stocks x 250 dates as if
     independent is what inflated the original SI t-stat to -20. The honest n is
     the number of DATES.
  2. NEWEY-WEST t. Overlapping forward windows autocorrelate the IC series.
  3. NULL CONTROL. Shuffle forward returns within each date; IC must collapse.
     Catches a broken pipeline. Does NOT catch beta.
  4. BETA CONTROL. <-- THE ONE THAT MATTERS. Three "discoveries" in this fund
     turned out to be market beta: PEAD, the ranker, and the h=40 model (IC +0.20
     that collapsed to +0.002 once beta was stripped). The null control passed on
     all of them, because shuffling destroys beta too. So: residualise the signal
     against beta WITHIN each date and re-measure. What survives is alpha.
  5. PER-TICKER Z-SCORE. Raw GEX scales with market cap and open interest --
     NVDA's gamma is ~100x SENS's for reasons that have nothing to do with signal.
     Cross-sectionally, raw GEX would just rank stocks by SIZE. Normalise per
     ticker (z-score over its own trailing history) so the object is "unusually
     positive/negative gamma FOR THIS NAME".
  6. BOTH SIGNS TESTED. GEX has no a-priori direction for RETURNS -- theory says
     it predicts VOLATILITY and mean-reversion-vs-momentum, not up-vs-down. So we
     test gex_z against forward return AND against forward absolute return
     (realized vol), and we report both honestly.

Usage:
  python validate_gex.py                 # h=5 default
  python validate_gex.py --hold 1 --hold 3 --hold 5 --hold 10
"""
import argparse, sqlite3, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--holds", type=int, nargs="+", default=[1, 3, 5, 10, 20])
ap.add_argument("--zwin", type=int, default=60, help="trailing window for the per-ticker z")
ap.add_argument("--min-names", type=int, default=30)
a = ap.parse_args()

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return np.nan
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * ((e[k:] @ e[:-k]) / n)
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan

# ── data ───────────────────────────────────────────────────────────────────
con = sqlite3.connect("accuracy.db")
gx = pd.read_sql("SELECT ticker, date, net_gamma, net_delta FROM options_greeks "
                 "WHERE net_gamma IS NOT NULL", con)
con.close()
if gx.empty:
    print("  options_greeks is EMPTY -- run backfill_greeks.py first."); sys.exit(1)

con = sqlite3.connect("prices.db")
# daily_prices is (ticker, date, adj_close) -- SPLIT-ADJUSTED, which is what a
# return series needs. raw_bars.close is unadjusted and would show a fake -50% on
# every split ex-date (the HON bug from this session).
px = pd.read_sql("SELECT ticker, date, adj_close AS close FROM daily_prices "
                 "WHERE date >= '2025-06-01'", con)
con.close()

gx["date"] = pd.to_datetime(gx["date"]); px["date"] = pd.to_datetime(px["date"])
gx["ticker"] = gx["ticker"].str.upper(); px["ticker"] = px["ticker"].str.upper()

print(f"  GEX   : {len(gx):,} rows / {gx.ticker.nunique()} tickers / "
      f"{gx.date.min().date()} .. {gx.date.max().date()}")
print(f"  prices: {len(px):,} rows / {px.ticker.nunique()} tickers")

# ── per-ticker z-score: raw GEX ranks stocks by SIZE, not by signal ─────────
gx = gx.sort_values(["ticker", "date"])
g = gx.groupby("ticker")["net_gamma"]
gx["gex_z"] = ((gx["net_gamma"] - g.transform(lambda s: s.rolling(a.zwin, min_periods=20).mean()))
               / g.transform(lambda s: s.rolling(a.zwin, min_periods=20).std()))
gx = gx.replace([np.inf, -np.inf], np.nan).dropna(subset=["gex_z"])
print(f"  gex_z : {len(gx):,} rows after per-ticker {a.zwin}d z-score\n")

px = px.sort_values(["ticker", "date"])
# beta proxy: 60d rolling corr(stock ret, equal-weight universe ret) * vol ratio.
px["ret"] = px.groupby("ticker")["close"].pct_change()
mkt = px.groupby("date")["ret"].mean().rename("mkt")
px = px.merge(mkt, on="date", how="left")
px["beta"] = (px.groupby("ticker")
                .apply(lambda d: d["ret"].rolling(60, min_periods=30)
                       .cov(d["mkt"]) / d["mkt"].rolling(60, min_periods=30).var())
                .reset_index(level=0, drop=True))

def per_date_ic(df, sig, ret):
    out = {}
    for d, grp in df.groupby("date"):
        if len(grp) < a.min_names: continue
        if grp[sig].nunique() < 5: continue
        ic = grp[sig].corr(grp[ret], method="spearman")
        if pd.notna(ic): out[d] = ic
    return pd.Series(out, dtype=float).sort_index()

def strip_beta(df, sig):
    def _r(grp):
        x = grp["beta"].astype(float).values; y = grp[sig].astype(float).values
        if len(grp) < 20 or np.nanstd(x) < 1e-9: return pd.Series(y, index=grp.index)
        x = np.nan_to_num(x, nan=float(np.nanmean(x)))
        b = np.polyfit(x, y, 1)
        return pd.Series(y - np.polyval(b, x), index=grp.index)
    out = df.copy()
    out["_resid"] = out.groupby("date", group_keys=False).apply(_r)
    return out

print("=" * 86)
print(f"  {'h':>3}{'signal':>16}{'mean IC':>10}{'IC IR':>8}{'NW-t':>8}"
      f"{'+sign':>7}{'dates':>7}{'null':>9}{'beta-strip':>12}")
print("=" * 86)

for h in a.holds:
    p = px.copy()
    p["fwd"] = p.groupby("ticker")["close"].shift(-h) / p["close"] - 1.0
    p["fwd_abs"] = p["fwd"].abs()          # realized vol -- what theory says GEX predicts
    m = gx.merge(p[["ticker", "date", "fwd", "fwd_abs", "beta"]],
                 on=["ticker", "date"], how="inner").dropna(subset=["fwd", "gex_z"])
    if m.empty:
        print(f"  {h:>3}  no overlap"); continue
    lag = max(1, h // 5)

    for sig_name, ret_col in [("gex_z -> ret", "fwd"), ("gex_z -> |ret|", "fwd_abs")]:
        ic = per_date_ic(m, "gex_z", ret_col)
        if len(ic) < 20:
            continue
        mn = m.copy()
        mn[ret_col] = mn.groupby("date")[ret_col].transform(
            lambda s: np.random.permutation(s.values))
        icn = per_date_ic(mn, "gex_z", ret_col)
        mb = strip_beta(m.dropna(subset=["beta"]), "gex_z")
        icb = per_date_ic(mb, "_resid", ret_col)
        ir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        print(f"  {h:>3}{sig_name:>16}{ic.mean():>+10.4f}{ir:>+8.3f}"
              f"{nw_t(ic.values, lag):>+8.2f}{100*(ic>0).mean():>6.0f}%{len(ic):>7d}"
              f"{icn.mean():>+9.4f}{icb.mean():>+12.4f}")

print("=" * 86)
print("  READ:")
print("   NW-t  : |t| > 2 = significant. > 3 = solid. This is the honest number.")
print("   null  : must be ~0. If not, the pipeline is broken -- ignore everything.")
print("   beta-strip : IC AFTER removing beta. If this collapses toward 0 while the")
print("                raw IC is large, THE SIGNAL IS BETA, not alpha. Three prior")
print("                'discoveries' in this fund died exactly here.")
print("   -> ret  : does gamma predict DIRECTION? (theory: weakly, if at all)")
print("   -> |ret|: does gamma predict VOLATILITY? (theory: YES -- negative gamma")
print("             amplifies moves. This is the mechanistically expected result.)")
