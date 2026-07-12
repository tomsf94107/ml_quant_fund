#!/usr/bin/env python3
"""
analysis/si_vol_sizing_ceiling.py -- does vol-sizing help the SI book?

WHY THIS EXISTS
  On 2026-07-12 I shipped --vol-target to si_positions_live.py (a LIVE money book)
  on the reasoning that equal DOLLARS != equal RISK. Then I found
  docs/vol_sizing_decision.md: an oracle pre-test on 2026-06-03 that had ALREADY
  tested this, on the MOMENTUM book:

      sizing                              mom_6_1 net Sh
      equal-weight                            +1.225
      naive trailing-40d-vol                  +1.437   <-- WINNER
      PERFECT future-vol (oracle ceiling)     +1.097

  Two things follow, and they point in OPPOSITE directions:
    1. naive trailing-vol BEAT equal-weight (+1.437 vs +1.225). Vol-sizing per se
       was NOT killed -- what I shipped (trailing vol) was the winner there.
    2. But PERFECT vol knowledge made it WORSE than naive. So a better vol
       FORECASTER cannot help. models/vol_forecast.py (OOS Spearman +0.706, built
       tonight) is therefore useless FOR SIZING on that book, whatever its accuracy.

  The stated mechanism was momentum-specific:
      "the names that keep running are often higher-vol; sizing toward future-calm
       names tilts into the momentum underperformers. Vol and momentum-return are
       ENTANGLED such that better vol forecasting is counterproductive."

  The SI book selects on DAYS-TO-COVER -- a positioning measure, not a price trend.
  There is no obvious reason that entanglement carries over. But "no obvious reason"
  is a hypothesis, not a result, and I have already put this live. So: run THE SAME
  TEST, on THE SI BOOK, with a pre-registered decision rule.

PRE-REGISTERED DECISION (written before the numbers exist)
    naive > equal  -> --vol-target STAYS. Same verdict momentum reached.
    naive <= equal -> REVERT --vol-target. It was shipped on an untested assumption.
    oracle < naive -> the forecaster is dead here too. vol_forecast.py is for regime
                      gating only, never for sizing.

MECHANICS (mirrors analysis/vol_sizing_ceiling.py so the two are comparable)
  Signal      : low days_to_cover = long (the validated sign, NW-t -4.76 @ h=40)
  PIT entry   : settlement_date + 8 BUSINESS DAYS (FINRA's dissemination lag).
                Entering on settlement_date is 8 BD of look-ahead.
  Hold        : 40 trading days (where the brick lives; at 1/3/5d it is not there)
  Overlap     : cohorts overlap, so the portfolio on any day is the average of all
                active cohorts. Sharpe is computed on the DAILY portfolio series,
                not on overlapping 40-day returns -- overlapping returns would
                inflate the t-stat, which is the bug behind PEAD's t=-20.
  Cost        : COST_BPS round-trip, charged at cohort entry (conservative).
  No fitting  : equal and naive have zero free parameters, so there is nothing to
                overfit; the oracle is deliberately look-ahead, to find the CEILING.
"""
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0
QUANTILE = 0.20          # low-DTC quintile
HOLD     = 40            # trading days -- the validated horizon
VOL_LB   = 40            # trailing window for naive vol (matches the momentum test)
LAG_BD   = 8             # FINRA dissemination
CLIP_DTC = 50.0

# ── data ───────────────────────────────────────────────────────────────────
si = pd.read_sql("SELECT ticker, settlement_date, days_to_cover FROM short_interest "
                 "WHERE days_to_cover IS NOT NULL",
                 sqlite3.connect(ROOT/"short_interest.db"))
si["settlement_date"] = pd.to_datetime(si["settlement_date"])
si = si[si["days_to_cover"] <= CLIP_DTC]
si["entry"] = si["settlement_date"] + pd.tseries.offsets.BDay(LAG_BD)
si["ticker"] = si["ticker"].str.upper()

px = pd.read_sql("SELECT ticker, date, adj_close FROM daily_prices "
                 "WHERE adj_close IS NOT NULL", sqlite3.connect(ROOT/"prices.db"))
px["date"] = pd.to_datetime(px["date"]); px["ticker"] = px["ticker"].str.upper()
close = px.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
ret   = close.pct_change()
tdays = close.index

print(f"  SI    : {len(si):,} rows / {si.ticker.nunique()} tickers / "
      f"{si.settlement_date.nunique()} settlements")
print(f"  prices: {close.shape[0]} days x {close.shape[1]} tickers")
print(f"  entry = settlement + {LAG_BD} BD | hold {HOLD}d | low-DTC quintile | "
      f"{COST_BPS:.0f}bps round-trip\n")

# forward realized vol per ticker (THE ORACLE -- deliberate look-ahead)
fwd_vol = ret[::-1].rolling(HOLD).std()[::-1].shift(-1)
trl_vol = ret.rolling(VOL_LB).std()

# ── build cohorts ──────────────────────────────────────────────────────────
def cohort_weights(entry_d, names, mode):
    w = {}
    for t in names:
        if mode == "equal":
            w[t] = 1.0
        elif mode == "naive":
            v = trl_vol.loc[entry_d, t] if (entry_d in trl_vol.index and t in trl_vol.columns) else np.nan
            w[t] = 1.0/v if (pd.notna(v) and v > 0) else np.nan
        else:                                    # oracle
            v = fwd_vol.loc[entry_d, t] if (entry_d in fwd_vol.index and t in fwd_vol.columns) else np.nan
            w[t] = 1.0/v if (pd.notna(v) and v > 0) else np.nan
    s = pd.Series(w, dtype=float)
    if not s.notna().any():
        return None
    s = s.fillna(s.median())
    return s / s.sum()

results = {}
for mode in ("equal", "naive", "oracle"):
    daily = pd.Series(0.0, index=tdays)
    active = pd.Series(0.0, index=tdays)
    n_cohorts = 0
    for entry_d, grp in si.groupby("entry"):
        # snap the entry to the next real trading day
        idx = tdays.searchsorted(entry_d)
        if idx >= len(tdays) - HOLD - 1:
            continue
        d = tdays[idx]
        uni = grp[grp.ticker.isin(close.columns)].sort_values("days_to_cover")
        if len(uni) < 30:
            continue
        k = max(1, int(len(uni)*QUANTILE))
        names = list(uni.head(k)["ticker"])         # LOW DTC = long
        w = cohort_weights(d, names, mode)
        if w is None:
            continue
        n_cohorts += 1
        window = tdays[idx+1: idx+1+HOLD]
        r = (ret.loc[window, names] * w.reindex(names).values).sum(axis=1)
        r.iloc[0] -= (COST_BPS/1e4) * 2.0           # round-trip cost at entry
        daily.loc[window] += r.values
        active.loc[window] += 1.0

    live = active > 0
    port = (daily[live] / active[live]).dropna()
    sd = port.std()
    sh = (port.mean()/sd) * np.sqrt(252) if sd > 0 else np.nan
    curve = (1 + port).cumprod()
    dd = float((curve/curve.cummax() - 1).min())
    results[mode] = (round(float(sh), 3), round(100*dd, 1), len(port), n_cohorts)

print("=" * 74)
print("  SI BOOK (long low-DTC, 40d hold) -- SIZING CEILING")
print("=" * 74)
print(f"  {'sizing':<38}{'net Sharpe':>12}{'maxDD':>9}{'days':>7}")
lbl = {"equal":  "equal-weight",
       "naive":  "naive trailing-40d-vol  <-- SHIPPED",
       "oracle": "PERFECT future-vol (ORACLE ceiling)"}
for m in ("equal", "naive", "oracle"):
    sh, dd, n, nc = results[m]
    print(f"  {lbl[m]:<38}{sh:>+12.3f}{dd:>8.1f}%{n:>7}")

eq, nv, orc = results["equal"][0], results["naive"][0], results["oracle"][0]
print()
print(f"  naive - equal   = {nv-eq:+.3f}   (is vol-sizing worth anything at all?)")
print(f"  oracle - naive  = {orc-nv:+.3f}   (room a FORECASTER could fill)")
print()
print("=" * 74)
if nv > eq:
    print(f"  >> --vol-target STAYS. Naive trailing-vol beats equal-weight by {nv-eq:+.3f}")
    print("     Sharpe. Same verdict the momentum book reached (+1.437 vs +1.225).")
    if orc <= nv:
        print(f"  >> But the ORACLE ({orc:+.3f}) does NOT beat naive ({nv:+.3f}): even PERFECT")
        print("     vol knowledge cannot improve on the lagged estimate. models/vol_forecast.py")
        print("     is confirmed USELESS FOR SIZING here too -- exactly as the Jun 3 oracle")
        print("     test found on momentum. Use it for regime gating or not at all.")
    else:
        print(f"  >> And the ORACLE ({orc:+.3f}) beats naive by {orc-nv:+.3f} -- unlike momentum,")
        print("     a better vol forecast COULD help this book. vol_forecast.py has a use.")
else:
    print(f"  >> REVERT --vol-target. Naive ({nv:+.3f}) does NOT beat equal-weight ({eq:+.3f}).")
    print("     It was shipped tonight on an untested assumption. Roll it back:")
    print("       git checkout si_positions_live.py.bak.emit.*  (or drop the flag)")
print("=" * 74)
