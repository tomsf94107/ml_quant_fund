#!/usr/bin/env python3
"""
analysis/combine_mom_si.py -- do momentum and short-interest combine?

THE GATE (docs/MASTER_TODO_LIST.md:302)
  "C1 COMBINER -- GATE: requires >=2 VALIDATED, DECORRELATED *RETURN* alphas.
   STATUS Jun 1: we have 1 return-alpha (momentum) + 1 RISK signal (vol) -- vol is
   NOT a return-alpha, so C1 is STILL BLOCKED."

  That was June 1. Since then the SI brick was validated (per-date IC -0.053,
  NW-t -4.76, negative every year 2021-2026, survives the 8-BD dissemination lag,
  80% survives sector-neutralisation). So there may now be TWO return alphas:

      MOMENTUM : 4/4 OOS folds, mean net Sharpe +0.96, 20d hold.  PRICE axis.
      SHORT INT: IC -0.053, NW-t -4.76, 40d hold.                 POSITIONING axis.

  The docs' own warning (line 77): "you can't diversify a price-momentum book with
  another price signal -- same information." DTC is not a price signal. That is
  exactly the gap it could fill.

WHY THE HEDGED STREAMS AND NOT THE RAW ONES  <-- THE WHOLE POINT
  The SI long book has beta 1.24 vs the universe (si_book_diagnostic). Momentum
  books are typically high-beta too (winners are volatile names). If both are
  secretly levered market bets, their RAW returns will correlate strongly -- but
  only because they are both long the market, which tells you nothing about whether
  their ALPHAS diversify.

  Worse, combining two levered market bets does not diversify anything. It doubles
  the beta and produces a smoother-looking curve that blows up in exactly the same
  drawdown.

  So the correlation MUST be measured on the BETA-STRIPPED streams. That is the
  control that has killed five findings in this fund today (PEAD, the ranker, the
  h=40 model, the BUY lift, and the SI book's own headline return). It applies here
  too.

PRE-REGISTERED DECISION RULE (written before the number exists)
    hedged corr < 0.30   -> COMBINE. Real diversification. C1 unblocks.
    hedged corr 0.30-0.60-> MARGINAL. Benefit exists but is smaller than it looks.
    hedged corr > 0.60   -> SAME BET. Do not build the combiner.

  Plus the direct test: does the 50/50 combined book beat BOTH singles on Sharpe?
  Grinold: IR = IC x sqrt(breadth). Two uncorrelated signals -> ~1.41x the IR of
  either. Two correlated signals -> nothing.
"""
import sys, sqlite3, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0
MOM_HOLD, MOM_DEC = 20, 0.10        # momentum: 20d hold, top decile  (docs 1.4)
SI_HOLD,  SI_Q    = 40, 0.20        # SI: 40d hold, low-DTC quintile  (validated)
LAG_BD            = 8               # FINRA dissemination lag

# ── prices ─────────────────────────────────────────────────────────────────
px = pd.read_sql("SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL",
                 sqlite3.connect(ROOT/"prices.db"))
px["date"] = pd.to_datetime(px["date"]); px["ticker"] = px["ticker"].str.upper()
close = px.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
close = close[close.index >= "2021-01-01"]
ret   = close.pct_change(fill_method=None)
tdays = close.index
mkt   = ret.mean(axis=1)                       # equal-weight universe = "the market"

# ── momentum book (mom_6_1: 126d return minus last 21d) ────────────────────
# mom_12_1, NOT mom_6_1. Tonight's purged-WF re-run (analysis/momentum_purged_wf):
#   mom_6_1  : 1/3 folds positive, mean net Sharpe +0.217  -> GATE FAIL (DECAYED;
#              the docs' 4/4 +0.96 was measured on 2021-26, it no longer holds
#              on 2024-26)
#   mom_12_1 : 3/3 folds positive, mean net Sharpe +1.750  -> GATE PASS
# The first combiner run used mom_6_1 -- i.e. it correlated SI against a signal
# that no longer works. That verdict is void. This is the live one.
mom = close.pct_change(252, fill_method=None) - close.pct_change(21, fill_method=None)

def book_returns(signal_fn, hold, rebal_dates):
    """Daily return of a long-only book, averaging OVERLAPPING cohorts.

    Overlapping cohorts must be AVERAGED, not summed. si_book_diagnostic cumsums
    them, which implicitly models ~2.7x leverage and is why it reports -84.5% DD
    where the 1x book draws -27%. Averaging is what si_positions_live actually does.
    """
    daily  = pd.Series(0.0, index=tdays)
    active = pd.Series(0.0, index=tdays)
    for d in rebal_dates:
        i = tdays.searchsorted(d)
        if i >= len(tdays) - hold - 1: continue
        d = tdays[i]
        names = signal_fn(d)
        if names is None or len(names) < 5: continue
        w = pd.Series(1.0/len(names), index=names)
        win = tdays[i+1 : i+1+hold]
        r = (ret.loc[win, names] * w.values).sum(axis=1)
        r.iloc[0] -= (COST_BPS/1e4)*2.0
        daily.loc[win]  += r.values
        active.loc[win] += 1.0
    live = active > 0
    return (daily[live]/active[live]).dropna()

# momentum: rebalance every 20 trading days, long top decile
mom_dates = tdays[::MOM_HOLD]
def mom_pick(d):
    row = mom.loc[d].dropna()
    if len(row) < 30: return None
    k = max(1, int(len(row)*MOM_DEC))
    return list(row.sort_values(ascending=False).head(k).index)
r_mom = book_returns(mom_pick, MOM_HOLD, mom_dates)

# SI: rebalance on each settlement + 8BD, long low-DTC quintile
si = pd.read_sql("SELECT ticker,settlement_date,days_to_cover FROM short_interest "
                 "WHERE days_to_cover IS NOT NULL", sqlite3.connect(ROOT/"short_interest.db"))
si["settlement_date"] = pd.to_datetime(si["settlement_date"])
si = si[si["days_to_cover"] <= 50.0]
si["entry"] = si["settlement_date"] + pd.tseries.offsets.BDay(LAG_BD)
si["ticker"] = si["ticker"].str.upper()
si_map = {d: g for d, g in si.groupby("entry")}
def si_pick(d):
    # the cohort keyed nearest this entry date
    cand = [k for k in si_map if abs((k-d).days) <= 5]
    if not cand: return None
    g = si_map[cand[0]]
    g = g[g.ticker.isin(close.columns)].sort_values("days_to_cover")
    if len(g) < 30: return None
    k = max(1, int(len(g)*SI_Q))
    return list(g.head(k)["ticker"])
r_si = book_returns(si_pick, SI_HOLD, sorted(si_map.keys()))

# ── align, then BETA-STRIP both ────────────────────────────────────────────
df = pd.DataFrame({"mom": r_mom, "si": r_si, "mkt": mkt}).dropna()
print(f"  overlapping days: {len(df)}  ({df.index.min().date()} .. {df.index.max().date()})\n")

def hedge(r, m):
    b = np.polyfit(m, r, 1)[0]
    return r - b*m, b

h_mom, b_mom = hedge(df["mom"], df["mkt"])
h_si,  b_si  = hedge(df["si"],  df["mkt"])

def stats(r):
    sd = r.std()
    sh = (r.mean()/sd)*math.sqrt(252) if sd > 0 else np.nan
    c  = (1+r).cumprod()
    return sh, float((c/c.cummax()-1).min())

print("="*78)
print("  THE TWO BOOKS")
print("="*78)
print(f"  {'':26}{'Sharpe':>9}{'maxDD':>9}{'beta':>8}")
for nm, r, b in [("MOMENTUM (raw)", df["mom"], b_mom), ("SHORT INTEREST (raw)", df["si"], b_si)]:
    sh, dd = stats(r); print(f"  {nm:26}{sh:>+9.2f}{100*dd:>8.1f}%{b:>8.2f}")
for nm, r in [("MOMENTUM (beta-hedged)", h_mom), ("SHORT INTEREST (hedged)", h_si)]:
    sh, dd = stats(r); print(f"  {nm:26}{sh:>+9.2f}{100*dd:>8.1f}%{'--':>8}")

raw_c = df["mom"].corr(df["si"])
hed_c = h_mom.corr(h_si)
print()
print("="*78)
print("  THE NUMBER THAT DECIDES IT")
print("="*78)
print(f"  correlation, RAW returns    : {raw_c:+.3f}   (inflated -- both are long the market)")
print(f"  correlation, BETA-HEDGED    : {hed_c:+.3f}   <-- THE HONEST ONE")
print()

# 50/50 combined, on the hedged (alpha) streams
comb  = 0.5*h_mom + 0.5*h_si
sh_c, dd_c = stats(comb)
sh_m, _    = stats(h_mom)
sh_s, _    = stats(h_si)
best = max(sh_m, sh_s)
print(f"  {'50/50 COMBINED (hedged)':26}{sh_c:>+9.2f}{100*dd_c:>8.1f}%")
print(f"  best single (hedged)      : {best:+.2f}")
print(f"  lift                      : {sh_c-best:+.2f}   "
      f"(Grinold: 2 uncorrelated signals -> ~1.41x IR)")
print()
print("="*78)
if hed_c < 0.30:
    print(f"  >> COMBINE. Hedged correlation {hed_c:+.3f} < 0.30. These are genuinely")
    print("     different bets: momentum is PRICE, days-to-cover is POSITIONING.")
    print("     The C1 combiner gate is MET -- it has been blocked since June 1 for")
    print("     lack of a 2nd return alpha. Build the combiner (HRP or equal-weight).")
elif hed_c < 0.60:
    print(f"  >> MARGINAL. Hedged correlation {hed_c:+.3f}. Some diversification, but")
    print("     less than the headline suggests. Combine only if the lift above is real.")
else:
    print(f"  >> SAME BET. Hedged correlation {hed_c:+.3f} > 0.60. Combining these adds")
    print("     nothing -- you would be making one bet twice. Do NOT build the combiner.")
    print("     Keep hunting a genuinely different axis.")
print("="*78)
print("  CAVEAT: in-sample, survivor-tilted, one bull-tape regime. A correlation")
print("  measured over 2021-2026 need not hold in a crisis, when everything")
print("  correlates to 1. Size accordingly.")
