#!/usr/bin/env python3
"""
scripts/research_reminders.py -- fire when a research question becomes ANSWERABLE.

WHY CONDITIONS, NOT DATES
  "Revisit in 6 months" is a calendar reminder. It fires whether or not the data is
  there, and it fires on a guess. Every item below is blocked on a MEASURABLE amount
  of accumulated data, so the reminder should check the data.

  It also means a stalled feed cannot hide. On 2026-07-12 the momentum shadow was
  found frozen for 14 days -- Pipeline C reported SUCCESS every day while writing
  zero new rows. A date-based reminder would have fired in September and told you
  nothing was ready, with no idea why. A condition-based one shows the counter not
  moving.

THE FOUR OPEN QUESTIONS (2026-07-12)
  1. MOMENTUM     -- backtest says +1.750 net Sharpe (3/3 folds). LIVE SHADOW says
                     -7.23pp edge. Those cannot both be right. Needs enough resolved
                     weeks to tell a 2-week semi selloff from a dead signal.
  2. GEX          -- predicts forward realized vol, survives the vol-proxy control
                     (104% retained) and a block-bootstrap null (p=0.002). But 231
                     days in ONE calm regime. UW's window is rolling and CANNOT be
                     backfilled -- it only accrues.
  3. DIRECTION BUY-- beats the base rate (+3.5pp h=3, +2.2pp h=5) and is NOT a beta
                     tilt (BUY beta 1.08 vs field 1.21). But BUY-vs-universe t=+1.18
                     on 69 dates: underpowered, not disproven.
  4. SI x BUY     -- does the direction model's BUY add anything on top of low-DTC?
                     Only 7 dates where both signals exist. Inconclusive.

  Every one is blocked on DATA, not on thinking. Nothing to do but wait -- and be
  told when the wait is over.
"""
import sqlite3, subprocess, sys
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
def con(db): return sqlite3.connect(f"file:{ROOT/db}?mode=ro", uri=True, timeout=20)

def notify(title, msg):
    subprocess.run(["osascript","-e",f'display notification "{msg}" with title "{title}"'],
                   check=False, capture_output=True)

RESULTS = []
def check(name, have, need, unit, note, ready_action):
    pct = min(100, int(100*have/need)) if need else 0
    RESULTS.append(dict(name=name, have=have, need=need, unit=unit, pct=pct,
                        note=note, action=ready_action, ready=have >= need))

# ── 1. MOMENTUM ────────────────────────────────────────────────────────────
# The gate needs enough RESOLVED WEEKS to judge consistency (>=60% positive weeks).
# 12 weeks ~= 3 months of live outcomes. Outcomes lag predictions by 20 trading days.
try:
    c = con("accuracy.db")
    d = pd.read_sql("SELECT DISTINCT prediction_date d FROM momentum_shadow_outcomes", c)
    c.close()
    d["d"] = pd.to_datetime(d["d"])
    weeks = d["d"].dt.to_period("W").nunique()
    check("MOMENTUM promotion gate", weeks, 12, "resolved weeks",
          "backtest +1.750 vs live -7.23pp -- one of them is wrong",
          "python3 -m scripts.momentum_promotion_check")
except Exception as e:
    check("MOMENTUM promotion gate", 0, 12, "resolved weeks", f"ERROR: {e}", "")

# ── 2. GEX ─────────────────────────────────────────────────────────────────
# 500 trading days ~= 2 years -> at least 2 distinct volatility regimes, and enough
# non-overlapping samples for the independence test that was underpowered at n=11.
try:
    c = con("accuracy.db")
    n = c.execute("SELECT COUNT(DISTINCT date) FROM options_greeks").fetchone()[0]
    c.close()
    check("GEX tradeable sample", n, 500, "trading days",
          "real vol signal, survives every control -- but ONE calm regime so far",
          "python3 validate_gex_block.py && python3 test_gex_vol_ab.py")
except Exception as e:
    check("GEX tradeable sample", 0, 500, "trading days", f"ERROR: {e}", "")

# ── 3. DIRECTION BUY ───────────────────────────────────────────────────────
# 150 dates roughly triples the current 69 -> t-stat scales ~sqrt(n), so a t=+1.18
# would land near +1.8 if the effect is real and stable. Still not conclusive, but it
# would move the question. Fewer than that and we are just re-reading noise.
try:
    c = con("accuracy.db")
    n = c.execute("""SELECT COUNT(DISTINCT p.prediction_date)
                     FROM predictions p JOIN outcomes o
                       ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
                      AND p.horizon=o.horizon
                     WHERE p.horizon=3 AND p.prob_up>=0.60""").fetchone()[0]
    c.close()
    check("DIRECTION BUY signal", n, 150, "resolved dates",
          "+3.5pp over base rate and NOT a beta tilt -- but t=+1.18, underpowered",
          "BUY-vs-universe test (see 2026-07-12 session)")
except Exception as e:
    check("DIRECTION BUY signal", 0, 150, "resolved dates", f"ERROR: {e}", "")

# ── 4. SI x BUY INTERSECTION ───────────────────────────────────────────────
# Dates where a FINRA settlement+8BD entry coincides with direction-model predictions.
# Settlements are ~24/yr, so 30 dates ~= 15 months of overlap. This is the slowest one.
try:
    c = con("short_interest.db")
    si = pd.read_sql("SELECT DISTINCT settlement_date FROM short_interest", c); c.close()
    si["e"] = pd.to_datetime(si["settlement_date"]) + pd.tseries.offsets.BDay(8)
    c = con("accuracy.db")
    pr = pd.read_sql("SELECT DISTINCT prediction_date d FROM predictions", c); c.close()
    pr["d"] = pd.to_datetime(pr["d"])
    n = len(set(si["e"]) & set(pr["d"]))
    check("SI x BUY intersection", n, 30, "overlapping dates",
          "does BUY add anything on top of low-DTC? 7 dates = inconclusive",
          "intersection test (see 2026-07-12 session)")
except Exception as e:
    check("SI x BUY intersection", 0, 30, "overlapping dates", f"ERROR: {e}", "")

# ── report ─────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"  RESEARCH REMINDERS — {datetime.now():%Y-%m-%d}")
print("=" * 78)
ready = []
for r in RESULTS:
    bar = "#" * (r["pct"] // 5) + "." * (20 - r["pct"] // 5)
    flag = "  ✅ READY" if r["ready"] else ""
    print(f"\n  {r['name']}{flag}")
    print(f"    [{bar}] {r['have']}/{r['need']} {r['unit']}  ({r['pct']}%)")
    print(f"    {r['note']}")
    if r["ready"]:
        ready.append(r)
        print(f"    >>> RUN: {r['action']}")

print()
print("=" * 78)
if ready:
    names = ", ".join(r["name"] for r in ready)
    print(f"  {len(ready)} QUESTION(S) NOW ANSWERABLE: {names}")
    notify("ML Quant — RESEARCH READY", f"{len(ready)} question(s) now have enough data: {names}")
else:
    print("  Nothing ready. All four are blocked on DATA, not on thinking.")
    print("  Nothing to do but let the feeds run.")
print()
print("  A counter that is NOT moving week over week means a feed is DEAD.")
print("  (momentum_shadow was frozen 14 days in June while Pipeline C reported OK.)")
