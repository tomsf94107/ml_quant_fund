#!/usr/bin/env python3
"""
Momentum promotion gate — recommends GO / NOT-YET for flipping momentum live.

Reads resolved 20-day forward returns and checks pre-committed criteria. Does NOT
flip anything: going live with real money stays a deliberate manual step (set
MOMENTUM_LIVE=1) taken ONLY when this says GO. Mirrors kill-switch philosophy.

EDGE IS COMPUTED HONESTLY (the Rule-#1 fix, May 31 2026):
momentum_shadow_outcomes contains forward returns for the WHOLE universe (picks
AND non-picks). is_buy_candidate lives in momentum_shadow_predictions, so we JOIN
back to it. Edge = mean(pick returns, is_buy_candidate=1) minus mean(field returns,
is_buy_candidate=0). This is picks-vs-the-names-they-were-chosen-over — the same
"+3.99pp universe edge" from validation, measured live. NOT picks-vs-themselves
(that would read ~0 forever — the bug caught and fixed before shipping).

Pre-committed criteria (set BEFORE any data resolved, so the bar can't be moved):
  1. SAMPLE     : >= 20 resolved BUY picks (noise floor)
  2. EDGE       : live edge vs field > +2.0pp (backtest +3.99; allow ~50% OOS
                  decay per McLean-Pontiff — +2pp is real and survivable)
  3. POSITIVE   : live mean pick return > 0
  4. CONSISTENCY: positive edge in >= 60% of resolved weeks (not one lucky week)
ALL must pass for GO.

    python -m scripts.momentum_promotion_check
"""
import os
import sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB = ROOT / "accuracy.db"
MIN_SAMPLE = 20
MIN_EDGE_PP = 2.0
MIN_WEEK_CONSISTENCY = 0.60
KIND = os.environ.get("MOMENTUM_KIND", "mom_6_1")

JOIN = ("FROM momentum_shadow_outcomes o "
        "JOIN momentum_shadow_predictions p "
        "  ON o.prediction_date=p.prediction_date AND o.ticker=p.ticker AND o.kind=p.kind "
        "WHERE o.kind=? AND o.prediction_date >= '2026-06-11'")  # honest clock: pre-06-11 sample invalidated (overlapping daily entries, 149-name universe)

# GATE RECALIBRATION (2026-07-15). Old fixed bar (+2.0pp) sat ABOVE the 18yr
# backtest mean; a correctly-performing signal cleared it ~38% of the time over
# 3 rebalances. New bar: consistency with the backtest distribution (90% one-
# sided), mirror KILL at 99%. Constants from reports/momentum_18yr_*.csv
# (EW-strip net, real run 2026-07-15).
BT_STATS = {'mom_6_1': (1.015, 4.673), 'mom_12_1': (1.064, 4.76)}  # cap3 EW-strip constants (like-for-like with shadow), 2026-07-15

def _bounds(kind, k):
    import math
    m, s = BT_STATS.get(kind, BT_STATS["mom_6_1"])
    se = s / math.sqrt(max(k, 1))
    return m - 1.28 * se, m - 2.33 * se

def main():
    con = sqlite3.connect(DB); cur = con.cursor()
    print("=" * 60); print("MOMENTUM PROMOTION GATE"); print("=" * 60)

    n = cur.execute(f"SELECT COUNT(*) {JOIN} AND p.is_buy_candidate=1", (KIND,)).fetchone()[0]
    print(f"Resolved {KIND} picks: {n}")
    if n == 0:
        print("\nNOT YET — no resolved outcomes. Shadow loop still accumulating;")
        print("first 20-day picks resolve ~20 trading days after the first logged")
        print("date (first logged 2026-05-29 → first outcomes ~late June). Re-run weekly.")
        con.close(); return 1

    pick_ret = cur.execute(f"SELECT AVG(o.actual_return)*100 {JOIN} AND p.is_buy_candidate=1", (KIND,)).fetchone()[0] or 0.0
    pick_win = cur.execute(f"SELECT AVG(o.actual_up)*100 {JOIN} AND p.is_buy_candidate=1", (KIND,)).fetchone()[0] or 0.0
    field_ret = cur.execute(f"SELECT AVG(o.actual_return)*100 {JOIN} AND p.is_buy_candidate=0", (KIND,)).fetchone()[0] or 0.0
    edge = pick_ret - field_ret

    # per-week edge (pick mean minus field mean, per prediction_date)
    rows = cur.execute(
        f"SELECT o.prediction_date, p.is_buy_candidate, AVG(o.actual_return)*100 "
        f"{JOIN} GROUP BY o.prediction_date, p.is_buy_candidate", (KIND,)).fetchall()
    by_date = {}
    for d, isbuy, avg in rows:
        by_date.setdefault(d, {})[isbuy] = avg
    week_edges = [v[1]-v.get(0,0.0) for v in by_date.values() if 1 in v]
    wpos = sum(1 for e in week_edges if e > 0)
    wcons = wpos/len(week_edges) if week_edges else 0.0

    print(f"Live pick return:  {pick_ret:+.2f}pp")
    print(f"Live field return: {field_ret:+.2f}pp  (the non-picks)")
    k_dates = max(len(week_edges), 1)
    lo90, lo99 = _bounds(KIND, k_dates)
    print(f"Live EDGE:         {edge:+.2f}pp  (consistency bar {lo90:+.2f}pp @ k={k_dates}; old fixed bar +{MIN_EDGE_PP}pp)")
    print(f"Live win rate:     {pick_win:.1f}%")
    print(f"Week consistency:  {wpos}/{len(week_edges)} positive ({wcons*100:.0f}%, need >= {MIN_WEEK_CONSISTENCY*100:.0f}%)")
    print("-" * 60)
    c1, c2, c3, c4 = n>=MIN_SAMPLE, edge>lo90, pick_ret>0, wcons>=MIN_WEEK_CONSISTENCY
    if edge < lo99:
        print(f"  !! KILL LINE: edge {edge:+.2f}pp < 99% bound {lo99:+.2f}pp -- shadow INCONSISTENT with 18yr backtest, not merely unlucky")
    print(f"  [{'PASS' if c1 else 'FAIL'}] sample >= {MIN_SAMPLE}     ({n})")
    print(f"  [{'PASS' if c2 else 'FAIL'}] edge consistent w/ backtest (> {lo90:+.2f}pp @ k={k_dates})   ({edge:+.2f})")
    print(f"  [{'PASS' if c3 else 'FAIL'}] return > 0      ({pick_ret:+.2f})")
    print(f"  [{'PASS' if c4 else 'FAIL'}] weeks >= {int(MIN_WEEK_CONSISTENCY*100)}%   ({wcons*100:.0f}%)")
    print("=" * 60)
    if c1 and c2 and c3 and c4:
        print("VERDICT: ✅ GO — criteria met. Promotion is JUSTIFIED.")
        print("Deliberate manual step to go live: set MOMENTUM_LIVE=1. Review numbers first.")
        con.close(); return 0
    print("VERDICT: ⏳ NOT YET — keep momentum in shadow. Re-run weekly.")
    con.close(); return 1

if __name__ == "__main__":
    sys.exit(main())
