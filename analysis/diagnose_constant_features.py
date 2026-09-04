#!/usr/bin/env python3
"""
diagnose_constant_features.py — WHY is each dead/constant feature dead?

READ-ONLY. Writes nothing. Diagnoses before anything is repaired.

WHY DIAGNOSE FIRST
    pipeline_audit found 1 dead and 15 constant columns of 119. Fixing them
    one at a time would be wrong: they almost certainly share a small number of
    root causes, and several are constant BY DESIGN.

    Known or strongly suspected causes, to be confirmed here:

      DELIBERATE   analyst_upside/analyst_buy_pct/analyst_mult were pinned to
                   0.0/0.5/1.0 on 2026-05-21 when the source was dropped -- the
                   builder comment says "No free historical source; train/serve
                   mismatch eliminated by dropping." fear_greed was dropped the
                   same day. These are correct as they stand and should NOT be
                   "fixed"; they should be REMOVED from OUTPUT_COLUMNS so they
                   stop diluting importance rankings.

      BLOCKED FEED yfinance is disabled by XProtect on this machine. The feature
                   build prints "index symbol ^VIX3M: yfinance disabled
                   (XProtect block), returning empty" on every run.
                   vix_term_structure needs VIX3M. warning.db ALREADY HOLDS
                   CBOE_VIX3M with 4,261 rows back to 2009 -- so the data is
                   local and the feature is reaching for a blocked source
                   instead. That one is repairable without any new vendor.

      EMPTY SOURCE options-derived (iv_skew_snap, pc_ratio_snap,
                   expected_move_perc) and earnings-derived (is_earnings_week,
                   pre/post_earnings_drift, rev_surprise) columns may be
                   constant because their upstream table is empty or the lookup
                   silently returns a default.

      DEFAULT-ON-FAIL a feature whose except branch returns 0.0 looks identical
                   to a feature that is genuinely 0.0. This is the single most
                   dangerous pattern in the codebase and has appeared repeatedly
                   this week.

WHAT THIS PRINTS
    For each dead/constant column: the constant VALUE it takes, whether the
    builder assigns it a literal, which source table it reads, and whether that
    table has data. A column pinned to a literal is deliberate; a column that
    reads a table which turns out to be empty is broken.

    Grouping by cause is the point -- the fix list should be 3-4 items, not 16.

    python analysis/diagnose_constant_features.py
"""
import argparse
import os
import re
import sqlite3
import sys

TARGETS = [
    "short_pct_float", "analyst_buy_pct", "analyst_mult", "analyst_upside",
    "expected_move_perc", "fear_greed", "is_earnings_week", "is_squeeze_setup",
    "iv_skew_snap", "monday_sentiment", "pc_ratio_snap", "post_earnings_drift",
    "pre_earnings_drift", "rev_surprise", "vix_term_structure", "vol_x_short",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--builder", default="features/builder.py")
    args = ap.parse_args()

    src = open(args.builder).read() if os.path.exists(args.builder) else ""

    print("=" * 74)
    print("A. WHAT VALUE DOES EACH TAKE, AND IS IT A LITERAL IN THE BUILDER?")
    print("=" * 74)
    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    df = build_feature_dataframe(args.ticker, start_date="2021-01-01",
                                 training_mode=True)
    print(f"  {args.ticker}: {len(df)} rows\n")
    print(f"  {'feature':<24}{'value':>12}{'nunique':>9}  assignment in builder.py")
    for c in TARGETS:
        if c not in df.columns:
            print(f"  {c:<24}{'ABSENT':>12}")
            continue
        s = df[c]
        nn = s.dropna()
        val = f"{nn.iloc[0]:.4g}" if len(nn) else "all NaN"
        # find a literal assignment like  df["x"] = 0.0
        lit = re.search(rf'df\[\s*["\']{re.escape(c)}["\']\s*\]\s*=\s*'
                        r'([0-9.\-]+)\s*$', src, re.M)
        note = (f'LITERAL  df["{c}"] = {lit.group(1)}' if lit
                else "computed / assigned from a variable")
        print(f"  {c:<24}{val:>12}{nn.nunique():>9}  {note}")

    print("\n" + "=" * 74)
    print("B. DO THE UPSTREAM SOURCES ACTUALLY HAVE DATA?")
    print("=" * 74)
    checks = [
        ("warning.db", "SELECT COUNT(*), MIN(obs_date), MAX(obs_date) FROM "
                       "data_vintages WHERE series_id='CBOE_VIX3M'",
         "VIX3M for vix_term_structure -- LOCAL, not yfinance"),
        ("warning.db", "SELECT COUNT(*), MIN(obs_date), MAX(obs_date) FROM "
                       "data_vintages WHERE series_id='CBOE_VIX'", "VIX"),
        ("earnings.db", "SELECT COUNT(*), MIN(date(report_date)), "
                        "MAX(date(report_date)) FROM earnings",
         "earnings calendar -> is_earnings_week, drifts, surprises"),
        ("earnings_monitor.db", "SELECT COUNT(*) FROM sqlite_master "
                                "WHERE type='table'", "monitor tables"),
        ("short_interest.db", "SELECT COUNT(*), MIN(settlement_date), "
                              "MAX(settlement_date) FROM short_interest",
         "short interest -> vol_x_short, is_squeeze_setup"),
    ]
    for db, sql, why in checks:
        if not os.path.exists(db):
            print(f"  {db:<22} FILE NOT FOUND        {why}")
            continue
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            r = c.execute(sql).fetchone()
            c.close()
            print(f"  {db:<22}{str(r):<38}{why}")
        except Exception as e:
            print(f"  {db:<22}{('ERR: '+str(e)[:30]):<38}{why}")

    print("\n" + "=" * 74)
    print("C. BLOCKED-FEED EVIDENCE")
    print("=" * 74)
    print("  The feature build prints this on every run:")
    print("    index symbol ^VIX3M: yfinance disabled (XProtect block), "
          "returning empty")
    print("    index symbol ES=F:   yfinance disabled (XProtect block), "
          "returning empty")
    print("  Any feature depending on those symbols cannot compute, and if its")
    print("  except branch returns a default it will look CONSTANT rather than")
    print("  broken. Grep for the fallback:\n")
    for pat in ("VIX3M", "ES=F", "yfinance disabled"):
        hits = [f"    line {i+1}: {l.strip()[:88]}"
                for i, l in enumerate(src.splitlines()) if pat in l]
        print(f"  '{pat}' in builder.py: {len(hits)} hit(s)")
        for h in hits[:4]:
            print(h)

    print("\n" + "=" * 74)
    print("D. HOW TO READ THIS")
    print("=" * 74)
    print("  LITERAL assignment  -> deliberate. Do not repair; REMOVE from")
    print("                         OUTPUT_COLUMNS so it stops diluting")
    print("                         importance rankings and inflating the")
    print("                         feature count.")
    print("  computed + source HAS data -> the lookup or the join is broken.")
    print("                         Repairable, and the highest-value case.")
    print("  computed + source EMPTY    -> a data-collection gap, not a code")
    print("                         bug. Fix the ingest first.")
    print("  blocked feed with a local alternative -> repoint the feature.")
    print("                         vix_term_structure is this case: VIX3M is")
    print("                         in warning.db and the builder asks yfinance.")


if __name__ == "__main__":
    main()
