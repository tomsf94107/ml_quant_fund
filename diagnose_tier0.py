#!/usr/bin/env python3
"""
diagnose_tier0.py — READ-ONLY forensics for the five Tier-0 repairs.

Writes nothing; every database opens mode=ro. Run and paste:

    python diagnose_tier0.py > tier0_$(date +%Y%m%d).txt 2>&1

T0.1  prob_up = 0.0        which overlay multiplier zeros it (the predictions
                           table carries prob_raw + 7 mults + overlay_reason,
                           so the guilty column names itself)
T0.2  intraday outcomes    dead since 2026-06-22 while predictions continue
T0.3  fear_greed           frozen value, since when, and whether fg_mult moves
T0.4  ff_factors_daily     fetch_log evidence (French posts monthly, ~1mo lag,
                           so a 2026-05-29 max in late August is a fetch fault)
T0.5  momentum shadow      predictions written after the book was killed
"""
import os
import sqlite3
from datetime import datetime

HOME = os.path.expanduser("~/Desktop/ML_Quant_Fund")


def ro(db):
    return sqlite3.connect(f"file:{os.path.join(HOME, db)}?mode=ro", uri=True)


def q(con, sql, args=()):
    try:
        return con.execute(sql, args).fetchall()
    except Exception as e:
        return [("ERROR", str(e)[:120])]


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


MULTS = ["risk_mult", "sent_mult", "regime_mult", "options_mult",
         "squeeze_mult", "intraday_mult", "fg_mult"]


def t01_prob_zero():
    hdr("T0.1  prob_up = 0.0 forensics")
    con = ro("accuracy.db")

    print("--- exact-0.0 rows by month/horizon (first appearance) ---")
    for r in q(con, """SELECT substr(prediction_date,1,7) m, horizon, COUNT(*)
        FROM predictions WHERE prob_up = 0.0
        GROUP BY m, horizon ORDER BY m, horizon"""):
        print("   ", r)

    print("\n--- earliest 0.0 rows (date, ticker, horizon) ---")
    for r in q(con, """SELECT prediction_date, ticker, horizon FROM predictions
        WHERE prob_up = 0.0 ORDER BY prediction_date, ticker LIMIT 8"""):
        print("   ", r)

    print("\n--- the chain on 0.0 rows: was the RAW model output already 0? ---")
    for r in q(con, """SELECT
        SUM(prob_raw IS NULL) raw_null, SUM(prob_raw = 0.0) raw_zero,
        SUM(prob_raw > 0.0) raw_pos,
        ROUND(AVG(CASE WHEN prob_raw > 0 THEN prob_raw END),3) raw_avg_when_pos,
        SUM(prob_eff_uncapped = 0.0) eff_zero, SUM(gate_block = 1) gated,
        SUM(overlay_downgraded = 1) downgraded
        FROM predictions WHERE prob_up = 0.0"""):
        print("   raw_null,raw_zero,raw_pos,raw_avg_when_pos,eff_zero,gated,downgraded")
        print("   ", r)

    print("\n--- each multiplier ON THE 0.0 ROWS (nulls, zeros, min/avg) ---")
    for m in MULTS:
        r = q(con, f"""SELECT SUM({m} IS NULL), SUM({m} = 0.0),
            ROUND(MIN({m}),3), ROUND(AVG({m}),3), ROUND(MAX({m}),3)
            FROM predictions WHERE prob_up = 0.0""")[0]
        print(f"   {m:<15} null={r[0]}  zero={r[1]}  min={r[2]}  avg={r[3]}  max={r[4]}")

    print("\n--- same multipliers on AUGUST rows where prob_up > 0 (control) ---")
    for m in MULTS:
        r = q(con, f"""SELECT SUM({m} IS NULL), SUM({m} = 0.0),
            ROUND(MIN({m}),3), ROUND(AVG({m}),3)
            FROM predictions WHERE prob_up > 0 AND prediction_date >= '2026-08'""")[0]
        print(f"   {m:<15} null={r[0]}  zero={r[1]}  min={r[2]}  avg={r[3]}")

    print("\n--- overlay_reason on 0.0 rows, top 10 ---")
    for r in q(con, """SELECT COALESCE(overlay_reason,'(null)'), COUNT(*)
        FROM predictions WHERE prob_up = 0.0
        GROUP BY overlay_reason ORDER BY 2 DESC LIMIT 10"""):
        print("   ", r)

    print("\n--- do the SAME ticker-dates have sane prob_raw elsewhere? sample ---")
    for r in q(con, """SELECT prediction_date, ticker, horizon, prob_raw, prob_up,
        risk_mult, sent_mult, regime_mult, options_mult, squeeze_mult,
        intraday_mult, fg_mult, gate_block, overlay_reason
        FROM predictions WHERE prob_up = 0.0
        ORDER BY prediction_date DESC LIMIT 5"""):
        print("   ", r)
    con.close()


def t03_fear_greed():
    hdr("T0.3  fear_greed freeze")
    con = ro("accuracy.db")
    print("--- distinct values by month ---")
    for r in q(con, """SELECT substr(prediction_date,1,7) m,
        COUNT(DISTINCT fear_greed) dv, ROUND(MIN(fear_greed),1),
        ROUND(MAX(fear_greed),1), SUM(fear_greed IS NULL)
        FROM prediction_features GROUP BY m ORDER BY m"""):
        print("   ", r)
    print("\n--- last date fear_greed CHANGED ---")
    for r in q(con, """SELECT MAX(prediction_date) FROM prediction_features pf
        WHERE EXISTS (SELECT 1 FROM prediction_features p2
            WHERE p2.prediction_date < pf.prediction_date
            AND p2.fear_greed != pf.fear_greed)"""):
        print("   ", r)
    print("\n--- does fg_mult still move while the feature is frozen? ---")
    for r in q(con, """SELECT substr(prediction_date,1,7) m,
        COUNT(DISTINCT fg_mult), ROUND(MIN(fg_mult),3), ROUND(MAX(fg_mult),3)
        FROM predictions WHERE prediction_date >= '2026-05'
        GROUP BY m ORDER BY m"""):
        print("   ", r)
    con.close()


def t02_t05_outcome_writers():
    hdr("T0.2 / T0.5  outcome writers")
    con = ro("accuracy.db")
    print("--- intraday: last 3 outcomes vs last 3 predictions ---")
    for r in q(con, "SELECT prediction_ts, ticker, horizon_hr FROM "
                    "intraday_outcomes ORDER BY prediction_ts DESC LIMIT 3"):
        print("   outcome ", r)
    for r in q(con, "SELECT prediction_ts, ticker, horizon_hr FROM "
                    "intraday_predictions ORDER BY prediction_ts DESC LIMIT 3"):
        print("   pred    ", r)
    for r in q(con, """SELECT COUNT(*) FROM intraday_predictions
        WHERE prediction_ts > (SELECT MAX(prediction_ts) FROM intraday_outcomes)"""):
        print("   predictions never scored:", r[0][0] if r and r[0] else r)

    print("\n--- momentum shadow: activity after the book was killed (2026-08-25) ---")
    for r in q(con, """SELECT substr(prediction_date,1,7) m, COUNT(*)
        FROM momentum_shadow_predictions GROUP BY m ORDER BY m"""):
        print("   preds ", r)
    for r in q(con, """SELECT substr(prediction_date,1,7) m, COUNT(*)
        FROM momentum_shadow_outcomes GROUP BY m ORDER BY m"""):
        print("   outs  ", r)
    con.close()


def t04_ff_factors():
    hdr("T0.4  ff_factors_daily fetch trail")
    con = ro("prices.db")
    cols = [r[1] for r in q(con, 'PRAGMA table_info("fetch_log")')]
    print("fetch_log columns:", cols)
    print("--- last 12 fetch_log rows ---")
    for r in q(con, 'SELECT * FROM fetch_log ORDER BY rowid DESC LIMIT 12'):
        print("   ", r)
    print("\n--- ff_factors_daily tail ---")
    for r in q(con, "SELECT date, mkt_rf, smb, hml, rf FROM ff_factors_daily "
                    "ORDER BY date DESC LIMIT 3"):
        print("   ", r)
    con.close()


def main():
    print(f"TIER-0 DIAGNOSTICS  {datetime.now():%Y-%m-%d %H:%M} local  (read-only)")
    for fn in (t01_prob_zero, t03_fear_greed, t02_t05_outcome_writers,
               t04_ff_factors):
        try:
            fn()
        except Exception as e:
            print(f"\n!! {fn.__name__} failed: {type(e).__name__}: {e}")
    print("\nEND")


if __name__ == "__main__":
    main()
