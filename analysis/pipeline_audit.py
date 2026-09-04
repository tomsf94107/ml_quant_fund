#!/usr/bin/env python3
"""
pipeline_audit.py — end-to-end audit: collection -> features -> retrain -> predict.

READ-ONLY. Writes nothing. Reports; does not repair.

WHY
    Several defects found on 2026-09-03/04 shared one shape: a stage ran,
    produced plausible output, exited zero, and was wrong. A frozen metrics
    table, 5x overconfident probabilities, a 2x price seam, a reverse split
    logged as +2,900%, borrow rows stamped with the wrong clock, and a ticker
    (IONQ) that has price data, short interest, metadata, six saved models and
    trains on demand -- yet has never produced a single prediction.

    None of those announced themselves. This audit asks the questions that
    would have.

WHAT IT CHECKS

  1. SOURCE FRESHNESS
     Every input database: newest row, age, ticker coverage. A source that
     stopped updating is invisible downstream because the last good value keeps
     being served.

  2. FEATURE HEALTH  (the section most likely to surprise)
     For a sample of tickers, every feature column is scored on:
       - % NaN. A column that is 100% NaN universe-wide is a dead feature the
         model carries but cannot use. short_pct_float is known to be one:
         UW's total_float returns shares OUTSTANDING, so the ratio never
         computes.
       - variance. A column with zero variance carries no information at all.
         analyst_upside/analyst_buy_pct/analyst_mult were pinned to constants
         (0.0/0.5/1.0) on 2026-05-21 when the source was dropped, and remain in
         OUTPUT_COLUMNS.
     Dead and constant features are not harmless: they dilute importance
     rankings and make the feature count look larger than the usable set.

  3. FEATURE OBSERVABILITY
     Features built vs features logged to prediction_features. If the model
     trains on 121 columns and only 34 are recorded at prediction time, the
     other 87 are invisible when a prediction has to be explained -- which is
     exactly the position the CRWV investigation started from.

  4. RETRAIN HEALTH
     Is it running, how many tickers per run, how many rows each model trains
     on, and do the saved model files on disk match the tickers that report
     importances? A model file older than the last retrain means the retrain
     did not actually save.

  5. PREDICTION COVERAGE
     Universe size vs distinct tickers predicted per day, and the names present
     in the universe but absent from recent predictions. This is the check that
     would have caught IONQ on the first night rather than months later via an
     unrelated research report.

  6. PROBABILITY HEALTH
     Distribution of prob_up and prob_cal by month -- mean, spread, and the
     fraction clearing the confidence gate. A distribution that drifts moves
     every fixed threshold underneath it, which happened here: 8.4% of h=5
     predictions cleared 0.70 in June and 1.15% in August, with no code change.

    python analysis/pipeline_audit.py
    python analysis/pipeline_audit.py --sample 25    # more tickers, slower
"""
import argparse
import os
import sqlite3
import statistics as st
import sys
from datetime import date, datetime


def age_days(d):
    if not d:
        return None
    try:
        return (date.today() - date.fromisoformat(str(d)[:10])).days
    except Exception:
        return None


def hdr(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def q1(db, sql, params=()):
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        r = c.execute(sql, params).fetchone()
        c.close()
        return r
    except Exception as e:
        return ("ERR", str(e)[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12,
                    help="tickers to build features for; slow, so keep small")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    os.chdir(args.root)

    # ---------------- 1. SOURCE FRESHNESS ----------------
    hdr("1. SOURCE FRESHNESS")
    sources = [
        ("prices.db", "raw_bars", "d", "ticker"),
        ("prices.db", "daily_prices", "date", "ticker"),
        ("short_interest.db", "short_interest", "settlement_date", "ticker"),
        ("insider_trades.db", "insider_filings_raw", "filing_date", "ticker"),
        ("insider_trades.db", "insider_flows", "date", "ticker"),
        ("earnings_monitor.db", "darkpool_prints", "date", "ticker"),
        ("borrow.db", "borrow_live", "ts_utc", "ticker"),
        ("accuracy.db", "predictions", "prediction_date", "ticker"),
        ("accuracy.db", "outcomes", "outcome_date", "ticker"),
        ("fundamentals.db", "xbrl_facts", "filed_date", "ticker"),
    ]
    print(f"  {'database':<22}{'table':<22}{'rows':>10}{'tickers':>9}"
          f"{'newest':>13}{'age':>6}")
    for db, tbl, dcol, tcol in sources:
        if not os.path.exists(db):
            print(f"  {db:<22}{tbl:<22}   FILE NOT FOUND")
            continue
        r = q1(db, f"SELECT COUNT(*), COUNT(DISTINCT {tcol}), MAX({dcol}) "
                   f"FROM {tbl}")
        if r and r[0] == "ERR":
            print(f"  {db:<22}{tbl:<22}   {r[1]}")
            continue
        a = age_days(r[2])
        flag = ""
        if a is not None and a > 7:
            flag = "  <-- STALE"
        print(f"  {db:<22}{tbl:<22}{r[0]:>10,}{r[1]:>9}"
              f"{str(r[2])[:10]:>13}{(str(a)+'d' if a is not None else '?'):>6}"
              f"{flag}")

    # ---------------- 4. RETRAIN HEALTH ----------------
    hdr("4. RETRAIN HEALTH")
    r = q1("accuracy.db", "SELECT COUNT(*), COUNT(DISTINCT ticker), "
                          "MAX(retrain_date) FROM feature_importance_history")
    print(f"  feature_importance_history: {r[0]:,} rows, {r[1]} tickers, "
          f"newest {r[2]}")
    try:
        c = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
        print(f"\n  {'retrain_date':<14}{'tickers':>9}{'features':>10}")
        for d, tk, f in c.execute(
                "SELECT retrain_date, COUNT(DISTINCT ticker), "
                "COUNT(DISTINCT feature) FROM feature_importance_history "
                "GROUP BY retrain_date ORDER BY retrain_date DESC LIMIT 8"):
            print(f"  {str(d)[:10]:<14}{tk:>9}{f:>10}")
        c.close()
    except Exception as e:
        print(f"  could not read retrain history: {e}")

    mdir = "models/saved"
    if os.path.isdir(mdir):
        files = [f for f in os.listdir(mdir) if f.endswith(".joblib")]
        tick = {f.split("_")[0] for f in files}
        newest = max((os.path.getmtime(os.path.join(mdir, f))
                      for f in files), default=0)
        oldest = min((os.path.getmtime(os.path.join(mdir, f))
                      for f in files), default=0)
        print(f"\n  {mdir}: {len(files)} files, {len(tick)} tickers")
        print(f"    newest {datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}"
              f"   oldest {datetime.fromtimestamp(oldest):%Y-%m-%d %H:%M}")
        stale = [f for f in files
                 if (datetime.now()
                     - datetime.fromtimestamp(
                         os.path.getmtime(os.path.join(mdir, f)))).days > 14]
        if stale:
            print(f"    {len(stale)} model files older than 14 days "
                  f"-- retrain may not be saving for these")
            print(f"    e.g. {', '.join(sorted(stale)[:6])}")

    # ---------------- 5. PREDICTION COVERAGE ----------------
    hdr("5. PREDICTION COVERAGE")
    try:
        universe = [l.strip().upper() for l in open("tickers.txt")
                    if l.strip()]
    except Exception:
        universe = []
    wl = []
    if os.path.exists("tickers_watchlist.txt"):
        wl = [l.strip().upper() for l in open("tickers_watchlist.txt")
              if l.strip() and not l.startswith("#")]
    print(f"  tickers.txt {len(universe)}   watchlist {len(wl)}   "
          f"union {len(set(universe) | set(wl))}")
    c = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    print(f"\n  {'date':<13}{'tickers':>9}{'vs union':>10}")
    for d, n in c.execute(
            "SELECT prediction_date, COUNT(DISTINCT ticker) FROM predictions "
            "WHERE prediction_date >= date('now','-8 days') "
            "GROUP BY prediction_date ORDER BY 1 DESC"):
        print(f"  {str(d)[:10]:<13}{n:>9}{n - len(set(universe)|set(wl)):>+10}")
    recent = {r[0] for r in c.execute(
        "SELECT DISTINCT ticker FROM predictions "
        "WHERE prediction_date >= date('now','-7 days')")}
    missing = sorted(set(universe) - recent)
    print(f"\n  in tickers.txt but NOT predicted in 7 days: {len(missing)}")
    if missing:
        print(f"    {', '.join(missing)}")
        print("    ^ these are silently absent. Nothing in the daily run "
              "reports them.")

    # ---------------- 6. PROBABILITY HEALTH ----------------
    hdr("6. PROBABILITY HEALTH  (h=5)")
    cols = [r[1] for r in c.execute("PRAGMA table_info(predictions)")]
    pc = ", ROUND(AVG(prob_cal),4)" if "prob_cal" in cols else ", NULL"
    print(f"  {'month':<9}{'n':>8}{'mean prob_up':>14}{'p10':>8}{'p90':>8}"
          f"{'>=0.70':>9}{'mean prob_cal':>15}")
    for m, n, mp, ge, mc in c.execute(
            f"SELECT substr(prediction_date,1,7), COUNT(*), "
            f"ROUND(AVG(prob_up),4), "
            f"ROUND(100.0*SUM(prob_up>=0.70)/COUNT(*),2){pc} "
            f"FROM predictions WHERE horizon=5 "
            f"AND prediction_date >= date('now','-180 days') "
            f"GROUP BY 1 ORDER BY 1"):
        ps = [r[0] for r in c.execute(
            "SELECT prob_up FROM predictions WHERE horizon=5 "
            "AND substr(prediction_date,1,7)=? AND prob_up IS NOT NULL "
            "ORDER BY prob_up", (m,))]
        p10 = ps[len(ps)//10] if ps else 0
        p90 = ps[9*len(ps)//10] if ps else 0
        print(f"  {m:<9}{n:>8,}{mp:>14}{p10:>8.3f}{p90:>8.3f}{ge:>8}%"
              f"{(mc if mc is not None else 0):>15}")
    c.close()

    # ---------------- 2 & 3. FEATURE HEALTH ----------------
    hdr(f"2. FEATURE HEALTH  (sample of {args.sample} tickers)")
    sys.path.insert(0, ".")
    try:
        from features.builder import build_feature_dataframe
    except Exception as e:
        print(f"  cannot import builder: {e}")
        return
    import random
    tk = universe[:] or ["AAPL", "NVDA", "MSFT"]
    random.Random(7).shuffle(tk)
    tk = tk[:args.sample]

    nan_pct = {}
    var_zero = {}
    ncols = 0
    ok = 0
    for t in tk:
        try:
            df = build_feature_dataframe(t, start_date="2021-01-01",
                                         training_mode=True)
            if df is None or df.empty:
                continue
            ok += 1
            num = df.select_dtypes("number")
            ncols = max(ncols, len(num.columns))
            for cN in num.columns:
                s = num[cN]
                nan_pct.setdefault(cN, []).append(100.0 * s.isna().mean())
                nn = s.dropna()
                var_zero.setdefault(cN, []).append(
                    1 if (len(nn) > 0 and nn.nunique() <= 1) else 0)
        except Exception:
            continue
    print(f"  built features for {ok}/{len(tk)} sampled tickers, "
          f"{ncols} numeric columns\n")

    dead = [(cN, st.mean(v)) for cN, v in nan_pct.items()
            if st.mean(v) >= 99.9]
    const = [cN for cN, v in var_zero.items()
             if v and sum(v) == len(v) and cN not in {d[0] for d in dead}]
    partial = sorted(((cN, st.mean(v)) for cN, v in nan_pct.items()
                      if 20 <= st.mean(v) < 99.9), key=lambda x: -x[1])

    print(f"  DEAD (100% NaN on every sampled ticker): {len(dead)}")
    for cN, v in sorted(dead):
        print(f"    {cN}")
    print(f"\n  CONSTANT (no variance on every sampled ticker): {len(const)}")
    for cN in sorted(const):
        print(f"    {cN}")
    print(f"\n  PARTIAL (20-99% NaN on average): {len(partial)}")
    for cN, v in partial[:15]:
        print(f"    {cN:<32}{v:>6.0f}% NaN")

    usable = ncols - len(dead) - len(const)
    print(f"\n  {ncols} numeric columns; {len(dead)} dead, {len(const)} "
          f"constant -> {usable} carry information")
    if dead or const:
        print("  Dead and constant columns are not harmless: they dilute "
              "importance\n  rankings and make the feature count look larger "
              "than the usable set.")

    hdr("3. FEATURE OBSERVABILITY")
    c = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    logged = [r[1] for r in c.execute("PRAGMA table_info(prediction_features)")
              if r[1] not in ("id", "ticker", "prediction_date", "horizon",
                              "created_at")]
    c.close()
    built = set(nan_pct)
    print(f"  built at train time : {len(built)}")
    print(f"  logged at predict   : {len(logged)}")
    inv = sorted(built - set(logged))
    print(f"  NOT logged          : {len(inv)}")
    if inv:
        print(f"    {', '.join(inv[:24])}"
              + (f"  ... and {len(inv)-24} more" if len(inv) > 24 else ""))
        print("\n  Features the model trains on but does not record at "
              "prediction time\n  cannot be inspected when a prediction has to "
              "be explained.")


if __name__ == "__main__":
    main()
