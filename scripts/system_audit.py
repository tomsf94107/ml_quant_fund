"""
scripts/system_audit.py - one-shot systematic health check, every known subsystem.
Jun 12 2026. PRINTS PASS/FAIL per check; fix list at end.
Run: python3 -m scripts.system_audit
"""
import os, sqlite3, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        ISSUES.append(f"{name}: {detail}")


def q(db, sql):
    try:
        conn = sqlite3.connect(str(ROOT / db))
        r = conn.execute(sql).fetchall()
        conn.close()
        return r
    except Exception as e:
        return [("ERR", repr(e))]


def main():
    today = datetime.now().strftime("%Y%m%d")
    print("=" * 64)
    print(" SYSTEM AUDIT", datetime.now().isoformat(timespec="seconds"))
    print("=" * 64)

    print("\n[1] UNIVERSE FILES")
    tk = [l.strip() for l in open(ROOT / "tickers.txt") if l.strip()]
    bad = [t for t in tk if t.startswith("#") or " " in t or len(t) > 6]
    check("tickers.txt clean", len(bad) == 0, f"{len(tk)} lines, bad: {bad[:5]}")
    import csv
    meta_rows = list(csv.DictReader(open(ROOT / "tickers_metadata.csv")))
    meta_t = {r[list(r.keys())[0]].upper() for r in meta_rows}
    missing_meta = [t for t in tk if t.upper() not in meta_t]
    check("metadata covers universe", len(missing_meta) == 0, f"missing: {missing_meta[:8]}")

    print("\n[2] PIPELINE MARKERS + LOGS (last 2 days)")
    markers = sorted((ROOT / "logs").glob(".pipeline_A_done_*"))
    last_a = markers[-1].name.split("_")[-1] if markers else "NONE"
    check("Pipeline A marker recent", last_a >= (datetime.now() - timedelta(days=2)).strftime("%Y%m%d"),
          f"last marker {last_a}")
    for p in ("A", "B", "C", "D"):
        dirs = sorted((ROOT / "logs").glob(f"pipeline_{p}_2*"))
        last = dirs[-1].name if dirs else "NONE"
        errs = 0
        if dirs:
            for lf in dirs[-1].glob("*.log"):
                try:
                    errs += sum(1 for L in open(lf, errors="ignore")
                                if "ERROR" in L or "FATAL" in L or "Traceback" in L)
                except Exception:
                    pass
        check(f"Pipeline {p} last run", bool(dirs), last)
        if dirs:
            check(f"Pipeline {p} error count ({dirs[-1].name})", errs < 20, f"{errs} error lines")

    print("\n[3] PREDICTIONS FRESHNESS + COVERAGE")
    r = q("accuracy.db", "SELECT MAX(prediction_date), COUNT(DISTINCT ticker) FROM predictions")
    check("predictions fresh", str(r[0][0]) >= (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
          f"max={r[0][0]}, tickers={r[0][1]}")
    r2 = q("accuracy.db",
           "SELECT COUNT(DISTINCT ticker) FROM predictions WHERE prediction_date=(SELECT MAX(prediction_date) FROM predictions)")
    check("latest-day ticker coverage vs 394", r2[0][0] >= 300, f"{r2[0][0]}/394 (new names pending B = expected until first full B run)")
    phantom = q("accuracy.db",
                "SELECT COUNT(*) FROM predictions WHERE ticker='CYBR' AND prediction_date >= '2026-06-11'")
    check("CYBR phantom stopped", phantom[0][0] == 0, f"{phantom[0][0]} rows since Jun 11")

    print("\n[4] MOMENTUM SHADOW (Jun 29 path)")
    r = q("accuracy.db",
          "SELECT MIN(prediction_date), MAX(prediction_date), COUNT(DISTINCT prediction_date), SUM(is_buy_candidate) FROM momentum_shadow_predictions")
    mn, mx, nd, buys = r[0]
    check("shadow logging current", str(mx) >= (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
          f"{mn} -> {mx}, {nd} nights, {buys} BUY cands")
    expected_nights = 0
    d = datetime.strptime(str(mn), "%Y-%m-%d")
    while d.strftime("%Y-%m-%d") <= str(mx):
        if d.weekday() < 5:
            expected_nights += 1
        d += timedelta(days=1)
    check("no missing shadow nights", nd >= expected_nights - 1, f"{nd} logged vs ~{expected_nights} trading days (1 known loss Jun 10)")

    print("\n[5] OUTCOMES INTEGRITY (zero-bug regression)")
    z = q("accuracy.db", "SELECT COUNT(*) FROM outcomes WHERE actual_return = 0.0")
    check("no exact-zero fake outcomes", z[0][0] == 0, f"{z[0][0]} rows")

    print("\n[6] DATA FEEDS FRESHNESS")
    f = q("fundamentals.db", "SELECT MAX(filed_date), COUNT(DISTINCT ticker) FROM xbrl_facts")
    check("fundamentals fresh+wide", f[0][1] >= 380, f"max filed {f[0][0]}, {f[0][1]} tickers")
    i = q("insider_trades.db", "SELECT MAX(filing_date), COUNT(DISTINCT ticker) FROM insider_filings_raw")
    check("insider raw current", str(i[0][0]) >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
          f"max {i[0][0]}, {i[0][1]} tickers (expansion crawl may still be running)")

    print("\n[7] FEATURE WIRING SMOKE (fundamentals in builder)")
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from features.fundamental_features import load_fundamental_features_pit
        import pandas as pd
        idx = pd.to_datetime(["2026-06-01"])
        df = load_fundamental_features_pit("AAPL", idx, db_path=ROOT / "fundamentals.db")
        check("fundamental loader live", df["fund_gp_assets"].notna().any() or df["fund_op_equity"].notna().any(),
              df.iloc[0].round(3).to_dict().__str__()[:80])
    except Exception as e:
        check("fundamental loader live", False, repr(e)[:80])

    print("\n[8] CRON SANITY")
    cr = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    for needle, label in [("pipeline_A_ingest", "cron A"), ("pipeline_B_train", "cron B"),
                          ("pipeline_C_preopen", "cron C"), ("pipeline_D", "cron D"),
                          ("etl_xbrl", "cron xbrl weekly"), ("weekly_insider", "cron insider weekly"),
                          ("recession", "cron recession weekly")]:
        check(label, needle in cr)

    print("\n[9] DISK/DB HYGIENE")
    twins = list(ROOT.glob("recession/recession.db")) 
    check("no empty recession twin", not any(t.stat().st_size == 0 for t in twins), str(twins))
    junk = list(ROOT.glob("* 2.db*"))
    check("no macOS dup files", len(junk) == 0, str([j.name for j in junk]))
    ahead = subprocess.run(["git", "rev-list", "--count", "origin/research-track..HEAD"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    check("commits pushed", ahead == "0", f"{ahead} unpushed")

    print("\n" + "=" * 64)
    if ISSUES:
        print(f" {len(ISSUES)} ISSUES TO FIX:")
        for i, s in enumerate(ISSUES, 1):
            print(f"  {i}. {s}")
    else:
        print(" ALL CHECKS PASS")
    print("=" * 64)


if __name__ == "__main__":
    main()
