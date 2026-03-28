#!/usr/bin/env python3
"""
Daily health check — runs at 1 PM Vietnam (2 AM ET)
Checks that last night's pipeline ran correctly.
"""
import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "accuracy.db"

def get_last_trading_date():
    """Return yesterday, or Friday if today is Monday."""
    today = datetime.now().date()
    if today.weekday() == 0:  # Monday
        return today - timedelta(days=3)
    return today - timedelta(days=1)

def check(label, passed, detail=""):
    status = "✅" if passed else "❌"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed

def main():
    last_date = get_last_trading_date()
    today = datetime.now().date()
    all_ok = True

    print("=" * 60)
    print(f"  ML Quant Fund — Daily Health Check")
    print(f"  Checking pipeline run for: {last_date}")
    print(f"  Health check time: {datetime.now().strftime('%Y-%m-%d %H:%M')} VN")
    print("=" * 60)

    con = sqlite3.connect(DB)

    # 1. Predictions populated
    n_pred = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction_date=?",
        (str(last_date),)
    ).fetchone()[0]
    ok = check("Predictions", n_pred >= 300, f"{n_pred} rows for {last_date} (expect ~303)")
    all_ok = all_ok and ok

    # 2. prediction_features populated
    n_feat = con.execute(
        "SELECT COUNT(*) FROM prediction_features WHERE prediction_date=?",
        (str(last_date),)
    ).fetchone()[0]
    ok = check("prediction_features", n_feat >= 300, f"{n_feat} rows for {last_date} (expect ~303)")
    all_ok = all_ok and ok

    # 3. Outcomes reconciled (previous day should have outcomes by now)
    n_out = con.execute(
        "SELECT COUNT(*) FROM outcomes WHERE prediction_date=?",
        (str(last_date),)
    ).fetchone()[0]
    ok = check("Outcomes reconciled", n_out >= 50, f"{n_out} rows for {last_date}")
    all_ok = all_ok and ok

    # 4. Retrain log is fresh
    retrain_log = ROOT / "logs" / "retrain.log"
    if retrain_log.exists():
        mtime = datetime.fromtimestamp(retrain_log.stat().st_mtime).date()
        ok = check("Retrain log", mtime >= last_date, f"last modified {mtime}")
    else:
        ok = check("Retrain log", False, "log file not found")
    all_ok = all_ok and ok

    # 5. daily_runner log is fresh
    runner_log = ROOT / "logs" / "daily_runner.log"
    if runner_log.exists():
        mtime = datetime.fromtimestamp(runner_log.stat().st_mtime).date()
        ok = check("daily_runner log", mtime >= last_date, f"last modified {mtime}")
    else:
        ok = check("daily_runner log", False, "log file not found")
    all_ok = all_ok and ok

    # 6. Accuracy check — last 3 days average
    rows = con.execute("""
        SELECT p.prediction_date,
               ROUND(100.0*SUM(CASE WHEN (p.prob_up>0.5 AND o.actual_return>0)
                                      OR (p.prob_up<=0.5 AND o.actual_return<0)
                               THEN 1 ELSE 0 END)/COUNT(*), 1) as acc
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
                       AND p.prediction_date=o.prediction_date
                       AND p.horizon=o.horizon
        WHERE p.prediction_date >= date('now', '-5 days')
        GROUP BY p.prediction_date
        ORDER BY p.prediction_date DESC
        LIMIT 3
    """).fetchall()
    if rows:
        avg_acc = sum(r[1] for r in rows) / len(rows)
        detail = " | ".join(f"{r[0]}: {r[1]}%" for r in rows)
        ok = check("Accuracy (3d avg)", avg_acc >= 45, f"{avg_acc:.1f}% | {detail}")
        all_ok = all_ok and ok
    else:
        check("Accuracy", False, "no outcomes to score")
        all_ok = False

    # 7. Intraday predictions populated (skip weekends)
    import datetime as dt
    today_wd = datetime.now().weekday()  # 0=Mon, 6=Sun
    if today_wd < 5:  # only check on weekdays
        n_intra = con.execute(
            "SELECT COUNT(DISTINCT ticker) FROM intraday_predictions WHERE prediction_date=?",
            (str(last_date),)
        ).fetchone()[0]
        ok = check("Intraday predictions", n_intra >= 90, f"{n_intra} tickers for {last_date} (expect ~101)")
        all_ok = all_ok and ok

    con.close()

    print("=" * 60)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — system healthy")
        # Desktop notification
        os.system('osascript -e \'display notification "All checks passed" with title "ML Quant Fund ✅ Healthy"\'')
    else:
        print("  ❌ ISSUES DETECTED — review above")
        os.system('osascript -e \'display notification "Pipeline issues detected — check logs" with title "ML Quant Fund ❌ Alert"\'')
    print("=" * 60)

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
