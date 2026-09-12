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
    """
    Return the most recent prediction_date with a real run (>= 50 rows).
    Filters out test/sentinel rows from manual pipeline runs.
    """
    import sqlite3
    con = sqlite3.connect(DB, timeout=30)
    result = con.execute("""
        SELECT prediction_date FROM predictions
        GROUP BY prediction_date
        HAVING COUNT(*) >= 50
        ORDER BY prediction_date DESC
        LIMIT 1
    """).fetchone()
    result = result[0] if result else None
    con.close()
    if result:
        from datetime import date
        return date.fromisoformat(result)
    # Fallback
    today = datetime.now().date()
    return today - timedelta(days=1)

_FAILURES: list = []


def check(label, passed, detail=""):
    status = "✅" if passed else "❌"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not passed:
        _FAILURES.append(label)
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

    con = sqlite3.connect(DB, timeout=30)

    # 1. Predictions populated
    n_pred = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction_date=?",
        (str(last_date),)
    ).fetchone()[0]
    # Universe-derived, not a frozen constant: "expect ~303" dated from a
    # smaller universe; the panel is now ~399 tickers x 3 horizons.
    _uni = con.execute(
        "SELECT COUNT(DISTINCT ticker) FROM predictions WHERE prediction_date=?",
        (str(last_date),)).fetchone()[0] or 0
    _hz = con.execute(
        "SELECT COUNT(DISTINCT horizon) FROM predictions WHERE prediction_date=?",
        (str(last_date),)).fetchone()[0] or 1
    _exp = _uni * _hz
    ok = check("Predictions", n_pred >= max(300, int(_exp * 0.95)),
               f"{n_pred} rows for {last_date} ({_uni} tickers x {_hz} horizons, expect ~{_exp})")
    all_ok = all_ok and ok

    # 2. prediction_features populated
    n_feat = con.execute(
        "SELECT COUNT(*) FROM prediction_features WHERE prediction_date=?",
        (str(last_date),)
    ).fetchone()[0]
    ok = check("prediction_features", n_feat >= max(300, int(_exp * 0.95)),
               f"{n_feat} rows for {last_date} (expect ~{_exp}; "
               f"{_exp - n_feat} short)" if n_feat < _exp else
               f"{n_feat} rows for {last_date} (expect ~{_exp})")
    all_ok = all_ok and ok

    # 3. Outcomes reconciled -- MATURITY-AWARE (Aug 1 2026).
    # WAS: counted outcomes for last_date, the PREDICTION RUN date. An h=1
    # outcome needs the NEXT session's close, so that row cannot exist yet by
    # construction -- the check only passed when the pipeline happened to lag.
    # Result: 77 "ISSUES DETECTED" lines in logs/health_check.log, i.e. the
    # health check cried wolf routinely, which is why nobody read it, which is
    # how four dead feeds went unnoticed for six days (Jul 26 - Aug 1).
    # NOW: check the newest prediction date that has had time to mature at
    # h=1 -- one completed session before the latest prediction date.
    _mature = con.execute(
        "SELECT MAX(prediction_date) FROM predictions WHERE prediction_date < ?",
        (str(last_date),)
    ).fetchone()[0]
    if _mature:
        n_out = con.execute(
            "SELECT COUNT(*) FROM outcomes WHERE prediction_date=? AND horizon=1",
            (str(_mature),)
        ).fetchone()[0]
        ok = check("Outcomes reconciled", n_out >= 50,
                   f"{n_out} h=1 rows for {_mature} (latest MATURED date; "
                   f"{last_date} cannot have outcomes yet -- h=1 needs the next close)")
    else:
        ok = check("Outcomes reconciled", True, "no prior prediction date to mature yet")
    all_ok = all_ok and ok

    # 4. Retrain log is fresh
    # Find newest 02_train_all.log across all pipeline_B_* folders
    pipeline_b_logs = sorted((ROOT / "logs").glob("pipeline_B_*/02_train_all.log"))
    if pipeline_b_logs:
        retrain_log = pipeline_b_logs[-1]
        mtime = datetime.fromtimestamp(retrain_log.stat().st_mtime).date()
        ok = check("Retrain log", mtime >= last_date - timedelta(days=1),
                   f"last modified {mtime} ({retrain_log.parent.name})")
    else:
        ok = check("Retrain log", False, "no pipeline_B logs found")
    all_ok = all_ok and ok

    # 5. daily_runner log is fresh
    # Newest daily_runner output across pipeline B (stage 3) and pipeline C (stage 2)
    runner_candidates = (
        list((ROOT / "logs").glob("pipeline_B_*/03_daily_runner.log")) +
        list((ROOT / "logs").glob("pipeline_C_*/02_daily_runner.log"))
    )
    if runner_candidates:
        runner_log = max(runner_candidates, key=lambda p: p.stat().st_mtime)
        mtime = datetime.fromtimestamp(runner_log.stat().st_mtime).date()
        ok = check("daily_runner log", mtime >= last_date,
                   f"last modified {mtime} ({runner_log.parent.name})")
    else:
        ok = check("daily_runner log", False, "no pipeline_B/C runner logs found")
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

    # ── FRESHNESS CHECKS (2026-09-11) ──────────────────────────────────────
    # Six defects in three days, none of which raised an exception: a crontab
    # placeholder API key that 401'd every Massive call while h40_shadow still
    # logged a top-3 indistinguishable from a good run; a wide-universe price
    # fetch that was never scheduled at all; the warning chain running inside
    # its own pub_date window; pipecheck reading ET dates against VN-written
    # logs; sector ETFs absent from the wrapper, leaving S7 comparing an
    # 11-day-old numerator to a live denominator; and a hardcoded share count
    # stale after a 4:1 split, printing 273.8% institutional ownership.
    #
    # Every one produced normal-shaped output. The common cause was a
    # hand-maintained value or job with nothing checking it still matched
    # reality, so these three check exactly that.
    #
    # Thresholds are measured, not chosen -- per the comment above about 77
    # consecutive false failures teaching everyone to ignore this file.
    import sqlite3 as _sq
    _PRICES = ROOT / "prices.db"
    _WARN = ROOT / "warning.db"

    # (a) raw_bars breadth. Measured over 60 sessions: min 1,991, median 2,000,
    # max 2,004. The two truncations found on 2026-09-10 were 444 and 863. A
    # floor of 1,500 sits in a ~700-wide empty gap, and anything from roughly
    # 900 to 1,900 gives the identical answer -- insensitive, not fitted.
    RAW_BARS_MIN = 1_500
    try:
        _c = _sq.connect(f"file:{_PRICES}?mode=ro", uri=True, timeout=30)
        _d, _n = _c.execute(
            "SELECT d, COUNT(*) FROM raw_bars GROUP BY d ORDER BY d DESC LIMIT 1"
        ).fetchone()
        _c.close()
        ok = check("raw_bars breadth", _n >= RAW_BARS_MIN,
                   f"{_n:,} tickers for {_d} (floor {RAW_BARS_MIN:,}; normal ~2,000)"
                   + ("" if _n >= RAW_BARS_MIN else
                      " -- wide-universe fetch did not complete; check "
                      "logs/universe_fetch_daily.log and the 10:00 VN job"))
        all_ok = all_ok and ok
    except Exception as _e:
        all_ok = check("raw_bars breadth", False, f"could not read prices.db: {_e}") and all_ok

    # (b) SPY_CLOSE recency. The warning chain's driver derives asof_date from
    # this series; if it stops advancing the composite silently freezes on an
    # old session. Two sessions of slack covers a normal one-session lag plus
    # a holiday.
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from utils.market_calendar import last_completed_session, previous_trading_day
        _c = _sq.connect(f"file:{_WARN}?mode=ro", uri=True, timeout=30)
        _obs = _c.execute("SELECT MAX(obs_date) FROM data_vintages "
                          "WHERE series_id='SPY_CLOSE'").fetchone()[0]
        _c.close()
        _floor = previous_trading_day(last_completed_session()).isoformat()
        ok = check("SPY_CLOSE recency", bool(_obs) and _obs >= _floor,
                   f"latest obs {_obs}, last completed session "
                   f"{last_completed_session()}"
                   + ("" if _obs and _obs >= _floor else
                      " -- ingest_spx has not run; the CEWS driver will refuse "
                      "to write (FEED_STALE) until it does"))
        all_ok = all_ok and ok
    except Exception as _e:
        all_ok = check("SPY_CLOSE recency", False, f"could not read warning.db: {_e}") and all_ok

    # (c) short_interest.db currency. This became load-bearing on 2026-09-12:
    # the monitor now prefers FINRA over UW for short interest whenever FINRA
    # is ahead, because UW lags by a full settlement and the stale row flagged
    # the PRIOR period's direction -- RDDT showed "covering wave -12.2%" when
    # 8/14 -> 8/31 was a +19.9% build. If this fetch stops, the preference
    # silently reverses and nothing says so.
    #
    # Settlements are semi-monthly and FINRA publishes ~8 business days later,
    # so the age of the newest settlement cycles between roughly 8 and 25 days.
    # 30 clears both ends. si_fetch_v2 logs ten ConnectionResetError(54) AUTH
    # ERRORs interleaved with successes -- the feed is flaky and self-recovers
    # on its every-third-day schedule, so this must not fire on one bad run.
    SI_MAX_AGE_DAYS = 30
    try:
        _c = _sq.connect(f"file:{ROOT / 'short_interest.db'}?mode=ro", uri=True, timeout=30)
        _sd, _age = _c.execute(
            "SELECT MAX(settlement_date), "
            "julianday('now') - julianday(MAX(settlement_date)) FROM short_interest"
        ).fetchone()
        _c.close()
        ok = check("short_interest currency", _age is not None and _age <= SI_MAX_AGE_DAYS,
                   f"newest settlement {_sd}, {_age:.0f} days old (limit "
                   f"{SI_MAX_AGE_DAYS}; normal cycle 8-25)"
                   + ("" if _age is not None and _age <= SI_MAX_AGE_DAYS else
                      " -- si_fetch_v2 has stopped; the monitor will fall back "
                      "to UW, which lags a full settlement and flags the prior "
                      "period's direction"))
        all_ok = all_ok and ok
    except Exception as _e:
        all_ok = check("short_interest currency", False,
                       f"could not read short_interest.db: {_e}") and all_ok

    # (d) Crontab snapshot age. The tracked copy drifted 261 lines behind the
    # live crontab because nothing regenerated it. A weekly job now writes it
    # Sundays 11:00 VN; 8 days is one cycle plus slack.
    try:
        import time as _t
        _snap = ROOT / "scripts" / "crontab_VN_anchored.txt"
        _age = (_t.time() - os.path.getmtime(_snap)) / 86400
        ok = check("crontab snapshot", _age <= 8.0,
                   f"{_age:.1f} days old"
                   + ("" if _age <= 8.0 else
                      " -- the Sunday 11:00 VN snapshot job has not run; the "
                      "tracked crontab no longer reflects what is scheduled"))
        all_ok = all_ok and ok
    except Exception as _e:
        all_ok = check("crontab snapshot", False, f"{_e}") and all_ok

    print("=" * 60)
    # STATUS FILE (Aug 1 2026) so `pipecheck` can show health without re-running
    # three DB queries -- and, critically, can show the AGE of the last check.
    # A detector that stopped running looks identical to one reporting "fine":
    # that is exactly how feed_freshness_check vanished for six days.
    import json as _json
    _status = {"status": "ok" if all_ok else "fail",
               "failures": _FAILURES,
               "checked_at": datetime.now().isoformat(timespec="seconds"),
               "last_date": str(last_date)}
    try:
        with open(os.path.expanduser(
                "~/Desktop/ML_Quant_Fund/logs/health_status.json"), "w") as _f:
            _json.dump(_status, _f, indent=1)
    except Exception as _e:
        print(f"  [warn] health_status.json write failed: {_e}")

    if all_ok:
        print("  ✅ ALL CHECKS PASSED — system healthy")
        # NO success notification. It fired EVERY day regardless of state --
        # zero information, and half the alert-fatigue mechanism (the other
        # half was 77 consecutive false failures from the outcomes-maturity
        # bug; ALL CHECKS PASSED had never once appeared in the log).
    else:
        print("  ❌ ISSUES DETECTED — review above")
        _msg = "FAILED: " + ", ".join(_FAILURES) if _FAILURES else "issues detected"
        os.system("osascript -e 'display notification \"" + _msg.replace('"', "'")
                  + "\" with title \"ML Quant Fund ALERT\"'")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
