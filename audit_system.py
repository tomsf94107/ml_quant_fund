#!/usr/bin/env python3
"""
audit_system.py — READ-ONLY inventory of the ML Quant Fund stack.

Writes nothing. Opens every database read-only (file:...?mode=ro). Introspects
schemas rather than assuming column names, because assuming them is how the last
three diagnostic queries in this project returned wrong answers.

    python audit_system.py > audit_$(date +%Y%m%d).txt 2>&1

Sections:
  1  database inventory: tables, row counts, date ranges, staleness
  2  cron inventory + log freshness
  3  feature panel: columns, null rate by month, zero-variance, families
  4  prediction pipeline: signal vocabulary, outcome join rate, accuracy with CI
  5  model artifacts: feature importance by retrain date, family concentration
  6  data-source freshness table (what is stale and by how much)
"""
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta

HOME = os.path.expanduser("~/Desktop/ML_Quant_Fund")
DBS = ["accuracy.db", "prices.db", "earnings_monitor.db", "short_interest.db",
       "institutional_trades.db", "warning.db"]

DATE_COLS = ("date", "prediction_date", "obs_date", "d", "snapshot_date",
             "trade_date", "retrain_date", "outcome_date", "asof_date",
             "prediction_ts", "report_date", "filing_date", "computed_at",
             "created_at", "pulled_at", "model_date", "run_date")

# Families, so we can see what the panel is actually made of.
# ORDER MATTERS: the first match wins, so specific families are checked before
# the broad price-technical catch-all. Word boundaries are mandatory -- without
# them "inst_signed_flow_5d" matches "low" and "vix_close" matches "close",
# which silently filed both under price_technical in the first run of this
# script. A miscounted family is exactly the error this audit exists to avoid.
FAMILY = [
    ("institutional",   r"\binst\b|\bblock\b|auction|dark pool|13f|whale"),
    ("short_interest",  r"\bshort\b|\bdtc\b|days to cover|borrow|float|squeeze|\bftd\b"),
    ("options",         r"\biv\b|skew|pc ratio|put call|gamma|greek|\boi\b|implied|vega|theta"),
    ("sentiment",       r"sentiment|fear greed|\bfg\b|news|wiki|social|analyst"),
    ("fundamental",     r"\bpe\b|\bpb\b|\beps\b|revenue|margin|profit|\broe\b|\broa\b|"
                        r"debt|\bbook\b|\bfcf\b|ebitda|accrual|valuation|cape"),
    ("macro",           r"\bcpi\b|\bgdp\b|payroll|unrate|\bfed\b|\brate\b|spread|\bted\b|"
                        r"sloos|housing|claims|\bpmi\b|\bism\b|yield|curve"),
    ("cross_asset",     r"\boil\b|\bspy\b|\bxlk\b|\bdxy\b|\bvix\b|\btnx\b|gold|copper|"
                        r"dollar|treasury"),
    ("sector",          r"sector|industry|peer|\bxl[a-z]\b|\bgroup\b"),
    ("regime",          r"regime|breadth|correlation|dispersion|\bbeta\b"),
    ("calendar",        r"day of week|month end|is month|season|earnings in|days to|"
                        r"quarter|\bdow\b"),
    ("price_technical", r"\brsi\b|macd|bb pct|\batr\b|\bobv\b|vwap|\bma \d|\bsma\b|"
                        r"\bema\b|\breturn \d|momentum|vol surge|premarket|\bgap\b|"
                        r"\bdma\b|\bprice\b|\bclose\b|\bhigh\b|\blow\b|\bopen\b|"
                        r"\bvolume\b|\bvol \d|\badx\b|stoch|\bcci\b|willr|\btrend\b"),
]


def q(con, sql, args=()):
    try:
        return con.execute(sql, args).fetchall()
    except Exception as e:
        return [("ERROR", str(e)[:90])]


def classify(name):
    """Underscores become spaces FIRST. '_' is a word char, so \bvix\b never
    matches 'vix_close' and \blow\b wrongly matches 'inst_signed_flow_5d'.
    Normalizing makes word boundaries mean what they look like they mean.
    Order matters: specific families are tested before the price catch-all."""
    n = re.sub(r"[_\-]+", " ", name.lower())
    for fam, pat in FAMILY:
        if re.search(pat, n):
            return fam
    return "unclassified"


def open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def section_databases():
    hdr("1  DATABASE INVENTORY")
    today = datetime.now().date()
    for db in DBS:
        path = os.path.join(HOME, db)
        if not os.path.exists(path):
            print(f"\n--- {db}: MISSING ---")
            continue
        size = os.path.getsize(path) / 1e6
        print(f"\n--- {db}  ({size:.1f} MB) ---")
        con = open_ro(path)
        tables = [r[0] for r in q(con, "SELECT name FROM sqlite_master "
                                       "WHERE type='table' AND name NOT LIKE "
                                       "'sqlite_%' ORDER BY name")]
        print(f"{'table':<34}{'rows':>10}  {'date column':<18}"
              f"{'min':<12}{'max':<12}{'stale':>7}")
        for t in tables:
            n = q(con, f'SELECT COUNT(*) FROM "{t}"')[0][0]
            cols = [r[1] for r in q(con, f'PRAGMA table_info("{t}")')]
            dcol = next((c for c in DATE_COLS if c in cols), None)
            if dcol and isinstance(n, int) and n:
                rng = q(con, f'SELECT MIN("{dcol}"), MAX("{dcol}") FROM "{t}"')[0]
                lo, hi = str(rng[0])[:10], str(rng[1])[:10]
                try:
                    stale = (today - datetime.fromisoformat(hi).date()).days
                    stale = f"{stale}d"
                except Exception:
                    stale = "?"
                print(f"{t:<34}{n:>10}  {dcol:<18}{lo:<12}{hi:<12}{stale:>7}")
            else:
                print(f"{t:<34}{n:>10}  {'(no date col)':<18}")
        con.close()


def section_cron():
    hdr("2  CRON + LOG FRESHNESS")
    try:
        ct = subprocess.run(["crontab", "-l"], capture_output=True,
                            text=True).stdout.splitlines()
    except Exception as e:
        print("crontab unreadable:", e)
        ct = []
    jobs = [l for l in ct if l.strip() and not l.strip().startswith("#")]
    print(f"{len(jobs)} active cron lines ({len(ct)} total incl. comments)\n")
    for l in jobs:
        m = re.match(r"^(\S+ \S+ \S+ \S+ \S+)\s+(.*)$", l.strip())
        if not m:
            continue
        sched, cmd = m.group(1), m.group(2)
        script = re.search(r"([\w/]+\.py)", cmd)
        log = re.search(r">>\s*(\S+\.log)", cmd)
        script = script.group(1).split("/")[-1] if script else "?"
        age = "no log"
        if log and os.path.exists(log.group(1)):
            mt = datetime.fromtimestamp(os.path.getmtime(log.group(1)))
            age = f"{(datetime.now() - mt).days}d ago ({mt:%Y-%m-%d %H:%M})"
        print(f"  {sched:<16} {script:<34} {age}")


def section_features():
    hdr("3  FEATURE PANEL")
    path = os.path.join(HOME, "accuracy.db")
    con = open_ro(path)
    cols = [r[1] for r in q(con, 'PRAGMA table_info("prediction_features")')]
    skip = {"id", "ticker", "prediction_date", "horizon", "created_at"}
    feats = [c for c in cols if c not in skip]
    fam = {}
    for c in feats:
        fam.setdefault(classify(c), []).append(c)
    print(f"{len(feats)} feature columns in prediction_features\n")
    for f in sorted(fam, key=lambda k: -len(fam[k])):
        print(f"  {f:<18} {len(fam[f]):>3}   {', '.join(sorted(fam[f]))[:110]}")

    print("\n--- null rate by month, last 8 months (>20% flagged) ---")
    months = [r[0] for r in q(con, "SELECT DISTINCT substr(prediction_date,1,7) m "
                                   "FROM prediction_features ORDER BY m DESC LIMIT 8")]
    for m in reversed(months):
        tot = q(con, "SELECT COUNT(*) FROM prediction_features "
                     "WHERE substr(prediction_date,1,7)=?", (m,))[0][0]
        if not tot:
            continue
        bad = []
        for c in feats:
            nn = q(con, f'SELECT SUM("{c}" IS NULL) FROM prediction_features '
                        f"WHERE substr(prediction_date,1,7)=?", (m,))[0][0] or 0
            if nn / tot > 0.20:
                bad.append(f"{c} {100*nn/tot:.0f}%")
        print(f"  {m}  n={tot:>6}   " + ("; ".join(bad) if bad else "all <20% null"))

    print("\n--- zero-variance columns in the last 90 days ---")
    cut = (datetime.now().date() - timedelta(days=90)).isoformat()
    flat = []
    for c in feats:
        r = q(con, f'SELECT COUNT(DISTINCT "{c}") FROM prediction_features '
                   f"WHERE prediction_date>=?", (cut,))[0][0]
        if isinstance(r, int) and r <= 1:
            flat.append(f"{c}({r})")
    print("  " + (", ".join(flat) if flat else "none"))
    con.close()


def section_predictions():
    hdr("4  PREDICTION PIPELINE")
    con = open_ro(os.path.join(HOME, "accuracy.db"))
    print("--- signal vocabulary by month ---")
    for r in q(con, "SELECT substr(prediction_date,1,7) m, signal, COUNT(*) "
                    "FROM predictions WHERE prediction_date>='2025-09' "
                    "GROUP BY m, signal ORDER BY m DESC, signal"):
        print("   ", r)

    print("\n--- predictions vs outcomes join rate ---")
    for r in q(con, """SELECT substr(p.prediction_date,1,7) m, p.horizon,
        COUNT(*) preds, SUM(o.actual_up IS NOT NULL) matched
        FROM predictions p LEFT JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.prediction_date>='2026-01'
        GROUP BY m, p.horizon ORDER BY m DESC, p.horizon LIMIT 24"""):
        print("   ", r)

    print("\n--- prob_up distribution (is the model even differentiating?) ---")
    for r in q(con, """SELECT substr(prediction_date,1,7) m, horizon, COUNT(*) n,
        ROUND(MIN(prob_up),3), ROUND(AVG(prob_up),3), ROUND(MAX(prob_up),3),
        SUM(prob_up>0.55), SUM(prob_up<0.45)
        FROM predictions WHERE prediction_date>='2026-04'
        GROUP BY m, horizon ORDER BY m DESC, horizon LIMIT 20"""):
        print("   ", r)

    print("\n--- accuracy_cache: every row, with n ---")
    for r in q(con, "SELECT ticker, horizon, window_days, ROUND(accuracy,3), "
                    "ROUND(roc_auc,3), n_predictions, computed_at "
                    "FROM accuracy_cache ORDER BY computed_at DESC LIMIT 25"):
        print("   ", r)
    con.close()


def section_models():
    hdr("5  MODEL ARTIFACTS")
    con = open_ro(os.path.join(HOME, "accuracy.db"))
    dates = [r[0] for r in q(con, "SELECT DISTINCT retrain_date FROM "
                                  "feature_importance_history ORDER BY "
                                  "retrain_date DESC LIMIT 5")]
    print("recent retrain dates:", dates)
    if dates:
        d = dates[0]
        print(f"\n--- top 20 features at {d}, aggregated across tickers ---")
        rows = q(con, """SELECT feature, ROUND(AVG(importance),4) imp,
            COUNT(DISTINCT ticker) tickers FROM feature_importance_history
            WHERE retrain_date=? GROUP BY feature ORDER BY imp DESC LIMIT 20""", (d,))
        tot = 0.0
        byfam = {}
        for feat, imp, nt in rows:
            fam = classify(feat)
            byfam[fam] = byfam.get(fam, 0.0) + (imp or 0)
            tot += (imp or 0)
            print(f"   {feat:<28} {imp:>9}  {fam:<18} tickers={nt}")
        print("\n--- importance share by family (top 20 only) ---")
        for f, v in sorted(byfam.items(), key=lambda x: -x[1]):
            print(f"   {f:<18} {100*v/tot:5.1f}%")
    con.close()


def main():
    print(f"AUDIT  {datetime.now():%Y-%m-%d %H:%M} local")
    print(f"repo   {HOME}")
    try:
        print("HEAD  ", subprocess.run(["git", "-C", HOME, "log", "--oneline", "-1"],
                                       capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    for fn in (section_databases, section_cron, section_features,
               section_predictions, section_models):
        try:
            fn()
        except Exception as e:
            print(f"\n!! {fn.__name__} failed: {type(e).__name__}: {e}")
    print("\n\nEND OF AUDIT")


if __name__ == "__main__":
    main()
