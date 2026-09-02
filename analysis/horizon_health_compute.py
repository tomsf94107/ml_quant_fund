"""
Horizon health computer — compute-once, display-many.
Writes high-confidence + overall accuracy per horizon to the horizon_health
table (history archive), prints a summary (captured by B's log -> pipecheck).
Read-only against predictions/outcomes; only writes its own table.
"""
import sqlite3
from datetime import date
from pathlib import Path

DB = Path.home() / "Desktop" / "ML_Quant_Fund" / "accuracy.db"

DDL = """
CREATE TABLE IF NOT EXISTS horizon_health (
  run_date     TEXT NOT NULL,
  horizon      INTEGER NOT NULL,
  window_days  INTEGER NOT NULL,
  band         TEXT NOT NULL,
  n            INTEGER,
  acc_pct      REAL,
  avg_ret_pct  REAL,
  PRIMARY KEY (run_date, horizon, window_days, band)
);
"""

CONFIGS = [
    (30, "highconf", "AND p.prob_up>=0.70"),
    (30, "overall",  ""),
    (90, "highconf", "AND p.prob_up>=0.70"),
    (90, "overall",  ""),
]

QUERY = """
SELECT p.horizon AS h, COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS acc,
  ROUND(AVG(o.actual_return)*100,3) AS avg_ret
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up IS NOT NULL {extra}
  AND o.prediction_date >= date('now','-{win} days')
GROUP BY p.horizon;
"""

def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, as percentages.

    Wilson rather than the normal approximation: it stays inside [0,1]
    and does not collapse at small n, which is precisely the regime
    these high-confidence cohorts live in (n=35 to n=61 on 30d windows).
    """
    import math
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100.0 * (c - s) / d, 100.0 * (c + s) / d)


def main():
    run_date = date.today().isoformat()
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.executescript(DDL)
    rows_written = 0
    summary = {}
    for win, band, extra in CONFIGS:
        q = QUERY.format(extra=extra, win=win)
        for h, n, acc, ret in cur.execute(q).fetchall():
            cur.execute(
                "INSERT OR REPLACE INTO horizon_health "
                "(run_date,horizon,window_days,band,n,acc_pct,avg_ret_pct) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_date, h, win, band, n, acc, ret))
            rows_written += 1
            summary.setdefault((band, win), {})[h] = (n, acc, ret)
    con.commit()
    con.close()
    print(f"[horizon_health] run_date={run_date} rows_written={rows_written}")
    for (band, win), hd in sorted(summary.items()):
        parts = []
        for h in (1, 3, 5):
            if h in hd:
                n, acc, ret = hd[h]
                lo, hi = wilson(round(n * acc / 100.0), n) \
                    if (acc is not None and n) else (None, None)
                ci = f" [{lo:.1f}-{hi:.1f}]" if lo is not None else ""
                parts.append(
                    f"h{h}: {acc}%{ci} (n={n}, ret={ret:+.2f}%)")
        print(f"  [{band:8s} {win}d] " + "  |  ".join(parts))
    # Flag horizons where the 30d interval CONTAINS the 90d point
    # estimate. That is the comparison that prompted this: 49.2% (n=61)
    # against 60.3% (n=847) looked like a collapse, but the 30d interval
    # spans it, so the two windows are not distinguishable from these
    # numbers alone.
    _hc30, _hc90 = (summary.get(("highconf", 30), {}),
                    summary.get(("highconf", 90), {}))
    _overlap = []
    for _h in sorted(set(_hc30) & set(_hc90)):
        _n30, _a30, _ = _hc30[_h]
        _n90, _a90, _ = _hc90[_h]
        if None in (_a30, _a90) or not _n30:
            continue
        _lo, _hi = wilson(round(_n30 * _a30 / 100.0), _n30)
        if _lo is not None and _lo <= _a90 <= _hi:
            _overlap.append(f"h{_h}")
    if _overlap:
        print(f"  note: 30d CI contains 90d for {', '.join(_overlap)}"
              f" -- not distinguishable; see pooled_accuracy.py")

    hc30 = summary.get(("highconf", 30), {})
    if 1 in hc30:
        n1, acc1, ret1 = hc30[1]
        if acc1 is not None:
            status = "RECOVERING" if acc1 >= 55 else ("WEAK" if acc1 >= 50 else "BROKEN")
            print(f"  [h1 watch] high-conf 30d acc={acc1}% -> {status} (>=55% for 2-3wk = regime passed)")

if __name__ == "__main__":
    main()
