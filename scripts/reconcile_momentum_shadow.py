"""
scripts/reconcile_momentum_shadow.py — point-in-time 20d outcomes for momentum shadow.

reconcile_outcomes() in accuracy/sink.py is hardwired to the predictions table at
horizon 1/3/5 (the broken direction model). The momentum shadow signal is 20d and
lives in momentum_shadow_predictions. This reconciles THAT table at 20d using the
SAME point-in-time forward-return logic (close[D] -> close[D+20 trading days]),
only emitting an outcome once D+20 has actually elapsed (no look-ahead — research:
"misaligned/forward labels not shifted correctly are the #1 source of leak").

Writes momentum_shadow_outcomes. Idempotent. Non-fatal in pipeline.
"""
from __future__ import annotations
import sqlite3, sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

DB = ROOT / "accuracy.db"
MOM_HORIZON = 20

DDL = """
CREATE TABLE IF NOT EXISTS momentum_shadow_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    kind            TEXT NOT NULL,
    horizon         INTEGER NOT NULL,
    outcome_date    TEXT NOT NULL,
    actual_return   REAL NOT NULL,
    actual_up       INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE(prediction_date, ticker, kind)
);
CREATE INDEX IF NOT EXISTS idx_momout_date ON momentum_shadow_outcomes(prediction_date);
"""


def _add_trading_days(start, n):
    d, added = start, 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def main():
    con = sqlite3.connect(str(DB))
    con.executescript(DDL)
    as_of = date.today()
    # picks not yet reconciled
    pend = pd.read_sql("""
        SELECT s.prediction_date, s.ticker, s.kind
        FROM momentum_shadow_predictions s
        LEFT JOIN momentum_shadow_outcomes o
          ON s.prediction_date=o.prediction_date AND s.ticker=o.ticker AND s.kind=o.kind
        WHERE o.id IS NULL
    """, con)
    if pend.empty:
        print("reconcile_momentum_shadow: nothing pending"); con.close(); return 0
    pend["pd_date"] = pd.to_datetime(pend["prediction_date"]).dt.date
    pend["outcome_date"] = pend["pd_date"].apply(lambda d: _add_trading_days(d, MOM_HORIZON))
    due = pend[pend["outcome_date"] <= as_of]
    if due.empty:
        print(f"reconcile_momentum_shadow: {len(pend)} picks pending, none mature yet "
              f"(need {MOM_HORIZON} trading days)"); con.close(); return 0

    from features.builder import _download
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    for tk, g in due.groupby("ticker"):
        try:
            px = _download(tk, str(g["pd_date"].min() - timedelta(days=5)), None).set_index("date")["close"]
            px.index = pd.to_datetime(px.index)
        except Exception:
            continue
        for r in g.itertuples():
            try:
                cp = float(px.asof(pd.Timestamp(r.pd_date)))
                co = float(px.asof(pd.Timestamp(r.outcome_date)))
                if cp == 0 or np.isnan(cp) or np.isnan(co):
                    continue
                ret = (co - cp) / cp
                con.execute("""INSERT OR REPLACE INTO momentum_shadow_outcomes
                    (prediction_date,ticker,kind,horizon,outcome_date,actual_return,actual_up,created_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (r.prediction_date, tk, r.kind, MOM_HORIZON, str(r.outcome_date),
                     ret, int(ret > 0), now))
                written += 1
            except Exception:
                continue
    con.commit(); con.close()
    print(f"reconcile_momentum_shadow: wrote {written} matured 20d outcomes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
