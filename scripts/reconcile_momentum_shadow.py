#!/usr/bin/env python3
"""
scripts/reconcile_momentum_shadow.py — point-in-time 20-SESSION outcomes.

REWRITTEN 2026-07-14. The prior version had THREE stacked date bugs and produced
a wrong return for 3,940 of 3,940 rows — i.e. EVERY row of the promotion
evidence for the one signal that passed the 18-year money gate.

  BUG 1 — _add_trading_days() counted WEEKDAYS, not SESSIONS:
              if d.weekday() < 5: added += 1
          Juneteenth, July-4-observed, Memorial Day were all counted as trading
          days, so a "20 trading day" window landed 1–2 sessions SHORT.

  BUG 2 — px.asof(outcome_date) SILENTLY ROLLS BACK to the last bar at or before
          the target. When the weekday-guess landed on an actual holiday
          (2026-06-05 → 2026-07-03), asof quietly returned the PREVIOUS session's
          close. No error, no warning.

  BUG 3 — as_of = date.today() is the Mac's VN local date — a day AHEAD of the US
          calendar. Picks were marked "due" before their outcome session existed
          (2026-06-12 → stored 2026-07-10, correct 2026-07-14).

  MEASURED (AAPL, prediction 2026-05-29, stored −9.0624%):
      close[2026-06-26] / close[2026-05-29] − 1 = −9.0624%   ← 19 sessions. STORED.
      close[2026-06-29] / close[2026-05-29] − 1 = −9.716%    ← 20 sessions. CORRECT.

WHY IT MATTERS
  analysis/momentum_18yr_test.py — the test momentum PASSED — uses
      fwd = panel.iloc[i + HOLD] / panel.iloc[i] - 1.0
  a POSITIONAL shift on a session index: exactly 20 sessions, holidays
  irrelevant. The shadow was measuring 18–19. The promotion gate and the
  backtest it validates were not measuring the same thing.

THE FIX — the pattern accuracy/sink.py already uses:
      idx     = px.index                          # sessions only, by construction
      pos     = idx.searchsorted(pred_ts, "right") - 1
      out_pos = pos + MOM_HORIZON                 # advance 20 REAL sessions
      if out_pos >= len(idx): continue            # not matured → stays pending
      outcome_date = idx[out_pos].date()          # the REAL session

  Holiday-proof (a session index contains no holidays). No silent rollback
  (positional, not asof). No look-ahead (the len() check IS the maturity test —
  date.today() is never consulted).

USAGE
    python scripts/reconcile_momentum_shadow.py                     # incremental (cron)
    python scripts/reconcile_momentum_shadow.py --rebuild --dry-run # show every diff
    python scripts/reconcile_momentum_shadow.py --rebuild           # recompute ALL
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute EVERY prediction, not just unreconciled ones "
                         "(INSERT OR REPLACE overwrites — no DELETE needed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change; write nothing")
    a = ap.parse_args()

    con = sqlite3.connect(str(DB))
    con.executescript(DDL)

    if a.rebuild:
        q = """
            SELECT s.prediction_date, s.ticker, s.kind,
                   o.outcome_date  AS old_date,
                   o.actual_return AS old_ret
            FROM momentum_shadow_predictions s
            LEFT JOIN momentum_shadow_outcomes o
              ON s.prediction_date=o.prediction_date
             AND s.ticker=o.ticker AND s.kind=o.kind
        """
    else:
        q = """
            SELECT s.prediction_date, s.ticker, s.kind,
                   NULL AS old_date, NULL AS old_ret
            FROM momentum_shadow_predictions s
            LEFT JOIN momentum_shadow_outcomes o
              ON s.prediction_date=o.prediction_date
             AND s.ticker=o.ticker AND s.kind=o.kind
            WHERE o.id IS NULL
        """
    pend = pd.read_sql(q, con)
    if pend.empty:
        print("reconcile_momentum_shadow: nothing to do")
        con.close()
        return 0

    pend["pd_date"] = pd.to_datetime(pend["prediction_date"]).dt.date
    print(f"reconcile_momentum_shadow: {len(pend)} picks "
          f"({'REBUILD' if a.rebuild else 'incremental'}"
          f"{', DRY RUN' if a.dry_run else ''})")

    from features.builder import _download

    now = datetime.now(timezone.utc).isoformat()
    written = immature = failed = 0
    changed, unchanged = 0, 0
    samples: list[str] = []

    for tk, g in pend.groupby("ticker"):
        try:
            px = (_download(tk, str(g["pd_date"].min() - timedelta(days=10)), None)
                  .set_index("date")["close"])
            px.index = pd.to_datetime(px.index)
            px = px[~px.index.duplicated(keep="last")].sort_index()
        except Exception:
            failed += len(g)
            continue
        idx = px.index
        if len(idx) == 0:
            failed += len(g)
            continue

        for r in g.itertuples():
            try:
                pos = idx.searchsorted(pd.Timestamp(r.pd_date), side="right") - 1
                if pos < 0:
                    failed += 1
                    continue
                out_pos = pos + MOM_HORIZON
                if out_pos >= len(idx):
                    immature += 1          # not matured yet — leave pending
                    continue

                cp = float(px.iloc[pos])
                co = float(px.iloc[out_pos])
                if cp == 0 or np.isnan(cp) or np.isnan(co):
                    failed += 1
                    continue

                ret = (co - cp) / cp
                outcome_date = idx[out_pos].date()

                if r.old_date is not None:
                    same_d = str(r.old_date)[:10] == str(outcome_date)
                    same_r = (r.old_ret is not None
                              and abs(float(r.old_ret) - ret) < 1e-9)
                    if same_d and same_r:
                        unchanged += 1
                    else:
                        changed += 1
                        if len(samples) < 12:
                            samples.append(
                                f"    {r.prediction_date[:10]} {tk:<6} {r.kind:<9} "
                                f"{str(r.old_date)[:10]} {float(r.old_ret)*100:+8.3f}%  ->  "
                                f"{outcome_date} {ret*100:+8.3f}%")

                if not a.dry_run:
                    con.execute(
                        """INSERT OR REPLACE INTO momentum_shadow_outcomes
                           (prediction_date,ticker,kind,horizon,outcome_date,
                            actual_return,actual_up,created_at)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (r.prediction_date, tk, r.kind, MOM_HORIZON,
                         str(outcome_date), ret, int(ret > 0), now))
                written += 1
            except Exception:
                failed += 1
                continue

    if not a.dry_run:
        con.commit()
    con.close()

    print(f"  matured & written : {written}")
    print(f"  not yet matured   : {immature}")
    print(f"  failed            : {failed}")
    if a.rebuild:
        print(f"  CHANGED           : {changed}")
        print(f"  unchanged         : {unchanged}")
        if samples:
            print("\n  sample diffs (old -> new):")
            print("\n".join(samples))
    if a.dry_run:
        print("\n  [DRY RUN] nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
