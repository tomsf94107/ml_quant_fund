#!/usr/bin/env python3
"""
probe_intraday_reconcile.py — find why reconcile_intraday_outcomes() stopped.

READ-ONLY. Replays the real function's body on a handful of genuinely
unreconciled rows, but WITHOUT the `except Exception: pass` that hides the
cause. Writes nothing to any database.

The production function swallows every error, so a per-row failure looks
identical to "nothing to do". This surfaces the actual exception, with a
traceback, for the first few rows -- and reports which stage each row reaches.

    python probe_intraday_reconcile.py 2>&1 | tail -60
"""
import sqlite3
import sys
import traceback
from datetime import datetime, timedelta

sys.path.insert(0, ".")


def main():
    from utils.timezone import now_et, ET

    now = now_et()
    print(f"now_et() = {now}")

    db = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    rows = db.execute("""
        SELECT p.ticker, p.prediction_ts, p.horizon_hr, p.price_at_pred, p.signal
        FROM intraday_predictions p
        LEFT JOIN intraday_outcomes o
            ON p.ticker=o.ticker AND p.prediction_ts=o.prediction_ts
               AND p.horizon_hr=o.horizon_hr
        WHERE o.id IS NULL
        ORDER BY p.prediction_ts
    """).fetchall()
    print(f"unreconciled rows: {len(rows)}")
    if not rows:
        print("nothing unreconciled -- the reconciler is not the problem")
        return
    print(f"oldest: {rows[0][1]}   newest: {rows[-1][1]}")

    # Rows spanning the gap: a few just after the last successful outcome, and
    # a few recent ones. If old rows fail but new ones would succeed (or vice
    # versa) that itself is diagnostic.
    sample = rows[:3] + rows[len(rows) // 2:len(rows) // 2 + 2] + rows[-3:]
    print(f"\nprobing {len(sample)} rows with exceptions SURFACED\n" + "-" * 70)

    stages = {}
    for ticker, pred_ts, horizon_hr, price_at_pred, signal in sample:
        tag = f"{ticker} {pred_ts} h={horizon_hr}"
        stage = "start"
        try:
            stage = "parse_ts"
            pred_dt = ET.localize(datetime.fromisoformat(pred_ts))
            outcome_dt = pred_dt + timedelta(hours=horizon_hr)
            if outcome_dt > now:
                print(f"  {tag}: SKIP -- outcome {outcome_dt} is in the future")
                stages["future"] = stages.get("future", 0) + 1
                continue

            stage = "import_massive_client"
            import pandas as pd
            from features import massive_client as mc

            stage = "download"
            hist = mc.download(ticker,
                               start=outcome_dt.strftime("%Y-%m-%d"),
                               end=(outcome_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1m", auto_adjust=True, progress=False)
            print(f"  {tag}: download -> "
                  f"{type(hist).__name__} rows={0 if hist is None else len(hist)}")
            if hist is None or hist.empty:
                print(f"      EMPTY for {outcome_dt:%Y-%m-%d} "
                      f"-- this is the silent 'continue' in production")
                stages["empty_download"] = stages.get("empty_download", 0) + 1
                continue

            stage = "columns"
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            stage = "tz_convert"
            hist.index = hist.index.tz_convert(ET)

            stage = "close_series"
            close = hist["Close"].squeeze()

            stage = "locate_price"
            if outcome_dt.hour >= 16:
                price_at_outcome = float(close.iloc[-1])
            else:
                idx = close.index.get_indexer([outcome_dt], method="nearest")[0]
                gap = abs((close.index[idx] - outcome_dt).total_seconds())
                if gap > 3600:
                    print(f"      nearest bar is {gap/60:.0f} min away (>60) "
                          f"-- production would silently continue")
                    stages["gap_too_large"] = stages.get("gap_too_large", 0) + 1
                    continue
                price_at_outcome = float(close.iloc[idx])

            r = (price_at_outcome - price_at_pred) / price_at_pred
            print(f"      OK  price {price_at_pred:.2f} -> {price_at_outcome:.2f}  "
                  f"ret {r:+.4f}")
            stages["would_succeed"] = stages.get("would_succeed", 0) + 1

        except Exception as e:
            print(f"  {tag}: EXCEPTION at stage '{stage}': "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            stages[f"exc:{stage}:{type(e).__name__}"] = \
                stages.get(f"exc:{stage}:{type(e).__name__}", 0) + 1

    print("\n" + "-" * 70)
    print("stage summary:")
    for k, v in sorted(stages.items(), key=lambda x: -x[1]):
        print(f"  {v:>4}  {k}")
    print("\nIn production every line above marked EXCEPTION is silently "
          "discarded by `except Exception: pass`.")
    db.close()


if __name__ == "__main__":
    main()
