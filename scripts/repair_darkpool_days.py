#!/usr/bin/env python3
"""Heal darkpool_prints gaps: per-day UW walks with date= + older_than cursor.
Use whenever the monitor skips days. Idempotent (UPSERT on tracking_id).
Stops: day-start reached (ET-derived, DST-safe) / short page / stall guard.
Usage: python repair_darkpool_days.py --ticker MSFT [--days 45] [--db earnings_monitor.db]
"""
import argparse, os, sqlite3, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.market_calendar import is_trading_day

import requests

ET = ZoneInfo("America/New_York")
UPSERT = """INSERT INTO darkpool_prints
(ticker, executed_at, size, price, value_usd, venue, tracking_id,
 nbbo_bid, nbbo_ask, canceled, ext_hours, et_date)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(tracking_id) DO UPDATE SET
 nbbo_bid=excluded.nbbo_bid, nbbo_ask=excluded.nbbo_ask,
 canceled=excluded.canceled, ext_hours=excluded.ext_hours,
 et_date=excluded.et_date"""


def repair(ticker: str, days: int, db: str) -> None:
    key = os.environ.get("UW_API_KEY")
    if not key:
        sys.exit("UW_API_KEY not set (run: set -a && . ./.env && set +a)")
    h = {"Authorization": f"Bearer {key}"}
    url = f"https://api.unusualwhales.com/api/darkpool/{ticker}"
    con = sqlite3.connect(db, timeout=30)
    end = datetime.now(ET).date() - timedelta(days=1)
    d = end - timedelta(days=days)
    total = 0
    while d <= end:
        if is_trading_day(d):
            day_start_utc = datetime(d.year, d.month, d.day, 4, 0,
                                     tzinfo=ET).astimezone(timezone.utc
                                     ).strftime("%Y-%m-%dT%H:%M:%SZ")
            iso, older, prev_oldest, pages, day_rows = d.isoformat(), None, None, 0, 0
            while pages < 80:
                p = {"date": iso, "limit": 500}
                if older:
                    p["older_than"] = older
                r = requests.get(url, headers=h, params=p, timeout=30)
                if r.status_code != 200:
                    print(f"  {iso}: HTTP {r.status_code}, stopping day")
                    break
                rows = (r.json() or {}).get("data") or []
                if not rows:
                    break
                for x in rows:
                    ts = x.get("executed_at")
                    if not ts:
                        continue
                    try:
                        et_dt = datetime.fromisoformat(
                            ts.replace("Z", "+00:00")).astimezone(ET)
                        sz = float(x.get("size") or 0)
                        px = float(x.get("price") or 0)
                    except (TypeError, ValueError):
                        continue
                    con.execute(UPSERT, (
                        ticker, ts, sz, px, sz * px,
                        x.get("market_center") or x.get("venue"),
                        str(x.get("tracking_id")) if x.get("tracking_id")
                            else f"{ticker}-{ts}-{sz}-{px}",
                        float(x["nbbo_bid"]) if x.get("nbbo_bid") else None,
                        float(x["nbbo_ask"]) if x.get("nbbo_ask") else None,
                        1 if x.get("canceled") else 0,
                        x.get("ext_hour_sold_codes"),
                        et_dt.date().isoformat()))
                    day_rows += 1
                pages += 1
                oldest = min((x["executed_at"] for x in rows
                              if x.get("executed_at")), default=None)
                if oldest is None or len(rows) < 500:
                    break
                if oldest <= day_start_utc:
                    break
                if oldest == prev_oldest:
                    raise RuntimeError(f"cursor stalled at {oldest} on {iso}")
                prev_oldest = oldest
                older = (datetime.fromisoformat(oldest.replace("Z", "+00:00"))
                         + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            con.commit()
            total += day_rows
            print(f"  {iso}: {pages:>2} page(s), {day_rows:>6} rows")
            time.sleep(0.2)
        d += timedelta(days=1)
    print(f"TOTAL upserted: {total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--db", default="earnings_monitor.db")
    a = ap.parse_args()
    repair(a.ticker.upper(), a.days, a.db)
