#!/usr/bin/env python3
"""
extend_credit_series.py — pull FULL history for the credit series.

WHY
    HY OAS (BAMLH0A0HYM2) was the strongest predictor of the universe's forward
    5-day up-rate in the 2026-09-03 timing test: rho +0.196, quintile spread
    +9.3pp, the only candidate clearing t=2 after the overlapping-window
    discount. High spreads predicted HIGHER forward up-rates -- buy fear.

    But warning.db holds it only from 2023-08-28, a rolling three-year window
    noted during the crash-warning build. That sample is a single regime: a bull
    market with dips, in which "buy the dip" works by construction. The
    relationship cannot be trusted until it is tested somewhere it might fail.

    FRED carries BAMLH0A0HYM2 daily from 1996-12-31 -- covering 2000, 2008 and
    2020. That is the sample that would actually test it.

WHAT IT FETCHES
    BAMLH0A0HYM2   ICE BofA US High Yield OAS, daily, 1996-12-31+
    BAMLC0A0CM     ICE BofA US Corporate (IG) OAS, daily, 1996-12-31+
    DGS2           2-year Treasury CMT, daily, 1976+
    T10Y2Y         10y-2y slope, daily, 1976+
    DTWEXBGS       Broad dollar index, daily, 2006+
    DCOILWTICO     WTI spot, daily, 1986+

    All are FRED, free, and not routinely revised (subject to occasional
    correction). No ALFRED vintage handling is applied: these are treated as
    non-revisable, which matches how series_meta.py already classifies the
    Treasury series. pub_date is stamped obs_date + 1 day, consistent with the
    rest of the non-revisable series in this database.

    Existing rows are preserved -- INSERT OR IGNORE on the full key -- so a
    re-run cannot overwrite a vintage that is already stored.

USAGE
    export FRED_API_KEY=...        # already in .env
    python analysis/extend_credit_series.py --db warning.db --dry-run
    python analysis/extend_credit_series.py --db warning.db
"""
import argparse
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

SERIES = {
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A0CM": "ICE BofA US Corporate (IG) OAS",
    "DGS2": "2-year Treasury CMT",
    "T10Y2Y": "10y-2y slope",
    "DTWEXBGS": "Broad dollar index",
    "DCOILWTICO": "WTI spot",
}
BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch(series, key, start="1900-01-01", timeout=60):
    q = urllib.parse.urlencode({
        "series_id": series, "api_key": key, "file_type": "json",
        "observation_start": start,
    })
    req = urllib.request.Request(f"{BASE}?{q}",
                                 headers={"User-Agent": "ml-quant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("observations", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY not set -- run: set -a && . ./.env && set +a")

    con = sqlite3.connect(args.db)
    print(f"{'series':<16}{'have':>8}{'have from':>13}"
          f"{'fetched':>9}{'from':>13}{'new':>8}")
    total_new = 0
    for sid, label in SERIES.items():
        cur = con.execute(
            "SELECT COUNT(*), MIN(obs_date) FROM data_vintages WHERE series_id=?",
            (sid,)).fetchone()
        have_n, have_from = cur[0], cur[1] or "-"
        try:
            obs = fetch(sid, key)
        except Exception as e:
            print(f"{sid:<16}  FAILED: {type(e).__name__}: {e}")
            continue
        rows = []
        for o in obs:
            v = o.get("value")
            if v in (".", "", None):
                continue
            try:
                rows.append((o["date"], float(v)))
            except ValueError:
                continue
        if not rows:
            print(f"{sid:<16}  no usable observations")
            continue

        new = 0
        if not args.dry_run:
            for d, v in rows:
                pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
                cur2 = con.execute(
                    "INSERT OR IGNORE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) "
                    "VALUES (?,?,?,?,?)", (sid, d, pub, v, "FRED"))
                new += cur2.rowcount
            con.commit()
        else:
            existing = {r[0] for r in con.execute(
                "SELECT obs_date FROM data_vintages WHERE series_id=?", (sid,))}
            new = sum(1 for d, _ in rows if d not in existing)

        print(f"{sid:<16}{have_n:>8}{have_from:>13}"
              f"{len(rows):>9}{rows[0][0]:>13}{new:>8}")
        total_new += new
        time.sleep(0.4)

    con.close()
    if args.dry_run:
        print(f"\nDRY RUN -- {total_new} new rows would be written.")
    else:
        print(f"\nwrote {total_new} new rows. Existing rows untouched "
              f"(INSERT OR IGNORE).")
        print("Re-run analysis/market_timing_test.py to test the HY OAS "
              "relationship\non a sample that includes 2000, 2008 and 2020 -- "
              "the regimes where\n'buy fear' would fail if it is going to.")


if __name__ == "__main__":
    main()
