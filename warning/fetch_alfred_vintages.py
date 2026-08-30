#!/usr/bin/env python3
"""
fetch_alfred_vintages.py — true point-in-time vintages for the revisable series.

WHAT THIS FIXES
    fetch_free_history stamps revisable series with the PULL date, because a bulk
    FRED download returns only today's revised values and there is no honest way
    to say when an earlier number was first published. series_meta.py therefore
    marks HOUST, CSUSHPINSA, ABCOMP and DRTSCILM as revisable, and every
    historical point-in-time read of them returns NA. That is correct but it
    blocks S3 and leaves S15's inputs unusable.

    ALFRED (the archival FRED) serves the value AS KNOWN on a given date.
    Verified 2026-08-30: HOUST for 2006-10 requested with
    realtime_start=realtime_end=2007-01-01 returns 1488 -- the first print, not
    today's revised figure.

HOW THE VINTAGES ARE BUILT
    ALFRED's `output_type=2` returns one row per (observation, vintage), so a
    single request per series yields the whole revision history. Each distinct
    (obs_date, realtime_start) pair becomes a data_vintages row with
    pub_date = realtime_start. A point-in-time read then sees exactly the number
    that was on the screen that day, and a later revision is a separate row --
    which is what the (series_id, obs_date, pub_date) primary key was designed
    for and what the bulk pull could never populate.

    Existing pull-date-stamped rows are left alone. They are wrong for history
    but harmless: a read at any historical date now finds the real vintage first,
    and a read today finds both and takes the latest.

RATE LIMIT
    FRED allows 120 requests/minute. Four series is nothing, but the loop sleeps
    briefly anyway so this stays safe if the series list grows.

USAGE
    export FRED_API_KEY=...            # already present in .env
    python warning/fetch_alfred_vintages.py --db warning.db
    python warning/fetch_alfred_vintages.py --db warning.db --dry-run
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request

# The four series series_meta.py marks revisable. Adding a series here without
# also marking it revisable would be harmless but pointless: non-revisable
# series already have correct derived pub_dates.
REVISABLE = ["HOUST", "CSUSHPINSA", "ABCOMP", "DRTSCILM"]
BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_all_vintages(series, key, timeout=60):
    """output_type=2 -> every (observation, vintage) pair for the series."""
    q = urllib.parse.urlencode({
        "series_id": series, "api_key": key, "file_type": "json",
        "output_type": 2, "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
    })
    req = urllib.request.Request(f"{BASE}?{q}",
                                 headers={"User-Agent": "warning-system/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--series", default=",".join(REVISABLE))
    args = ap.parse_args()

    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY not set -- run: set -a && . ./.env && set +a")

    con = None if args.dry_run else sqlite3.connect(args.db)
    grand = 0
    for series in [s.strip() for s in args.series.split(",") if s.strip()]:
        try:
            payload = fetch_all_vintages(series, key)
        except Exception as e:
            print(f"  [{series}] FAILED: {type(e).__name__}: {e}")
            continue

        obs = payload.get("observations", [])
        rows, vintages = [], set()
        revised_obs = 0
        for o in obs:
            d = o.get("date")
            # output_type=2 names each vintage column "<SERIES>_YYYYMMDD" --
            # eight digits, NO dashes. An earlier version of this parser required
            # a dashed 10-character date and silently rejected every column,
            # reporting "no vintage rows parsed" from 811 good observations.
            seen = {}
            for k, v in o.items():
                if k == "date" or v in (".", "", None):
                    continue
                tail = k.rsplit("_", 1)[-1]
                if len(tail) != 8 or not tail.isdigit():
                    continue
                vint = f"{tail[:4]}-{tail[4:6]}-{tail[6:]}"
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                seen[vint] = fv
                rows.append((d, vint, fv))
                vintages.add(vint)
            if len(set(seen.values())) > 1:
                revised_obs += 1

        if not rows:
            print(f"  [{series}] no vintage rows parsed from "
                  f"{len(obs)} observations -- check the response shape")
            continue

        dates = sorted({r[0] for r in rows})
        n_obs = len({r[0] for r in rows})
        print(f"  [{series}] {len(rows)} (obs,vintage) pairs   "
              f"obs {dates[0]}..{dates[-1]}   {len(vintages)} vintages "
              f"{min(vintages)}..{max(vintages)}")
        print(f"      observations that were EVER revised: {revised_obs}/{n_obs}"
              f"  ({100.0 * revised_obs / max(n_obs, 1):.1f}%)")
        print(f"      PIT REACH: vintages begin {min(vintages)}, so a "
              f"point-in-time read before that date sees NOTHING for this\n"
              f"         series -- ALFRED simply has no record of what was "
              f"published then.")
        if revised_obs == 0:
            print(f"      -> {series} is NEVER REVISED in ALFRED's record. Its "
                  f"pub_date should be DERIVED from obs_date + publication_lag\n"
                  f"         (series_meta.py), not left to vintages: ALFRED's "
                  f"earliest vintage here is {min(vintages)}, so vintage-based\n"
                  f"         point-in-time reads cannot reach before that, while "
                  f"a derived lag reaches the full history.")

        # DEDUPLICATE TO CHANGE POINTS.
        # ALFRED repeats a value in every later vintage until it is revised, so
        # the raw feed is enormously redundant: HOUST alone returns 325,600
        # (obs,vintage) pairs for 811 observations. pit.series_asof takes the
        # latest pub_date <= asof, so storing only the vintages where the value
        # CHANGED gives byte-identical point-in-time answers from a fraction of
        # the rows. Writing all 1,060,940 would have quintupled warning.db to
        # store the same information.
        by_obs = {}
        for obs_date, pub, val in rows:
            by_obs.setdefault(obs_date, []).append((pub, val))
        kept = []
        for obs_date, pairs in by_obs.items():
            pairs.sort()
            last = None
            for pub, val in pairs:
                if last is None or val != last:
                    kept.append((obs_date, pub, val))
                    last = val
        print(f"      deduped to change points: {len(rows)} -> {len(kept)} rows "
              f"({100.0 * len(kept) / max(len(rows), 1):.1f}%)")

        if con is not None:
            for obs_date, pub, val in kept:
                con.execute(
                    "INSERT OR IGNORE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) "
                    "VALUES (?,?,?,?,?)",
                    (series, obs_date, pub, val, "ALFRED"))
            con.commit()
        grand += len(kept)
        time.sleep(0.6)

    if con is not None:
        n = con.execute("SELECT COUNT(*) FROM data_vintages "
                        "WHERE source='ALFRED'").fetchone()[0]
        con.close()
        print(f"\nwrote {grand} rows; data_vintages ALFRED rows now {n}")
        print("Historical point-in-time reads of these series now return the "
              "value that was actually published, not today's revision.")
    else:
        print(f"\nDRY RUN -- {grand} rows would be written.")


if __name__ == "__main__":
    main()
