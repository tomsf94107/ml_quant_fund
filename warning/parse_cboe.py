#!/usr/bin/env python3
"""
parse_cboe.py — Cboe archive CSVs -> data_vintages.

fetch_free_history saves the Cboe files as raw CSVs; only the FRED leg upserts to
the database. This closes that gap. Idempotent: INSERT OR IGNORE on
(series_id, obs_date, pub_date).

SIX DISTINCT FORMATS, verified against the actual files 2026-08-28:
  1. *_History.csv OHLC   DATE,OPEN,HIGH,LOW,CLOSE   MM/DD/YYYY, no preamble
  2. *_History.csv single DATE,SKEW | DATE,VVIX      MM/DD/YYYY, no preamble
  3. vxocurrent.csv       3 preamble rows, "Date    ,Open,High ,Low  ,Close",
                          M/D/YYYY unpadded, values may carry leading spaces
  4. pcratioarchive.csv   3 preamble rows; header's last field is a multi-line
                          quoted disclaimer; 3 ratio columns
  5. totalpc/equitypc/indexpc  3 preamble rows, DATE,CALL(S),PUT(S),TOTAL,P/C Ratio
  6. vixpc.csv            2 preamble rows, Date,VIX P/C Ratio,Put Vol,Call Vol,Total

ENCODING: pcratioarchive.csv and vixpc.csv contain non-UTF-8 bytes (0xa0). Read
with errors="replace"; the affected bytes are all inside disclaimer text.

SERIES IDs are prefixed CBOE_ so they never collide with the FRED-distributed
VIXCLS / VXVCLS / VXOCLS already in data_vintages. Where both exist, prefer FRED:
verified 2026-08-28, FRED VXVCLS starts 2007-12-04 while Cboe's own
VIX3M_History.csv starts only 2009-09-18.

COVERAGE FINDINGS (recorded, not repaired):
  - Put/call archives END 2019-10-04 (totalpc, equitypc, indexpc, vixpc) and
    pcratioarchive ends 2003-12-31, leaving a 2004-01..2006-10 hole. F1 has no
    post-2019 P/C history from this route.
  - VXO_History.csv starts 1993-01; the 1986-2003 span needs vxoarchive.xls
    (Excel, not parsed here). FRED VXOCLS covers 1986-01-02..2021-09-23 and is
    the better single source.
  - VVIX's earliest rows are erratic (2006-03-06 71.73 then 2006-03-15 15.71).
    Loaded as-is and flagged; do not silently clean vendor data.

USAGE
    python warning/parse_cboe.py --dir data/raw/cboe --db warning.db
    python warning/parse_cboe.py --dir data/raw/cboe --db warning.db --dry-run
"""
import argparse
import csv
import io
import os
import sqlite3
from datetime import date, timedelta

# filename -> (kind, {series_id: column_index})
SPEC = {
    "VIX_History.csv":   ("ohlc",   {"CBOE_VIX": 4}),
    "VIX3M_History.csv": ("ohlc",   {"CBOE_VIX3M": 4}),
    "VIX6M_History.csv": ("ohlc",   {"CBOE_VIX6M": 4}),
    "VIX9D_History.csv": ("ohlc",   {"CBOE_VIX9D": 4}),
    "COR1M_History.csv": ("ohlc",   {"CBOE_COR1M": 4}),
    "COR3M_History.csv": ("ohlc",   {"CBOE_COR3M": 4}),
    "VXO_History.csv":   ("ohlc",   {"CBOE_VXO": 4}),
    "SKEW_History.csv":  ("single", {"CBOE_SKEW": 1}),
    "VVIX_History.csv":  ("single", {"CBOE_VVIX": 1}),
    "vxocurrent.csv":    ("pre3",   {"CBOE_VXO_CURRENT": 4}),
    "pcratioarchive.csv": ("pre3",  {"CBOE_PC_TOTAL": 1,
                                     "CBOE_PC_INDEX": 2,
                                     "CBOE_PC_EQUITY": 3}),
    "totalpc.csv":       ("pre3",   {"CBOE_PC_TOTAL": 4}),
    "indexpc.csv":       ("pre3",   {"CBOE_PC_INDEX": 4}),
    "equitypc.csv":      ("pre3",   {"CBOE_PC_EQUITY": 4}),
    "vixpc.csv":         ("pre2",   {"CBOE_PC_VIX": 1}),
}

SKIP_ROWS = {"ohlc": 1, "single": 1, "pre3": 3, "pre2": 2}


def parse_date(s):
    s = s.strip()
    if not s:
        return None
    try:
        m, d, y = s.split("/")
    except ValueError:
        return None
    if len(y) != 4:
        return None
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def parse_file(path, kind, cols):
    """Yield (series_id, obs_date, value). Unparseable rows are skipped and
    counted by the caller -- never guessed at."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(io.StringIO(f.read())))
    out, skipped = [], 0
    for row in rows[SKIP_ROWS[kind]:]:
        if not row:
            continue
        d = parse_date(row[0])
        if d is None:
            skipped += 1
            continue
        for series, idx in cols.items():
            if idx >= len(row):
                continue
            raw = row[idx].strip()
            if raw in ("", ".", "n/a", "N/A"):
                continue
            try:
                out.append((series, d, float(raw)))
            except ValueError:
                skipped += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/cboe")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = None if args.dry_run else sqlite3.connect(args.db)
    total_new = 0
    print(f"{'file':<22}{'series':<20}{'rows':>8}{'skipped':>9}  range")
    print("-" * 88)
    for fname, (kind, cols) in SPEC.items():
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            print(f"{fname:<22}{'-- missing --':<20}")
            continue
        recs, skipped = parse_file(path, kind, cols)
        by_series = {}
        for s, d, v in recs:
            by_series.setdefault(s, []).append((d, v))
        for series, vals in sorted(by_series.items()):
            vals.sort()
            if con is not None:
                for d, v in vals:
                    pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
                    con.execute(
                        "INSERT OR IGNORE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)", (series, d, pub, v, f"Cboe/{fname}"))
                con.commit()
            total_new += len(vals)
            print(f"{fname:<22}{series:<20}{len(vals):>8}{skipped:>9}  "
                  f"{vals[0][0]}..{vals[-1][0]}")

    if con is not None:
        n = con.execute("SELECT COUNT(*) FROM data_vintages "
                        "WHERE series_id LIKE 'CBOE_%'").fetchone()[0]
        con.close()
        print(f"\ndata_vintages now holds {n} CBOE_* rows ({total_new} parsed)")
    else:
        print(f"\nDRY RUN -- {total_new} rows would be written. Nothing written.")
    print("\nNOTE: prefer FRED VIXCLS/VXVCLS/VXOCLS over the CBOE_* copies where "
          "both exist; FRED VXVCLS reaches back to 2007-12 vs Cboe's 2009-09.")


if __name__ == "__main__":
    main()
