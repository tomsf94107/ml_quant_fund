#!/usr/bin/env python3
"""
parse_french.py — Ken French daily portfolios -> data_vintages.

STRUCTURE, verified against the real files 2026-08-30:
    Each CSV holds EXACTLY TWO stacked tables, no annual blocks and no firm
    counts:
        line   8/11  "Average Value Weighted Returns -- Daily"
        line   9/12  header row, leading comma then column names
        ...          ~26,277 rows of  YYYYMMDD,ret,ret,...
        line  26286   "Average Equal Weighted Returns -- Daily"
        ...          the same dates again, equal-weighted
        last          copyright line
    Returns are PERCENT. Missing is -99.99 or -999.

WHY BOTH WEIGHTINGS MATTER
    S6 compares equal-weight against cap-weight. French publishes both for the
    same portfolios in one file, so the comparison needs no construction at all
    -- unlike the RSP/SPY route, where two separate ETFs with different
    inception dates and tracking must be aligned.

WHAT IS WRITTEN
    FR_IND_VW:<name>   FR_IND_EW:<name>     49 industries
    FR_ME_VW:<name>    FR_ME_EW:<name>      size portfolios

    The colon keeps these clear of any LIKE 'X_%' pattern -- SQL treats '_' as a
    single-character wildcard, which already produced a miscounted total once in
    this project (parse_cfe, 2026-08-30).

    Only the columns a signal actually needs are written by default (--columns),
    because writing all 49 industries x 2 weightings x 26,277 days would add
    2.6M rows to a database currently holding ~250k. The default set covers S6.

VINTAGE STAMPING
    French recomputes portfolios each June and republishes the whole history, so
    an observation's "publication date" is not well defined. pub_date is stamped
    obs_date + 1 day, matching every other non-revised daily series here. That is
    a simplification: French DOES restate history when CRSP is revised, so a
    strict point-in-time read of 1970 would want the vintage as known then, which
    no source provides. Recorded rather than hidden -- see DECISIONS.md.

USAGE
    python warning/parse_french.py --dir data/raw/french --dry-run
    python warning/parse_french.py --dir data/raw/french --db warning.db --apply
"""
import argparse
import os
import sqlite3
from datetime import date, timedelta

VW_MARK = "Average Value Weighted Returns -- Daily"
EW_MARK = "Average Equal Weighted Returns -- Daily"
MISSING = {-99.99, -999.0, -999.99}

FILES = {
    "F-F_Research_Data_Factors_daily.CSV": "FR_F",
    "12_Industry_Portfolios_daily.csv": "FR_I12",
    "49_Industry_Portfolios_Daily.csv": "FR_IND",
    "Portfolios_Formed_on_ME_daily.csv": "FR_ME",
}
# S6 needs the broad market under both weightings. "Hi 30" is the large-cap
# tercile (the cap-weight proxy); the equal-weighted version of the same
# portfolio is the equal-weight comparison. Kept deliberately small: see the
# row-count note above.
DEFAULT_COLUMNS = ["Lo 30", "Med 40", "Hi 30"]


def parse(path):
    """-> {(weighting, column): [(obs_date, value)]}"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    # Single-table files (the research factors) carry no weighting marker at
    # all: prose, then a header row, then data. Detect that up front rather than
    # returning nothing, which is what a marker-only parser would do.
    single_table = not any(VW_MARK in l or EW_MARK in l for l in lines)

    out, weighting, header = {}, ("" if single_table else None), None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if VW_MARK in s:
            weighting, header = "VW", None
            continue
        if EW_MARK in s:
            weighting, header = "EW", None
            continue
        if s.startswith("Copyright"):
            break
        if weighting is None:
            continue
        if header is None:
            if s.startswith(","):
                header = [c.strip() for c in s.split(",")[1:]]
            continue
        parts = [p.strip() for p in s.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            continue
        d = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"
        for name, raw in zip(header, parts[1:]):
            try:
                v = float(raw)
            except ValueError:
                continue
            if v in MISSING:
                continue
            out.setdefault((weighting, name), []).append((d, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/french")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--columns", default=",".join(DEFAULT_COLUMNS),
                    help="comma-separated column names to write; 'ALL' writes "
                         "every column (millions of rows)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    wanted = None if args.columns.upper() == "ALL" else \
        [c.strip() for c in args.columns.split(",") if c.strip()]

    con = None if not args.apply else sqlite3.connect(args.db)
    grand = 0
    for fname, prefix in FILES.items():
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            print(f"[{fname}] MISSING")
            continue
        tables = parse(path)
        cols = sorted({c for _, c in tables})
        print(f"\n[{fname}] {len(tables)} (weighting,column) series, "
              f"{len(cols)} distinct columns")
        sample = next(iter(tables.values()))
        print(f"  dates {sample[0][0]}..{sample[-1][0]}  "
              f"{len(sample)} rows per series")
        if wanted:
            missing = [c for c in wanted if c not in cols]
            if missing:
                print(f"  columns not in this file (skipped): {missing}")

        for (w, name), rows in sorted(tables.items()):
            if wanted and name not in wanted:
                continue
            sid = f"{prefix}_{w}:{name}" if w else f"{prefix}:{name}"
            print(f"  {sid:<24} {len(rows):>7} rows  "
                  f"{rows[0][0]}..{rows[-1][0]}")
            grand += len(rows)
            if con is not None:
                for d, v in rows:
                    pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
                    con.execute(
                        "INSERT OR IGNORE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)",
                        (sid, d, pub, v, f"French/{fname}"))
                con.commit()

    if con is not None:
        n = con.execute(r"SELECT COUNT(*) FROM data_vintages "
                        r"WHERE series_id LIKE 'FR\_%' ESCAPE '\'").fetchone()[0]
        con.close()
        print(f"\nwrote {grand} rows; data_vintages FR_* now {n}")
    else:
        print(f"\nDRY RUN -- {grand} rows would be written. Add --apply.")


if __name__ == "__main__":
    main()
