#!/usr/bin/env python3
"""
dump_universe.py -- pull the COMPLETE ticker universe from the system.

It does not assume a single source of truth. It collects the union of every
ticker found in:
  - tickers_metadata.csv
  - every table in each DB that has a ticker/symbol column (schema is
    introspected via PRAGMA, so exact table/column names are NOT hardcoded)

Outputs:
  - the sorted union, one ticker per line (paste-ready), also written to
    universe_all.txt
  - per-source counts, so you can see which source is authoritative and where
    a ticker is present/missing
  - optional --matrix: ticker x source presence grid

USAGE
  python dump_universe.py                 # print union + source counts
  python dump_universe.py --matrix        # also print presence matrix
  python dump_universe.py --out uni.txt   # custom output path
  ML_QUANT_ROOT=~/ML_Quant_Fund python dump_universe.py
"""
import argparse
import csv
import os
import sqlite3
import sys

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
META = os.path.join(ROOT, "tickers_metadata.csv")
DBS = {
    "prices":         os.path.join(ROOT, "prices.db"),
    "monitor":        os.path.join(ROOT, "earnings_monitor.db"),
    "accuracy":       os.path.join(ROOT, "accuracy.db"),
    "short_interest": os.path.join(ROOT, "short_interest.db"),
    "institutional":  os.path.join(ROOT, "institutional_trades.db"),
    # earnings.db was MISSING from v1.0 and is a real universe source --
    # it held 63 ECHO rows that the 498-ticker union never saw (2026-08-15).
    "earnings":       os.path.join(ROOT, "earnings.db"),
}
TICKER_COLS = {"ticker", "symbol", "sym", "tickers", "ticker_symbol"}


def norm(x):
    return (x or "").strip().upper()


def from_csv(sources):
    if not os.path.isfile(META):
        print(f"# WARN: {META} not found", file=sys.stderr)
        return
    with open(META, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    tcol = 0
    for i, h in enumerate(header):
        if h.strip().lower() in TICKER_COLS:
            tcol = i
            break
    for r in rows[1:]:
        if r and len(r) > tcol and norm(r[tcol]):
            sources.setdefault(norm(r[tcol]), set()).add("csv")


def from_db(dbname, path, sources):
    if not os.path.isfile(path):
        print(f"# WARN: db not found: {path}", file=sys.stderr)
        return
    try:
        con = sqlite3.connect(path, timeout=30)
    except sqlite3.Error as e:
        print(f"# WARN: cannot open {dbname}: {e}", file=sys.stderr)
        return
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for tbl in tables:
            try:
                cols = con.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            except sqlite3.Error:
                continue
            tcol = next((c[1] for c in cols if c[1].lower() in TICKER_COLS), None)
            if not tcol:
                continue
            label = f"{dbname}:{tbl}"
            try:
                for (val,) in con.execute(f'SELECT DISTINCT "{tcol}" FROM "{tbl}"'):
                    if norm(val):
                        sources.setdefault(norm(val), set()).add(label)
            except sqlite3.Error as e:
                print(f"# WARN: read {label}: {e}", file=sys.stderr)
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="universe_all.txt", help="write union here")
    ap.add_argument("--matrix", action="store_true", help="print ticker x source grid")
    ap.add_argument("--root", help="override ML_QUANT_ROOT")
    args = ap.parse_args()

    global ROOT, META, DBS
    if args.root:
        ROOT = os.path.expanduser(args.root)
        META = os.path.join(ROOT, "tickers_metadata.csv")
        DBS = {k: os.path.join(ROOT, os.path.basename(v)) for k, v in DBS.items()}

    sources = {}  # ticker -> set(source labels)
    from_csv(sources)
    for name, path in DBS.items():
        from_db(name, path, sources)

    union = sorted(sources)
    with open(args.out, "w") as f:
        f.write("\n".join(union) + "\n")

    # per-source counts
    per_source = {}
    for labs in sources.values():
        for l in labs:
            per_source[l] = per_source.get(l, 0) + 1

    print(f"# universe union: {len(union)} tickers  (written to {args.out})")
    print("# per-source counts:")
    for l in sorted(per_source):
        print(f"#   {l:32s} {per_source[l]}")
    print()

    if args.matrix:
        labels = sorted(per_source)
        print("ticker," + ",".join(labels))
        for t in union:
            print(t + "," + ",".join("x" if l in sources[t] else "" for l in labels))
    else:
        # paste-ready plain list
        print("\n".join(union))


if __name__ == "__main__":
    main()
