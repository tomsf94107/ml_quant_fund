#!/usr/bin/env python3
"""
ticker_lifecycle.py -- handle the OTHER half of universe management: RENAME and
RETIRE. Onboarding (add_ticker.py) was only ever half the lifecycle.

WHY THIS EXISTS (found 2026-08-14)
  Three "dead feeds" turned out to be three DIFFERENT corporate actions:
    CYBR  delisted  (PANW merger closed 2026-02-11)      -> RETIRE
    EA    delisted  (PIF take-private closed 2026-08-04) -> RETIRE
    SATS  RENAMED   (-> ECHO effective 2026-06-24)       -> RENAME, not retire
  Retiring SATS would have silently dropped a LIVE constituent. A dead vendor
  feed does not tell you which of the three happened -- a human must verify the
  corporate action. This tool executes the decision once it is made.

DESIGN RULES
  - RETIRE NEVER DELETES DB ROWS. Deleting history manufactures survivorship
    bias -- the exact bias the fund haircuts for. Retire = remove from the
    ACTIVE universe (so it stops emitting predictions); history stays intact.
  - RENAME rewrites the ticker across every table that has a ticker column,
    because a rename is the SAME security (CUSIP unchanged) and the history
    must stay continuous.
  - Every DB is backed up before write. Conflicts are detected BEFORE any
    UPDATE and abort that table.
  - Schema is introspected (PRAGMA), so table/column names are not hardcoded.

USAGE
  python scripts/ticker_lifecycle.py --rename SATS:ECHO --dry-run
  python scripts/ticker_lifecycle.py --rename SATS:ECHO
  python scripts/ticker_lifecycle.py --retire CYBR,EA --reason "delisted" --dry-run
  python scripts/ticker_lifecycle.py --retire CYBR,EA --reason "delisted"
  python scripts/ticker_lifecycle.py --status ECHO
"""
import argparse
import csv
import os
import shutil
import sqlite3
import sys
from datetime import date

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
DB_NAMES = ["prices.db", "earnings_monitor.db", "accuracy.db",
            "short_interest.db", "institutional_trades.db", "earnings.db"]
TICKER_COLS = {"ticker", "symbol", "sym", "ticker_symbol"}


def dbs():
    return [(n, os.path.join(ROOT, n)) for n in DB_NAMES
            if os.path.isfile(os.path.join(ROOT, n))]


def meta_path():
    return os.path.join(ROOT, "tickers_metadata.csv")


def retired_path():
    return os.path.join(ROOT, "tickers_retired.csv")


def watchlist_path():
    return os.path.join(ROOT, "tickers_watchlist.txt")


def runner_path():
    """THE universe file: daily_runner.load_tickers() reads tickers.txt.
    tickers_metadata.csv is METADATA ONLY and does not enrol anything."""
    return os.path.join(ROOT, "tickers.txt")


def _rename_line_file(path, label, old, new, dry):
    if not os.path.isfile(path):
        return
    lines = list(open(path))
    hits = sum(1 for l in lines if l.strip().upper() == old)
    if hits and not dry:
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            for l in lines:
                f.write(new + "\n" if l.strip().upper() == old else l)
    if hits:
        print(f"  {label:24s} {hits} row(s) {'would change' if dry else 'renamed'}")


def _remove_from_line_file(path, label, want, dry):
    if not os.path.isfile(path):
        return 0
    lines = list(open(path))
    keep = [l for l in lines if l.strip().upper() not in want]
    removed = len(lines) - len(keep)
    if removed and not dry:
        shutil.copy2(path, path + ".bak")
        open(path, "w").writelines(keep)
    if removed:
        print(f"  {label:24s} -{removed} row(s){' (dry)' if dry else ''}")
    return removed


def ticker_tables(con):
    """[(table, ticker_col)] for every table exposing a ticker-like column."""
    out = []
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        try:
            cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        except sqlite3.Error:
            continue
        col = next((c[1] for c in cols if c[1].lower() in TICKER_COLS), None)
        if col:
            out.append((t, col))
    return out


def do_status(ticker):
    print(f"# status: {ticker}\n")
    for name, path in dbs():
        con = sqlite3.connect(path, timeout=30)
        try:
            for tbl, col in ticker_tables(con):
                try:
                    n = con.execute(
                        f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col}"=?', (ticker,)).fetchone()[0]
                except sqlite3.Error:
                    continue
                if n:
                    print(f"  {name:24s} {tbl:32s} {n}")
        finally:
            con.close()
    mp = meta_path()
    if os.path.isfile(mp):
        hit = any(r and r[0].strip().upper() == ticker
                  for r in csv.reader(open(mp)))
        print(f"\n  tickers_metadata.csv     {'PRESENT' if hit else 'absent'}")
    for path, label in [(runner_path(), "tickers.txt (RUNNER)"),
                        (watchlist_path(), "tickers_watchlist.txt")]:
        if os.path.isfile(path):
            hit = any(l.strip().upper() == ticker for l in open(path))
            print(f"  {label:24s} {'PRESENT' if hit else 'absent'}")
    return 0


def do_rename(old, new, dry):
    print(f"# rename {old} -> {new}   dry_run={dry}\n")
    total = 0
    for name, path in dbs():
        con = sqlite3.connect(path, timeout=30)
        try:
            tabs = ticker_tables(con)
        except sqlite3.Error as e:
            print(f"  {name}: cannot read schema: {e}")
            con.close()
            continue

        plan = []
        for tbl, col in tabs:
            try:
                n_old = con.execute(
                    f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col}"=?', (old,)).fetchone()[0]
                n_new = con.execute(
                    f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col}"=?', (new,)).fetchone()[0]
            except sqlite3.Error:
                continue
            if n_old:
                plan.append((tbl, col, n_old, n_new))
        con.close()

        if not plan:
            continue
        print(f"  == {name} ==")
        if not dry:
            shutil.copy2(path, path + f".bak.rename_{date.today().isoformat()}")
        con = sqlite3.connect(path, timeout=30)
        try:
            for tbl, col, n_old, n_new in plan:
                warn = f"  (target already has {n_new} rows)" if n_new else ""
                if dry:
                    print(f"     {tbl:32s} {n_old:7d} rows would change{warn}")
                    total += n_old
                    continue
                try:
                    con.execute(f'UPDATE "{tbl}" SET "{col}"=? WHERE "{col}"=?', (new, old))
                    con.commit()
                    left = con.execute(
                        f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col}"=?', (old,)).fetchone()[0]
                    print(f"     {tbl:32s} {n_old:7d} renamed, {left} left{warn}")
                    total += n_old
                except sqlite3.IntegrityError as e:
                    con.rollback()
                    print(f"     {tbl:32s} ABORTED (constraint): {e}")
                except sqlite3.Error as e:
                    con.rollback()
                    print(f"     {tbl:32s} ERROR: {e}")
        finally:
            con.close()

    # universe CSV
    mp = meta_path()
    if os.path.isfile(mp):
        rows = list(csv.reader(open(mp)))
        hits = sum(1 for r in rows[1:] if r and r[0].strip().upper() == old)
        if hits and not dry:
            shutil.copy2(mp, mp + ".bak")
            for r in rows[1:]:
                if r and r[0].strip().upper() == old:
                    r[0] = new
            with open(mp, "w", newline="") as f:
                csv.writer(f).writerows(rows)
        print(f"\n  tickers_metadata.csv     {hits} row(s) {'would change' if dry else 'renamed'}")

    # THE runner universe + watchlist
    _rename_line_file(runner_path(), "tickers.txt", old, new, dry)
    _rename_line_file(watchlist_path(), "tickers_watchlist.txt", old, new, dry)

    print(f"\n# total rows {'to change' if dry else 'changed'}: {total}")
    if dry:
        print("# re-run without --dry-run to apply")
    return 0


def do_retire(tickers, reason, dry):
    print(f"# retire {', '.join(tickers)}   reason={reason!r}   dry_run={dry}")
    print("# NOTE: DB history is PRESERVED (deleting it would manufacture")
    print("#       survivorship bias). Retire only removes from the ACTIVE universe.\n")

    # report retained history so the decision is informed
    for t in tickers:
        counts = []
        for name, path in dbs():
            con = sqlite3.connect(path, timeout=30)
            try:
                n = 0
                for tbl, col in ticker_tables(con):
                    try:
                        n += con.execute(
                            f'SELECT COUNT(*) FROM "{tbl}" WHERE "{col}"=?', (t,)).fetchone()[0]
                    except sqlite3.Error:
                        pass
                if n:
                    counts.append(f"{name}={n}")
            finally:
                con.close()
        print(f"  {t:8s} history retained: {', '.join(counts) if counts else 'none'}")

    mp, rp, wp = meta_path(), retired_path(), watchlist_path()
    if not os.path.isfile(mp):
        print(f"\n! {mp} not found")
        return 1

    rows = list(csv.reader(open(mp)))
    header, body = rows[0], rows[1:]
    want = {t.upper() for t in tickers}
    keep = [r for r in body if not (r and r[0].strip().upper() in want)]
    moved = [r for r in body if r and r[0].strip().upper() in want]

    print(f"\n  tickers_metadata.csv     {len(body)} -> {len(keep)} rows "
          f"({len(moved)} removed)")
    missing = want - {r[0].strip().upper() for r in moved}
    if missing:
        print(f"  ! not present in CSV: {', '.join(sorted(missing))}")

    rp_n = _remove_from_line_file(runner_path(), "tickers.txt", want, dry)
    if not rp_n and os.path.isfile(runner_path()):
        print("  tickers.txt              not present (already out of runner universe)")
    _remove_from_line_file(watchlist_path(), "tickers_watchlist.txt", want, dry)

    if dry:
        print("\n# re-run without --dry-run to apply")
        return 0

    shutil.copy2(mp, mp + ".bak")
    with open(mp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep)

    new_file = not os.path.isfile(rp)
    with open(rp, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ticker", "retired_date", "reason"])
        for r in moved:
            w.writerow([r[0].strip().upper(), date.today().isoformat(), reason])
    print(f"  tickers_retired.csv      +{len(moved)} row(s)")

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rename", help="OLD:NEW")
    ap.add_argument("--retire", help="comma list")
    ap.add_argument("--reason", default="delisted")
    ap.add_argument("--status", help="report where a ticker appears")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT
    if args.root:
        ROOT = os.path.expanduser(args.root)
    if not os.path.isdir(ROOT):
        sys.exit(f"FATAL: root not found: {ROOT}")

    if args.status:
        return do_status(args.status.strip().upper())
    if args.rename:
        if ":" not in args.rename:
            ap.error("--rename needs OLD:NEW")
        old, new = (x.strip().upper() for x in args.rename.split(":", 1))
        return do_rename(old, new, args.dry_run)
    if args.retire:
        ts = [t.strip().upper() for t in args.retire.split(",") if t.strip()]
        return do_retire(ts, args.reason, args.dry_run)
    ap.error("one of --rename / --retire / --status required")


if __name__ == "__main__":
    sys.exit(main())
