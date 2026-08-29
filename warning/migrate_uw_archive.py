#!/usr/bin/env python3
"""
migrate_uw_archive.py — make uw_archive multi-vintage per session date.

THE BUG THIS FIXES (found 2026-08-28, before the first cron run)
    uw_archive's primary key was (endpoint, query_params, snapshot_date) and the
    archiver inserts with INSERT OR IGNORE. Once snapshot_date became the ET date,
    two pulls covering the SAME ET session collide, and the FIRST one wins.

    Concretely: the manual pull at 08:18 ET on 2026-08-28 ran before the open and
    captured 2026-08-27 data. The cron at 06:30 ICT Sat = 19:30 ET Fri stamps the
    same 2026-08-28 and would have been IGNORED -- the fresh post-close payload
    discarded, the stale pre-open one locked in permanently. No error, no log
    line, and uw_archive is append-only so it could never be corrected.

THE FIX
    Add pulled_at to the primary key. This is exactly the data_vintages pattern:
    snapshot_date names the SESSION, pulled_at names the OBSERVATION of it, and
    several observations of one session may coexist. Nothing is overwritten and
    nothing is deleted, which is what rule #8 actually asks for -- the old key
    silently dropped writes, which is a form of overwriting.

    A parse step should take, for each (endpoint, snapshot_date), the row with the
    LATEST pulled_at at or before its own as-of moment. Same discipline as
    pit.series_asof.

SAFETY
    Dry-run by default. Rebuilds via a new table + copy + rename inside one
    transaction; SQLite cannot ALTER a primary key in place. Row counts are
    compared before and after and the migration aborts if they differ.

USAGE
    python warning/migrate_uw_archive.py --db warning.db
    python warning/migrate_uw_archive.py --db warning.db --apply
"""
import argparse
import sqlite3

NEW_DDL = """
CREATE TABLE uw_archive_new (
    endpoint      TEXT NOT NULL,
    query_params  TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,          -- the ET session being captured
    payload_json  TEXT NOT NULL,
    pulled_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (endpoint, query_params, snapshot_date, pulled_at)
)
"""


def current_pk(con):
    row = con.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                      "AND name='uw_archive'").fetchone()
    return row[0] if row else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)

    ddl = current_pk(con)
    if ddl is None:
        raise SystemExit("uw_archive does not exist -- apply warning_schema.sql first")
    already = "pulled_at)" in ddl.replace(" ", "").replace("\n", "")[-60:]
    before = con.execute("SELECT COUNT(*) FROM uw_archive").fetchone()[0]
    dates = con.execute("SELECT snapshot_date, COUNT(*) FROM uw_archive "
                        "GROUP BY snapshot_date ORDER BY snapshot_date").fetchall()

    print(f"uw_archive: {before} rows")
    for d, n in dates:
        print(f"  {d}  {n} rows")
    print(f"\npulled_at already in the primary key: {already}")

    if already:
        print("nothing to do.")
        return

    if not args.apply:
        print("\nDRY RUN -- would rebuild uw_archive with PK "
              "(endpoint, query_params, snapshot_date, pulled_at).")
        print("Re-run with --apply. Do this BEFORE the next cron run, or that "
              "run's payloads are silently dropped.")
        return

    con.execute("BEGIN")
    try:
        con.execute("DROP TABLE IF EXISTS uw_archive_new")
        con.executescript(NEW_DDL)
        con.execute("INSERT INTO uw_archive_new "
                    "(endpoint, query_params, snapshot_date, payload_json, pulled_at) "
                    "SELECT endpoint, query_params, snapshot_date, payload_json, "
                    "pulled_at FROM uw_archive")
        moved = con.execute("SELECT COUNT(*) FROM uw_archive_new").fetchone()[0]
        if moved != before:
            raise RuntimeError(f"row count changed: {before} -> {moved}")
        con.execute("DROP TABLE uw_archive")
        con.execute("ALTER TABLE uw_archive_new RENAME TO uw_archive")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    after = con.execute("SELECT COUNT(*) FROM uw_archive").fetchone()[0]
    print(f"\nmigrated. rows {before} -> {after} (must be equal)")
    print("uw_archive now keyed on (endpoint, query_params, snapshot_date, "
          "pulled_at); repeat pulls of one session all persist.")
    con.close()


if __name__ == "__main__":
    main()
