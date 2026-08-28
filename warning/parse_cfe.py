#!/usr/bin/env python3
"""
parse_cfe.py — CFE VIX futures settlement files -> data_vintages.

FORMAT (verified against the real files 2026-08-28):
    Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
    10/21/2004,F (Jan 05),168.1,168.9,168,168.5,167,167,13,0,13
  Date is MM/DD/YYYY. Settle is column index 6. The contract is taken from the
  FILENAME (CFE_<code><yy>_VX.csv), not the "Futures" label, because the filename
  is machine-generated and the label is free text.

WHAT IT WRITES
    VX_<CODE><YY>   raw settle per contract, e.g. VX_F05
    VX_FRONT        settle of the nearest-expiry contract trading that day
    VX_SECOND       settle of the next contract
  Front/second need NO expiry calendar: each file spans exactly one contract's
  trading life, so the set of files containing a date IS the set of live
  contracts on that date. Sorting them by (year, month-code) gives the order.

THE 10x SCALE BREAK -- DETECTED, NOT ASSUMED
    CFE's original VIX futures were quoted on a multiplied index (the Oct-2004
    settles above are ~167 while VIX was in the mid-teens), and the contract was
    later de-multiplied to 1x. The exact changeover is NOT hardcoded here. The
    parser stores RAW settles and prints the median ratio of VX_FRONT to same-day
    VIXCLS per year, so the break is visible in the output and can be ruled on
    with evidence. Nothing downstream should consume VX_FRONT until that ruling
    is recorded -- see DECISIONS.md.

USAGE
    python warning/parse_cfe.py --dir data/raw/cfe --db warning.db --dry-run
    python warning/parse_cfe.py --dir data/raw/cfe --db warning.db
"""
import argparse
import csv
import io
import os
import re
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

MONTH_CODES = "FGHJKMNQUVXZ"          # F=Jan ... Z=Dec
FNAME = re.compile(r"CFE_([FGHJKMNQUVXZ])(\d{2})_VX\.csv$", re.I)
SETTLE_COL = 6


def contract_key(fname):
    """'CFE_F05_VX.csv' -> (2005, 1, 'F05'). None if the name does not match."""
    m = FNAME.search(os.path.basename(fname))
    if not m:
        return None
    code, yy = m.group(1).upper(), int(m.group(2))
    year = 2000 + yy
    return (year, MONTH_CODES.index(code) + 1, f"{code}{yy:02d}")


def parse_date(s):
    s = (s or "").strip()
    try:
        m, d, y = s.split("/")
        return date(int(y), int(m), int(d)).isoformat()
    except Exception:
        return None


def parse_file(path):
    """-> [(obs_date, settle)]. Rows that do not parse are counted, not guessed."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(io.StringIO(f.read())))
    out, skipped = [], 0
    for row in rows:
        if len(row) <= SETTLE_COL:
            skipped += 1
            continue
        d = parse_date(row[0])
        if d is None:
            skipped += 1                     # header and any preamble
            continue
        raw = row[SETTLE_COL].strip()
        try:
            v = float(raw)
        except ValueError:
            skipped += 1
            continue
        if v > 0:
            out.append((d, v))
    return out, skipped


def build(dirpath):
    """-> (per_contract, front, second). front/second are [(date, settle)]."""
    per_contract = {}
    by_date = defaultdict(list)              # date -> [(year, month, code, settle)]
    skipped_total = 0
    for fname in sorted(os.listdir(dirpath)):
        key = contract_key(fname)
        if key is None:
            continue
        year, month, code = key
        rows, skipped = parse_file(os.path.join(dirpath, fname))
        skipped_total += skipped
        if not rows:
            continue
        per_contract[f"VX_{code}"] = rows
        for d, v in rows:
            by_date[d].append((year, month, code, v))

    front, second = [], []
    for d in sorted(by_date):
        live = sorted(by_date[d])            # (year, month) ascending = expiry order
        front.append((d, live[0][3]))
        if len(live) > 1:
            second.append((d, live[1][3]))
    return per_contract, front, second, skipped_total


def scale_report(con, front):
    """Median VX_FRONT / same-day VIXCLS by year. Makes the 10x break visible."""
    vix = dict(con.execute(
        "SELECT obs_date, value FROM data_vintages WHERE series_id='VIXCLS'").fetchall())
    by_year = defaultdict(list)
    for d, v in front:
        iv = vix.get(d)
        if iv:
            by_year[d[:4]].append(v / iv)
    out = []
    for y in sorted(by_year):
        r = sorted(by_year[y])
        out.append((y, len(r), r[len(r) // 2]))
    return out


MULTIPLIER = 10.0
RATIO_SPLIT = 5.0      # the two regimes sit at ~1 and ~10; 5 separates them with
                       # enormous margin, so no changeover DATE has to be asserted


def normalize(con, rows):
    """Divide multiplied-era settles by 10, decided per row against same-day VIX.

    D13. The alternative -- hardcoding a changeover date -- would be an assertion
    about CFE contract history that no ingested source states. The ratio is
    either ~1 or ~10, so classifying each row on its own evidence is both safer
    and self-documenting. Rows with no same-day VIXCLS inherit the previous row's
    classification; if none has been made yet they are dropped, because an
    unclassifiable settle is worse than a missing one.
    """
    vix = dict(con.execute(
        "SELECT obs_date, value FROM data_vintages WHERE series_id='VIXCLS'").fetchall())
    out, last_div, dropped, switch = [], None, 0, None
    for d, v in rows:
        iv = vix.get(d)
        if iv:
            div = MULTIPLIER if (v / iv) > RATIO_SPLIT else 1.0
            if last_div is not None and div != last_div and switch is None:
                switch = d
            last_div = div
        elif last_div is None:
            dropped += 1
            continue
        else:
            div = last_div
        out.append((d, v / div))
    return out, dropped, switch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/cfe")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-contract", action="store_true",
                    help="also write the 150+ VX_<CODE><YY> per-contract series")
    ap.add_argument("--normalize", action="store_true",
                    help="divide multiplied-era settles by 10, decided PER ROW "
                         "against same-day VIXCLS (see DECISIONS.md D13)")
    args = ap.parse_args()

    per_contract, front, second, skipped = build(args.dir)
    print(f"contracts parsed: {len(per_contract)}   unparseable rows: {skipped} "
          f"(headers are expected here)")
    if front:
        print(f"VX_FRONT  {len(front)} rows  {front[0][0]}..{front[-1][0]}")
        print(f"VX_SECOND {len(second)} rows  {second[0][0]}..{second[-1][0]}")

    con = sqlite3.connect(args.db)

    print("\n=== SCALE CHECK: median VX_FRONT / VIXCLS by year ===")
    print("    a ratio near 10 means the multiplied contract; near 1 means "
          "de-multiplied.\n")
    rep = scale_report(con, front)
    if not rep:
        print("    no overlapping VIXCLS dates -- load the FRED leg first")
    for y, n, ratio in rep:
        flag = "  <== BREAK" if rep and abs(ratio - 1.0) > 0.3 and ratio < 5 else ""
        print(f"    {y}  n={n:>4}  median ratio {ratio:8.3f}{flag}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        con.close()
        return

    def write(series, rows):
        for d, v in rows:
            pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            con.execute("INSERT OR IGNORE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)", (series, d, pub, v, "CFE"))

    if args.normalize:
        front, dropped_f, switch = normalize(con, front)
        second, dropped_s, _ = normalize(con, second)
        print(f"\n  normalized per row against same-day VIXCLS "
              f"(D13). first 10x->1x switch observed at: {switch}")
        if dropped_f or dropped_s:
            print(f"  dropped {dropped_f + dropped_s} unclassifiable rows "
                  f"(no same-day VIXCLS and no prior classification)")
    write("VX_FRONT", front)
    write("VX_SECOND", second)
    if args.per_contract:
        for series, rows in per_contract.items():
            write(series, rows)
    con.commit()
    # NB: '_' is a single-char wildcard in SQL LIKE, so 'VX_%' also matches
    # VXOCLS and VXVCLS. Escape it or the count silently includes the FRED
    # volatility series (found 2026-08-28: reported 20720 instead of 7006).
    n = con.execute(r"SELECT COUNT(*) FROM data_vintages "
                    r"WHERE series_id LIKE 'VX\_%' ESCAPE ''").fetchone()[0]
    con.close()
    print(f"\nwrote VX_FRONT/VX_SECOND"
          f"{' + per-contract' if args.per_contract else ''}; "
          f"data_vintages now holds {n} VX_* rows")
    if args.normalize:
        print("Scale-normalized per D13; VX_FRONT/VX_SECOND are on the same scale "
              "as VIXCLS throughout.")
    else:
        print("RAW settles, NOT scale-normalized. Re-run with --normalize, or do "
              "not consume VX_FRONT downstream.")


if __name__ == "__main__":
    main()
