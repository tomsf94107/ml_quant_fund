#!/usr/bin/env python3
"""One-time heal: re-fetch each persisted Form 4 accession, re-parse with the
issuer-aware parser, and remove other-issuer rows (venture-arm portfolio
dispositions) from all form4* tables. Marks form4_parsed so the accession is
never re-fetched. Dry-run default; --apply to write.
Usage: python scripts/repair_form4_other_issuer.py --ticker GOOG --cik 1652044 [--apply]"""
import argparse, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.monitor_ticker import fetch_form4_xml, parse_form4_xml, now_iso, DB_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--cik", type=int, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    tkr = a.ticker.upper()

    con = sqlite3.connect(str(DB_PATH), timeout=30)
    cur = con.cursor()
    accs = [r[0] for r in cur.execute(
        "SELECT accession_number FROM form4_parsed WHERE ticker=? "
        "AND filer_name NOT LIKE '[OTHER ISSUER%'", (tkr,)).fetchall()]
    print(f"{tkr}: {len(accs)} persisted accession(s) to re-check")

    tables = []
    for (name,) in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'form4%'"):
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({name})")]
        if "accession_number" in cols:
            tables.append((name, "ticker" in cols))
    print("form4 tables with accession_number:", [t for t, _ in tables])

    bad = []
    for acc in accs:
        xml = fetch_form4_xml(a.cik, acc, None)
        time.sleep(0.15)
        if not xml:
            print(f"  {acc}: fetch failed -- SKIP (left as-is)")
            continue
        p = parse_form4_xml(xml)
        isym = (p.get("issuer_symbol") or "").upper()
        if isym and isym != tkr:
            bad.append((acc, isym, p.get("filer_name") or "?"))
            print(f"  {acc}: issuer={isym}  filer={p.get('filer_name')}  -> REMOVE")
    print(f"\nother-issuer accessions: {len(bad)} of {len(accs)}")

    if not a.apply:
        print("DRY RUN ONLY. Re-run with --apply to write.")
        return
    for acc, isym, filer in bad:
        for tbl, has_tkr in tables:
            if tbl == "form4_parsed":
                continue
            q = f"DELETE FROM {tbl} WHERE accession_number=?"
            p_ = [acc]
            if has_tkr:
                q += " AND ticker=?"
                p_.append(tkr)
            cur.execute(q, p_)
            print(f"  {acc}: {tbl} -{cur.rowcount} row(s)")
        cur.execute("""UPDATE form4_parsed SET
                         filer_name=?, is_director=0, is_officer=0,
                         is_ten_percent_owner=0, transaction_count=0,
                         aggregate_p_value=0, aggregate_s_value=0, parsed_at=?
                       WHERE accession_number=? AND ticker=?""",
                    (f"[OTHER ISSUER {isym}] {filer}", now_iso(), acc, tkr))
    con.commit()
    print("APPLIED.")


if __name__ == "__main__":
    main()
