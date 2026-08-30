#!/usr/bin/env python3
"""
parse_ritter.py — Ritter IPO statistics Table 1 -> data_vintages, for S11.

SOURCE
    data/raw/ritter/IPO-Statistics.pdf, Jay Ritter, University of Florida.
    Page 3 carries Table 1: "Mean First-day Returns and Money Left on the Table,
    1980-2025", one row per year:

        Year  NumberOfIPOs  EW-mean  PW-mean  median  amountLeft  aggregate
        1980      71         14.3%    20.0%    6.9%   $0.18bn     $0.91bn

    Page 4 is Table 1a, which repeats Year, count and the EW mean with a
    different tail. That redundancy is used as a CROSS-CHECK rather than ignored:
    if the two tables disagree on a year, the extraction is wrong and the run
    stops rather than writing a plausible-looking number.

SERIES WRITTEN
    RITTER_IPO_COUNT        number of IPOs that year
    RITTER_IPO_FIRSTDAY     equal-weighted mean first-day return, in percent

    The registry's S11 formula is "IPO count + mean first-day return in top
    decile (hist to date)", so both legs come from this one table.

VINTAGE STAMPING
    obs_date is 31 December of the reference year. Ritter publishes the prior
    year's figures in the following spring; the PDF's Table 1 header is dated
    March 16 2026 for the 1980-2025 table. pub_date is therefore stamped as
    31 March of the FOLLOWING year, which is deliberately conservative: a
    point-in-time read in January cannot see the year that just ended.

    Getting this wrong in the optimistic direction would let S11 "know" IPO
    statistics months before Ritter published them -- exactly the look-ahead the
    pub_date discipline exists to prevent.

USAGE
    python warning/parse_ritter.py --pdf data/raw/ritter/IPO-Statistics.pdf
    python warning/parse_ritter.py --pdf ... --db warning.db --apply
"""
import argparse
import re
import sqlite3

ROW = re.compile(
    r"^\s*((?:19|20)\d\d)\s+([\d,]+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%")
PUB_MONTH_DAY = "-03-31"          # published the spring after the reference year


def parse_page(pdf_path, page_index):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_index].extract_text() or ""
    out = {}
    for line in text.split("\n"):
        m = ROW.match(line)
        if not m:
            continue
        year = int(m.group(1))
        count = int(m.group(2).replace(",", ""))
        ew = float(m.group(3))
        out[year] = (count, ew)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/raw/ritter/IPO-Statistics.pdf")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t1 = parse_page(args.pdf, 2)          # page 3: Table 1
    t1a = parse_page(args.pdf, 3)         # page 4: Table 1a
    print(f"Table 1  (page 3): {len(t1)} years "
          f"{min(t1) if t1 else '-'}..{max(t1) if t1 else '-'}")
    print(f"Table 1a (page 4): {len(t1a)} years "
          f"{min(t1a) if t1a else '-'}..{max(t1a) if t1a else '-'}")

    if not t1:
        raise SystemExit("no rows parsed from Table 1 -- the PDF layout has "
                         "changed; re-cut the regex against the real text.")

    # CROSS-CHECK WITH TOLERANCE.
    #
    # The two tables are revised on DIFFERENT DATES -- verified 2026-08-30, the
    # PDF headers read "Table 1 ... (March 16, 2026)" and "Table 1a (March 17,
    # 2026)". Three of 46 years differ by one IPO or a few tenths of a percent,
    # which is Ritter revising, not a parsing fault. Demanding exact equality
    # rejected good data.
    #
    # A real extraction error looks nothing like this: a misread column is wildly
    # wrong or wrong everywhere. So the check keeps its teeth via a tolerance
    # tight enough that any column confusion still trips it.
    #
    # Table 1 remains the SOURCE -- it is the table whose title matches S11's
    # formula. Table 1a is only a bound.
    COUNT_TOL = 2          # IPOs
    RETURN_TOL = 1.0       # percentage points

    shared = sorted(set(t1) & set(t1a))
    small, large = [], []
    for y in shared:
        dc = abs(t1[y][0] - t1a[y][0])
        dr = abs(t1[y][1] - t1a[y][1])
        if dc or dr:
            (large if (dc > COUNT_TOL or dr > RETURN_TOL) else small).append(
                (y, t1[y], t1a[y], dc, dr))

    print(f"cross-check on {len(shared)} shared years: "
          f"{len(small)} within tolerance, {len(large)} beyond")
    for y, a, b, dc, dr in small:
        print(f"  ok  {y}: Table 1 {a} vs 1a {b}  (d_count {dc}, d_ret {dr:.1f}pp)"
              f" -- separate revision dates")
    for y, a, b, dc, dr in large:
        print(f"  !!  {y}: Table 1 {a} vs 1a {b}  (d_count {dc}, d_ret {dr:.1f}pp)")
    if large:
        raise SystemExit(f"{len(large)} year(s) differ by more than "
                         f"{COUNT_TOL} IPOs or {RETURN_TOL}pp -- that is too "
                         f"large for a revision and suggests a column was "
                         f"misread. Nothing written.")

    years = sorted(t1)
    print(f"\n{'year':>6}{'IPOs':>7}{'EW 1st-day':>12}   pub_date")
    for y in years[:3] + years[-3:]:
        c, ew = t1[y]
        print(f"{y:>6}{c:>7}{ew:>11.1f}%   {y + 1}{PUB_MONTH_DAY}")

    if not args.apply:
        print(f"\nDRY RUN -- {2 * len(years)} rows would be written "
              f"(count + first-day return per year). Re-run with --apply.")
        return

    con = sqlite3.connect(args.db)
    n = 0
    for y in years:
        c, ew = t1[y]
        obs = f"{y}-12-31"
        pub = f"{y + 1}{PUB_MONTH_DAY}"
        for sid, val in (("RITTER_IPO_COUNT", float(c)),
                         ("RITTER_IPO_FIRSTDAY", ew)):
            con.execute("INSERT OR IGNORE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)",
                        (sid, obs, pub, val, "Ritter IPO-Statistics.pdf"))
            n += 1
    con.commit()
    tot = con.execute("SELECT COUNT(*) FROM data_vintages "
                      "WHERE series_id LIKE 'RITTER%'").fetchone()[0]
    con.close()
    print(f"\nwrote {n} rows; data_vintages RITTER* now {tot}")


if __name__ == "__main__":
    main()
