#!/usr/bin/env python3
"""
french_fire_value.py — do S7F/S8F fires precede worse outcomes than chance?

D21 recorded "roughly 3 hits in 13" for S7F. That number is meaningless without
a base rate: every month-end is followed by SOME drawdown, and a century
containing 1929, 1937, 1973, 1987, 2000 and 2008 has plenty. This computes the
forward return and worst drawdown after every fire and compares them with the
unconditional distribution over all scored month-ends.

Same test already applied to S5 on SPDR data, where RED's mean 126-day drawdown
(-8.94%) was indistinguishable from all new-high days (-8.05%) at n=11.

CALIBRATION ONLY (D20): French restates history, so these are not real-time
replays. The comparison is still valid -- both arms use the same data.

    python warning/french_fire_value.py --db warning.db
"""
import argparse
import os
import sqlite3
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders.s7_defensive_rotation import (RS_WINDOW as S7_W, RS_ARM, RS_RED,
                                            NEAR_HIGH_PCT, HIGH_WINDOW)
from builders.s8_epicenter_fracture import (RS_WINDOW as S8_W, DMA_WINDOW,
                                            ARM_DRAWDOWN, RED_DRAWDOWN,
                                            INDEX_NEAR_HIGH)

INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
DEFENSIVE = ["NoDur", "Utils", "Hlth"]
FWD = 126          # ~6 months
DD_WIN = 252       # worst drawdown over the following year


def load(con, sid):
    return con.execute("SELECT obs_date, value FROM data_vintages "
                       "WHERE series_id=? ORDER BY obs_date", (sid,)).fetchall()


def compound(rows):
    lvl, out = 100.0, []
    for d, r in rows:
        lvl *= (1.0 + r / 100.0)
        out.append((d, lvl))
    return out


def summarise(label, rows, n_total):
    if not rows:
        print(f"  {label:<26} n=0")
        return
    f = sorted(r[0] for r in rows)
    d = sorted(r[1] for r in rows)
    print(f"  {label:<26} n={len(rows):>4} ({100.0*len(rows)/n_total:>4.1f}%)  "
          f"fwd126 mean {st.mean(f):+6.2f}% median {st.median(f):+6.2f}%   "
          f"worst-252 mean {st.mean(d):+6.2f}% median {st.median(d):+6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    ex = dict(load(con, "FR_F:Mkt-RF"))
    rf = dict(load(con, "FR_F:RF"))
    dates = sorted(set(ex) & set(rf))
    mkt = [v for _, v in compound([(d, ex[d] + rf[d]) for d in dates])]
    ind = {}
    for n in INDUSTRIES:
        raw = dict(load(con, f"FR_I12_VW:{n}"))
        ind[n] = [v for _, v in compound([(d, raw[d]) for d in dates
                                          if d in raw])]
    con.close()

    ends = [i - 1 for i in range(1, len(dates))
            if dates[i][:7] != dates[i - 1][:7]]
    need = max(HIGH_WINDOW, S8_W + 1, DMA_WINDOW)

    all_rows, s7_red, s8_red = [], [], []
    for i in ends:
        if i < need or i + DD_WIN >= len(mkt):
            continue
        m, hi = mkt[i], max(mkt[i - HIGH_WINDOW + 1:i + 1])
        fwd = (mkt[i + FWD] - m) / m * 100
        worst = (min(mkt[i:i + DD_WIN]) - m) / m * 100
        all_rows.append((fwd, worst))

        mret7 = (m - mkt[i - S7_W]) / mkt[i - S7_W]
        rs = [(ind[n][i] - ind[n][i - S7_W]) / ind[n][i - S7_W] - mret7
              for n in DEFENSIVE]
        if m >= hi * (1 - NEAR_HIGH_PCT) and sum(rs) / len(rs) > RS_RED:
            s7_red.append((fwd, worst))

        mret8 = (m - mkt[i - S8_W]) / mkt[i - S8_W]
        rs8 = {n: (ind[n][i] - ind[n][i - S8_W]) / ind[n][i - S8_W] - mret8
               for n in INDUSTRIES}
        L = ind[max(rs8, key=rs8.get)]
        lhigh = max(L[i - HIGH_WINDOW + 1:i + 1])
        if (m >= hi * (1 - INDEX_NEAR_HIGH)
                and L[i] < sum(L[i - DMA_WINDOW + 1:i + 1]) / DMA_WINDOW
                and (lhigh - L[i]) / lhigh >= RED_DRAWDOWN):
            s8_red.append((fwd, worst))

    n = len(all_rows)
    print(f"\n{n} scored month-ends with a full forward window "
          f"({dates[0]}..{dates[-1]})\n")
    summarise("ALL month-ends", all_rows, n)
    summarise("S7F RED", s7_red, n)
    summarise("S8F RED", s8_red, n)
    print("\nA fire is only informative if its forward return is WORSE and its "
          "worst drawdown DEEPER than the unconditional row above.")


if __name__ == "__main__":
    main()
