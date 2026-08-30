#!/usr/bin/env python3
"""
french_century_scan.py — firing rates for S7F and S8F across 1930-2026.

REPLACES s8f_century_scan.py, WHICH WAS QUADRATIC.
    The first version called the builder once per month-end, and each call
    re-read and re-compounded all thirteen century-long series from SQLite --
    roughly 200 million row-operations and 15,000 queries to answer a question
    where the underlying data never changes. It took over four minutes.

    This version loads every series once, compounds once, and walks forward,
    which is the shape the problem actually has.

CALIBRATION ONLY (D20). French restates history when CRSP is revised, so a fire
date here is not a claim the signal would have fired in real time. What the scan
supports is a statement about FREQUENCY over a long sample -- which is the thing
neither S7 (1 fire in 10 years) nor S8 (1 fire in 10 years) has enough SPDR-era
data to establish.

    python warning/french_century_scan.py --db warning.db
"""
import argparse
import os
import sqlite3
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders.s7_defensive_rotation import (RS_WINDOW as S7_W, RS_ARM, RS_RED,
                                            NEAR_HIGH_PCT, HIGH_WINDOW)
from builders.s8_epicenter_fracture import (RS_WINDOW as S8_W, DMA_WINDOW,
                                            ARM_DRAWDOWN, RED_DRAWDOWN,
                                            INDEX_NEAR_HIGH, MIN_SECTORS)

INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
DEFENSIVE = ["NoDur", "Utils", "Hlth"]


def load(con, sid):
    return con.execute("SELECT obs_date, value FROM data_vintages "
                       "WHERE series_id=? ORDER BY obs_date", (sid,)).fetchall()


def compound(rows):
    lvl, out = 100.0, []
    for d, r in rows:
        lvl *= (1.0 + r / 100.0)
        out.append((d, lvl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    ex = dict(load(con, "FR_F:Mkt-RF"))
    rf = dict(load(con, "FR_F:RF"))
    dates = sorted(set(ex) & set(rf))
    mkt = compound([(d, ex[d] + rf[d]) for d in dates])
    mkt_lvl = [v for _, v in mkt]

    ind = {}
    for name in INDUSTRIES:
        raw = dict(load(con, f"FR_I12_VW:{name}"))
        ind[name] = [v for _, v in compound([(d, raw[d]) for d in dates
                                             if d in raw])]
    con.close()
    print(f"loaded {len(dates)} sessions {dates[0]}..{dates[-1]}, "
          f"{len(ind)} industries\n")

    # month-end index positions
    ends = []
    for i in range(1, len(dates)):
        if dates[i][:7] != dates[i - 1][:7]:
            ends.append(i - 1)

    s7 = {"G": 0, "Y": 0, "R": 0, "NA": 0}
    s8 = {"G": 0, "Y": 0, "R": 0, "NA": 0}
    s7_fires, s8_fires = [], []
    prev7 = prev8 = None

    need = max(HIGH_WINDOW, S8_W + 1, DMA_WINDOW)
    for i in ends:
        if i < need:
            s7["NA"] += 1
            s8["NA"] += 1
            continue
        d = dates[i]
        hi = max(mkt_lvl[i - HIGH_WINDOW + 1:i + 1])
        m = mkt_lvl[i]

        # ---- S7F ----
        mret7 = (m - mkt_lvl[i - S7_W]) / mkt_lvl[i - S7_W]
        rs = [(ind[n][i] - ind[n][i - S7_W]) / ind[n][i - S7_W] - mret7
              for n in DEFENSIVE]
        mean_rs = sum(rs) / len(rs)
        near7 = m >= hi * (1.0 - NEAR_HIGH_PCT)
        st7 = ("R" if (near7 and mean_rs > RS_RED) else
               "Y" if (near7 and mean_rs > RS_ARM) else "G")
        s7[st7] += 1
        if st7 == "R" and prev7 != "R":
            s7_fires.append((d, mean_rs * 100, 100 * (hi - m) / hi))
        prev7 = st7

        # ---- S8F ----
        mret8 = (m - mkt_lvl[i - S8_W]) / mkt_lvl[i - S8_W]
        rs8 = {n: (ind[n][i] - ind[n][i - S8_W]) / ind[n][i - S8_W] - mret8
               for n in INDUSTRIES}
        leader = max(rs8, key=rs8.get)
        lv = ind[leader]
        lhigh = max(lv[i - HIGH_WINDOW + 1:i + 1])
        ldd = (lhigh - lv[i]) / lhigh
        ldma = sum(lv[i - DMA_WINDOW + 1:i + 1]) / DMA_WINDOW
        near8 = m >= hi * (1.0 - INDEX_NEAR_HIGH)
        below = lv[i] < ldma
        st8 = ("R" if (near8 and below and ldd >= RED_DRAWDOWN) else
               "Y" if (near8 and below and ldd >= ARM_DRAWDOWN) else "G")
        s8[st8] += 1
        if st8 == "R" and prev8 != "R":
            s8_fires.append((d, leader, ldd * 100, 100 * (hi - m) / hi))
        prev8 = st8

    for label, counts, fires in (("S7F defensive rotation", s7, s7_fires),
                                 ("S8F epicenter fracture", s8, s8_fires)):
        scored = sum(counts.values()) - counts["NA"]
        print(f"=== {label} ===")
        print(f"  month-ends scored: {scored}   "
              + "  ".join(f"{k} {counts[k]} ({100.0*counts[k]/max(scored,1):.1f}%)"
                          for k in ("G", "Y", "R")))
        print(f"  {len(fires)} distinct RED episodes:")
        for f in fires:
            if len(f) == 3:
                print(f"    {f[0]}  mean RS {f[1]:+6.2f}%   "
                      f"market {f[2]:.2f}% below high")
            else:
                print(f"    {f[0]}  leader {f[1]:<7} DD {f[2]:5.1f}%   "
                      f"market {f[3]:.2f}% below high")
        print()


if __name__ == "__main__":
    main()
