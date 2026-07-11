#!/usr/bin/env python3
"""audit_splits.py -- a split record must be CONFIRMED BY THE TAPE.

A split with ratio R (split_to/split_from) must move the raw traded price by 1/R on
its ex-date. CRWD 1->4: raw 772.74 -> 193.98 = /3.98. CONFIRMED.
HON 2->1 should DOUBLE the price. Raw: 232.21 -> 227.80. CONTRADICTED -> the
adjustment manufactures a phantom -51% into adj_close, outcomes, and training labels.
"""
import sqlite3
con = sqlite3.connect("prices.db")
rows = con.execute("SELECT ticker, exec_date, split_from, split_to FROM splits "
                   "WHERE exec_date >= '2022-01-01' ORDER BY exec_date DESC").fetchall()
print(f"{'ticker':7s} {'ex_date':11s} {'ratio':>7s} {'prev_close':>11s} {'close':>9s} "
      f"{'gap':>7s} {'expect':>7s}  verdict")
print("-" * 82)
bad = []
for tk, ed, sf, st in rows:
    if not sf or not st: continue
    R = st / sf
    p = con.execute("SELECT close FROM raw_bars WHERE ticker=? AND d<? ORDER BY d DESC LIMIT 1",
                    (tk, ed)).fetchone()
    c = con.execute("SELECT close FROM raw_bars WHERE ticker=? AND d>=? ORDER BY d ASC LIMIT 1",
                    (tk, ed)).fetchone()
    if not p or not c or not p[0] or not c[0]: continue
    gap = c[0] / p[0]          # observed
    exp = 1.0 / R              # expected
    ok = abs(gap / exp - 1.0) < 0.25
    if not ok: bad.append((tk, ed, R, p[0], c[0], gap, exp))
    print(f"{tk:7s} {ed:11s} {R:7.3f} {p[0]:11.2f} {c[0]:9.2f} {gap:7.3f} {exp:7.3f}  "
          f"{'ok' if ok else 'CONTRADICTED <<<'}")
print()
print(f"  {len(bad)} SPURIOUS split records (tape does not confirm them):")
for tk, ed, R, p, c, g, e in bad:
    print(f"    {tk} {ed}  ratio {R:.3f} -> expected gap {e:.3f}, tape says {g:.3f}")
con.close()
