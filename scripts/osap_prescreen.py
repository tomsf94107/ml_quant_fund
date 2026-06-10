#!/usr/bin/env python3
"""OSAP pre-screen: published gross L/S decay for candidate signals BEFORE building.
Usage: python scripts/osap_prescreen.py Mom12m STreversal  (acronyms; no args = list all)
Data: data/osap/ (Chen-Zimmermann Oct-2025 release). GROSS, long-short, full CRSP --
apply Chen-Velikov haircut mentally: avg anomaly nets ~4bps/mo after costs."""
import sys, pandas as pd
ls = pd.read_csv("data/osap/PredictorPortsFull.csv")
ls = ls[ls["port"]=="LS"]; ls["date"]=pd.to_datetime(ls["date"])
names = sorted(ls["signalname"].unique())
if len(sys.argv)<2:
    print("\n".join(names)); sys.exit()
print(f"{'signal':28s} {'full':>7s} {'2015+':>7s} {'2020+':>7s}  n")
for s in sys.argv[1:]:
    m = [n for n in names if n.lower()==s.lower()] or [n for n in names if s.lower() in n.lower()]
    for n in m or [s]:
        d = ls[ls["signalname"]==n]
        if d.empty: print(f"{n:28s}  NOT FOUND"); continue
        f=d['ret'].mean()*12; p=d[d['date']>='2015-01-01']['ret'].mean()*12; q=d[d['date']>='2020-01-01']['ret'].mean()*12
        print(f"{n:28s} {f:+7.2f} {p:+7.2f} {q:+7.2f}  {len(d)}")
