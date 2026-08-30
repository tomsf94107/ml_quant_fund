#!/usr/bin/env python3
"""
fetch_french_portfolios.py — Ken French daily portfolios, the registry's own
pre-2003 fallback for S6 and S8.

WHY THIS EXISTS
    Tested 2026-08-30: Massive's history is capped at 2016-07-18 regardless of
    the requested start date. Every price-derived signal in this build -- S2's
    equity leg, S5, S6, S7, S8, S14 leg (a) -- is therefore confined to a decade
    containing no credit crisis. That is the mechanical cause of D17, and it
    cannot be fixed inside that data source.

    But the registry already names an alternative for two of them:
        S6  "RSP/SPXEW 2003+; French size pre-2003"
        S8  "own universe / sector ETFs / French industries"

    French's library is free, public, survivorship-safe by construction (it is
    built from CRSP including delisted names), and daily back to 1926 -- which
    covers 1929, 1973, 1987, 2000 and 2008. Using it is following the
    specification, not substituting for it.

WHAT IS FETCHED
    49_Industry_Portfolios_daily        -> S8's leader/RS legs
    Portfolios_Formed_on_ME_daily       -> S6's equal- vs value-weight legs

    Both files carry MULTIPLE tables in one CSV (value-weighted returns, then
    equal-weighted, then firm counts, then average firm size), separated by
    blank lines and headed by prose. Nothing is parsed here: this script only
    downloads and reports structure, so the parser can be cut against the real
    layout rather than a guess. Same discipline used for the Cboe and CFE files,
    both of which had formats nothing like what was assumed.

RETURNS, NOT PRICES
    French publishes daily RETURNS in percent, with -99.99 and -999 as missing
    markers. S6 wants a relative return, which returns give directly. S8 needs
    drawdown-from-high and a 200-day average, which require a price index built
    by compounding -- a step that must be done from a fixed base and documented,
    since a compounded index is not the same object as a traded price.

USAGE
    python warning/fetch_french_portfolios.py --out data/raw/french
    python warning/fetch_french_portfolios.py --out data/raw/french --inspect
"""
import argparse
import io
import os
import urllib.request
import zipfile

BASE = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/")
FILES = {
    "49_Industry_Portfolios_daily_CSV.zip": "S8: industry RS and leader",
    "Portfolios_Formed_on_ME_daily_CSV.zip": "S6: size portfolios, EW vs VW",
}
UA = {"User-Agent": "warning-system/1.0"}


def fetch(name, out_dir, timeout=120):
    url = BASE + name
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = r.read()
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, name)
    with open(zpath, "wb") as f:
        f.write(blob)
    members = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for m in z.namelist():
            z.extract(m, out_dir)
            members.append((m, z.getinfo(m).file_size))
    return zpath, members


def inspect(path, n=28):
    print(f"\n--- {os.path.basename(path)} ---")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(f"  {i:>3}| {line.rstrip()[:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/french")
    ap.add_argument("--inspect", action="store_true",
                    help="print the head of each extracted CSV so the parser "
                         "can be written against the real layout")
    args = ap.parse_args()

    extracted = []
    for name, why in FILES.items():
        print(f"[{name}]  {why}")
        try:
            zpath, members = fetch(name, args.out)
            for m, size in members:
                print(f"  extracted {m}  ({size/1e6:.1f} MB)")
                if m.lower().endswith(".csv"):
                    extracted.append(os.path.join(args.out, m))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    if args.inspect:
        for p in extracted:
            inspect(p)
    elif extracted:
        print(f"\n{len(extracted)} CSV(s) extracted. Re-run with --inspect to "
              f"see their structure before any parser is written.")


if __name__ == "__main__":
    main()
