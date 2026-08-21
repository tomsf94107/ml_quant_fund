#!/usr/bin/env python3
"""
patch_sector_etf_fallback.py -- give EVERY ticker a real sector benchmark.

THE DEFECT (2026-08-21)
  features/builder.py:1563
      sector_sym = SECTOR_ETF_MAP.get(ticker, SECTOR_ETF)
  SECTOR_ETF_MAP is a hand-maintained dict of ~456 tickers. Anything not in it
  silently falls back to the market ETF -- so "sector-relative" becomes
  "market-relative", the same number twice. Reports render 'SPY (sector)' beside
  'SPY (market)' and every sector_rel_ret for those names is beta, not sector.

  16 of 411 tickers are unmapped, and ALL SIXTEEN were added in the last two
  days: BLND BTQ CCJ CERO DY GEMI LEU MSTR MTZ NVT POWL RXT SCCO SLNH TRAW
  (+QQQ, itself a benchmark). add_ticker does not touch this map, so EVERY
  future add inherits the same gap. Patching 16 names would leave the mechanism
  intact; this patches the mechanism.

THE FIX
  Resolution order:  SECTOR_ETF_MAP  ->  bucket from tickers_metadata.csv  ->
                     SECTOR_ETF (market)
  The bucket taxonomy already exists (45 groups, zero empty as of 2026-08-21) and
  add_ticker writes it on every add -- so new tickers inherit a sector benchmark
  automatically from here on. The hand map is untouched and still wins, so no
  existing behaviour changes.

ALSO FIXED
  The bare `except Exception: df["sector_rel_ret"] = 0.0` swallowed an 8-second
  SIGALRM timeout and wrote a REAL-LOOKING ZERO the model then consumed as data.
  Now logs which ticker/symbol failed. Zero is retained as the value (changing it
  to NaN would alter model input); the point is that it stops being silent.

USAGE
  python scripts/patch_sector_etf_fallback.py --dry-run
  python scripts/patch_sector_etf_fallback.py
  python scripts/patch_sector_etf_fallback.py --verify
"""
import argparse
import os
import re
import shutil
import sys
from datetime import date

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
TARGET = os.path.join(ROOT, "features", "builder.py")

# Only ETFs that actually exist in the price DB -- verified against the universe
# union 2026-08-21. No XBI/ITA/CIBR: not carried, would silently fail to fetch.
BUCKET_ETF = {
    "Ad Tech": "XLC",                 "AI": "XLK",
    "Automotive": "XLY",              "Biotech": "XLV",
    "Commodities": "XLB",             "Consumer": "XLY",
    "Consumer Disc": "XLY",           "Consumer Staples": "XLP",
    "Consumer Tech": "XLK",           "Core Silicon": "SMH",
    "Crypto": "XLF",                  "Custom Silicon": "SMH",
    "Cybersecurity": "XLK",           "DC REIT": "XLRE",
    "Defense": "XLI",                 "E-commerce": "XLY",
    "Energy": "XLE",                  "Energy Storage": "XLI",
    "Enterprise Software": "IGV",     "Financials": "XLF",
    "Fintech": "XLF",                 "Healthcare": "XLV",
    "Hyperscaler": "XLK",             "Industrial Gases": "XLB",
    "Industrials": "XLI",             "Infrastructure": "XLI",
    "Market ETF": "SPY",              "Materials": "XLB",
    "Memory": "SMH",                  "Neoclouds": "XLK",
    "Networking": "XLK",              "Nuclear": "XLU",
    "Physical AI": "XLI",             "Power": "XLU",
    "Power/Industrial": "XLI",        "PropTech": "XLRE",
    "Quantum Computing": "XLK",       "REITs": "XLRE",
    "SaaS Victim": "IGV",             "Scientific Instruments": "XLV",
    "Sector ETF": "SPY",              "Semiconductor Equipment": "SMH",
    "Server Hardware": "XLK",         "Space Tech": "XLI",
    "Telecom": "XLC",
}

BLOCK = '''
# ── Bucket-derived sector ETF fallback (added 2026-08-21) ───────────────────
# SECTOR_ETF_MAP is hand-maintained and therefore always behind the universe:
# 16 of 411 tickers were unmapped, all added in the prior two days, and every
# one silently fell back to the MARKET etf -- making "sector-relative" identical
# to "market-relative". add_ticker writes tickers_metadata.csv on every add, so
# resolving through the bucket keeps new tickers covered automatically.
BUCKET_ETF_MAP = %(bucket_map)s


def _bucket_etf_lookup():
    """{TICKER: etf} derived from tickers_metadata.csv buckets. Cached."""
    global _BUCKET_ETF_CACHE
    try:
        return _BUCKET_ETF_CACHE
    except NameError:
        pass
    import csv as _csv, os as _os
    out = {}
    _p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                       "tickers_metadata.csv")
    try:
        with open(_p, newline="") as _f:
            _rows = list(_csv.reader(_f))
        _hdr = [h.strip().lower() for h in _rows[0]]
        _tc = next((i for i, h in enumerate(_hdr) if h in ("ticker", "symbol")), 0)
        _bc = next((i for i, h in enumerate(_hdr)
                    if h in ("bucket", "sector", "industry", "group")), None)
        if _bc is not None:
            for _r in _rows[1:]:
                if _r and len(_r) > max(_tc, _bc) and _r[_tc].strip():
                    _e = BUCKET_ETF_MAP.get(_r[_bc].strip())
                    if _e:
                        out[_r[_tc].strip().upper()] = _e
    except Exception as _e:
        log.warning("bucket ETF lookup unavailable (%%s); sector falls back to market", _e)
    _BUCKET_ETF_CACHE = out
    return out


def resolve_sector_etf(ticker):
    """SECTOR_ETF_MAP -> bucket -> market. Never silently duplicates the market."""
    _t = (ticker or "").upper()
    _e = SECTOR_ETF_MAP.get(_t)
    if _e:
        return _e
    _e = _bucket_etf_lookup().get(_t)
    if _e:
        return _e
    log.warning("no sector ETF for %%s (not in SECTOR_ETF_MAP, no usable bucket) "
                "-- sector-relative will equal market-relative", _t)
    return SECTOR_ETF
# ────────────────────────────────────────────────────────────────────────────
'''


def fmt_map():
    items = sorted(BUCKET_ETF.items())
    lines = ["{"]
    for k, v in items:
        lines.append(f'    "{k}": "{v}",')
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--path")
    args = ap.parse_args()
    target = args.path or TARGET
    if not os.path.isfile(target):
        sys.exit(f"FATAL: {target} not found")
    s = open(target).read()

    if args.verify:
        print("resolve_sector_etf present :", "def resolve_sector_etf" in s)
        print("call site patched          :", "resolve_sector_etf(ticker)" in s)
        print("timeout still silent       :",
              'except Exception:\n        df["sector_rel_ret"] = 0.0' in s)
        return 0

    if "def resolve_sector_etf" in s:
        print("# already patched")
        return 0

    anchor = re.search(r"^SECTOR_ETF_MAP\s*=\s*\{", s, re.M)
    if not anchor:
        sys.exit("FATAL: SECTOR_ETF_MAP not found")
    # end of the dict literal
    i = s.index("{", anchor.start())
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    else:
        sys.exit("FATAL: could not find end of SECTOR_ETF_MAP")

    block = BLOCK % {"bucket_map": fmt_map()}

    call_old = "sector_sym = SECTOR_ETF_MAP.get(ticker, SECTOR_ETF)"
    if s.count(call_old) != 1:
        sys.exit(f"FATAL: call site found {s.count(call_old)} times, expected 1")
    call_new = "sector_sym = resolve_sector_etf(ticker)"

    exc_old = ('    except Exception:\n'
               '        df["sector_rel_ret"] = 0.0')
    exc_new = ('    except Exception as _sec_e:\n'
               '        # was SILENT: an 8s SIGALRM timeout wrote a real-looking 0.0\n'
               '        # that the model consumed as data. Value kept (changing it to\n'
               '        # NaN would alter model input); it just stops being invisible.\n'
               '        log.warning("sector_rel_ret failed for %s vs %s (%s) -- set 0.0",\n'
               '                    ticker, sector_sym, type(_sec_e).__name__)\n'
               '        df["sector_rel_ret"] = 0.0')
    has_exc = exc_old in s

    if args.dry_run:
        print(f"# DRY-RUN on {target}")
        print(f"#   insert resolver after SECTOR_ETF_MAP (ends char {end})")
        print(f"#   rewrite call site: {call_old!r} -> {call_new!r}")
        print(f"#   timeout logging  : {'yes' if has_exc else 'ANCHOR NOT FOUND (skipped)'}")
        print(f"#   buckets mapped   : {len(BUCKET_ETF)}")
        return 0

    shutil.copy2(target, f"{target}.bak.sectoretf.{date.today().isoformat()}")
    out = s[:end] + "\n" + block + s[end:]
    out = out.replace(call_old, call_new)
    if has_exc:
        out = out.replace(exc_old, exc_new)
    open(target, "w").write(out)

    import py_compile
    try:
        py_compile.compile(target, doraise=True)
    except Exception as e:
        sys.exit(f"FATAL: patched file does not compile: {e}")
    print(f"# patched {target}")
    print(f"#   backup: {target}.bak.sectoretf.{date.today().isoformat()}")
    print(f"#   {len(BUCKET_ETF)} buckets mapped; timeout logging "
          f"{'added' if has_exc else 'SKIPPED (anchor not found)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
