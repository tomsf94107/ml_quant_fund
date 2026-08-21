#!/usr/bin/env python3
"""
patch_price_cache_empty_gap.py -- add the missing `else` to the silent
empty-gap no-op in features/price_cache.py.

THE BUG (found 2026-08-14)
    if gap_start <= end_s:
        gap = fetch_raw_fn(ticker, gap_start, end_s)
        if gap is not None and not gap.empty:
            _write_raw(con, ticker, gap)
        <-- NO else. An empty vendor response is a SILENT no-op: no log, no
            retry, no alert. A feed can die and nothing ever says so.

IMPACT: 10 tickers accumulated trailing staleness undetected. CYBR was dead
~6 months (delisted 2026-02-11) and nothing surfaced it. The tickers were
found only by manually querying MAX(d) per ticker.

THE FIX: log a WARNING when the gap fetch comes back empty. One branch. This
is what would have surfaced CYBR in February instead of August.

Idempotent: re-running detects the patch is already applied and exits 0.
Backs up to features/price_cache.py.bak.emptygap.<date> before writing.

USAGE
  python scripts/patch_price_cache_empty_gap.py --dry-run
  python scripts/patch_price_cache_empty_gap.py
"""
import argparse
import os
import shutil
import sys
from datetime import date

import re

# Indentation-agnostic: capture the real indent from the file rather than
# hardcoding it (the block sits inside try/else and depth has changed before).
ANCHOR = re.compile(
    r"^(?P<i1>[ \t]+)if gap is not None and not gap\.empty:\n"
    r"(?P<i2>[ \t]+)_write_raw\(con, ticker, gap\)\n",
    re.MULTILINE)

MARKER = "EMPTY gap fetch"


def _build(m):
    i1, i2 = m.group("i1"), m.group("i2")
    return (
        f"{i1}if gap is not None and not gap.empty:\n"
        f"{i2}_write_raw(con, ticker, gap)\n"
        f"{i1}else:\n"
        f"{i2}# An empty gap fetch means the vendor served nothing for a window\n"
        f"{i2}# we believe should have bars. Silent before Aug 2026: 10 tickers\n"
        f"{i2}# went stale undetected, CYBR for ~6 months (delisted 2026-02-11).\n"
        f"{i2}log.warning(\n"
        f"{i2}    \"price_cache: EMPTY gap fetch %s %s..%s (last=%s) -- \"\n"
        f"{i2}    \"feed may be dead/delisted; run repair_stale_feeds.py\",\n"
        f"{i2}    ticker, gap_start, end_s, last)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
    path = args.path or os.path.join(root, "features", "price_cache.py")
    if not os.path.isfile(path):
        sys.exit(f"FATAL: not found: {path}")

    src = open(path).read()

    if MARKER in src:
        print(f"# already patched: {path}")
        return 0

    matches = list(ANCHOR.finditer(src))
    if not matches:
        print("# FAILED: anchor not found. The code may have changed.")
        print("# Expected, inside cached_daily():")
        print("#     if gap is not None and not gap.empty:")
        print("#         _write_raw(con, ticker, gap)")
        print("# Apply manually: add an `else:` that log.warning()s")
        print("# ticker, gap_start, end_s, last.")
        return 1
    if len(matches) > 1:
        print(f"# FAILED: anchor found {len(matches)} times -- ambiguous, patch manually.")
        return 1
    m = matches[0]

    if "import logging" not in src and "log = " not in src:
        print("# WARNING: no logger detected in this module; verify `log` exists.")

    replacement = _build(m)
    if args.dry_run:
        print(f"# DRY-RUN: would patch {path}  (line {src[:m.start()].count(chr(10)) + 1})")
        print("# ---- inserting ----")
        print(replacement[len(m.group(0)):])
        return 0

    bak = f"{path}.bak.emptygap.{date.today().isoformat()}"
    shutil.copy2(path, bak)
    open(path, "w").write(src[:m.start()] + replacement + src[m.end():])

    check = open(path).read()
    ok = MARKER in check
    print(f"# backup : {bak}")
    print(f"# patched: {path}  [{'OK' if ok else 'VERIFY FAILED'}]")
    if not ok:
        return 1
    print("# next: python -c 'import features.price_cache' to confirm it imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
