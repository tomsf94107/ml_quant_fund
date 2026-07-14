#!/usr/bin/env python3
"""
scripts/prefetch_prices.py -- STAGE 0. Fetch every price bar BEFORE anything predicts.

THE BUG THIS FIXES
    Prices are fetched LAZILY, inside build_feature_dataframe(), which is called
    from inside the daily_runner prediction loop. So:
        Massive 429 -> massive_client returns an EMPTY frame (does not raise)
          -> price_cache serves the last good data (a stale panel)
            -> the model predicts on it
              -> daily_runner stamps run_date = _last_completed_session()  [CORRECT date]
                -> a Friday-priced prediction is published as Monday's signal.
    2026-07-13: 225 of 337 predictions were built on 2026-07-10 prices. Silent.

THE FIX
    Fetch is its own stage. It paces itself, retries, then VERIFIES that every
    ticker has a bar for the target session. If any are missing it exits non-zero
    and the pipeline stops. Nothing downstream ever sees a stale panel, because
    the cache is complete before the first prediction is made -- and with a warm
    cache + ML_QUANT_NO_TAIL_REFETCH=1 the runner makes ZERO API calls.

PACING
    massive_client's adaptive throttle is capped at 1.0s (60 req/min) and cannot
    back off further. If the real tier limit is below that, it 429s forever. This
    script paces itself INDEPENDENTLY of that throttle, so it works regardless.
        --sleep 12   ->  5 req/min   (free/basic tier)
        --sleep 1.0  -> 60 req/min
        --sleep 0.1  -> paid tier

USAGE
    python scripts/prefetch_prices.py --sleep 12          # fetch + verify
    python scripts/prefetch_prices.py --sleep 12 --check  # verify only, no fetch

EXIT
    0 = every ticker has a bar for the target session (or is a known no-data name)
    1 = incomplete. DO NOT PREDICT.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ML_QUANT_NO_TAIL_REFETCH", "1")   # never re-fetch what we have

import pandas as pd  # noqa: E402

from features.massive_client import download, _last_completed_session  # noqa: E402

PRICES = ROOT / "prices.db"
TICKERS_FILE = ROOT / "tickers.txt"


def load_tickers() -> list[str]:
    return [t.strip().upper() for t in TICKERS_FILE.read_text().splitlines()
            if t.strip() and not t.strip().startswith("#")]


def have_bar(target: str) -> set[str]:
    con = sqlite3.connect(f"file:{PRICES}?mode=ro", uri=True)
    s = {t for (t,) in con.execute(
        "SELECT DISTINCT ticker FROM raw_bars WHERE d = ?", (target,))}
    con.close()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=12.0,
                    help="seconds between tickers. 12 = 5 req/min (basic tier)")
    ap.add_argument("--check", action="store_true", help="verify only, no fetching")
    ap.add_argument("--start", default="2022-01-01")
    a = ap.parse_args()

    target = _last_completed_session().strftime("%Y-%m-%d")
    tickers = load_tickers()
    print(f"target session : {target}")
    print(f"universe       : {len(tickers)} tickers")
    print(f"pacing         : {a.sleep}s  (~{60/a.sleep:.0f} req/min)")

    got = have_bar(target)
    todo = [t for t in tickers if t not in got]
    print(f"already have   : {len(got)}")
    print(f"to fetch       : {len(todo)}\n")

    if a.check:
        missing = [t for t in tickers if t not in got]
        print(f"MISSING ({len(missing)}): {', '.join(missing[:40])}"
              f"{' ...' if len(missing) > 40 else ''}")
        sys.exit(0 if not missing else 1)

    if not todo:
        print("nothing to fetch. cache is complete.")
        sys.exit(0)

    ok, empty, err = 0, [], []
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            df = download(t, start=a.start, end=target, auto_adjust=True)
            if df is None or len(df) == 0:
                empty.append(t)
                status = "EMPTY"
            else:
                ok += 1
                status = f"{len(df)} bars"
        except Exception as e:
            err.append(t)
            status = f"ERR {str(e)[:40]}"
        el = time.time() - t0
        eta = (el / i) * (len(todo) - i) / 60
        print(f"  [{i:>3}/{len(todo)}] {t:<7} {status:<24} "
              f"eta {eta:>5.1f}m", flush=True)
        time.sleep(a.sleep)

    # ---- VERIFY: the whole point of this script -----------------------------
    got = have_bar(target)
    missing = [t for t in tickers if t not in got]

    print(f"\n{'='*66}")
    print(f"  fetched OK      : {ok}")
    print(f"  empty returns   : {len(empty)}  {empty[:12]}")
    print(f"  errors          : {len(err)}    {err[:12]}")
    print(f"  bars for {target}: {len(got)}/{len(tickers)}")
    print(f"{'='*66}")

    if missing:
        print(f"\n  INCOMPLETE -- {len(missing)} tickers have NO bar for {target}:")
        print(f"  {', '.join(missing[:50])}{' ...' if len(missing) > 50 else ''}")
        print("\n  DO NOT PREDICT. Re-run with a larger --sleep, or these names")
        print("  genuinely have no bar (halted/delisted) and belong on a skip list.")
        sys.exit(1)

    print("\n  COMPLETE. Every ticker has a bar. Safe to predict.")
    print("  The runner will now make ZERO API calls (warm cache + NO_TAIL_REFETCH).")
    sys.exit(0)


if __name__ == "__main__":
    main()
