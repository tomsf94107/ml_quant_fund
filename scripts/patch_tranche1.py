#!/usr/bin/env python3
"""
scripts/patch_tranche1.py -- the guards. DRY-RUN BY DEFAULT.

WHAT IT FIXES

  [1] HOLIDAY BLINDNESS
      features/builder.py:707  and  features/massive_client.py:465  each hold a
      COPY of _last_completed_session(), both doing only:
          while _d.weekday() >= 5: _d -= 1
      Weekends. No holidays. Measured damage:
          2026-06-19 Juneteenth   387 predictions  0 bars  1,138 outcomes
          2026-07-03 July-4 obs.  398 predictions  0 bars  1,186 outcomes
      Both are repointed at utils.market_calendar (validated: 1,134 sessions,
      zero disagreements with raw_bars). 13 call sites, one source of truth.

  [2] is_trading_day() in scripts/daily_runner.py
      Its own docstring: "basic check, ignores holidays". Repointed.

  [3] STALE-PANEL PREDICTIONS  <-- the one that caused 2026-07-13
      daily_runner.py:~400 already has a pre-check. It asks the WRONG QUESTION:
          if _check_df.empty:  skip
      On a 429, price_cache serves cached bars through the LAST GOOD DAY. The
      frame is NOT empty. The check passes. A stale panel is built, the model
      predicts on it, and the signal publishes stamped with today's date.
      2026-07-13: 225 of 337 predictions were computed from 2026-07-10 prices.

      The fix asks the RIGHT question: does the last bar equal run_date?

      RULE-8: build_feature_dataframe is called TWICE -- the main loop (~412)
      and the watchlist loop (~716). The watchlist path has NO pre-check at all.
      Both are guarded.

USAGE
    python scripts/patch_tranche1.py           # dry run: show every diff
    python scripts/patch_tranche1.py --apply   # write (backs up each file first)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── edit 1: massive_client._last_completed_session -> delegate ───────────────
MC_OLD = '''def _last_completed_session():
    """Last US trading day whose 17:00 ET close + publish margin has passed.
    Mirrors builder._last_completed_session() so both modules agree. Replaces
    date.today() (Mac VN local date = a day ahead of US calendar), which caused
    requests for not-yet-traded / plan-unauthorized bars (403 NOT_AUTHORIZED)."""
    from datetime import timedelta as _td
    try:
        from utils.timezone import now_et as _now_et
        _et = _now_et()
    except Exception:
        _et = datetime.datetime.now()  # last resort; better than VN date.today()
    _d = _et.date()
    if _et.hour < 17:
        _d = _d - _td(days=1)
    while _d.weekday() >= 5:
        _d = _d - _td(days=1)
    return _d'''

MC_NEW = '''def _last_completed_session():
    """Last US trading day whose 17:00 ET close + publish margin has passed.

    2026-07-14: was weekends-only (`while _d.weekday() >= 5`). It returned
    Juneteenth and July-4-observed as trading days -> 785 predictions and 2,324
    outcomes written for sessions that never happened. Now delegates to
    utils.market_calendar, which is HOLIDAY-AWARE and validated against 1,134
    real sessions in raw_bars (zero disagreements, both directions).
    """
    from utils.market_calendar import last_completed_session as _lcs
    return _lcs()'''

# ── edit 2: builder._last_completed_session -> delegate ──────────────────────
B_OLD = '''def _last_completed_session():
    """Last US trading day whose close has passed (17:00 ET, with publish-delay
    margin). Replaces _date.today(), which used the Mac's VN local date — a day
    ahead of the US trading calendar — causing requests for not-yet-traded /
    plan-unauthorized bars (the 403 NOT_AUTHORIZED storm)."""
    from datetime import timedelta as _td
    from utils.timezone import now_et as _now_et
    _et = _now_et()
    _d = _et.date()
    if _et.hour < 17:
        _d = _d - _td(days=1)
    while _d.weekday() >= 5:  # Sat=5, Sun=6 -> walk back to Friday
        _d = _d - _td(days=1)
    return _d'''

B_NEW = '''def _last_completed_session():
    """Last US trading day whose close has passed (17:00 ET + publish margin).

    2026-07-14: was a DUPLICATE of massive_client._last_completed_session, and
    both were weekends-only. Now both delegate to utils.market_calendar --
    one source of truth, holiday-aware, validated against raw_bars.
    """
    from utils.market_calendar import last_completed_session as _lcs
    return _lcs()'''

# ── edit 3: daily_runner.is_trading_day -> delegate ──────────────────────────
DR_TD_OLD = '''def is_trading_day() -> bool:
    """Return True if today is a weekday (basic check, ignores holidays)."""
    from datetime import date as _date
    t = today_et()
    if isinstance(t, str):
        return _date.fromisoformat(t).weekday() < 5
    return t.weekday() < 5'''

DR_TD_NEW = '''def is_trading_day() -> bool:
    """Return True if today (ET) is a US trading day -- HOLIDAYS INCLUDED.

    2026-07-14: the old body was `weekday() < 5` and its own docstring said
    "ignores holidays". It let the runner fire on Juneteenth and July-4-observed,
    producing 785 predictions with no entry price. Now holiday-aware.
    """
    from utils.market_calendar import is_trading_day as _itd
    return _itd(today_et())'''

# ── edit 4: the stale-panel guard, MAIN loop ────────────────────────────────
DR_G1_OLD = '''                if _check_df.empty:
                    log.warning(f"  ⚠ {ticker} has no Massive data — skipping (avoid yfinance fallback chain)")
                    failed.append(ticker)
                    time.sleep(SLEEP_BETWEEN)
                    continue'''

DR_G1_NEW = '''                if _check_df.empty:
                    log.warning(f"  ⚠ {ticker} has no Massive data — skipping (avoid yfinance fallback chain)")
                    failed.append(ticker)
                    time.sleep(SLEEP_BETWEEN)
                    continue
                # ── STALE-PANEL GUARD (added 2026-07-14) ─────────────────────
                # The .empty check above asks "is there ANY data?". On a 429,
                # massive_client returns an empty frame WITHOUT raising, and
                # price_cache then serves the last good bars. The frame is NOT
                # empty -- it is STALE. On 2026-07-13, 225 of 337 predictions
                # were computed from 2026-07-10 prices and published as Monday's
                # signals. Every job reported OK and exited 0.
                # The right question is: does the newest bar equal run_date?
                try:
                    _last_bar = _check_df.index[-1].strftime("%Y-%m-%d")
                except Exception:
                    _last_bar = str(_check_df.index[-1])[:10]
                if _last_bar != run_date:
                    log.error(f"  🔴 {ticker} STALE PANEL: newest bar {_last_bar} "
                              f"!= run_date {run_date} — REFUSING to predict")
                    failed.append(ticker)
                    time.sleep(SLEEP_BETWEEN)
                    continue'''

# ── edit 5: the stale-panel guard, WATCHLIST loop (has NO check at all) ─────
DR_G2_OLD = '''            try:
                from features.builder import build_feature_dataframe
                from signals.generator import generate_signals
                df = build_feature_dataframe(ticker, start_date=TRAIN_START)'''

DR_G2_NEW = '''            try:
                from features.builder import build_feature_dataframe
                from signals.generator import generate_signals
                df = build_feature_dataframe(ticker, start_date=TRAIN_START)
                # ── STALE-PANEL GUARD (added 2026-07-14) ─────────────────────
                # RULE-8: the main loop's pre-check never existed here at all.
                # A guard on one of two identical paths is not a guard.
                if df is None or df.empty:
                    log.error(f"  🔴 {ticker} watchlist: empty panel — skipping")
                    continue
                _wl_last = str(df["date"].iloc[-1])[:10]
                if _wl_last != run_date:
                    log.error(f"  🔴 {ticker} watchlist STALE PANEL: newest bar "
                              f"{_wl_last} != run_date {run_date} — REFUSING")
                    continue'''

EDITS = [
    ("features/massive_client.py", MC_OLD,    MC_NEW,    "massive_client._last_completed_session -> market_calendar"),
    ("features/builder.py",        B_OLD,     B_NEW,     "builder._last_completed_session -> market_calendar (dedup)"),
    ("scripts/daily_runner.py",    DR_TD_OLD, DR_TD_NEW, "is_trading_day -> holiday-aware"),
    ("scripts/daily_runner.py",    DR_G1_OLD, DR_G1_NEW, "STALE-PANEL GUARD, main loop"),
    ("scripts/daily_runner.py",    DR_G2_OLD, DR_G2_NEW, "STALE-PANEL GUARD, watchlist loop"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    print("=" * 74)
    print(f"  TRANCHE 1 -- {'APPLY' if a.apply else 'DRY RUN'}")
    print("=" * 74)

    # verify every anchor BEFORE writing anything
    ok = True
    for path, old, _new, label in EDITS:
        p = ROOT / path
        if not p.exists():
            print(f"  MISSING FILE : {path}")
            ok = False
            continue
        n = p.read_text().count(old)
        mark = "OK" if n == 1 else "FAIL"
        print(f"  [{mark:>4}] {label}")
        print(f"         {path}  anchor found {n}x (need exactly 1)")
        if n != 1:
            ok = False

    if not ok:
        print("\n  ABORT: an anchor is missing or ambiguous. The source has drifted")
        print("  from what was audited. NOTHING WAS WRITTEN. Re-audit before patching.")
        sys.exit(1)

    if not a.apply:
        print("\n  All 5 anchors verified. Re-run with --apply to write.")
        print("  Each file is backed up to <file>.bak.tranche1.<stamp> first.")
        sys.exit(0)

    touched = set()
    for path, old, new, label in EDITS:
        p = ROOT / path
        if path not in touched:
            shutil.copy2(p, f"{p}.bak.tranche1.{stamp}")
            touched.add(path)
        p.write_text(p.read_text().replace(old, new, 1))
        print(f"  PATCHED  {path}  <-  {label}")

    print("\n  Backups: " + ", ".join(f"{t}.bak.tranche1.{stamp}" for t in sorted(touched)))
    print("\n  VERIFY NOW:")
    print("    python -c \"import ast;[ast.parse(open(f).read()) for f in "
          "['features/massive_client.py','features/builder.py','scripts/daily_runner.py']];"
          "print('SYNTAX OK')\"")
    print("    python -c \"from features.massive_client import _last_completed_session as a;"
          "from features.builder import _last_completed_session as b;"
          "print('lcs:', a(), b(), 'agree:', a()==b())\"")
    print("    python -c \"import sys;sys.path.insert(0,'.');"
          "from scripts.daily_runner import is_trading_day;print('today trading:', is_trading_day())\"")


if __name__ == "__main__":
    main()
