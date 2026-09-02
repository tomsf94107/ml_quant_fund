#!/usr/bin/env python3
"""
reconcile_intraday.py — cron entry point for intraday outcome reconciliation.

WHY THIS EXISTS
    reconcile_intraday_outcomes() had exactly one caller: ui/1_Dashboard.py.
    Outcomes were therefore only written when someone opened that Streamlit
    page. Combined with a bare `except Exception: pass`, the function stopped
    working after the Polygon -> Massive migration and nobody could tell --
    15,692 predictions went unscored.

    This violates the project's own point-in-time rule: a feature is not done
    until its data source is on a cron, not a manual UI button.

CRON (VN-anchored, matching the existing convention in
scripts/crontab_VN_anchored.txt). Run twice per session: once mid-session and
once after the close, so a failed run is visible the same day rather than at the
next dashboard visit.

    0 23 * * 1-5  ... scripts/reconcile_intraday.py   # 12:00 ET, mid-session
    0 6 * * 2-6   ... scripts/reconcile_intraday.py   # 19:00 ET prev day, post-close

EXIT CODES
    0  ran, whatever it reconciled
    1  the function raised -- cron mail / log will show it

Note the backlog before roughly the last few weeks is NOT recoverable: Massive's
1-minute history has a limited lookback and April-June bars now return empty.
Those rows stay unscored permanently. Running on a cron is what stops the next
gap forming.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from utils.timezone import ts_et
    print(f"[reconcile_intraday] start {ts_et()} ET")
    try:
        from accuracy.sink import reconcile_intraday_outcomes
        reconcile_intraday_outcomes()
    except Exception as e:
        import traceback
        print(f"[reconcile_intraday] RAISED {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print(f"[reconcile_intraday] done {ts_et()} ET")
    return 0


if __name__ == "__main__":
    sys.exit(main())
