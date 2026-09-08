#!/usr/bin/env python3
"""
warning/rebuild_persistence.py -- B1.

WHY
    schema_meta.engine_state carried per-signal counts up to 13 against 6 real
    asof_dates. The counter increments once per INVOCATION, not once per DATE:
    2026-08-28 was stepped six times, and signal_values/composite_scores use
    INSERT OR REPLACE so the rows collapsed to one while the counter did not.
    Persistence is the engine's hysteresis input, so an inflated count makes a
    signal reach effective state early.

    signal_values.effective_state has also never been written -- persist() puts
    a literal None in that column. The engine computes it and keeps it only
    inside the engine_state JSON blob, so there is no historical record of what
    the engine concluded on any past date.

WHAT THIS DOES
    Replays the stored per-date states through the REAL engine: resets
    EngineState, calls warning_engine.step() once per asof_date in order, and
    writes the resulting effective_state back to signal_values.

    It does NOT reimplement apply_persistence. A gaps-and-islands SQL recompute
    looks equivalent and is not: apply_persistence RETURNS EARLY on NA
    (warning_engine.py:119), so a run FREEZES across NA rather than resetting.
    That subtlety is invisible in a SQL sketch and is shipped, tested behaviour.

    Reads the STORED signal_values.state, not the builders. Re-deriving would
    give "what we would conclude today about that date", not what was observed.

SAFETY
    - Refuses to run without --write; default is a dry-run diff.
    - Refuses to run if any asof_date is not a real trading session.
    - No longer deletes alerts. It used to, because persist() used a plain
      INSERT and replay would duplicate them -- which destroyed real history:
      F2's 2026-09-03 transition survived in effective_state with no alert.
      B3 made alerts idempotent via a unique index and INSERT OR REPLACE, so
      the workaround is gone.

USAGE
    python warning/rebuild_persistence.py --db warning.db
    python warning/rebuild_persistence.py --db warning.db --write
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from utils.market_calendar import is_trading_day          # noqa: E402
from warning_engine import SignalReading, EngineState, step  # noqa: E402


def min_persistence_map() -> dict[str, int]:
    """signal_id -> persistence_days from the registry.

    Builders pass result["persistence_days"] into SignalReading.min_persistence,
    so this is the same source. Falls back to the dataclass default of 1 for
    ids absent from the registry (L4A-E are derived, not registry rows).
    """
    out: dict[str, int] = {}
    with open(os.path.join(_HERE, "signal_registry.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["id"]] = int(row["persistence_days"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_rows(con):
    """asof_date -> [(signal_id, layer, state, stale)], date-ordered."""
    days: dict[str, list] = {}
    for asof, sid, layer, state, stale in con.execute(
        "SELECT asof_date, signal_id, layer, state, stale "
        "FROM signal_values ORDER BY asof_date, signal_id"
    ):
        days.setdefault(asof, []).append((sid, layer, state, bool(stale)))
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.path.dirname(_HERE),
                                                 "warning.db"))
    ap.add_argument("--write", action="store_true",
                    help="apply. Default is a dry-run diff.")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    mp = min_persistence_map()
    days = load_rows(con)
    if not days:
        sys.exit("signal_values is empty; nothing to rebuild.")

    dates = sorted(days)
    bad = [d for d in dates if not is_trading_day(d)]
    if bad:
        sys.exit(f"non-trading asof_dates present: {bad}. "
                 f"Fix those rows before rebuilding persistence.")

    # Gaps are reported, not silently absorbed. Policy for what a gap DOES to a
    # run belongs in DECISIONS.md, not in this script.
    missing = []
    d0, d1 = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    step_d = d0
    while step_d <= d1:
        iso = step_d.isoformat()
        if is_trading_day(step_d) and iso not in days:
            missing.append(iso)
        step_d = date.fromordinal(step_d.toordinal() + 1)
    if missing:
        print(f"GAP: trading sessions with no row inside the replay span: "
              f"{missing}\n     Replay treats them as absent, so runs continue "
              f"across them.\n")

    print(f"replaying {len(dates)} dates: {dates[0]} .. {dates[-1]}\n")

    st = EngineState()
    updates: list[tuple[str, str, str]] = []
    for asof in dates:
        readings = [
            SignalReading(signal_id=sid, layer=layer, state=state, stale=stale,
                          min_persistence=mp.get(sid, 1))
            for sid, layer, state, stale in days[asof]
        ]
        res = step(asof, readings, st)
        for sid, (s, n, eff) in st.persistence.items():
            updates.append((eff, asof, sid))
        print(f"  {asof}  band {res.band:<18} "
              f"composite {res.composite if res.composite is not None else 'n/a'}")

    print("\nfinal per-signal persistence (state, consecutive days, effective):")
    for sid in sorted(st.persistence):
        s, n, eff = st.persistence[sid]
        flag = "" if eff == s else f"   <- not yet effective (needs {mp.get(sid, 1)})"
        print(f"  {sid:<5} {s}  {n:>3}d  eff {eff}{flag}")

    if not args.write:
        print(f"\nDRY RUN. {len(updates)} effective_state values computed, "
              f"none written. Re-run with --write to apply.")
        return

    con.executemany(
        "UPDATE signal_values SET effective_state=? "
        "WHERE asof_date=? AND signal_id=?", updates)
    blob = json.dumps({"band": st.band, "candidate_band": st.candidate_band,
                       "candidate_days": st.candidate_days,
                       "persistence": {k: list(v)
                                       for k, v in st.persistence.items()}})
    con.execute("INSERT OR REPLACE INTO schema_meta (key, value) "
                "VALUES ('engine_state', ?)", (blob,))
    con.commit()
    print(f"\nwrote {len(updates)} effective_state values; engine_state reset "
          f"to date-derived counts. Alerts untouched -- "
          f"idempotent since B3.")


if __name__ == "__main__":
    main()
