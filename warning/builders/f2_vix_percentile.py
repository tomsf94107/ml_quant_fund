"""
f2_vix_percentile.py — builder for F2, VIX percentile.

REGISTRY ROW (implemented verbatim):
    id              F2                      layer: L1      tier: dashboard
    formula         VIX vs 504d percentile
    data_source     Cboe VIX_History.csv    history_start: 1990-01-02
    threshold_arm   <20th
    threshold_red   <10th w/ L2>=0.5
    persistence     5 days                  max_staleness: 3 days
    direction       low_vol_complacent
    verdicts        2000: VXO mid-20s
                    2008: 9.89 Jan-07; 16.12 at peak
                    2022: 14.38 Feb-20 exhibit

DIRECTION IS INVERTED, WHICH IS THE POINT
    LOW volatility is the warning, not high. A VIX in the bottom decile of its own
    two-year distribution is complacency being priced. The report's defining
    exhibit is Oct-2007: VIX 16.12 at a record equity high while funding markets
    were already ruptured -- "the trap" (line 424).

THE RED CONDITION IS CROSS-LAYER
    `<10th w/ L2>=0.5` needs the L2 LAYER score, which a single builder cannot
    know. `l2_score` is therefore an argument. Passed None, F2 can ARM but not
    fire red, and says so -- the same discipline as S2's equity leg.

    Consequence for the daily driver: evaluation is TWO-PASS. Compute the L2
    signals, let the engine produce the L2 layer score, then compute F2 with it.
    F2 is L1 and no L2 signal depends on F2, so there is no circularity.

DATA ROUTE
    Registry names Cboe's VIX_History.csv. As of 2026-08-28 `cdn.cboe.com` is
    DNS-blackholed to 127.0.0.1 from the operator's host, so the series is pulled
    as FRED `VIXCLS` -- the same Cboe index redistributed, 1990-01-02 onward,
    matching the registry's history_start exactly. Same data, reachable route.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_bdays  # noqa: E402

SIGNAL_ID = "F2"
LAYER = "L1"
SERIES = "VIXCLS"
MAX_STALENESS_DAYS = 3
PERSISTENCE_DAYS = 5

AMBER_STATE = "Y"          # DECISIONS.md D1 (ratified)
WINDOW = 504               # registry: "504d percentile" (~2 trading years)
ARM_PCTILE = 20.0          # registry: "<20th"
RED_PCTILE = 10.0          # registry: "<10th ..."
RED_L2_MIN = 0.5           # registry: "... w/ L2>=0.5"


def percentile_of_last(values) -> float:
    """Percentile rank of the final value within `values`, 0-100.

    Fraction of the window at or below the current level. A value at the very
    bottom scores near 0, at the top near 100.
    """
    cur = values[-1]
    return 100.0 * sum(1 for v in values if v <= cur) / len(values)


def compute(con, asof, l2_score=None):
    rows = series_asof(con, SERIES, asof)
    if len(rows) < WINDOW:
        return _na(asof, f"need {WINDOW} obs of {SERIES}, have {len(rows)}")

    window = [v for _, v in rows[-WINDOW:]]
    vix = window[-1]
    pct = percentile_of_last(window)
    stale_days = staleness_bdays(con, SERIES, asof)

    armed = pct < ARM_PCTILE
    red_pct = pct < RED_PCTILE
    if red_pct and l2_score is not None and l2_score >= RED_L2_MIN:
        state = "R"
    elif armed:
        state = AMBER_STATE
    else:
        state = "G"

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state,
        "raw_value": vix,
        "zscore": None,
        "stale": (stale_days is None or stale_days > MAX_STALENESS_DAYS),
        "stale_days": stale_days,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": rows[-1][0],
        "detail": {
            "series": SERIES,
            "vix": round(vix, 2),
            "percentile_504d": round(pct, 1),
            "window_low": round(min(window), 2),
            "window_high": round(max(window), 2),
            "armed_below_20th": armed,
            "below_10th": red_pct,
            "l2_score": l2_score,
            "l2_note": (None if l2_score is not None else
                        "L2 layer score not supplied -- the red condition "
                        "(<10th WITH L2>=0.5) cannot be evaluated, so F2 can arm "
                        "but not fire red. The daily driver must pass it."),
        },
    }


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(
        signal_id=result["signal_id"], layer=result["layer"],
        state=result["state"], stale=bool(result["stale"]),
        min_persistence=result["persistence_days"])
