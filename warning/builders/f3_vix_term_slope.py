"""
f3_vix_term_slope.py — builder for F3, VIX term slope.

REGISTRY ROW:
    id F3 | layer L3 | role confirmer | tier dashboard
    formula        (VIX3M-VIX)/VIX and futures front-second
    data_source    Cboe csv + CFE          series: VIX3M csv; CFE_*.csv
    history_start  2004 (futures) / 2007-12 (VIX3M)
    publication_lag 0                      persistence: 5   max_staleness: 3
    threshold_arm  slope<0 1d              threshold_red: inverted >=5d
    direction      inversion_stress
    verdict_2000   impossible (no futures)
    verdict_2008   Aug-07 inversion; contango at top

ROLE IS 'CONFIRMER', NOT PREDICTOR -- AND THE REGISTRY SAYS WHY
    Report Part VII: "inverts in every correction (1997/2010/2011/2018) -- L3
    only". F3 marks stress that is already underway. Its 2008 verdict is the
    clearest statement of its limits: it inverted in August 2007 during the
    funding rupture, then returned to CONTANGO at the October top. A signal that
    is calm at the peak is not a predictor, and the registry places it in L3
    (imminent risk, forward-monitoring only) rather than L1/L2 for that reason.

DATA ROUTE
    VIX3M comes from FRED's VXVCLS, not Cboe's VIX3M_History.csv: verified
    2026-08-28, FRED starts 2007-12-04 -- matching the registry's stated
    history_start -- while the Cboe file starts only 2009-09-18. Same Cboe index,
    longer history via the redistributor. CBOE_VIX3M is loaded and available as a
    cross-check.

TWO LEGS, ONE PRIMARY (DECISIONS.md D14)
    vix3m     (VIX3M - VIX) / VIX          2007-12-04 onward
    futures   (SECOND - FRONT) / FRONT     2004-03-26 .. 2018-02-23, D13-normalized

    The registry's thresholds ("slope<0 1d", "inverted >=5d") are written against
    the VIX3M formula, so the vix3m leg is PRIMARY wherever it has fresh data.
    The futures leg carries dates before 2007-12 that the vix3m leg cannot reach
    -- which is the only reason F3's 2008 verdict ("Aug-07 inversion; contango at
    top") is testable at all. Where both exist, both are reported and the primary
    is named; they are never averaged, because averaging two term-structure
    measures with different tenors would produce a number matching neither.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "F3"
LAYER = "L3"
MAX_STALENESS_DAYS = 3
PERSISTENCE_DAYS = 5
AMBER_STATE = "Y"                 # DECISIONS.md D1

VIX_SERIES = "VIXCLS"
VIX3M_SERIES = "VXVCLS"           # FRED copy: 2007-12-04+, longer than the Cboe file
FUT_FRONT = "VX_FRONT"            # CFE, 2004-03-26..2018-02-23, D13-normalized
FUT_SECOND = "VX_SECOND"
RED_INVERTED_DAYS = 5             # registry: "inverted >=5d"
ARM_INVERTED_DAYS = 1             # registry: "slope<0 1d"


def _slopes(con, asof, long_s, short_s):
    """(date, slope) where slope = (longer_tenor - shorter) / shorter."""
    a, b = series_asof(con, long_s, asof), series_asof(con, short_s, asof)
    if not a or not b:
        return []
    return [(d, (x - y) / y) for d, x, y in align(a, b) if y]


def compute(con, asof):
    csv_slopes = _slopes(con, asof, VIX3M_SERIES, VIX_SERIES)
    fut_slopes = _slopes(con, asof, FUT_SECOND, FUT_FRONT)

    csv_fresh = bool(csv_slopes) and _fresh(con, asof, VIX3M_SERIES, VIX_SERIES)
    if csv_fresh:
        slopes, leg = csv_slopes, "vix3m"
    elif fut_slopes:
        slopes, leg = fut_slopes, "futures"
    else:
        return _na(asof, f"neither leg has data: {VIX3M_SERIES}/{VIX_SERIES} "
                         f"({len(csv_slopes)} obs) and {FUT_SECOND}/{FUT_FRONT} "
                         f"({len(fut_slopes)} obs)")

    if len(slopes) < RED_INVERTED_DAYS:
        return _na(asof, f"{leg} leg has {len(slopes)} obs, "
                         f"need {RED_INVERTED_DAYS}")

    cur_date, cur_slope = slopes[-1]
    # consecutive inverted days ending at the latest observation
    run = 0
    for _, s in reversed(slopes):
        if s < 0:
            run += 1
        else:
            break

    pair = (VIX3M_SERIES, VIX_SERIES) if leg == "vix3m" else (FUT_SECOND, FUT_FRONT)
    stale = max(x for x in (staleness_bdays(con, pair[0], asof),
                            staleness_bdays(con, pair[1], asof)) if x is not None)

    if run >= RED_INVERTED_DAYS:
        state = "R"
    elif run >= ARM_INVERTED_DAYS:
        state = AMBER_STATE
    else:
        state = "G"

    def _last(series):
        r = series_asof(con, series, asof)
        return round(r[-1][1], 2) if r else None

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": cur_slope, "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": cur_date,
        "detail": {
            "leg": leg, "legs_available": [n for n, s in
                                           (("vix3m", csv_slopes),
                                            ("futures", fut_slopes)) if s],
            "vix": _last(VIX_SERIES), "vix3m": _last(VIX3M_SERIES),
            "vx_front": _last(FUT_FRONT), "vx_second": _last(FUT_SECOND),
            "slope": round(cur_slope, 4),
            "slope_pct": round(cur_slope * 100, 2),
            "slope_vix3m_pct": (round(csv_slopes[-1][1] * 100, 2)
                                if csv_slopes else None),
            "slope_futures_pct": (round(fut_slopes[-1][1] * 100, 2)
                                  if fut_slopes else None),
            "inverted": cur_slope < 0,
            "inverted_run_days": run,
            "red_at_days": RED_INVERTED_DAYS,
        },
    }


def _fresh(con, asof, *series):
    """True when every named series has a published obs inside the stale limit."""
    for s in series:
        d = staleness_bdays(con, s, asof)
        if d is None or d > MAX_STALENESS_DAYS:
            return False
    return True


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof), "state": "NA",
            "raw_value": None, "zscore": None, "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(signal_id=result["signal_id"], layer=result["layer"],
                         state=result["state"], stale=bool(result["stale"]),
                         min_persistence=result["persistence_days"])
