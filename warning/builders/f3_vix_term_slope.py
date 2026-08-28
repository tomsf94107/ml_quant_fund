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

THE FUTURES LEG IS NOT BUILT
    "futures front-second" needs CFE per-contract settlement files (2004+), which
    fetch_free_history pulls only under --only cfe (~276 requests) and which no
    parser yet loads into data_vintages. Without it F3 cannot reach back to 2004,
    and the 2000-era verdict is "impossible (no futures)" regardless. The reading
    declares which legs were used; the futures leg is reported as absent rather
    than silently omitted.
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
RED_INVERTED_DAYS = 5             # registry: "inverted >=5d"
ARM_INVERTED_DAYS = 1             # registry: "slope<0 1d"


def compute(con, asof):
    vix = series_asof(con, VIX_SERIES, asof)
    v3m = series_asof(con, VIX3M_SERIES, asof)
    if not vix or not v3m:
        missing = [s for s, r in ((VIX_SERIES, vix), (VIX3M_SERIES, v3m)) if not r]
        return _na(asof, f"no visible observations for {','.join(missing)}")

    joined = align(v3m, vix)                       # (date, vix3m, vix)
    if len(joined) < RED_INVERTED_DAYS:
        return _na(asof, f"need {RED_INVERTED_DAYS} overlapping obs of "
                         f"{VIX3M_SERIES}/{VIX_SERIES}, have {len(joined)}")

    slopes = [(d, (a - b) / b) for d, a, b in joined if b]
    if not slopes:
        return _na(asof, f"{VIX_SERIES} is zero on every overlapping date")

    cur_date, cur_slope = slopes[-1]
    # consecutive inverted days ending at the latest observation
    run = 0
    for _, s in reversed(slopes):
        if s < 0:
            run += 1
        else:
            break

    stale = max(x for x in (staleness_bdays(con, VIX_SERIES, asof),
                            staleness_bdays(con, VIX3M_SERIES, asof)) if x is not None)

    if run >= RED_INVERTED_DAYS:
        state = "R"
    elif run >= ARM_INVERTED_DAYS:
        state = AMBER_STATE
    else:
        state = "G"

    v3m_v, vix_v = joined[-1][1], joined[-1][2]
    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": cur_slope, "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": cur_date,
        "detail": {
            "vix": round(vix_v, 2), "vix3m": round(v3m_v, 2),
            "slope": round(cur_slope, 4),
            "slope_pct": round(cur_slope * 100, 2),
            "inverted": cur_slope < 0,
            "inverted_run_days": run,
            "red_at_days": RED_INVERTED_DAYS,
            "legs_used": [VIX3M_SERIES, VIX_SERIES],
            "futures_leg": None,
            "futures_note": "CFE front-second settles not ingested; the futures "
                            "leg of the formula is absent, so F3 starts 2007-12 "
                            "rather than 2004",
        },
    }


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
