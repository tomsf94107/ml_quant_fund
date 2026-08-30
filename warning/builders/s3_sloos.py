"""
s3_sloos.py — builder for S3, SLOOS lending standards.

REGISTRY ROW:
    id S3 | layer L2 | role predictor | tier shortlist
    formula   C&I net tightening: amber >+10, red >+20; CRE red >+30
    source    Fed/FRED         series: DRTSCILM (+CRE tables)
    history   1990-04-01       frequency: quarterly
    publication_lag ~5 weeks   max_staleness: 120 days
    persistence 1 obs          direction: tightening_bearish
    verdicts  2000 REPRODUCIBLE 1999-2000; 2008 fired Jul-07 +7.5 / Oct-07 +19.2;
              2022 silent
    notes     verified prints: 7.5 / 19.2 / 32.1 / 83.6 (2007Q3-2008Q4)

WHAT IT MEASURES
    The Senior Loan Officer Opinion Survey asks banks whether they tightened
    standards on commercial and industrial loans. DRTSCILM is the NET percentage
    tightening: positive means more banks tightened than eased. It is a direct
    read on credit availability from the people who supply it, which is why the
    report places it in L2 alongside the market-priced credit signals rather
    than treating it as macro colour.

    No z-score, no detrend: the registry specifies raw LEVELS. A net +20 means
    the same thing in 1998 and 2026, unlike a spread whose normal range shifts.
    That also makes S3 immune to the self-neutralization problem recorded in D12
    -- a level threshold cannot be absorbed by its own trailing window.

POINT-IN-TIME REACH IS 2010, NOT 1990
    The observations run to 1990-04-01, but ALFRED's earliest DRTSCILM VINTAGE
    is 2010-04-20 (verified 2026-08-30). Before that date ALFRED has no record of
    what was published, so pit.series_asof correctly returns nothing and S3 reads
    NA at 2000 and 2008.

    The registry's 2000 and 2008 verdicts are therefore NOT reproducible here,
    despite the underlying data existing. What CAN be checked is whether the
    stored VALUES match the registry's verified prints -- see run_signal.py's
    S3 anchors, which read the 2007-2008 observations from today's vantage.
    That validates the data, not the signal's real-time behaviour.

CRE LEG IS NOT BUILT
    The registry says "DRTSCILM (+CRE tables)" with a separate red at +30 for
    commercial real estate. No CRE series is ingested, so that leg is reported
    as unavailable rather than silently folded into the C&I reading. S3 is
    currently the C&I leg alone.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_days  # noqa: E402

SIGNAL_ID = "S3"
LAYER = "L2"
MAX_STALENESS_DAYS = 120           # quarterly survey; the registry's own limit
PERSISTENCE_DAYS = 1               # registry: "1 obs"
AMBER_STATE = "Y"                  # DECISIONS.md D1

CI_SERIES = "DRTSCILM"
CRE_SERIES = "DRTSCRE"             # not ingested; declared, never assumed
ARM_LEVEL = 10.0                   # registry threshold_arm: "+10"
RED_LEVEL = 20.0                   # registry threshold_red: "+20"
CRE_RED_LEVEL = 30.0               # registry: "CRE red >+30"


def compute(con, asof):
    rows = series_asof(con, CI_SERIES, asof)
    if not rows:
        return _na(asof, f"no visible observations for {CI_SERIES}. ALFRED's "
                         f"earliest vintage is 2010-04-20, so point-in-time "
                         f"reads before then see nothing even though the "
                         f"observations run to 1990.")

    obs_date, level = rows[-1]
    stale = staleness_days(con, CI_SERIES, asof)

    # CRE leg: declared unavailable rather than folded into the C&I reading.
    cre = series_asof(con, CRE_SERIES, asof)
    cre_level = cre[-1][1] if cre else None
    cre_fired = (cre_level is not None and cre_level > CRE_RED_LEVEL)

    if level > RED_LEVEL or cre_fired:
        state = "R"
    elif level > ARM_LEVEL:
        state = AMBER_STATE
    else:
        state = "G"

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": level, "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS),
        "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": obs_date,
        "detail": {
            "ci_net_tightening": round(level, 1),
            "arm_above": ARM_LEVEL, "red_above": RED_LEVEL,
            "survey_quarter": obs_date,
            "days_since_survey": stale,
            "cre_net_tightening": cre_level,
            "cre_red_above": CRE_RED_LEVEL,
            "cre_note": (None if cre else
                         f"{CRE_SERIES} not ingested -- the registry's CRE leg "
                         f"(red >+{CRE_RED_LEVEL:.0f}) is NOT evaluated. S3 is "
                         f"the C&I leg alone."),
            "pit_note": "ALFRED vintages for DRTSCILM begin 2010-04-20; before "
                        "that a point-in-time read returns nothing, so the "
                        "registry's 2000 and 2008 verdicts cannot be reproduced "
                        "here even though the observations exist.",
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
    return SignalReading(signal_id=result["signal_id"], layer=result["layer"],
                         state=result["state"], stale=bool(result["stale"]),
                         min_persistence=result["persistence_days"])
