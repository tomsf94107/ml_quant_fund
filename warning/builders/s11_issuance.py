"""
s11_issuance.py — builder for S11, Issuance quantity & quality.

REGISTRY ROW:
    id S11 | layer L1 | tier shortlist | history_start 1980
    formula  equity leg: IPO count + mean first-day return in top decile
             (hist to date). credit leg: HY share of gross corp issuance
    source   Ritter site; SIFMA      series: annual/qtrly tables
    arm      top quintile            red: top decile
    frequency annual                 direction: hot_issuance_bearish

WHY THIS SIGNAL IS DIFFERENT FROM EVERY OTHER ONE BUILT SO FAR
    Its data begins in 1980 and it is annual, so it spans 2000 AND 2008. Every
    signal built after S1 has been confined to a post-2010 or post-2016 sample
    with no credit crisis in it -- the core of D17. S11 can actually be checked
    against both bubbles.

PERCENTILES ARE EXPANDING, NOT FULL-SAMPLE
    "top decile (hist to date)" is read literally: at each evaluation year the
    percentile is computed against every year up to and including that one, and
    nothing after. Ranking against the full 1980-2025 sample would tell 1999
    what 2021 looked like. That is the same look-ahead the pub_date discipline
    exists to prevent, and it would flatter the signal precisely at the peaks.

    Consequence worth stating: early years rank against a handful of
    observations, so a percentile there is nearly meaningless. MIN_YEARS holds
    the signal at NA until the history can support a decile at all.

BOTH LEGS MUST BE HOT, WHICH IS THE POINT
    The registry's equity leg is "IPO count AND mean first-day return" -- volume
    and mania together. A high count alone is a healthy new-issue market; a high
    first-day pop alone can be a handful of small deals. The conjunction is what
    distinguishes a bubble from an active calendar.

CREDIT LEG NOT BUILT
    "HY share of gross corporate issuance" comes from SIFMA, which is not
    ingested. That leg is reported unavailable rather than folded into the
    equity reading. S11 is currently the equity leg alone.

PUBLICATION LAG
    parse_ritter stamps pub_date as 31 March of the following year, because
    Ritter publishes each year's figures the following spring. A point-in-time
    read in, say, January 2000 therefore sees 1998 and earlier -- not the 1999
    data that had not yet been published. This matters enormously for the 2000
    anchor: the mania year is invisible in real time until spring 2000.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_days  # noqa: E402

SIGNAL_ID = "S11"
LAYER = "L1"
MAX_STALENESS_DAYS = 550           # annual series published each spring
PERSISTENCE_DAYS = 1
AMBER_STATE = "Y"                  # DECISIONS.md D1

COUNT_SERIES = "RITTER_IPO_COUNT"
POP_SERIES = "RITTER_IPO_FIRSTDAY"
ARM_PCTILE = 80.0                  # registry threshold_arm: "top quintile"
RED_PCTILE = 90.0                  # registry threshold_red: "top decile"
MIN_YEARS = 10                     # a decile over fewer years is meaningless


def _pctile(values, value):
    return 100.0 * sum(1 for v in values if v <= value) / len(values)


def compute(con, asof):
    counts = series_asof(con, COUNT_SERIES, asof)
    pops = series_asof(con, POP_SERIES, asof)
    if not counts or not pops:
        missing = [s for s, r in ((COUNT_SERIES, counts), (POP_SERIES, pops))
                   if not r]
        return _na(asof, f"no visible observations for {','.join(missing)}")
    if len(counts) < MIN_YEARS or len(pops) < MIN_YEARS:
        return _na(asof, f"need {MIN_YEARS} published years for an expanding "
                         f"decile, have {min(len(counts), len(pops))}")

    # Expanding percentile: ranked against history TO DATE only.
    cvals = [v for _, v in counts]
    pvals = [v for _, v in pops]
    c_pct = _pctile(cvals, cvals[-1])
    p_pct = _pctile(pvals, pvals[-1])
    worst = min(c_pct, p_pct)      # BOTH legs must be hot; the weaker one binds

    if worst >= RED_PCTILE:
        state = "R"
    elif worst >= ARM_PCTILE:
        state = AMBER_STATE
    else:
        state = "G"

    stale = staleness_days(con, COUNT_SERIES, asof)
    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": pvals[-1], "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS),
        "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": counts[-1][0],
        "detail": {
            "reference_year": counts[-1][0][:4],
            "ipo_count": int(cvals[-1]),
            "ipo_count_pctile": round(c_pct, 1),
            "first_day_return_pct": round(pvals[-1], 1),
            "first_day_pctile": round(p_pct, 1),
            "binding_pctile": round(worst, 1),
            "arm_at": ARM_PCTILE, "red_at": RED_PCTILE,
            "years_in_history": len(cvals),
            "pctile_note": "expanding: ranked against years published up to this "
                           "date only, never the full sample",
            "credit_leg": None,
            "credit_note": "SIFMA HY share of gross corporate issuance is not "
                           "ingested; the registry's credit leg is NOT evaluated. "
                           "S11 is the equity leg alone.",
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
