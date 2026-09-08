"""
s10_margin_debt.py — builder for S10, Margin-debt blowoff & reversal.

REGISTRY ROW:
    id S10 | layer L1+trigger | tier shortlist | history_start 1997-01
    formula  fragility: YoY>+40%.
             trigger:   first 3m decline from peak (as of publication date)
    source   FINRA margin statistics   series: margin stats xlsx
    arm      YoY>+30%                  red: YoY>+40% then 3m reversal
    frequency monthly                  publication_lag ~3 weeks
    persistence_days 1                 max_staleness_days 45
    direction reversal_bearish         role: confirmer
    verdicts  peak Mar-00 coincident, trigger ~Jun-Jul00 with 96% of the
              decline remaining | peak Oct-07, trigger ~Jan-08 with 49%
              remaining | peak Oct-21 led index peaks by 1-2 months
    notes    coincident confirmer with small lag - valuable because slow bears
             are long

THIS SIGNAL IS STATEFUL, WHICH MAKES IT DIFFERENT FROM S11 AND S13
    Both of those rank one number against its own history. S10 tracks a running
    PEAK and counts months since it. Two consequences.

    The peak must be the peak AS KNOWN THEN, not the eventual peak. Computed
    from the visible series at each asof, so a peak set in month t is only the
    peak until something exceeds it. A full-sample peak would let the signal
    know in 1999 that March 2000 was the top.

    The count is of PUBLISHED months, not elapsed ones. If FINRA skips a
    release the count does not advance on a month nobody could see.

WHY THE LAG IS THE POINT, NOT A DEFECT
    FINRA publishes in the third week of the month FOLLOWING the reference
    month. So the March 2000 peak became visible in late April, and a
    three-month decline from it could not be confirmed until roughly June or
    July. The registry records exactly that: trigger ~Jun-Jul00, with 96% of
    the decline still ahead.

    That is the whole claim of the signal. It does not call tops. It confirms
    that leverage has turned, late, while most of the drawdown is still to
    come -- which is useful precisely because, as the registry note says, slow
    bears are long. A version tuned to fire at the peak would be fitted.

TWO CONDITIONS, AND THEY ARE SEQUENTIAL
    Fragility comes first: YoY above +30% is amber, above +40% is the red
    precondition. The reversal is second: the first three consecutive declines
    from a peak that was itself set while fragile. Red requires BOTH -- a
    reversal without a preceding blowoff is an ordinary decline in leverage,
    and a blowoff without a reversal is a bull market.

    Read literally from "red: YoY>+40% then 3m reversal". The fragility test is
    applied AT THE PEAK, not at the current month, because by the time three
    months of decline have accumulated the YoY reading has usually rolled over
    on its own -- testing it late would make the red condition nearly
    unreachable. This is the one place where the registry's compressed phrasing
    admits more than one reading; the peak-anchored version is what reproduces
    the recorded verdicts, and the alternative is noted in the detail block.

L1 AND L4D
    The registry layer is "L1+trigger": S10 counts toward L1 coverage and also
    feeds L4D, forced deleveraging. l4_propagation.py currently hard-codes an
    NA reason for L4D naming this signal as the missing input. That wiring is
    a separate change; this builder only produces the reading.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_days  # noqa: E402

SIGNAL_ID = "S10"
LAYER = "L1"
MAX_STALENESS_DAYS = 45
PERSISTENCE_DAYS = 1
AMBER_STATE = "Y"                  # DECISIONS.md D1

DEBIT_SERIES = "FINRA_MARGIN_DEBIT"
ARM_YOY = 30.0                     # registry threshold_arm
RED_YOY = 40.0                     # registry threshold_red
REVERSAL_MONTHS = 3                # "first 3m decline from peak"
MIN_MONTHS = 24                    # need a year of history plus a year to compare


def _yoy(rows, i):
    """YoY % at index i, or None if the -12 month is absent."""
    if i < 12:
        return None
    return 100.0 * (rows[i][1] / rows[i - 12][1] - 1.0)


def compute(con, asof):
    rows = series_asof(con, DEBIT_SERIES, asof)
    if not rows:
        return _na(asof, f"no visible observations for {DEBIT_SERIES}; run "
                         f"warning/ingest_finra_margin.py")
    if len(rows) < MIN_MONTHS:
        return _na(asof, f"need {MIN_MONTHS} published months, have {len(rows)}")

    vals = [v for _, v in rows]
    cur_yoy = _yoy(rows, len(rows) - 1)
    if cur_yoy is None:
        return _na(asof, "fewer than 12 published months; YoY undefined")

    # Running peak AS KNOWN AT THIS ASOF. Ties resolve to the earlier month, so
    # a flat top does not keep resetting the decline count.
    peak_i = max(range(len(vals)), key=lambda i: (vals[i], -i))
    months_since_peak = len(vals) - 1 - peak_i
    peak_yoy = _yoy(rows, peak_i)

    # Consecutive down-months, reported only. NOT the trigger -- see below.
    declines = 0
    for i in range(peak_i + 1, len(vals)):
        if vals[i] < vals[i - 1]:
            declines += 1
        else:
            break

    # "3m decline from peak" is CUMULATIVE: three published months have elapsed
    # since the peak and the level is still below it. NOT three consecutive
    # down months. Measured -- 2000 went down, down, UP (Apr 268,716 / May
    # 256,862 / Jun 264,471) and 2007 the same (Aug 362,475 / Sep 359,104 /
    # Oct 376,979). A consecutive reading never reaches three in either
    # episode and reproduces neither recorded verdict.
    fragile_at_peak = peak_yoy is not None and peak_yoy > RED_YOY
    below_peak = vals[-1] < vals[peak_i]
    reversal = months_since_peak >= REVERSAL_MONTHS and below_peak

    # "FIRST 3m decline" read as an EVENT: red in the month the condition first
    # confirms, not for as long as it holds. An all-time-peak anchor that keeps
    # firing gives six years of red after both 2000 and 2007 -- the Mar-00 peak
    # was not exceeded until 2006-12 and the Jul-07 peak not until 2013-09.
    # Six years of red through a recovery is not a signal. See DECISIONS.md.
    first_confirm = months_since_peak == REVERSAL_MONTHS and below_peak
    drawdown = 100.0 * (vals[-1] / vals[peak_i] - 1.0)

    if fragile_at_peak and first_confirm:
        state = "R"
    elif cur_yoy > ARM_YOY or (fragile_at_peak and months_since_peak > 0):
        state = AMBER_STATE
    else:
        state = "G"

    stale = staleness_days(con, DEBIT_SERIES, asof)
    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": cur_yoy, "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS),
        "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": rows[-1][0],
        "detail": {
            "reference_month": rows[-1][0][:7],
            "debit_balances_musd": round(vals[-1]),
            "yoy_pct": round(cur_yoy, 1),
            "arm_at_yoy": ARM_YOY, "red_at_yoy": RED_YOY,
            "peak_month": rows[peak_i][0][:7],
            "peak_musd": round(vals[peak_i]),
            "peak_yoy_pct": round(peak_yoy, 1) if peak_yoy is not None else None,
            "months_since_peak": months_since_peak,
            "consecutive_declines": declines,
            "below_peak": below_peak,
            "trigger_armed": reversal,
            "trigger_first_confirm": first_confirm,
            "reversal_needs": REVERSAL_MONTHS,
            "drawdown_from_peak_pct": round(drawdown, 1),
            "fragile_at_peak": fragile_at_peak,
            "reversal_confirmed": reversal,
            "months_in_history": len(vals),
            "pit_note": "peak is the peak AS KNOWN at this asof, from months "
                        "published on or before it -- never the eventual peak",
            "lag_note": "FINRA publishes in the third week of the month "
                        "following the reference month, so a peak is not "
                        "visible for ~3 weeks and a 3-month reversal not for "
                        "~4 months. That lag is the signal: the registry "
                        "records the Mar-00 peak triggering ~Jun-Jul00 with "
                        "96% of the decline still ahead.",
            "registry_2007_peak_note": "the registry records the 2008 peak as "
                        "Oct-07. FINRA's series peaks Jul-07 at 416,403; "
                        "Oct-07 is 376,979, 9.5% lower and not even a local "
                        "max (Nov-07 is higher). Oct-07 is the S&P 500 peak. "
                        "The 2021 row says margin LED the index by 1-2 months, "
                        "which is consistent. Not fitted to the wrong date.",
            "reading_note": "fragility is tested AT THE PEAK, not at the "
                            "current month. By the time three declines have "
                            "accumulated the YoY reading has usually rolled "
                            "over, so testing it late would make red nearly "
                            "unreachable. The registry's 'YoY>+40% then 3m "
                            "reversal' admits both readings; this one "
                            "reproduces the recorded verdicts.",
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
