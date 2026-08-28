"""
s1_term_spread.py — builder for S1, Term-spread regime.

REGISTRY ROW (signal_registry.csv, implemented verbatim):
    id              S1
    layer           L2          role: predictor
    formula         m_avg(DGS10-DTB3); ARM if <0 for >=2 of 3 consecutive months;
                    ESCALATE if re-steepens +50bp from trough after >=6m inversion
    data_source     FRED/ALFRED     series: DGS10;DTB3
    history_start   1962-01-02      frequency: daily->monthly
    publication_lag 1 day
    threshold_arm   inversion 2/3mo
    threshold_red   re-steepening escalation
    persistence     21 days         direction: inversion_bearish
    max_staleness   7 days
    verdicts        2000: fired Feb+Jul 2000
                    2008: fired Aug06-May07 + Jun07 re-steepen
                    2022: post-peak only

WHY RE-STEEPENING IS THE RED, NOT THE INVERSION
    Inversion alone arms; it does not fire. Report line 469: "Inversion alone =
    arm, don't act. Layer-1 amber can persist for years." The Aug-2006 inversion
    led the bear by 14-16 months. The re-steepening off the trough after a long
    inversion is the late-cycle marker.

AMBER -> 'Y' (0.33), NOT 'O' (0.66)               <-- RATIFICATION NEEDED
    The registry/report use three states (green/amber/red); the engine has five
    (G/Y/O/R/B). The amber mapping is UNDERSPECIFIED by both documents. 'Y' is
    chosen because report line 469 says L1/L2 amber "can persist for years --
    exposure gates only"; at 0.66 a multi-year amber would hold the composite
    elevated indefinitely. This is a reasoned default, NOT a sourced spec value.
    Change AMBER_STATE if ratified otherwise; it is deliberately one constant.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, monthly_mean, align, staleness_days  # noqa: E402

SIGNAL_ID = "S1"
LAYER = "L2"
SERIES = ("DGS10", "DTB3")
MAX_STALENESS_DAYS = 7
PERSISTENCE_DAYS = 21

AMBER_STATE = "Y"          # see docstring -- awaiting ratification
INVERSION_MONTHS_OF_3 = 2  # registry: "<0 for >=2 of 3 consecutive months"
MIN_INVERSION_MONTHS = 6   # registry: "after >=6m inversion"
RESTEEPEN_BP = 0.50        # registry: "+50bp from trough" (series are in percent)

# ESCALATE RECENCY WINDOW                          <-- RATIFICATION NEEDED
# The registry formula says "re-steepens +50bp from trough after >=6m inversion"
# but sets NO time limit on how old that inversion may be. Without one, the
# 1980-11..1981-08 inversion (trough -2.65) made every later positive spread look
# like a +300bp "re-steepening": S1 fired R in Feb-2000 off a 19-year-old trough.
#
# NOT A FITTED PARAMETER: the historical anchors are invariant across a very wide
# range. 2007-10 needs >= 6 months (inversion ended 2007-04); excluding the 1981
# artifact at 2000-02 needs < 222 months. Any value in [6, 222) gives identical
# verdicts on every anchor, so 24 is a round choice inside a flat region, not a
# tuned one. Escalation also clears whenever a NEW inversion run begins.
ESCALATE_WINDOW_MONTHS = 24


def compute(con, asof, full_history: bool = False):
    """Compute S1 as of `asof`. Returns a dict; caller wraps it in SignalReading.

    Missing data returns state 'NA' with a reason -- never a fabricated 0.
    """
    d10 = series_asof(con, "DGS10", asof)
    d3m = series_asof(con, "DTB3", asof)
    if not d10 or not d3m:
        missing = [s for s, r in zip(SERIES, (d10, d3m)) if not r]
        return _na(asof, f"no visible observations for {','.join(missing)}")

    joined = align(d10, d3m)
    if not joined:
        return _na(asof, "DGS10 and DTB3 have no overlapping observation dates")

    spread_daily = [(d, a - b) for d, a, b in joined]
    months = monthly_mean(spread_daily)
    if len(months) < 3:
        return _na(asof, f"only {len(months)} monthly obs; need >=3 for the 2-of-3 rule")

    # staleness: worst of the two inputs
    st = [staleness_days(con, s, asof) for s in SERIES]
    stale_days = max(x for x in st if x is not None)
    stale = stale_days > MAX_STALENESS_DAYS

    last3 = months[-3:]
    n_inverted = sum(1 for _, v in last3 if v < 0)
    armed = n_inverted >= INVERSION_MONTHS_OF_3

    # ESCALATE: a run of >=6 inverted months, then a >=50bp re-steepen off its trough.
    escalated, detail = _resteepen(months)

    if escalated:
        state = "R"
    elif armed:
        state = AMBER_STATE
    else:
        state = "G"

    current = months[-1]
    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state,
        "raw_value": current[1],
        "zscore": None,                       # M1 standardization happens at fit time
        "stale": stale, "stale_days": stale_days,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": max(d10[-1][0], d3m[-1][0]),
        "detail": {
            "current_month": current[0],
            "current_spread": round(current[1], 4),
            "last3": [(m, round(v, 4)) for m, v in last3],
            "inverted_of_last3": n_inverted,
            "armed": armed,
            "escalated": escalated,
            "resteepen": detail,
        },
    }


def _resteepen(months):
    """True if the LAST completed inversion run was >=6 months and the spread has
    since risen >=50bp off that run's trough. Returns (bool, detail)."""
    runs, cur = [], []
    for m, v in months:
        if v < 0:
            cur.append((m, v))
        else:
            if cur:
                runs.append(cur); cur = []
    ongoing = bool(cur)
    if cur:
        runs.append(cur)
    if not runs:
        return False, {"reason": "no inversion run on record"}

    run = runs[-1]
    trough_m, trough_v = min(run, key=lambda x: x[1])
    latest_m, latest_v = months[-1]
    rise = latest_v - trough_v
    age = _months_between(run[-1][0], latest_m)
    detail = {"run_len": len(run), "run_start": run[0][0], "run_end": run[-1][0],
              "ongoing": ongoing, "trough_month": trough_m,
              "trough": round(trough_v, 4), "latest": round(latest_v, 4),
              "rise_bp": round(rise * 100, 1), "months_since_run_end": age}

    if len(run) < MIN_INVERSION_MONTHS:
        detail["reason"] = (f"last inversion run only {len(run)}m "
                            f"(<{MIN_INVERSION_MONTHS}m)")
        return False, detail
    if not ongoing and age > ESCALATE_WINDOW_MONTHS:
        detail["reason"] = (f"inversion ended {age}m ago "
                            f"(>{ESCALATE_WINDOW_MONTHS}m window) -- stale, cannot escalate")
        return False, detail
    return (rise >= RESTEEPEN_BP), detail


def _months_between(m_a: str, m_b: str) -> int:
    """Whole months from 'YYYY-MM' m_a to m_b."""
    ya, ma = int(m_a[:4]), int(m_a[5:7])
    yb, mb = int(m_b[:4]), int(m_b[5:7])
    return (yb - ya) * 12 + (mb - ma)


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason}}


def to_reading(result):
    """dict -> warning_engine.SignalReading."""
    from warning_engine import SignalReading
    return SignalReading(
        signal_id=result["signal_id"], layer=result["layer"],
        state=result["state"], stale=bool(result["stale"]),
        min_persistence=result["persistence_days"])
