"""
s9_short_interest.py — builder for S9, Aggregate short interest.

REGISTRY ROW:
    id S9 | layer L2 | role predictor | tier shortlist
    formula   detrended log aggregate SI (linear trend to date) z > +1.5 (12m)
    arm       z>1            red: z>1.5
    source    FINRA files (2014+)      frequency: semi-monthly
    publication_lag ~8 bus days        max_staleness: 20 days
    direction rising_SI_bearish        persistence: 1 obs
    verdicts  2000 UNDECIDABLE (institutional); 2008 UNDECIDABLE;
              2022 REPRODUCIBLE
    notes     Rapach et al 2016 evidence class

THE PANEL IS CHOSEN POINT-IN-TIME, AND THAT IS THE WHOLE DIFFICULTY
    An aggregate over a drifting universe measures the ingest history, not
    positioning. Measured on the real data 2026-08-30: the naive sum over all
    available names rose 4.42bn -> 9.88bn (+124%) while a fixed 362-name panel
    rose 4.40bn -> 7.88bn (+79%). About 45 percentage points of the apparent
    surge was coverage expansion. One step, 2026-05-29 to 2026-06-15, added 25
    tickers and 0.66bn -- a naive reading calls that a 6% short-interest surge.

    But selecting "names present on every date" using today's data is
    look-ahead: it presumes knowledge of which tickers would still be covered.
    So the panel is rebuilt at EVERY evaluation date from names present on all
    settlement dates inside the trailing z-window, using only rows already
    published as of that date. The panel size is reported in every reading, so a
    shrinking panel is visible rather than silently changing what is measured.

HISTORY IS SHORTER THAN THE REGISTRY CLAIMS
    The registry says 2014+. short_interest.db starts 2021-04-15 -- about 113
    semi-monthly observations. A 12-month z-window is therefore n=24, and the
    expanding linear detrend consumes further degrees of freedom. S9 is thin by
    construction here, and only the 2022 verdict is checkable at all (the
    registry already marks 2000 and 2008 UNDECIDABLE).

DETREND
    "linear trend to date" is read as an EXPANDING OLS fit of log(aggregate) on
    a time index, using every observation visible at the evaluation date and no
    more. The residual is then z-scored over the trailing 12 months. Fitting the
    trend on the full sample would leak the future into every historical
    reading -- the same defect class as the pub_date bug fixed on 2026-08-28.
"""

from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import _d  # noqa: E402

SIGNAL_ID = "S9"
LAYER = "L2"
MAX_STALENESS_DAYS = 20
PERSISTENCE_DAYS = 1               # registry: "1 obs"
AMBER_STATE = "Y"                  # DECISIONS.md D1

Z_WINDOW_OBS = 24                  # 12 months of semi-monthly observations
MIN_TREND_OBS = 36                 # expanding OLS needs materially more than the z-window
ARM_Z = 1.0                        # registry threshold_arm: "z>1"
RED_Z = 1.5                        # registry threshold_red: "z>1.5"
MIN_PANEL = 100                    # an "aggregate" over a handful of names is not one


def _visible_rows(con, asof):
    """All published SI:* rows as of `asof`. Same rule as pit.series_asof, applied
    across many series in one pass -- reading them one at a time would be ~440
    queries per evaluation date."""
    return con.execute(
        "SELECT series_id, obs_date, value FROM data_vintages "
        "WHERE series_id LIKE 'SI:%' AND pub_date <= ? ORDER BY obs_date",
        (_d(asof),)).fetchall()


def _ols_residual_last(ys):
    """Residual of the final point from an OLS line fitted to all of ys."""
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    resid = [ys[i] - (a + b * xs[i]) for i in range(n)]
    return resid, a, b


def compute(con, asof):
    rows = _visible_rows(con, asof)
    if not rows:
        return _na(asof, "no published SI:* rows visible")

    by_date = {}
    for sid, obs, val in rows:
        by_date.setdefault(obs, {})[sid] = val
    dates = sorted(by_date)
    if len(dates) < MIN_TREND_OBS:
        return _na(asof, f"need {MIN_TREND_OBS} settlement dates for the "
                         f"expanding trend, have {len(dates)}")

    # POINT-IN-TIME panel: names present on every date inside the trailing
    # z-window. Never the full-sample intersection, which would be look-ahead.
    window_dates = dates[-Z_WINDOW_OBS:]
    panel = set(by_date[window_dates[0]])
    for d in window_dates[1:]:
        panel &= set(by_date[d])
    if len(panel) < MIN_PANEL:
        return _na(asof, f"point-in-time panel is {len(panel)} names "
                         f"(<{MIN_PANEL}) over the trailing {Z_WINDOW_OBS} dates")

    # The trend is fitted on the same panel across every visible date, so a
    # changing universe cannot tilt it. Dates missing a panel member are dropped.
    agg = []
    for d in dates:
        row = by_date[d]
        if not panel <= set(row):
            continue
        total = sum(row[s] for s in panel)
        if total > 0:
            agg.append((d, math.log(total)))
    if len(agg) < MIN_TREND_OBS:
        return _na(asof, f"only {len(agg)} dates carry the full panel; "
                         f"need {MIN_TREND_OBS}")

    resid, _a, slope = _ols_residual_last([v for _, v in agg])
    if resid is None:
        return _na(asof, "degenerate time index")

    tail = resid[-Z_WINDOW_OBS:]
    if len(tail) < Z_WINDOW_OBS:
        return _na(asof, f"z-window needs {Z_WINDOW_OBS} residuals, "
                         f"have {len(tail)}")
    m = sum(tail) / len(tail)
    sd = math.sqrt(sum((x - m) ** 2 for x in tail) / (len(tail) - 1))
    if sd == 0:
        return _na(asof, "zero residual variance over the z-window")
    z = (resid[-1] - m) / sd

    if z > RED_Z:
        state = "R"
    elif z > ARM_Z:
        state = AMBER_STATE
    else:
        state = "G"

    last_obs = agg[-1][0]
    stale = (__import__("datetime").date.fromisoformat(_d(asof))
             - __import__("datetime").date.fromisoformat(last_obs)).days

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": math.exp(agg[-1][1]), "zscore": z,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": last_obs,
        "detail": {
            "z": round(z, 2), "arm_z": ARM_Z, "red_z": RED_Z,
            "aggregate_short_bn": round(math.exp(agg[-1][1]) / 1e9, 2),
            "panel_names": len(panel),
            "panel_note": "chosen point-in-time from names present on every "
                          "date in the trailing window; a full-sample "
                          "intersection would be look-ahead",
            "obs_used": len(agg), "z_window_obs": Z_WINDOW_OBS,
            "trend_slope_per_obs": round(slope, 5),
            "detrended_residual": round(resid[-1], 4),
            "settlement_date": last_obs,
            "days_since_settlement": stale,
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
