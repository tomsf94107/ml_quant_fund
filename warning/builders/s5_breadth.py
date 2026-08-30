"""
s5_breadth.py — builder for S5, Breadth divergence composite.

REGISTRY ROW:
    id S5 | layer L2 | role predictor | tier shortlist
    formula  2 of 3:
             (a) cumulative A/D makes a lower high vs the index's 52w high,
                 over >= 126 trading days
             (b) % of names above their 200DMA < 60% while the index is at a
                 52-week high
             (c) new 52w lows > 2.5% of issues on >= 5 of the last 21 days,
                 with the index within 5% of its high
    arm      1 of 3          red: 2 of 3
    persistence 21 days      max_staleness: 3 days
    history  forward-only (historical cells literature-sourced)
    verdicts 2000: fired Apr-98 (23mo lead); 2008: fired ~Jun-07 (4mo);
             2022: fired late-2021
    notes    survivorship-safe forward only

ALL THREE CONDITIONS ARE DIVERGENCES, NOT LEVELS
    Each requires the index to be at or near its high. Weak breadth in a falling
    market is not a divergence -- it is the market falling. The signal is the
    index making highs on deteriorating internals, which is why every leg is
    gated on the index rather than measured alone.

SURVIVORSHIP BIAS MAKES HISTORICAL READINGS UNUSABLE AS EVIDENCE
    The breadth series come from prices.db, which holds 443 tickers that exist
    TODAY. Names that delisted between 2016 and 2026 are gone. Breadth counts
    how many constituents are failing, and the failures have been removed, so
    historical readings are systematically healthier than reality and S5 will
    UNDER-FIRE on history.

    The registry says exactly this -- `history_start: forward-only`,
    `survivorship-safe forward only`, and its 2000/2008/2022 verdicts are marked
    literature-sourced rather than computed. A quiet S5 in 2020 or 2022 is
    therefore NOT evidence the signal was quiet then, and must not be counted as
    a miss. Every reading carries the caveat.

REGISTRY VERDICTS CANNOT BE REPRODUCED HERE AT ALL
    Apr-1998, Jun-2007 and late-2021 all predate or straddle the 2016 start of
    the universe, and two of the three predate it entirely. S5 has no
    reproducible historical anchor. Its record starts now.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_bdays  # noqa: E402

SIGNAL_ID = "S5"
LAYER = "L2"
MAX_STALENESS_DAYS = 3
PERSISTENCE_DAYS = 21
AMBER_STATE = "Y"                     # DECISIONS.md D1

AD = "BREADTH_AD_CUM"
PCT200 = "BREADTH_PCT_200DMA"
NEWLOWS = "BREADTH_NEW_LOWS_PCT"
BENCH = "SPY_CLOSE"                   # PROXY for SPX

HIGH_WINDOW = 252
AD_LOOKBACK = 126                     # registry: ">= 126td"
PCT200_THRESHOLD = 60.0               # registry: "< 60%"
NEWLOW_PCT = 2.5                      # registry: "> 2.5% of issues"
NEWLOW_DAYS = 5                       # registry: "on >= 5 of 21td"
NEWLOW_WINDOW = 21
NEAR_HIGH_PCT = 0.05                  # registry: "within 5% of high" (leg c)


def _at_52w_high(bench):
    return bench[-1][1] >= max(v for _, v in bench[-HIGH_WINDOW:])


def _near_high(bench, pct=NEAR_HIGH_PCT):
    hi = max(v for _, v in bench[-HIGH_WINDOW:])
    return bench[-1][1] >= hi * (1.0 - pct)


def compute(con, asof):
    bench = series_asof(con, BENCH, asof)
    if len(bench) < HIGH_WINDOW:
        return _na(asof, f"need {HIGH_WINDOW} obs of {BENCH}, have {len(bench)}")

    ad = series_asof(con, AD, asof)
    p200 = series_asof(con, PCT200, asof)
    nl = series_asof(con, NEWLOWS, asof)

    at_high = _at_52w_high(bench)
    near_high = _near_high(bench)
    hi = max(v for _, v in bench[-HIGH_WINDOW:])
    pct_below = 100 * (hi - bench[-1][1]) / hi

    legs, detail = {}, {}

    # (a) cumulative A/D makes a LOWER HIGH while the index makes a new one
    if len(ad) < AD_LOOKBACK:
        detail["a"] = {"reason": f"need {AD_LOOKBACK} obs of {AD}, have {len(ad)}"}
    else:
        window = [v for _, v in ad[-AD_LOOKBACK:]]
        prior_max = max(window[:-1])
        lower_high = window[-1] < prior_max
        legs["a"] = bool(at_high and lower_high)
        detail["a"] = {"index_at_52w_high": at_high,
                       "ad_cum": round(window[-1], 1),
                       "ad_prior_max_126d": round(prior_max, 1),
                       "ad_lower_high": lower_high, "fired": legs["a"]}

    # (b) breadth thin at a 52-week high
    if not p200:
        detail["b"] = {"reason": f"no {PCT200} data"}
    else:
        pct = p200[-1][1]
        legs["b"] = bool(at_high and pct < PCT200_THRESHOLD)
        detail["b"] = {"index_at_52w_high": at_high,
                       "pct_above_200dma": round(pct, 1),
                       "threshold": PCT200_THRESHOLD, "fired": legs["b"]}

    # (c) new lows expanding while the index holds near its high
    if len(nl) < NEWLOW_WINDOW:
        detail["c"] = {"reason": f"need {NEWLOW_WINDOW} obs of {NEWLOWS}, "
                                 f"have {len(nl)}"}
    else:
        recent = [v for _, v in nl[-NEWLOW_WINDOW:]]
        days = sum(1 for v in recent if v > NEWLOW_PCT)
        legs["c"] = bool(near_high and days >= NEWLOW_DAYS)
        detail["c"] = {"index_near_high_5pct": near_high,
                       "days_above_2p5pct_of_21": days,
                       "needs_days": NEWLOW_DAYS,
                       "latest_new_lows_pct": round(recent[-1], 2),
                       "fired": legs["c"]}

    if not legs:
        return _na(asof, "no breadth legs computable: "
                         + "; ".join(f"{k}: {v.get('reason')}"
                                     for k, v in detail.items()))

    n_fired = sum(1 for v in legs.values() if v)
    if n_fired >= 2:
        state = "R"
    elif n_fired >= 1:
        state = AMBER_STATE
    else:
        state = "G"

    stale = max(x for x in (staleness_bdays(con, BENCH, asof),
                            staleness_bdays(con, AD, asof))
                if x is not None)

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": float(n_fired), "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": bench[-1][0],
        "detail": {
            "legs_fired": n_fired, "legs_computable": sorted(legs),
            "leg_a_ad_divergence": detail.get("a"),
            "leg_b_pct_above_200dma": detail.get("b"),
            "leg_c_new_lows": detail.get("c"),
            "index_pct_below_52w_high": round(pct_below, 2),
            "SURVIVORSHIP": "breadth computed from surviving tickers only; "
                            "delisted names absent, so historical readings are "
                            "healthier than reality and S5 under-fires on "
                            "history. Forward-only per the registry.",
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
