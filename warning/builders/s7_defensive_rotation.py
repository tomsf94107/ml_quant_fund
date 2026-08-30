"""
s7_defensive_rotation.py — builder for S7, Defensive rotation.

REGISTRY ROW:
    id S7 | layer L2 | tier shortlist
    formula  mean RS(XLP; XLU; healthcare) vs SPY 63td > +5%
             WHILE SPX within 3% of high
    source   sector ETFs 1998-12+; French industries pre
    series   XLP; XLU; XLV; SPY        history_start 1998-12-16

WHAT IT MEASURES, AND WHY THE CONJUNCTION MATTERS
    Defensives outperforming is ordinary in a decline -- staples and utilities
    fall less than the market, so their relative strength rises mechanically.
    That carries no information: it is arithmetic, not positioning.

    The signal is defensives leading WHILE THE INDEX IS STILL AT ITS HIGH. That
    is money rotating to safety before the tape confirms anything, which is the
    only configuration in which the observation is a warning rather than a
    description. The near-high gate is the whole signal; the RS leg alone would
    fire in every correction and predict nothing.

    Same construction as S2's credit-equity divergence, and the same reason: the
    conjunction is the signal, not either half.

COVERAGE
    prices.db carries XLP, XLU, XLV and SPY from 2016-07-18 (verified
    2026-08-30), so S7 runs from roughly 2016-10 once the 63-day window fills.
    The registry's 1998-12 start assumes a longer ETF history than the stack
    holds; before 2016 the signal reports NA rather than pretending.

    SPY stands in for SPX in the near-high gate, exactly as in S2. It is a proxy
    and is labelled as one in every reading.

DECISIONS
    The registry says "mean RS(XLP; XLU; healthcare)". Healthcare is read as
    XLV. The mean is taken across whichever of the three have data, and the
    count is reported -- a two-ETF mean is declared, never silently substituted
    for a three-ETF one. Below MIN_DEFENSIVE_ETFS the signal is NA: one sector is
    a sector, not a rotation.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "S7"
LAYER = "L2"
MAX_STALENESS_DAYS = 7
PERSISTENCE_DAYS = 10
AMBER_STATE = "Y"                    # DECISIONS.md D1

DEFENSIVE = ["XLP_CLOSE", "XLU_CLOSE", "XLV_CLOSE"]
BENCH = "SPY_CLOSE"                  # PROXY for SPX
RS_WINDOW = 63                       # registry: "63td"
RS_THRESHOLD = 0.05                  # registry: "> +5%"
NEAR_HIGH_PCT = 0.03                 # registry: "within 3% of high"
HIGH_WINDOW = 252                    # 52 weeks
MIN_DEFENSIVE_ETFS = 2               # one sector is not a rotation


def _rel_strength(sect, bench, window=RS_WINDOW):
    """Relative return of `sect` over `bench` across the trailing window."""
    joined = align(sect, bench)
    if len(joined) < window + 1:
        return None
    d0, s0, b0 = joined[-1 - window]
    _d1, s1, b1 = joined[-1]
    if not s0 or not b0:
        return None
    return ((s1 - s0) / s0) - ((b1 - b0) / b0)


def compute(con, asof):
    bench = series_asof(con, BENCH, asof)
    if len(bench) < HIGH_WINDOW:
        return _na(asof, f"need {HIGH_WINDOW} obs of {BENCH} for the near-high "
                         f"gate, have {len(bench)}")

    rs, missing = {}, []
    for name in DEFENSIVE:
        s = series_asof(con, name, asof)
        if not s:
            missing.append(f"{name}: no data")
            continue
        st = staleness_bdays(con, name, asof)
        if st is None or st > MAX_STALENESS_DAYS:
            missing.append(f"{name}: stale {st}d")
            continue
        v = _rel_strength(s, bench)
        if v is None:
            missing.append(f"{name}: <{RS_WINDOW + 1} overlapping obs")
            continue
        rs[name] = v

    if len(rs) < MIN_DEFENSIVE_ETFS:
        return _na(asof, f"need >={MIN_DEFENSIVE_ETFS} defensive ETFs, have "
                         f"{len(rs)} ({'; '.join(missing) or 'none'})")

    mean_rs = sum(rs.values()) / len(rs)

    window = [v for _, v in bench[-HIGH_WINDOW:]]
    high = max(window)
    last = bench[-1][1]
    near_high = last >= high * (1.0 - NEAR_HIGH_PCT)

    rs_leg = mean_rs > RS_THRESHOLD
    if rs_leg and near_high:
        state = "R"                  # the divergence: defensives lead AT the high
    elif rs_leg or near_high:
        state = AMBER_STATE
    else:
        state = "G"

    stale = max(x for x in (staleness_bdays(con, BENCH, asof),
                            *(staleness_bdays(con, n, asof) for n in rs))
                if x is not None)

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": mean_rs, "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": bench[-1][0],
        "detail": {
            "mean_rs_63d_pct": round(mean_rs * 100, 2),
            "threshold_pct": RS_THRESHOLD * 100,
            "per_etf_rs_pct": {k.replace("_CLOSE", ""): round(v * 100, 2)
                               for k, v in rs.items()},
            "n_defensive": len(rs), "omitted": missing or None,
            "rs_leg": rs_leg,
            "bench": BENCH + " (PROXY for SPX)",
            "bench_last": round(last, 2),
            "bench_52w_high": round(high, 2),
            "pct_below_high": round(100 * (high - last) / high, 2),
            "near_high_leg": near_high,
            "note": "defensives outperforming in a DECLINE is arithmetic; the "
                    "signal is defensives leading while the index is still at "
                    "its high",
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
