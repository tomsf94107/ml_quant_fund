"""
s2_credit.py — builder for S2, Credit trend & credit-equity divergence.

REGISTRY ROW (implemented verbatim):
    id              S2                      layer: L2      role: predictor
    formula         spread>200d MA AND spread>=126d_low+75bp
                    WHILE SPX within 3% of 52w high (21d)
    data_source     FRED. BAA10YM primary history;
                    BAMLH0A0HYM2 rolling 3y window (archive it)
    history_start   1953-04-01              frequency: daily/monthly
    publication_lag 1 day                   persistence: 10 days
    threshold_arm   half-condition          threshold_red: full condition
    direction       widening_bearish        max_staleness: 7 days
    verdicts        2000: weak fire (+44bp Jan-Mar00)
                    2008: strong fire (Jun-Oct07)
                    2022: correctly silent
    notes           PAIR WITH S3 or S15 to kill 1998/2011/2015/2018 false fires;
                    FRED ICE truncation flagged

TWO MODES (DECISIONS.md D7)
    daily   BAMLH0A0HYM2, 200d MA / 126d low. Only from ~2024-05: FRED serves a
            rolling 3-year window, so 200 daily observations do not exist before
            then. The archive is the only long history you will ever have.
    monthly BAA10YM, 10-month MA / 6-month low, 1953+. 200 and 126 trading days
            at ~21/month. The +75bp threshold is unchanged.
    Every reading declares its mode. The two are never silently mixed.

HALF vs FULL (DECISIONS.md D6)
    credit leg alone      -> amber (arm)
    credit AND equity leg -> red   (the divergence)

THE EQUITY LEG IS NOT YET COMPUTABLE
    "SPX within 3% of its 52w high for 21 days" needs an SPX price series.
    warning.db has no price data: fetch_free_history pulls macro, Cboe vol and
    factor files, none of which is the S&P 500 index level. Rather than
    substitute a proxy (SPY starts 1993; a Ken French cumulative market index is
    not SPX), the equity leg returns None and the reading says so. S2 can
    therefore ARM but never fire RED until an SPX source is wired.

    That is a finding, not a gap to paper over: an S2 that silently dropped the
    equity leg would fire red on every credit widening -- exactly the 1998/2011/
    2015/2018 false fires the registry's notes column warns about.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, monthly_mean_complete, staleness_bdays  # noqa: E402

SIGNAL_ID = "S2"
LAYER = "L2"
MAX_STALENESS_DAYS = 7
PERSISTENCE_DAYS = 10
PUB_LAG_DAYS = 1

AMBER_STATE = "Y"            # DECISIONS.md D1 (ratified)
WIDEN_BP = 0.75              # registry: "+75bp"; series are in percentage points
EQUITY_NEAR_HIGH_PCT = 0.03  # registry: "within 3% of 52w high"
EQUITY_PERSIST_DAYS = 21     # registry: "(21d)"

DAILY_SERIES = "BAMLH0A0HYM2"
MONTHLY_SERIES = "BAA10YM"
DAILY_MA = 200               # trading days
DAILY_LOW = 126              # trading days
MONTHLY_MA = 10              # months ~= 200 trading days   (D7)
MONTHLY_LOW = 6              # months ~= 126 trading days   (D7)


EQUITY_SERIES = "SPY_CLOSE"      # PROXY for SPX; see ingest_spx.py


def compute(con, asof, mode: str = "auto", spx_near_high=None):
    """Compute S2 as of `asof`.

    mode: 'auto' | 'daily' | 'monthly'. 'auto' prefers daily when enough daily
    observations are visible, else falls back to monthly.

    spx_near_high: None (unknown -> cannot evaluate the full condition), or a
    bool supplied by the caller once an SPX series is wired.
    """
    daily = series_asof(con, DAILY_SERIES, asof)
    use_daily = (mode == "daily") or (mode == "auto" and len(daily) >= DAILY_MA)

    if use_daily:
        if len(daily) < DAILY_MA:
            return _na(asof, f"daily mode needs {DAILY_MA} obs of {DAILY_SERIES}, "
                             f"have {len(daily)}", "daily")
        values = [v for _, v in daily]
        series_used, ma_n, low_n = DAILY_SERIES, DAILY_MA, DAILY_LOW
        last_label = daily[-1][0]
        stale_days = staleness_bdays(con, DAILY_SERIES, asof)
    else:
        rows = series_asof(con, MONTHLY_SERIES, asof)
        if not rows:
            return _na(asof, f"no visible observations for {MONTHLY_SERIES}", "monthly")
        months = monthly_mean_complete(rows, asof, PUB_LAG_DAYS)
        if len(months) < MONTHLY_MA:
            return _na(asof, f"monthly mode needs {MONTHLY_MA} complete months of "
                             f"{MONTHLY_SERIES}, have {len(months)}", "monthly")
        values = [v for _, v in months]
        series_used, ma_n, low_n = MONTHLY_SERIES, MONTHLY_MA, MONTHLY_LOW
        last_label = months[-1][0]
        stale_days = staleness_bdays(con, MONTHLY_SERIES, asof)

    spread = values[-1]
    ma = sum(values[-ma_n:]) / ma_n
    low = min(values[-low_n:])

    above_ma = spread > ma
    off_low = spread >= low + WIDEN_BP
    credit_leg = above_ma and off_low

    # Equity leg: computed from the ingested proxy unless the caller overrides.
    equity_source = None
    if spx_near_high is None:
        closes = series_asof(con, EQUITY_SERIES, asof)
        if closes:
            spx_near_high = equity_leg_from_prices(closes, asof)
            equity_source = EQUITY_SERIES

    if not credit_leg:
        state = "G"
    elif spx_near_high is True:
        state = "R"
    else:
        state = AMBER_STATE          # half-condition, or equity leg unknown

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state,
        "raw_value": spread,
        "zscore": None,
        "stale": (stale_days is None or stale_days > MAX_STALENESS_DAYS),
        "stale_days": stale_days,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": last_label,
        "detail": {
            "mode": "daily" if use_daily else "monthly",
            "series": series_used,
            "last_period": last_label,
            "spread": round(spread, 4),
            f"ma{ma_n}": round(ma, 4),
            f"low{low_n}": round(low, 4),
            "above_ma": above_ma,
            "off_low_by_bp": round((spread - low) * 100, 1),
            "off_low": off_low,
            "credit_leg": credit_leg,
            "equity_leg": spx_near_high,
            "equity_source": equity_source,
            "equity_note": (
                None if spx_near_high is not None else
                f"{EQUITY_SERIES} absent or too short at this date (needs 273 "
                f"sessions; coverage starts 2016-07-18) -- the full condition "
                f"cannot be evaluated, so S2 can arm but not fire red"),
        },
    }


def equity_leg_from_prices(closes, asof) -> bool | None:
    """SPX within 3% of its 52w high on every one of the last 21 sessions.

    closes: [(date, close)] ascending, point-in-time. Returns None if there is
    not enough history to judge -- never False by default, because "not enough
    data" and "not near the high" are different findings.
    """
    rows = [(d, c) for d, c in closes if d <= str(asof)]
    if len(rows) < 252 + EQUITY_PERSIST_DAYS:
        return None
    for i in range(len(rows) - EQUITY_PERSIST_DAYS, len(rows)):
        window = [c for _, c in rows[max(0, i - 252):i + 1]]
        if not window:
            return None
        if rows[i][1] < max(window) * (1.0 - EQUITY_NEAR_HIGH_PCT):
            return False
    return True


def _na(asof, reason, mode):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason, "mode": mode}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(
        signal_id=result["signal_id"], layer=result["layer"],
        state=result["state"], stale=bool(result["stale"]),
        min_persistence=result["persistence_days"])
