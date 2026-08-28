"""
series_meta.py — revisability and publication lag, per series.

WHY THIS EXISTS
    A bulk FRED pull stamps every row with pub_date = the day we downloaded it.
    That is when WE got it, not when it became publicly knowable. For a 2000
    observation pulled in 2026 the two differ by 26 years, and a point-in-time
    read for 2000 correctly sees nothing -- which makes historical validation
    of every macro signal impossible.

    The correction is legitimate for exactly one class of series: those that are
    NEVER REVISED. For those, the value published on obs_date + publication_lag
    is the final value, so deriving pub_date reproduces precisely what ALFRED
    would return. It is not an approximation and not a backfill.

    For REVISABLE series it is NOT legitimate. A first-print payrolls number
    differs from today's revised one, and only true ALFRED vintages can tell you
    what was on the screen at the time. Those series stay stamped with the pull
    date until the ALFRED leg runs; a historical read returning NA for them is
    the correct, honest answer.

RULE #5 (backfill guard) IS NOT VIOLATED
    That rule bars treating a series as real-time-observable for dates it did
    not exist (SKEW pre-2011, VIX pre-2003, GZ/EBP entirely). These series did
    exist and were published daily; we are recording when, not inventing it.

EPISTEMIC STATUS
    `revisable` is a per-series property. Market-quoted yields and index levels
    are not restated; statistical estimates from surveys are. The classification
    below is [fact?] -- from domain knowledge, not a vendor-published revision
    policy document. Anything uncertain is marked revisable, which is the safe
    direction: it yields NA rather than a wrong number.
"""

# series_id -> (revisable, publication_lag_days, note)
SERIES_META = {
    # ---- NOT revised: market-quoted rates/prices. pub_date may be derived. ----
    "DGS10":        (False, 1, "Treasury constant maturity, H.15 daily. Not restated."),
    "DTB3":         (False, 1, "3m T-bill secondary market, H.15 daily. Not restated."),
    "BAA":          (False, 1, "Moody's Baa corporate yield, daily. Not restated."),
    "AAA":          (False, 1, "Moody's Aaa corporate yield, daily. Not restated."),
    "BAA10YM":      (False, 1, "Baa minus 10y, monthly, derived from unrevised inputs."),
    "BAMLH0A0HYM2": (False, 1, "ICE BofA HY OAS index level, daily. Not restated. "
                               "FRED serves a rolling 3y window -- archive is the history."),
    "BAMLC0A0CM":   (False, 1, "ICE BofA IG OAS index level, daily. Not restated. "
                               "Rolling 3y window."),
    "SOFR":         (False, 1, "NY Fed published next business day. Revisions are rare "
                               "and pre-announced; treated as final."),
    "VIXCLS":       (False, 1, "Cboe VIX close via FRED. Index level, not restated."),
    # Cboe archive copies, loaded by parse_cboe.py. All are published index levels
    # or exchange-reported volume ratios: not restated. Prefer the FRED copies of
    # VIX/VIX3M/VXO where both exist (FRED VXVCLS reaches 2007-12 vs Cboe 2009-09).
    "CBOE_VIX":     (False, 1, "Cboe VIX close, direct archive. 1990-01-02+."),
    "CBOE_VIX3M":   (False, 1, "Cboe VIX3M close. 2009-09-18+ -- SHORTER than FRED VXVCLS."),
    "CBOE_VIX6M":   (False, 1, "Cboe VIX6M close. 2008-01-02+."),
    "CBOE_VIX9D":   (False, 1, "Cboe VIX9D close. 2011-01-04+."),
    "CBOE_SKEW":    (False, 1, "Cboe SKEW index. 1990-01-02+. F4, Phase 5."),
    "CBOE_VVIX":    (False, 1, "Cboe VVIX. 2006-03+. Earliest rows erratic "
                               "(2006-03-06 71.73 then 2006-03-15 15.71) -- loaded "
                               "as-is, not cleaned."),
    "CBOE_COR1M":   (False, 1, "Cboe 1-month implied correlation. 2006-01-03+."),
    "CBOE_COR3M":   (False, 1, "Cboe 3-month implied correlation. 2006-01-03+."),
    "CBOE_VXO":     (False, 1, "Cboe VXO close. 1993-01-29+. FRED VXOCLS is longer."),
    "CBOE_VXO_CURRENT": (False, 1, "Cboe VXO 2004-2021 file."),
    "CBOE_PC_TOTAL":  (False, 1, "Cboe total put/call ratio. ENDS 2019-10-04; "
                                 "2004-01..2006-10 hole between archive files."),
    "CBOE_PC_INDEX":  (False, 1, "Cboe index put/call ratio. ENDS 2019-10-04."),
    "CBOE_PC_EQUITY": (False, 1, "Cboe equity put/call ratio. ENDS 2019-10-04."),
    "CBOE_PC_VIX":    (False, 1, "Cboe VIX put/call ratio. ENDS 2019-10-04."),
    "VXVCLS":        (False, 1, "Cboe VIX3M close via FRED. Not restated."),
    "VXOCLS":       (False, 1, "Cboe VXO close via FRED. Not restated. "
                               "DISCONTINUED 2021-09-23 -- do not treat a missing "
                               "value after that date as staleness."),
    "SPY_CLOSE":    (False, 1, "SPY daily close from prices.db/raw_bars. A settled "
                               "close is not restated (splits are handled upstream). "
                               "PROXY for SPX -- labelled in every S2 reading. "
                               "Coverage from 2016-07-18 only."),

    # ---- REVISED: statistical estimates. Require true ALFRED vintages. ----
    "CSUSHPINSA":   (True,  60, "Case-Shiller: revised monthly. ALFRED required."),
    "HOUST":        (True,  17, "Housing starts: routinely revised. ALFRED required."),
    "ABCOMP":       (True,   7, "ABCP outstanding, Fed weekly. Revisable. ALFRED required."),
    "DRTSCILM":     (True,  35, "SLOOS, quarterly, ~5w lag per registry. ALFRED required."),
    "PAYEMS":       (True,   5, "Payrolls: heavily revised. ALFRED required."),
    "GDPC1":        (True,  30, "Real GDP: three estimates + annual revisions. ALFRED."),
    "INDPRO":       (True,  17, "Industrial production: revised. ALFRED required."),
    "UNRATE":       (True,   5, "Unemployment rate: revised w/ annual benchmarks. ALFRED."),
}


def is_revisable(series_id: str) -> bool:
    """Unknown series default to REVISABLE -- the safe direction (yields NA)."""
    meta = SERIES_META.get(series_id)
    return True if meta is None else meta[0]


def pub_lag_days(series_id: str) -> int:
    meta = SERIES_META.get(series_id)
    return 1 if meta is None else meta[1]


def derivable_pub_date(series_id: str) -> bool:
    """True only when pub_date may be derived as obs_date + publication_lag."""
    return series_id in SERIES_META and not SERIES_META[series_id][0]


def note(series_id: str) -> str:
    meta = SERIES_META.get(series_id)
    return "" if meta is None else meta[2]
