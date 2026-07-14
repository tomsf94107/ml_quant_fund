#!/usr/bin/env python3
"""
utils/market_calendar.py -- ONE source of truth for "was the market open?"

WHY THIS EXISTS
    TWO copies of _last_completed_session() exist:
        features/builder.py:707
        features/massive_client.py:465
    Both do exactly this:
        while _d.weekday() >= 5:      # Sat=5, Sun=6
            _d -= timedelta(days=1)
    Weekends only. NO HOLIDAYS. And is_trading_day() in scripts/daily_runner.py
    says so in its own docstring: "basic check, ignores holidays".

    Measured consequence, 2026-07-14:
        2026-06-19 (Juneteenth)    387 predictions   0 bars   1,138 outcomes
        2026-07-03 (July 4 obs.)   398 predictions   0 bars   1,186 outcomes
    785 predictions and 2,324 outcomes written against sessions that never
    happened. No entry price. No forward return. They corrupt every live
    accuracy number: the dashboard, the SELL-signal validation query, and
    Pipeline B's nightly sanity guard.

DESIGN
    Holidays are COMPUTED from NYSE rules, not hardcoded. A hardcoded list runs
    out and then fails SILENTLY -- the exact bug class this file kills.

    One-off closures (9/11, state funerals, Sandy) cannot be computed and ARE
    listed, with an explicit horizon: ask about a date past HORIZON_YEAR and
    this raises instead of quietly guessing.

    validate_against_db() cross-checks the calendar against prices.db raw_bars
    (~1,133 real sessions, 2022-2026). If calendar and reality disagree, the
    CALENDAR is wrong. Fix it here, not in the data.

USAGE
    from utils.market_calendar import is_trading_day, last_completed_session

    is_trading_day(date(2026, 6, 19))     -> False   (Juneteenth)
    last_completed_session()              -> date, holiday-aware, ET-anchored

SELF-TEST
    python -m utils.market_calendar          # validates against raw_bars
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Past this year the computed rules are still fine, but the one-off closure list
# is not maintained. Raise rather than guess.
HORIZON_YEAR = 2035

# Closures that have no rule. Extend as they happen.
SPECIAL_CLOSURES: set[date] = {
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),                      # Reagan funeral
    date(2007, 1, 2),                       # Ford funeral
    date(2012, 10, 29), date(2012, 10, 30), # Sandy
    date(2018, 12, 5),                      # Bush funeral
    date(2025, 1, 9),                       # Carter funeral
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of month. n=-1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    d -= timedelta(days=(d.weekday() - weekday) % 7)
    return d


def _easter(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date | None:
    """NYSE observance shift. Sat -> preceding Fri. Sun -> following Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> set[date]:
    """Every NYSE full-day closure in `year`."""
    if year > HORIZON_YEAR:
        raise ValueError(
            f"market_calendar: {year} is past HORIZON_YEAR={HORIZON_YEAR}. "
            f"The computed rules still hold, but SPECIAL_CLOSURES is unmaintained. "
            f"Review and bump HORIZON_YEAR rather than trusting this silently."
        )
    h: set[date] = set()

    # New Year's Day. NYSE quirk: if Jan 1 is a Saturday the market does NOT
    # close the preceding Friday (Dec 31). Only the Sunday->Monday shift applies.
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        h.add(ny + timedelta(days=1))
    elif ny.weekday() < 5:
        h.add(ny)

    h.add(_nth_weekday(year, 1, 0, 3))          # MLK, 3rd Mon Jan
    h.add(_nth_weekday(year, 2, 0, 3))          # Presidents, 3rd Mon Feb
    h.add(_easter(year) - timedelta(days=2))    # Good Friday
    h.add(_nth_weekday(year, 5, 0, -1))         # Memorial, last Mon May

    if year >= 2022:                            # Juneteenth, from 2022
        j = _observed(date(year, 6, 19))
        if j:
            h.add(j)

    ind = _observed(date(year, 7, 4))
    if ind:
        h.add(ind)

    h.add(_nth_weekday(year, 9, 0, 1))          # Labor, 1st Mon Sep
    h.add(_nth_weekday(year, 11, 3, 4))         # Thanksgiving, 4th Thu Nov

    xm = _observed(date(year, 12, 25))
    if xm:
        h.add(xm)

    h |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return {d for d in h if d.weekday() < 5}


def is_trading_day(d: date | str) -> bool:
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    if d.weekday() >= 5:
        return False
    return d not in holidays(d.year)


def previous_trading_day(d: date | str) -> date:
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_completed_session(now: datetime | None = None) -> date:
    """Last US session whose 17:00 ET close + publish margin has passed.

    Replaces the two weekend-only copies. ET-anchored: the Mac's VN local date
    is a day AHEAD of the US calendar, which produced 403 NOT_AUTHORIZED storms
    and predictions stamped for sessions that had not happened.
    """
    if now is None:
        try:
            from utils.timezone import now_et
            now = now_et()
        except Exception:
            now = datetime.now()
    d = now.date()
    if now.hour < 17:
        d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ── self-test: the calendar must agree with 4.5 years of real bars ───────────
def validate_against_db(db: str | Path | None = None) -> int:
    """Cross-check every computed date against raw_bars. Returns exit code."""
    import sqlite3
    db = Path(db) if db else ROOT / "prices.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    real = {date.fromisoformat(r[0]) for r in
            con.execute("SELECT DISTINCT d FROM raw_bars")}
    con.close()
    if not real:
        print("no bars in raw_bars -- cannot validate")
        return 1

    lo, hi = min(real), max(real)
    print(f"raw_bars: {len(real)} sessions, {lo} -> {hi}\n")

    said_closed_but_traded, said_open_but_no_bars = [], []
    d = lo
    while d <= hi:
        if d.weekday() < 5:
            open_ = is_trading_day(d)
            traded = d in real
            if traded and not open_:
                said_closed_but_traded.append(d)
            if open_ and not traded:
                said_open_but_no_bars.append(d)
        d += timedelta(days=1)

    print(f"CALENDAR SAYS CLOSED, MARKET TRADED : {len(said_closed_but_traded)}")
    for x in said_closed_but_traded:
        print(f"   {x}  <-- CALENDAR IS WRONG. Fix the rule.")

    print(f"\nCALENDAR SAYS OPEN, NO BARS         : {len(said_open_but_no_bars)}")
    for x in said_open_but_no_bars:
        print(f"   {x}  ({x.strftime('%a')})  -- holiday I missed, OR a data gap")

    print("\n--- known holidays in range, for eyeballing ---")
    for y in range(lo.year, hi.year + 1):
        hs = sorted(h for h in holidays(y) if lo <= h <= hi)
        print(f"  {y}: {', '.join(str(h) for h in hs)}")

    if said_closed_but_traded:
        print("\nFAIL: the calendar closed a day the market actually traded.")
        return 1
    print("\nPASS: no session in raw_bars is marked closed.")
    if said_open_but_no_bars:
        print("      (review the 'no bars' list -- missed holiday vs data gap)")
    return 0


if __name__ == "__main__":
    sys.exit(validate_against_db())


def early_closes(year: int) -> set:
    """NYSE 13:00 ET early closes, derived. Rules verified against NYSE published
    2024-2028 calendars (ICE press release + nyse.com/trade/hours-calendars):
    day-after-Thanksgiving always; Dec 24 when it is a trading day (excludes
    Dec-25-on-Saturday years, when Dec 24 is the observed full holiday);
    Jul 3 when it is a trading day AND Jul 4 falls Mon-Fri (excludes
    Jul-4-on-Saturday years like 2026, when Jul 3 is the observed FULL closure,
    and Jul-4-on-Sunday years, when Fri Jul 3 is a regular full session)."""
    from datetime import date as _d, timedelta as _td
    nov1 = _d(year, 11, 1)
    first_thu = nov1 + _td(days=(3 - nov1.weekday()) % 7)
    ec = {first_thu + _td(weeks=3, days=1)}
    dec24 = _d(year, 12, 24)
    if is_trading_day(dec24):
        ec.add(dec24)
    jul3 = _d(year, 7, 3)
    if is_trading_day(jul3) and _d(year, 7, 4).weekday() < 5:
        ec.add(jul3)
    return {d for d in ec if is_trading_day(d)}


def close_time_et(d):
    """Session close for date d: 13:00 on early-close days, else 16:00."""
    from datetime import date as _d, time as _t
    if isinstance(d, str):
        d = _d.fromisoformat(d)
    return _t(13, 0) if d in early_closes(d.year) else _t(16, 0)


def is_market_open(now=None) -> bool:
    """RTH open test: trading day AND 09:30 <= ET time <= session close
    (early-close aware). Injectable clock for tests. Canonical replacement
    for the three holiday-blind copies in intraday_builder / uw_client /
    intraday_kill_switch."""
    from datetime import datetime as _dt, time as _t
    from zoneinfo import ZoneInfo as _Z
    et = _Z("America/New_York")
    if now is None:
        now = _dt.now(et)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=et)
    else:
        now = now.astimezone(et)
    d = now.date()
    if not is_trading_day(d):
        return False
    return _t(9, 30) <= now.time() <= close_time_et(d)
