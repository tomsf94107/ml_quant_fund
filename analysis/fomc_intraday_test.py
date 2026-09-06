#!/usr/bin/env python3
"""
fomc_intraday_test.py — is there an intraday pattern around FOMC announcements?

Fetches SPY 1-minute bars for FOMC dates only. Writes nothing to prices.db
beyond whatever price_cache persists; stores its own results nowhere.

THE OBSERVATION THIS TESTS
    A single Warsh speech was observed with the index rising ~30 minutes before
    and falling 15-30 minutes into it. One instance is what randomness looks
    like; 89 FOMC dates can tell the difference.

WHAT THE LITERATURE SAYS, AND WHY THE ANSWER MATTERS
    Lucca & Moench (2015) documented a large pre-FOMC drift: US equities rose
    on average 49 basis points in the 24 hours before scheduled announcements,
    roughly 80% of the annual equity premium, over September 1994 to March 2011.

    It then died. Kurov, Wolfe & Gilbert (2020) extend the sample to December
    2019 and find the drift "essentially disappeared after 2015" in
    announcements both with and without a press conference, attributing it to
    reduced uncertainty as the Fed became more transparent. Kurov & Gu (2016)
    put the break at 2011; Ben Dor & Rosa (2019) find no evidence from 2011 to
    2017.

    This fund's own cross-sectional test on 2016-2026 found nothing: the
    beta-return IC on announcement days was LOWER than on other days at every
    horizon, and FOMC days were the worst at h=1 (IC -0.0335). That is the
    expected result for a sample beginning after the effect died.

    So the prior is that this test finds nothing either. It is worth running
    anyway because it measures a DIFFERENT quantity -- an intraday, index-level
    move rather than a daily cross-sectional one -- and because a single vivid
    observation deserves a real check rather than a dismissal.

WHY IT IS A DIFFERENT PRODUCT
    Any result here is MARKET TIMING, not stock selection. It would say "the
    index tends to move this way at this hour", which is not something the h=40
    cross-sectional book can use -- that model ranks names against each other on
    the same date, and an index-level move is common to all of them.

    It also cannot currently be traded: the intraday model is documented broken
    (DOWN h=1 at 39% accuracy, UP h=4 at 44%, with an explicit "do NOT trust
    intraday signals" note and a fix plan in docs/intraday/).

METHOD
    For each FOMC date, SPY 1-minute bars. Returns measured in windows around
    14:00 ET, the scheduled announcement time for most of the sample:

      pre_24h     previous close to 13:45 ET      the Lucca-Moench window
      pre_2h      12:00 to 13:45
      pre_30m     13:30 to 13:45
      ann_30m     14:00 to 14:30                  the reaction
      post_2h     14:30 to 16:00
      full_day    previous close to 16:00

    Compared against the same windows on NON-FOMC days in the same weeks, so
    the baseline is contemporaneous rather than a long-run average.

    Split by era: 2016-2019 and 2020-2026, since the literature dates the
    disappearance to 2015 and this sample starts after it.

    python analysis/fomc_intraday_test.py --dry-run
    python analysis/fomc_intraday_test.py
"""
import argparse
import math
import os
import sqlite3
import statistics as st
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

WINDOWS = [
    ("pre_2h",   "12:00", "13:45"),
    ("pre_30m",  "13:30", "13:45"),
    ("ann_30m",  "14:00", "14:30"),
    ("post_2h",  "14:30", "16:00"),
    ("full_day", "09:30", "16:00"),
]


def nw_t(v, lag=1):
    n = len(v)
    if n < 8:
        return None
    m = sum(v) / n
    d = [x - m for x in v]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    con = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    try:
        fomc = [r[0] for r in con.execute(
            "SELECT event_date FROM macro_events WHERE event_code='FOMC' "
            "ORDER BY event_date")]
    except Exception as e:
        con.close()
        raise SystemExit(f"macro_events unavailable ({e}) -- run "
                         f"analysis/etl_macro_calendar.py first")
    con.close()
    today = datetime.now().strftime("%Y-%m-%d")
    fomc = [d for d in fomc if d <= today]
    print(f"{len(fomc)} past FOMC dates, {fomc[0]} .. {fomc[-1]}")
    print(f"  2016-2019: {sum(1 for d in fomc if d < '2020-01-01')}")
    print(f"  2020-2026: {sum(1 for d in fomc if d >= '2020-01-01')}")

    if args.dry_run:
        print(f"\nDRY RUN -- would fetch {args.ticker} 1-minute bars for "
              f"{len(fomc)} FOMC dates plus a control day near each.")
        print("  Windows (ET):")
        for n, a, b in WINDOWS:
            print(f"    {n:<10}{a} - {b}")
        print("\n  Prior from the literature: NOTHING. Kurov, Wolfe & Gilbert")
        print("  (2020) find the pre-FOMC drift 'essentially disappeared after")
        print("  2015'. This sample starts 2016.")
        return

    sys.path.insert(0, ".")
    from features import massive_client as mc
    import pandas as pd

    # Massive's minute history starts between 2016 and 2017 on this tier:
    # 2016-03-16 returns 403, 2017-02-01 returns 696 bars. So the sample is
    # 2017 onward, which still spans the era boundary the literature cares
    # about -- Kurov, Wolfe & Gilbert date the disappearance to 2015, while
    # Boguth et al. found the drift persisted for press-conference meetings
    # through 2017.
    dates = [d for d in fomc if d >= "2017-01-01"]
    if args.limit:
        dates = dates[:args.limit]
    print(f"\n  fetching {args.ticker} 1-minute bars for {len(dates)} FOMC "
          f"dates from 2017\n")

    def windows_for(day):
        """Returns per window for one session, as fractions."""
        try:
            df = mc.download(args.ticker, start=day, end=day, interval="1m",
                             progress=False)
        except Exception:
            return None
        if df is None or len(df) < 200:
            return None
        if hasattr(df.columns, "levels"):
            df = df.xs(args.ticker, axis=1, level=1, drop_level=True) \
                if args.ticker in df.columns.get_level_values(1) else df
        try:
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx = idx.tz_convert("America/New_York")
        except Exception:
            return None
        close = list(df["Close"]) if "Close" in df.columns else None
        if close is None:
            return None
        hm = [f"{t.hour:02d}:{t.minute:02d}" for t in idx]
        out = {}
        for name, a, b in WINDOWS:
            ia = next((i for i, t in enumerate(hm) if t >= a), None)
            ib = next((i for i in range(len(hm) - 1, -1, -1)
                       if hm[i] <= b), None)
            if ia is None or ib is None or ib <= ia:
                continue
            p0, p1 = close[ia], close[ib]
            if p0:
                out[name] = (p1 - p0) / p0
        return out or None

    fomc_res = {n: [] for n, _, _ in WINDOWS}
    ctrl_res = {n: [] for n, _, _ in WINDOWS}
    era = {"2017-2019": {n: [] for n, _, _ in WINDOWS},
           "2020-2026": {n: [] for n, _, _ in WINDOWS}}
    n_ok = n_fail = 0
    fomc_set = set(fomc)

    for i, d in enumerate(dates, 1):
        w = windows_for(d)
        if w:
            n_ok += 1
            for k, v in w.items():
                fomc_res[k].append(v)
                era["2017-2019" if d < "2020-01-01" else "2020-2026"][k].append(v)
        else:
            n_fail += 1
        # control: the nearest weekday 7 days earlier that is NOT an FOMC date
        c = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=7))
        while c.weekday() >= 5 or c.strftime("%Y-%m-%d") in fomc_set:
            c -= timedelta(days=1)
        cw = windows_for(c.strftime("%Y-%m-%d"))
        if cw:
            for k, v in cw.items():
                ctrl_res[k].append(v)
        if i % 20 == 0:
            print(f"    {i}/{len(dates)}  {n_ok} ok  {n_fail} failed")

    print(f"\n  {n_ok} FOMC sessions fetched, {n_fail} failed, "
          f"{len(ctrl_res['full_day'])} controls\n")
    if n_ok < 20:
        print("  too few sessions to conclude anything")
        return

    print(f"  {'window':<10}{'FOMC':>10}{'NW t':>8}{'control':>11}"
          f"{'diff':>10}{'n':>6}")
    for name, _, _ in WINDOWS:
        f = fomc_res[name]
        c = ctrl_res[name]
        if len(f) < 20:
            continue
        t_ = nw_t(f) or 0.0
        diff = st.mean(f) - (st.mean(c) if c else 0.0)
        print(f"  {name:<10}{100*st.mean(f):>+9.3f}%{t_:>+8.2f}"
              + (f"{100*st.mean(c):>+10.3f}%" if c else f"{'-':>11}")
              + f"{100*diff:>+9.3f}%{len(f):>6}")

    print(f"\n  BY ERA (FOMC sessions only)")
    print(f"  {'window':<10}{'2017-2019':>12}{'n':>5}{'2020-2026':>12}{'n':>5}")
    for name, _, _ in WINDOWS:
        a = era["2017-2019"][name]
        b = era["2020-2026"][name]
        if len(a) < 5 or len(b) < 5:
            continue
        print(f"  {name:<10}{100*st.mean(a):>+11.3f}%{len(a):>5}"
              f"{100*st.mean(b):>+11.3f}%{len(b):>5}")

    print("\n  Lucca & Moench measured +49bp over the 24 hours before the")
    print("  announcement, roughly 80% of the annual equity premium, for")
    print("  1994-2011. Kurov, Wolfe & Gilbert find it 'essentially")
    print("  disappeared after 2015'. A pre_2h or pre_30m figure near zero, or")
    print("  no different from the control column, confirms that here.\n")
    print("  ann_30m is the separate question: does the index move on the")
    print("  release itself, as observed anecdotally? That is a reaction, not")
    print("  a drift, and no paper found rules it out for this era.\n")
    print("  Any result is MARKET TIMING, not stock selection. The h=40 book")
    print("  ranks names against each other on one date and cannot use an")
    print("  index-level move. And the intraday model is documented broken --")
    print("  DOWN h=1 at 39% accuracy -- so nothing here is tradeable today.")


if __name__ == "__main__":
    main()
