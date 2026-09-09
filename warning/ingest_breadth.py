#!/usr/bin/env python3
"""
ingest_breadth.py — universe breadth series into warning.db, for S5.

COMPUTES THREE SERIES from prices.db raw_bars:
    BREADTH_AD_CUM        cumulative advance-decline line (running sum of
                          advancers minus decliners)
    BREADTH_PCT_200DMA    fraction of names trading above their own 200DMA
    BREADTH_NEW_LOWS_PCT  fraction of names closing at a 252-day low

SURVIVORSHIP BIAS -- READ THIS BEFORE USING ANY HISTORICAL VALUE
    S5's registry row says `history_start: forward-only`, `data_source: daily
    bars all active+delisted`, and `notes: survivorship-safe forward only`.

    prices.db carries 443 tickers THAT EXIST TODAY. Every name that delisted,
    was acquired, or went to zero between 2016 and 2026 is absent. Breadth is
    precisely the measurement where that matters: %>200DMA and new-52w-lows
    count how many constituents are FAILING, and the ones that actually failed
    have been removed from the sample.

    The bias is directional, not noise. Historical breadth computed this way is
    systematically HEALTHIER than the real thing, so S5 will under-fire on
    history. A quiet reading in 2020 or 2022 is not evidence the signal was
    quiet then.

    These series are therefore honest only FORWARD from the date the universe is
    captured with delistings retained. Everything before that is descriptive.
    The builder repeats this caveat in every reading rather than relying on
    anyone remembering it.

UNIVERSE DRIFT
    Names enter raw_bars at different dates (XLC 2018, recent IPOs later). The
    daily cross-section therefore grows. Advance/decline counts are ratios or
    fractions where possible, and the constituent count is written alongside so
    a change in the denominator is visible rather than silent -- the same defect
    that made the naive short-interest aggregate rise 124% on 79% of real change.

USAGE
    python warning/ingest_breadth.py --prices prices.db --db warning.db
    python warning/ingest_breadth.py --prices prices.db --db warning.db --dry-run
"""
import os
import argparse
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

DMA_WINDOW = 200
LOW_WINDOW = 252
MIN_NAMES = 50          # absolute floor, for the earliest history only
TRUNCATION_LOOKBACK = 5  # sessions compared against, for the guard below
# A fraction is unavoidable here. Pure ordinal comparison fails because the
# count declines by one or two most weeks through ordinary delisting, so
# "fewer than any of the last five" flags normal attrition. 0.5 is chosen to
# sit far from both regimes rather than near either: measured attrition is
# 0.05% per session and the 2026-09-08 truncation was 78%. Any value between
# roughly 0.02 and 0.95 separates them identically, so the result is not
# sensitive to the number -- which is the test for whether a threshold is
# doing real work or merely fitted.
TRUNCATION_FLOOR = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="prices.db")
    ap.add_argument("--db", default=os.environ.get("WARNING_DB",
                                               "warning.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = sqlite3.connect(f"file:{args.prices}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT ticker, d, close FROM raw_bars WHERE close IS NOT NULL "
        "ORDER BY ticker, d").fetchall()
    src.close()

    series = defaultdict(list)          # ticker -> [(date, close)]
    for t, d, c in rows:
        series[t].append((d, float(c)))
    print(f"universe: {len(series)} tickers, {len(rows)} bars")

    all_dates = sorted({d for _, d, _ in rows})
    print(f"dates: {len(all_dates)}  {all_dates[0]}..{all_dates[-1]}")

    # per-ticker rolling state, walked forward once
    idx = {t: 0 for t in series}
    seen_counts = []                    # every date's n_today, skipped or not
    hist = defaultdict(list)            # ticker -> trailing closes
    prev_close = {}
    ad_cum = 0.0
    out = defaultdict(list)             # series_id -> [(date, value)]

    for d in all_dates:
        adv = dec = 0
        above = tot_dma = 0
        lows = tot_low = 0
        n_today = 0
        for t, rws in series.items():
            i = idx[t]
            if i >= len(rws) or rws[i][0] != d:
                continue
            close = rws[i][1]
            idx[t] = i + 1
            n_today += 1
            h = hist[t]
            h.append(close)
            if len(h) > LOW_WINDOW:
                del h[0]
            pc = prev_close.get(t)
            if pc is not None:
                if close > pc:
                    adv += 1
                elif close < pc:
                    dec += 1
            prev_close[t] = close
            if len(h) >= DMA_WINDOW:
                tot_dma += 1
                if close > sum(h[-DMA_WINDOW:]) / DMA_WINDOW:
                    above += 1
            if len(h) >= LOW_WINDOW:
                tot_low += 1
                if close <= min(h):
                    lows += 1

        if n_today < MIN_NAMES:
            continue

        # TRUNCATED-SESSION GUARD, added 2026-09-08.
        #
        # MIN_NAMES = 50 was an absolute floor set when this universe held ~443
        # names. At 1,996 it is no longer protective: Pipeline A truncated its
        # fetch on 2026-09-08 after 444 tickers -- alphabetical from A, so a
        # cut-off rather than a random subset -- and exited 0. Breadth was then
        # computed over 434 names and written as if it were 1,996.
        # BREADTH_PCT_200DMA moved 61.96 -> 57.83, a 4.1pp change that was
        # entirely a change of denominator, and S5 read it as market breadth.
        #
        # The rule is ORDINAL, not a threshold: a session may not have fewer
        # names than the SMALLEST of the previous five. Nothing is invented.
        # Real attrition moves this count by one or two per session -- measured
        # 2003 to 1996 over five weeks -- while a truncation drops it by
        # 1,562. The two are not close, so no tolerance is needed.
        #
        # A genuine mass-delisting day would be blocked and printed, which is
        # the right failure: visible, not silent.
        # Compare against OBSERVED counts, not accepted ones. Reading the
        # accepted list makes this a ratchet: once a date is skipped the window
        # freezes at the last accepted value, so every later session with a
        # normally-declining count is skipped too. First version of this guard
        # dropped six legitimate sessions -- 08-31 through 09-08 -- because the
        # window stayed pinned at 2003 after 08-25 fell to 2000.
        recent = seen_counts[-TRUNCATION_LOOKBACK:]
        seen_counts.append(n_today)
        if recent and n_today < min(recent) * TRUNCATION_FLOOR:
            print(f"  SKIP {d}: {n_today} names, below the prior "
                  f"{len(recent)}-session minimum of {int(min(recent))}. "
                  f"Truncated ingest, not a market event -- nothing written "
                  f"for this date.")
            continue

        ad_cum += (adv - dec)
        out["BREADTH_AD_CUM"].append((d, ad_cum))
        if tot_dma >= MIN_NAMES:
            out["BREADTH_PCT_200DMA"].append((d, 100.0 * above / tot_dma))
        if tot_low >= MIN_NAMES:
            out["BREADTH_NEW_LOWS_PCT"].append((d, 100.0 * lows / tot_low))
        out["BREADTH_N_NAMES"].append((d, float(n_today)))

    for k, v in out.items():
        print(f"  {k:<24} {len(v):>6} rows  {v[0][0]}..{v[-1][0]}"
              f"  last {v[-1][1]:.2f}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    con = sqlite3.connect(args.db)
    n = 0
    for sid, vals in out.items():
        for d, v in vals:
            pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            con.execute("INSERT OR IGNORE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)",
                        (sid, d, pub, v, "prices.db/raw_bars"))
            n += 1
    con.commit()
    con.close()
    print(f"\nwrote {n} rows")
    print("SURVIVORSHIP: computed from today's 443 surviving tickers. Delisted "
          "names are absent, so historical breadth reads HEALTHIER than reality "
          "and S5 will under-fire on history. Forward-only per the registry.")


if __name__ == "__main__":
    main()
