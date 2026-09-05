#!/usr/bin/env python3
"""
darkpool_volume_si_test.py — dark-pool VOLUME conditioned on short interest.

READ-ONLY. Writes nothing.

WHY THIS IS NOT A REPEAT OF THE TEN FAILED TESTS
    Dark-pool SIGNED SKEW was closed on this fund's own evidence: 10+ direct
    directional tests on MU found it coincident, not leading. The literature
    explains why. FINRA ATS data never reveals whether a trade was a buy or a
    sell, so any "signed" skew is an INFERENCE from contemporaneous price
    context -- which makes it coincident by construction. The test was not
    wrong; the construction could not have worked.

    A second construction fails for a separate reason. The short-volume ratio
    from FINRA's daily file has a median near 48% across ~12,000 symbols,
    because market makers internalising retail flow book the offsetting side as
    short. A high reading is normal, not bearish.

    The academic construction is neither. Brogaard uses the ratio of dark-pool
    volume to CONSOLIDATED volume -- a volume share, no sign inferred. And
    Boulton et al. ("Short selling and dark pool volume") find the predictive
    form is CONDITIONAL: subsequent returns are lower for HEAVILY SHORTED stocks
    with GREATER dark pool volume, and the effect is stronger for stocks with
    higher uncertainty.

    This fund holds both sides of that interaction and has never tested it:
    institutional_trades.db for dark-pool volume, short_interest.db for
    days-to-cover. Every prior test used skew alone.

WHAT IS MEASURED
    Per rebalance date, cross-sectionally:

      dp_share   dark-pool dollar volume over the trailing window, divided by
                 consolidated dollar volume from raw_bars. No sign, no
                 inference -- just how much of the tape went dark.
      dtc        FINRA days-to-cover as of the last PUBLISHED settlement
                 (settlement + 12 calendar days, since FINRA disseminates ~8
                 business days later). The lag is not optional: entry at
                 settlement is a look-ahead worth ~10% of the SI brick's edge.

    Three tests, in increasing specificity:
      1. dp_share alone -- per-date IC against forward return
      2. dp_share within DTC terciles -- does the sign or size differ?
      3. the Boulton cell: HIGH dtc AND HIGH dp_share, against everything else

    Reported at h=20 and h=40, since this fund's own horizon sweep found the
    feature set carries 40-day information and nothing at 5, and multi-seed
    because three single-seed results reversed on replication on 2026-09-05.

THE BAR
    Harvey, Liu & Zhu argue that given hundreds of published factors and
    extensive data mining, a NEW factor needs a t-ratio above 3.0, not 2.0.
    That is the standard applied here. Anything between 2 and 3 is recorded as
    not established rather than as a finding.

    python analysis/darkpool_volume_si_test.py --seeds 3 --tickers 80
"""
import argparse
import datetime
import math
import os
import sqlite3
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

HORIZONS = (20, 40)
T_HURDLE = 3.0          # Harvey-Liu-Zhu, not the conventional 2.0


def nw_t(series, lag):
    n = len(series)
    if n < 10:
        return None
    m = sum(series) / n
    d = [x - m for x in series]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def spearman(x, y):
    n = len(x)
    if n < 8:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--window", type=int, default=20,
                    help="trailing sessions for the dark-pool volume share")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--universe", default="tickers.txt")
    args = ap.parse_args()

    for f in ("institutional_trades.db", "short_interest.db", "prices.db"):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found -- run from the repo root")

    ic = sqlite3.connect("file:institutional_trades.db?mode=ro", uri=True)
    tabs = [r[0] for r in ic.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    print(f"institutional_trades.db tables: {', '.join(tabs)}")
    dp_tab = next((t for t in tabs if t == "institutional_trades"), None) \
        or next((t for t in tabs if "dark" in t.lower()
                 or "print" in t.lower()), None)
    if dp_tab is None:
        ic.close()
        raise SystemExit(
            "no dark-pool table found. Show the schema and this will be "
            "re-cut:\n  sqlite3 institutional_trades.db '.tables'")
    cols = [r[1] for r in ic.execute(f"PRAGMA table_info({dp_tab})")]
    print(f"  using {dp_tab}: {', '.join(cols)}")
    tcol = next((c for c in cols if c.lower() in
                 ("ticker", "symbol")), None)
    dcol = next((c for c in cols if c.lower() == "trade_date"), None) \
        or next((c for c in cols if c.lower() in
                 ("d", "date", "executed_at")), None)
    # notional_usd, not shares: a volume SHARE must be dollar-on-dollar
    # against raw_bars close*volume, or the ratio is meaningless.
    vcol = next((c for c in cols if c.lower() == "notional_usd"), None) \
        or next((c for c in cols if "premium" in c.lower()
                 or "notional" in c.lower()), None)
    if not all((tcol, dcol, vcol)):
        ic.close()
        raise SystemExit(f"could not identify ticker/date/value columns in "
                         f"{dp_tab}: {cols}")
    print(f"  ticker={tcol}  date={dcol}  value={vcol}\n")

    dp = defaultdict(lambda: defaultdict(float))
    for t, d, v in ic.execute(
            f"SELECT {tcol}, {dcol}, {vcol} FROM {dp_tab} "
            f"WHERE {vcol} IS NOT NULL AND is_dark_pool = 1 "
            f"AND is_canceled = 0"):
        if t and d:
            dp[str(t).upper()][str(d)[:10]] += float(v or 0)
    ic.close()
    print(f"dark-pool: {len(dp)} tickers")

    sc = sqlite3.connect("file:short_interest.db?mode=ro", uri=True)
    si_hist = defaultdict(list)
    for t, d, v in sc.execute(
            "SELECT ticker, settlement_date, days_to_cover FROM "
            "short_interest WHERE days_to_cover IS NOT NULL "
            "AND days_to_cover <= 50 ORDER BY settlement_date"):
        si_hist[str(t).upper()].append((str(d)[:10], float(v)))
    sc.close()
    print(f"short interest: {len(si_hist)} tickers")

    def dtc_asof(t, d):
        """Latest settlement at least 12 days old. FINRA disseminates ~8
        business days after settlement, so anything fresher was not public --
        entry at settlement is worth ~10% of the SI brick's measured edge."""
        h = si_hist.get(t)
        if not h:
            return None
        cut = (datetime.date.fromisoformat(d)
               - datetime.timedelta(days=12)).isoformat()
        best = None
        for sd, v in h:
            if sd <= cut:
                best = v
            else:
                break
        return best

    pc = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    bars = defaultdict(list)
    for t, d, c, v in pc.execute(
            "SELECT ticker, d, close, volume FROM raw_bars "
            "WHERE close > 0 AND volume > 0"):
        bars[t].append((str(d)[:10], float(c), float(v)))
    pc.close()
    for t in bars:
        bars[t].sort()
    print(f"prices: {len(bars)} tickers\n")

    import random
    uni = [l.strip().upper() for l in open(args.universe) if l.strip()]
    have = [t for t in uni if t in dp and t in si_hist and t in bars]
    print(f"{len(have)} of {len(uni)} names have dark-pool AND short-interest "
          f"AND price data\n")
    if len(have) < 40:
        raise SystemExit("too few names carry all three sources")

    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = have[:]
        random.Random(seed).shuffle(u)
        sample = u[:args.tickers]

        for H in HORIZONS:
            per_date = defaultdict(list)
            for t in sample:
                b = bars[t]
                idx = {d: i for i, (d, _, _) in enumerate(b)}
                for i in range(args.window, len(b) - H):
                    d, c, _ = b[i]
                    dollar = sum(b[k][1] * b[k][2]
                                 for k in range(i - args.window, i))
                    if dollar <= 0:
                        continue
                    dpv = sum(dp[t].get(b[k][0], 0.0)
                              for k in range(i - args.window, i))
                    if dpv <= 0:
                        continue
                    share = dpv / dollar
                    dtc = dtc_asof(t, d)
                    if dtc is None:
                        continue
                    fwd = (b[i + H][1] - c) / c
                    if abs(fwd) > 1.5:
                        continue
                    per_date[d].append((share, dtc, fwd))

            ics, hi_hi, rest = [], [], []
            terc = defaultdict(list)
            for d, rows in per_date.items():
                if len(rows) < args.min_names:
                    continue
                r = spearman([x[0] for x in rows], [x[2] for x in rows])
                if r is not None:
                    ics.append(r)
                ds = sorted(x[1] for x in rows)
                dhi = ds[int(len(ds) * 0.67)]
                dlo = ds[int(len(ds) * 0.33)]
                ss = sorted(x[0] for x in rows)
                shi = ss[int(len(ss) * 0.67)]
                mkt = st.mean(x[2] for x in rows)
                cell = [x[2] for x in rows if x[1] >= dhi and x[0] >= shi]
                other = [x[2] for x in rows if not (x[1] >= dhi
                                                    and x[0] >= shi)]
                if cell and other:
                    hi_hi.append(st.mean(cell) - mkt)
                    rest.append(st.mean(other) - mkt)
                for lab, lo, hi in (("low dtc", -1e9, dlo),
                                    ("mid dtc", dlo, dhi),
                                    ("high dtc", dhi, 1e9)):
                    sub = [x for x in rows if lo <= x[1] < hi]
                    if len(sub) >= 10:
                        rr = spearman([x[0] for x in sub],
                                      [x[2] for x in sub])
                        if rr is not None:
                            terc[lab].append(rr)

            if len(ics) < 20:
                print(f"  seed {seed} h={H}: only {len(ics)} dates, skipped")
                continue
            t_all = nw_t(ics, max(1, H // 15)) or 0.0
            print(f"SEED {seed}, h={H} — {len(ics)} dates")
            print(f"  dp_share IC            {st.mean(ics):+.4f}  "
                  f"NW t {t_all:+.2f}"
                  + ("   PASSES t>3" if abs(t_all) > T_HURDLE else ""))
            for lab in ("low dtc", "mid dtc", "high dtc"):
                v = terc.get(lab, [])
                if len(v) >= 20:
                    tt = nw_t(v, max(1, H // 15)) or 0.0
                    print(f"    within {lab:<9} IC {st.mean(v):+.4f}  "
                          f"NW t {tt:+.2f}")
            if hi_hi:
                tc = nw_t(hi_hi, max(1, H // 15)) or 0.0
                print(f"  Boulton cell (high dtc AND high dp_share)")
                print(f"    excess {100*st.mean(hi_hi):+.3f}pp  NW t {tc:+.2f}"
                      + ("   PASSES t>3" if abs(tc) > T_HURDLE else ""))
                print(f"    everything else {100*st.mean(rest):+.3f}pp")
                agg[("cell", H)].append(st.mean(hi_hi))
            agg[("ic", H)].append(st.mean(ics))
            print()

    print("=" * 62)
    print("ACROSS SEEDS")
    print("=" * 62)
    for H in HORIZONS:
        v = agg.get(("ic", H), [])
        c = agg.get(("cell", H), [])
        if v:
            print(f"  h={H}  dp_share IC {st.mean(v):+.4f}   "
                  f"seeds same sign {sum(1 for x in v if (x<0)==(st.mean(v)<0))}"
                  f"/{len(v)}")
        if c:
            print(f"        Boulton cell {100*st.mean(c):+.3f}pp   "
                  f"seeds + {sum(1 for x in c if x>0)}/{len(c)}")

    print(f"\n  Bar is NW t > {T_HURDLE}, not 2.0: Harvey, Liu & Zhu argue that")
    print("  given hundreds of published factors and extensive data mining, a")
    print("  new factor needs a t-ratio above 3.0. Anything between 2 and 3 is")
    print("  NOT ESTABLISHED, not a finding.\n")
    print("  Boulton predicts NEGATIVE excess in the high-dtc/high-dark-volume")
    print("  cell -- heavily shorted names with heavy dark trading go on to")
    print("  underperform. A positive result there contradicts the paper and")
    print("  should be treated as noise until it repeats.")


if __name__ == "__main__":
    main()
