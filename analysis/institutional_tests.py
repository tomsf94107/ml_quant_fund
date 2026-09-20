#!/usr/bin/env python3
"""
institutional_tests.py — T1 through T4, against the bars fixed in
PREREG_institutional_13f.md before the data landed.

READ-ONLY. Benchmarked against SPY over identical bars.

THE PRIOR, WHICH IS NOT ENCOURAGING
    Fama & French (2010), 1984-2006: in aggregate active managers have ZERO
    alpha before costs and negative after, and an investor buying a portfolio
    from the TOP THREE PERCENT can expect an alpha of zero. The copycat
    literature's edge (Frank et al. 2004; Verbeek & Wang 2013) is the FEE
    SAVING -- copycats match gross return and skip the ~1% expense ratio -- and
    that does not apply to someone who was never paying the fee.

    13F also runs on the same 45-day clock as congressional disclosure, which
    produced eight nulls across eight angles on 2026-09-18, and Form 4 open-
    market purchases -- faster, attributed, own money -- came back NEGATIVE at
    every horizon on 2026-09-20.

    T1 is expected to fail. It is run anyway, because "institutions in
    aggregate have no edge" measured here is worth more than the same claim
    inherited from a paper.

ENTRY AT THE FILING DATE
    13F is due 45 days after quarter end -- Feb 17, May 15, Aug 14, Nov 16.
    Entry is the close AFTER report_date + 45 days. Most managers file at the
    deadline (Berkshire, Coatue and Alkeon all filed 2026-08-14 for Q2); some
    file up to 8 days early. Using +45 for everyone is therefore conservative:
    it assumes the information arrived later than it did for the early filers.
    Keying on quarter end instead would embed 45 days of look-ahead, which is
    the error the congressional ingest was built to avoid.

EXCLUSIONS, AND WHY EACH ONE MATTERS
    Market makers        Citadel, CTC, Millennium and Morgan Stanley hold
                         options as dealer INVENTORY, not as a view. Citadel's
                         2026-06-30 book is 68% Options.
    Options              13F reports LONG positions only. Short calls and short
                         puts never appear, so a put line may hedge something
                         unreported. Shares only, throughout.
    Fund / Debt / Pref   not equity bets.
    Unpriced tickers     1,621 of 3,587 held names are in raw_bars. The rest
                         are foreign listings, delisted names and Bloomberg-
                         style identifiers like 1023761D.

USAGE
    python analysis/institutional_tests.py
    python analysis/institutional_tests.py --test 2
"""
import argparse
import sqlite3
import statistics as st
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INST, PX, DARK = (ROOT / "institutions.db", ROOT / "prices.db",
                  ROOT / "institutional_trades.db")

MM = {"CITADEL ADVISORS LLC", "CTC LLC", "MILLENNIUM MANAGEMENT LLC",
      "SUSQUEHANNA INTERNATIONAL GROUP, LLP", "JANE STREET GROUP, LLC",
      "OPTIVER HOLDING B.V.", "IMC-CHICAGO, LLC", "MORGAN STANLEY"}


def filed(report_date):
    """13F is due 45 days after quarter end. Conservative for early filers."""
    y, m, dd = (int(x) for x in report_date.split("-"))
    return (date(y, m, dd) + timedelta(days=45)).isoformat()


def load_bars(con, tickers):
    q = ",".join("?" * len(tickers))
    acc = defaultdict(lambda: ([], []))
    for tk, d, c in con.execute(
            f"SELECT ticker, d, close FROM raw_bars WHERE ticker IN ({q}) "
            f"AND close > 0 ORDER BY ticker, d", tickers):
        acc[tk][0].append(d)
        acc[tk][1].append(float(c))
    return dict(acc)


def fwd(b, anchor, h):
    ds, cs = b
    i = bisect_right(ds, anchor)
    if i >= len(ds) or i + h >= len(ds):
        return None
    return cs[i + h] / cs[i] - 1.0


def mdd(b, anchor, h):
    """Worst peak-to-trough inside the window. T2 is a RISK claim, and a mean
    return hides the path -- two names can end flat with very different
    drawdowns, and the drawdown is what forces a sale."""
    ds, cs = b
    i = bisect_right(ds, anchor)
    if i >= len(ds) or i + h >= len(ds):
        return None
    w = cs[i:i + h + 1]
    peak, worst = w[0], 0.0
    for p in w:
        peak = max(peak, p)
        worst = min(worst, p / peak - 1.0)
    return worst


def report(label, vals, bar_pp, need_pos=0.55, split=None):
    if len(vals) < 30:
        print(f"  {label:34}{len(vals):>6}   too few")
        return False
    n = len(vals)
    m = st.mean(vals) * 100
    pos = sum(1 for v in vals if v > 0) / n
    sd = st.stdev(vals) * 100 if n > 1 else 0.0
    t = m / (sd / n ** 0.5) if sd else 0.0
    e = l = float("nan")
    if split:
        early = [v for v, y in split if y <= 2020]
        late = [v for v, y in split if y > 2020]
        if len(early) > 20: e = st.mean(early) * 100
        if len(late) > 20: l = st.mean(late) * 100
    ok = (m > bar_pp and pos >= need_pos
          and (e != e or l != l or (e > 0) == (l > 0)))
    print(f"  {label:34}{n:>6}{m:>+9.2f}{100*pos:>7.0f}%{t:>7.2f}"
          f"{e:>+9.2f}{l:>+9.2f}  {'PASS' if ok else ''}")
    return ok


def load_panel(con):
    rows = con.execute("""
        SELECT name, report_date, ticker, units_change, value, pct_total
        FROM inst_holdings
        WHERE security_type = 'Share' AND ticker IS NOT NULL
          AND report_date >= '2016-01-01'
    """).fetchall()
    return [r for r in rows if r[0] not in MM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=60)
    a = ap.parse_args()
    H = a.horizon

    ic = sqlite3.connect(f"file:{INST}?mode=ro", uri=True)
    panel = load_panel(ic)
    ic.close()
    if not panel:
        print("inst_holdings is empty — run the ingest first.")
        return
    qs = sorted({r[1] for r in panel})
    print(f"{len(panel):,} share rows, {len({r[0] for r in panel})} managers "
          f"(market makers excluded), {len(qs)} quarters {qs[0]}..{qs[-1]}\n")

    pc = sqlite3.connect(f"file:{PX}?mode=ro", uri=True)
    want = sorted({r[2] for r in panel} | {"SPY"})
    B = load_bars(pc, want)
    pc.close()
    spy = B.get("SPY")
    if not spy:
        print("SPY missing from raw_bars"); return
    print(f"{sum(1 for t in want if t in B):,} of {len(want):,} tickers priced\n")

    hdr = (f"  {'cut':34}{'n':>6}{'mean%':>9}{'pos':>7}{'t':>7}"
           f"{'<=2020':>9}{'>2020':>9}")

    # ── T1: does consensus BUYING predict? ──────────────────────────────
    if a.test in (0, 1):
        print(f"T1  consensus buying, h={H}, bar > 1pp and >= 70% positive")
        print(hdr)
        byq = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for nm, q, tk, uc, val, pct in panel:
            if uc is None:
                continue
            byq[q][tk][0 if uc > 0 else 1] += 1
        top, bot = [], []
        for q, tks in byq.items():
            f = filed(q)
            scored = [(tk, b - s) for tk, (b, s) in tks.items()
                      if tk in B and (b + s) >= 3]
            if len(scored) < 20:
                continue
            scored.sort(key=lambda x: -x[1])
            k = max(1, len(scored) // 10)
            y = int(q[:4])
            for tk, _ in scored[:k]:
                r, s_ = fwd(B[tk], f, H), fwd(spy, f, H)
                if r is not None and s_ is not None:
                    top.append((r - s_, y))
            for tk, _ in scored[-k:]:
                r, s_ = fwd(B[tk], f, H), fwd(spy, f, H)
                if r is not None and s_ is not None:
                    bot.append((r - s_, y))
        report("top decile bought", [v for v, _ in top], 1.0, 0.70, top)
        report("bottom decile", [v for v, _ in bot], 1.0, 0.70, bot)
        print()

    # ── T2: does CROWDING predict drawdown? ─────────────────────────────
    if a.test in (0, 2):
        print(f"T2  crowding -> drawdown, h={H}, bar 2pp WORSE for crowded")
        print(hdr)
        hi, rest = [], []
        for q in qs:
            f = filed(q)
            cnt = defaultdict(int)
            for nm, qq, tk, uc, val, pct in panel:
                if qq == q and tk in B:
                    cnt[tk] += 1
            if len(cnt) < 50:
                continue
            vals = sorted(cnt.values())
            cut = vals[int(len(vals) * 0.8)]
            y = int(q[:4])
            for tk, c in cnt.items():
                dd = mdd(B[tk], f, H)
                if dd is None:
                    continue
                (hi if c >= cut else rest).append((dd, y))
        # Sign convention: drawdown is negative, so a MORE negative mean for
        # the crowded group is the hypothesis. Reported as-is.
        report("crowded quintile, worst DD", [v for v, _ in hi], -99, 0.0, hi)
        report("the rest, worst DD", [v for v, _ in rest], -99, 0.0, rest)
        if hi and rest:
            gap = (st.mean([v for v, _ in hi]) - st.mean([v for v, _ in rest])) * 100
            print(f"  crowded are {gap:+.2f}pp {'worse' if gap < 0 else 'better'}"
                  f" — bar is 2pp worse")
        print()

    # ── T3: thesis shift ────────────────────────────────────────────────
    if a.test in (0, 3):
        print("T3  liquidate-and-rebuild, bar 20+ events and > 2pp at 2 quarters")
        counts = defaultdict(dict)
        for nm, q, tk, uc, val, pct in panel:
            counts[nm][q] = counts[nm].get(q, 0) + 1
        events = 0
        for nm, byq2 in counts.items():
            seq = sorted(byq2.items())
            for i in range(1, len(seq)):
                prev_n, now_n = seq[i - 1][1], seq[i][1]
                if prev_n >= 5 and now_n <= prev_n * 0.3:
                    events += 1
        print(f"  {events} liquidation events (position count fell > 70% "
              f"quarter on quarter)")
        if events < 20:
            print("  FAILS the 20-event minimum — not testable on this panel.")
        print()

    # ── T4: crowding x daily dark-pool flow ─────────────────────────────
    if a.test in (0, 4):
        print(f"T4  crowded + negative dark-pool flow, h={H}")
        print(hdr)
        dc = sqlite3.connect(f"file:{DARK}?mode=ro", uri=True)
        # Dark pool covers the recent window only, so T4 is restricted to
        # quarters where flow data exists -- stated rather than silently
        # producing a tiny sample.
        flow = {}
        for tk, b, s in dc.execute("""
            SELECT ticker,
                   SUM(CASE WHEN side='BUY'  THEN notional_usd ELSE 0 END),
                   SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END)
            FROM institutional_trades
            WHERE side IN ('BUY','SELL')
              AND trade_date >= date((SELECT MAX(trade_date) FROM institutional_trades),
                                     '-90 days')
            GROUP BY ticker"""):
            t = (b or 0) + (s or 0)
            if t > 0:
                flow[tk] = ((b or 0) - (s or 0)) / t
        dc.close()
        q = qs[-1]
        f = filed(q)
        cnt = defaultdict(int)
        for nm, qq, tk, uc, val, pct in panel:
            if qq == q and tk in B:
                cnt[tk] += 1
        if cnt:
            vals = sorted(cnt.values())
            cut = vals[int(len(vals) * 0.8)]
            dist = [tk for tk, c in cnt.items()
                    if c >= cut and flow.get(tk, 0) < -0.10]
            print(f"  latest quarter {q}, filed {f}")
            print(f"  {len(dist)} crowded names with dark-pool flow < -10%")
            print("  Forward returns cannot be measured yet — the window has")
            print("  not matured. T4 needs one more quarter, re-run after "
                  f"{filed(q)[:7]} + 3 months.")
        print()

    print("  Bars are fixed in PREREG_institutional_13f.md, written before the")
    print("  backfill completed. A cut that fails is closed, not retuned.")
    print("  SURVIVORSHIP: the roster came from the CURRENT institution list,")
    print("  so managers who closed were never probed. Every result above is")
    print("  optimistic by an unknown amount.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
