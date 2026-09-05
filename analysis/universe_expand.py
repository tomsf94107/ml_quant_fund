#!/usr/bin/env python3
"""
universe_expand.py — screen a ~2,000-name liquid US universe.

READ-ONLY by default. Writes nothing unless --write is passed, and then only a
proposed file, never tickers.txt itself.

WHY EXPAND
    The cross-sectional signal found at h=40 on 2026-09-05 shows wide dispersion
    across ticker samples -- prob>=0.70 ranged +2.14pp to +5.46pp over three
    draws of 80 names. That is what a thin cross-section looks like.

    Gu, Kelly & Xiu (2020) run ~30,000 stocks over 60 years; this fund runs 420
    over 5. Their out-of-sample R2 is ~0.4% monthly, detectable only because the
    panel is ~1.8M stock-months. Cross-sectional rank signals get sharper with
    breadth: a top decile of 420 names is 42 stocks, of 2,000 it is 200, and the
    HEAD of that ranking is far more selective.

    Universe size is plausibly a bigger lever than any estimator change. Twenty
    model configurations were tested on 2026-09-05 and all landed in the same
    place; none of them changed the number of names being ranked.

SCREENS, AND WHY EACH
    1. DOLLAR ADV >= $5M. Below that a $105k book cannot fill without impact.
       The h=40 deployment check found median position at 0.001% of dollar ADV,
       so this is not currently binding -- but at 2,000 names it would be, and
       the screen is what keeps it so.
    2. PRICE >= $5. Sub-$5 names carry wide spreads and different microstructure;
       a backtest on them overstates what is achievable.
    3. >= 250 BARS. Matches the existing history gate. Note the separate
       1,000-bar gate on HIGH confidence stays in force downstream.
    4. TOP N BY ADV among what survives.

ON SECTOR BALANCE -- DELIBERATELY NOT IMPOSED
    Forcing equal sector weights distorts the cross-section. If one sector
    genuinely holds more predictable names, an equal-weight mandate discards
    that. GKX take the market as it is, and so does this screen.

    A concentration GUARD is reported instead: if any one sector exceeds ~30% of
    the universe, that is flagged for a human decision rather than silently
    corrected. The distinction matters -- a guard catches something pathological,
    a mandate overrides the data.

SURVIVORSHIP, STATED PLAINLY
    This screens the names that EXIST TODAY. Every company that delisted between
    2016 and now is absent, so any backtest on this universe is survivor-tilted
    and overstates returns.

    Excluding delisted names is CORRECT for a live universe and WRONG for a
    backtest. The bias cannot be removed without delisted price history; it can
    only be sized. si_leg_decomp.py sized it for the SI brick and found it
    near-moot there (79% of the edge in low days-to-cover names, which rarely
    delist). Nobody has sized it for the h=40 signal, and a model picking six
    high-conviction names could well be selecting the volatile ones where
    delisting matters most.

    Expanding the universe makes this WORSE, not better: more names means more
    that will eventually delist, while the history still lacks the ones that
    already did.

    python analysis/universe_expand.py
    python analysis/universe_expand.py --target 2000 --write
"""
import argparse
import os
import sqlite3
import statistics as st
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--min-adv", type=float, default=5e6)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-bars", type=int, default=250)
    ap.add_argument("--sector-cap", type=float, default=0.30)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.prices_db):
        raise SystemExit(f"{args.prices_db} not found -- run from repo root")

    current = []
    if os.path.exists("tickers.txt"):
        current = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    meta = {}
    if os.path.exists("tickers_metadata.csv"):
        for i, line in enumerate(open("tickers_metadata.csv")):
            if i == 0:
                continue
            p = line.strip().split(",")
            if len(p) >= 2 and p[0]:
                meta[p[0].strip().upper()] = p[1].strip()

    con = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    print("WHAT IS ALREADY HERE")
    for tbl, dcol in (("raw_bars", "d"), ("daily_prices", "date")):
        try:
            n, tk, lo, hi = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT ticker), MIN({dcol}), "
                f"MAX({dcol}) FROM {tbl}").fetchone()
            print(f"  {tbl:<14}{n:>12,} rows  {tk:>5} tickers  {lo} .. {hi}")
        except Exception as e:
            print(f"  {tbl:<14}unavailable: {e}")
    print(f"  tickers.txt   {len(current):>12} names")
    print(f"  metadata      {len(meta):>12} names with a sector\n")

    rows = con.execute("""
        SELECT ticker, COUNT(*), MAX(d),
               AVG(CASE WHEN d >= date('now','-60 days')
                        THEN close * volume END),
               AVG(CASE WHEN d >= date('now','-60 days') THEN close END)
        FROM raw_bars WHERE close > 0 AND volume > 0
        GROUP BY ticker
    """).fetchall()
    con.close()

    print(f"SCREENS  (target {args.target}, ADV >= ${args.min_adv/1e6:.0f}M, "
          f"price >= ${args.min_price:.0f}, >= {args.min_bars} bars)")
    cand, drop = [], Counter()
    for tk, nbars, last, adv, px in rows:
        if nbars < args.min_bars:
            drop["too few bars"] += 1
            continue
        if adv is None or px is None:
            drop["no recent data"] += 1
            continue
        if px < args.min_price:
            drop["price < min"] += 1
            continue
        if adv < args.min_adv:
            drop["ADV < min"] += 1
            continue
        cand.append((tk, adv, px, nbars))
    print(f"  {len(rows)} tickers in raw_bars")
    for k, v in drop.most_common():
        print(f"    -{v:>5}  {k}")
    print(f"  {len(cand)} pass all screens")

    cand.sort(key=lambda x: -x[1])
    sel = cand[:args.target]
    print(f"  {len(sel)} selected (top by dollar ADV)\n")

    if len(cand) < args.target:
        print(f"  NOTE: only {len(cand)} names clear the screens, short of the")
        print(f"  {args.target} target. raw_bars holds {len(rows)} tickers, so")
        print(f"  expanding the universe needs a WIDER FETCH first -- the screen")
        print(f"  can only choose among what has been ingested.\n")

    keep = {t for t, _, _, _ in sel}
    print("OVERLAP WITH THE CURRENT UNIVERSE")
    print(f"  currently listed          {len(current):>6}")
    print(f"  of those, pass the screen {len(set(current) & keep):>6}")
    dropped = sorted(set(current) - keep)
    print(f"  currently listed, DROPPED {len(dropped):>6}")
    if dropped:
        print(f"    {', '.join(dropped[:20])}"
              + (f"  ... and {len(dropped)-20} more" if len(dropped) > 20 else ""))
        print("    ^ review before adopting: some are deliberate discretionary")
        print("      coverage, not screen failures.")
    print(f"  NEW names added           {len(keep - set(current)):>6}\n")

    print("SECTOR MIX (from tickers_metadata.csv where known)")
    sec = Counter(meta.get(t, "unknown") for t in keep)
    tot = sum(sec.values())
    for s, n in sec.most_common(12):
        flag = ("  <-- above cap" if s != "unknown"
                and n / tot > args.sector_cap else "")
        print(f"  {s[:28]:<30}{n:>6}{100*n/tot:>7.1f}%{flag}")
    unk = sec.get("unknown", 0)
    if unk:
        print(f"\n  {unk} of {tot} selected names have no sector label. At this")
        print("  size the hand-curated bucket/tier metadata no longer covers the")
        print("  universe -- fine for a systematic model, but the discretionary")
        print("  side of the fund uses those labels.")

    print(f"\n  Sector balance is NOT imposed. The cap is a guard: anything")
    print(f"  above {args.sector_cap:.0%} is flagged for a decision, not corrected.")

    if sel:
        advs = [a for _, a, _, _ in sel]
        print(f"\n  selected ADV: median ${st.median(advs)/1e6:.1f}M, "
              f"min ${min(advs)/1e6:.1f}M, max ${max(advs)/1e9:.1f}B")

    if args.write:
        out = "tickers_expanded.txt"
        with open(out, "w") as f:
            for t, _, _, _ in sorted(sel):
                f.write(t + "\n")
        print(f"\n  wrote {out} ({len(sel)} names). tickers.txt UNTOUCHED --")
        print("  adopting this is a separate, deliberate step.")
    else:
        print("\n  DRY RUN. Re-run with --write to emit tickers_expanded.txt.")


if __name__ == "__main__":
    main()
