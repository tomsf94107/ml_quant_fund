#!/usr/bin/env python3
"""
long_short_book.py — does hedging out market direction leave a tradeable edge?

READ-ONLY. Reads logged predictions, outcomes and borrow fees. Writes nothing.

WHY THIS EXISTS
    The per-ticker direction model cannot call the market. Measured at the
    production gate against the rest of the same day's universe, its SELECTION
    beats the field by +8 to +9pp consistently -- including September 2026, when
    the selection hit 37.9% and the field hit 28.6%. The ranking works. The
    direction does not, and nothing in the model targets direction: it is
    trained on `1 if fwd > 0 else 0` per name, which learns which stocks beat
    which stocks, not whether stocks go up.

    Long the top and short the bottom cancels the market move and leaves the
    ranking. That is what a market-neutral book is for, and it is why long/short
    equity exists as a category rather than as a refinement.

    Monthly gross spreads, top decile minus bottom, h=5:
        Mar +3.29%  Apr -1.73%  May +0.46%  Jun +2.09%
        Jul +2.13%  Aug +0.61%  Sep +1.19%
    Positive in six of seven, and POSITIVE IN SEPTEMBER, when long-only was
    flat. This script turns those monthly averages into something with a cost
    model and a t-statistic.

BORROW IS NOT THE OBSTACLE, AND THAT IS NOT OBVIOUS
    si_positions_live.py's docstring warns at length that its own short leg
    shorts high-days-to-cover names -- exactly the hardest and most expensive to
    borrow -- and that backtest costs of 10-40bps did not include borrow fees,
    which would erode it in live trading. That warning is correct FOR THAT BOOK
    and does not transfer here, because low prob_up and high days-to-cover
    select different names.

    Measured PIT on the names this book would actually short:
        Jun 172.9bps   Jul 164.6bps   Aug 134.9bps   Sep 127.0bps
    against a universe average of 196bps. The bottom decile is CHEAPER to
    borrow than the average stock. Over a five-day hold, 127bps annualised is
    about 2.5bps -- roughly 2% of September's 119bps spread.

    The tail is the part that bites: individual fees reach 19,531bps (195%) in
    June and 5,524bps in September. Those are cheap to exclude and ruinous to
    hold, so this caps the fee rather than only filtering is_htb.

WHAT DECIDES IT
    Execution cost, which is in no database here. At 10bps/leg the book nets
    roughly +76bps per five days; at 40bps/leg it is negative. The ladder is
    printed rather than a single number chosen, because the answer depends on
    the broker, the order type and the position size -- none of which this
    script can know.

POINT-IN-TIME BORROW
    Fees are joined on the most recent settlement AT OR BEFORE each prediction
    date, never the latest available. FINRA settles semi-monthly and publishes
    about eight business days later, so using a settlement dated after the
    prediction would be look-ahead: it would price a short with information that
    did not exist when it was opened.

USAGE
    python analysis/long_short_book.py
    python analysis/long_short_book.py --long-gate 0.70 --short-gate 0.40
"""
import argparse
import math
import sqlite3
from collections import defaultdict

import numpy as np


def nw_t(x, lag):
    """Newey-West t on the mean of x, `lag` overlapping periods."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if s <= 0:
        return float("nan")
    return x.mean() / math.sqrt(s / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--borrow-db", default="borrow.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--long-gate", type=float, default=0.70)
    ap.add_argument("--short-gate", type=float, default=0.40)
    ap.add_argument("--max-fee-bps", type=float, default=1000.0,
                    help="skip shorts above this annualised borrow fee. 1000 = "
                         "10%%/yr. Fees reach 19,531bps in this data and a name "
                         "at 195%% is not a short, it is a squeeze.")
    ap.add_argument("--min-names", type=int, default=5,
                    help="per side, per date")
    ap.add_argument("--start", default="2026-04-01",
                    help="default skips March, which has 1 usable date")
    a = ap.parse_args()
    H = a.horizon

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    try:
        con.execute(f"ATTACH ? AS b", (a.borrow_db,))
        have_borrow = True
    except sqlite3.Error as e:
        print(f"[warn] borrow.db not attached ({e}); shorts costed at 0 fee")
        have_borrow = False

    # PIT borrow: the most recent settlement AT OR BEFORE the prediction date.
    fee_sql = """
        , (SELECT f.borrow_fee_bps FROM b.borrow_fees f
            WHERE f.ticker = p.ticker AND f.asof_date <= p.prediction_date
            ORDER BY f.asof_date DESC LIMIT 1) AS fee
        , (SELECT f.is_htb FROM b.borrow_fees f
            WHERE f.ticker = p.ticker AND f.asof_date <= p.prediction_date
            ORDER BY f.asof_date DESC LIMIT 1) AS htb
    """ if have_borrow else ", NULL AS fee, NULL AS htb"

    rows = list(con.execute(f"""
        SELECT p.prediction_date, p.ticker, p.prob_up, o.actual_return {fee_sql}
        FROM predictions p
        JOIN outcomes o ON o.ticker = p.ticker
                       AND o.prediction_date = p.prediction_date
                       AND o.horizon = p.horizon
        WHERE p.horizon = ? AND p.prob_up IS NOT NULL
          AND o.actual_return IS NOT NULL
          AND p.prediction_date >= ?
    """, (H, a.start)))
    con.close()

    if not rows:
        print("no rows")
        return

    byd = defaultdict(lambda: {"L": [], "S": [], "skip_htb": 0, "skip_fee": 0})
    for d, tk, prob, ret, fee, htb in rows:
        prob = float(prob)
        if prob >= a.long_gate:
            byd[d]["L"].append(float(ret))
        elif prob < a.short_gate:
            if htb:
                byd[d]["skip_htb"] += 1
                continue
            if fee is not None and float(fee) > a.max_fee_bps:
                byd[d]["skip_fee"] += 1
                continue
            # Borrow charged over the holding period, annualised fee / 252 * H.
            cost = (float(fee) / 1e4) * (H / 252.0) if fee is not None else 0.0
            byd[d]["S"].append(-float(ret) - cost)

    daily, n_l, n_s, htb_tot, fee_tot = {}, [], [], 0, 0
    for d, v in byd.items():
        htb_tot += v["skip_htb"]
        fee_tot += v["skip_fee"]
        if len(v["L"]) < a.min_names or len(v["S"]) < a.min_names:
            continue
        # Equal dollars each side: the book is the average of the two legs, so a
        # market move that lifts both cancels rather than adding.
        daily[d] = 0.5 * (sum(v["L"]) / len(v["L"]) + sum(v["S"]) / len(v["S"]))
        n_l.append(len(v["L"]))
        n_s.append(len(v["S"]))

    if len(daily) < 10:
        print(f"only {len(daily)} usable dates -- need both sides to carry "
              f">= {a.min_names} names")
        return

    dates = sorted(daily)
    v = np.array([daily[d] for d in dates])

    print(f"\nLONG/SHORT BOOK — h={H}, long >= {a.long_gate:.2f}, "
          f"short < {a.short_gate:.2f}")
    print(f"{len(dates)} dates {dates[0]} .. {dates[-1]}, "
          f"avg {np.mean(n_l):.0f} long / {np.mean(n_s):.0f} short per date")
    print(f"borrow charged PIT at the settlement in force; {htb_tot} "
          f"hard-to-borrow and {fee_tot} above {a.max_fee_bps:.0f}bps excluded\n")

    print(f"  gross of execution, net of borrow: {v.mean():+.3%} per {H}-day "
          f"period   NW t {nw_t(v, H):+.2f}   {float((v>0).mean()):.0%} of dates > 0")

    print("\nBY MONTH")
    bym = defaultdict(list)
    for d in dates:
        bym[d[:7]].append(daily[d])
    for m in sorted(bym):
        arr = np.array(bym[m])
        print(f"  {m}   {len(arr):>3} dates   {arr.mean():+.3%}   "
              f"{float((arr>0).mean()):.0%} > 0")

    print(f"\nEXECUTION COST LADDER (both legs, one round trip per rebalance)")
    print(f"  {'bps/leg':>9}{'net':>12}{'NW t':>9}")
    for bps in (0, 5, 10, 20, 40, 60, 100):
        net = v - 4.0 * (bps / 1e4)   # 2 legs x (entry + exit)
        print(f"  {bps:>9}{net.mean():>+12.3%}{nw_t(net, H):>+9.2f}")

    print(f"\n  Four crossings per rebalance: buy and sell the long leg, sell")
    print(f"  short and cover the short leg. A ladder rather than one number,")
    print(f"  because execution cost is not in any database here -- it depends")
    print(f"  on the broker, the order type and the position size.")
    print(f"\n  The bar for deployment in this fund is NW t > 3.0, and it applies")
    print(f"  to the NET row at whatever cost you actually pay, not to the gross.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
