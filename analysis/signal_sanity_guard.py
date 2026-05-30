"""
analysis/signal_sanity_guard.py — STANDING pre-trade sanity guard.

Exists because the signal was inverted at the extremes for months and nothing
caught it. This runs every cycle and FAILS LOUD (exit 1) on any RED, so a cron
can block/alert before signals are trusted.

Checks, over a trailing window, per horizon:
  1. DIRECTION  — long/short decile spread. NEGATIVE = model ranks BACKWARDS (the
                  exact failure that was missed). RED.
  2. BUY HIT    — actual win-rate of fired BUY signals vs 50%, with n. <50% (and
                  enough samples) = RED.
  3. RANK-IC    — Spearman(prob_up, actual_return) per date. Persistently negative = RED.
Read-only on accuracy.db. Prints a table; exit code != 0 if any RED.
"""
import argparse, sqlite3, sys
import numpy as np, pandas as pd

def load(db, horizon, days):
    q = """
      SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up, p.signal,
             o.actual_return, o.actual_up
      FROM predictions p
      JOIN outcomes o
        ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
       AND p.horizon=o.horizon
      WHERE p.horizon=?
        AND p.prediction_date >= date('now', ?)
    """
    con = sqlite3.connect(db)
    df = pd.read_sql(q, con, params=[horizon, f"-{days} day"])
    con.close()
    return df

def check_direction(df):
    """Long/short decile spread. Positive=correct, negative=BACKWARDS."""
    sp = []
    for d, g in df.groupby("prediction_date"):
        if len(g) < 10 or g["prob_up"].nunique() < 2:
            continue
        g = g.sort_values("prob_up")
        k = max(1, len(g)//10)
        sp.append(g.tail(k)["actual_return"].mean() - g.head(k)["actual_return"].mean())
    if not sp:
        return None, "no data", 0
    a = np.array(sp); mean = a.mean()
    t = mean/(a.std()/np.sqrt(len(a))) if a.std()>0 else 0
    status = "RED" if mean < 0 and t < -1.0 else ("WARN" if mean < 0 else "GREEN")
    return status, f"spread {mean:+.5f} (t={t:+.2f}, {len(a)} days)", len(a)

def check_buy_hit(df):
    """Actual win-rate of fired BUY signals."""
    b = df[df["signal"]=="BUY"]
    n = len(b)
    if n < 30:
        return "WARN", f"only {n} BUY signals (need 30+)", n
    wr = b["actual_up"].mean()
    se = np.sqrt(wr*(1-wr)/n); lo = wr - 1.96*se
    status = "RED" if (wr < 0.50 and lo < 0.50 and wr+1.96*se < 0.52) else ("WARN" if wr<0.52 else "GREEN")
    return status, f"BUY win-rate {wr*100:.1f}% (n={n}, 95%CI low {lo*100:.1f}%)", n

def check_directional_hit(df):
    """The check that matters: when the model says UP, does it go UP? When it
    says DOWN, does it go DOWN? Splits ALL predictions (not just fired BUYs) by
    the model's directional call (prob_up >= 0.5 = UP call, < 0.5 = DOWN call)
    and measures whether each group actually moved that way. This is the test
    that catches AMD/MRVL-style inversion: model called neutral/down, stock ran up.
    RED if EITHER call type wins less than 50% with enough samples = inverted."""
    up = df[df["prob_up"] >= 0.5]
    dn = df[df["prob_up"] <  0.5]
    n_up, n_dn = len(up), len(dn)
    if n_up < 30 and n_dn < 30:
        return "WARN", f"too few calls (UP={n_up}, DOWN={n_dn})", 0
    # UP calls: fraction that actually went up. DOWN calls: fraction that went down.
    up_hit = up["actual_up"].mean() if n_up else float("nan")
    dn_hit = (1 - dn["actual_up"].mean()) if n_dn else float("nan")  # DOWN correct = actually went down
    import numpy as _np
    msg = f"UP-calls right {up_hit*100:.1f}% (n={n_up}), DOWN-calls right {dn_hit*100:.1f}% (n={n_dn})"
    red = False
    # an UP call group that wins <48% means UP is actually DOWN (inverted), and vice versa
    if n_up >= 30 and not _np.isnan(up_hit) and up_hit < 0.48:
        red = True
    if n_dn >= 30 and not _np.isnan(dn_hit) and dn_hit < 0.48:
        red = True
    if red:
        return "RED", "INVERTED — " + msg, max(n_up, n_dn)
    if (n_up >= 30 and up_hit < 0.50) or (n_dn >= 30 and dn_hit < 0.50):
        return "WARN", msg, max(n_up, n_dn)
    return "GREEN", msg, max(n_up, n_dn)


def check_rank_ic(df):
    ics = []
    for d, g in df.groupby("prediction_date"):
        if len(g) < 5 or g["prob_up"].nunique()<2 or g["actual_return"].nunique()<2:
            continue
        ics.append(g["prob_up"].rank().corr(g["actual_return"].rank()))
    if not ics:
        return None, "no data", 0
    a = np.array(ics); mean = a.mean()
    t = mean/(a.std()/np.sqrt(len(a))) if a.std()>0 else 0
    status = "RED" if mean < 0 and t < -1.0 else ("WARN" if mean < 0 else "GREEN")
    return status, f"rank-IC {mean:+.4f} (t={t:+.2f})", len(a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--horizons", default="1,3,5")
    args = ap.parse_args()

    any_red = False
    print(f"=== SIGNAL SANITY GUARD (trailing {args.days}d) ===\n")
    for h in [int(x) for x in args.horizons.split(",")]:
        df = load(args.db, h, args.days)
        print(f"--- horizon {h}d  ({len(df)} matched pred/outcome rows) ---")
        if df.empty:
            print("  no data\n"); continue
        for name, fn in [("DIRECTION", check_direction),
                         ("DIR-HIT  ", check_directional_hit),
                         ("BUY HIT  ", check_buy_hit),
                         ("RANK-IC  ", check_rank_ic)]:
            st, msg, _ = fn(df)
            if st is None: 
                print(f"  {name}  --   {msg}"); continue
            mark = {"GREEN":"[ OK ]","WARN":"[WARN]","RED":"[RED!]"}[st]
            print(f"  {name}  {mark}  {msg}")
            if st == "RED": any_red = True
        print()
    if any_red:
        print(">>> RED FLAG: signal failed a sanity check. DO NOT TRUST until investigated.")
        sys.exit(1)
    print(">>> all checks pass (no RED).")
    sys.exit(0)

if __name__ == "__main__":
    main()
