"""
analysis/signal_sanity_guard.py — STANDING pre-trade sanity guard.

Exists because the signal was inverted at the extremes for months and nothing
caught it. This runs every cycle and FAILS LOUD (exit 1) on any RED, so a cron
can block/alert before signals are trusted.

Checks, over a trailing window, per horizon:
  1. DIRECTION  — long/short decile spread. NEGATIVE = model ranks BACKWARDS.
                  INFORMATIONAL ONLY for a long-only book (we never trade the
                  short leg), so this never RED-gates; it WARNs at most.
  2. DIR-HIT    — per-call directional hit. The DOWN/short leg is structurally
                  weak (documented since May 7), and we don't trade it, so a weak
                  DOWN leg WARNs, never REDs. A broken UP (long) leg still REDs.
  3. BUY HIT    — actual win-rate of fired BUY signals vs 50%. RED-gating.
  4. RANK-IC    — Spearman(prob_up, actual_return) per date. RED-gating.
Read-only on accuracy.db. Prints a table; exit code != 0 if any RED.
"""
import argparse, sqlite3, sys
import numpy as np, pandas as pd

# Horizons whose models are not yet through the fitness gate. They print
# their metrics but cannot emit RED (informational until fit). Edit as the
# fitness re-run completes each horizon.
UNFIT_HORIZONS = {5}

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
    """Long/short decile spread. Positive=correct, negative=BACKWARDS.
    INFORMATIONAL for a long-only book: a negative spread reflects the weak
    short leg we never trade, so this caps at WARN and never REDs."""
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
    status = "WARN" if mean < 0 else "GREEN"
    return status, f"spread {mean:+.5f} (t={t:+.2f}, {len(a)} days) [info: short leg untraded]", len(a)

def check_buy_hit(df):
    """Actual win-rate of fired BUY signals. RED-gating."""
    b = df[df["signal"]=="BUY"]
    n = len(b)
    if n < 30:
        return "WARN", f"only {n} BUY signals (need 30+)", n
    wr = b["actual_up"].mean()
    se = np.sqrt(wr*(1-wr)/n); lo = wr - 1.96*se
    status = "RED" if (wr < 0.50 and lo < 0.50 and wr+1.96*se < 0.52) else ("WARN" if wr<0.52 else "GREEN")
    return status, f"BUY win-rate {wr*100:.1f}% (n={n}, 95%CI low {lo*100:.1f}%)", n

def check_directional_hit(df):
    """Per-call directional hit. The UP (long) leg is the one we trade: if it
    inverts, that is RED. The DOWN (short) leg is structurally weak and untraded,
    so a weak/inverted DOWN leg WARNs but never REDs."""
    up = df[df["prob_up"] >= 0.5]
    dn = df[df["prob_up"] <  0.5]
    n_up, n_dn = len(up), len(dn)
    if n_up < 30 and n_dn < 30:
        return "WARN", f"too few calls (UP={n_up}, DOWN={n_dn})", 0
    up_hit = up["actual_up"].mean() if n_up else float("nan")
    dn_hit = (1 - dn["actual_up"].mean()) if n_dn else float("nan")
    import numpy as _np
    msg = f"UP-calls right {up_hit*100:.1f}% (n={n_up}), DOWN-calls right {dn_hit*100:.1f}% (n={n_dn})"
    if n_up >= 30 and not _np.isnan(up_hit) and up_hit < 0.48:
        return "RED", "LONG-LEG INVERTED — " + msg, max(n_up, n_dn)
    if (n_up >= 30 and up_hit < 0.50) or (n_dn >= 30 and not _np.isnan(dn_hit) and dn_hit < 0.50):
        suffix = "" if (n_up >= 30 and up_hit < 0.50) else " [info: short leg untraded]"
        return "WARN", msg + suffix, max(n_up, n_dn)
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
        unfit = h in UNFIT_HORIZONS
        tag = "  [UNFIT - informational, cannot RED]" if unfit else ""
        print(f"--- horizon {h}d  ({len(df)} matched pred/outcome rows){tag} ---")
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
            if st == "RED" and unfit:
                print(f"  {name}  [RED*]  {msg}  (suppressed: horizon unfit)")
            else:
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
