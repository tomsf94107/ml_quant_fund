#!/usr/bin/env python3
"""
feature_liveness.py — catch a feature dying, across the WHOLE set.

READ-ONLY on every source. Writes one row per feature per run to
accuracy.db :: feature_liveness.

THE GAP THIS CLOSES
    pipeline_audit.log records it exactly:

        built at train time : 121
        logged at predict   :  31
        NOT logged          :  90

    prediction_features carries 31 columns. health_check's constant-feature
    assertion, added 2026-09-14, can only see those 31. The other 90 -- the
    dark-pool pair, the 8-K family, FinBERT, fundamentals, analyst data,
    days_to_earnings, the twelve sector ETF returns -- are built, trained on,
    and invisible. A break in any of them shows up as degraded accuracy months
    later with no way to attribute it.

    vix_ret is the worked example. Constant 0.0 in 100% of stored rows from
    July, 56% of June, and found on 2026-09-14 by hand. It WAS logged. The 90
    unlogged ones have no such backstop at all.

WHY BASELINE-AND-DIFF, NOT THRESHOLDS
    An audit run on 2026-09-18 flagged six dead features. Every one was wrong:

      high_52w_ratio, low_52w_ratio   NaN only because the test built 75 bars
                                      and they need rolling(252)
      short_pct_float                 NaN deliberately -- builder.py:1159, UW
                                      total_float is shares OUTSTANDING
                                      mislabelled, verified against GME
      is_squeeze_setup                correctly always False, downstream of that
      sentiment_score                 live; the 8 sampled tickers had no scored
                                      news
      insider_net_shares              sparse by nature -- 12 nonzero days of 160

    Six false positives from one pass. Any rule that tries to classify a
    feature as healthy or broken from its CURRENT state will reproduce them,
    because "constant" means something different for a macro scalar, a sparse
    event feature, a documented NaN and a genuine break.

    So this stores a fingerprint and alerts on CHANGE. A feature that has
    always been constant stays quiet forever. A feature that HAD variance and
    loses it is the signal -- and that is precisely the vix_ret signature.

THE FOUR NUMBERS PER FEATURE
    x_distinct   distinct values ACROSS tickers on the last row. A per-ticker
                 feature collapsing to 1 is broken; a macro scalar at 1 is
                 correct, and the baseline knows which is which without being
                 told.
    t_distinct   distinct values ACROSS TIME for one ticker. Catches a feature
                 frozen at its last good value -- which cross-sectional
                 variance alone will not see.
    nan_rate     across tickers.
    spread       (max - min) / |mean| across tickers, so a feature that varies
                 but by nothing is visible.

USAGE
    python analysis/feature_liveness.py --baseline    # first run, or after a
                                                      # deliberate change
    python analysis/feature_liveness.py               # compare and alert
"""
import argparse
import json
import sqlite3
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "accuracy.db"

# Deliberately mixed: mega and small, six sectors, one ETF, one recent listing.
# A feature that varies across THESE is alive; one that does not is either a
# date-level scalar or broken, and the baseline distinguishes them.
SAMPLE = ["AAPL", "NVDA", "JPM", "XOM", "PFE", "WMT", "BA", "MU", "SLV", "PLTR"]

DDL = """
CREATE TABLE IF NOT EXISTS feature_liveness (
    run_date    TEXT NOT NULL,
    feature     TEXT NOT NULL,
    x_distinct  INTEGER,   -- across tickers, last row
    t_distinct  INTEGER,   -- across time, first ticker
    nan_rate    REAL,
    spread      REAL,
    is_baseline INTEGER DEFAULT 0,
    PRIMARY KEY (run_date, feature)
);
"""


def fingerprint(start="2024-01-01"):
    """Build the full feature frame for the sample and measure each column.

    start defaults to two years: rolling(252) windows need a year of history
    before high_52w_ratio and friends stop being NaN, and a shorter window
    manufactures exactly the false positives this script exists to avoid.
    """
    import pandas as pd
    from features.builder import build_feature_dataframe
    from models.classifier import FEATURE_COLUMNS

    last, series = {}, None
    for t in SAMPLE:
        try:
            d = build_feature_dataframe(t, start_date=start, training_mode=True)
            if d is None or d.empty:
                print(f"  [warn] {t}: empty frame")
                continue
            last[t] = d.iloc[-1]
            if series is None:
                series = d.tail(60)          # time-variance probe, one ticker
        except Exception as e:
            print(f"  [warn] {t}: {type(e).__name__}: {str(e)[:60]}")
    if not last:
        raise SystemExit("no ticker built -- cannot fingerprint")

    df = pd.DataFrame(last).T
    out = {}
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            out[c] = dict(x_distinct=-1, t_distinct=-1, nan_rate=1.0, spread=0.0)
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        tv = (pd.to_numeric(series[c], errors="coerce")
              if series is not None and c in series else None)
        mean = float(v.mean()) if v.notna().any() else 0.0
        spread = (float(v.max() - v.min()) / abs(mean)) if mean else 0.0
        out[c] = dict(
            x_distinct=int(v.nunique(dropna=True)),
            t_distinct=int(tv.nunique(dropna=True)) if tv is not None else -1,
            nan_rate=float(v.isna().mean()),
            spread=round(spread, 6),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="store this run as the reference. Use after a "
                         "deliberate feature change, never to silence an alert")
    ap.add_argument("--start", default="2024-01-01")
    a = ap.parse_args()

    con = sqlite3.connect(DB, timeout=30)
    con.executescript(DDL)

    print(f"building {len(SAMPLE)} tickers from {a.start} ...")
    fp = fingerprint(a.start)
    # TIMESTAMP, not date. The primary key is (run_date, feature), so two
    # runs on the same day with INSERT OR REPLACE overwrite each other -- and
    # a comparison run immediately after a baseline run silently DESTROYED the
    # baseline it was meant to compare against. Found on the first use.
    today = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    con.executemany(
        "INSERT OR REPLACE INTO feature_liveness "
        "(run_date, feature, x_distinct, t_distinct, nan_rate, spread, is_baseline) "
        "VALUES (?,?,?,?,?,?,?)",
        [(today, k, v["x_distinct"], v["t_distinct"], v["nan_rate"],
          v["spread"], 1 if a.baseline else 0) for k, v in fp.items()])
    con.commit()
    print(f"{len(fp)} features fingerprinted")

    base = {r[0]: r[1:] for r in con.execute(
        "SELECT feature, x_distinct, t_distinct, nan_rate, spread "
        "FROM feature_liveness WHERE is_baseline=1 AND run_date=("
        "  SELECT MAX(run_date) FROM feature_liveness WHERE is_baseline=1)")}

    if a.baseline:
        print("stored as BASELINE. Future runs compare against it.")
        con.close()
        return
    if not base:
        print("no baseline stored -- run once with --baseline first")
        con.close()
        return

    # ALERT ON LOSS OF VARIANCE, not on its absence. A feature that was always
    # constant stays quiet; one that HAD variance and lost it is the signal.
    alerts = []
    for c, v in fp.items():
        if c not in base:
            alerts.append((c, "NEW", "not in baseline"))
            continue
        bx, bt, bn, bs = base[c]
        if v["x_distinct"] == -1 and bx != -1:
            alerts.append((c, "ABSENT", f"was in the frame, now missing"))
        elif bx > 1 and v["x_distinct"] <= 1:
            alerts.append((c, "FLAT", f"across tickers {bx} -> {v['x_distinct']}"))
        elif bt > 1 and 0 <= v["t_distinct"] <= 1:
            alerts.append((c, "FROZEN", f"across 60 days {bt} -> {v['t_distinct']}"))
        elif v["nan_rate"] > 0.5 and bn <= 0.5:
            alerts.append((c, "NULL", f"nan {bn:.0%} -> {v['nan_rate']:.0%}"))
        elif bs > 0.01 and v["spread"] < bs * 0.1:
            alerts.append((c, "NARROW", f"spread {bs:.3f} -> {v['spread']:.3f}"))

    print()
    if not alerts:
        print(f"OK — no feature lost variance against the baseline "
              f"({len(base)} compared)")
    else:
        print(f"{len(alerts)} FEATURE(S) CHANGED")
        for c, kind, why in alerts:
            print(f"  {kind:8} {c:30} {why}")
        print()
        print("  FLAT   = lost cross-ticker variance. For a per-ticker feature")
        print("           this is a break; for a date-level scalar it is normal")
        print("           and would not fire, because the baseline already")
        print("           recorded it as constant.")
        print("  FROZEN = lost variance over 60 days. This is the vix_ret")
        print("           signature: the value still exists, it stopped moving.")
        print()
        print("  Do NOT re-baseline to clear an alert. The baseline is a")
        print("  reference for what working looked like, not a mute button.")
    con.close()
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
