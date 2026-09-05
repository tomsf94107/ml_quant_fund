#!/usr/bin/env python3
"""
h40_shadow.py — freeze the h=40 model and log its picks daily. Never trades.

WRITES to accuracy.db, table h40_shadow_predictions. Creates nothing else.

WHY FROZEN
    The h=40 cross-sectional book measures +1.70pp excess per 40-day period over
    2021-2025, positive in four of five clean years, independent of the SI
    brick. Every one of those numbers is a BACKTEST -- a question asked of the
    past by someone who already knew the answer.

    PCT7 is the counter-example that matters. It was trained 2026-05-25, wired
    into shadow mode, and then nobody scored it for fourteen weeks. That neglect
    produced the strongest evidence in this system: 24,741 predictions on a
    model that was never retrained, genuinely out-of-sample throughout, and
    impossible for any researcher to have tuned after the fact.

    So this trains ONCE and never retrains. The model ages, and that is the
    point: if it still works in six months untouched, the claim is far stronger
    than a continuously-refit book could support.

    A rolling version would be more realistic as an operational design, but every
    refit is a decision point where something could be adjusted, and this exists
    precisely to remove those.

THE CLOCK IS LONG BY CONSTRUCTION
    At a 40-day horizon, three months of logging gives roughly THREE independent
    outcomes. Six months gives six. Meaningful evidence needs closer to a year.

    That is the reason to start now rather than after the next piece of analysis:
    every day not logging is a day of evidence not accruing, and no amount of
    later effort recovers it.

WHAT IS LOGGED
    Per ticker per run: the frozen model's probability, its rank that day, the
    universe size, the price at entry, and the date 40 trading days ahead. The
    forward return is filled in later by a scorer -- this script never computes
    an outcome, so it cannot be accused of choosing one.

    Also logged: model_sha, so a retrain that changes the artifact is visible
    rather than silent. If that value ever changes mid-record, the frozen claim
    is void for everything after it.

WHAT IT DOES NOT DO
    No trading. No signal emission. No interaction with signals/generator.py or
    the STEP 1 kill switch. It writes to its own table and nothing reads it.

    python analysis/h40_shadow.py --train      # once, builds the frozen model
    python analysis/h40_shadow.py              # daily, logs picks
    python analysis/h40_shadow.py --status     # what has accumulated
"""
import argparse
import hashlib
import os
import sqlite3
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")

MODEL_PATH = "models/saved/H40_SHADOW_frozen.joblib"
DDL = """
CREATE TABLE IF NOT EXISTS h40_shadow_predictions (
    run_date      TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    prob          REAL NOT NULL,
    rank_today    INTEGER NOT NULL,
    universe_n    INTEGER NOT NULL,
    entry_close   REAL,
    model_sha     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (run_date, ticker)
)
"""


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--top", type=int, default=10,
                    help="log this many ranks; cap-3 is the studied book but "
                         "logging deeper costs nothing and allows the cap to be "
                         "chosen later without re-running history")
    args = ap.parse_args()
    H = args.horizon
    sys.path.insert(0, ".")

    con = sqlite3.connect(args.db, timeout=60)
    con.execute(DDL)
    con.commit()

    if args.status:
        r = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT run_date), MIN(run_date), "
            "MAX(run_date), COUNT(DISTINCT model_sha) "
            "FROM h40_shadow_predictions").fetchone()
        print(f"h40_shadow_predictions: {r[0]:,} rows, {r[1]} run dates, "
              f"{r[2]} .. {r[3]}")
        print(f"  distinct model_sha: {r[4]}"
              + ("   <-- MORE THAN ONE: the frozen claim is void after the "
                 "change" if r[4] and r[4] > 1 else "   (frozen, as intended)"))
        if r[1]:
            days = 0
            try:
                days = (date.fromisoformat(r[3]) - date.fromisoformat(r[2])).days
            except Exception:
                pass
            print(f"  elapsed: {days} calendar days ~ "
                  f"{days/56.0:.1f} independent {H}-day outcomes")
            print(f"  meaningful evidence needs roughly a year")
        con.close()
        return

    universe = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    from features.builder import build_feature_dataframe

    if args.train:
        if os.path.exists(MODEL_PATH):
            raise SystemExit(
                f"{MODEL_PATH} already exists. Refusing to overwrite -- the "
                f"whole point is that it is frozen. Delete it deliberately if "
                f"you mean to restart the clock, and note that doing so voids "
                f"every observation logged so far.")
        from xgboost import XGBClassifier
        import joblib
        X, y, cols = [], [], None
        built = 0
        print(f"training the frozen model on {len(universe)} tickers "
              f"from {args.start}, h={H}")
        for i, t in enumerate(universe, 1):
            try:
                df = build_feature_dataframe(t, start_date=args.start,
                                             training_mode=True)
                if df is None or len(df) < 300 or "close" not in df.columns:
                    continue
                num = df.select_dtypes("number")
                num = num.drop(columns=[c for c in num.columns
                                        if c.startswith("target_")],
                               errors="ignore")
                if cols is None:
                    cols = list(num.columns)
                cl = list(df["close"])
                for j in range(20, len(cl) - H):
                    a, b = cl[j], cl[j + H]
                    if not a or not b:
                        continue
                    r = (b - a) / a
                    if abs(r) > 1.5:
                        continue
                    X.append([float(v) if v == v else float("nan")
                              for v in num.iloc[j].tolist()])
                    y.append(1 if r > 0 else 0)
                built += 1
                if i % 50 == 0:
                    print(f"  ...{i} tickers, {len(X):,} rows")
            except Exception:
                continue
        if len(X) < 20000:
            raise SystemExit(f"only {len(X)} training rows -- too few")
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
        m.fit(X, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"model": m, "cols": cols, "horizon": H,
                     "trained_on": str(date.today()),
                     "n_rows": len(X), "n_tickers": built}, MODEL_PATH)
        print(f"\n  saved {MODEL_PATH}")
        print(f"  {len(X):,} rows, {built} tickers, {len(cols)} features")
        print(f"  sha {sha_of(MODEL_PATH)}")
        print("\n  This model must NEVER be retrained. Add the daily run to")
        print("  cron and leave it alone. Score no earlier than six months,")
        print("  and prefer a year.")
        con.close()
        return

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found -- run --train once first")
    import joblib
    art = joblib.load(MODEL_PATH)
    m, cols = art["model"], art["cols"]
    sha = sha_of(MODEL_PATH)
    run_date = str(date.today())

    rows = []
    for t in universe:
        try:
            # Scoring uses training_mode=True on purpose: the frozen model was
            # trained that way and never saw non-zero expected_move_perc,
            # pre/post_earnings_drift or is_earnings_week, which builder.py
            # zeroes in that path for PIT honesty. Live values would be a
            # train/serve mismatch, and skipping the UW calls is much faster.
            df = build_feature_dataframe(t, start_date="2024-01-01",
                                         training_mode=True)
            if df is None or df.empty:
                continue
            num = df.select_dtypes("number")
            vec = []
            for c in cols:
                v = num[c].iloc[-1] if c in num.columns else float("nan")
                vec.append(float(v) if v == v else float("nan"))
            p = float(m.predict_proba([vec])[0][1])
            px = float(df["close"].iloc[-1]) if "close" in df.columns else None
            rows.append((t, p, px))
        except Exception:
            continue

    if not rows:
        print("no tickers scored -- nothing logged")
        con.close()
        return
    rows.sort(key=lambda x: -x[1])
    n = len(rows)
    now = str(date.today())
    con.executemany(
        "INSERT OR REPLACE INTO h40_shadow_predictions "
        "(run_date, ticker, prob, rank_today, universe_n, entry_close, "
        "model_sha, created_at) VALUES (?,?,?,?,?,?,?,?)",
        [(run_date, t, p, i + 1, n, px, sha, now)
         for i, (t, p, px) in enumerate(rows[:args.top])])
    con.commit()
    tot = con.execute("SELECT COUNT(*) FROM h40_shadow_predictions").fetchone()[0]
    con.close()
    print(f"{run_date}: scored {n} tickers, logged top {min(args.top, n)}")
    print(f"  top 3: " + ", ".join(f"{t} {p:.3f}" for t, p, _ in rows[:3]))
    print(f"  table now holds {tot:,} rows, model sha {sha}")


if __name__ == "__main__":
    main()
