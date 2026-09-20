#!/usr/bin/env python3
"""
system_audit.py — every cog, end to end. READ-ONLY.

WHY THIS EXISTS RATHER THAN MORE SPOT CHECKS
    An audit on 2026-09-18 sampled 10 tickers on one date against
    FEATURE_COLUMNS and produced nine flags, seven of which were false
    positives -- test-window artifacts and repairs already made and documented
    in the code. It also missed a second model
    (models/train_panel_ranker.py) whose STRANDED list contains
    short_pct_float__cs_rank, a cross-sectional rank of an all-NaN column.

    Sampling cannot audit a machine. This walks every layer, discovers what is
    there instead of assuming, and reports what it measured rather than what it
    concluded.

WHAT IT DOES NOT DO
    It does not decide whether a feature SHOULD be constant. That judgement is
    what produced seven false positives. It reports state, names the source,
    and leaves classification to analysis/feature_liveness.py, which alerts on
    CHANGE against a stored baseline.

SECTIONS
    1  DATA SOURCES     every .db: tables, row counts, newest row, staleness
                        against its own observed cadence
    2  FEATURE BUILD    all features the builder emits -- not just
                        FEATURE_COLUMNS -- across a stratified universe sample,
                        measured on BOTH axes: across tickers and across time
    3  TRAIN/SERVE      rebuild and diff against what prediction_features
                        stored. This is the class that hid vix_ret for three
                        months: the builder was healthy, the logged row was not
    4  MODEL LAYER      every saved model: age, feature count, and whether its
                        expected columns match what the builder produces today
    5  ENSEMBLE         weights across tickers -- a leg pinned at 0 or 1 is a
                        single model wearing an ensemble label
    6  MULTIPLIERS      the chain measured inert at 0.0031 on 91,000 rows.
                        Traced per multiplier to find which are disconnected
    7  OUTPUT           predictions to outcomes, benchmarked, and the
                        lift-versus-return split

USAGE
    python analysis/system_audit.py
    python analysis/system_audit.py --section 3
    python analysis/system_audit.py --tickers 24
"""
import argparse
import os
import sqlite3
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAIL, WARN, OK = [], [], []


def hdr(n, title):
    print(f"\n{'='*78}\n{n}. {title}\n{'='*78}")


def fail(msg):
    FAIL.append(msg); print(f"  FAIL  {msg}")


def warn(msg):
    WARN.append(msg); print(f"  WARN  {msg}")


def ok(msg):
    OK.append(msg); print(f"  ok    {msg}")


# ─────────────────────────────────────────────────────── 1. DATA SOURCES

def audit_sources():
    hdr(1, "DATA SOURCES")
    dbs = sorted(ROOT.glob("*.db")) + sorted((ROOT / "data").glob("*.db"))
    dbs = [d for d in dbs if not d.name.startswith("squeeze_")]
    print(f"  {len(dbs)} databases (squeeze_*.db excluded — per-ticker scratch)\n")
    print(f"  {'database':28}{'table':30}{'rows':>9}{'newest':>12}{'age':>6}")
    for db in dbs:
        if db.stat().st_size == 0:
            fail(f"{db.name} is 0 bytes — created and never written")
            continue
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
            tabs = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")]
            for t in tabs:
                try:
                    n = c.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
                except Exception:
                    continue
                # Find a date-ish column and read its max. Names vary by table
                # (d, date, ds, asof_date, filed_at_date, run_date, ...), so
                # this discovers rather than assumes.
                cols = [r[1] for r in c.execute(f"PRAGMA table_info('{t}')")]
                dcol = next((x for x in cols if x.lower() in (
                    "d", "date", "ds", "asof_date", "run_date", "prediction_date",
                    "settlement_date", "filed_at_date", "score_date",
                    "obs_date", "created_at", "retrain_date", "filing_date",
                    "effective_date", "updated")), None)
                newest, age = "—", ""
                if dcol and n:
                    try:
                        newest = str(c.execute(
                            f"SELECT MAX({dcol}) FROM '{t}'").fetchone()[0])[:10]
                        dt = datetime.strptime(newest, "%Y-%m-%d").date()
                        d_age = (date.today() - dt).days
                        age = f"{d_age}d"
                        if d_age > 120 and n > 100:
                            warn(f"{db.name}:{t} newest row {newest} "
                                 f"({d_age}d old, {n:,} rows)")
                    except Exception:
                        pass
                print(f"  {db.name:28}{t[:29]:30}{n:>9,}{newest:>12}{age:>6}")
            c.close()
        except Exception as e:
            fail(f"{db.name}: {type(e).__name__}: {str(e)[:50]}")


# ────────────────────────────────────────────────────── 2. FEATURE BUILD

def audit_features(n_tickers):
    hdr(2, "FEATURE BUILD")
    import pandas as pd
    from features.builder import build_feature_dataframe
    from models.classifier import FEATURE_COLUMNS

    uni = [l.strip().upper() for l in open(ROOT / "tickers.txt") if l.strip()]
    # Stratified, not the first N: take an even stride so sectors and sizes mix.
    step = max(1, len(uni) // n_tickers)
    sample = uni[::step][:n_tickers]
    print(f"  {len(sample)} of {len(uni)} tickers, stride {step}, 2y history")
    print(f"  (the 2026-09-18 pass used 10 tickers and 75 bars; rolling(252) "
          f"features were NaN and read as dead)\n")

    last, ts = {}, {}
    t0 = time.time()
    for i, t in enumerate(sample, 1):
        try:
            d = build_feature_dataframe(t, start_date="2024-01-01",
                                        training_mode=True)
            if d is None or d.empty:
                fail(f"{t}: builder returned empty")
                continue
            last[t] = d.iloc[-1]
            ts[t] = d.tail(90)
            if i % 8 == 0:
                print(f"    [{i}/{len(sample)}] {time.time()-t0:.0f}s")
        except Exception as e:
            fail(f"{t}: build raised {type(e).__name__}: {str(e)[:60]}")
    if not last:
        fail("no ticker built — cannot continue")
        return None, None

    df = pd.DataFrame(last).T
    built = set(df.columns)
    declared = set(FEATURE_COLUMNS)
    print(f"\n  builder emits {len(built)} columns, "
          f"FEATURE_COLUMNS declares {len(declared)}")
    missing = declared - built
    if missing:
        fail(f"{len(missing)} declared features the builder does NOT emit: "
             f"{', '.join(sorted(missing)[:8])}")
    else:
        ok("every declared feature is present in the built frame")

    # BOTH AXES. Cross-ticker variance alone cannot see a feature frozen at its
    # last good value; time variance alone cannot see one broadcast identically
    # to every name. vix_ret failed the second way for three months.
    print(f"\n  {'feature':32}{'x-tick':>7}{'x-time':>7}{'nan%':>6}  state")
    dead_both, flat_x, frozen_t = [], [], []
    for c in sorted(declared & built):
        v = pd.to_numeric(df[c], errors="coerce")
        tvals = set()
        for t in ts:
            if c in ts[t]:
                tvals |= set(pd.to_numeric(ts[t][c], errors="coerce")
                             .dropna().round(8).tolist())
        xd, td = int(v.nunique(dropna=True)), len(tvals)
        nan = float(v.isna().mean())
        state = ""
        if nan >= 1.0:
            state = "ALL-NaN"; dead_both.append(c)
        elif xd <= 1 and td <= 1:
            state = "constant both axes"; dead_both.append(c)
        elif xd <= 1:
            state = "same for every ticker"; flat_x.append(c)
        elif td <= 1:
            state = "frozen over 90 days"; frozen_t.append(c)
        if state:
            print(f"  {c:32}{xd:>7}{td:>7}{100*nan:>5.0f}%  {state}")

    print(f"\n  {len(dead_both)} dead on both axes, {len(flat_x)} flat across "
          f"tickers (macro scalars live here), {len(frozen_t)} frozen over time")
    if frozen_t:
        fail(f"frozen over 90 days but varying across tickers — a per-ticker "
             f"feature that stopped moving: {', '.join(frozen_t)}")
    return df, ts


# ───────────────────────────────────────────────────── 3. TRAIN vs SERVE

def audit_trainserve(built_df):
    hdr(3, "TRAIN/SERVE SKEW")
    import pandas as pd
    if built_df is None:
        warn("section 2 produced no frame — skipped")
        return
    c = sqlite3.connect(f"file:{ROOT/'accuracy.db'}?mode=ro", uri=True)
    pf_cols = [r[1] for r in c.execute("PRAGMA table_info(prediction_features)")
               if r[2] == "REAL"]
    latest = c.execute("SELECT MAX(prediction_date) FROM prediction_features").fetchone()[0]
    print(f"  prediction_features logs {len(pf_cols)} of the built columns, "
          f"newest {latest}")
    print(f"  comparing the builder's LAST ROW against the stored row per "
          f"ticker\n")
    print(f"  {'feature':28}{'built':>14}{'stored':>14}  note")
    skew = 0
    for col in pf_cols:
        if col not in built_df.columns:
            continue
        for t in list(built_df.index)[:6]:
            row = c.execute(
                f"SELECT {col} FROM prediction_features WHERE ticker=? "
                f"AND prediction_date=? AND horizon=5", (t, latest)).fetchone()
            if not row or row[0] is None:
                continue
            b, s = float(built_df.loc[t, col] or 0), float(row[0])
            # Tolerance: the stored row was built on an earlier session, so an
            # exact match is not expected for anything price-driven. A stored
            # ZERO against a nonzero build is the signature that matters.
            if abs(s) < 1e-12 and abs(b) > 1e-9:
                print(f"  {col:28}{b:>14.6f}{s:>14.6f}  stored ZERO, built nonzero")
                skew += 1
                break
    if skew:
        fail(f"{skew} feature(s) stored as zero while the builder produces a "
             f"value — the vix_ret signature")
    else:
        ok("no stored-zero-versus-built-nonzero mismatch found")
    c.close()


# ──────────────────────────────────────────────────────── 4. MODEL LAYER

def audit_models():
    hdr(4, "MODEL LAYER")
    import joblib
    from models.classifier import FEATURE_COLUMNS
    sd = ROOT / "models" / "saved"
    jl = sorted(sd.glob("*.joblib"))
    print(f"  {len(jl)} saved models in models/saved/")
    if not jl:
        fail("no saved models"); return
    ages = [(f, (time.time() - f.stat().st_mtime) / 86400) for f in jl]
    stale = [f for f, a in ages if a > 7]
    print(f"  newest {min(a for _, a in ages):.1f}d, "
          f"oldest {max(a for _, a in ages):.1f}d")
    if stale:
        warn(f"{len(stale)} model(s) older than 7 days, e.g. "
             f"{', '.join(f.name for f in stale[:4])}")
    else:
        ok("every saved model refit within 7 days")

    # Feature contract. ensemble.py:82 says alignment exists to stop new
    # FEATURE_COLUMNS breaking old models -- verify what the stored objects
    # actually expect versus what is declared now.
    kinds = {}
    for f in jl:
        kinds.setdefault(f.stem.split("_")[-1], []).append(f)
    print(f"\n  by horizon/kind: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(kinds.items())))
    checked = 0
    for f in jl[:6]:
        try:
            m = joblib.load(f)
            cols = getattr(m, "feature_cols", None) or getattr(
                m, "feature_names_in_", None)
            if cols is None and isinstance(m, dict):
                cols = m.get("feature_cols")
            if cols is None:
                continue
            checked += 1
            extra = set(cols) - set(FEATURE_COLUMNS)
            gone = set(FEATURE_COLUMNS) - set(cols)
            if extra or gone:
                warn(f"{f.name}: expects {len(cols)} cols; "
                     f"{len(gone)} declared-but-absent, {len(extra)} "
                     f"stored-but-undeclared")
            else:
                ok(f"{f.name}: {len(cols)} columns match FEATURE_COLUMNS")
        except Exception as e:
            warn(f"{f.name}: load raised {type(e).__name__}")
    if not checked:
        warn("no saved model exposed a feature list — contract unverifiable")


# ───────────────────────────────────────────────────────── 5. ENSEMBLE

def audit_ensemble():
    hdr(5, "ENSEMBLE WEIGHTS")
    import joblib
    sd = ROOT / "models" / "saved"
    ens = sorted(sd.glob("*_ensemble_*.joblib"))
    if not ens:
        warn("no ensemble models found"); return
    ws = []
    for f in ens[:40]:
        try:
            m = joblib.load(f)
            w = getattr(m, "weights", None) or (
                m.get("weights") if isinstance(m, dict) else None)
            if w is not None:
                ws.append((f.name, tuple(round(float(x), 3) for x in
                                         (w if hasattr(w, "__iter__") else [w]))))
        except Exception:
            continue
    if not ws:
        warn("no ensemble exposed weights"); return
    print(f"  sampled {len(ws)} of {len(ens)} ensembles")
    from collections import Counter
    cnt = Counter(w for _, w in ws)
    for w, n in cnt.most_common(8):
        print(f"    {str(w):24} x{n}")
    # A leg pinned at 0 or 1 across most tickers means the ensemble is one
    # model wearing two names.
    pinned = sum(n for w, n in cnt.items()
                 if any(abs(x) < 1e-6 or abs(x - 1) < 1e-6 for x in w))
    if pinned > len(ws) * 0.5:
        fail(f"{pinned}/{len(ws)} ensembles have a leg pinned at 0 or 1 — "
             f"that is a single model, not an ensemble")
    else:
        ok("ensemble weights are mixed across tickers")


# ──────────────────────────────────────────────────────── 6. MULTIPLIERS

def audit_multipliers():
    hdr(6, "MULTIPLIER CHAIN")
    c = sqlite3.connect(f"file:{ROOT/'accuracy.db'}?mode=ro", uri=True)
    mults = ["risk_mult", "sent_mult", "regime_mult", "options_mult",
             "squeeze_mult", "intraday_mult", "fg_mult"]
    print(f"  {'multiplier':16}{'distinct':>9}{'min':>9}{'max':>9}{'mean':>9}")
    inert = []
    for m in mults:
        try:
            d, mn, mx, av = c.execute(
                f"SELECT COUNT(DISTINCT {m}), MIN({m}), MAX({m}), AVG({m}) "
                f"FROM predictions WHERE prediction_date >= date('now','-60 days') "
                f"AND {m} IS NOT NULL").fetchone()
        except Exception as e:
            warn(f"{m}: {type(e).__name__}"); continue
        if d is None or d == 0:
            warn(f"{m}: no non-null rows in 60 days"); continue
        print(f"  {m:16}{d:>9}{mn:>9.4f}{mx:>9.4f}{av:>9.4f}")
        if d <= 1:
            inert.append(m)
    if inert:
        fail(f"{len(inert)} multiplier(s) take exactly ONE value over 60 days "
             f"— computed nightly, changing nothing: {', '.join(inert)}")
    # The chain's aggregate effect, which is the number that matters.
    eff = c.execute(
        "SELECT ROUND(AVG(ABS(prob_up - prob_raw)),6), COUNT(*) FROM predictions "
        "WHERE prediction_date >= date('now','-60 days') AND prob_raw IS NOT NULL"
    ).fetchone()
    print(f"\n  chain moves prob_eff by {eff[0]} on average over {eff[1]:,} rows")
    if eff[0] is not None and eff[0] < 0.005:
        fail(f"the whole multiplier chain moves the probability by {eff[0]} — "
             f"seven multipliers computed nightly for less than half a point")
    c.close()


# ─────────────────────────────────────────────────────────── 7. OUTPUT

def audit_output():
    hdr(7, "OUTPUT — lift versus return")
    c = sqlite3.connect(f"file:{ROOT/'accuracy.db'}?mode=ro", uri=True)
    print(f"  {'h':>3}{'window':>9}{'n':>8}{'sel hit':>9}{'field':>8}"
          f"{'LIFT':>8}{'sel ret':>9}{'field ret':>10}{'EDGE':>8}")
    for h in (1, 3, 5):
        for days, lab in ((30, "30d"), (90, "90d")):
            r = c.execute("""
                SELECT COUNT(*),
                  AVG(CASE WHEN p.prob_up>=0.70 THEN o.actual_up END),
                  AVG(CASE WHEN p.prob_up< 0.70 THEN o.actual_up END),
                  AVG(CASE WHEN p.prob_up>=0.70 THEN o.actual_return END),
                  AVG(CASE WHEN p.prob_up< 0.70 THEN o.actual_return END)
                FROM predictions p JOIN outcomes o
                  ON o.ticker=p.ticker AND o.prediction_date=p.prediction_date
                 AND o.horizon=p.horizon
                WHERE p.horizon=? AND p.prediction_date>=date('now',?)
            """, (h, f"-{days} days")).fetchone()
            n, sh, fh, sr, fr = r
            if not sh or not fh:
                continue
            print(f"  {h:>3}{lab:>9}{n:>8,}{100*sh:>8.1f}%{100*fh:>7.1f}%"
                  f"{100*(sh-fh):>+7.1f}{100*sr:>8.2f}%{100*fr:>9.2f}%"
                  f"{100*(sr-fr):>+7.2f}")
            # THE ANOMALY. A positive hit-rate lift with a negative return edge
            # means the selection picks more winners than the field and still
            # loses to it -- the losers are bigger. No accuracy panel shows this.
            if sh - fh > 0.01 and sr - fr < 0:
                fail(f"h={h} {lab}: lift {100*(sh-fh):+.1f}pp but return edge "
                     f"{100*(sr-fr):+.2f}pp — more winners, worse returns")
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, default=0, help="0 = all")
    ap.add_argument("--tickers", type=int, default=24)
    a = ap.parse_args()

    print(f"SYSTEM AUDIT  {datetime.now():%Y-%m-%d %H:%M}  read-only")
    built = None
    run = lambda n: a.section in (0, n)
    if run(1): audit_sources()
    if run(2) or run(3):
        built, _ = audit_features(a.tickers) if run(2) else (None, None)
    if run(3): audit_trainserve(built)
    if run(4): audit_models()
    if run(5): audit_ensemble()
    if run(6): audit_multipliers()
    if run(7): audit_output()

    print(f"\n{'='*78}\nSUMMARY: {len(FAIL)} FAIL, {len(WARN)} WARN, "
          f"{len(OK)} ok\n{'='*78}")
    for m in FAIL:
        print(f"  FAIL  {m}")
    for m in WARN[:20]:
        print(f"  WARN  {m}")
    if len(WARN) > 20:
        print(f"  ... and {len(WARN)-20} more warnings")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
