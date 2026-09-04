#!/usr/bin/env python3
"""
confidence_gate_test.py — is the MODEL less accurate when supply pressure is high?

READ-ONLY. Writes nothing.

A DIFFERENT DEPENDENT VARIABLE
    Five tests have asked whether insider data predicts RETURNS. All null:
    flow, trajectory, overhang, buying conditioned on short interest, and the
    Cohen-Malloy-Pomorski routine/opportunistic split.

    This asks something else: is the MODEL'S OWN ACCURACY lower when insider
    selling, short interest or option skew is elevated?

    Those are not the same question. A predictor needs an edge over the market.
    A GATE only needs the model's calls to be worse in an identifiable state --
    the market can be perfectly efficient about supply pressure while this
    particular model, which sees only price, volume and volatility, is
    systematically fooled by it.

    CRWV is the motivating case. The model scored 9.1% on 11 high-confidence
    h=5 calls while a 30% holder distributed 70M shares. It was not wrong about
    the market. It was wrong about a name whose dominant driver was outside its
    feature set. If that generalises, the fix is not a feature -- it is a
    refusal to emit high confidence in that state.

WHAT IS TESTED
    For every prediction with a resolved outcome, the state variables as of the
    prediction date:

      insider_sell_dov   90-day insider dollar sales / 20-day dollar ADV
      days_to_cover      FINRA days-to-cover, latest settlement public by then
      iv_skew_snap       option skew already logged in prediction_features
      pc_ratio_snap      put/call ratio already logged

    For each, predictions are split into terciles of the state variable, and
    accuracy is compared across terciles -- overall and for high-confidence
    calls specifically.

    The comparison that matters is HIGH-STATE accuracy against LOW-STATE
    accuracy within the high-confidence cohort. A gate is justified only if
    high-confidence calls are materially and consistently worse in the high
    state.

WHY THIS COULD FIND SOMETHING WHERE FIVE TESTS FOUND NOTHING
    It is a conditional statement about a specific model's errors, not a claim
    about market efficiency. The bar is also lower in a defensible way: a gate
    that removes bad calls does not need to beat the market, only to beat
    emitting them.

    It could equally find nothing, in which case CRWV was idiosyncratic and the
    existing 1,000-bar history gate already covers the population.

POINT-IN-TIME
    Insider data keyed on filing_date. Short interest uses only settlements at
    least 12 days old, since FINRA publishes roughly 8 business days after
    settlement. Skew and put/call come from prediction_features, recorded at
    prediction time by construction.

    python analysis/confidence_gate_test.py
"""
import argparse
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy-db", default="accuracy.db")
    ap.add_argument("--insider-db", default="insider_trades.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--si-db", default="short_interest.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.55)
    args = ap.parse_args()

    ac = sqlite3.connect(f"file:{args.accuracy_db}?mode=ro", uri=True)
    feat_cols = [r[1] for r in ac.execute("PRAGMA table_info(prediction_features)")]
    have_skew = "iv_skew_snap" in feat_cols
    have_pc = "pc_ratio_snap" in feat_cols
    sel = ", ".join(
        ["p.ticker", "p.prediction_date", "p.prob_up", "o.actual_up"]
        + (["f.iv_skew_snap"] if have_skew else [])
        + (["f.pc_ratio_snap"] if have_pc else []))
    preds = ac.execute(f"""
        SELECT {sel}
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        LEFT JOIN prediction_features f ON f.ticker=p.ticker
          AND f.prediction_date=p.prediction_date AND f.horizon=p.horizon
        WHERE p.horizon=? AND o.actual_up IS NOT NULL AND p.prob_up IS NOT NULL
    """, (args.horizon,)).fetchall()
    ac.close()
    print(f"{len(preds):,} scored h={args.horizon} predictions")

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close, volume = defaultdict(dict), defaultdict(dict)
    for t, d, c, v in px.execute(
            "SELECT ticker, d, close, volume FROM raw_bars WHERE d>='2025-01-01'"):
        if c:
            close[t][d] = c
        if v:
            volume[t][d] = v
    px.close()
    dadv = {}
    for t, s in close.items():
        vs = volume.get(t, {})
        vd = sorted(vs)
        for i in range(20, len(vd)):
            w = [vs[vd[j]] * s.get(vd[j], 0) for j in range(i - 20, i)]
            m = sum(w) / len(w)
            if m > 0:
                dadv[(t, vd[i])] = m

    ic = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    sell = defaultdict(float)
    for t, fd, code, sh, pps in ic.execute(
            "SELECT ticker, filing_date, transaction_code, shares, "
            "price_per_share FROM insider_filings_raw "
            "WHERE filing_date >= '2024-10-01' AND transaction_code='S'"):
        sell[(t, str(fd)[:10])] += abs(sh or 0) * (pps or 0)
    ic.close()

    si = defaultdict(dict)
    try:
        sc = sqlite3.connect(f"file:{args.si_db}?mode=ro", uri=True)
        cols = [r[1] for r in sc.execute("PRAGMA table_info(short_interest)")]
        if "days_to_cover" in cols:
            for t, d, v in sc.execute(
                    "SELECT ticker, settlement_date, days_to_cover "
                    "FROM short_interest WHERE days_to_cover IS NOT NULL"):
                si[t][str(d)[:10]] = v
        sc.close()
    except Exception:
        pass

    def si_asof(t, d):
        s = si.get(t)
        if not s:
            return None
        cut = (date.fromisoformat(d) - timedelta(days=12)).isoformat()
        cand = [k for k in s if k <= cut]
        return s[max(cand)] if cand else None

    def sell90(t, d):
        d0 = date.fromisoformat(d)
        return sum(sell.get((t, (d0 - timedelta(days=k)).isoformat()), 0.0)
                   for k in range(90))

    states = ["insider_sell_dov", "days_to_cover"]
    if have_skew:
        states.append("iv_skew_snap")
    if have_pc:
        states.append("pc_ratio_snap")

    recs = []
    for row in preds:
        t, d, prob, actual = row[0], row[1], row[2], row[3]
        i = 4
        skew = row[i] if have_skew else None
        if have_skew:
            i += 1
        pc = row[i] if have_pc else None
        a = dadv.get((t, d))
        s = {}
        if a:
            s["insider_sell_dov"] = sell90(t, d) / a
        dtc = si_asof(t, d)
        if dtc is not None:
            s["days_to_cover"] = dtc
        if skew is not None:
            s["iv_skew_snap"] = skew
        if pc is not None:
            s["pc_ratio_snap"] = pc
        recs.append((t, d, prob, actual, s))

    for state in states:
        vals = sorted(r[4][state] for r in recs if state in r[4])
        if len(vals) < 300:
            print(f"\n=== {state}: only {len(vals)} observations, skipped ===")
            continue
        lo_cut = vals[len(vals) // 3]
        hi_cut = vals[2 * len(vals) // 3]
        print(f"\n=== {state} ===")
        print(f"  terciles at {lo_cut:.4g} / {hi_cut:.4g}, "
              f"{len(vals):,} observations")
        print(f"  {'cohort':<16}{'tercile':<8}{'n':>7}{'acc':>8}"
              f"{'base':>8}{'lift':>9}{'95% CI':>18}")
        for cohort, filt in (("all", lambda p: True),
                             (f"prob>={args.conf}", lambda p: p >= args.conf),
                             ("prob>=0.70", lambda p: p >= 0.70)):
            for label, lo, hi in (("LOW", -1e18, lo_cut),
                                  ("MID", lo_cut, hi_cut),
                                  ("HIGH", hi_cut, 1e18)):
                sub = [r for r in recs
                       if state in r[4] and lo <= r[4][state] < hi
                       and filt(r[2])]
                if len(sub) < 30:
                    print(f"  {cohort:<16}{label:<8}{len(sub):>7}   too few")
                    continue
                n = len(sub)
                k = sum(1 for r in sub if r[3] == 1)
                # base rate: up-rate of ALL predictions in this tercile,
                # so market direction within the state is controlled for
                allsub = [r for r in recs
                          if state in r[4] and lo <= r[4][state] < hi]
                base = 100.0 * sum(1 for r in allsub if r[3] == 1) / len(allsub)
                acc = 100.0 * k / n
                cl, ch = wilson(k, n)
                print(f"  {cohort:<16}{label:<8}{n:>7}{acc:>7.1f}%"
                      f"{base:>7.1f}%{acc-base:>+8.1f}pp"
                      f"   [{cl:>5.1f},{ch:>5.1f}]")
            print()

    print("  The comparison that matters is LIFT in the HIGH tercile against "
          "LIFT in\n  the LOW tercile, within the high-confidence cohorts. "
          "Accuracy alone is\n  confounded by market direction, which is why "
          "the base rate is computed\n  WITHIN each tercile.\n")
    print("  A gate is justified only if high-confidence lift is materially "
          "and\n  consistently WORSE in the HIGH state, with non-overlapping "
          "intervals.\n  Anything less is not worth suppressing calls over.")


if __name__ == "__main__":
    main()
