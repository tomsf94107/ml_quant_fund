#!/usr/bin/env python3
"""
multiplier_audit.py — does the multiplier chain help or hurt?

READ-ONLY. Writes nothing.

WHY
    In fourteen weeks the h=5 production gate emitted 49 BUYs against 24,692
    HOLDs -- about 3.5 actionable signals a week across 420 tickers. Phase 2H
    proposes an overlay filter on those BUYs, which would be filtering a book
    that barely exists.

    The cause is upstream. signals/generator.py computes

        prob_eff = prob_up x risk_mult x sent_mult x regime_mult
                           x options_mult x squeeze_mult

    and BUY requires prob_eff >= 0.70 (DEFAULT_CONFIDENCE_THRESHOLD, raised from
    0.55 on 2026-05-08). The builder's own comment shows the effect: "NVDA at
    54.6% raw x 0.78 = 42.6% effective -> HOLD".

    Every multiplier is logged as its own column, so the chain is fully
    auditable after the fact -- and as far as this audit can tell, never has
    been.

THE QUESTION THAT MATTERS
    Not "why so few BUYs" -- that is arithmetic. The question is whether the
    chain IMPROVES discrimination. A multiplier that shrinks probabilities
    without reordering them changes only how many signals clear a fixed gate;
    it adds nothing and costs coverage.

    So: does prob_eff rank outcomes better than prob_up? If AUC is the same,
    the chain is a volume control, not a signal. If AUC is WORSE, it is
    destroying information that the model produced.

    This project has been here before. The squeeze multiplier is already gated
    off behind ML_QUANT_ENABLE_SQUEEZE_MULT, and the options aggressor tilt is
    documented as descriptive-only and explicitly NOT to be gated on. Several
    of these were shipped without validation.

WHAT IS MEASURED
    1. How far each multiplier moves probabilities: distribution, and how often
       it is neutral (1.0).
    2. AUC of prob_up vs prob_eff_uncapped vs prob_cal, on the same rows.
       Same rows matters -- comparing across different subsets would confound.
    3. Per-multiplier: AUC of prob_up alone against prob_up x that multiplier,
       one at a time, so a single bad actor is visible rather than averaged into
       the chain.
    4. What the BUY count would be at several gates, with and without the chain.

    AUC is the right measure because multipliers are monotone transforms ONLY if
    applied uniformly. They are not: each row gets a different multiplier, so the
    chain genuinely reorders, and reordering can help or hurt.

    python analysis/multiplier_audit.py
"""
import argparse
import math
import sqlite3
import statistics as st

MULTS = ["risk_mult", "sent_mult", "regime_mult", "options_mult",
         "squeeze_mult", "intraday_mult", "fg_mult"]


def auc(scores, labels):
    n = len(scores)
    pos = sum(labels)
    neg = n - pos
    if not pos or not neg:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        a = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = a
        i = j + 1
    rs = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rs - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--since", default="2026-05-25")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    have = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    mults = [m for m in MULTS if m in have]
    eff = "prob_eff_uncapped" if "prob_eff_uncapped" in have else None
    cal = "prob_cal" if "prob_cal" in have else None
    sel = ["p.prob_up", "o.actual_up"] + [f"p.{m}" for m in mults]
    if eff:
        sel.append(f"p.{eff}")
    if cal:
        sel.append(f"p.{cal}")

    rows = con.execute(f"""
        SELECT {', '.join(sel)}
        FROM predictions p JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=? AND o.actual_up IS NOT NULL
          AND p.prob_up IS NOT NULL AND p.prediction_date >= ?
    """, (args.horizon, args.since)).fetchall()
    con.close()

    if len(rows) < 500:
        print(f"only {len(rows)} rows since {args.since}")
        return
    print(f"h={args.horizon}, {len(rows):,} scored predictions "
          f"since {args.since}")
    print(f"multipliers logged: {', '.join(mults)}\n")

    prob = [r[0] for r in rows]
    y = [r[1] for r in rows]

    print("1. HOW FAR EACH MULTIPLIER MOVES PROBABILITIES")
    print(f"  {'multiplier':<16}{'n':>7}{'mean':>8}{'min':>7}{'max':>7}"
          f"{'% at 1.0':>10}")
    for i, m in enumerate(mults):
        v = [r[2 + i] for r in rows if r[2 + i] is not None]
        if not v:
            print(f"  {m:<16}{'0':>7}   never logged")
            continue
        neutral = 100.0 * sum(1 for x in v if abs(x - 1.0) < 1e-9) / len(v)
        print(f"  {m:<16}{len(v):>7}{st.mean(v):>8.3f}{min(v):>7.2f}"
              f"{max(v):>7.2f}{neutral:>9.0f}%")

    print("\n2. DOES THE CHAIN IMPROVE RANKING? (same rows, so comparable)")
    base = auc(prob, y)
    print(f"  {'score':<22}{'AUC':>9}{'vs prob_up':>13}")
    print(f"  {'prob_up (raw model)':<22}{base:>9.4f}{'--':>13}")
    idx = 2 + len(mults)
    if eff:
        e = [r[idx] for r in rows]
        ok = [i for i in range(len(rows)) if e[i] is not None]
        if len(ok) > 500:
            a = auc([e[i] for i in ok], [y[i] for i in ok])
            b = auc([prob[i] for i in ok], [y[i] for i in ok])
            print(f"  {'prob_eff_uncapped':<22}{a:>9.4f}{a-b:>+13.4f}")
        idx += 1
    if cal:
        c = [r[idx] for r in rows]
        ok = [i for i in range(len(rows)) if c[i] is not None]
        if len(ok) > 500:
            a = auc([c[i] for i in ok], [y[i] for i in ok])
            b = auc([prob[i] for i in ok], [y[i] for i in ok])
            print(f"  {'prob_cal':<22}{a:>9.4f}{a-b:>+13.4f}")

    print("\n3. EACH MULTIPLIER ALONE (prob_up x that one, nothing else)")
    print("   isolates a single bad actor instead of averaging it into the chain")
    print(f"  {'multiplier':<16}{'n':>7}{'AUC':>9}{'vs prob_up':>13}")
    for i, m in enumerate(mults):
        ok = [j for j in range(len(rows)) if rows[j][2 + i] is not None]
        if len(ok) < 500:
            print(f"  {m:<16}{len(ok):>7}   too few")
            continue
        s = [prob[j] * rows[j][2 + i] for j in ok]
        b = auc([prob[j] for j in ok], [y[j] for j in ok])
        a = auc(s, [y[j] for j in ok])
        flag = "  <-- HURTS" if a - b <= -0.005 else ""
        print(f"  {m:<16}{len(ok):>7}{a:>9.4f}{a-b:>+13.4f}{flag}")

    print("\n4. BUY COUNT AT EACH GATE, with and without the chain")
    print(f"  {'gate':>6}{'on prob_up':>13}{'on prob_eff':>14}")
    for g in (0.55, 0.60, 0.65, 0.70):
        a = sum(1 for p in prob if p >= g)
        if eff:
            b = sum(1 for r in rows
                    if r[2 + len(mults)] is not None
                    and r[2 + len(mults)] >= g)
            print(f"  {g:>6.2f}{a:>13,}{b:>14,}")
        else:
            print(f"  {g:>6.2f}{a:>13,}{'n/a':>14}")

    print("\n  The production gate is prob_eff >= 0.70, raised from 0.55 on")
    print("  2026-05-08. If the chain does not improve AUC, it is a volume")
    print("  control rather than a signal: it changes how many trades clear a")
    print("  fixed bar without improving which ones do.")


if __name__ == "__main__":
    main()
