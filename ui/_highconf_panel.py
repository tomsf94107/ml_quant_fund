"""
_highconf_panel.py — per-ticker high-confidence accuracy, with the statistics
that stop it being read as a stock picker.

Used by both ui/1_Dashboard.py (collapsed) and ui/pages/2_Accuracy.py (open).
Written once so the two cannot drift.

WHY THE STATISTICS ARE NOT OPTIONAL HERE
    A plain ranking of tickers by recent accuracy is the most misleading table
    this project could ship. Measured 2026-09-02:

      - At prob_up>=0.70 over 30 days there are ~1-3 predictions PER TICKER.
        A ticker with one correct call ranks 100%.
      - Ranking by a prior window barely predicts the next: Spearman rho was
        +0.26 (h=3) and +0.21 (h=5) across ~72 tickers. Real but weak.
      - Individual rows swing violently: IR 75.0% -> 25.0%, KVUE 60.0% -> 0.0%,
        GOOG 57.9% -> 11.1%.
      - With ~194 tickers tested at 5%, about 10 clear 50% by luck alone.

    So the panel reports four things a bare ranking would not: the sample size,
    the Wilson interval, the prior window's value, and a Benjamini-Hochberg
    false-discovery correction across all tickers tested.

TWO-SIDED FDR
    Both tails are tested. A ticker significantly BELOW 50% is as non-random as
    one above, and is a diagnostic worth seeing -- it usually means the model
    is mishandling that name (bad features, a corporate action, a regime it
    cannot read), not that the signal should be inverted. The daily SELL signal
    was closed in July 2026 after an apparent inversion at n=1,234 flipped sign.

WHAT SURVIVES FDR MEANS
    At q=0.10, at most ~10% of the flagged set is expected to be false. That is
    a named set you can reason about, unlike "31 above 50%, about 10 of them
    luck, unidentified".
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

import pandas as pd
import streamlit as st

DB = "accuracy.db"

ETF_EXCLUDE = {
    "XLE", "XLF", "XLK", "XLV", "XLP", "XLU", "XLI", "XLY", "XLB", "XLRE",
    "XLC", "SPY", "QQQ", "IWM", "RSP", "SMH", "IGV", "ARKK", "TLT", "GLD",
    "USO", "VXX", "DIA", "EEM", "EFA", "HYG", "LQD", "XBI", "XOP", "XRT",
    "KRE", "SOXX", "VNQ", "SLV", "VOO", "VTI",
}

_SQL = """
SELECT p.ticker,
       COUNT(*) AS n,
       SUM(CASE WHEN (p.prob_up >= 0.5) = (o.actual_up = 1) THEN 1 ELSE 0 END) AS hits
FROM predictions p
JOIN outcomes o
  ON p.ticker = o.ticker
 AND p.prediction_date = o.prediction_date
 AND p.horizon = o.horizon
WHERE p.horizon = ?
  AND p.prob_up >= ?
  AND o.actual_up IS NOT NULL
  AND p.prediction_date >= ?
  AND p.prediction_date < ?
GROUP BY p.ticker
"""


def _wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval, as percentages.

    Wilson rather than the normal approximation: it stays inside [0, 100] and
    does not collapse at small n, which is the regime every row here lives in.
    """
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value against p=0.5.

    Exact rather than normal-approximate because n is often under 20, where the
    approximation is poor in exactly the tail that matters.
    """
    if not n:
        return 1.0
    from math import comb
    obs = comb(n, k) * (p ** n)
    total = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * (p ** n)
        if pi <= obs * (1 + 1e-9):
            total += pi
    return min(1.0, total)


def _bh(pvals, q: float):
    """Benjamini-Hochberg. Returns a boolean list, True = survives at level q.

    Sort p ascending; find the largest i with p_i <= (i/m) * q; everything up to
    that rank is a discovery. Controls the expected PROPORTION of false
    positives among discoveries, which is the right guarantee when testing
    ~200 tickers at once -- unlike a raw 0.05 cut, which yields ~10 false
    positives by construction.
    """
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    cutoff_rank = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * q:
            cutoff_rank = rank
    out = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff_rank:
            out[idx] = True
    return out


def _spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for m_ in range(i, j + 1):
                r[o[m_]] = avg
            i = j + 1
        return r
    n = len(xs)
    if n < 4:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


@st.cache_data(ttl=600)
def _fetch(horizon: int, thresh: float, start: str, end: str):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(_SQL, (horizon, thresh, start, end)).fetchall()
    con.close()
    return {t: (n, k) for t, n, k in rows if t not in ETF_EXCLUDE}


def render(key_prefix: str = "hc", default_expanded: bool = True):
    """Render the panel. key_prefix keeps widget keys unique across pages."""
    st.subheader("High-confidence accuracy by ticker")
    st.caption(
        "What each ticker has done lately — not which to trust. "
        "Rank persistence is weak (ρ ≈ 0.2); read the prior column before "
        "trusting any row."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        preset = st.radio("Period", ["14d", "30d", "60d", "90d", "Custom"],
                          index=1, horizontal=True, key=f"{key_prefix}_period")
    with c2:
        horizon = st.radio("Horizon", [3, 5], index=1, horizontal=True,
                           key=f"{key_prefix}_h",
                           help="Independent of the sidebar horizon setting.")
    with c3:
        q = st.select_slider("FDR q", options=[0.05, 0.10, 0.15, 0.20, 0.25],
                             value=0.10, key=f"{key_prefix}_q",
                             help="Looser q gives a longer list with more "
                                  "false positives.")

    today = date.today()
    if preset == "Custom":
        d1, d2 = st.columns(2)
        start = d1.date_input("From", today - timedelta(days=30),
                              key=f"{key_prefix}_from")
        end = d2.date_input("To", today, key=f"{key_prefix}_to")
        start, end = str(start), str(end)
        span = max((date.fromisoformat(end) - date.fromisoformat(start)).days, 1)
    else:
        span = int(preset.rstrip("d"))
        start = str(today - timedelta(days=span))
        end = str(today)

    s1, s2 = st.columns(2)
    thresh = s1.slider("Confidence threshold (prob_up ≥)", 0.50, 0.90, 0.55,
                       0.01, key=f"{key_prefix}_thresh")
    min_n = s2.slider("Minimum predictions per ticker", 1, 40, 15, 1,
                      key=f"{key_prefix}_minn",
                      help="Below ~10 a single call moves the number by "
                           "10 points or more.")

    now = _fetch(horizon, thresh, start, end)
    prior_start = str(date.fromisoformat(start) - timedelta(days=span))
    prior = _fetch(horizon, thresh, prior_start, start)

    rows = [(t, n, k) for t, (n, k) in now.items() if n >= min_n]
    if not rows:
        st.info(f"No tickers with at least {min_n} predictions at "
                f"prob_up ≥ {thresh:.2f} in this window. Lower the threshold, "
                f"widen the period, or reduce the minimum.")
        return

    pvals = [_binom_two_sided(k, n) for _, n, k in rows]
    surv = _bh(pvals, q)

    recs = []
    for (t, n, k), pv, sv in zip(rows, pvals, surv):
        acc = 100.0 * k / n
        lo, hi = _wilson(k, n)
        pn, pk = prior.get(t, (0, 0))
        pacc = 100.0 * pk / pn if pn else None
        if sv:
            flag = "FDR ▲" if acc > 50 else "FDR ▼"
        elif lo > 50:
            flag = "raw ▲"
        elif hi < 50:
            flag = "raw ▼"
        else:
            flag = "—"
        recs.append({
            "Ticker": t, "Now %": round(acc, 1), "n": n,
            "CI low": round(lo, 1), "CI high": round(hi, 1),
            "Prior %": round(pacc, 1) if pacc is not None else None,
            "Δ pp": round(acc - pacc, 1) if pacc is not None else None,
            "Prior n": pn or None,
            "p": round(pv, 4), "Flag": flag,
        })
    df = pd.DataFrame(recs).sort_values("Now %", ascending=False)

    raw_above = sum(1 for r in recs if r["CI low"] > 50)
    raw_below = sum(1 for r in recs if r["CI high"] < 50)
    fdr_n = sum(surv)
    by_chance = 0.05 * len(rows)
    stable = sum(1 for r in recs
                 if r["Prior %"] is not None and r["Now %"] > 50 and r["Prior %"] > 50)
    with_prior = sum(1 for r in recs if r["Prior %"] is not None)
    stable_chance = 0.25 * with_prior

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tickers", len(rows))
    m2.metric("Above 50% (raw)", raw_above, f"~{by_chance:.0f} by chance",
              delta_color="off")
    m3.metric(f"Survives FDR q={q}", fdr_n,
              "named set" if fdr_n else "none clear", delta_color="off")
    m4.metric("Below 50% (raw)", raw_below, delta_color="off")
    m5.metric("Stable both windows", stable,
              f"~{stable_chance:.0f} by chance", delta_color="off")

    pri = [r["Prior %"] for r in recs if r["Prior %"] is not None]
    lat = [r["Now %"] for r in recs if r["Prior %"] is not None]
    rho = _spearman(pri, lat) if len(pri) >= 4 else None
    med_n = sorted(r["n"] for r in recs)[len(recs) // 2]
    if rho is not None:
        st.caption(f"Rank persistence prior → now: ρ = {rho:+.2f} "
                   f"on {len(pri)} tickers. Near zero means the ranking "
                   f"describes the past only.")
    if med_n < 10:
        st.warning(
            f"Median n is {med_n}. At this sample size a single call moves a "
            f"ticker by more than 10 points, and ρ is mostly noise. Read at "
            f"60d or 90d, or raise the minimum."
        )

    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=420 if default_expanded else 280)

    st.caption(
        "**Flag** — FDR ▲ survives multiple-testing correction and is above "
        "50%; FDR ▼ survives and is BELOW 50%, meaning the model is reliably "
        "wrong on that name (a diagnostic, not a signal to invert — the daily "
        "SELL signal was closed in July after exactly that pattern flipped "
        "sign). raw ▲/▼ clears 50% on its own interval but not after "
        "correction. — is inconclusive, which most rows are."
    )
    st.download_button(
        "Download CSV", df.to_csv(index=False).encode(),
        file_name=f"highconf_h{horizon}_{start}_{end}.csv",
        mime="text/csv", key=f"{key_prefix}_csv")
