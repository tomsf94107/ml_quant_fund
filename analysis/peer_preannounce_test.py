#!/usr/bin/env python3
"""Same-quarter peer pre-announcement -> later announcer. Read-only.
Signal: mean PIT trailing SUE of same-bucket peers announced 1-45d earlier
(same fiscal quarter +-20d), snapshot A_Y - 6 sessions, usable +1BD after
each peer's announce. Outcomes: (A) through Y's print (A_Y+1 close),
(B) pre-print drift only. Per-week Spearman IC, NW-t lag 4. Null (within-
week shuffle, seed 42) runs in the same execution as the real result."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MIN_PRIORS = 4
MIN_PEERS = 3
SNAP_OFFSET = 6
SEED = 42


def load():
    con = sqlite3.connect(f"file:{ROOT/'earnings.db'}?mode=ro", uri=True)
    ev = pd.read_sql("""SELECT ticker, announce_date, fiscal_end, eps_surprise,
                               report_time
                        FROM earnings_events
                        WHERE eps_surprise IS NOT NULL AND fiscal_end IS NOT NULL
                          AND announce_date >= '2016-06-01'""", con)
    con.close()
    ev["announce_date"] = pd.to_datetime(ev.announce_date.str[:10])
    ev["fiscal_end"] = pd.to_datetime(ev.fiscal_end.str[:10])
    meta = pd.read_csv(ROOT / "tickers_metadata.csv")[["ticker", "bucket"]]
    ev = ev.merge(meta, on="ticker", how="inner").sort_values(
        ["ticker", "announce_date"]).reset_index(drop=True)
    # PIT trailing SUE: expanding std of PRIOR surprises, min MIN_PRIORS
    ev["sue"] = np.nan
    for _, idx in ev.groupby("ticker").groups.items():
        s = ev.loc[idx, "eps_surprise"].values
        out = np.full(len(s), np.nan)
        for i in range(len(s)):
            if i >= MIN_PRIORS:
                sd = np.std(s[:i], ddof=1)
                if sd > 0:
                    out[i] = s[i] / sd
        ev.loc[idx, "sue"] = out
    con = sqlite3.connect(f"file:{ROOT/'prices.db'}?mode=ro", uri=True)
    px = pd.read_sql("SELECT ticker, date, adj_close FROM daily_prices "
                     "WHERE adj_close IS NOT NULL", con)
    con.close()
    panel = px.pivot(index="date", columns="ticker",
                     values="adj_close").sort_index()
    panel.index = pd.to_datetime(panel.index)
    return ev, panel


def build_events(ev, panel):
    cal = panel.index
    pos = pd.Series(np.arange(len(cal)), index=cal)

    def next_sess(d):
        i = cal.searchsorted(d, side="right")
        return cal[i] if i < len(cal) else pd.NaT

    ev = ev[ev.sue.notna()].copy()
    ev["eff_date"] = ev.announce_date.map(next_sess)   # usable +1 session
    rows = []
    for bkt, g in ev.groupby("bucket"):
        g = g.sort_values("announce_date")
        for _, y in g.iterrows():
            i_y = cal.searchsorted(y.announce_date)
            if i_y - SNAP_OFFSET < 0 or i_y + 1 >= len(cal):
                continue
            snap, prep = cal[i_y - SNAP_OFFSET], cal[i_y - 1]
            post = cal[min(i_y + 1, len(cal) - 1)]
            z = g[(g.ticker != y.ticker)
                  & (abs((g.fiscal_end - y.fiscal_end).dt.days) <= 20)
                  & (g.announce_date < y.announce_date)
                  & ((y.announce_date - g.announce_date).dt.days <= 45)
                  & (g.eff_date <= snap)]
            if len(z) < MIN_PEERS or y.ticker not in panel.columns:
                continue
            p = panel[y.ticker]
            p0 = p.get(snap, np.nan)
            if not np.isfinite(p0) or p0 <= 0:
                continue
            ra = p.get(post, np.nan) / p0 - 1.0
            rb = p.get(prep, np.nan) / p0 - 1.0
            rows.append({"ticker": y.ticker, "bucket": bkt,
                         "snap": snap, "week": snap.to_period("W").start_time,
                         "year": snap.year, "n_peers": len(z),
                         "sig": z.sue.mean(),
                         "ret_through": ra, "ret_preprint": rb})
    return pd.DataFrame(rows)


def weekly_ic(df, sig_col, ret_col):
    ics = []
    for wk, g in df.dropna(subset=[sig_col, ret_col]).groupby("week"):
        if len(g) >= 6 and g[sig_col].nunique() > 2:
            ic = stats.spearmanr(g[sig_col], g[ret_col]).statistic
            if np.isfinite(ic):
                ics.append({"week": wk, "year": wk.year, "n": len(g), "ic": ic})
    return pd.DataFrame(ics)


def nw_t(x, lag=4):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        v += 2 * w * (e[:-L] @ e[L:]) / n
    return x.mean() / np.sqrt(v / n)


def line(tag, w):
    if w.empty:
        print(f"  {tag:<26} no weeks")
        return
    t = nw_t(w.ic.values)
    yr = w.groupby("year").ic.mean()
    print(f"  {tag:<26} IC {w.ic.mean():+.4f}  NW-t {t:+5.2f}  "
          f"weeks {len(w)}  yrs+ {(yr > 0).sum()}/{len(yr)}")


def main():
    ev, panel = load()
    df = build_events(ev, panel)
    print(f"Y-events usable: {len(df)}  ({df.ticker.nunique()} tickers, "
          f"{df.week.nunique()} weeks, {df.snap.min().date()} -> {df.snap.max().date()})")
    print(f"median peers/event: {df.n_peers.median():.0f}")

    for label, col in (("A: through Y's print", "ret_through"),
                       ("B: pre-print drift", "ret_preprint")):
        print(f"\n=== OUTCOME {label} ===")
        line("REAL", weekly_ic(df, "sig", col))
        rng = np.random.default_rng(SEED)
        d2 = df.copy()
        d2["sig_null"] = d2.groupby("week").sig.transform(
            lambda s: rng.permutation(s.values))
        line("NULL (within-week shuffle)", weekly_ic(d2, "sig_null", col))
        w_all = weekly_ic(df, "sig", col)
        if not w_all.empty:
            yr = w_all.groupby("year").ic.mean()
            print("  yearly IC:", "  ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))
        print("  per-bucket (n>=100 events):")
        for bkt, g in df.groupby("bucket"):
            if len(g) >= 100:
                w = weekly_ic(g, "sig", col)
                if len(w) >= 30:
                    print(f"    {bkt:<18} IC {w.ic.mean():+.4f}  "
                          f"NW-t {nw_t(w.ic.values):+5.2f}  ev {len(g)}")
        # tercile spread, pooled diagnostic
        dd = df.dropna(subset=["sig", col]).copy()
        dd["terc"] = dd.groupby("week").sig.transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)
            if len(s) >= 6 else np.nan)
        sp = dd[dd.terc == 2][col].mean() - dd[dd.terc == 0][col].mean()
        print(f"  top-bottom tercile spread (pooled, diag): {sp*100:+.3f}%")


if __name__ == "__main__":
    sys.exit(main())
