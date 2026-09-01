#!/usr/bin/env python3
"""
squeeze_radar.py -- two-axis short-squeeze screen: FUEL x IGNITION.

WHAT THIS IS
  A SCREEN. It ranks names by how much short-side fuel is loaded and whether
  anything is lighting it right now. It is NOT a validated signal: the weights
  below are hand-set and have never passed per-date rank-IC / Newey-West /
  shuffle null. Do NOT size positions from this score. Candidate generation only.
  (The fund has ONE validated brick: FINRA SI, low-DTC long leg. This is not it.)

WHY IT EXISTS
  squeeze_live.py  ranks by borrow fee alone -- FUEL, one axis, no timing.
                   A name sits fully fuelled for months and the scan can't tell.
  squeeze_scan.py  finds the multi-day RAMP/IGNITION pattern, but one ticker at
                   a time with ABSOLUTE thresholds calibrated on BYND
                   (ret_1d >= 20%). Verified 2026-08-26: those can never fire on
                   a mega-cap -- NVDA printed 19 RAMPs and 0 IGNITIONs in 6mo.
                   Absolute thresholds do not travel across market caps.
  this file        combines both, CROSS-SECTIONALLY PERCENTILE-RANKED so the
                   ignition axis means the same thing at every market cap, and
                   scores fuel DISCOUNTED by ignition.

DENOMINATOR HONESTY
  SI %-of-float is deliberately NOT reported. short_interest.db has no float
  column, and UW's `total_float` is shares OUTSTANDING mislabelled -- verified
  2026-08-26: GME total_float 448,691,257 == shares outstanding; true float is
  ~409.4M. A wrong denominator is worse than none. DTC is denominator-free.

DTC IS RECOMPUTED
  FINRA's stored avg_daily_vol is as-of the settlement date, weeks stale.
  We recompute days-to-cover against live trailing-20d ADV from raw_bars and
  show BOTH. Measured spread on this universe: -82% to +123%.

RUN
  radar                      # full universe, live fee probe on all
  radar --probe 40           # probe only the top 40 by local fuel (fast)
  radar --probe 0            # zero API calls, local only
  radar --only BYND RZLV     # specific names
  radar --top 30 --save
"""

import argparse, os, sqlite3, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from squeeze_scan import compute_signals          # verified, ET-clamped 2026-08-26

BARS_LOOKBACK = 130          # calendar days of bars pulled for rvol/thrust
DTC_JUNK_CEILING = 50.0      # FINRA OTC junk rule
UW_PAUSE = 0.12              # matches squeeze_live.py's verified-working pace
IGNITION_FLOOR = 0.15        # score = fuel * (FLOOR + (1-FLOOR)*ignition)
# Ignition is SELF-NORMALIZED, not cross-sectionally ranked. Percentile rank
# was wrong (fixed 2026-08-26): it made ignition relative, so the top of any
# sorted list scored high even when nothing was moving, and it discarded
# magnitude -- SCCO (+10.6% on rvol 0.78, i.e. BELOW-average volume) ranked
# 93 against MRNA (+19.1% on rvol 8.96) whose raw thrust was ~20x larger.
# Sigma units travel across market caps; raw % does not. rvol is already
# self-normalized. Both conditions must hold, so the form is MULTIPLICATIVE:
# a big move on no volume is drift, and volume with no move is churn.
THRUST_Z_FULL = 3.0          # 3-day move at 3 sigma of own vol -> full credit
RVOL_FULL     = 3.0          # 3x normal volume -> full credit
RVOL_MIN      = 1.0          # at/below average volume -> zero ignition
SPLIT_RET_MIN = 0.60         # |1d move| that triggers split inspection
SPLIT_PROD_LO, SPLIT_PROD_HI = 0.30, 3.00

# Starter exclusion set: ETF/ETN short interest is not squeeze fuel --
# creation/redemption makes the float elastic, so shorts are never trapped.
# VERIFY against signals/generator.py's own _is_etf logic and extend.
ETF_EXCLUDE = {
    "SPY", "QQQ", "IWM", "DIA", "IGV", "SMH", "ARKK", "TLT", "HYG", "LQD",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
}


def pct_rank(s):
    return s.rank(pct=True, na_option="keep")


def fee_tier(fee_pct):
    """Borrow-fee tier. Cutoffs identical to squeeze_live.py so the TIER label
    means exactly the same thing in both tools -- deliberately not 'improved'."""
    if fee_pct is None or fee_pct != fee_pct:
        return "-"
    if fee_pct >= 20:
        return "EXTREME"
    if fee_pct >= 5:
        return "HIGH"
    if fee_pct >= 1:
        return "MODERATE"
    return "easy"


def load_universe_safe(prices_db, si_db):
    try:
        from si_fetch_v2 import load_universe
        return sorted(load_universe(prices_db, si_db) or [])
    except Exception as e:
        print(f"  load_universe unavailable ({e}); falling back to raw_bars")
        con = sqlite3.connect(prices_db)
        u = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM raw_bars")]
        con.close()
        return sorted(u)


def load_bars(prices_db, tickers):
    con = sqlite3.connect(prices_db)
    cutoff = (pd.Timestamp.utcnow().tz_localize(None)
              - pd.Timedelta(days=BARS_LOOKBACK)).strftime("%Y-%m-%d")
    q = ("SELECT ticker, d AS date, open, high, low, close, volume "
         "FROM raw_bars WHERE d >= ? ORDER BY ticker, d")
    df = pd.read_sql(q, con, params=(cutoff,))
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    keep = set(tickers)
    return {t: g.reset_index(drop=True)
            for t, g in df.groupby("ticker") if t in keep and len(g) >= 25}


def load_si(si_db):
    """Last two settlements per ticker."""
    con = sqlite3.connect(si_db)
    df = pd.read_sql(
        "SELECT ticker, settlement_date, current_short, avg_daily_vol, days_to_cover "
        "FROM short_interest ORDER BY ticker, settlement_date DESC", con)
    con.close()
    df.loc[df["days_to_cover"] > DTC_JUNK_CEILING, "days_to_cover"] = np.nan
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.head(2).reset_index(drop=True)
        prev = float(g.loc[1, "current_short"]) if len(g) > 1 else np.nan
        cur = float(g.loc[0, "current_short"])
        out[t] = {
            "settle": g.loc[0, "settlement_date"],
            "short": cur,
            "dtc_finra": g.loc[0, "days_to_cover"],
            "si_chg": (cur - prev) / prev if prev and prev > 0 else np.nan,
        }
    return out


def load_splits(prices_db):
    """prices.db:splits is AUTHORITATIVE -- 330 rows/193 tickers, maintained from
    Massive's splits endpoint by price_cache / massive_client / backfill_raw_bars.
    Convention (verified 2026-08-26 against CRWD 4:1 = 1.0|4.0):
        adj_price(d) = close(d) * (split_from/split_to) for exec_date > d
    """
    con = sqlite3.connect(prices_db)
    rows = con.execute("SELECT ticker, exec_date, split_from, split_to FROM splits "
                       "WHERE split_from > 0 AND split_to > 0").fetchall()
    con.close()
    out = {}
    for tk, ed, sf, st in rows:
        out.setdefault(tk, []).append((str(ed)[:10], float(sf) / float(st)))
    for v in out.values():
        v.sort()
    return out


def apply_splits(g, splits):
    """Rebase raw (UNADJUSTED) bars onto the current basis using TABLE data only."""
    g = g.copy()
    applied = []
    d0 = g["date"].iloc[0].strftime("%Y-%m-%d")
    for ed, f in splits:
        if ed <= d0 or f == 1.0:
            continue
        idx = g.index[g["date"] < pd.Timestamp(ed)]
        if not len(idx):
            continue
        for c in ("open", "high", "low", "close"):
            g.loc[idx, c] = g.loc[idx, c] * f
        g.loc[idx, "volume"] = g.loc[idx, "volume"] / f
        applied.append((ed, f))
    return g, applied


def residual_discontinuity(g):
    """FLAG-ONLY. Never adjusts.

    A prior version inferred splits from bars via price_ratio*volume_ratio ~ 1.
    Over full history that flagged GME 2021-01-26/27/29 and AMC 2021-06-02 --
    the canonical short squeezes -- as splits, in a short-squeeze scanner.
    Implied ratios were 1.60-2.35; real split ratios are round. The heuristic
    does not generalize and must never rewrite prices. It now only reports a
    suspected gap so the row can be SUPPRESSED rather than silently wrong.
    """
    pr = g["close"] / g["close"].shift(1)
    for i in range(max(1, len(g) - 25), len(g)):
        p = pr.iat[i]
        if p == p and (p >= 5.0 or p <= 0.20):
            return g["date"].iat[i].strftime("%Y-%m-%d"), float(p)
    return None, None


def bar_metrics(g, splits):
    g, _splits = apply_splits(g, splits)
    _sd, _sr = residual_discontinuity(g)
    s = compute_signals(g)
    last = s.iloc[-1]
    f = lambda k: float(last[k]) if pd.notna(last[k]) else np.nan
    # Vol window EXCLUDES the 3-day thrust window (fix 2026-08-26). Including
    # it let a violent move inflate its own denominator and suppress its own
    # z-score: MRNA +19.1% on rvol 8.96 scored ignition 9 because vol20d came
    # back ~41% daily -- contaminated by the very days being measured.
    # MAD, not std. One 177% observation destroys a 20-point standard deviation:
    # MRNA (+19.1% over 3d on rvol 8.96) scored ignition 9 because its own +177%
    # day of 2026-08-19 sat in the vol window and inflated the denominator to
    # ~41% daily. MAD is unmoved by a single outlier.
    _r = s["ret_1d"].shift(3).tail(20).dropna()
    v = 1.4826 * (_r - _r.median()).abs().median() if len(_r) >= 10 else np.nan
    return {"close": f("close"), "ret_3d": f("ret_3d"), "rvol": f("rvol"),
            "vol20d": float(v) if pd.notna(v) else np.nan,
            "adv20": f("vol20"), "up_streak": int(last["up_streak"]),
            "ramp": bool(last["FLAG_ramp"]), "top": bool(last["FLAG_top"]),
            "split_dates": [d for d, _ in _splits],
            "split_factor": float(np.prod([p for _, p in _splits])) if _splits else 1.0,
            "suspect_date": _sd, "suspect_ratio": _sr,
            "bar_date": last["date"].strftime("%Y-%m-%d")}


def probe_borrow(ticker):
    from monitor_ticker import uw_get
    rows = (uw_get(f"/api/shorts/{ticker}/data") or {}).get("data") or []
    if not rows:
        return None
    r = rows[0]
    # UW returns newest-first; rows[0] is correct. What was missing is the
    # OBSERVATION time -- without it the caller stamped its own clock and a
    # 20-hour-old snapshot was recorded as a fresh reading.
    obs_ts = r.get("timestamp")
    try:
        fee = float(r.get("fee_rate")) if r.get("fee_rate") is not None else None
    except (TypeError, ValueError):
        fee = None
    try:
        av = int(r.get("short_shares_available")) if r.get("short_shares_available") is not None else None
    except (TypeError, ValueError):
        av = None
    stale_h = None
    if obs_ts:
        try:
            from datetime import datetime as _dt, timezone as _tz
            _o = _dt.fromisoformat(str(obs_ts).replace("Z", "+00:00"))
            stale_h = (_dt.now(_tz.utc) - _o).total_seconds() / 3600.0
        except Exception:
            stale_h = None
    return {"fee": fee, "avail": av, "obs_ts": obs_ts, "stale_h": stale_h}


def log_borrow_live(borrow_db, rows, ts):
    """Point-in-time borrow series. CANNOT be honestly backfilled later.

    ts_utc is UW'S OBSERVATION TIME, not our probe time. Recording the probe
    time made three probes of one unchanged 20-hour-old snapshot look like
    three separate observations (found 2026-09-01). Since ts_utc is in the
    primary key, repeat probes of unchanged data now collapse to a single row,
    which is what a point-in-time series is supposed to do.

    probed_at keeps our own clock, so the two are separable and the lag is
    always recoverable.
    """
    con = sqlite3.connect(borrow_db, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS borrow_live(
        ticker TEXT NOT NULL, ts_utc TEXT NOT NULL, fee_bps REAL,
        shares_avail INTEGER, source TEXT, PRIMARY KEY(ticker, ts_utc))""")
    cols = [c[1] for c in con.execute("PRAGMA table_info(borrow_live)")]
    if "probed_at" not in cols:
        con.execute("ALTER TABLE borrow_live ADD COLUMN probed_at TEXT")
    con.executemany(
        "INSERT OR REPLACE INTO borrow_live "
        "(ticker, ts_utc, fee_bps, shares_avail, source, probed_at) "
        "VALUES (?,?,?,?,?,?)",
        [(r["ticker"],
          r.get("obs_ts") or ts,          # UW's clock, falling back to ours
          (r["fee"] * 100.0) if r["fee"] is not None else None,
          r["avail"], "UW:/api/shorts/{t}/data", ts) for r in rows
         if r.get("fee") is not None or r.get("avail") is not None])
    con.commit(); con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--prices-db"); ap.add_argument("--si-db"); ap.add_argument("--borrow-db")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--probe", type=int, default=-1,
                    help="probe live fee for top N by local fuel; -1 = all, 0 = none")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--keep-etf", action="store_true")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    prices_db = a.prices_db or os.path.join(a.root, "prices.db")
    si_db     = a.si_db     or os.path.join(a.root, "short_interest.db")
    borrow_db = a.borrow_db or os.path.join(a.root, "borrow.db")

    # FUEL is a cross-sectional percentile, meaningful ONLY against the full
    # universe. Ranking within an --only set made BYND read fuel 100 in a 2-name
    # call and 83 in a 3-name call at the same instant (observed 2026-08-26).
    # Always rank the whole universe; --only filters the DISPLAY, not the rank.
    _focus = [t.upper() for t in a.only] if a.only else None
    universe = load_universe_safe(prices_db, si_db)
    if _focus:
        universe = sorted(set(universe) | set(_focus))
    n_raw = len(universe)
    # Exclude ETFs from the RANKING population always -- gating this on --only
    # made squeezeselect rank against 441 names while squeeze ranked against 423,
    # so the same ticker read a different fuel percentile depending on the call.
    # Explicitly-named ETFs are still kept so squeezeselect SPY shows something.
    if not a.keep_etf:
        universe = [t for t in universe
                    if t not in ETF_EXCLUDE or (_focus and t in _focus)]
    n_etf = n_raw - len(universe)

    t0 = time.time()
    bars = load_bars(prices_db, universe)
    splits_map = load_splits(prices_db)
    si = load_si(si_db)

    rows = []
    for t in universe:
        g = bars.get(t)
        if g is None:
            continue
        try:
            m = bar_metrics(g, splits_map.get(t, []))
        except Exception:
            continue
        s = si.get(t, {})
        adv = m["adv20"]
        short = s.get("short", np.nan)
        # FINRA shares_short is denominated in PRE-split shares. If a split
        # landed after the settlement date, rebase the numerator too or DTC is
        # off by the split factor (BYND: 33x).
        _post = [d for d in m["split_dates"] if s.get("settle") and d > s["settle"]]
        _f = m["split_factor"] if _post else 1.0
        dtc_live = ((short / _f) / adv) if (adv and adv > 0 and short == short) else np.nan
        rows.append({"ticker": t, **m, "settle": s.get("settle"), "split_adj": bool(m["split_dates"]),
                     "dtc_finra": s.get("dtc_finra", np.nan),
                     "dtc_live": dtc_live, "si_chg": s.get("si_chg", np.nan),
                     "fee": np.nan, "avail": np.nan,
                     "stale_h": np.nan})

    if not rows:
        sys.exit("  no rows -- check raw_bars coverage and --si-db path")
    df = pd.DataFrame(rows)

    # ---- IGNITION: self-normalized, multiplicative, absolute ----------------
    denom = df["vol20d"] * np.sqrt(3.0)
    df["thrust_z"] = np.where(denom > 0, df["ret_3d"] / denom, np.nan)
    t = np.clip(df["thrust_z"].fillna(0) / THRUST_Z_FULL, 0, 1)
    v = np.clip((df["rvol"].fillna(0) - RVOL_MIN) / (RVOL_FULL - RVOL_MIN), 0, 1)
    df["ignition"] = 100.0 * t * v

    # ---- Stage 1 fuel (local only) → decides who gets probed ----------------
    df["fuel"] = 100.0 * pct_rank(df["dtc_live"]).fillna(0)

    # ---- Stage 2: live borrow fee ------------------------------------------
    n_probe = len(df) if a.probe < 0 else min(a.probe, len(df))
    if _focus:
        n_probe = len(_focus)
    probed = []
    if n_probe:
        # Probe by PROVISIONAL SCORE, not fuel. The board ranks on score, so
        # selecting probes on fuel systematically skipped the igniting names:
        # RZLV (2026-08-26) scored #2 overall on ignition 100 but sat ~131st by
        # fuel (69th pct), so --probe 60 left its fee blank on the one row that
        # most needed it.
        _prov = df["fuel"] * (IGNITION_FLOOR + (1 - IGNITION_FLOOR) * df["ignition"] / 100.0)
        order = (_focus if _focus else
                 df.assign(_prov=_prov).sort_values("_prov", ascending=False)
                   ["ticker"].head(n_probe).tolist())
        order = [t for t in order if t in set(df["ticker"])]
        # NOT "live": the UW feed updates 05:00-11:23 ET only, then
        # stops for the day. See the freshness block below.
        print(f"  probing borrow fee for {len(order)} names "
              f"(UW feed updates 05:00-11:23 ET only) ...")
        for i, t in enumerate(order, 1):
            try:
                p = probe_borrow(t)
            except Exception:
                p = None
            if p:
                df.loc[df["ticker"] == t, "fee"] = p["fee"]
                df.loc[df["ticker"] == t, "avail"] = p["avail"]
                # stale_h too, or the freshness block below has nothing
                # to read. probe_borrow returns it; only fee and avail
                # were being copied across (found 2026-09-01).
                df.loc[df["ticker"] == t, "stale_h"] = p.get("stale_h")
                probed.append({"ticker": t, **p})
            if i % 50 == 0:
                print(f"    ... {i}/{len(order)}")
            time.sleep(UW_PAUSE)
        if probed:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                log_borrow_live(borrow_db, probed, ts)
                print(f"  logged {len(probed)} rows -> borrow.db:borrow_live @ {ts}")
            except Exception as e:
                print(f"  borrow_live LOG FAILED: {e}")
        # fold fee into fuel only if it covers the whole displayed population
        if len(probed) == len(df) and not _focus:
            df["fuel"] = 100.0 * (pct_rank(df["dtc_live"]).fillna(0)
                                  + pct_rank(df["fee"]).fillna(0)) / 2.0
            fuel_basis = "DTC + live borrow fee"
        else:
            fuel_basis = f"DTC only (fee probed on {len(probed)}/{len(df)}, not folded in)"
    else:
        fuel_basis = "DTC only (--probe 0, no API calls)"

    # Suppress rows with an unexplained price discontinuity: rvol / DTC_live /
    # ignition are all corrupt when the bars straddle an unrecorded split.
    _susp = df["suspect_date"].notna()
    if _susp.any():
        df.loc[_susp, ["dtc_live", "rvol", "ret_3d", "ignition"]] = np.nan
    df["ignition"] = df["ignition"].fillna(0.0)
    n_ranked = len(df)
    _ign_hi  = int((df["ignition"] >= 50).sum())
    _ign_mid = int(((df["ignition"] >= 10) & (df["ignition"] < 50)).sum())
    _ign_lo  = int((df["ignition"] < 10).sum())
    df["score"] = df["fuel"] * (IGNITION_FLOOR + (1 - IGNITION_FLOOR) * df["ignition"] / 100.0)
    df = df.sort_values("score", ascending=False)
    if _focus:
        df = df[df["ticker"].isin(_focus)]
    show = df[df["score"] >= a.min_score].head(a.top)

    def tag(r):
        # Absolute thresholds -- meaningful only because ignition is now
        # self-normalized. Under the old percentile version everything at the
        # top of the list read IGNITING by construction.
        if r["ignition"] >= 50 and r["fuel"] >= 60: return "IGNITING"
        if r["fuel"] >= 60 and r["ignition"] < 10:  return "LOADED-cold"
        if r["fuel"] >= 60:                          return "LOADED"
        if r["ignition"] >= 50:                      return "moving-nofuel"
        return "watch"

    L = []
    L.append("=" * 104)
    L.append(f"  SQUEEZE RADAR -- FUEL x IGNITION   "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    _scope = (f"  ranked vs {n_ranked} universe, showing {len(df)} selected"
              if _focus else
              f"  universe {n_ranked} scored ({n_etf} ETFs excluded of {n_raw})")
    L.append(_scope + f"  |  bars as-of {df['bar_date'].mode().iat[0]}"
                      f"  |  FINRA settle {df['settle'].mode().iat[0]}")
    L.append(f"  FUEL basis: {fuel_basis}")
    L.append(f"  IGNITION across universe: >=50 {_ign_hi}"
             f"  |  10-50 {_ign_mid}"
             f"  |  <10 {_ign_lo}"
             f"   (0 everywhere = nothing is igniting today; that is a valid answer)")
    L.append("  SCREEN ONLY -- hand-set weights, never gated. Not a buy signal, not a brick.")
    L.append("  SI %-of-float withheld: no honest float source (UW total_float = outstanding).")
    L.append("=" * 104)
    # BORROW FRESHNESS. The radar exists to catch borrow drying up, so
    # serving yesterday's availability without saying so is the worst
    # failure it can have. On 2026-09-01 it showed RZLV at 3,500,000
    # while the true figure that morning was 0 -- the number was 20
    # hours old and nothing said so.
    #
    # The age comes from the "stale_h" COLUMN of df (probe_borrow now
    # returns it). An earlier version of this patch read _fees, which is
    # a pandas Series of floats, not a dict of probe results -- it would
    # have found nothing even where it did not crash outright.
    if "stale_h" in df.columns:
        _ages = df["stale_h"].dropna()
        if len(_ages):
            _newest = float(_ages.min())
            if _newest >= 18:
                L.append(f"  !! BORROW DATA IS {_newest:.1f}h OLD -- the "
                         f"UW feed has produced nothing for today.")
                L.append("     AVAIL and FEE% below are STALE. Borrow can "
                         "vanish overnight; treat every availability "
                         "figure here as unverified.")
            elif _newest >= 4:
                L.append(f"  ! borrow data {_newest:.1f}h old (UW feed "
                         f"stops 11:23 ET; run before then for "
                         f"~30-min-old data)")
            else:
                L.append(f"  borrow data {_newest*60:.0f} min old")

    _fees = df["fee"].dropna()
    if len(_fees):
        _tc = _fees.map(fee_tier).value_counts()
        L.append(f"  BORROW TIERS (of {len(_fees)} probed): "
                 f"EXTREME {_tc.get('EXTREME',0)}  HIGH {_tc.get('HIGH',0)}  "
                 f"MODERATE {_tc.get('MODERATE',0)}  easy {_tc.get('easy',0)}"
                 f"   [>=20% / >=5% / >=1% / <1%]")
    L.append(f"  {'#':>3} {'TICKER':<7}{'SCORE':>6}{'FUEL':>6}{'IGN':>5}"
             f"{'DTC_live':>9}{'DTC_fin':>8}{'SI_chg':>7}{'FEE%':>7} {'TIER':<9}"
             f"{'AVAIL':>11}{'3d%':>8}{'rvol':>6}  STATE")
    L.append("  " + "-" * 100)
    for i, (_, r) in enumerate(show.iterrows(), 1):
        # g() right-justifies the "n/a" case too. The old lambda returned a bare
        # 3-char "n/a" with no padding, so every missing value shifted the rest
        # of the line left -- visible as "8%n/a" in the 2026-08-26 --probe 0 run.
        def g(v, fmt, w):
            return (fmt % v).rjust(w) if v == v else "n/a".rjust(w)
        _avail = (f"{int(r['avail']):,}" if r["avail"] == r["avail"] else "n/a").rjust(11)
        _sic = (f"{r['si_chg']*100:+.0f}%" if r["si_chg"] == r["si_chg"] else "n/a").rjust(7)
        _r3d = r["ret_3d"] * 100 if r["ret_3d"] == r["ret_3d"] else np.nan
        L.append(f"  {i:>3} {r['ticker']:<7}{r['score']:>6.0f}{r['fuel']:>6.0f}"
                 f"{r['ignition']:>5.0f}{g(r['dtc_live'],'%.2f',9)}{g(r['dtc_finra'],'%.2f',8)}"
                 f"{_sic}{g(r['fee'],'%.2f',7)} {fee_tier(r['fee']):<9}"
                 f"{_avail}{g(_r3d,'%+.1f%%',8)}{g(r['rvol'],'%.2f',6)}  {tag(r)}"
                 + ("  RAMP" if r["ramp"] else "") + ("  TOP" if r["top"] else "")
                 + (f"  SPLIT-adj x{r['split_factor']:.2f}"
                    if r["split_adj"] and abs(r["split_factor"] - 1.0) >= 0.10 else "")
                 + (f"  !! SUSPECT GAP {r['suspect_date']} x{r['suspect_ratio']:.1f}"
                    " -- unrecorded split? row suppressed" if r["suspect_date"] else ""))
    L.append("  " + "-" * 100)
    L.append(f"  DTC_live = shares_short / trailing-20d ADV (raw_bars). DTC_fin = FINRA as-published.")
    L.append(f"  scan {time.time()-t0:.0f}s")
    L.append("=" * 104)
    out = "\n".join(L)
    print(out)

    if a.save:
        os.makedirs(os.path.join(a.root, "logs"), exist_ok=True)
        fn = os.path.join(a.root, "logs",
                          f"squeeze_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
        open(fn, "w").write(out + "\n")
        print(f"  saved -> {fn}")


if __name__ == "__main__":
    main()
