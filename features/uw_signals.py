# features/uw_signals.py
# Unusual Whales signal fetchers:
#   1. Options flow alerts — unusual activity per ticker
#   2. Market tide — market-wide call/put flow
#   3. OI change — open interest change as GEX proxy
#   4. Institutional ownership — quarterly holdings change

import os
import sqlite3
import requests
from datetime import date, datetime
from pathlib import Path

UW_KEY   = os.getenv("UW_API_KEY", "")
HEADERS  = {"Authorization": f"Bearer {UW_KEY}"}
BASE_URL = "https://api.unusualwhales.com"
DB_PATH  = Path(__file__).parent.parent / "accuracy.db"


# ══════════════════════════════════════════════════════════════════════
# 1. OPTIONS FLOW ALERTS
# ══════════════════════════════════════════════════════════════════════

def get_options_flow_score(ticker: str) -> dict:
    """
    Compute options flow score for a ticker from UW flow alerts.
    Positive score = bullish unusual activity.
    Negative score = bearish unusual activity.
    """
    result = {
        "ticker":          ticker,
        "flow_score":      0.0,
        "bullish_prem":    0.0,
        "bearish_prem":    0.0,
        "sweep_count":     0,
        "alert_count":     0,
        "flow_signal":     "NEUTRAL",
        "error":           None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/option-trades/flow-alerts",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        alerts = [
            a for a in r.json().get("data", [])
            if a.get("ticker", "").upper() == ticker.upper()
        ]

        if not alerts:
            return result

        bullish_prem = 0.0
        bearish_prem = 0.0
        sweep_count  = 0

        for a in alerts:
            prem = float(a.get("total_premium", 0) or 0)
            typ  = a.get("type", "").lower()
            if typ == "call":
                bullish_prem += prem
            elif typ == "put":
                bearish_prem += prem
            if a.get("has_sweep"):
                sweep_count += 1

        total = bullish_prem + bearish_prem
        if total > 0:
            flow_score = (bullish_prem - bearish_prem) / total
        else:
            flow_score = 0.0

        result["flow_score"]   = round(flow_score, 4)
        result["bullish_prem"] = bullish_prem
        result["bearish_prem"] = bearish_prem
        result["sweep_count"]  = sweep_count
        result["alert_count"]  = len(alerts)

        if flow_score > 0.3:
            result["flow_signal"] = "BULLISH"
        elif flow_score < -0.3:
            result["flow_signal"] = "BEARISH"
        else:
            result["flow_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def flow_score_to_multiplier(flow_score: float) -> float:
    """
    Convert options flow score to prob_eff multiplier.
    Strong bullish flow → boost. Strong bearish → suppress.
    """
    if flow_score > 0.5:   return 1.08
    if flow_score > 0.3:   return 1.04
    if flow_score < -0.5:  return 0.92
    if flow_score < -0.3:  return 0.96
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. MARKET TIDE
# ══════════════════════════════════════════════════════════════════════

def get_market_tide() -> dict:
    """
    Get today's market tide — net call vs put premium across entire market.
    Positive = bullish market flow. Negative = bearish.
    """
    result = {
        "tide_score":       0.0,
        "net_call_premium": 0.0,
        "net_put_premium":  0.0,
        "net_volume":       0,
        "tide_signal":      "NEUTRAL",
        "error":            None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/market/market-tide",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = r.json().get("data", [])
        if not data:
            result["error"] = "No market tide data"
            return result

        # Sum all intraday buckets for today
        today = str(date.today())
        today_data = [d for d in data if d.get("date", "") == today]
        if not today_data:
            today_data = data  # fallback to all data

        net_call = sum(float(d.get("net_call_premium", 0) or 0) for d in today_data)
        net_put  = sum(float(d.get("net_put_premium",  0) or 0) for d in today_data)
        net_vol  = sum(int(d.get("net_volume", 0) or 0)         for d in today_data)

        total = abs(net_call) + abs(net_put)
        tide_score = (net_call + net_put) / total if total > 0 else 0.0

        result["tide_score"]       = round(tide_score, 4)
        result["net_call_premium"] = net_call
        result["net_put_premium"]  = net_put
        result["net_volume"]       = net_vol

        if tide_score > 0.2:
            result["tide_signal"] = "BULLISH"
        elif tide_score < -0.2:
            result["tide_signal"] = "BEARISH"
        else:
            result["tide_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def tide_score_to_multiplier(tide_score: float) -> float:
    """
    Convert market tide score to multiplier.
    Strong bullish tide → mild boost across all tickers.
    Strong bearish tide → mild suppression.
    """
    if tide_score > 0.4:  return 1.05
    if tide_score > 0.2:  return 1.02
    if tide_score < -0.4: return 0.95
    if tide_score < -0.2: return 0.98
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. OI CHANGE (GEX PROXY)
# ══════════════════════════════════════════════════════════════════════

def get_oi_change_score(ticker: str) -> dict:
    """
    Get open interest change score for a ticker.
    Rising OI + rising price = strong conviction = bullish signal.
    Rising OI + falling price = bearish signal.
    """
    result = {
        "ticker":      ticker,
        "oi_score":    0.0,
        "oi_change":   0.0,
        "oi_signal":   "NEUTRAL",
        "error":       None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/market/oi-change",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = [
            d for d in r.json().get("data", [])
            if d.get("underlying_symbol", "").upper() == ticker.upper()
        ]

        if not data:
            return result

        # Aggregate OI changes for this ticker
        total_oi_change = sum(
            float(d.get("oi_change", 0) or 0) for d in data
        )
        days_increasing = max(
            int(d.get("days_of_oi_increases", 0) or 0) for d in data
        )

        # Normalize: more days increasing = stronger signal
        oi_score = min(days_increasing / 5.0, 1.0) * (1 if total_oi_change > 0 else -1)

        result["oi_score"]  = round(oi_score, 4)
        result["oi_change"] = round(total_oi_change, 2)

        if oi_score > 0.4:
            result["oi_signal"] = "BULLISH"
        elif oi_score < -0.4:
            result["oi_signal"] = "BEARISH"
        else:
            result["oi_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def oi_score_to_multiplier(oi_score: float) -> float:
    if oi_score > 0.6:  return 1.04
    if oi_score > 0.4:  return 1.02
    if oi_score < -0.6: return 0.96
    if oi_score < -0.4: return 0.98
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 4. INSTITUTIONAL OWNERSHIP
# ══════════════════════════════════════════════════════════════════════

def get_institutional_score(ticker: str) -> dict:
    """
    Get net institutional buying/selling from 13F filings.
    Positive = net buying. Negative = net selling.
    Note: 45-day lag — use as slow-moving background signal.
    """
    result = {
        "ticker":         ticker,
        "inst_score":     0.0,
        "units_changed":  0,
        "inst_signal":    "NEUTRAL",
        "filing_date":    None,
        "error":          None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/institution/{ticker}/ownership",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = r.json().get("data", [])
        if not data:
            result["error"] = "No institutional data"
            return result

        # Sum units_changed across top institutions
        total_changed = sum(int(d.get("units_changed", 0) or 0) for d in data[:20])
        shares_out    = float(data[0].get("shares_outstanding", 1) or 1)
        latest_filing = data[0].get("filing_date", None)

        # Normalize by shares outstanding
        inst_score = total_changed / shares_out * 1000  # scale up

        result["inst_score"]    = round(min(max(inst_score, -1.0), 1.0), 4)
        result["units_changed"] = total_changed
        result["filing_date"]   = latest_filing

        if inst_score > 0.1:
            result["inst_signal"] = "BULLISH"
        elif inst_score < -0.1:
            result["inst_signal"] = "BEARISH"
        else:
            result["inst_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def inst_score_to_multiplier(inst_score: float) -> float:
    """
    Institutional score is slow-moving — use very mild multiplier.
    """
    if inst_score > 0.5:  return 1.03
    if inst_score > 0.1:  return 1.01
    if inst_score < -0.5: return 0.97
    if inst_score < -0.1: return 0.99
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# COMBINED SIGNAL
# ══════════════════════════════════════════════════════════════════════

# Module-level cache for market-wide signals (fetched once per run)
_MW_CACHE = {"tide": None, "flow_alerts": None, "oi": None, "ts": None}


def _get_marketwide_cached() -> dict:
    """Fetch market-wide signals once and cache for 5 minutes."""
    from datetime import datetime
    now = datetime.now()
    if _MW_CACHE["ts"] is not None:
        age_mins = (now - _MW_CACHE["ts"]).total_seconds() / 60
        if age_mins < 5:
            return _MW_CACHE

    # Refresh cache
    _MW_CACHE["tide"]        = get_market_tide()
    _MW_CACHE["flow_alerts"] = _get_flow_alerts_raw()
    _MW_CACHE["oi"]          = _get_oi_raw()
    _MW_CACHE["ts"]          = now
    return _MW_CACHE


def _get_flow_alerts_raw() -> list:
    """Fetch raw flow alerts once."""
    try:
        r = requests.get(f"{BASE_URL}/api/option-trades/flow-alerts",
                         headers=HEADERS, timeout=10)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return []


def _get_oi_raw() -> list:
    """Fetch raw OI change data once."""
    try:
        r = requests.get(f"{BASE_URL}/api/market/oi-change",
                         headers=HEADERS, timeout=10)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return []


def get_combined_uw_multiplier(ticker: str) -> dict:
    """
    Fetch all 4 UW signals and return combined multiplier.
    Market-wide signals (tide, flow, OI) fetched ONCE and cached.
    Institutional reads from DB. No per-ticker API calls for market-wide data.
    """
    # Get market-wide signals from cache (1 API call total, not 126)
    mw    = _get_marketwide_cached()
    tide  = mw["tide"]

    # Compute per-ticker flow score from cached alerts
    alerts    = mw["flow_alerts"]
    t_alerts  = [a for a in alerts if a.get("ticker","").upper() == ticker.upper()]
    bull_prem = sum(float(a.get("total_premium",0) or 0) for a in t_alerts if a.get("type","").lower()=="call")
    bear_prem = sum(float(a.get("total_premium",0) or 0) for a in t_alerts if a.get("type","").lower()=="put")
    total_p   = bull_prem + bear_prem
    flow_score= round((bull_prem - bear_prem) / total_p, 4) if total_p > 0 else 0.0
    flow      = {
        "flow_score":   flow_score,
        "flow_signal":  "BULLISH" if flow_score > 0.3 else "BEARISH" if flow_score < -0.3 else "NEUTRAL",
        "alert_count":  len(t_alerts),
    }

    # Compute per-ticker OI score from cached data
    oi_data   = mw["oi"]
    t_oi      = [d for d in oi_data if d.get("underlying_symbol","").upper() == ticker.upper()]
    if t_oi:
        oi_chg    = sum(float(d.get("oi_change",0) or 0) for d in t_oi)
        days_inc  = max(int(d.get("days_of_oi_increases",0) or 0) for d in t_oi)
        oi_score  = min(days_inc / 5.0, 1.0) * (1 if oi_chg > 0 else -1)
    else:
        oi_score  = 0.0
    oi = {
        "oi_score":  round(oi_score, 4),
        "oi_signal": "BULLISH" if oi_score > 0.4 else "BEARISH" if oi_score < -0.4 else "NEUTRAL",
    }

    # Institutional from DB (no API call)
    inst  = get_institutional_score(ticker)

    flow_mult = flow_score_to_multiplier(flow["flow_score"])
    tide_mult = tide_score_to_multiplier(tide["tide_score"])
    oi_mult   = oi_score_to_multiplier(oi["oi_score"])
    inst_mult = inst_score_to_multiplier(inst["inst_score"])

    combined  = round(flow_mult * tide_mult * oi_mult * inst_mult, 4)
    combined  = round(min(max(combined, 0.80), 1.20), 4)

    return {
        "ticker":      ticker,
        "combined":    combined,
        "flow_mult":   flow_mult,
        "tide_mult":   tide_mult,
        "oi_mult":     oi_mult,
        "inst_mult":   inst_mult,
        "flow_score":  flow["flow_score"],
        "flow_signal": flow["flow_signal"],
        "tide_score":  tide["tide_score"],
        "tide_signal": tide["tide_signal"],
        "oi_score":    oi["oi_score"],
        "oi_signal":   oi["oi_signal"],
        "inst_score":  inst["inst_score"],
        "inst_signal": inst["inst_signal"],
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\nUW Signals for {ticker}:")
    result = get_combined_uw_multiplier(ticker)
    for k, v in result.items():
        print(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED UW SIGNALS — SHORT INTEREST, LIT FLOW, EXPIRY, NET IMPACT
# ══════════════════════════════════════════════════════════════════════════════

def get_short_interest_score(ticker: str) -> dict:
    """Short interest % float + days to cover — reads from DB cache first."""
    result = {"si_float": 0.0, "days_to_cover": 0.0,
              "si_signal": "NEUTRAL", "error": None}
    try:
        from datetime import timedelta
        cutoff = str(date.today() - timedelta(days=14))
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT si_float, days_to_cover, si_signal
                FROM short_interest_cache WHERE ticker=? AND market_date>=?
                ORDER BY market_date DESC LIMIT 1
            """, (ticker, cutoff)).fetchone()
        if row:
            result["si_float"]      = row[0] or 0.0
            result["days_to_cover"] = row[1] or 0.0
            result["si_signal"]     = row[2] or "NEUTRAL"
            return result
    except Exception:
        pass
    try:
        r = requests.get(f"{BASE_URL}/api/shorts/{ticker}/interest-float/v2",
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data = r.json().get("data", [])
        if not data:
            return result
        latest    = data[0]
        si_float  = float(latest.get("si_float", 0) or 0)
        dtc       = float(latest.get("days_to_cover", 0) or 0)
        result["si_float"]     = round(si_float, 4)
        result["days_to_cover"]= round(dtc, 2)
        # High SI + high DTC = squeeze potential = BULLISH
        # Moderate SI = mild bearish (crowded short)
        if si_float > 0.20 and dtc > 5:
            result["si_signal"] = "SQUEEZE"
        elif si_float > 0.10:
            result["si_signal"] = "HIGH_SHORT"
        elif si_float < 0.02:
            result["si_signal"] = "LOW_SHORT"
        else:
            result["si_signal"] = "NEUTRAL"
    except Exception as e:
        result["error"] = str(e)
    return result


def si_score_to_multiplier(si_float: float, days_to_cover: float) -> float:
    """Squeeze potential boosts BUY, heavy shorting suppresses."""
    if si_float > 0.20 and days_to_cover > 5:  return 1.06  # squeeze
    if si_float > 0.15:                          return 0.96  # crowded short
    if si_float > 0.10:                          return 0.98  # elevated short
    return 1.0


def get_top_net_impact_score(ticker: str) -> dict:
    """Check if ticker appears in top net options premium impact list."""
    result = {"net_premium": 0.0, "in_top_impact": False, "error": None}
    try:
        r = requests.get(f"{BASE_URL}/api/market/top-net-impact",
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data = r.json().get("data", [])
        match = [d for d in data if d.get("ticker","").upper() == ticker.upper()]
        if match:
            net = float(match[0].get("net_premium", 0) or 0)
            result["net_premium"]   = net
            result["in_top_impact"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


def get_lit_flow_score(ticker: str) -> dict:
    """Lit exchange large trade flow — complement to dark pool."""
    result = {"lit_ratio": 0.0, "lit_signal": "NEUTRAL", "error": None}
    try:
        r = requests.get(f"{BASE_URL}/api/lit-flow/{ticker}",
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data = r.json().get("data", [])
        if not data:
            return result
        # Ratio of lit to total volume
        total  = sum(float(d.get("volume", 0) or 0) for d in data[:20])
        if total > 0:
            result["lit_ratio"] = round(min(total / 1e7, 1.0), 4)
            if result["lit_ratio"] > 0.6:
                result["lit_signal"] = "HIGH"
    except Exception as e:
        result["error"] = str(e)
    return result


def get_expiry_breakdown_score(ticker: str) -> dict:
    """Options expiry concentration — high OI near current expiry = pinning."""
    result = {"near_expiry_oi": 0, "pinning_risk": False, "error": None}
    try:
        r = requests.get(f"{BASE_URL}/api/stock/{ticker}/expiry-breakdown",
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data = r.json().get("data", [])
        if not data:
            return result
        # First expiry = nearest — high OI = pinning risk
        nearest = data[0]
        oi      = int(nearest.get("open_interest", 0) or 0)
        result["near_expiry_oi"] = oi
        result["pinning_risk"]   = oi > 50000
    except Exception as e:
        result["error"] = str(e)
    return result


def _get_si_cached() -> dict:
    """Cache short interest for all tickers — fetched once per session."""
    return {}  # Will be populated per-ticker on demand


def get_extended_uw_multiplier(ticker: str) -> dict:
    """
    Extended UW multiplier combining short interest + lit flow + expiry.
    Called per-ticker — SI and expiry are per-ticker endpoints.
    """
    si      = get_short_interest_score(ticker)
    expiry  = get_expiry_breakdown_score(ticker)

    si_mult     = si_score_to_multiplier(si["si_float"], si["days_to_cover"])
    expiry_mult = 0.97 if expiry["pinning_risk"] else 1.0

    combined = round(min(max(si_mult * expiry_mult, 0.85), 1.15), 4)

    return {
        "ticker":        ticker,
        "combined":      combined,
        "si_mult":       si_mult,
        "expiry_mult":   expiry_mult,
        "si_float":      si["si_float"],
        "days_to_cover": si["days_to_cover"],
        "si_signal":     si["si_signal"],
        "pinning_risk":  expiry["pinning_risk"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 2 — MODEL FEATURES (needs retrain)
# Seasonality, Analyst ratings, FTDs
# ══════════════════════════════════════════════════════════════════════════════

def get_seasonality_features(ticker: str) -> dict:
    """
    Monthly seasonality features — reads from DB cache first.
    Falls back to live UW API if DB empty.
    """
    result = {
        "seasonal_avg_return":    0.0,
        "seasonal_positive_pct":  0.5,
        "seasonal_signal":        "NEUTRAL",
        "error":                  None,
    }
    month = date.today().month
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT avg_change, positive_months_perc, seasonal_signal
                FROM seasonality_cache WHERE ticker=? AND month=?
            """, (ticker, month)).fetchone()
        if row:
            result["seasonal_avg_return"]   = row[0] or 0.0
            result["seasonal_positive_pct"] = row[1] or 0.5
            result["seasonal_signal"]       = row[2] or "NEUTRAL"
            return result
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BASE_URL}/api/seasonality/{ticker}/monthly",
            headers=HEADERS, timeout=8
        )
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data  = r.json().get("data", [])
        month = date.today().month
        match = [d for d in data if int(d.get("month", 0)) == month]
        if match:
            m = match[0]
            avg_ret  = float(m.get("avg_change", 0) or 0)
            pos_pct  = float(m.get("positive_months_perc", 0.5) or 0.5)
            result["seasonal_avg_return"]   = round(avg_ret, 4)
            result["seasonal_positive_pct"] = round(pos_pct, 4)
            if avg_ret > 0.02 and pos_pct > 0.6:
                result["seasonal_signal"] = "BULLISH"
            elif avg_ret < -0.02 and pos_pct < 0.4:
                result["seasonal_signal"] = "BEARISH"
    except Exception as e:
        result["error"] = str(e)
    return result


def get_analyst_score(ticker: str) -> dict:
    """
    Analyst rating changes — reads from DB cache first.
    Falls back to live UW API if DB empty.
    """
    result = {
        "analyst_score":   0.0,
        "upgrades_30d":    0,
        "downgrades_30d":  0,
        "avg_target":      0.0,
        "analyst_signal":  "NEUTRAL",
        "error":           None,
    }
    try:
        from datetime import timedelta
        cutoff = str(date.today() - timedelta(days=7))
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT analyst_score, upgrades_30d, downgrades_30d,
                       avg_target, analyst_signal
                FROM analyst_cache WHERE ticker=? AND date>=?
                ORDER BY date DESC LIMIT 1
            """, (ticker, cutoff)).fetchone()
        if row:
            result["analyst_score"]   = row[0] or 0.0
            result["upgrades_30d"]    = row[1] or 0
            result["downgrades_30d"]  = row[2] or 0
            result["avg_target"]      = row[3] or 0.0
            result["analyst_signal"]  = row[4] or "NEUTRAL"
            return result
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BASE_URL}/api/screener/analysts",
            headers=HEADERS,
            params={"ticker": ticker},
            timeout=8
        )
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data     = r.json().get("data", [])
        if not data:
            return result

        from datetime import datetime, timedelta
        cutoff   = (datetime.now() - timedelta(days=30)).isoformat()
        recent   = [d for d in data if d.get("timestamp","") >= cutoff]

        upgrades   = sum(1 for d in recent if d.get("action","").lower() in
                        ("upgrade","initiated","reiterated") and
                        d.get("recommendation","").lower() in ("buy","strong buy","outperform"))
        downgrades = sum(1 for d in recent if d.get("action","").lower() == "downgrade")

        targets = [float(d["target"]) for d in recent if d.get("target")]
        avg_tgt = sum(targets) / len(targets) if targets else 0.0

        score = (upgrades - downgrades) / max(len(recent), 1)
        result["analyst_score"]   = round(score, 4)
        result["upgrades_30d"]    = upgrades
        result["downgrades_30d"]  = downgrades
        result["avg_target"]      = round(avg_tgt, 2)

        if score > 0.2:
            result["analyst_signal"] = "BULLISH"
        elif score < -0.2:
            result["analyst_signal"] = "BEARISH"

    except Exception as e:
        result["error"] = str(e)
    return result


def get_ftd_score(ticker: str) -> dict:
    """
    Failures to deliver — reads from DB cache first.
    Falls back to live UW API if DB empty.
    """
    result = {
        "ftd_shares":  0,
        "ftd_signal":  "NEUTRAL",
        "error":       None,
    }
    try:
        from datetime import timedelta
        cutoff = str(date.today() - timedelta(days=7))
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT ftd_shares, ftd_signal FROM ftd_cache
                WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1
            """, (ticker, cutoff)).fetchone()
        if row:
            result["ftd_shares"] = row[0] or 0
            result["ftd_signal"] = row[1] or "NEUTRAL"
            return result
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BASE_URL}/api/shorts/{ticker}/ftds",
            headers=HEADERS, timeout=8
        )
        if r.status_code != 200:
            result["error"] = f"UW {r.status_code}"
            return result
        data = r.json().get("data", [])
        if not data:
            return result

        # Sum last 5 days FTDs
        recent_ftds = sum(int(d.get("quantity", 0) or 0) for d in data[:5])
        result["ftd_shares"] = recent_ftds

        if recent_ftds > 500_000:
            result["ftd_signal"] = "HIGH"
        elif recent_ftds > 100_000:
            result["ftd_signal"] = "ELEVATED"

    except Exception as e:
        result["error"] = str(e)
    return result
