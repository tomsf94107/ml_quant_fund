#!/usr/bin/env python3
# data/alpha_sources.py
# ─────────────────────────────────────────────────────────────────────────────
# Free alpha data sources to improve model accuracy:
#
#   1. Analyst estimate revisions  (yfinance — free, no key needed)
#   2. 13F institutional filings   (SEC EDGAR — free, no key needed)
#   3. Earnings call NLP           (SEC EDGAR 8-K — free, no key needed)
#   4. Unusual Whales options flow (API — $30/month)
#
# Each returns a multiplier (0.85-1.15) for the signal generator,
# AND a feature dict for model training.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/cache/alpha")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. ANALYST ESTIMATE REVISIONS (free via yfinance)
# ══════════════════════════════════════════════════════════════════════════════

def get_analyst_revisions(ticker: str) -> dict:
    """
    Fetch analyst price target revisions and recommendations via yfinance.

    What this captures:
    - Price target upgrades/downgrades (strong forward signal)
    - Buy/Sell/Hold ratio changes
    - EPS estimate revisions (most predictive)

    Returns multiplier + features dict.
    """
    import yfinance as yf

    result = {
        "ticker":              ticker,
        "analyst_multiplier":  1.0,
        "n_buy":               0,
        "n_hold":              0,
        "n_sell":              0,
        "buy_pct":             0.5,
        "mean_target":         None,
        "target_upside":       None,
        "recent_upgrade":      False,
        "recent_downgrade":    False,
        "error":               None,
    }

    try:
        tkr  = yf.Ticker(ticker)
        info = tkr.info

        # Current price
        price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Analyst price targets
        target_mean = info.get("targetMeanPrice")
        target_low  = info.get("targetLowPrice")
        target_high = info.get("targetHighPrice")

        if price and target_mean:
            upside = (target_mean / price) - 1.0
            result["mean_target"]   = round(float(target_mean), 2)
            result["target_upside"] = round(float(upside), 4)

        # Recommendation counts
        n_buy    = (info.get("numberOfAnalystOpinions") or 0)
        rec      = info.get("recommendationMean")  # 1=Strong Buy, 5=Sell

        if rec is not None:
            # Convert 1-5 scale to buy_pct
            buy_pct = max(0, min(1, (5 - float(rec)) / 4))
            result["buy_pct"] = round(buy_pct, 3)

        # Recent recommendations from recommendations_summary
        try:
            rec_df = tkr.recommendations_summary
            if rec_df is not None and not rec_df.empty:
                latest = rec_df.iloc[0]
                result["n_buy"]  = int(latest.get("strongBuy", 0) +
                                       latest.get("buy", 0))
                result["n_hold"] = int(latest.get("hold", 0))
                result["n_sell"] = int(latest.get("sell", 0) +
                                       latest.get("strongSell", 0))
        except Exception:
            pass

        # Recent upgrades/downgrades (last 30 days)
        try:
            upgrades = tkr.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                recent = upgrades[
                    upgrades.index >= (datetime.now() - timedelta(days=30))
                ]
                grades = recent.get("ToGrade", pd.Series()).str.lower()
                result["recent_upgrade"]   = any(
                    g in ["buy", "strong buy", "outperform", "overweight"]
                    for g in grades
                )
                result["recent_downgrade"] = any(
                    g in ["sell", "strong sell", "underperform", "underweight"]
                    for g in grades
                )
        except Exception:
            pass

        # ── Compute multiplier ────────────────────────────────────────────────
        mult = 1.0
        upside = result.get("target_upside", 0) or 0

        if upside > 0.20:     mult += 0.06   # >20% upside target → bullish
        elif upside > 0.10:   mult += 0.03
        elif upside < -0.05:  mult -= 0.05   # below current price → bearish

        if result["buy_pct"] > 0.70:  mult += 0.04
        elif result["buy_pct"] < 0.30: mult -= 0.04

        if result["recent_upgrade"]:   mult += 0.04
        if result["recent_downgrade"]: mult -= 0.05

        result["analyst_multiplier"] = round(max(0.80, min(1.15, mult)), 3)

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  2. 13F INSTITUTIONAL FILINGS (free via SEC EDGAR)
# ══════════════════════════════════════════════════════════════════════════════

# Known CIKs for major hedge funds (add more as needed)
HEDGE_FUND_CIKS = {
    "Citadel":       "0001423053",
    "Bridgewater":   "0001350694",
    "Renaissance":   "0001037389",
    "TwoSigma":      "0001450683",
    "Millennium":    "0001273931",
    "Point72":       "0001535778",
    "DE Shaw":       "0001009207",
}

SEC_HEADERS = {"User-Agent": "ml-quant-fund research@example.com"}


def get_institutional_sentiment(ticker: str, lookback_quarters: int = 2) -> dict:
    """
    Check recent 13F filings for institutional buying/selling of a ticker.

    SEC EDGAR 13F filings are public and free.
    Filed quarterly with 45-day lag (Q4 filed by Feb 15, etc.)

    Returns dict with institutional_multiplier and sentiment.
    """
    result = {
        "ticker":                   ticker,
        "institutional_multiplier": 1.0,
        "net_institutional":        "UNKNOWN",
        "n_funds_increased":        0,
        "n_funds_decreased":        0,
        "error":                    None,
    }

    # Check cache (13F data doesn't change often)
    cache_file = CACHE_DIR / f"13f_{ticker}.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    try:
        # Use SEC EDGAR full-text search for ticker in 13F filings
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
               f"&dateRange=custom&startdt="
               f"{(datetime.now()-timedelta(days=180)).strftime('%Y-%m-%d')}"
               f"&enddt={datetime.now().strftime('%Y-%m-%d')}"
               f"&forms=13F-HR")

        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            result["error"] = f"SEC API returned {resp.status_code}"
            return result

        data   = resp.json()
        hits   = data.get("hits", {}).get("hits", [])
        n_hits = len(hits)

        # More filings mentioning this ticker = more institutional interest
        if n_hits >= 20:
            result["net_institutional"]        = "BULLISH"
            result["institutional_multiplier"] = 1.05
            result["n_funds_increased"]        = n_hits
        elif n_hits >= 5:
            result["net_institutional"]        = "NEUTRAL"
            result["institutional_multiplier"] = 1.02
        else:
            result["net_institutional"]        = "LOW_INTEREST"
            result["institutional_multiplier"] = 1.0

        # Cache result
        cache_file.write_text(json.dumps(result))

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. EARNINGS CALL NLP (free via SEC EDGAR 8-K filings)
# ══════════════════════════════════════════════════════════════════════════════

def get_earnings_call_sentiment(ticker: str) -> dict:
    """
    Fetch most recent earnings release (8-K) from SEC EDGAR and run
    FinBERT sentiment on the text.

    What this captures:
    - Management tone (confident vs hedging language)
    - Guidance raised/lowered/withdrawn
    - Surprise language ("exceeded", "disappointed", "headwinds")
    """
    result = {
        "ticker":               ticker,
        "earnings_multiplier":  1.0,
        "sentiment_score":      0.0,
        "guidance":             "UNKNOWN",
        "filing_date":          None,
        "error":                None,
    }

    # Check cache
    cache_file = CACHE_DIR / f"earnings_nlp_{ticker}.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    try:
        # Step 1: Get company CIK from SEC EDGAR
        cik_url  = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=8-K&dateb=&owner=include&count=5&search_text=&action=getcompany&output=atom"
        resp     = requests.get(cik_url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            result["error"] = "CIK lookup failed"
            return result

        # Parse CIK from response
        import re
        cik_match = re.search(r'CIK=(\d+)', resp.text)
        if not cik_match:
            result["error"] = "CIK not found"
            return result

        cik = cik_match.group(1).zfill(10)

        # Step 2: Get most recent 8-K filing
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp2       = requests.get(filings_url, headers=SEC_HEADERS, timeout=10)
        if resp2.status_code != 200:
            result["error"] = "Filings lookup failed"
            return result

        filings = resp2.json()
        recent  = filings.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accns = recent.get("accessionNumber", [])

        # Find most recent 8-K
        for form, d, accn in zip(forms, dates, accns):
            if form == "8-K":
                result["filing_date"] = d

                # Step 3: Get filing text
                accn_clean = accn.replace("-", "")
                doc_url    = (f"https://www.sec.gov/Archives/edgar/full-index/"
                              f"2024/QTR1/{accn_clean}.txt")

                # Use EDGAR viewer instead (more reliable)
                viewer_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                              f"action=getcompany&CIK={cik}&type=8-K&dateb=&"
                              f"owner=include&count=1&search_text=")

                # Step 4: Run FinBERT on excerpt
                text_snippet = _extract_8k_text(cik, accn)
                if text_snippet:
                    score = _run_finbert(text_snippet)
                    result["sentiment_score"] = round(score, 4)

                    # Guidance keywords
                    text_lower = text_snippet.lower()
                    if any(w in text_lower for w in
                           ["raised guidance", "increased outlook", "raised our",
                            "exceeded expectations", "record revenue"]):
                        result["guidance"] = "RAISED"
                        result["earnings_multiplier"] = 1.10
                    elif any(w in text_lower for w in
                             ["withdrew guidance", "lowered guidance",
                              "reduced outlook", "challenging environment",
                              "below expectations"]):
                        result["guidance"] = "LOWERED"
                        result["earnings_multiplier"] = 0.85
                    else:
                        result["guidance"] = "MAINTAINED"
                        mult = 1.0 + (score * 0.08)
                        result["earnings_multiplier"] = round(
                            max(0.88, min(1.10, mult)), 3
                        )
                break

        cache_file.write_text(json.dumps(result))

    except Exception as e:
        result["error"] = str(e)

    return result


def _extract_8k_text(cik: str, accn: str, max_chars: int = 3000) -> Optional[str]:
    """Extract text from an 8-K filing."""
    try:
        accn_fmt = f"{accn[:10]}-{accn[10:12]}-{accn[12:]}"
        idx_url  = (f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accn}/{accn_fmt}-index.htm")
        resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        # Find the main document
        import re
        docs = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"',
                          resp.text)
        if not docs:
            return None

        doc_resp = requests.get(f"https://www.sec.gov{docs[0]}",
                                headers=SEC_HEADERS, timeout=10)
        if doc_resp.status_code != 200:
            return None

        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return None


def _run_finbert(text: str) -> float:
    """Run FinBERT on text, return score -1 to +1."""
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification",
                        model="ProsusAI/finbert",
                        truncation=True, max_length=512)
        result = pipe(text[:512])[0]
        label  = result["label"].lower()
        score  = result["score"]
        return score if label == "positive" else (-score if label == "negative" else 0.0)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  4. UNUSUAL WHALES OPTIONS FLOW ($30/month)
# ══════════════════════════════════════════════════════════════════════════════

def get_unusual_whales_flow(ticker: str) -> dict:
    """
    Fetch real options flow from Unusual Whales API.
    Requires API key ($30/month at unusualwhales.com)

    Set in .streamlit/secrets.toml:
        UNUSUAL_WHALES_KEY = "your_api_key"

    Returns dict with flow_signal and multiplier.
    """
    result = {
        "ticker":          ticker,
        "uw_multiplier":   1.0,
        "flow_signal":     "NEUTRAL",
        "bullish_premium": 0,
        "bearish_premium": 0,
        "net_flow":        0,
        "error":           None,
    }

    try:
        import streamlit as st
        api_key = st.secrets.get("UNUSUAL_WHALES_KEY",
                                  os.getenv("UNUSUAL_WHALES_KEY", ""))
    except Exception:
        api_key = os.getenv("UNUSUAL_WHALES_KEY", "")

    if not api_key:
        result["error"] = "No Unusual Whales API key configured"
        return result

    try:
        # Unusual Whales API v2
        url  = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-alerts"
        resp = requests.get(url,
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10)
        if resp.status_code != 200:
            result["error"] = f"API returned {resp.status_code}"
            return result

        data   = resp.json().get("data", [])
        today  = datetime.now().date().isoformat()

        bull_premium = 0
        bear_premium = 0

        for flow in data:
            if flow.get("date", "")[:10] != today:
                continue
            premium    = float(flow.get("premium", 0))
            sentiment  = flow.get("sentiment", "").lower()
            if sentiment == "bullish":
                bull_premium += premium
            elif sentiment == "bearish":
                bear_premium += premium

        total = bull_premium + bear_premium
        if total > 0:
            net = (bull_premium - bear_premium) / total
            result["bullish_premium"] = int(bull_premium)
            result["bearish_premium"] = int(bear_premium)
            result["net_flow"]        = round(net, 3)

            if net > 0.30:
                result["flow_signal"]   = "BULLISH"
                result["uw_multiplier"] = min(1.12, 1.0 + net * 0.20)
            elif net < -0.30:
                result["flow_signal"]   = "BEARISH"
                result["uw_multiplier"] = max(0.88, 1.0 + net * 0.15)

            result["uw_multiplier"] = round(result["uw_multiplier"], 3)

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED ALPHA MULTIPLIER
# ══════════════════════════════════════════════════════════════════════════════

def get_combined_alpha_multiplier(ticker: str) -> tuple[float, dict]:
    """
    Fetch all alpha sources and return combined multiplier.

    Returns (combined_multiplier, details_dict)
    Combined = analyst × institutional × earnings × options_flow
    Capped at [0.75, 1.20] to prevent extreme swings.
    """
    details = {}

    # Analyst revisions
    analyst = get_analyst_revisions(ticker)
    details["analyst"] = analyst
    m_analyst = analyst.get("analyst_multiplier", 1.0)

    # Institutional (13F)
    institutional = get_institutional_sentiment(ticker)
    details["institutional"] = institutional
    m_inst = institutional.get("institutional_multiplier", 1.0)

    # Earnings NLP
    earnings = get_earnings_call_sentiment(ticker)
    details["earnings"] = earnings
    m_earn = earnings.get("earnings_multiplier", 1.0)

    # Unusual Whales (if key available)
    uw = get_unusual_whales_flow(ticker)
    details["unusual_whales"] = uw
    m_uw = uw.get("uw_multiplier", 1.0)

    combined = m_analyst * m_inst * m_earn * m_uw
    combined = round(max(0.75, min(1.20, combined)), 3)

    return combined, details


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    tickers = ["AAPL", "NVDA", "TSLA"]
    print(f"\nAlpha Sources — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    print(f"{'─'*70}")

    for ticker in tickers:
        print(f"\n{ticker}:")

        analyst = get_analyst_revisions(ticker)
        print(f"  Analyst:  upside={analyst.get('target_upside', 'N/A')}  "
              f"buy_pct={analyst.get('buy_pct', 'N/A')}  "
              f"mult={analyst.get('analyst_multiplier', 1.0):.2f}x")

        inst = get_institutional_sentiment(ticker)
        print(f"  13F:      {inst.get('net_institutional', 'N/A')}  "
              f"mult={inst.get('institutional_multiplier', 1.0):.2f}x")

        combined, _ = get_combined_alpha_multiplier(ticker)
        print(f"  COMBINED: {combined:.3f}x")
