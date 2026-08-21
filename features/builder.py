# features/builder.py
# ─────────────────────────────────────────────────────────────────────────────
# THE canonical feature pipeline. One function. One output schema. No compat
# wrappers anywhere else in the codebase.
#
# Output schema (all lowercase, snake_case):
#   date, ticker, close, volume,
#   return_1d, return_3d, return_5d,
#   ma_5, ma_10, ma_20,
#   volatility_5d, volatility_10d,
#   rsi_14, macd, macd_signal,
#   bb_upper, bb_lower, bb_width,
#   volume_zscore, volume_spike,
#   vwap, obv, atr,
#   spy_ret, xlk_ret,
#   sentiment_score,          ← 0.0 placeholder until sentiment pipeline runs
#   insider_net_shares, insider_7d, insider_21d,
#   congress_net_shares,
#   risk_today, risk_next_1d, risk_next_3d, risk_prev_1d,
#   is_pandemic
#
# Rules:
#   - All optional signals (sentiment, insider, congress, risk) default to 0.0
#     silently. They NEVER crash the pipeline.
#   - yfinance MultiIndex columns are flattened immediately on download.
#   - Output always has a clean RangeIndex (not DatetimeIndex) so it plays
#     nicely with both sklearn and Streamlit dataframes.
#   - This file has zero Streamlit imports. Zero UI code. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sqlite3
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3
import yfinance as yf
from features import massive_client as mc

# ── Constants ─────────────────────────────────────────────────────────────────
SPY_TICKER      = "SPY"
SECTOR_ETF      = "XLK"
VIX_TICKER      = "^VIX"
VOL_LOOKBACK    = 20        # sessions for volume z-score

# Sector ETF map — stock → best matching sector ETF
# Complete sector map — all 157 tickers (universe + watchlist).
# Rebuilt 2026-05-28 (F2): previously ~30 mapped, rest defaulted to XLK (wrong).
# GOOG/META→XLC (comm svcs), HOOD/CRCL→XLF, D/NEE/OKLO→XLU, EQIX/DLR/RC→XLRE, etc.
SECTOR_ETF_MAP = {
    # ── S&P 500 expansion (May 30): 429 large-caps, true GICS sector ──
    "A":"XLV",
    "ABBV":"XLV",
    "ACGL":"XLF",
    "ACN":"XLK",
    "ADBE":"XLK",
    "ADI":"XLK",
    "ADM":"XLP",
    "ADP":"XLI",
    "AEE":"XLU",
    "AEP":"XLU",
    "AES":"XLU",
    "AFL":"XLF",
    "AIG":"XLF",
    "AIZ":"XLF",
    "AJG":"XLF",
    "AKAM":"XLK",
    "ALB":"XLB",
    "ALGN":"XLV",
    "ALL":"XLF",
    "ALLE":"XLI",
    "AMCR":"XLB",
    "AME":"XLI",
    "AMGN":"XLV",
    "AMP":"XLF",
    "AMT":"XLRE",
    "AON":"XLF",
    "AOS":"XLI",
    "APA":"XLE",
    "APH":"XLK",
    "APO":"XLF",
    "APP":"XLK",
    "APTV":"XLY",
    "ARE":"XLRE",
    "ARES":"XLF",
    "ATO":"XLU",
    "AVB":"XLRE",
    "AVY":"XLB",
    "AWK":"XLU",
    "AXON":"XLI",
    "AZO":"XLY",
    "BAC":"XLF",
    "BALL":"XLB",
    "BAX":"XLV",
    "BBY":"XLY",
    "BDX":"XLV",
    "BEN":"XLF",
    "BF-B":"XLP",
    "BG":"XLP",
    "BIIB":"XLV",
    "BKNG":"XLY",
    "BKR":"XLE",
    "BLDR":"XLI",
    "BLK":"XLF",
    "BMY":"XLV",
    "BNY":"XLF",
    "BR":"XLI",
    "BRK-B":"XLF",
    "BRO":"XLF",
    "BX":"XLF",
    "BXP":"XLRE",
    "C":"XLF",
    "CAG":"XLP",
    "CAH":"XLV",
    "CARR":"XLI",
    "CASY":"XLP",
    "CAT":"XLI",
    "CB":"XLF",
    "CBOE":"XLF",
    "CBRE":"XLRE",
    "CCI":"XLRE",
    "CCL":"XLY",
    "CDNS":"XLK",
    "CDW":"XLK",
    "CF":"XLB",
    "CFG":"XLF",
    "CHD":"XLP",
    "CHRW":"XLI",
    "CIEN":"XLK",
    "CINF":"XLF",
    "CL":"XLP",
    "CLX":"XLP",
    "CMCSA":"XLC",
    "CME":"XLF",
    "CMG":"XLY",
    "CMI":"XLI",
    "CMS":"XLU",
    "CNP":"XLU",
    "COF":"XLF",
    "COHR":"XLK",
    "COO":"XLV",
    "COP":"XLE",
    "COR":"XLV",
    "CPAY":"XLF",
    "CPB":"XLP",
    "CPRT":"XLI",
    "CPT":"XLRE",
    "CRH":"XLB",
    "CRL":"XLV",
    "CSCO":"XLK",
    "CSGP":"XLRE",
    "CSX":"XLI",
    "CTAS":"XLI",
    "CTSH":"XLK",
    "CTVA":"XLB",
    "CVNA":"XLY",
    "CVS":"XLV",
    "CVX":"XLE",
    "DAL":"XLI",
    "DASH":"XLY",
    "DD":"XLB",
    "DE":"XLI",
    "DECK":"XLY",
    "DG":"XLP",
    "DGX":"XLV",
    "DHI":"XLY",
    "DHR":"XLV",
    "DIS":"XLC",
    "DLTR":"XLP",
    "DOC":"XLRE",
    "DOV":"XLI",
    "DOW":"XLB",
    "DPZ":"XLY",
    "DRI":"XLY",
    "DTE":"XLU",
    "DUK":"XLU",
    "DVA":"XLV",
    "DVN":"XLE",
    "DXCM":"XLV",
    "EA":"XLC",
    "EBAY":"XLY",
    "ECL":"XLB",
    "ED":"XLU",
    "EFX":"XLI",
    "EG":"XLF",
    "EIX":"XLU",
    "EL":"XLP",
    "ELV":"XLV",
    "EMR":"XLI",
    "EOG":"XLE",
    "EPAM":"XLK",
    "EQR":"XLRE",
    "EQT":"XLE",
    "ERIE":"XLF",
    "ES":"XLU",
    "ESS":"XLRE",
    "ETR":"XLU",
    "EVRG":"XLU",
    "EW":"XLV",
    "EXC":"XLU",
    "EXE":"XLE",
    "EXPD":"XLI",
    "EXPE":"XLY",
    "EXR":"XLRE",
    "F":"XLY",
    "FANG":"XLE",
    "FAST":"XLI",
    "FCX":"XLB",
    "FDS":"XLF",
    "FDX":"XLI",
    "FE":"XLU",
    "FFIV":"XLK",
    "FICO":"XLK",
    "FIS":"XLF",
    "FISV":"XLF",
    "FITB":"XLF",
    "FIX":"XLI",
    "FOX":"XLC",
    "FOXA":"XLC",
    "FRT":"XLRE",
    "FSLR":"XLK",
    "FTV":"XLI",
    "GD":"XLI",
    "GDDY":"XLK",
    "GE":"XLI",
    "GEHC":"XLV",
    "GEN":"XLK",
    "GILD":"XLV",
    "GIS":"XLP",
    "GL":"XLF",
    "GNRC":"XLI",
    "GOOGL":"XLC",
    "GPC":"XLY",
    "GPN":"XLF",
    "GRMN":"XLY",
    "GS":"XLF",
    "GWW":"XLI",
    "HAL":"XLE",
    "HAS":"XLY",
    "HBAN":"XLF",
    "HCA":"XLV",
    "HD":"XLY",
    "HIG":"XLF",
    "HII":"XLI",
    "HLT":"XLY",
    "HON":"XLI",
    "HPE":"XLK",
    "HPQ":"XLK",
    "HRL":"XLP",
    "HSIC":"XLV",
    "HST":"XLRE",
    "HSY":"XLP",
    "HUBB":"XLI",
    "HWM":"XLI",
    "IBKR":"XLF",
    "IBM":"XLK",
    "ICE":"XLF",
    "IDXX":"XLV",
    "IEX":"XLI",
    "IFF":"XLB",
    "INCY":"XLV",
    "INTU":"XLK",
    "INVH":"XLRE",
    "IP":"XLB",
    "IQV":"XLV",
    "IR":"XLI",
    "IRM":"XLRE",
    "ISRG":"XLV",
    "IT":"XLK",
    "ITW":"XLI",
    "IVZ":"XLF",
    "J":"XLI",
    "JBHT":"XLI",
    "JCI":"XLI",
    "JKHY":"XLF",
    "JPM":"XLF",
    "KDP":"XLP",
    "KEY":"XLF",
    "KEYS":"XLK",
    "KHC":"XLP",
    "KIM":"XLRE",
    "KKR":"XLF",
    "KMB":"XLP",
    "KMI":"XLE",
    "KO":"XLP",
    "KR":"XLP",
    "L":"XLF",
    "LDOS":"XLI",
    "LEN":"XLY",
    "LH":"XLV",
    "LHX":"XLI",
    "LII":"XLI",
    "LITE":"XLK",
    "LMT":"XLI",
    "LNT":"XLU",
    "LOW":"XLY",
    "LUV":"XLI",
    "LVS":"XLY",
    "LYB":"XLB",
    "LYV":"XLC",
    "MA":"XLF",
    "MAA":"XLRE",
    "MAR":"XLY",
    "MAS":"XLI",
    "MCD":"XLY",
    "MCK":"XLV",
    "MCO":"XLF",
    "MDLZ":"XLP",
    "MDT":"XLV",
    "MET":"XLF",
    "MGM":"XLY",
    "MKC":"XLP",
    "MLM":"XLB",
    "MMM":"XLI",
    "MNST":"XLP",
    "MO":"XLP",
    "MOS":"XLB",
    "MPC":"XLE",
    "MPWR":"XLK",
    "MRK":"XLV",
    "MRSH":"XLF",
    "MS":"XLF",
    "MSCI":"XLF",
    "MSI":"XLK",
    "MTB":"XLF",
    "MTD":"XLV",
    "NCLH":"XLY",
    "NDAQ":"XLF",
    "NDSN":"XLI",
    "NEM":"XLB",
    "NI":"XLU",
    "NKE":"XLY",
    "NOC":"XLI",
    "NRG":"XLU",
    "NSC":"XLI",
    "NTAP":"XLK",
    "NTRS":"XLF",
    "NUE":"XLB",
    "NVR":"XLY",
    "NWS":"XLC",
    "NWSA":"XLC",
    "O":"XLRE",
    "ODFL":"XLI",
    "OKE":"XLE",
    "OMC":"XLC",
    "ON":"XLK",
    "ORCL":"XLK",
    "ORLY":"XLY",
    "OTIS":"XLI",
    "OXY":"XLE",
    "PANW":"XLK",
    "PAYX":"XLI",
    "PCAR":"XLI",
    "PCG":"XLU",
    "PEG":"XLU",
    "PEP":"XLP",
    "PFG":"XLF",
    "PG":"XLP",
    "PGR":"XLF",
    "PH":"XLI",
    "PHM":"XLY",
    "PKG":"XLB",
    "PLD":"XLRE",
    "PM":"XLP",
    "PNC":"XLF",
    "PNR":"XLI",
    "PNW":"XLU",
    "PODD":"XLV",
    "POOL":"XLY",
    "PPG":"XLB",
    "PPL":"XLU",
    "PRU":"XLF",
    "PSA":"XLRE",
    "PSKY":"XLC",
    "PSX":"XLE",
    "PTC":"XLK",
    "PWR":"XLI",
    "Q":"XLK",
    "QCOM":"XLK",
    "RCL":"XLY",
    "REG":"XLRE",
    "REGN":"XLV",
    "RF":"XLF",
    "RJF":"XLF",
    "RL":"XLY",
    "RMD":"XLV",
    "ROK":"XLI",
    "ROL":"XLI",
    "ROP":"XLK",
    "RSG":"XLI",
    "RTX":"XLI",
    "RVTY":"XLV",
    "SATS":"XLC",
    "SBAC":"XLRE",
    "SBUX":"XLY",
    "SCHW":"XLF",
    "SHW":"XLB",
    "SJM":"XLP",
    "SLB":"XLE",
    "SNA":"XLI",
    "SNDK":"XLK",
    "SNPS":"XLK",
    "SO":"XLU",
    "SOLV":"XLV",
    "SPG":"XLRE",
    "SPGI":"XLF",
    "SRE":"XLU",
    "STE":"XLV",
    "STLD":"XLB",
    "STT":"XLF",
    "STZ":"XLP",
    "SW":"XLB",
    "SWK":"XLI",
    "SWKS":"XLK",
    "SYF":"XLF",
    "SYK":"XLV",
    "SYY":"XLP",
    "T":"XLC",
    "TAP":"XLP",
    "TDG":"XLI",
    "TDY":"XLK",
    "TECH":"XLV",
    "TEL":"XLK",
    "TER":"XLK",
    "TFC":"XLF",
    "TKO":"XLC",
    "TMO":"XLV",
    "TMUS":"XLC",
    "TPL":"XLE",
    "TRGP":"XLE",
    "TRMB":"XLK",
    "TROW":"XLF",
    "TRV":"XLF",
    "TSCO":"XLY",
    "TSN":"XLP",
    "TT":"XLI",
    "TTD":"XLC",
    "TTWO":"XLC",
    "TXT":"XLI",
    "TYL":"XLK",
    "UBER":"XLI",
    "UDR":"XLRE",
    "UHS":"XLV",
    "ULTA":"XLY",
    "UNP":"XLI",
    "UPS":"XLI",
    "URI":"XLI",
    "USB":"XLF",
    "VEEV":"XLV",
    "VICI":"XLRE",
    "VLO":"XLE",
    "VLTO":"XLI",
    "VMC":"XLB",
    "VRSK":"XLI",
    "VRSN":"XLK",
    "VRTX":"XLV",
    "VTR":"XLRE",
    "VTRS":"XLV",
    "WAB":"XLI",
    "WAT":"XLV",
    "WBD":"XLC",
    "WDAY":"XLK",
    "WEC":"XLU",
    "WELL":"XLRE",
    "WFC":"XLF",
    "WM":"XLI",
    "WMB":"XLE",
    "WRB":"XLF",
    "WSM":"XLY",
    "WST":"XLV",
    "WTW":"XLF",
    "WY":"XLRE",
    "WYNN":"XLY",
    "XEL":"XLU",
    "XOM":"XLE",
    "XYL":"XLI",
    "YUM":"XLY",
    "ZBH":"XLV",
    "ZBRA":"XLK",
    "ZTS":"XLV",

    # Technology (XLK)
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","AVGO":"XLK","CRM":"XLK",
    "CRWD":"XLK","DDOG":"XLK","SNOW":"XLK","DUOL":"XLK","PLTR":"XLK","SMCI":"XLK",
    "TSM":"XLK","INTC":"XLK","MU":"XLK","ARM":"XLK","AMAT":"XLK","LRCX":"XLK",
    "ASML":"XLK","MRVL":"XLK","ALAB":"XLK","ANET":"XLK","NET":"XLK","FTNT":"XLK",
    "TEAM":"XLK","NOW":"XLK","ASAN":"XLK","MNDY":"XLK","AI":"XLK","FIVN":"XLK",
    "PUBM":"XLK","NVMI":"XLK","ONTO":"XLK","WDC":"XLK","STX":"XLK","QUBT":"XLK",
    "RZLV":"XLK","FSLY":"XLK","S":"XLK","ADSK":"XLK","CLS":"XLK","FVRR":"XLK",
    "IREN":"XLK","APLD":"XLK","QS":"XLK","KLAC":"XLK","TXN":"XLK","MCHP":"XLK",
    "NXPI":"XLK","DELL":"XLK","GLW":"XLK","FLEX":"XLK","JBL":"XLK","GFS":"XLK",
    "MDB":"XLK","CYBR":"XLK","CRWV":"XLK","NBIS":"XLK","FIG":"XLK",
    # Financials (XLF)
    "V":"XLF","AXP":"XLF","PYPL":"XLF","COIN":"XLF","XYZ":"XLF","HOOD":"XLF","CRCL":"XLF",
    # Healthcare (XLV)
    "JNJ":"XLV","PFE":"XLV","UNH":"XLV","MRNA":"XLV","NVO":"XLV","BSX":"XLV",
    "CNC":"XLV","LLY":"XLV","ABT":"XLV","AZN":"XLV","CI":"XLV","HUM":"XLV",
    "INSM":"XLV","VKTX":"XLV","SMMT":"XLV","QURE":"XLV","ORIC":"XLV","DNA":"XLV",
    "BRKR":"XLV","BIO":"XLV","ALT":"XLV","SANA":"XLV","SENS":"XLV","VXRT":"XLV",
    # Consumer Discretionary (XLY)
    "TSLA":"XLY","AMZN":"XLY","SHOP":"XLY","ABNB":"XLY","LULU":"XLY","ROST":"XLY",
    "TJX":"XLY","TPR":"XLY","CAVA":"XLY","ETSY":"XLY","GM":"XLY","NIO":"XLY",
    "OPEN":"XLY","BETR":"XLY","GME":"XLY","STLA":"XLY",
    # Consumer Staples (XLP)
    "COST":"XLP","WMT":"XLP","TGT":"XLP","KVUE":"XLP","BYND":"XLP",
    # Communication Services (XLC)
    "GOOG":"XLC","META":"XLC","NFLX":"XLC","ZM":"XLC","ROKU":"XLC","VZ":"XLC",
    "NOK":"XLC","ASTS":"XLC","RDDT":"XLC","PINS":"XLC","CHTR":"XLC","AMC":"XLC",
    # Industrials (XLI)
    "BA":"XLI","ALK":"XLI","UAL":"XLI","ETN":"XLI","VRT":"XLI","GEV":"XLI",
    "SYM":"XLI","EME":"XLI","HY":"XLI","RKLB":"XLI","PL":"XLI","AMPX":"XLI","LYFT":"XLI",
    # Utilities (XLU)
    "CEG":"XLU","VST":"XLU","D":"XLU","NEE":"XLU","OKLO":"XLU",
    # Materials (XLB)
    "MP":"XLB","LIN":"XLB","APD":"XLB","USAR":"XLB",
    # Real Estate (XLRE)
    "EQIX":"XLRE","DLR":"XLRE","RC":"XLRE",
    # ETFs (map to self)
    "SPY":"SPY","QQQ":"QQQ","GLD":"GLD","SLV":"SLV",
    "XLF":"XLF","XLE":"XLE","XLV":"XLV","XLI":"XLI","XLU":"XLU",
}

# ── Bucket-derived sector ETF fallback (added 2026-08-21) ───────────────────
# SECTOR_ETF_MAP is hand-maintained and therefore always behind the universe:
# 16 of 411 tickers were unmapped, all added in the prior two days, and every
# one silently fell back to the MARKET etf -- making "sector-relative" identical
# to "market-relative". add_ticker writes tickers_metadata.csv on every add, so
# resolving through the bucket keeps new tickers covered automatically.
BUCKET_ETF_MAP = {
    "AI": "XLK",
    "Ad Tech": "XLC",
    "Automotive": "XLY",
    "Biotech": "XLV",
    "Commodities": "XLB",
    "Consumer": "XLY",
    "Consumer Disc": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Tech": "XLK",
    "Core Silicon": "SMH",
    "Crypto": "XLF",
    "Custom Silicon": "SMH",
    "Cybersecurity": "XLK",
    "DC REIT": "XLRE",
    "Defense": "XLI",
    "E-commerce": "XLY",
    "Energy": "XLE",
    "Energy Storage": "XLI",
    "Enterprise Software": "IGV",
    "Financials": "XLF",
    "Fintech": "XLF",
    "Healthcare": "XLV",
    "Hyperscaler": "XLK",
    "Industrial Gases": "XLB",
    "Industrials": "XLI",
    "Infrastructure": "XLI",
    "Market ETF": "SPY",
    "Materials": "XLB",
    "Memory": "SMH",
    "Neoclouds": "XLK",
    "Networking": "XLK",
    "Nuclear": "XLU",
    "Physical AI": "XLI",
    "Power": "XLU",
    "Power/Industrial": "XLI",
    "PropTech": "XLRE",
    "Quantum Computing": "XLK",
    "REITs": "XLRE",
    "SaaS Victim": "IGV",
    "Scientific Instruments": "XLV",
    "Sector ETF": "SPY",
    "Semiconductor Equipment": "SMH",
    "Server Hardware": "XLK",
    "Space Tech": "XLI",
    "Telecom": "XLC",
}


def _bucket_etf_lookup():
    """{TICKER: etf} derived from tickers_metadata.csv buckets. Cached."""
    global _BUCKET_ETF_CACHE
    try:
        return _BUCKET_ETF_CACHE
    except NameError:
        pass
    import csv as _csv, os as _os
    out = {}
    _p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                       "tickers_metadata.csv")
    try:
        with open(_p, newline="") as _f:
            _rows = list(_csv.reader(_f))
        _hdr = [h.strip().lower() for h in _rows[0]]
        _tc = next((i for i, h in enumerate(_hdr) if h in ("ticker", "symbol")), 0)
        _bc = next((i for i, h in enumerate(_hdr)
                    if h in ("bucket", "sector", "industry", "group")), None)
        if _bc is not None:
            for _r in _rows[1:]:
                if _r and len(_r) > max(_tc, _bc) and _r[_tc].strip():
                    _e = BUCKET_ETF_MAP.get(_r[_bc].strip())
                    if _e:
                        out[_r[_tc].strip().upper()] = _e
    except Exception as _e:
        log.warning("bucket ETF lookup unavailable (%s); sector falls back to market", _e)
    _BUCKET_ETF_CACHE = out
    return out


def resolve_sector_etf(ticker):
    """SECTOR_ETF_MAP -> bucket -> market. Never silently duplicates the market."""
    _t = (ticker or "").upper()
    _e = SECTOR_ETF_MAP.get(_t)
    if _e:
        return _e
    _e = _bucket_etf_lookup().get(_t)
    if _e:
        return _e
    log.warning("no sector ETF for %s (not in SECTOR_ETF_MAP, no usable bucket) "
                "-- sector-relative will equal market-relative", _t)
    return SECTOR_ETF
# ────────────────────────────────────────────────────────────────────────────

INSIDER_DB      = os.getenv("INSIDER_DB_PATH", "insider_trades.db")
CONGRESS_DB     = os.getenv("CONGRESS_DB_PATH", "congress_trades.db")

PANDEMIC_START  = pd.Timestamp("2020-03-01")
PANDEMIC_END    = pd.Timestamp("2023-12-31")

# ── Output schema (enforced at the end) ──────────────────────────────────────
OUTPUT_COLUMNS = [
    "date", "ticker",
    "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_20",
    "volatility_5d", "volatility_10d",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_width",
    "volume_zscore", "volume_spike",
    "vwap", "obv", "atr",
    "spy_ret", "xlk_ret",
    "xle_ret_5d", "xlv_ret_5d", "xlf_ret_5d", "xlk_ret_5d",
    "xlu_ret_5d", "xli_ret_5d", "xlp_ret_5d", "xly_ret_5d",
    "xlc_ret_5d", "xlre_ret_5d", "xlb_ret_5d",
    "sentiment_score",
    "insider_net_shares", "insider_7d", "insider_21d", "insider_60d", "insider_90d",
    "risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d",
    "is_pandemic",
    # Earnings surprise
    "eps_surprise", "rev_surprise",
    "days_to_earnings",
    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
    "expected_move_perc", "pre_earnings_drift", "post_earnings_drift", "is_earnings_week",
    # ── NEW v2 features ──────────────────────────────────────────────────────
    "return_20d", "return_60d",          # medium-term momentum
    "ma_50",                             # 50-day MA
    "ma5_above_ma20", "ma20_above_ma50", # MA crossover signals
    "high_52w_ratio", "low_52w_ratio",   # distance from 52w extremes
    "bb_pct",                            # BB position (0=at lower, 1=at upper)
    "rsi_above_70", "rsi_below_30",      # RSI extreme flags
    "vwap_dev_eod", "vol_surge_eod", "intraday_momentum",  # intraday-derived
    "obv_trend",                         # OBV vs 10d OBV mean
    "vix_close", "vix_ret",              # market fear gauge
    "oil_ret", "oil_spy_corr",           # crude oil price signal
    "dxy_ret", "yield_10y", "fear_greed", "beta_60d",
    "short_ratio", "short_pct_float", "vix_term_structure", "monday_sentiment",
    "sector_rel_ret",                    # stock return - sector ETF return
    "day_of_week", "is_month_end",       # calendar effects
    # ── NEW v3 features ──────────────────────────────────────────────────────
    "premarket_gap",                     # open vs prev close
    "iv_skew_snap", "pc_ratio_snap",     # options IV skew + put/call ratio
    "analyst_upside", "analyst_buy_pct", "analyst_mult",  # analyst revisions
    "finbert_sentiment", "finbert_sentiment_earnings", "finbert_mult", # FinBERT NLP sentiment
    # ── Session E Phase 3 (May 22 2026): 8-K Item code features ──────────────
    "eightk_exec_change_30d",
    "eightk_material_agreement_30d",
    "eightk_reg_fd_30d",
    "eightk_other_events_30d",
    "eightk_filings_30d",
    "eightk_days_since_last",
    # ── Session E Phase 2 (May 22 2026): Revenue growth from Polygon ─────────
    "rev_growth_yoy",
    "rev_growth_qoq",
    # ── NEW v4 features ──────────────────────────────────────────────────────
    "vix_5d_above_25",          # binary: VIX > 25 for 5 consecutive days
    "semi_etf_momentum_60d",    # SMH ETF 60-day cumulative return
    "igv_vs_sp500_ret_30d",     # IGV vs SPY 30-day spread (software regime)
    "lqd_hyg_spread",           # credit stress: LQD vs HYG 30d spread
    # ── Phase 1 D interaction/normalized features (May 25 2026) ──────────────
    # Informed by A8 finding: volatility * short = squeeze; rev_growth signals matter.
    # All per-ticker, no train/serve mismatch.
    "vol_x_short",              # A: volatility_10d * short_pct_float
    "rev_x_low52w",             # A: rev_growth_yoy * low_52w_ratio
    "vol_10d_self_rank",        # C: rolling 252d rank of own volatility_10d
    "vol_zscore_60d",           # D: (vol_10d - mean_60d) / std_60d
    "is_squeeze_setup",         # E: vol_10d > 0.04 AND short_pct_float > 0.10
    # -- PIT fundamentals (Jun 11 2026, fundamentals.db / Track C) -----------
    "fund_gp_assets",           # (revenue - cogs) / total_assets, Novy-Marx quality
    "fund_op_equity",           # operating_income / equity
    "fund_ni_margin",           # net_income / revenue
    "fund_bm",                  # equity / market cap (value)
    "fund_ep",                  # net_income / market cap (earnings yield)
    # NOTE: short_self_rank + short_zscore_60d removed May 25 2026 — short_pct_float
    # is constant per ticker (single yfinance broadcast), so rolling rank/zscore degenerate.
]

# ── Institutional darkpool features (UW Lee-Ready flow) ──────────────────────
# Audit-validated May 17 2026 (n=458, 3 rounds). 4 of 8 candidates survived:
#   inst_block_buy_sell_7d  p=0.0017  max|rho|=0.34 vs 87-feat panel
#   inst_signed_flow_30d    p=0.011   max|rho|=0.17
#   inst_auction_imbal_5d   p=0.014   max|rho|=0.20
#   inst_signed_flow_5d     p=0.18    max|rho|=0.19  (kept for timescale div.)
# Dropped: block_notional_7d (rho=0.64 price-scale proxy), block_count_7d
#   (p=0.72 no signal), dp_signed_flow_5d (redundant), sweep_count_7d (zero).
# Gated: when OFF (default) these are NOT in OUTPUT_COLUMNS -- true no-op,
# existing 303 models unaffected. Flip ML_QUANT_INST_FEATURES=1 before a
# Pipeline B retrain to bake them into the next model generation.
_INST_FEATURES_ENABLED = os.environ.get("ML_QUANT_INST_FEATURES", "0") == "1"
_FUND_FEATURES_ENABLED = os.environ.get("ML_QUANT_DISABLE_FUND_FEATURES", "0") != "1"
_INST_FEATURE_COLS = [
    "inst_block_buy_sell_7d",
    "inst_signed_flow_30d",
    "inst_auction_imbal_5d",
    "inst_signed_flow_5d",
]
if _INST_FEATURES_ENABLED:
    OUTPUT_COLUMNS = OUTPUT_COLUMNS + _INST_FEATURE_COLS

# ── MISSING-INDICATOR columns for sparse features (May 27 2026, Phase 2) ──
# When ML_QUANT_MISSING_INDICATORS=1, builder emits {feature}_has_value
# binary columns alongside the 4 inst_* features. Trees split on 'data present?'
# vs 'value when present' independently. Mirrors classifier.py SPARSE_INDICATOR_COLS.
_MISSING_INDICATORS_ENABLED = os.environ.get("ML_QUANT_MISSING_INDICATORS", "0") == "1"
_SPARSE_FEATURE_COLS = _INST_FEATURE_COLS  # 4 inst features that get indicators
_SPARSE_INDICATOR_COLS = [f"{c}_has_value" for c in _SPARSE_FEATURE_COLS]
if _MISSING_INDICATORS_ENABLED:
    OUTPUT_COLUMNS = OUTPUT_COLUMNS + _SPARSE_INDICATOR_COLS

# ── PANEL TRANSFORMS A/B (P3.5, May 29 2026) ─────────────────────────────────
# 7 transforms of existing features, surfaced by the P3.2 alpha gate as the
# only panel features adding material ABSOLUTE IC uplift (~+0.012-0.024) over
# their raw bases. Definitions match analysis/alpha_transformations.py EXACTLY
# so the gate-measured IC transfers. Flag OFF (default) = true no-op.
_PANEL_TRANSFORMS_ENABLED = os.environ.get("ML_QUANT_PANEL_TRANSFORMS", "0") == "1"
_PANEL_TRANSFORM_COLS = [
    "ma_20__ts_std__w5",
    "ma_10__ts_std__w10",
    "rsi_14__ts_mean__w20",
    "bb_upper__ts_delta__w20",
    "post_earnings_3d__ts_mean__w10",
    "rev_growth_qoq__ts_std__w10",
    "is_squeeze_setup__ts_argmax__w20",
]
if _PANEL_TRANSFORMS_ENABLED:
    OUTPUT_COLUMNS = OUTPUT_COLUMNS + _PANEL_TRANSFORM_COLS

# ── A8 cross-sectional prob_top_decile feature (Phase 2A, May 27 2026) ────────
# When ML_QUANT_A8_FEATURE=1, builder reads data/a8_oos_panel.parquet and joins
# a8_prob_top_decile as a feature column. A8 = top-decile cross-sectional model
# (OOS AUC 0.677). Per-ticker model learns when A8 conviction boosts/tempers signal.
#
# Walk-forward guarantee: each row's a8_prob uses A8 trained on data BEFORE that
# date (5-day purge), so no look-ahead leakage.
#
# Missing values: 2020-01 to 2020-08 (pre-training), or sparse tickers like BYND
# fall back to 0.10 (universe top-decile base rate).
_A8_FEATURE_ENABLED = os.environ.get("ML_QUANT_A8_FEATURE", "0") == "1"
_A8_PANEL_PATH = "data/a8_oos_panel.parquet"
_A8_PANEL_CACHE = None  # lazy-loaded; cached at module level

def _load_a8_panel():
    """Lazy-load the A8 OOS panel. Returns DataFrame indexed by (ticker, date)."""
    global _A8_PANEL_CACHE
    if _A8_PANEL_CACHE is not None:
        return _A8_PANEL_CACHE
    import pandas as pd
    if not os.path.exists(_A8_PANEL_PATH):
        # File missing — return empty DataFrame, all lookups will fall to fallback
        _A8_PANEL_CACHE = pd.DataFrame(columns=["ticker", "date", "a8_prob"])
        return _A8_PANEL_CACHE
    df = pd.read_parquet(_A8_PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    _A8_PANEL_CACHE = df.set_index(["ticker", "date"])["a8_prob"]
    return _A8_PANEL_CACHE

if _A8_FEATURE_ENABLED:
    OUTPUT_COLUMNS = OUTPUT_COLUMNS + ["a8_prob_top_decile"]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Per-ticker OHLCV cache, keyed by ticker only.
# Walk-forward calls _download 3,739 times (one per outcome). Without this
# cache that's 3,739 fresh Massive API calls per process, triggering rate
# limits and yfinance fallback hangs. With this cache, 3,739 calls become
# ~125 (one per unique ticker), each subsequent call slicing from memory.
# OHLCV historical data is immutable so caching the widest range is safe.
# Added May 5 2026.
_TICKER_OHLCV_CACHE: dict[str, "pd.DataFrame"] = {}


def _last_completed_session():
    """Last US trading day whose close has passed (17:00 ET + publish margin).

    2026-07-14: was a DUPLICATE of massive_client._last_completed_session, and
    both were weekends-only. Now both delegate to utils.market_calendar --
    one source of truth, holiday-aware, validated against raw_bars.
    """
    from utils.market_calendar import last_completed_session as _lcs
    return _lcs()


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance/Massive and flatten any MultiIndex columns.

    May 5 2026: Per-ticker module-level cache with full-range fetch on first
    call, in-memory slice for subsequent calls. Same V2 pattern as FRED and
    _MACRO_CACHE. Eliminates rate-limit issues during walk-forward where the
    same ticker is fetched thousands of times with varying end_dates.
    """
    if ticker not in _TICKER_OHLCV_CACHE:
        # First call for this ticker — fetch widest range (start through today)
        from datetime import date as _date
        widest_end = _last_completed_session().strftime("%Y-%m-%d")
        full_df = mc.download(
            ticker, start=start, end=widest_end,
            auto_adjust=True, progress=False,
        )
        if full_df.empty:
            raise ValueError(f"yfinance returned no data for {ticker} ({start} → {widest_end})")

        # Flatten MultiIndex once at cache time
        if isinstance(full_df.columns, pd.MultiIndex):
            full_df.columns = full_df.columns.get_level_values(0)

        _TICKER_OHLCV_CACHE[ticker] = full_df

    full = _TICKER_OHLCV_CACHE[ticker]

    # Slice cached DataFrame to requested [start, end]
    df = full
    try:
        if start is not None:
            start_ts = pd.to_datetime(start)
            df = df.loc[df.index >= start_ts]
        if end is not None:
            end_ts = pd.to_datetime(end)
            df = df.loc[df.index <= end_ts]
    except Exception:
        # If index isn't datetime, return full DataFrame unchanged
        pass

    df = df.copy()

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} ({start} → {end})")

    # Note: MultiIndex flattening already done at cache time, but check defensively
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Normalise column names to lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename 'date' if yfinance called it something else
    for alias in ("datetime", "index"):
        if alias in df.columns and "date" not in df.columns:
            df = df.rename(columns={alias: "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date   # keep as date, not timestamp
    df["ticker"] = ticker.upper()
    return df


# ── Macro indicator cache ──────────────────────────────────────────────────
# Caches ^VIX, ES=F, ^VIX3M (and any other macro symbol) at module level so
# Pipeline B fetches each macro ONCE per process run, not per ticker.
# Was causing curl_cffi DNS thread exhaustion ~ticker 47 (May 2 2026 incident).
_MACRO_CACHE: dict[str, "pd.DataFrame"] = {}

# ── Provider circuit breaker (May 6 2026) ─────────────────────────────────
# Per ChatGPT walk-forward consult. When a provider+symbol fetch fails,
# mark it disabled for 30 min. Prevents walk-forward (~thousands of calls)
# from hitting the same doomed call repeatedly.
import time as _time_mod
_PROVIDER_FAILURE_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_PROVIDER_FAILURE_TTL = 30 * 60  # 30 minutes


def _provider_disabled(provider: str, symbol: str) -> bool:
    """Check if (provider, symbol) is in failure cooldown."""
    key = (provider, symbol)
    item = _PROVIDER_FAILURE_CACHE.get(key)
    if not item:
        return False
    failed_at, _err = item
    if _time_mod.time() - failed_at > _PROVIDER_FAILURE_TTL:
        _PROVIDER_FAILURE_CACHE.pop(key, None)
        return False
    return True


def _mark_provider_failure(provider: str, symbol: str, err: Exception) -> None:
    """Mark (provider, symbol) as failed; will skip for 30 min."""
    _PROVIDER_FAILURE_CACHE[(provider, symbol)] = (_time_mod.time(), repr(err))
    import logging
    logging.getLogger(__name__).warning(
        f"provider.circuit_mark provider={provider} symbol={symbol} err={err!r}"
    )


def _get_macro_cached(symbol: str, start: str, end: str) -> "pd.DataFrame":
    """Fetch a macro symbol via mc.download, cached for the process lifetime.
    Returns a deep copy so callers can mutate freely.

    May 4 2026 V2: Cache key is `symbol` only. First call fetches the WIDEST
    range from `start` through today, caches the full DataFrame, and slices
    in memory for the [start, end] requested by caller.

    V1 (key=symbol|start|end) was broken for walk-forward where end_date
    varies per call → cache always missed → thousands of fresh API calls →
    rate limits. Macro/OHLCV historical data is immutable so caching the
    widest range is correct.
    """
    if symbol not in _MACRO_CACHE:
        # Circuit breaker: skip if recently failed (May 6 2026)
        if _provider_disabled("macro", symbol):
            import logging
            logging.getLogger(__name__).warning(
                f"_get_macro_cached: skipping {symbol} (in failure cooldown)"
            )
            return pd.DataFrame()
        # First call for this symbol — fetch widest range (start through today)
        from datetime import date as _date
        widest_end = _last_completed_session().strftime("%Y-%m-%d")
        try:
            _MACRO_CACHE[symbol] = mc.download(
                symbol, start=start, end=widest_end,
                auto_adjust=True, progress=False,
            )
        except Exception as e:
            _mark_provider_failure("macro", symbol, e)
            _MACRO_CACHE[symbol] = pd.DataFrame()  # cache the empty so we don't retry

    full = _MACRO_CACHE[symbol]
    if full is None or full.empty:
        return full.copy() if full is not None else pd.DataFrame()

    # Slice cached full-range DataFrame to [start, end]
    sliced = full
    try:
        if start is not None:
            start_ts = pd.to_datetime(start)
            sliced = sliced.loc[sliced.index >= start_ts]
        if end is not None:
            end_ts = pd.to_datetime(end)
            sliced = sliced.loc[sliced.index <= end_ts]
    except Exception:
        # If index isn't datetime, fall back to returning full DataFrame
        pass

    return sliced.copy()


def _market_return(etf: str, start: str, end: str,
                   index: pd.Index, return_close: bool = False):
    """Fetch ETF daily return, reindexed to match the main df's date index.
    If return_close=True, returns a DataFrame with both 'close' and 'ret' columns.
    """
    try:
        tmp = _get_macro_cached(etf, start, end)
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = tmp.columns.get_level_values(0)
        # Massive returns an UNNAMED DatetimeIndex; yfinance named it 'Date'.
        # reset_index() therefore yields 'index' not 'date' -> KeyError ->
        # silent NaN. Name the datetime column explicitly. (Fix Jun 30 2026.)
        tmp = tmp.reset_index()
        tmp.columns = [str(c).strip().lower() for c in tmp.columns]
        if "date" not in tmp.columns:
            # first column is the (former) datetime index whatever it was named
            tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        tmp = tmp.set_index("date")
        close_s = tmp["close"].reindex(index).ffill()
        ret_s   = close_s.pct_change().rename(f"{etf.lower()}_ret")
        if return_close:
            return pd.DataFrame({"close": close_s.values, "ret": ret_s.values},
                                 index=index)
        return ret_s
    except Exception:
        if return_close:
            return pd.DataFrame({"close": np.full(len(index), 20.0),
                                  "ret":   np.zeros(len(index))}, index=index)
        return pd.Series(np.nan, index=index, name=f"{etf.lower()}_ret")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI — same formula used in TradingView / FactSet."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series,
         close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP (resets each day — here it's daily running VWAP)."""
    cum_vol = volume.cumsum()
    cum_pv  = (close * volume).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


# ── Optional signal loaders (all return pd.Series indexed by date) ───────────

def _load_insider_uw(ticker: str, dates: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load insider trades from Unusual Whales API.
    Routed through features.uw_client (market-hours gated). On gate-closed /
    rate-limited / failed, falls through to _load_insider (SQLite).
    """
    zeros = pd.Series(0.0, index=dates, name="insider_net_shares")
    try:
        from features.uw_client import uw_get
        payload = uw_get(f"/api/insider/{ticker}/ticker-flow")
        if payload is None:
            return _load_insider(ticker, dates)
        trades = payload.get("data", [])
        if not trades:
            return _load_insider(ticker, dates)

        rows = []
        for t in trades:
            try:
                # UW "volume" field is pre-signed: positive for buy, negative for sell.
                # See features/uw_client docs for /api/insider/{ticker}/ticker-flow schema.
                rows.append({
                    "date":       pd.Timestamp(t["date"]).date(),
                    "net_shares": float(t.get("volume", 0))
                })
            except Exception:
                continue

        if not rows:
            return _load_insider(ticker, dates)

        idf = pd.DataFrame(rows).groupby("date")["net_shares"].sum()
        idf.index = pd.to_datetime(idf.index)
        net   = idf.reindex(dates).fillna(0.0).rename("insider_net_shares")
        roll7 = net.rolling(7,  min_periods=1).sum().rename("insider_7d")
        roll21= net.rolling(21, min_periods=1).sum().rename("insider_21d")
        roll60= net.rolling(60, min_periods=1).sum().rename("insider_60d")
        roll90= net.rolling(90, min_periods=1).sum().rename("insider_90d")
        return net, roll7, roll21, roll60, roll90

    except Exception:
        return _load_insider(ticker, dates)


def _load_insider(ticker: str, dates: pd.Index, as_of: str | date | None = None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load insider net_shares + 7d/21d/60d/90d rolling sums from SQLite."""
    zeros = pd.Series(0.0, index=dates)
    try:
        conn = sqlite3.connect(INSIDER_DB, timeout=30)
        if as_of is not None:
            # Point-in-time honesty: filter by created_at (γ backfill: trade_date + 2 BD)
            df = pd.read_sql(
                "SELECT date, net_shares FROM insider_flows "
                "WHERE ticker = ? AND (created_at IS NULL OR DATE(created_at) <= ?) "
                "ORDER BY date",
                conn, params=(ticker.upper(), str(as_of)), parse_dates=["date"]
            )
        else:
            df = pd.read_sql(
                "SELECT date, net_shares FROM insider_flows WHERE ticker = ? ORDER BY date",
                conn, params=(ticker.upper(),), parse_dates=["date"]
            )
        conn.close()
        if df.empty:
            return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        df = df.set_index(df["date"].dt.date)["net_shares"]
        net    = df.reindex(dates).fillna(0.0)
        roll7  = net.rolling(7,  min_periods=1).sum()
        roll21 = net.rolling(21, min_periods=1).sum()
        roll60 = net.rolling(60, min_periods=1).sum()
        roll90 = net.rolling(90, min_periods=1).sum()
        return (net.rename("insider_net_shares"),
                roll7.rename("insider_7d"),
                roll21.rename("insider_21d"),
                roll60.rename("insider_60d"),
                roll90.rename("insider_90d"))
    except Exception:
        return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()


def _load_congress(ticker: str, dates: pd.Index) -> pd.Series:
    """Load congressional net shares from SQLite."""
    zeros = pd.Series(0.0, index=dates, name="congress_net_shares")
    try:
        conn = sqlite3.connect(CONGRESS_DB, timeout=30)
        df = pd.read_sql(
            "SELECT ds as date, congress_net_shares FROM congress_flows WHERE ticker = ? ORDER BY date",
            conn, params=(ticker.upper(),), parse_dates=["date"]
        )
        conn.close()
        if df.empty:
            return zeros
        df = df.set_index(df["date"].dt.date)["congress_net_shares"]
        return df.reindex(dates).fillna(0.0).rename("congress_net_shares")
    except Exception:
        return zeros


def _load_risk_flags(dates: pd.Index) -> pd.DataFrame:
    """Load pre-computed risk flags. Falls back to zeros if unavailable."""
    cols = ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]
    zero_df = pd.DataFrame(0.0, index=dates, columns=cols)
    try:
        from signals.risk_gate import build_risk_features
        rf = build_risk_features(dates[0], dates[-1])
        # 2026-05-20 dead-feature fix: build_risk_features already returns dates as
        # the index. The old code did rf.set_index("date") which raised KeyError
        # (no "date" column), got swallowed by bare except, and returned zeros for
        # all 4 risk_* features across every ticker for months. See audit notes.
        rf = rf[cols].copy()
        rf.index = pd.to_datetime(rf.index).normalize()
        target = pd.to_datetime(pd.Index(dates)).normalize()
        out = rf.reindex(target).fillna(0.0)
        out.index = dates  # restore caller's index type
        return out
    except Exception:
        return zero_df


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  ←  THE ONLY FUNCTION YOU SHOULD IMPORT FROM THIS MODULE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_dataframe(
    ticker: str,
    start_date: str | date = "2018-01-01",
    end_date:   str | date | None = None,
    include_sentiment: bool = True,    # reads from SQLite cache, 0.0 if no data
    training_mode: bool = False,       # if True, skip slow live API calls
) -> pd.DataFrame:
    """
    Build the canonical feature DataFrame for `ticker`.

    Parameters
    ----------
    ticker         : e.g. "AAPL"
    start_date     : ISO string or date object  (default: 2018-01-01)
    end_date       : ISO string or date object  (default: today)
    include_sentiment : if True, calls sentiment pipeline (slow, costs API calls)

    Returns
    -------
    pd.DataFrame with exactly OUTPUT_COLUMNS columns, clean RangeIndex,
    NaNs dropped from warm-up period only.
    """
    if end_date is None:
        # VN-date bug: datetime.today() = Mac VN local = a day ahead of the last
        # completed ET session, requesting a non-existent future daily bar.
        # Cap at last completed session. Fix Jul 1 2026.
        end_date = _last_completed_session().strftime("%Y-%m-%d")

    start_str = str(start_date)
    end_str   = str(end_date)
    ticker    = ticker.upper().strip()

    # ── 1. Price data ─────────────────────────────────────────────────────────
    df = _download(ticker, start_str, end_str)

    required = {"date", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after download: {missing}")

    # ── 2. Date index for reindexing optional signals ─────────────────────────
    date_index = pd.Index(df["date"])

    # ── 3. Price-based features ───────────────────────────────────────────────
    # Ensure Series (not DataFrame if duplicate columns exist)
    c = df["close"]
    if hasattr(c, "columns"):  # it's a DataFrame (duplicate "close" columns)
        c = c.iloc[:, 0]
    o = df["open"] if "open" in df.columns else c
    if hasattr(o, "columns"):  # it's a DataFrame (duplicate "open" columns)
        o = o.iloc[:, 0]

    # ── Pre-market gap ────────────────────────────────────────────────────────
    # How much did stock gap up/down from yesterday's close to today's open
    df["premarket_gap"] = (o - c.shift(1)) / c.shift(1)

    df["return_1d"] = c.pct_change(1)
    df["return_3d"] = c.pct_change(3)
    df["return_5d"] = c.pct_change(5)

    df["ma_5"]  = c.rolling(5).mean()
    df["ma_10"] = c.rolling(10).mean()
    df["ma_20"] = c.rolling(20).mean()

    df["volatility_5d"]  = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # ── 4. Oscillators ────────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(c)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = (4 * std20 / ma20.replace(0, np.nan))

    # ── 5. Volume features ────────────────────────────────────────────────────
    v = df["volume"].replace(0, np.nan)
    vol_mean = v.rolling(VOL_LOOKBACK).mean()
    vol_std  = v.rolling(VOL_LOOKBACK).std()

    df["volume_zscore"] = (v - vol_mean) / vol_std.replace(0, np.nan)
    df["volume_spike"]  = (df["volume_zscore"] > 2).astype(int)
    df["vwap"]          = _vwap(c, v)
    df["obv"]           = _obv(c, v)

    # ATR needs high/low — check they exist
    if {"high", "low"}.issubset(df.columns):
        df["atr"] = _atr(df["high"], df["low"], c)
    else:
        df["atr"] = np.nan

    # ── 6. Market / sector context ────────────────────────────────────────────
    spy = _market_return(SPY_TICKER, start_str, end_str, date_index)
    xlk = _market_return(SECTOR_ETF, start_str, end_str, date_index)
    df["spy_ret"] = spy.values
    df["xlk_ret"] = xlk.values

    # ── 6a. Sector ETF 5-day returns ───────────────────────────────────────────
    # Added 2026-05-21: gives model thematic regime signal that single-day
    # xlk_ret + sector_rel_ret can't capture. Hypothesis: May 2026 regression
    # in OKLO/CEG/ORIC/MRNA was sector rollover the model couldn't see.
    # F2 (2026-05-28): extended to all 11 SPDR sectors (added XLC, XLRE, XLB)
    # to match the complete SECTOR_ETF_MAP. Previously only 8 sectors had 5d ret.
    for _etf in ("XLE", "XLV", "XLF", "XLU", "XLI", "XLP", "XLY", "XLC", "XLRE", "XLB"):
        try:
            _ret = _market_return(_etf, start_str, end_str, date_index)
            _ret_5d = (1 + _ret).rolling(5).apply(lambda x: x.prod() - 1, raw=True)
            df[f"{_etf.lower()}_ret_5d"] = _ret_5d.values
        except Exception as _e:
            # Rule #1(b): log the failure instead of silently swallowing it.
            import logging as _lg
            _lg.warning("sector 5d ret failed for %s: %s; filling 0.0", _etf, _e)
            df[f"{_etf.lower()}_ret_5d"] = 0.0
    # XLK 5d separately (XLK already loaded above)
    try:
        _xlk_5d = (1 + xlk).rolling(5).apply(lambda x: x.prod() - 1, raw=True)
        df["xlk_ret_5d"] = _xlk_5d.values
    except Exception:
        df["xlk_ret_5d"] = 0.0

    # ── 6b. Macro features — DXY, 10Y yield, Fear & Greed, Beta, Short interest ──
    # DXY via FRED Trade Weighted USD Index (DTWEXBGS) — replaces yfinance
    try:
        from features.fred_client import fred_get_as_series
        _dxy_series = fred_get_as_series("DTWEXBGS", start=start_str, end=end_str)
        if _dxy_series is not None and not _dxy_series.empty:
            # FRED gives index level — compute daily returns
            _dxy_ret = _dxy_series.pct_change().fillna(0.0)
            _dxy_map = {d.date(): v for d, v in _dxy_ret.items()}
            df["dxy_ret"] = df["date"].map(_dxy_map).fillna(0.0).values
        else:
            df["dxy_ret"] = 0.0
    except Exception:
        df["dxy_ret"] = 0.0

    # VIX term structure — VIX/VIX3M ratio (Hull Chapter 20)
    # ratio > 1 = inverted = acute panic (short-term fear > long-term) = mean reverting, less dangerous
    # ratio < 1 = normal = sustained fear = more dangerous, suppresses BUY signals harder
    # VIX from FRED (VIXCLS, official CBOE redistribution)
    # VIX3M from yfinance (CBOE proprietary, FRED doesn't have it) - uses resilient wrapper
    try:
        from features.fred_client import fred_get_as_series
        from features.yf_resilient import safe_yf_download

        _vix_s = fred_get_as_series("VIXCLS", start=start_str, end=end_str)
        # Use cached fetch instead of safe_yf_download (60s TTL was too short
        # for full Pipeline B run; module cache lasts the whole process)
        _vix3m_raw = _get_macro_cached("^VIX3M", start_str, end_str)

        if (_vix_s is not None and not _vix_s.empty
            and _vix3m_raw is not None and not _vix3m_raw.empty):
            if isinstance(_vix3m_raw.columns, pd.MultiIndex):
                _vix3m_raw.columns = _vix3m_raw.columns.get_level_values(0)
            _vix3m_raw.index = pd.to_datetime(_vix3m_raw.index).normalize()
            _vix3m_s = _vix3m_raw["Close"].squeeze()

            _ratio = (_vix_s / _vix3m_s).dropna()
            _ratio_map = {d.date(): v for d, v in _ratio.items()}
            df["vix_term_structure"] = df["date"].map(_ratio_map).ffill().fillna(1.0)
        else:
            df["vix_term_structure"] = 1.0
    except Exception:
        df["vix_term_structure"] = 1.0



    # 10Y Treasury yield via FRED DGS10 (replaces yfinance ^TNX)
    # DGS10 is already in % (e.g. 4.25), divide by 100 to match yfinance ^TNX format (0.0425)
    try:
        from features.fred_client import fred_get_as_series
        tnx_series = fred_get_as_series("DGS10", start=start_str, end=end_str)
        if tnx_series is not None and not tnx_series.empty:
            tnx_series = tnx_series / 100.0  # Convert % to decimal
            tnx_map = {d.date(): v for d, v in tnx_series.items()}
            df["yield_10y"] = df["date"].map(tnx_map).ffill().fillna(0.04)
        else:
            df["yield_10y"] = 0.04
    except Exception:
        df["yield_10y"] = 0.04

    # Fear & Greed Index — dropped from model 2026-05-21 (no historical source).
    # Kept as constant in df so OUTPUT_COLUMNS / prediction_features schema unchanged.
    df["fear_greed"] = 0.5

    # Monday sentiment score (Anthropic API — scored Sunday night)
    try:
        conn_sent = sqlite3.connect("data/sentiment.db", timeout=30)
        sent_row = conn_sent.execute("""
            SELECT sentiment_score, confidence FROM monday_sentiment
            WHERE ticker=? ORDER BY score_date DESC LIMIT 1
        """, (ticker,)).fetchone()
        conn_sent.close()
        if sent_row:
            # Decay sentiment signal over the week — full strength Monday, zero by Friday
            from utils.timezone import now_et
            dow = now_et().weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
            if dow >= 5:  # weekend — treat as Monday (preparing for next week)
                decay = 1.0
            else:
                decay = max(0.0, 1.0 - (dow * 0.25))  # Mon=1.0, Tue=0.75, Wed=0.5, Thu=0.25, Fri=0.0
            df["monday_sentiment"] = sent_row[0] * sent_row[1] * decay
        else:
            df["monday_sentiment"] = 0.0
    except Exception:
        df["monday_sentiment"] = 0.0

    # 60-day rolling beta vs SPY
    try:
        _spy_ret = pd.Series(spy.values, index=df.index)
        _stk_ret = c.pct_change()
        _cov = _stk_ret.rolling(60).cov(_spy_ret)
        _var = _spy_ret.rolling(60).var()
        df["beta_60d"] = (_cov / _var.replace(0, np.nan)).fillna(1.0)
    except Exception:
        df["beta_60d"] = 1.0

    # Short interest ratio (from yfinance — updates bi-weekly)
    try:
        if not training_mode:
            _info = mc.get_short_interest(ticker)
            df["short_ratio"]   = float(_info.get("shortRatio") or 0.0)
            df["short_pct_float"] = float(_info.get("shortPercentOfFloat") or 0.0)
        else:
            df["short_ratio"]   = 0.0
            df["short_pct_float"] = 0.0
    except Exception:
        df["short_ratio"]   = 0.0
        df["short_pct_float"] = 0.0

    # ── 7. Sentiment — reads from SQLite cache (run etl_sentiment.py daily) ────
    # Historical rows default to 0.0 (no past headlines available).
    # Live predictions use today's cached FinBERT score.
    # Set include_sentiment=False to skip entirely (faster, for batch training).
    if include_sentiment:
        try:
            from data.etl_sentiment import load_sentiment_scores
            # Point-in-time honest: filter to scores known on or before end_date
            sent_df = load_sentiment_scores(ticker, start_date=start_str, end_date=end_str, as_of=end_str)
            if not sent_df.empty:
                sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.date
                sent_map = sent_df.set_index("date")["score"].to_dict()
                df["sentiment_score"] = df["date"].map(sent_map).fillna(0.0)
            else:
                df["sentiment_score"] = 0.0
        except Exception:
            df["sentiment_score"] = 0.0
    else:
        df["sentiment_score"] = 0.0

    # ── 8. Earnings surprise features ────────────────────────────────────────
    try:
        from data.etl_earnings import load_earnings_features, load_uw_earnings_features
        # Point-in-time honest: only use earnings rows known on or before end_date
        earn = load_earnings_features(ticker, date_index, as_of=end_str)
        for col in ["eps_surprise", "rev_surprise", "days_to_earnings",
                    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d"]:
            df[col] = earn[col].values if col in earn.columns else 0.0
        # UW earnings — richer features (LIVE API, skipped in training_mode for PIT honesty)
        if not training_mode:
            uw_earn = load_uw_earnings_features(ticker, date_index)
            for col in ["expected_move_perc", "pre_earnings_drift",
                        "post_earnings_drift", "is_earnings_week"]:
                df[col] = uw_earn[col].values if col in uw_earn.columns else 0.0
        else:
            for col in ["expected_move_perc", "pre_earnings_drift",
                        "post_earnings_drift", "is_earnings_week"]:
                df[col] = 0.0
    except Exception:
        for col in ["eps_surprise", "rev_surprise", "days_to_earnings",
                    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
                    "expected_move_perc", "pre_earnings_drift",
                    "post_earnings_drift", "is_earnings_week"]:
            df[col] = 0.0

    # ── UW extended features — seasonality, analyst, FTDs ─────────────────
    try:
        from features.uw_signals import (
            get_seasonality_features, get_analyst_score, get_ftd_score
        )
        seas    = get_seasonality_features(ticker)
        analyst = get_analyst_score(ticker)
        ftd     = get_ftd_score(ticker)
        df["seasonal_avg_return"]   = seas["seasonal_avg_return"]
        df["seasonal_positive_pct"] = seas["seasonal_positive_pct"]
        df["analyst_score"]         = analyst["analyst_score"]
        df["upgrades_30d"]          = float(analyst["upgrades_30d"])
        df["downgrades_30d"]        = float(analyst["downgrades_30d"])
        df["ftd_shares"]            = float(ftd["ftd_shares"]) / 1e6
    except Exception:
        for col in ["seasonal_avg_return","seasonal_positive_pct",
                    "analyst_score","upgrades_30d","downgrades_30d","ftd_shares"]:
            df[col] = 0.0

    # ── 9. Insider flows ──────────────────────────────────────────────────────
    # In training_mode, skip the LIVE UW API and go directly to SQLite with PIT filter
    if training_mode:
        ins_net, ins_7d, ins_21d, ins_60d, ins_90d = _load_insider(ticker, date_index, as_of=end_str)
    else:
        ins_net, ins_7d, ins_21d, ins_60d, ins_90d = _load_insider_uw(ticker, date_index)
    df["insider_net_shares"] = ins_net.values
    df["insider_7d"]         = ins_7d.values
    df["insider_21d"]        = ins_21d.values
    df["insider_60d"]        = ins_60d.values
    df["insider_90d"]        = ins_90d.values

    # ── 10. Congressional trading ──────────────────────────────────────────────
    # 2026-05-20: congress_net_shares removed from FEATURE_COLUMNS.
    # _load_congress function preserved in case revived as risk-gate signal.
    # Rationale: 45-day STOCK Act disclosure lag exceeds 1/3/5d horizons;
    # post-2012 academic outperformance signal weak; NANC/KRUZ ETFs underperform.

    # ── 11. Risk flags ────────────────────────────────────────────────────────
    risk = _load_risk_flags(date_index)
    for col in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]:
        df[col] = risk[col].values if col in risk.columns else 0.0

    # ── 11. Pandemic regime ───────────────────────────────────────────────────
    dates_ts = pd.to_datetime(df["date"])
    df["is_pandemic"] = (
        (dates_ts >= PANDEMIC_START) & (dates_ts <= PANDEMIC_END)
    ).astype(int)

    # ── 12. NEW v2 features ───────────────────────────────────────────────────

    # Medium-term momentum
    df["return_20d"] = c.pct_change(20)
    df["return_60d"] = c.pct_change(60)

    # MA50 and crossover signals
    df["ma_50"]           = c.rolling(50).mean()
    df["ma5_above_ma20"]  = (df["ma_5"] > df["ma_20"]).astype(int)
    df["ma20_above_ma50"] = (df["ma_20"] > df["ma_50"]).astype(int)

    # 52-week high/low ratios
    high_52w = c.rolling(252).max()
    low_52w  = c.rolling(252).min()
    df["high_52w_ratio"] = (c / high_52w.replace(0, np.nan)) - 1.0   # 0 = at 52w high
    df["low_52w_ratio"]  = (c / low_52w.replace(0, np.nan)) - 1.0    # 0 = at 52w low

    # BB position (0 = at lower band, 1 = at upper band)
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / bb_range

    # RSI extreme flags
    df["rsi_above_70"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_below_30"] = (df["rsi_14"] < 30).astype(int)

    # ── Intraday-derived daily features ──────────────────────────────────────
    # vwap_dev_eod: how far close was from VWAP at end of day
    df["vwap_dev_eod"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan)

    # vol_surge_eod: today's volume vs 20d average
    vol_avg = v.rolling(20).mean().replace(0, np.nan)
    df["vol_surge_eod"] = v / vol_avg

    # intraday_momentum: vwap_dev weighted by vol_surge
    df["intraday_momentum"] = df["vwap_dev_eod"] * df["vol_surge_eod"].fillna(1)

    # OBV trend (OBV minus its 10d mean, normalized by std)
    obv_ma  = df["obv"].rolling(10).mean()
    obv_std = df["obv"].rolling(10).std().replace(0, np.nan)
    df["obv_trend"] = (df["obv"] - obv_ma) / obv_std

    # VIX
    # ── Overnight futures (S&P500 futures ES=F) ─────────────────────────────
    try:
        es_ret = _market_return("ES=F", start_str, end_str, date_index)
        df["es_overnight"] = es_ret.values
    except Exception:
        df["es_overnight"] = 0.0

    # VIX close from FRED VIXCLS (^VIX is a true index Massive cannot serve;
    # yfinance disabled by XProtect). Mirrors the proven vix_term_structure
    # FRED pattern above. Fix Jun 30 2026 (was defaulting to constant 20.0).
    try:
        from features.fred_client import fred_get_as_series
        _vixc = fred_get_as_series("VIXCLS", start=start_str, end=end_str)
        if _vixc is not None and not _vixc.empty:
            _vix_map = {d.date(): v for d, v in _vixc.items()}
            _vix_close_s = df["date"].map(_vix_map).ffill().bfill()
            df["vix_close"] = _vix_close_s.values
            df["vix_ret"]   = _vix_close_s.pct_change().fillna(0.0).values
        else:
            df["vix_close"] = 20.0
            df["vix_ret"]   = 0.0
    except Exception:
        df["vix_close"] = 20.0
        df["vix_ret"]   = 0.0

    # ── Crude oil (USO as proxy for WTI) ────────────────────────────────────
    try:
        oil_raw = _market_return("USO", start_str, end_str, date_index)
        df["oil_ret"] = oil_raw.values
        df["oil_spy_corr"] = df["oil_ret"].rolling(20).corr(df["return_1d"]).fillna(0)
    except Exception:
        df["oil_ret"]      = 0.0
        df["oil_spy_corr"] = 0.0
    # ── FinBERT NLP sentiment (PIT historical lookup, added 2026-05-21) ──────
    # Same code path for training AND inference → no train/serve mismatch.
    # Reads from data/sentiment.db.finbert_filings populated by
    # data.etl_finbert_filings (Session A: 8-K + NT-*).
    try:
        from data.alpha_sources import load_finbert_pit
        _fb = load_finbert_pit(ticker, date_index)
        df["finbert_sentiment"]          = _fb["finbert_sentiment"].values
        df["finbert_sentiment_earnings"] = _fb["finbert_sentiment_earnings"].values
        df["finbert_mult"]               = _fb["finbert_mult"].values
    except Exception as _e:
        df["finbert_sentiment"]          = 0.0
        df["finbert_sentiment_earnings"] = 0.0
        df["finbert_mult"]               = 1.0

    # ── 8-K Item code features (Session E Phase 3, May 22 2026) ──────────────
    # PIT-correct loader reads earnings.db.eightk_items populated by edgartools.
    # Same code path training + inference (no train/serve mismatch).
    # Defaults: 0 / 90 days. Foreign filers (no 8-K) get defaults.
    try:
        from data.alpha_sources import load_eightk_pit
        _8k = load_eightk_pit(ticker, date_index)
        df["eightk_exec_change_30d"]        = _8k["eightk_exec_change_30d"].values
        df["eightk_material_agreement_30d"] = _8k["eightk_material_agreement_30d"].values
        df["eightk_reg_fd_30d"]             = _8k["eightk_reg_fd_30d"].values
        df["eightk_other_events_30d"]       = _8k["eightk_other_events_30d"].values
        df["eightk_filings_30d"]            = _8k["eightk_filings_30d"].values
        df["eightk_days_since_last"]        = _8k["eightk_days_since_last"].values
    except Exception as _e:
        df["eightk_exec_change_30d"]        = 0
        df["eightk_material_agreement_30d"] = 0
        df["eightk_reg_fd_30d"]             = 0
        df["eightk_other_events_30d"]       = 0
        df["eightk_filings_30d"]            = 0
        df["eightk_days_since_last"]        = 90

    # ── Revenue growth features (Session E Phase 2, May 22 2026) ─────────────
    # PIT-safe loader reads earnings.db.earnings_surprises.rev_actual
    # populated by data/etl_polygon_revenue.py (Polygon Financials).
    # Foreign filers/ETFs default to 0.0 (no Polygon coverage).
    try:
        from data.alpha_sources import load_rev_growth_pit
        _rg = load_rev_growth_pit(ticker, date_index)
        df["rev_growth_yoy"] = _rg["rev_growth_yoy"].values
        df["rev_growth_qoq"] = _rg["rev_growth_qoq"].values
    except Exception as _e:
        df["rev_growth_yoy"] = 0.0
        df["rev_growth_qoq"] = 0.0

    # ── Analyst revisions — dropped from model 2026-05-21 ────────────────────
    # No free historical source; train/serve mismatch eliminated by dropping.
    # Kept as constants in df so OUTPUT_COLUMNS / prediction_features unchanged.
    df["analyst_upside"]  = 0.0
    df["analyst_buy_pct"] = 0.5
    df["analyst_mult"]    = 1.0

    # ── Options IV skew + PC ratio (daily snapshot via UW) ───────────────────
    # Migration 2026-05-01: replaced yfinance options_flow with UW endpoints
    # to eliminate Pipeline B crashes from yfinance/curl_cffi thread pool corruption.
    # UW provides real per-strike Greeks (skew) and aggregated volume (PC ratio).
    if training_mode:
        df["iv_skew_snap"]  = 0.0
        df["pc_ratio_snap"] = 1.0
    else:
        # IV skew via UW (with Massive fallback for future Advanced upgrade)
        try:
            from features.massive_options import get_25delta_skew_with_fallback
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(get_25delta_skew_with_fallback, ticker)
                try:
                    skew_result = fut.result(timeout=10)
                    iv_skew_val = skew_result.get("skew_25d") or 0.0
                except Exception:
                    iv_skew_val = 0.0
            df["iv_skew_snap"] = iv_skew_val
        except Exception:
            df["iv_skew_snap"] = 0.0

        # PC ratio via UW options-volume endpoint
        try:
            from features.uw_options import get_pc_ratio_uw
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(get_pc_ratio_uw, ticker)
                try:
                    pc_result = fut.result(timeout=10)
                    pc_ratio_val = pc_result.get("pc_ratio") or 1.0
                except Exception:
                    pc_ratio_val = 1.0
            df["pc_ratio_snap"] = pc_ratio_val
        except Exception:
            df["pc_ratio_snap"] = 1.0

    # Sector-relative return (stock 1d return minus its sector ETF return)
    sector_sym = resolve_sector_etf(ticker)
    try:
        import signal as _signal
        def _timeout_handler(signum, frame): raise TimeoutError()
        _signal.signal(_signal.SIGALRM, _timeout_handler)
        _signal.alarm(8)  # 8 second timeout
        try:
            sec_ret = _market_return(sector_sym, start_str, end_str, date_index)
            df["sector_rel_ret"] = df["return_1d"].values - sec_ret.values
        finally:
            _signal.alarm(0)
    except Exception as _sec_e:
        # was SILENT: an 8s SIGALRM timeout wrote a real-looking 0.0
        # that the model consumed as data. Value kept (changing it to
        # NaN would alter model input); it just stops being invisible.
        log.warning("sector_rel_ret failed for %s vs %s (%s) -- set 0.0",
                    ticker, sector_sym, type(_sec_e).__name__)
        df["sector_rel_ret"] = 0.0

    # Calendar effects
    df["day_of_week"]  = pd.to_datetime(df["date"]).dt.dayofweek   # 0=Mon 4=Fri
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    # ── 11b. Regime / credit features ───────────────────────────────────────
    # VIX 5d above 25 binary flag
    try:
        _vix_s2 = pd.Series(df["vix_close"].values, index=df.index)
        df["vix_5d_above_25"] = (_vix_s2 > 25).rolling(5).min().fillna(0).astype(int)
    except Exception:
        df["vix_5d_above_25"] = 0

    # SMH 60d momentum (semiconductor ETF regime)
    try:
        _smh = _market_return("SMH", start_str, end_str, date_index)
        _smh_s = pd.Series(_smh.values, index=df.index)
        df["semi_etf_momentum_60d"] = _smh_s.rolling(60).sum().fillna(0.0)
    except Exception:
        df["semi_etf_momentum_60d"] = 0.0

    # IGV vs SPY 30d spread (software sector vs market)
    try:
        _igv = _market_return("IGV", start_str, end_str, date_index)
        _igv_s = pd.Series(_igv.values, index=df.index)
        _spy_s = pd.Series(spy.values, index=df.index)
        df["igv_vs_sp500_ret_30d"] = (
            _igv_s.rolling(30).sum() - _spy_s.rolling(30).sum()
        ).fillna(0.0)
    except Exception:
        df["igv_vs_sp500_ret_30d"] = 0.0

    # Credit stress proxy: LQD vs HYG 30d spread
    try:
        _lqd = _market_return("LQD", start_str, end_str, date_index)
        _hyg = _market_return("HYG", start_str, end_str, date_index)
        _lqd_s = pd.Series(_lqd.values, index=df.index)
        _hyg_s = pd.Series(_hyg.values, index=df.index)
        df["lqd_hyg_spread"] = (
            _lqd_s.rolling(30).sum() - _hyg_s.rolling(30).sum()
        ).fillna(0.0)
    except Exception:
        df["lqd_hyg_spread"] = 0.0

    # ── 11c. Institutional darkpool features (UW Lee-Ready flow) ──────────────
    # New data axis: signed institutional flow from UW darkpool prints.
    # PIT-honest -- institutional_features uses STRICT trade_date < as_of_date.
    # Reads local SQLite (institutional_trades.db), NOT a live API, so the same
    # path is correct in training_mode and live mode (no branch needed).
    #
    # DELIBERATE HOUSE-STYLE DEVIATION: surrounding blocks fall back to 0.0.
    # This block falls back to np.nan because:
    #   (1) inst_block_buy_sell_7d's neutral is 1.0, not 0.0 -- a 0.0 fill would
    #       code as "extremely bearish blocks" and mislead the model.
    #   (2) XGBoost handles NaN natively (learns optimal split direction).
    #   (3) ~5% of rows have no institutional data; NaN preserves "unknown".
    # MUST run before the OUTPUT_COLUMNS enforcement below.
    # PIT-correct: load per-row, not single broadcast (fix 2026-05-21)
    if _INST_FEATURES_ENABLED:
        try:
            from features.institutional_features import load_institutional_features_pit
            _inst_df = load_institutional_features_pit(ticker, date_index)
            for _col in _INST_FEATURE_COLS:
                df[_col] = _inst_df[_col].values
        except Exception:
            for _col in _INST_FEATURE_COLS:
                df[_col] = np.nan

    # -- 11c-2. PIT fundamentals from fundamentals.db (Jun 11 2026, Track C wire) --
    # Quarterly step-functions, strict filed_date < as_of (no same-day leak).
    # NaN before first filing = honest unknown. Fail-loud pattern per Rule 1.
    _FUND_FEATURE_COLS = ["fund_gp_assets", "fund_op_equity", "fund_ni_margin",
                          "fund_bm", "fund_ep"]
    if _FUND_FEATURES_ENABLED:
        try:
            from features.fundamental_features import load_fundamental_features_pit
            _close_ser = df["close"] if "close" in df.columns else None
            if _close_ser is not None:
                _close_ser = pd.Series(_close_ser.values, index=date_index)
            _fund_df = load_fundamental_features_pit(ticker, date_index, close=_close_ser)
            for _col in _FUND_FEATURE_COLS:
                df[_col] = _fund_df[_col].values
        except Exception as _fund_e:
            import logging as _flg
            _flg.getLogger(__name__).error(f"fundamental_features failed {ticker}: {_fund_e!r}")
            for _col in _FUND_FEATURE_COLS:
                df[_col] = np.nan

    # ── 11d. Phase 1 D — interaction/normalized features (May 25 2026) ───────
    # Per-ticker only (no train/serve mismatch risk). Informed by A8 finding.
    try:
        # Option A: raw-value interactions (heavy-tailed but XGB can handle)
        _vol = df["volatility_10d"].fillna(0.0)
        _short = df["short_pct_float"].fillna(0.0)
        _rev = df["rev_growth_yoy"].fillna(0.0)
        _low52 = df["low_52w_ratio"].fillna(1.0)  # 1.0 = neutral (1x from 52w low)

        df["vol_x_short"] = _vol * _short
        df["rev_x_low52w"] = _rev * _low52

        # Option C: rolling self-rank (where does this ticker sit in its own recent history)
        # NOTE: short_self_rank removed — short_pct_float is constant per ticker.
        df["vol_10d_self_rank"] = _vol.rolling(252, min_periods=20).rank(pct=True).fillna(0.5)

        # Option D: rolling z-score (deviation from ticker's own baseline)
        # NOTE: short_zscore_60d removed — short_pct_float is constant per ticker.
        _vol_mean = _vol.rolling(60, min_periods=20).mean()
        _vol_std = _vol.rolling(60, min_periods=20).std().replace(0, np.nan)
        df["vol_zscore_60d"] = ((_vol - _vol_mean) / _vol_std).fillna(0.0).clip(-5, 5)

        # Option E: binary squeeze setup indicator
        df["is_squeeze_setup"] = ((_vol > 0.04) & (_short > 0.10)).astype(float)

        # ── P3.5 panel transforms (gated; match alpha_transformations defs) ──
        if _PANEL_TRANSFORMS_ENABLED:
            def _ts_argmax(s, w):
                return s.rolling(w, min_periods=1).apply(
                    lambda x: float(len(x) - 1 - x.values.argmax())
                    if not __import__("numpy").isnan(x).all() else float("nan"),
                    raw=False)
            df["ma_20__ts_std__w5"] = df["ma_20"].rolling(5, min_periods=2).std()
            df["ma_10__ts_std__w10"] = df["ma_10"].rolling(10, min_periods=5).std()
            df["rsi_14__ts_mean__w20"] = df["rsi_14"].rolling(20, min_periods=10).mean()
            df["bb_upper__ts_delta__w20"] = df["bb_upper"].diff(20)
            df["post_earnings_3d__ts_mean__w10"] = df["post_earnings_3d"].rolling(10, min_periods=5).mean()
            df["rev_growth_qoq__ts_std__w10"] = df["rev_growth_qoq"].rolling(10, min_periods=5).std()
            df["is_squeeze_setup__ts_argmax__w20"] = _ts_argmax(df["is_squeeze_setup"], 20)
    except Exception as _e:
        # Fail loud per Rule #1 (b)
        import logging as _lg
        _lg.getLogger(__name__).error(f"interaction features failed for {ticker}: {_e}")
        for _col in ["vol_x_short", "rev_x_low52w", "vol_10d_self_rank",
                     "vol_zscore_60d", "is_squeeze_setup"]:
            df[_col] = 0.0
        if _PANEL_TRANSFORMS_ENABLED:
            for _col in _PANEL_TRANSFORM_COLS:
                df[_col] = 0.0

    # ── 12. Enforce output schema ─────────────────────────────────────────────
    # Add any missing columns as 0.0
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[OUTPUT_COLUMNS]

    # Drop warm-up NaNs (first ~26 rows from EMA/BB calculations)
    required_non_null = ["close", "ma_20", "rsi_14", "macd", "bb_upper"]
    df = df.dropna(subset=required_non_null).reset_index(drop=True)

    # ── R1 refactor (May 26 2026): attach feature_cols list as metadata ──
    # Downstream callers (predict_proba, train_ensemble) can use this to
    # select only model-input columns, suppressing the extras_columns warning
    # for known non-feature columns like close, volume, macd_signal, etc.
    # Note: df.attrs is fragile — gets stripped by many pandas ops (groupby,
    # merge, etc.). predict_proba MUST have a fallback to self.feature_cols.
    # Join A8 prob_top_decile from OOS panel (Phase 2A, May 27 2026)
    # When ML_QUANT_A8_FEATURE=1, lookup a8_prob from data/a8_oos_panel.parquet
    # Missing values fall back to 0.10 (universe top-decile base rate)
    if _A8_FEATURE_ENABLED:
        try:
            _a8_panel = _load_a8_panel()
            if len(_a8_panel) > 0:
                # _a8_panel is a Series indexed by (ticker, date)
                # df has a date column; lookup per row
                _ticker_upper = ticker.upper()
                df["a8_prob_top_decile"] = df["date"].apply(
                    lambda d: _a8_panel.get((_ticker_upper, d.date() if hasattr(d, 'date') else d), 0.10)
                )
            else:
                df["a8_prob_top_decile"] = 0.10  # fallback when panel missing
        except Exception as _a8_e:
            import logging as _lg
            _lg.getLogger(__name__).error(f"A8 join failed for {ticker}: {_a8_e}")
            df["a8_prob_top_decile"] = 0.10

    # Compute missing-value indicators for sparse features (Phase 2)
    if _MISSING_INDICATORS_ENABLED:
        for _sparse_col in _SPARSE_FEATURE_COLS:
            _ind_col = f"{_sparse_col}_has_value"
            if _sparse_col in df.columns:
                df[_ind_col] = df[_sparse_col].notna().astype(int)
            else:
                df[_ind_col] = 0  # source feature absent → has_value=0

    try:
        from models.classifier import FEATURE_COLUMNS as _FC
        _fc_set = set(_FC)
        # feature_cols = the MODEL INPUTS (97 cols when inst flag on, 101 with indicators)
        df.attrs['feature_cols'] = list(_FC)
        # output_only_cols = OUTPUT_COLUMNS - FEATURE_COLUMNS - {'date','ticker'}
        # These are diagnostic/dashboard cols that downstream models should NOT warn about
        df.attrs['output_only_cols'] = [c for c in df.columns if c not in _fc_set and c not in ('date', 'ticker')]
        df.attrs['feature_cols_set_by'] = 'build_feature_dataframe'
    except Exception:
        pass  # If FEATURE_COLUMNS import fails, just skip — non-critical metadata

    return df


def add_forecast_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Add binary classification targets to a feature DataFrame.

    For each horizon h, adds:
      target_{h}d = 1 if close[t+h] > close[t], else 0

    These are TRUE forward returns — no lookahead if you train only on rows
    where target is not NaN (i.e. exclude the last h rows).
    """
    df = df.copy()
    for h in horizons:
        df[f"target_{h}d"] = (df["close"].shift(-h) > df["close"]).astype(float)
        df.loc[df.index[-h:], f"target_{h}d"] = np.nan   # last h rows have no target
    return df
