#!/usr/bin/env python3
"""Patch monitor_earnings.py: swap OPEN -> OKLO/QUBT/CRWD/SNOW/NVMI. Idempotent."""
import ast, re, sys
from pathlib import Path

PATH = Path("scripts/monitor_earnings.py")
if not PATH.exists():
    sys.exit(f"ERROR: {PATH} not found. cd to ~/Desktop/ML_Quant_Fund first.")

src = PATH.read_text()
orig = src

# Fix 1: critical SyntaxError — DDOG missing }, before OKLO
broken = ('        "news_search_term": "Datadog DDOG",\n'
          '        "OKLO": {\n'
          '        "sector_etf": "URA",')
fixed = ('        "news_search_term": "Datadog DDOG",\n'
         '    },\n'
         '    "OKLO": {\n'
         '        "sector_etf": "URA",')
src = src.replace(broken, fixed)

# Fix 2: SECTOR_COHORTS
new_cohorts = '''SECTOR_COHORTS: dict[str, list[str]] = {
    "NVDA": ["AVGO", "AMD", "MRVL", "TSM"],
    "SMCI": ["DELL", "HPE", "ANET", "AVGO"],
    "DDOG": ["MDB", "SNOW", "NET", "TEAM"],
    "OKLO": ["SMR", "NNE", "LEU", "BWXT"],
    "QUBT": ["IONQ", "QBTS", "RGTI", "ARQQ"],
    "CRWD": ["PANW", "ZS", "S", "FTNT"],
    "SNOW": ["DDOG", "MDB", "NET", "TEAM"],
    "NVMI": ["KLAC", "AMAT", "LRCX", "ONTO"],
}'''
src = re.sub(r"SECTOR_COHORTS:\s*dict\[str,\s*list\[str\]\]\s*=\s*\{[^}]*\}",
             new_cohorts, src, flags=re.DOTALL)

# Fix 3: TICKER_MACRO_CONTEXT
new_macro = '''TICKER_MACRO_CONTEXT: dict[str, list[str]] = {
    "NVDA": [
        "AI chip China export controls",
        "Taiwan TSMC semiconductor",
    ],
    "SMCI": [
        "AI server demand hyperscaler capex",
    ],
    "DDOG": [
        "cloud spending enterprise software AI budget",
    ],
    "OKLO": [
        "small modular reactor SMR nuclear policy",
        "AI data center power demand",
    ],
    "QUBT": [
        "quantum computing breakthrough government funding",
    ],
    "CRWD": [
        "cybersecurity spending CISO budget",
        "ransomware breach incident",
    ],
    "SNOW": [
        "cloud spending enterprise software AI budget",
    ],
    "NVMI": [
        "semiconductor capex foundry equipment",
        "TSMC Samsung Intel fab investment",
    ],
}'''
src = re.sub(r"TICKER_MACRO_CONTEXT:\s*dict\[str,\s*list\[str\]\]\s*=\s*\{.*?^\}",
             new_macro, src, flags=re.DOTALL | re.MULTILINE)

# Fix 4: cosmetic comments
for old, new in [
    ("# default 4 tickers", "# default 8 tickers"),
    ("Default ticker universe: NVDA, DDOG, SMCI, OPEN.",
     "Default ticker universe: NVDA, DDOG, SMCI, OKLO, QUBT, CRWD, SNOW, NVMI."),
    ("to small-cap (OPEN ~$3, several million shares/day).",
     "to small-cap (NVMI, low ADV)."),
    ("routine for NVDA (3T market cap) but huge for OPEN (3B market cap).",
     "routine for NVDA (3T market cap) but huge for NVMI (~$7B market cap)."),
    ("controls; OPEN cares about mortgage rates.",
     "controls; NVMI cares about semi-cycle headlines."),
    ("E.g. NVDA gets China chip export controls; OPEN gets mortgage rates.",
     "E.g. NVDA gets China chip export controls; NVMI gets semi-cycle headlines."),
]:
    src = src.replace(old, new)

if src == orig:
    print("No changes needed — already up to date.")
else:
    PATH.with_suffix(".py.bak").write_text(orig)
    PATH.write_text(src)
    print(f"Patched. Backup at {PATH.with_suffix('.py.bak')}")

try:
    ast.parse(src)
    print("Syntax OK ✓")
except SyntaxError as e:
    sys.exit(f"⚠️ Still broken at line {e.lineno}: {e.msg}")
