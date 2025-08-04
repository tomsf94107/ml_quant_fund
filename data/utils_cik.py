# utils_cik.py (safe local version)
# ──────────────────────────────────────────────
import json
import os

CACHE_FILE = "data/cik_to_ticker.json"

def load_cik_to_ticker_map():
    """
    Load CIK → ticker mapping from local cache file only.
    Assumes cik_to_ticker.json exists in /data.
    """
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"❌ CIK cache file not found: {CACHE_FILE}")
    
    with open(CACHE_FILE, "r") as f:
        data = json.load(f)
    
    return {str(v["cik_str"]).zfill(10): v["ticker"] for v in data.values()}
