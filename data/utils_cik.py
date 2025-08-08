import os, json

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cik_to_ticker.json")

def load_cik_to_ticker_map():
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"❌ CIK cache file not found: {CACHE_FILE}")

    with open(CACHE_FILE, "r") as f:
        data = json.load(f)

    # If this is the raw SEC dump (values are dicts with 'cik_str' & 'ticker'), unwrap:
    sample = next(iter(data.values()))
    if isinstance(sample, dict) and "cik_str" in sample and "ticker" in sample:
        return {
            str(v["cik_str"]).lstrip("0"): v["ticker"]
            for v in data.values()
        }

    # Otherwise assume it's already { "1045810": "NVDA", ... }
    return data
