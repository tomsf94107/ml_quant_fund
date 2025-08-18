# data/convert_cik_map.py
import json, os

# Point at this script’s directory
data_dir = os.path.dirname(__file__)
raw_file = os.path.join(data_dir, "cik_to_ticker.json")
flat_file = raw_file  # overwrite same file

# Load the raw SEC-format JSON
with open(raw_file, "r") as f:
    raw = json.load(f)

# Build a flat CIK→ticker map (cast CIK to string)
flat = {
    str(entry["cik_str"]).lstrip("0"): entry["ticker"]
    for entry in raw.values()
}

# Write it back
with open(flat_file, "w") as f:
    json.dump(flat, f, indent=2)

print("Rewrote cik_to_ticker.json as flat CIK→ticker map.")
