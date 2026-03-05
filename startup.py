# startup.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Cloud startup script.
# Trains models if saved joblib files are missing.
# Called automatically by 1_Dashboard.py on first load.
#
# On Streamlit Cloud:
#   - models/saved/ is empty (joblib files not in git)
#   - This script detects that and trains all models
#   - Takes 10-15 minutes on first deploy
#   - Subsequent loads use cached models (fast)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def models_are_trained(min_models: int = 10) -> bool:
    """Check if enough models exist to run the dashboard."""
    saved = _ROOT / "models" / "saved"
    if not saved.exists():
        return False
    joblibs = list(saved.glob("*.joblib"))
    return len(joblibs) >= min_models


def run_startup(verbose: bool = True) -> bool:
    """
    Train models if missing. Returns True if training ran, False if skipped.
    Called by 1_Dashboard.py before rendering anything.
    """
    if models_are_trained():
        return False   # already trained — skip

    if verbose:
        print("⚙️  No trained models found. Running train_all.py...")
        print("   This takes 10-15 minutes on first deploy.")

    try:
        from models.train_all import train_all, DEFAULT_TICKERS
        from pathlib import Path as P

        # Load tickers from file if available
        tickers_file = _ROOT / "tickers.txt"
        if tickers_file.exists():
            tickers = [t.strip().upper() for t in
                      tickers_file.read_text().splitlines() if t.strip()]
        else:
            tickers = DEFAULT_TICKERS

        results = train_all(tickers=tickers, horizons=[1, 3, 5], verbose=verbose)
        trained  = sum(1 for r in results if not r.get("error"))

        if verbose:
            print(f"✅ Training complete — {trained} models ready")
        return True

    except Exception as e:
        if verbose:
            print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    run_startup(verbose=True)
