"""R1 driver — 8 tickers × 3 horizons, leak-free single-fold eval."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from models.walk_forward import _load_df_for_ticker
from research.leak_free_clf import train_leak_free

TICKERS  = ["MSFT", "NVDA", "TSLA", "AMD", "RZLV", "BYND", "JPM", "XOM"]
HORIZONS = [1, 3, 5]
RESULTS  = Path("research/results"); RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    rows, probs = [], []
    for tk in TICKERS:
        try:
            df = _load_df_for_ticker(tk)
        except Exception as e:
            print(f"{tk}: load failed — {e}"); continue
        for h in HORIZONS:
            try:
                out = train_leak_free(df, f"target_{h}d", tk, h)
            except Exception as e:
                print(f"{tk} h={h}: {e}"); continue
            rows.append({k: v for k, v in out.items() if k not in ("y_test","p_test")})
            for p, y in zip(out["p_test"], out["y_test"]):
                probs.append({"ticker": tk, "horizon": h, "prob": float(p), "actual": int(y)})
            print(f"{tk} h={h}: AUC_cal={out['auc_calibrated']} acc={out['accuracy']} n_te={out['n_test']}")

    pd.DataFrame(rows).to_csv(RESULTS / "r1_leak_free.csv", index=False)
    pd.DataFrame(probs).to_parquet(RESULTS / "r1_test_probs.parquet")
    print(f"\nWrote {RESULTS}/r1_leak_free.csv + r1_test_probs.parquet")


if __name__ == "__main__":
    main()
