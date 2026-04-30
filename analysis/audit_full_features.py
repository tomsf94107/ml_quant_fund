"""
analysis/audit_full_features.py — Leak audit on the FULL 79-feature production set.

Bypasses the prediction_features table (which only logs 27 features) and uses
build_feature_dataframe() directly to assemble the same feature set the
production model trains on. Computes per-feature correlation with target_1d,
target_3d, target_5d.

Severity rules (same as analysis/walk_forward.py):
    |corr| < 0.10  expected for an honest equity-prediction feature
    |corr| 0.10–0.30  worth attention but plausible
    |corr| 0.30–0.50  suspicious — verify how feature is computed
    |corr| > 0.50  almost certainly a leak

Drop in: ml_quant_fund/analysis/audit_full_features.py

Usage:
    # Default: 8 tickers (mix of liquid + sector representatives), all horizons
    ~/.pyenv/versions/ml_quant_310/bin/python -m analysis.audit_full_features

    # Specific tickers, faster
    ~/.pyenv/versions/ml_quant_310/bin/python -m analysis.audit_full_features --tickers NVDA,AAPL,MSFT,SPY

    # All tickers (slow, definitive)
    ~/.pyenv/versions/ml_quant_310/bin/python -m analysis.audit_full_features --all-tickers --csv full_audit.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.builder import build_feature_dataframe, add_forecast_targets
from models.classifier import FEATURE_COLUMNS


# Reasonable default: a mix of liquid majors + sector representatives.
# Enough variety to catch leaks that show up across asset types,
# small enough to run in a few minutes.
DEFAULT_TICKERS = [
    "NVDA",   # core silicon
    "AAPL",   # consumer tech
    "MSFT",   # hyperscaler
    "JPM",    # financials
    "XOM",    # energy
    "PFE",    # healthcare
    "SPY",    # market ETF
    "MU",     # memory
]


def load_panel_for_tickers(
    tickers: List[str],
    start_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the full 79-feature dataframe + targets for each ticker, concat."""
    dfs = []
    for i, t in enumerate(tickers):
        if verbose:
            print(f"  [{i+1}/{len(tickers)}] building features for {t}...",
                  end="", flush=True)
        try:
            kwargs = {}
            if start_date:
                kwargs["start_date"] = start_date
            df = build_feature_dataframe(t, **kwargs)
            df = add_forecast_targets(df)
            df["__ticker"] = t
            dfs.append(df)
            if verbose:
                print(f" {len(df)} rows, {len(df.columns)} cols")
        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def audit_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ("target_1d", "target_3d", "target_5d"),
) -> pd.DataFrame:
    """Per-feature correlation with each target. Severity-graded."""
    rows = []

    for feat in feature_cols:
        if feat not in df.columns:
            rows.append({
                "feature": feat,
                "status": "MISSING_FROM_PANEL",
                "n_obs": 0,
                "corr_target_1d": None,
                "corr_target_3d": None,
                "corr_target_5d": None,
                "max_abs_corr": None,
                "severity": "UNKNOWN",
            })
            continue

        x = df[feat]
        if not pd.api.types.is_numeric_dtype(x):
            rows.append({
                "feature": feat,
                "status": "NON_NUMERIC",
                "n_obs": int(x.notna().sum()),
                "corr_target_1d": None,
                "corr_target_3d": None,
                "corr_target_5d": None,
                "max_abs_corr": None,
                "severity": "UNKNOWN",
            })
            continue

        x = x.astype(float)
        if x.notna().sum() < 100 or x.std() == 0 or x.std() != x.std():
            rows.append({
                "feature": feat,
                "status": "INSUFFICIENT_DATA",
                "n_obs": int(x.notna().sum()),
                "corr_target_1d": None,
                "corr_target_3d": None,
                "corr_target_5d": None,
                "max_abs_corr": None,
                "severity": "UNKNOWN",
            })
            continue

        corrs = {}
        for tc in target_cols:
            if tc not in df.columns:
                corrs[tc] = None
                continue
            c = x.corr(df[tc].astype(float))
            corrs[tc] = round(c, 4) if c == c else None

        valid_corrs = [abs(c) for c in corrs.values() if c is not None]
        max_abs = max(valid_corrs) if valid_corrs else 0.0

        if max_abs > 0.50:
            severity = "LEAK_LIKELY"
        elif max_abs > 0.30:
            severity = "SUSPICIOUS"
        elif max_abs > 0.10:
            severity = "ELEVATED"
        else:
            severity = "OK"

        rows.append({
            "feature": feat,
            "status": "OK",
            "n_obs": int(x.notna().sum()),
            "corr_target_1d": corrs.get("target_1d"),
            "corr_target_3d": corrs.get("target_3d"),
            "corr_target_5d": corrs.get("target_5d"),
            "max_abs_corr": round(max_abs, 4),
            "severity": severity,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["severity", "max_abs_corr"],
        ascending=[True, False],
        key=lambda s: s.map({
            "LEAK_LIKELY": 0, "SUSPICIOUS": 1, "ELEVATED": 2,
            "OK": 3, "UNKNOWN": 4,
        }) if s.name == "severity" else s,
        na_position="last",
    )
    return out


def print_report(audit: pd.DataFrame, n_rows_data: int, n_tickers: int) -> None:
    print("\n" + "=" * 76)
    print("FULL FEATURE LEAK AUDIT")
    print("=" * 76)
    print(f"Audited {len(audit)} features over {n_rows_data} rows × {n_tickers} tickers")

    by_sev = audit.groupby("severity").size()
    print("\nFeatures by severity:")
    for sev in ["LEAK_LIKELY", "SUSPICIOUS", "ELEVATED", "OK", "UNKNOWN"]:
        n = int(by_sev.get(sev, 0))
        marker = "⚠" if sev in ("LEAK_LIKELY", "SUSPICIOUS") else " "
        print(f"  {marker} {sev:14s} {n}")

    leak_likely = audit[audit["severity"] == "LEAK_LIKELY"]
    if not leak_likely.empty:
        print("\n" + "⚠" * 38)
        print("LEAK_LIKELY features (|corr| > 0.50 — fix immediately):")
        print("⚠" * 38)
        print(leak_likely[[
            "feature", "n_obs", "corr_target_1d", "corr_target_3d",
            "corr_target_5d", "max_abs_corr",
        ]].to_string(index=False))

    suspicious = audit[audit["severity"] == "SUSPICIOUS"]
    if not suspicious.empty:
        print("\n" + "⚠" * 38)
        print("SUSPICIOUS features (|corr| 0.30–0.50 — verify computation):")
        print("⚠" * 38)
        print(suspicious[[
            "feature", "n_obs", "corr_target_1d", "corr_target_3d",
            "corr_target_5d", "max_abs_corr",
        ]].to_string(index=False))

    elevated = audit[audit["severity"] == "ELEVATED"]
    if not elevated.empty:
        print(f"\nELEVATED features (|corr| 0.10–0.30, n={len(elevated)}):")
        print(elevated[[
            "feature", "n_obs", "corr_target_1d", "corr_target_3d",
            "corr_target_5d", "max_abs_corr",
        ]].to_string(index=False))

    unknown = audit[audit["severity"] == "UNKNOWN"]
    if not unknown.empty:
        print(f"\n{len(unknown)} features couldn't be evaluated:")
        for _, r in unknown.iterrows():
            print(f"  {r['feature']:30s} {r['status']}  (n={r['n_obs']})")

    print("\n" + "=" * 76)
    print("INTERPRETATION GUIDE")
    print("=" * 76)
    if not leak_likely.empty or not suspicious.empty:
        print("⚠ LEAKS DETECTED. Likely paths to investigate, by feature category:")
        print("   - analyst_*       → are analyst targets revised retroactively?")
        print("                       Use point-in-time data, not as-of-today.")
        print("   - eps_surprise    → was 'surprise' computed using post-earnings price?")
        print("                       Should use pre-announcement consensus only.")
        print("   - sentiment_*     → is sentiment computed using same-day news?")
        print("                       FinBERT scores broken Mar 5+ per other chat.")
        print("   - congress_*      → is filing_date or trade_date being used?")
        print("                       Trade dates leak; use filing dates only.")
        print("   - es_overnight    → does this include the prediction-day open?")
        print("                       Should be prior-day futures only.")
        print("   - days_to_earnings → does this know when earnings ARE, post-fact?")
        print("                       Should use scheduled-as-of-now only.")
    else:
        print("✓ No leaks detected at |corr| > 0.30 threshold.")
        print("  If your live AUC is still much lower than training AUC, the cause")
        print("  is likely overfitting on the model side, not feature contamination.")
        print("  Action: regularize the model (already validated in walk-forward).")


def main() -> None:
    p = argparse.ArgumentParser(description="Full 79-feature leak audit.")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated ticker list (default: 8 representatives)")
    p.add_argument("--all-tickers", action="store_true",
                   help="Use every ticker in tickers.txt (slow but definitive)")
    p.add_argument("--start-date", default=None,
                   help="ISO start date passed to build_feature_dataframe")
    p.add_argument("--csv", default=None,
                   help="Optional CSV output path")
    args = p.parse_args()

    if args.all_tickers:
        with open("tickers.txt") as f:
            tickers = [t.strip().upper() for t in f.read().split() if t.strip()]
        print(f"Auditing all {len(tickers)} tickers from tickers.txt")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS
        print(f"Using default set: {tickers}")

    print(f"\nBuilding feature dataframes...")
    panel = load_panel_for_tickers(tickers, start_date=args.start_date)

    if panel.empty:
        print("No panel built. Check that build_feature_dataframe works for these tickers.")
        return

    print(f"\nCombined panel: {len(panel)} rows × {len(panel.columns)} columns")
    print(f"Production feature list: {len(FEATURE_COLUMNS)} features")
    overlap = sum(1 for f in FEATURE_COLUMNS if f in panel.columns)
    print(f"Features available in panel: {overlap}/{len(FEATURE_COLUMNS)}")

    if overlap < len(FEATURE_COLUMNS):
        missing = [f for f in FEATURE_COLUMNS if f not in panel.columns]
        print(f"\n⚠ {len(missing)} production features missing from panel:")
        print(f"  {missing}")
        print(f"  These will be flagged as MISSING_FROM_PANEL in the report.")

    audit = audit_correlations(panel, FEATURE_COLUMNS)
    print_report(audit, len(panel), len(tickers))

    if args.csv:
        audit.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
