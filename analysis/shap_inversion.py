"""
analysis/shap_inversion.py

SHAP analysis to diagnose calibration inversion in h=3 and h=5 horizons.
Uses model-agnostic Explainer that works with EnsembleResult's predict_proba.
"""
from __future__ import annotations
import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from features.builder import build_feature_dataframe
from models.classifier import FEATURE_COLUMNS

DB = ROOT / "accuracy.db"
MODELS = ROOT / "models" / "saved"
OUT = ROOT / "baselines"
OUT.mkdir(exist_ok=True)


def load_high_conf_samples(horizon: int, n_per_group: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB))
    df = pd.read_sql(f"""
        SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up,
               o.actual_return, o.actual_up
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker
         AND p.prediction_date = o.prediction_date
         AND p.horizon = o.horizon
        WHERE p.horizon = {horizon}
          AND p.prob_up >= 0.70
          AND o.actual_up IS NOT NULL
        ORDER BY p.prediction_date DESC
    """, conn)
    conn.close()

    wins = df[df['actual_up'] == 1].head(n_per_group).copy()
    losses = df[df['actual_up'] == 0].head(n_per_group).copy()
    print(f"Available: {len(df)} high-conf ({(df['actual_up']==1).sum()} wins, {(df['actual_up']==0).sum()} losses)", flush=True)
    print(f"Sampled:   {len(wins)} wins, {len(losses)} losses", flush=True)
    return pd.concat([wins, losses], ignore_index=True)


def build_features_at(ticker: str, prediction_date: str) -> pd.Series | None:
    try:
        df = build_feature_dataframe(ticker, end_date=prediction_date)
        if df.empty:
            return None
        if hasattr(df.index, 'date'):
            target = pd.to_datetime(prediction_date).date()
            mask = df.index.date == target
            row = df[mask].iloc[-1] if mask.any() else df.iloc[-1]
        else:
            row = df.iloc[-1]
        missing = [c for c in FEATURE_COLUMNS if c not in row.index]
        if missing:
            return None
        return row[FEATURE_COLUMNS]
    except Exception:
        return None


def compute_shap(samples: pd.DataFrame, horizon: int) -> pd.DataFrame:
    import shap

    print(f"\nBuilding feature vectors for {len(samples)} samples (~3s each, ~{len(samples)*3/60:.1f}min total)...", flush=True)
    feature_rows = []
    valid_idx = []
    t0 = time.time()
    for i, row in samples.iterrows():
        feats = build_features_at(row['ticker'], row['prediction_date'])
        if feats is not None:
            feature_rows.append(feats.values)
            valid_idx.append(i)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(samples)}] feature builds, {len(feature_rows)} valid, elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nBuilt {len(feature_rows)}/{len(samples)} feature vectors", flush=True)

    if not feature_rows:
        return pd.DataFrame()

    samples = samples.loc[valid_idx].reset_index(drop=True)
    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    # Build a background sample for KernelExplainer (subset of all features)
    background = shap.sample(X, min(20, len(X)), random_state=42)

    # Group samples by ticker so we load each model once
    print(f"\nComputing SHAP per ticker (KernelExplainer, ~30s/sample)...", flush=True)
    print(f"This is the slow part. Reduce --n-per-group if needed.", flush=True)

    shap_rows = []
    skipped = 0
    last_ticker = None
    explainer = None

    for i, row in samples.iterrows():
        ticker = row['ticker']
        if ticker != last_ticker:
            model_path = MODELS / f"{ticker}_ensemble_{horizon}d.joblib"
            if not model_path.exists():
                skipped += 1
                continue
            try:
                er = joblib.load(model_path)
                # Wrap predict_proba to return prob_up (positive class) only
                def predict_fn(X_arr, _er=er):
                    df_in = pd.DataFrame(X_arr, columns=FEATURE_COLUMNS)
                    proba = _er.predict_proba(df_in)
                    # predict_proba returns [P(0), P(1)] — return P(1) only as 1D
                    return proba[:, 1] if proba.ndim == 2 else proba
                explainer = shap.KernelExplainer(predict_fn, background)
                last_ticker = ticker
                print(f"  Loaded {ticker} model", flush=True)
            except Exception as e:
                skipped += 1
                last_ticker = None
                continue

        if explainer is None:
            skipped += 1
            continue

        try:
            sample_t0 = time.time()
            shap_vals = explainer.shap_values(X.iloc[i:i+1].values, nsamples=50, silent=True)
            shap_vals = np.array(shap_vals).flatten()
            print(f"  [{i+1}/{len(samples)}] {ticker} {row['prediction_date']} "
                  f"actual={row['actual_up']} prob={row['prob_up']:.3f} "
                  f"({time.time()-sample_t0:.1f}s)", flush=True)

            shap_rows.append({
                'ticker': ticker,
                'prediction_date': row['prediction_date'],
                'actual_up': int(row['actual_up']),
                'prob_up': float(row['prob_up']),
                **{f'shap_{c}': float(v) for c, v in zip(FEATURE_COLUMNS, shap_vals)}
            })
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {ticker} SHAP failed: {str(e)[:80]}", flush=True)
            skipped += 1
            continue

    print(f"\nComputed SHAP for {len(shap_rows)} samples ({skipped} skipped)", flush=True)
    return pd.DataFrame(shap_rows)


def analyze(shap_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    wins = shap_df[shap_df['actual_up'] == 1]
    losses = shap_df[shap_df['actual_up'] == 0]

    print(f"\n{'='*80}", flush=True)
    print(f"SHAP ANALYSIS — h={horizon}d", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Wins:   {len(wins)} samples (high-conf BUYs that worked)", flush=True)
    print(f"Losses: {len(losses)} samples (high-conf BUYs that failed)", flush=True)

    feat_cols = [c for c in shap_df.columns if c.startswith('shap_')]
    rows = []
    for col in feat_cols:
        feat = col.replace('shap_', '')
        win_signed = wins[col].mean() if len(wins) else 0
        loss_signed = losses[col].mean() if len(losses) else 0
        win_abs = wins[col].abs().mean() if len(wins) else 0
        loss_abs = losses[col].abs().mean() if len(losses) else 0
        rows.append({
            'feature': feat,
            'win_signed_shap': round(win_signed, 4),
            'loss_signed_shap': round(loss_signed, 4),
            'sign_disagreement': round(abs(win_signed - loss_signed), 4),
            'win_abs_shap': round(win_abs, 4),
            'loss_abs_shap': round(loss_abs, 4),
            'mag_diff_loss_minus_win': round(loss_abs - win_abs, 4),
        })

    out = pd.DataFrame(rows)

    print(f"\nTOP 20 — biggest SIGN DISAGREEMENT (feature pushes BUY differently for wins vs losses):", flush=True)
    top_d = out.sort_values('sign_disagreement', ascending=False).head(20)
    print(top_d.to_string(index=False), flush=True)

    print(f"\nTOP 20 — feature 'shouts louder' for losses than wins (potential misleading signal):", flush=True)
    top_m = out[out['mag_diff_loss_minus_win'] > 0].sort_values('mag_diff_loss_minus_win', ascending=False).head(20)
    print(top_m.to_string(index=False), flush=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, choices=[1, 3, 5])
    parser.add_argument("--n-per-group", type=int, default=30)
    args = parser.parse_args()

    print(f"SHAP inversion analysis — h={args.horizon}d, n_per_group={args.n_per_group}", flush=True)
    print(f"Total compute estimate: ~{args.n_per_group * 2 * 35 / 60:.0f} min for {args.n_per_group * 2} samples", flush=True)

    samples = load_high_conf_samples(args.horizon, args.n_per_group)
    if len(samples) < 4:
        print(f"❌ Not enough samples ({len(samples)}) — need 4+", flush=True)
        return

    shap_df = compute_shap(samples, args.horizon)
    if shap_df.empty:
        print("❌ No SHAP results", flush=True)
        return

    analysis = analyze(shap_df, args.horizon)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    raw_path = OUT / f"shap_inversion_h{args.horizon}_raw_{timestamp}.csv"
    summary_path = OUT / f"shap_inversion_h{args.horizon}_summary_{timestamp}.csv"
    shap_df.to_csv(raw_path, index=False)
    analysis.to_csv(summary_path, index=False)
    print(f"\n✅ Saved raw SHAP: {raw_path}", flush=True)
    print(f"✅ Saved summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
