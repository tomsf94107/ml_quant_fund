"""
analysis/build_alpha_panel.py
─────────────────────────────────────────────────────────────────────────────
Build full alpha panel via 17-operator explode of base 83-feature universe.

Sprint 2 Stage 2 implementation per Gap_Check_and_Roadmap(04292026.md) Lever 2.

PIPELINE:
  1. Load tickers from tickers.txt (default 125 tickers)
  2. For each ticker, call features.builder.build_feature_dataframe
  3. Stack into wide panels: dict[feat_name → DataFrame(dates × tickers)]
  4. Apply 17 operators from features.alpha_transformations.ALPHA_OPS
     - Time-series ops at windows [5, 10, 20]
     - Cross-sectional ops (no window)
     - Group/neutralization ops using bucket map from tickers_metadata.csv
  5. Stack output into per-date wide DataFrames
  6. Write to data/alpha_panel/YYYY-MM-DD.parquet (one file per date)

OUTPUT FORMAT:
  Each parquet file = wide DataFrame:
    rows    = tickers in panel that date
    columns = alpha names (e.g. "rsi_14__ts_decay_linear__w10")
    values  = float64

NAMING CONVENTION:
  Base feature:        "rsi_14"
  Single op:           "rsi_14__cs_rank"
  Windowed op:         "rsi_14__ts_mean__w10"
  Composed (future):   "rsi_14__cs_rank__ts_mean__w10"

USAGE:
  from analysis.build_alpha_panel import build_alpha_panel, write_alpha_panel
  panel = build_alpha_panel(tickers=["AAPL", "MSFT", ...])
  write_alpha_panel(panel, output_dir="data/alpha_panel/")

  # OR via CLI:
  python scripts/pipeline_D_alpha_panel.sh

DESIGN DECISIONS:
  - Per-date parquet (not single file): atomic writes, easy resume,
    matches WorldQuant's storage pattern
  - Bucket map from tickers_metadata.csv (CSV not DB — source of truth)
  - Group-of-1 buckets passed through unchanged (per group_neutralize)
  - Orphan tickers (not in tickers_metadata.csv) skipped from group ops
"""
from __future__ import annotations
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np

# Path setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from features.alpha_transformations import (
    ALPHA_OPS,
    # Cross-sectional
    cs_rank, cs_zscore, cs_demean,
    # Time-series
    ts_mean, ts_std, ts_rank, ts_delta, ts_max, ts_min,
    ts_argmax, ts_decay_linear,
    # Pointwise
    signed_power, scale,
    # Group
    group_neutralize,
)

DEFAULT_TICKERS_FILE   = ROOT / "tickers.txt"
DEFAULT_METADATA_FILE  = ROOT / "tickers_metadata.csv"
DEFAULT_OUTPUT_DIR     = ROOT / "data" / "alpha_panel"
NON_FEATURE_COLS = {
    "date", "ticker", "__ticker__",
    "open", "high", "low", "close", "volume", "adj_close",
    "Open", "High", "Low", "Close", "Volume", "Adj Close",
}

# Operators that DON'T fit explode_panel because they need special args
# - cs_mean / cs_std: return Series not DataFrame (used for composition)
# - ts_sma: alias of ts_mean
# - ts_corr: binary op, needs two panels — skip until Stage 2.5
# - indneutralize: alias of group_neutralize
SKIP_FROM_EXPLODE = {"cs_mean", "cs_std", "ts_sma", "ts_corr", "indneutralize"}

# build_feature_dataframe drops rows with NaN in ma_20, bb_upper, etc.
# Those features need ~26 trading days of price history to compute.
# Use 60 trading days minimum for safety (covers MA50 + warmup buffer).
WARMUP_MIN_DAYS = 60

log = logging.getLogger("build_alpha_panel")


def load_bucket_map(metadata_file: Path = DEFAULT_METADATA_FILE) -> dict:
    """Load ticker → bucket map from tickers_metadata.csv."""
    df = pd.read_csv(metadata_file)
    return dict(zip(df["ticker"], df["bucket"]))


def load_tickers(tickers_file: Path = DEFAULT_TICKERS_FILE) -> list[str]:
    """Load ticker universe."""
    return [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]


def build_panels_from_tickers(
    tickers: list[str],
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build per-feature wide panels by calling build_feature_dataframe
    for each ticker and stacking.

    Returns: dict[feature_name → DataFrame(dates × tickers)]
    """
    from features.builder import build_feature_dataframe
    from datetime import datetime as _dt
    
    # Warmup guard: build_feature_dataframe drops rows missing MA20/BB/etc.
    # If start_date is too recent, ALL tickers return empty silently.
    _start = _dt.strptime(start_date, "%Y-%m-%d")
    _today = _dt.now() if end_date is None else _dt.strptime(end_date, "%Y-%m-%d")
    _calendar_days = (_today - _start).days
    _trading_days_est = int(_calendar_days * 5 / 7)  # rough trading-day estimate
    if _trading_days_est < WARMUP_MIN_DAYS:
        raise ValueError(
            f"start_date={start_date} too recent: only ~{_trading_days_est} "
            f"trading days to {end_date or 'today'}. build_feature_dataframe "
            f"requires ≥{WARMUP_MIN_DAYS} for MA/BB warmup. "
            f"Use start_date ≤ {(_today - __import__('datetime').timedelta(days=WARMUP_MIN_DAYS * 2)).date()}."
        )

    per_ticker = {}
    feature_cols = None
    for i, t in enumerate(tickers, 1):
        try:
            t_start = time.time()
            df = build_feature_dataframe(t, start_date=start_date,
                                         end_date=end_date)
            if df is None or df.empty:
                if verbose:
                    log.warning(f"[{i}/{len(tickers)}] {t}: empty, skip")
                continue
            # Standard column set on first valid ticker
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
            # Ensure date index
            if "date" in df.columns:
                df = df.set_index("date")
            per_ticker[t] = df[feature_cols]
            if verbose:
                log.info(f"[{i}/{len(tickers)}] {t}: {len(df)} rows "
                         f"({time.time()-t_start:.1f}s)")
        except Exception as e:
            log.warning(f"[{i}/{len(tickers)}] {t}: FAILED {e}")

    if not per_ticker or feature_cols is None:
        raise RuntimeError(
            f"ALL {len(tickers)} tickers returned empty from build_feature_dataframe. "
            f"Likely causes: start_date={start_date} too recent for warmup, "
            f"or upstream data feeds unavailable. Check pipeline_A log."
        )
    
    # Warn if many tickers failed (potential systemic issue)
    fail_pct = (len(tickers) - len(per_ticker)) / len(tickers) * 100
    if fail_pct > 50:
        log.warning(
            f"⚠️  {len(tickers) - len(per_ticker)}/{len(tickers)} tickers empty "
            f"({fail_pct:.0f}%). Possible data feed issue."
        )

    # Stack into wide panels per feature
    panels = {}
    for feat in feature_cols:
        cols = {}
        for tk, df in per_ticker.items():
            if feat in df.columns:
                cols[tk] = df[feat]
        panels[feat] = pd.DataFrame(cols)

    return panels


def explode_panels(
    panels: dict[str, pd.DataFrame],
    bucket_map: Optional[dict] = None,
    ts_windows: tuple[int, ...] = (5, 10, 20),
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Apply all 17 operators to base panels. Returns expanded alpha dict.

    Args:
      panels:     base feature panels from build_panels_from_tickers
      bucket_map: ticker → bucket for group_neutralize (optional)
      ts_windows: rolling windows for ts_* ops (default 5, 10, 20)
      verbose:    log per-feature progress

    Returns: dict[alpha_name → DataFrame] with all transformations applied
    """
    alphas: dict[str, pd.DataFrame] = {}
    n_features = len(panels)

    for fi, (feat_name, panel) in enumerate(panels.items(), 1):
        if verbose:
            log.info(f"[{fi}/{n_features}] Exploding {feat_name}...")

        # Pointwise
        try:
            alphas[f"{feat_name}__signed_power_p05"] = signed_power(panel, p=0.5)
        except Exception as e:
            log.warning(f"  signed_power failed: {e}")

        try:
            alphas[f"{feat_name}__scale"] = scale(panel)
        except Exception as e:
            log.warning(f"  scale failed: {e}")

        # Cross-sectional (return DataFrames)
        try:
            alphas[f"{feat_name}__cs_rank"] = cs_rank(panel)
        except Exception as e:
            log.warning(f"  cs_rank failed: {e}")

        try:
            alphas[f"{feat_name}__cs_zscore"] = cs_zscore(panel)
        except Exception as e:
            log.warning(f"  cs_zscore failed: {e}")

        try:
            alphas[f"{feat_name}__cs_demean"] = cs_demean(panel)
        except Exception as e:
            log.warning(f"  cs_demean failed: {e}")

        # Time-series operators × windows
        ts_ops = {
            "ts_mean":         ts_mean,
            "ts_std":          ts_std,
            "ts_rank":         ts_rank,
            "ts_delta":        ts_delta,
            "ts_max":          ts_max,
            "ts_min":          ts_min,
            "ts_argmax":       ts_argmax,
            "ts_decay_linear": ts_decay_linear,
        }
        for op_name, op_fn in ts_ops.items():
            for w in ts_windows:
                try:
                    alphas[f"{feat_name}__{op_name}__w{w}"] = op_fn(panel, window=w)
                except Exception as e:
                    log.warning(f"  {op_name}(w={w}) failed: {e}")

        # Group neutralization (one variant)
        if bucket_map is not None:
            try:
                alphas[f"{feat_name}__group_neutralize"] = group_neutralize(
                    panel, bucket_map
                )
            except Exception as e:
                log.warning(f"  group_neutralize failed: {e}")

    return alphas


def write_alpha_panel(
    alphas: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_dates: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Write alpha dict to per-date parquet files.

    For each date, builds a wide DataFrame:
      rows   = tickers
      cols   = alpha names
      values = the alpha value for that (date, ticker) pair

    Args:
      alphas:        dict[alpha_name → DataFrame(dates × tickers)]
      output_dir:    where to write parquet files
      target_dates:  if None, writes all dates in the panel. If specified,
                     writes only those (e.g. last 5 trading days).
      verbose:       log per-date progress

    Returns: dict with summary {dates_written, alphas_written, output_files}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not alphas:
        return {"dates_written": 0, "alphas_written": 0, "output_files": []}

    # Determine dates to write
    sample_df = next(iter(alphas.values()))
    all_dates = sample_df.index
    if target_dates is not None:
        target_set = set(pd.to_datetime(target_dates))
        all_dates = [d for d in all_dates if d in target_set]

    output_files = []
    dates_written = 0
    for d in all_dates:
        # Build per-date wide DataFrame: rows=tickers, cols=alpha_names
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        per_date_cols = {}
        for alpha_name, panel in alphas.items():
            if d in panel.index:
                per_date_cols[alpha_name] = panel.loc[d]

        if not per_date_cols:
            continue

        wide_df = pd.DataFrame(per_date_cols)
        wide_df.index.name = "ticker"
        out_path = output_dir / f"{date_str}.parquet"
        wide_df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        output_files.append(str(out_path))
        dates_written += 1
        if verbose:
            log.info(f"  Wrote {date_str}.parquet ({wide_df.shape[0]} tickers × "
                     f"{wide_df.shape[1]} alphas)")

    return {
        "dates_written":   dates_written,
        "alphas_written":  len(alphas),
        "output_files":    output_files,
    }


def build_alpha_panel(
    tickers: Optional[list[str]] = None,
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_dates: Optional[list[str]] = None,
    ts_windows: tuple[int, ...] = (5, 10, 20),
    bucket_map: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end: build base panels, explode via 17 operators, write parquet.

    Returns: summary dict from write_alpha_panel.
    """
    if tickers is None:
        tickers = load_tickers()
    if bucket_map is None:
        try:
            bucket_map = load_bucket_map()
        except Exception as e:
            log.warning(f"Could not load bucket_map: {e} — skipping group ops")
            bucket_map = None

    log.info(f"Building base panels for {len(tickers)} tickers "
             f"({start_date} → {end_date or 'today'})...")
    panels = build_panels_from_tickers(tickers, start_date, end_date, verbose)
    if not panels:
        log.error("No base panels built — aborting")
        return {"dates_written": 0, "alphas_written": 0, "output_files": [],
                "error": "no_base_panels"}

    log.info(f"Base panels: {len(panels)} features × "
             f"{next(iter(panels.values())).shape}")
    log.info(f"Exploding via {len(ALPHA_OPS) - len(SKIP_FROM_EXPLODE)} operators...")

    alphas = explode_panels(panels, bucket_map=bucket_map,
                            ts_windows=ts_windows, verbose=verbose)

    log.info(f"Generated {len(alphas)} alphas. Writing parquet...")
    return write_alpha_panel(alphas, output_dir, target_dates, verbose)


if __name__ == "__main__":
    # CLI smoke test: 3 tickers, 1 month
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = build_alpha_panel(
        tickers=["AAPL", "MSFT", "NVDA"],
        start_date="2026-04-01",
        target_dates=None,
        verbose=True,
    )
    print(f"\nSummary: {summary['dates_written']} dates, "
          f"{summary['alphas_written']} alphas, "
          f"{len(summary['output_files'])} files")
