# core/feature_utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotent finisher for any OHLCV/feature DataFrame.

    - Ensure DatetimeIndex (tz-naive), drop invalid index rows, sort ascending
    - Add VWAP if missing and Volume exists (guards against early zero volumes)
    - Add Bollinger-width fallback if missing (BBP_20_2.0 or (BBU-BBL)/Close)
    - Replace ±inf → NaN
    - Optionally downcast float64 → float32 to reduce memory

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with at least ["Close"] and optionally ["Volume"].

    Returns
    -------
    pd.DataFrame
        Cleaned/standardized DataFrame.
    """
    out = df.copy()

    # 0) Ensure DatetimeIndex (tz-naive), drop bad index rows, sort ascending
    try:
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
        # Drop tz to avoid tz-aware/naive join issues elsewhere
        out.index = out.index.tz_localize(None)
        # Drop rows with invalid index
        if getattr(out.index, "isna", None) is not None:
            out = out[~out.index.isna()]
    except Exception:
        # If index coercion fails, keep original index but still try to sort
        pass

    out.sort_index(inplace=True)

    # 1) VWAP (only if not already present)
    if "VWAP" not in out.columns and "Volume" in out.columns and out["Volume"].notna().any():
        try:
            vol = out["Volume"].fillna(0)
            pv = (out["Close"] * vol).cumsum()
            vc = vol.cumsum().replace(0, np.nan)  # guard against early zeros
            out["VWAP"] = (pv / vc).bfill()
        except Exception:
            # leave VWAP absent if anything goes wrong
            pass

    # 2) Bollinger width fallback (if not already present)
    if "BB_width" not in out.columns and "Close" in out.columns:
        try:
            bb = ta.bbands(out["Close"])
            if isinstance(bb, pd.DataFrame) and not bb.empty:
                if "BBP_20_2.0" in bb.columns:
                    out["BB_width"] = bb["BBP_20_2.0"]
                elif {"BBU_20_2.0", "BBL_20_2.0"}.issubset(bb.columns):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        out["BB_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / out["Close"]
        except Exception:
            pass

    # 3) Inf → NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4) Optional memory slim
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")

    return out
