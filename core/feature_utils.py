# core/feature_utils.py
import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
except Exception:
    ta = None

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotent finisher:
      - sort index chronologically
      - add VWAP if (Close, Volume) present (guard against early zero volume)
      - add BB_width if missing (BBP_20_2.0 or (BBU-BBL)/Close) when Close present
      - replace ±inf → NaN
      - downcast float64 → float32
    """
    df = df.copy()
    df.sort_index(inplace=True)

    if "VWAP" not in df.columns and "Volume" in df.columns and df["Volume"].notna().any() and "Close" in df.columns:
        vol = df["Volume"].fillna(0)
        pv  = (df["Close"] * vol).cumsum()
        vc  = vol.cumsum().replace(0, np.nan)
        df["VWAP"] = (pv / vc).fillna(method="bfill")

    if "BB_width" not in df.columns and "Close" in df.columns and ta is not None:
        try:
            bb = ta.bbands(df["Close"])
            if isinstance(bb, pd.DataFrame) and not bb.empty:
                if "BBP_20_2.0" in bb.columns:
                    df["BB_width"] = bb["BBP_20_2.0"]
                elif {"BBU_20_2.0", "BBL_20_2.0"}.issubset(bb.columns):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        df["BB_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["Close"]
        except Exception:
            pass

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")

    return df
