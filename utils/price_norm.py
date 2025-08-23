# utils/price_norm.py
import pandas as pd

def normalize_price_df(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a price df to columns:
      ['ticker','date','close','shares_outstanding','market_cap']
    Works even if `px` has MultiIndex columns like ('Close','AAPL') or mixed/blank
    second levels (e.g., ('date','')).
    """
    if px is None or len(px) == 0:
        return pd.DataFrame(columns=["ticker","date","close","shares_outstanding","market_cap"])

    df = px.copy()
    T = str(ticker).upper()

    # 1) Flatten / slice MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        sliced = False
        for lev in range(df.columns.nlevels):
            try:
                if any(str(v).upper() == T for v in df.columns.get_level_values(lev)):
                    try:
                        tmp = df.xs(T, axis=1, level=lev, drop_level=True)
                        if tmp.shape[1] > 0:
                            df = tmp
                            sliced = True
                            break
                    except Exception:
                        pass
            except Exception:
                pass
        if isinstance(df.columns, pd.MultiIndex):
            flat = []
            for tpl in df.columns.to_flat_index():
                parts = [str(p) for p in tpl if str(p) not in ("", "None", "nan")]
                if len(parts) >= 2:
                    name = f"{parts[0]}_{parts[1]}" if parts[1].upper() != T else parts[0]
                else:
                    name = parts[0]
                flat.append(name)
            df.columns = flat

    # 2) Ensure a date column
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        else:
            df = df.reset_index().rename(columns={"index": "date"})

    # 3) Find/rename close
    def _find_first(names):
        lower_map = {c.lower(): c for c in df.columns}
        for n in names:
            if n in lower_map:
                return lower_map[n]
        for n in names:
            pretty = n.replace("_"," ").title()
            if pretty in df.columns:
                return pretty
        return None

    df.columns = [str(c) for c in df.columns]
    close_candidates = [
        "adj close", "adj_close", "adjclose", "close",
        f"adj close_{T.lower()}", f"adj_close_{T.lower()}", f"adjclose_{T.lower()}",
        f"close_{T.lower()}", f"adj close_{T}", f"adj_close_{T}", f"adjclose_{T}", f"close_{T}",
    ]
    ccol = _find_first(close_candidates)
    if ccol:
        df = df.rename(columns={ccol: "close"})
    elif "close" not in df.columns:
        numcols = df.select_dtypes(include="number").columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]: "close"})
        else:
            df["close"] = pd.NA

    # 4) Ticker column
    df["ticker"] = T

    # 5) shares_outstanding / market_cap
    for base in ("shares_outstanding", "market_cap"):
        if base not in df.columns:
            cand = _find_first([base, f"{base}_{T.lower()}", f"{base}_{T}"])
            if cand:
                df = df.rename(columns={cand: base})
            else:
                df[base] = pd.NA

    # 6) Final tidy
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    keep = [c for c in ["ticker","date","close","shares_outstanding","market_cap"] if c in df.columns]
    out = df[keep].copy().dropna(subset=["date"])
    return out
