#!/usr/bin/env python3
"""
Patch features/massive_client.py to add hybrid index routing.
Indices (^VIX, ^GSPC, etc.) auto-route to yfinance.
Stocks/ETFs go to Massive.

Run from project root:
  python3 patch_massive_client_hybrid.py

Idempotent.
"""
from pathlib import Path

PATH = Path("features/massive_client.py")
text = PATH.read_text()
original_len = len(text)

# Insert INDEX_SYMBOLS constant after the BASE_URL constants block
old_constants = '''BASE_URL = "https://api.polygon.io"
API_KEY  = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")'''

new_constants = '''BASE_URL = "https://api.polygon.io"
API_KEY  = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")

# Index symbols not available on Massive Stocks Developer plan.
# These auto-route to yfinance.
INDEX_SYMBOLS = {
    "^VIX", "^VIX3M", "^VIX9D",        # CBOE volatility indices
    "^TNX", "^TYX", "^FVX", "^IRX",    # Treasury yield indices
    "^GSPC", "^DJI", "^IXIC", "^RUT",  # Major US equity indices
    "^FTSE", "^N225", "^HSI", "^GDAXI", # International indices
    "ES=F", "NQ=F", "YM=F", "RTY=F",   # Equity index futures
    "CL=F", "GC=F", "SI=F", "HG=F",    # Commodity futures
}


def _is_index(symbol):
    """Return True if symbol should route to yfinance instead of Massive."""
    return symbol in INDEX_SYMBOLS or symbol.startswith("^") or symbol.endswith("=F")'''

if old_constants in text and "INDEX_SYMBOLS" not in text:
    text = text.replace(old_constants, new_constants, 1)
    print("[OK] Added INDEX_SYMBOLS constant")
else:
    print("[SKIP] Constants already patched or marker not found")

# Replace the download() function to add hybrid routing
old_download = '''def download(
    tickers,
    start=None,
    end=None,
    period=None,
    interval="1d",
    auto_adjust=True,
    progress=False,
    **kwargs,
):
    """
    Download OHLCV data. Drop-in replacement for yfinance.download().

    Returns a DataFrame with:
      - single ticker: columns ['Open', 'High', 'Low', 'Close', 'Volume']
      - multi ticker: MultiIndex columns [(field, ticker), ...]
    """
    _check_key()

    if period and not start:
        end = end or date.today()
        if isinstance(end, str):
            end = pd.Timestamp(end).date()
        start = _resolve_period(end, period)

    if not start or not end:
        raise ValueError("download requires either (start, end) or period")

    start_str = _normalize_date(start)
    end_str = _normalize_date(end)

    mult, span = _interval_to_massive(interval)

    if isinstance(tickers, str):
        return _download_single(tickers, start_str, end_str, mult, span, auto_adjust)
    else:
        return _download_batch(tickers, start_str, end_str, mult, span, auto_adjust)'''

new_download = '''def download(
    tickers,
    start=None,
    end=None,
    period=None,
    interval="1d",
    auto_adjust=True,
    progress=False,
    **kwargs,
):
    """
    Download OHLCV data. Drop-in replacement for yfinance.download().

    HYBRID ROUTING:
    - Index symbols (^VIX, ^GSPC, futures like ES=F) auto-route to yfinance
    - Stocks/ETFs route to Massive
    - Mixed list: each ticker fetched from correct source, merged into
      yfinance-shape MultiIndex DataFrame

    Returns a DataFrame with:
      - single ticker: columns ['Open', 'High', 'Low', 'Close', 'Volume']
      - multi ticker: MultiIndex columns [(field, ticker), ...]
    """
    if period and not start:
        end = end or date.today()
        if isinstance(end, str):
            end = pd.Timestamp(end).date()
        start = _resolve_period(end, period)

    if not start or not end:
        raise ValueError("download requires either (start, end) or period")

    start_str = _normalize_date(start)
    end_str = _normalize_date(end)

    if isinstance(tickers, str):
        # Single ticker — route based on type
        if _is_index(tickers):
            return _download_yfinance(tickers, start_str, end_str, interval, auto_adjust)
        else:
            _check_key()
            mult, span = _interval_to_massive(interval)
            return _download_single(tickers, start_str, end_str, mult, span, auto_adjust)
    else:
        # List of tickers — split, fetch each from correct source, merge
        indices = [t for t in tickers if _is_index(t)]
        stocks = [t for t in tickers if not _is_index(t)]

        frames = {}

        # Fetch indices from yfinance
        for t in indices:
            try:
                df = _download_yfinance(t, start_str, end_str, interval, auto_adjust)
                if not df.empty:
                    frames[t] = df
            except Exception as e:
                log.warning(f"yfinance failed for index {t}: {e}")

        # Fetch stocks from Massive
        if stocks:
            _check_key()
            mult, span = _interval_to_massive(interval)
            for t in stocks:
                try:
                    df = _download_single(t, start_str, end_str, mult, span, auto_adjust)
                    if not df.empty:
                        frames[t] = df
                except Exception as e:
                    log.warning(f"massive failed for {t}: {e}")

        if not frames:
            return pd.DataFrame()

        # Merge into yfinance-shape MultiIndex
        out = pd.concat(frames, axis=1)
        out.columns = out.columns.swaplevel(0, 1)
        out = out.sort_index(axis=1)
        return out


def _download_yfinance(ticker, start_str, end_str, interval, auto_adjust):
    """Fallback fetch via yfinance for index symbols not in Massive."""
    import yfinance as yf
    df = yf.download(ticker, start=start_str, end=end_str,
                     interval=interval, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    # Flatten MultiIndex if yfinance returned one
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    if interval == "1d":
        df.index = df.index.normalize()
    return df'''

if old_download in text:
    text = text.replace(old_download, new_download, 1)
    print("[OK] download() function patched with hybrid routing")
else:
    print("[FAIL] download() function marker not found")

if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")
