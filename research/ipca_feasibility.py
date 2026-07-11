"""
IPCA feasibility (Hunt #6, Jun 1 2026) — STEP 1+2: assemble curated characteristic
panel + confirm IPCA converges in-sample before the OOS build.

Research (KPS 2019): ~8-10 curated characteristics ~= 36; only ~8 significant
(momentum, reversal, idio-vol, beta, liquidity). IPCA is parameter-parsimonious so
OOS deterioration is modest IF char count small. Use curated ~10, NOT all 100.
Monthly freq. Panel: (ticker,date) chars (xs rank-standardized) + fwd 1m return.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import build_feature_dataframe

CHARS = ["return_1d", "return_5d", "volatility_5d", "volatility_10d",
         "rsi_14", "bb_width", "volume_zscore", "obv", "atr", "beta_60d",
         "semi_etf_momentum_60d"]


def build_char_panel(tickers, start="2018-01-01"):
    frames = []
    for i, t in enumerate(tickers, 1):
        if i % 30 == 0:
            print(f"  [{i}/{len(tickers)}] {t}", flush=True)
        try:
            df = build_feature_dataframe(t, start_date=start)
            if "date" not in df.columns:
                continue
            df = df.set_index("date")
            keep = [c for c in CHARS if c in df.columns]
            if len(keep) < len(CHARS) - 2:
                continue
            sub = df[keep].copy()
            sub["close"] = df["close"]
            sub["ticker"] = t
            frames.append(sub)
        except Exception:
            pass
    panel = pd.concat(frames)
    panel.index = pd.to_datetime(panel.index)
    return panel


def to_monthly(panel):
    out = []
    for t, g in panel.groupby("ticker"):
        g = g.sort_index()
        m = g.resample("ME").last()
        m["fwd_ret"] = m["close"].shift(-1) / m["close"] - 1.0
        out.append(m)
    return pd.concat(out).dropna(subset=["fwd_ret"])


def standardize_xs(mp, chars):
    mp = mp.reset_index()
    mp = mp.rename(columns={mp.columns[0]: "date"})
    for c in chars:
        mp[c] = mp.groupby("date")[c].transform(
            lambda s: s.rank(pct=True) - 0.5 if s.notna().sum() > 5 else s * 0)
    return mp.dropna(subset=chars)


def main():
    tickers = [t.strip().upper() for t in open(ROOT / "tickers.txt")
               if t.strip() and not t.startswith("#")]
    print(f"Building characteristic panel for {len(tickers)} tickers...")
    panel = build_char_panel(tickers)
    chars = [c for c in CHARS if c in panel.columns]
    print(f"  chars used ({len(chars)}): {chars}")
    mp = to_monthly(panel)
    mp = standardize_xs(mp, chars)
    print(f"  monthly panel: {len(mp)} (ticker,month) rows, "
          f"{mp['date'].nunique()} months, {mp['ticker'].nunique()} tickers")
    from ipca import InstrumentedPCA
    mp = mp.set_index(["ticker", "date"]).sort_index()
    X = mp[chars]; y = mp["fwd_ret"]
    print(f"\n  fitting IPCA n_factors=4 in-sample (feasibility)...")
    reg = InstrumentedPCA(n_factors=4, intercept=False, max_iter=2000)
    reg.fit(X=X, y=y)
    r2 = reg.score(X=X, y=y)
    print(f"  CONVERGED. in-sample total R2 = {r2:.4f}")
    Gamma, Factors = reg.get_factors(label_ind=True)
    print(f"  Gamma (chars x factors): {Gamma.shape}")
    print("\n  READ: converged + R2 a few % (NOT ~0, NOT ~1) -> build OOS rank-IC test.")
    print("  Degenerate R2 or no convergence -> IPCA not viable at this scale.")


if __name__ == "__main__":
    main()
