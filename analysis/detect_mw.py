# Data-driven market-wide detector — classifies at the BASE-FEATURE level.
#
# Why base-level: transforms (ts_std, ts_argmax) destroy the signal needed to
# classify from the alpha panel. ts_argmax returns 0..w (a few values) whether
# or not the base is per-ticker, so nunique/ratio on the TRANSFORMED column
# fails (verified: fund_op_equity__ts_argmax__w5 looked market-wide). The
# base feature panel is where oil_ret is genuinely 1-value-per-date and
# fund_op_equity is genuinely per-ticker. Classify there, map transforms by base.
#
# market_wide: base panel has median distinct-across-tickers-per-date <= 2
# per_ticker:  median distinct >> 2 (≈ n_tickers)
# Each alpha 'base__op__w' maps to its base's class via prefix.

from analysis.build_alpha_panel import build_panels_from_tickers, load_tickers


def classify_bases(tickers=None, start_date="2025-06-01", n_sample=40):
    """Build base panels for a ticker sample, classify each base feature by
    median distinct-values-across-tickers-per-date. Returns dict[base -> class]."""
    if tickers is None:
        tickers = load_tickers()[:n_sample]
    panels = build_panels_from_tickers(tickers, start_date, None, verbose=False)
    out = {}
    for base, p in panels.items():
        if p.shape[1] < 5:
            out[base] = "per_ticker"; continue
        med = p.nunique(axis=1).median()
        out[base] = "market_wide" if med <= 2 else "per_ticker"
    return out


def alpha_base(alpha_name: str) -> str:
    """base feature = substring before first '__'."""
    return alpha_name.split("__")[0]


def classify_alpha(alpha_name: str, base_classes: dict) -> str:
    """Map a transformed alpha to its base's class. Unknown base -> per_ticker
    (conservative: keep it in the stock-picking pool rather than wrongly drop)."""
    return base_classes.get(alpha_base(alpha_name), "per_ticker")


def _selftest():
    bc = classify_bases(n_sample=30)
    expect = {"oil_ret": "market_wide", "dxy_ret": "market_wide",
              "fund_op_equity": "per_ticker", "fund_ni_margin": "per_ticker",
              "vix_close": "market_wide", "return_1d": "per_ticker"}
    ok = True
    for base, want in expect.items():
        got = bc.get(base, "MISSING")
        flag = "OK" if got == want else "*** MISMATCH ***"
        if got != want:
            ok = False
        print(f"  {base:20s} got={got:12s} want={want:12s} {flag}")
    from collections import Counter
    print("\nbase class counts:", dict(Counter(bc.values())))
    print("total bases:", len(bc))
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
