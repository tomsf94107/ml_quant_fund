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
        # SAMPLE FIX (2026-08-22): was load_tickers()[:n_sample] -- the first N
        # ALPHABETICALLY. For a sparse feature that is a biased sample:
        # inst_signed_flow_30d covers ~47 of 411 names per date, so most of the
        # first 40 are NaN, distinct-count collapses, and it was classified
        # market_wide despite 46.6 distinct values/date on the full panel.
        # It carried IC +0.034, t 3.24, Sharpe +0.55, mono 0.697 and was
        # excluded from every survivor list.
        import random as _rnd
        _all = load_tickers()
        tickers = (_rnd.Random(17).sample(_all, min(n_sample, len(_all)))
                   if len(_all) > n_sample else _all)
    panels = build_panels_from_tickers(tickers, start_date, None, verbose=False)
    out = {}
    for base, p in panels.items():
        if p.shape[1] < 5:
            out[base] = "per_ticker"; continue
        # COVERAGE GUARD: too few names carrying a value makes distinct-count
        # meaningless. Default to per_ticker (keeps it in the pool) rather than
        # silently excluding a sparse stock-level signal.
        _cov = p.notna().sum(axis=1).median()
        if _cov < 5:
            out[base] = "per_ticker"; continue
        med = p.nunique(axis=1).median()
        # THRESHOLD FIX (2026-08-22): was `med <= 2`, which cannot distinguish a
        # genuine market-wide feature from a BINARY PER-TICKER FLAG.
        # Measured on the live panel, distinct-values-per-date:
        #   vix_close / oil_ret / xlv_ret_5d          = 1   market-wide
        #   ma5_above_ma20 / is_squeeze_setup /
        #   post_earnings_1d                          = 2   PER-TICKER 0/1 flags
        # At <=2 every binary stock-level flag was excluded from the
        # stock-picking pool. A market-wide feature is CONSTANT: exactly 1.
        out[base] = "market_wide" if med <= 1 else "per_ticker"
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
