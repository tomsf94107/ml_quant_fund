"""
Probabilistic Sharpe Ratio (PSR) + small-sample guards.
Bailey & Lopez de Prado (2012), The Sharpe Ratio Efficient Frontier.

Drop-in helpers for the per-ticker backtest metrics. PSR answers "what is the
probability the TRUE Sharpe > benchmark, given sample length, skew, kurtosis" —
so a high raw SR on few/noisy trades collapses to a low PSR. Far harder to game
than the raw sqrt(252)*mu/sd number.

Usage in generator._compute_backtest_metrics (after you have ret_strat):
    from signals.sharpe_psr import sharpe_with_psr
    sr, psr, min_trl, reliable = sharpe_with_psr(ret_strat, n_trades)
"""
import numpy as np
from scipy import stats

ANN = 252
MIN_TRADES_RELIABLE = 30   # below this, treat per-ticker Sharpe as unreliable


def _moments(returns):
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return None
    mu = r.mean()
    sd = r.std(ddof=1)
    sr_period = mu / sd                       # per-period (daily) Sharpe
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)    # non-excess (normal=3)
    return n, sr_period, skew, kurt


def probabilistic_sharpe_ratio(returns, sr_benchmark_annual=0.0):
    """P(true SR > benchmark). In [0,1]. Adjusts for n, skew, kurtosis."""
    m = _moments(returns)
    if m is None:
        return np.nan
    n, sr_p, skew, kurt = m
    sr_bench_p = sr_benchmark_annual / np.sqrt(ANN)   # benchmark to per-period
    # SR estimator std (Bailey-LdP / Mertens-Christie-Opdyke)
    denom = 1.0 - skew * sr_p + ((kurt - 1.0) / 4.0) * sr_p ** 2
    if denom <= 0 or n < 2:
        return np.nan
    sr_std = np.sqrt(denom / (n - 1))
    if sr_std == 0:
        return np.nan
    psr = stats.norm.cdf((sr_p - sr_bench_p) / sr_std)
    return float(psr)


def min_track_record_length(returns, sr_benchmark_annual=0.0, prob=0.95):
    """How many observations needed for PSR(benchmark) > prob, at current stats."""
    m = _moments(returns)
    if m is None:
        return np.nan
    n, sr_p, skew, kurt = m
    sr_bench_p = sr_benchmark_annual / np.sqrt(ANN)
    if sr_p <= sr_bench_p:
        return np.inf   # never reaches benchmark at this SR
    z = stats.norm.ppf(prob)
    denom = (sr_p - sr_bench_p) ** 2
    num = (1.0 - skew * sr_p + ((kurt - 1.0) / 4.0) * sr_p ** 2) * z ** 2
    return float(1.0 + num / denom)


def sharpe_with_psr(returns, n_trades=None, sr_benchmark_annual=0.0):
    """Returns (annualized_SR, PSR, minTRL, reliable_bool).
    reliable = enough trades AND PSR materially > 0.5."""
    m = _moments(returns)
    if m is None:
        return np.nan, np.nan, np.nan, False
    n, sr_p, skew, kurt = m
    sr_ann = np.sqrt(ANN) * sr_p
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark_annual)
    trl = min_track_record_length(returns, sr_benchmark_annual)
    nt = n_trades if n_trades is not None else n
    reliable = (nt >= MIN_TRADES_RELIABLE) and (not np.isnan(psr)) and (psr >= 0.95)
    return float(sr_ann), psr, trl, bool(reliable)
