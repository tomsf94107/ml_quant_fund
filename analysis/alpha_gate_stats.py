"""
analysis/alpha_gate_stats.py
Multiple-testing / overfitting statistics for the P3.2 alpha gate.

Two complementary gates:
  (a) Harvey-Liu-Zhu hurdle: require |t-stat| > 3.0 for a single feature.
  (b) Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): probability
      that an observed Sharpe (here: IC information ratio) exceeds the
      EXPECTED MAXIMUM across N independent trials. With N=3390 the
      expected-max bar rises sharply — this is what stops us mistaking
      the best-of-3390 noise feature for signal.

For an alpha feature we treat the per-date IC series like a return series:
  IC_IR (mean_IC / std_IC) plays the role of Sharpe; n_dates plays T.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_trial_sr: float = 1.0) -> float:
    """E[max] of N iid standard-normal-ish Sharpe estimates (Bailey-LdP).

    Uses the analytic approximation:
      E[max] ~ sqrt(var) * [ (1-g)*Z(1 - 1/N) + g*Z(1 - 1/(N*e)) ]
    where Z = inverse normal CDF, g = Euler-Mascheroni, e = exp(1).
    var_trial_sr = cross-sectional variance of trial Sharpe estimates;
    default 1.0 gives the bar in 'standard-error' units.
    """
    if n_trials < 2:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return np.sqrt(var_trial_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(observed_sr: float, benchmark_sr: float,
                         T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR: prob that true Sharpe > benchmark, given observed SR over T obs.

    SR and benchmark are in PER-OBSERVATION units (not annualized).
    skew/kurt are of the underlying return (IC) series.
    """
    if T < 2:
        return np.nan
    denom = np.sqrt(1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr ** 2)
    if denom <= 0:
        return np.nan
    z = (observed_sr - benchmark_sr) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe(observed_sr: float, n_trials: int, T: int,
                    var_trial_sr: float, skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR = PSR with benchmark = expected-max-Sharpe under N trials.

    observed_sr, var_trial_sr in per-observation Sharpe units.
    Returns prob the feature's true Sharpe beats the best-of-N null.
    Gate: DSR > 0.95.
    """
    sr_star = expected_max_sharpe(n_trials, var_trial_sr)
    return probabilistic_sharpe(observed_sr, sr_star, T, skew, kurt)


# ───────────────────────── unit tests ─────────────────────────
if __name__ == "__main__":
    print("UNIT TESTS — DSR/PSR math sanity\n")

    # Test 1: expected-max-Sharpe MONOTONIC increasing in N
    e1, e10, e100, e3390 = (expected_max_sharpe(n) for n in (1, 10, 100, 3390))
    print(f"E[max] sharpe:  N=1:{e1:.3f}  N=10:{e10:.3f}  N=100:{e100:.3f}  N=3390:{e3390:.3f}")
    assert e1 < e10 < e100 < e3390, "FAIL: E[max] must rise with N"
    print("  PASS: E[max] rises with N (more trials -> higher bar)\n")

    # Test 2: PSR rises with observed SR
    p_lo = probabilistic_sharpe(0.05, 0.0, T=500)
    p_hi = probabilistic_sharpe(0.20, 0.0, T=500)
    print(f"PSR(SR=0.05)={p_lo:.4f}  PSR(SR=0.20)={p_hi:.4f}")
    assert p_hi > p_lo, "FAIL: PSR must rise with observed SR"
    print("  PASS: PSR rises with observed Sharpe\n")

    # Test 3: THE KEY ONE — same feature, DSR DROPS sharply as N rises.
    # IC_IR ~ 0.30 over T=500 dates; trial-SR variance ~ 1/T (std err of SR)
    sr = 0.30                      # per-date IC information ratio
    T = 500
    var_trial = 1.0 / T            # variance of a Sharpe estimate ~ 1/T
    d_n1   = deflated_sharpe(sr, 1,    T, var_trial)
    d_n50  = deflated_sharpe(sr, 50,   T, var_trial)
    d_n3390= deflated_sharpe(sr, 3390, T, var_trial)
    print(f"DSR(IC_IR=0.30, T=500):  N=1:{d_n1:.4f}  N=50:{d_n50:.4f}  N=3390:{d_n3390:.4f}")
    assert d_n1 > d_n50 > d_n3390, "FAIL: DSR must DROP as N rises"
    print("  PASS: DSR drops as N rises (the multiple-testing penalty works)\n")

    # Test 4: a STRONG feature should still pass at N=3390; a WEAK one should not
    strong = deflated_sharpe(0.55, 3390, 500, 1.0/500)
    weak   = deflated_sharpe(0.15, 3390, 500, 1.0/500)
    print(f"At N=3390, T=500:  strong(IR=0.55) DSR={strong:.3f}  weak(IR=0.15) DSR={weak:.3f}")
    print(f"  strong passes 0.95? {strong > 0.95}   weak passes 0.95? {weak > 0.95}")

    print("\nALL CORE ASSERTIONS PASSED — math is sane, ready to wire into gate.")
