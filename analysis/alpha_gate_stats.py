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


from itertools import combinations


def cscv_pbo(perf, S=10):
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein,
    Lopez de Prado, Zhu 2017).

    perf: 2D array [T observations x N trials] of per-period performance
          (here: per-date IC for each feature). Higher = better.
    S:    number of equal time-blocks (must be even). Default 10 -> C(10,5)=252
          balanced IS/OOS splits.

    Method: split rows into S blocks. For every way to choose S/2 blocks as
    IS (rest OOS), pick the trial with best mean IS performance, record its
    RANK among all trials OOS, map to logit. PBO = fraction of splits where
    the IS-best ranks BELOW median OOS (logit < 0).

    PBO > 0.5  -> selection is overfit (IS-best is OOS-mediocre).
    Gate: PBO < 0.3.
    """
    perf = np.asarray(perf, float)
    T, N = perf.shape
    if N < 2 or T < S or S % 2 != 0:
        return np.nan
    # trim to a multiple of S, split into S contiguous blocks
    rows_per = T // S
    perf = perf[:rows_per * S]
    blocks = [perf[i*rows_per:(i+1)*rows_per] for i in range(S)]
    half = S // 2
    n_below = 0; n_tot = 0
    for is_idx in combinations(range(S), half):
        oos_idx = [b for b in range(S) if b not in is_idx]
        is_mat  = np.vstack([blocks[b] for b in is_idx])   # rows x N
        oos_mat = np.vstack([blocks[b] for b in oos_idx])
        is_perf  = np.nanmean(is_mat,  axis=0)             # N
        oos_perf = np.nanmean(oos_mat, axis=0)             # N
        if np.all(np.isnan(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))                 # IS-chosen trial
        # rank of chosen trial OOS (1=worst .. N=best); omega = rank fraction
        order = np.argsort(np.argsort(np.nan_to_num(oos_perf, nan=-np.inf)))
        rank = order[best] + 1
        omega = rank / (N + 1)                             # in (0,1)
        logit = np.log(omega / (1.0 - omega))
        n_below += (logit < 0); n_tot += 1
    return float(n_below / n_tot) if n_tot else np.nan


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

    # Test 5: PBO — random noise should be ~0.5 (overfit); a planted strong
    # trial should pull PBO down (real signal generalizes OOS).
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((500, 100))               # 100 noise trials
    pbo_noise = cscv_pbo(noise, S=10)
    signal = noise.copy(); signal[:, 0] += 0.5            # trial 0 truly strong
    pbo_signal = cscv_pbo(signal, S=10)
    print(f"\nPBO noise(100 trials)={pbo_noise:.3f}  PBO with planted signal={pbo_signal:.3f}")
    assert 0.3 < pbo_noise < 0.7, "FAIL: pure noise PBO should be near 0.5"
    assert pbo_signal < pbo_noise, "FAIL: planted signal should lower PBO"
    print("  PASS: PBO ~0.5 for noise, drops when real signal present\n")

    print("\nALL CORE ASSERTIONS PASSED — math is sane, ready to wire into gate.")
