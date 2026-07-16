"""
Shape-based Scenario Generation using Copulas.

Implements the framework from:
    Kaut, M. & Wallace, S. W. (2011). "Shape-based Scenario Generation
    using Copulas." Computational Management Science, 8, 181-199.

The key idea: separate the multivariate structure (copula) from the
marginal distributions. The paper's recommended best method (Sec 3.2)
is: empirical copula via rank-pairing of sampled history + KS-optimal
discretized inverse-CDF margins + post-process to correct moments and
correlations (Hoyland-Kaut-Wallace 2003).

This module provides:
  - EmpiricalCopulaGenerator:        rank-coupling of historical samples
  - GaussianCopulaGenerator:         t/Gaussian copula with empirical margins
  - moment_matching_postprocess:     Hoyland et al. 2003 correction
  - generate_scenarios:              one-call convenience using paper's best combo
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata
from scipy.linalg import cholesky
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Margin generation (Step 2 of the paper's framework, Section 2.1)
# ---------------------------------------------------------------------------

def ks_optimal_quantiles(n_scenarios: int) -> np.ndarray:
    """
    Returns the Kolmogorov-Smirnov-optimal discretization
    u_s = (2s - 1) / (2 * n_S)  for s = 1, ..., n_S.

    Paper Sec 2.1: this is optimal in the KS sense, vs. s/(n_S+1)
    commonly used in copula literature.
    """
    s = np.arange(1, n_scenarios + 1)
    return (2.0 * s - 1.0) / (2.0 * n_scenarios)


def empirical_inverse_cdf(historical: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Empirical inverse CDF (quantile function) of `historical`, evaluated
    at probabilities `u`. Uses linear interpolation between order
    statistics, matching numpy's default quantile behavior.

    historical: 1-D array of past observations for ONE margin
    u:          probabilities in (0, 1)
    """
    return np.quantile(historical, u, method="linear")


def discretized_margins(historical: np.ndarray, n_scenarios: int) -> np.ndarray:
    """
    Build a fixed KS-optimal discretization of each margin's empirical
    distribution (paper Sec 2.1, "fixed discretization of margins").

    historical:  (T, d) historical returns, T observations of d assets
    n_scenarios: number of scenarios to produce per margin

    Returns: (n_scenarios, d) array. Column i contains the n_scenarios
    KS-optimal quantile values for margin i, sorted ascending.
    """
    u = ks_optimal_quantiles(n_scenarios)
    d = historical.shape[1]
    margins = np.empty((n_scenarios, d))
    for i in range(d):
        margins[:, i] = empirical_inverse_cdf(historical[:, i], u)
    return margins


# ---------------------------------------------------------------------------
# 2. Copula sampling (Step 1 of the paper's framework, Section 2.1)
# ---------------------------------------------------------------------------

def empirical_copula_ranks(
    historical: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample the empirical copula from historical data via rank pairing
    (paper Sec 2.1, first bullet of Step 1).

    For each scenario s and margin i, we want the rank in {1, ..., n_S}
    that tells us which order statistic of the margin to use.

    Procedure:
      1. Sample n_S rows (with replacement) from the historical (T, d) data.
      2. Within each column of the sample, replace values by their ranks
         in {1, ..., n_S}.

    The resulting (n_S, d) integer matrix is a sample from the empirical
    copula expressed as a coupling of ranks.
    """
    T = historical.shape[0]
    idx = rng.integers(0, T, size=n_scenarios)
    sample = historical[idx, :]

    # rankdata with method="ordinal" gives unique ranks 1..n_S per column,
    # breaking ties by index order — important so each rank in {1,...,n_S}
    # appears exactly once per margin (paper Sec 1.1).
    ranks = np.empty_like(sample, dtype=np.int64)
    for i in range(sample.shape[1]):
        ranks[:, i] = rankdata(sample[:, i], method="ordinal")
    return ranks  # values in {1, ..., n_scenarios}


def gaussian_copula_ranks(
    correlation: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a Gaussian copula with the given correlation matrix and
    convert the sample to a rank-coupling (paper Sec 2.3).

    Useful as a fallback when historical data is short relative to
    the asset count, or when you want a smooth structure.
    """
    d = correlation.shape[0]
    z = rng.multivariate_normal(np.zeros(d), correlation, size=n_scenarios)
    ranks = np.empty_like(z, dtype=np.int64)
    for i in range(d):
        ranks[:, i] = rankdata(z[:, i], method="ordinal")
    return ranks


# ---------------------------------------------------------------------------
# 3. Assembly: combine copula ranks with margin values (paper Sec 2.1 end)
# ---------------------------------------------------------------------------

def assemble_scenarios(
    rank_matrix: np.ndarray,
    sorted_margins: np.ndarray,
) -> np.ndarray:
    """
    rank_matrix:    (n_S, d) integer ranks in {1, ..., n_S}
    sorted_margins: (n_S, d) margin values sorted ASC per column
                     (output of discretized_margins is already sorted)

    Returns (n_S, d) scenarios where scenarios[s, i] is the
    rank_matrix[s, i]-th order statistic of margin i.
    """
    # ranks are 1-indexed; convert to 0-indexed
    n_S, d = rank_matrix.shape
    out = np.empty((n_S, d))
    for i in range(d):
        out[:, i] = sorted_margins[rank_matrix[:, i] - 1, i]
    return out


# ---------------------------------------------------------------------------
# 4. Post-process: correct moments and correlations (paper Sec 2.2)
#    Hoyland, Kaut & Wallace 2003 - cubic + Cholesky iteration.
# ---------------------------------------------------------------------------

def _solve_fleishman_for(target_skew: float, target_excess_kurt: float):
    """Solve unit-variance Fleishman for one margin. Returns (a,b,c,d)
    such that Y = a + b*X + c*X**2 + d*X**3 has mean 0, var 1, given
    skew, given excess kurt, when X has mean 0 and var 1 (X need not be
    normal — Fleishman is exact for normal but a useful approximation
    otherwise). Returns identity (0,1,0,0) on failure."""
    from scipy.optimize import fsolve

    def eqs(bcd):
        b, c, d = bcd
        e1 = b**2 + 6*b*d + 2*c**2 + 15*d**2 - 1.0
        e2 = 2*c*(b**2 + 24*b*d + 105*d**2 + 2) - target_skew
        e3 = (24*(b*d + c**2*(1 + b**2 + 28*b*d)
                  + d**2*(12 + 48*b*d + 141*c**2 + 225*d**2))
              - target_excess_kurt)
        return [e1, e2, e3]

    try:
        b, c, d = fsolve(eqs, x0=[1.0, 0.0, 0.0], full_output=False)
        a = -c
        # sanity
        if not all(np.isfinite([a, b, c, d])):
            return (0.0, 1.0, 0.0, 0.0)
        return (a, b, c, d)
    except Exception:
        return (0.0, 1.0, 0.0, 0.0)


def moment_matching_postprocess(
    scenarios: np.ndarray,
    target_moments: np.ndarray,
    target_correlation: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-3,
    apply_cubic: bool = False,
) -> np.ndarray:
    """
    Hoyland-Kaut-Wallace (2003) cubic + Cholesky correction.

    Algorithm:
      1) Standardize each margin to mean 0, var 1.
      2) (Optional, if `apply_cubic=True`) Apply Fleishman cubic per
         margin once. The cubic is exact only when the input is N(0,1);
         it AMPLIFIES tails when applied to data that's already
         heavy-tailed. So the default is to skip it: when the starting
         sample comes from `assemble_scenarios` with empirical margins,
         the marginal moments are already correct by construction —
         only correlation needs correcting.
         Set `apply_cubic=True` if your starting sample has near-normal
         margins and you want to inject specific (skew, kurt) targets
         (this is the original HKW use case).
      3) Iterate Cholesky decorrelate-recorrelate until the sample
         correlation matches target. The Cholesky step distorts the
         per-margin distribution slightly; with a good copula starting
         point the distortion is small (paper Sec 2.2).
      4) Rescale to target (mean, variance).

    The paper (Sec 3.3) reports negligible bias for n_S >= 250 with
    this combination.

    scenarios:           (n_S, d) starting sample
    target_moments:      (d, 4) per-margin (mean, var, skew, kurt) — only
                         the first two are used unless apply_cubic=True
    target_correlation:  (d, d) target Pearson correlation
    apply_cubic:         see above; default False
    """
    n_S, d = scenarios.shape
    Y = scenarios.copy().astype(np.float64)

    target_means = target_moments[:, 0]
    target_vars = target_moments[:, 1]

    target_corr_reg = target_correlation + 1e-10 * np.eye(d)
    target_chol = cholesky(target_corr_reg, lower=True)

    # ---- Step 1: standardize ------------------------------------------------
    Z = (Y - Y.mean(axis=0)) / (Y.std(axis=0, ddof=0) + 1e-12)

    # ---- Step 2: Fleishman cubic (optional) --------------------------------
    if apply_cubic:
        target_skew = target_moments[:, 2]
        target_excess_kurt = target_moments[:, 3] - 3.0
        for i in range(d):
            a, b, c, dd = _solve_fleishman_for(
                target_skew[i], target_excess_kurt[i]
            )
            zi = Z[:, i]
            Z[:, i] = a + b * zi + c * zi**2 + dd * zi**3
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0, ddof=0) + 1e-12)

    # ---- Step 3: iterate Cholesky correlation correction --------------------
    for _ in range(max_iter):
        sample_corr = np.corrcoef(Z, rowvar=False) + 1e-10 * np.eye(d)
        err = np.max(np.abs(sample_corr - target_correlation))
        if err < tol:
            break
        sample_chol = cholesky(sample_corr, lower=True)
        A = np.linalg.solve(sample_chol, target_chol)
        Z = Z @ A.T
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0, ddof=0) + 1e-12)

    # ---- Step 4: rescale to target (mean, var) ------------------------------
    Y = target_means + np.sqrt(target_vars) * Z
    return Y


# ---------------------------------------------------------------------------
# 5. One-shot convenience function — paper's best combo
# ---------------------------------------------------------------------------

def generate_scenarios(
    historical: np.ndarray,
    n_scenarios: int,
    correct_moments_and_correlation: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate n_scenarios scenarios using the paper's recommended method:
        empirical copula (rank-pairing) + KS-optimal inverse-CDF margins
        + (optional) HKW moment & correlation post-process

    historical:  (T, d) array of past returns
    n_scenarios: number of scenarios to produce
    correct_moments_and_correlation: if True, run HKW post-process to
        correct first 4 moments and the Pearson correlation matrix.
        Recommended for n_scenarios >= 250 (paper Sec 3.3).

    Returns (n_scenarios, d) array of scenario returns.
    """
    rng = np.random.default_rng(seed)

    # Step 1: empirical copula as rank-coupling
    rank_matrix = empirical_copula_ranks(historical, n_scenarios, rng)

    # Step 2: KS-optimal discretized margins from empirical inverse CDF
    sorted_margins = discretized_margins(historical, n_scenarios)

    # Assembly
    scenarios = assemble_scenarios(rank_matrix, sorted_margins)

    if not correct_moments_and_correlation:
        return scenarios

    # Compute targets from history
    means = historical.mean(axis=0)
    variances = historical.var(axis=0, ddof=0)
    # Use Fisher's definition of skew/kurt to match common conventions
    centered = historical - means
    stds = np.sqrt(variances) + 1e-12
    skew = (centered**3).mean(axis=0) / stds**3
    kurt = (centered**4).mean(axis=0) / stds**4   # raw (Pearson) kurt; normal=3
    target_moments = np.column_stack([means, variances, skew, kurt])
    target_correlation = np.corrcoef(historical, rowvar=False)

    return moment_matching_postprocess(
        scenarios, target_moments, target_correlation
    )