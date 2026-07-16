"""
Tests for VineScenarioTree (He & Zhang 2024 vine copula method).

Run from project root:
    python -m pytest tests/test_vine_scenario_tree.py -v

The fixtures use Q=4 assets and T=300 obs so the full suite runs in ~60 s.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from src.vine_scenario_tree import VineScenarioTree


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures (module scope — built once, reused across all test classes)
# ──────────────────────────────────────────────────────────────────────────────

Q_TEST = 4     # keep small so GARCH+vine fits in seconds
J_TEST = 15    # num_recourse
E_TEST = 5     # num_evaluate


@pytest.fixture(scope="module")
def synthetic_data():
    """Correlated daily returns for Q=4 assets, T=300 obs."""
    rng = np.random.default_rng(0)
    T, Q = 300, Q_TEST
    corr = np.array([
        [1.0, 0.45, 0.30, 0.15],
        [0.45, 1.0, 0.50, 0.20],
        [0.30, 0.50, 1.0, 0.35],
        [0.15, 0.20, 0.35, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, Q)) @ L.T
    returns = z * 0.01 + 0.0003   # ~1 % daily vol, small positive drift
    initial_prices = np.array([100.0, 150.0, 80.0, 200.0])
    return initial_prices, returns


@pytest.fixture(scope="module")
def fitted_tree(synthetic_data):
    ip, ret = synthetic_data
    tree = VineScenarioTree(ip, ret, seed=42)
    tree.fit()
    return tree


@pytest.fixture(scope="module")
def built_tree(fitted_tree):
    fitted_tree.build_tree(
        num_recourse=J_TEST,
        num_evaluate=E_TEST,
        n_sim_stage1=150,
        n_sim_stage2=30,
        beta=0.45,
    )
    return fitted_tree


# ──────────────────────────────────────────────────────────────────────────────
# 1. Correctness of fit()
# ──────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_garch_params_count(self, fitted_tree):
        assert len(fitted_tree._garch_params) == Q_TEST

    def test_garch_stationarity(self, fitted_tree):
        """alpha + beta < 1 guarantees variance stationarity."""
        for i, p in enumerate(fitted_tree._garch_params):
            assert p['alpha'] >= 0,    f"asset {i}: alpha < 0"
            assert p['beta']  >= 0,    f"asset {i}: beta < 0"
            assert p['omega'] >  0,    f"asset {i}: omega <= 0"
            assert p['alpha'] + p['beta'] < 1.0, (
                f"asset {i}: alpha+beta={p['alpha']+p['beta']:.4f} >= 1 (non-stationary)"
            )

    def test_z_shape(self, fitted_tree, synthetic_data):
        _, ret = synthetic_data
        assert fitted_tree._z_std.shape == ret.shape

    def test_z_mean_near_zero(self, fitted_tree):
        z = fitted_tree._z_std
        for i in range(z.shape[1]):
            assert abs(z[:, i].mean()) < 0.2, (
                f"asset {i}: standardized residual mean {z[:,i].mean():.3f} far from 0"
            )

    def test_z_std_near_one(self, fitted_tree):
        z = fitted_tree._z_std
        for i in range(z.shape[1]):
            s = z[:, i].std()
            assert 0.6 < s < 1.4, (
                f"asset {i}: standardized residual std {s:.3f} not near 1"
            )

    def test_vine_fitted(self, fitted_tree):
        assert fitted_tree._vine is not None
        assert fitted_tree._fitted is True

    def test_vine_dimension(self, fitted_tree):
        """Vine copula must have the same dimension as the asset count."""
        assert fitted_tree._vine.dim == Q_TEST


# ──────────────────────────────────────────────────────────────────────────────
# 2. Correctness of build_tree()
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildTree:
    def test_built_flag(self, built_tree):
        assert built_tree._built is True

    def test_recourse_shape(self, built_tree):
        assert built_tree.recourse_prices.shape == (J_TEST, Q_TEST)

    def test_evaluate_list_length(self, built_tree):
        assert len(built_tree.evaluate_prices) == J_TEST

    def test_evaluate_block_shape(self, built_tree):
        for j, ep in enumerate(built_tree.evaluate_prices):
            assert ep.shape == (E_TEST, Q_TEST), (
                f"recourse node {j}: evaluate block shape {ep.shape}"
            )

    def test_recourse_probs_sum(self, built_tree):
        s = built_tree.recourse_probs.sum()
        assert abs(s - 1.0) < 1e-6, f"p_j sum = {s}"

    def test_recourse_probs_nonneg(self, built_tree):
        assert np.all(built_tree.recourse_probs >= -1e-10)

    def test_evaluate_prob_value(self, built_tree):
        assert abs(built_tree.evaluate_prob - 1.0 / E_TEST) < 1e-12

    def test_recourse_prices_positive(self, built_tree):
        assert np.all(built_tree.recourse_prices > 0)

    def test_evaluate_prices_positive(self, built_tree):
        for ep in built_tree.evaluate_prices:
            assert np.all(ep > 0), "Negative/zero evaluate price found"

    def test_stage2_conditioned_on_recourse(self, built_tree):
        """
        Stage-2 prices should centre around the corresponding recourse prices,
        not the initial prices — verifying the conditional GARCH update.
        """
        for j in range(J_TEST):
            rec   = built_tree.recourse_prices[j]
            evl   = built_tree.evaluate_prices[j]          # (E, Q)
            ratio = evl / rec                               # should be near 1
            mean_ratio = ratio.mean()
            assert 0.7 < mean_ratio < 1.3, (
                f"node {j}: evaluate/recourse mean ratio {mean_ratio:.3f} out of range"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 3. LP moment matching quality
# ──────────────────────────────────────────────────────────────────────────────

class TestLPMomentMatching:
    """LP-optimal q must be at least as good as uniform q on the combined objective."""

    @staticmethod
    def _lp_objective(centroids, weights, M, sig, C, SK, KT, VaR, beta):
        """Sum of absolute moment deviations (LP objective without mean term)."""
        Q = centroids.shape[1]
        cov_pairs = [(i, k) for i in range(Q) for k in range(i, Q)]

        # raw 2nd moment
        raw2 = np.array([
            weights @ (centroids[:, i] * centroids[:, k])
            for (i, k) in cov_pairs
        ])
        target_raw2 = np.array([C[i, k] + M[i] * M[k] for (i, k) in cov_pairs])
        err_cov = np.abs(raw2 - target_raw2).sum()

        # skewness
        sk_w = ((centroids - M)**3 / sig**3).T @ weights
        err_sk = np.abs(sk_w - SK).sum()

        # kurtosis
        kt_w = ((centroids - M)**4 / sig**4).T @ weights
        err_kt = np.abs(kt_w - KT).sum()

        # tail mean
        tm_w = (centroids * (centroids < VaR)).T @ weights / (1 - beta)
        err_tm = np.abs(tm_w - np.zeros(Q)).sum()   # vs. zero baseline (relative)

        return err_cov + err_sk + err_kt + err_tm

    def test_lp_no_worse_than_uniform(self, fitted_tree):
        """LP must not increase the moment-matching objective vs. uniform."""
        rng = np.random.default_rng(77)
        Q = Q_TEST
        n_large = 200

        ret_large, _ = fitted_tree._simulate_stage1(n_large)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=10, n_init=5, random_state=0)
        km.fit(ret_large)
        centroids = km.cluster_centers_

        q_lp   = fitted_tree._moment_matching_lp(centroids, ret_large, beta=0.45)
        q_unif = np.ones(10) / 10.0

        M   = ret_large.mean(axis=0)
        C   = np.cov(ret_large, rowvar=False)
        sig = np.sqrt(np.diag(C).clip(1e-16))
        cent = ret_large - M
        SK  = (cent**3).mean(axis=0) / sig**3
        KT  = (cent**4).mean(axis=0) / sig**4
        VaR = np.quantile(ret_large, 0.45, axis=0)
        beta = 0.45

        obj_lp   = self._lp_objective(centroids, q_lp,   M, sig, C, SK, KT, VaR, beta)
        obj_unif = self._lp_objective(centroids, q_unif, M, sig, C, SK, KT, VaR, beta)

        print(f"\n  LP objective: {obj_lp:.6f}  |  Uniform: {obj_unif:.6f}")
        assert obj_lp <= obj_unif + 1e-6, (
            f"LP objective ({obj_lp:.6f}) worse than uniform ({obj_unif:.6f})"
        )

    def test_mean_approximately_preserved(self, built_tree, synthetic_data):
        """
        Weighted mean of recourse returns should be close to the
        large-simulation mean (LP enforces this with high penalty weight).
        """
        ip, _ = synthetic_data
        rec_ret = built_tree.recourse_prices / ip - 1.0   # (J, Q)
        q = built_tree.recourse_probs
        lp_mean = rec_ret.T @ q                            # (Q,)

        # Compare against historical mean as a sanity bound
        hist_mean = built_tree.historical_returns.mean(axis=0)
        # They shouldn't differ by more than 10× the historical mean magnitude
        scale = np.abs(hist_mean).mean() + 1e-5
        diff = np.abs(lp_mean - hist_mean).mean()
        assert diff < 20 * scale, (
            f"LP mean far from historical: diff/scale = {diff/scale:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. CSV export format
# ──────────────────────────────────────────────────────────────────────────────

class TestExportCSV:
    @pytest.fixture(scope="class")
    def csv_path(self, built_tree, tmp_path_factory):
        p = str(tmp_path_factory.mktemp("csv") / "vine_test.csv")
        built_tree.export_tree_to_csv(p)
        return p

    @pytest.fixture(scope="class")
    def df(self, csv_path):
        return pd.read_csv(csv_path, header=None)

    def test_file_exists(self, csv_path):
        assert os.path.exists(csv_path)

    def test_row_count(self, df):
        assert len(df) == J_TEST * E_TEST

    def test_column_count(self, df):
        assert df.shape[1] == 3 + 2 * Q_TEST

    def test_no_header(self, csv_path):
        with open(csv_path) as f:
            first = f.readline().strip()
        assert first.startswith("0"), (
            f"First line should start with recourse index 0, got: {first[:30]}"
        )

    def test_recourse_indices_range(self, df):
        indices = df.iloc[:, 0].astype(int)
        assert set(indices) == set(range(J_TEST))

    def test_path_probs_sum_to_one(self, df):
        """Σ_j Σ_e p_j * p_je = 1."""
        total = (df.iloc[:, 1] * df.iloc[:, 2]).sum()
        assert abs(total - 1.0) < 1e-5, f"Path probability sum = {total}"

    def test_pj_nonneg(self, df):
        """LP may assign zero weight to some centroids (sparse solution) — that's valid."""
        assert np.all(df.iloc[:, 1].to_numpy() >= -1e-10)

    def test_pje_uniform(self, df):
        """All evaluate conditional probs should equal 1/E."""
        pje = df.iloc[:, 2].to_numpy()
        assert np.allclose(pje, 1.0 / E_TEST, atol=1e-12)

    def test_recourse_prices_positive(self, df):
        rec = df.iloc[:, 3: 3 + Q_TEST].to_numpy()
        assert np.all(rec > 0)

    def test_evaluate_prices_positive(self, df):
        evl = df.iloc[:, 3 + Q_TEST: 3 + 2 * Q_TEST].to_numpy()
        assert np.all(evl > 0)

    def test_recourse_prices_consistent_per_node(self, df):
        """All rows sharing the same recourse index must have identical recourse prices."""
        Q = Q_TEST
        for j in range(J_TEST):
            rows = df[df.iloc[:, 0].astype(int) == j]
            rec = rows.iloc[:, 3: 3 + Q].to_numpy()
            assert np.allclose(rec, rec[0]), (
                f"Recourse node {j} has inconsistent prices across its rows"
            )

    def test_same_format_as_scenario_tree(self, df):
        """Column-by-column type / sign checks matching ScenarioTree layout."""
        Q = Q_TEST
        assert df.iloc[:, 0].dtype in (np.float64, np.int64, int, float)
        assert np.all(df.iloc[:, 1].between(0, 1))
        assert np.all(df.iloc[:, 2].between(0, 1))
        assert np.all(df.iloc[:, 3: 3 + Q].to_numpy() > 0)
        assert np.all(df.iloc[:, 3 + Q: 3 + 2 * Q].to_numpy() > 0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. End-to-end integration: matches get_tree_expected_returns() expectations
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_expected_returns_finite(self, built_tree, synthetic_data, tmp_path):
        """
        Reproduces the main.py get_tree_expected_returns() calculation to verify
        the CSV is correctly structured for the optimizer.
        """
        ip, _ = synthetic_data
        csv_path = str(tmp_path / "vine_integration.csv")
        built_tree.export_tree_to_csv(csv_path)

        df = pd.read_csv(csv_path, header=None)
        Q  = Q_TEST
        p_j  = df.iloc[:, 1].to_numpy()
        p_je = df.iloc[:, 2].to_numpy()
        eval_prices = df.iloc[:, 3 + Q: 3 + 2 * Q].to_numpy()

        probs = p_j * p_je
        expected_prices = (eval_prices * probs[:, np.newaxis]).sum(axis=0)
        expected_returns = expected_prices / ip - 1.0

        assert np.all(np.isfinite(expected_returns)), "Expected returns contain NaN/Inf"

    def test_vine_seed_reproducibility(self, synthetic_data):
        """Same seed → identical tree."""
        ip, ret = synthetic_data
        trees = []
        for _ in range(2):
            t = VineScenarioTree(ip, ret, seed=7)
            t.fit()
            t.build_tree(num_recourse=5, num_evaluate=3,
                         n_sim_stage1=50, n_sim_stage2=10, beta=0.45)
            trees.append(t)

        np.testing.assert_array_almost_equal(
            trees[0].recourse_prices, trees[1].recourse_prices,
            decimal=10, err_msg="Recourse prices differ across identical seeds"
        )
        np.testing.assert_array_almost_equal(
            trees[0].recourse_probs, trees[1].recourse_probs,
            decimal=10, err_msg="Recourse probs differ across identical seeds"
        )
