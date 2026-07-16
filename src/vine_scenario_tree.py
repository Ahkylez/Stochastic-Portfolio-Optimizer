"""
He & Zhang (2024) vine copula scenario tree generator.

Pipeline:
  1. Fit GARCH(1,1)-SkewStudent per asset (arch library)
  2. Empirical PIT → fit R-vine copula (pyvinecopulib)
  3. Simulate large stage-1 set → K-means → LP moment matching for p_j
  4. For each recourse node: update GARCH state, simulate stage-2 paths
  5. Export in the same CSV format as ScenarioTree

Required: pip install arch pyvinecopulib scikit-learn scipy
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, List
from scipy.optimize import linprog
from sklearn.cluster import KMeans

try:
    from arch import arch_model
except ImportError:
    raise ImportError("pip install arch")

try:
    import pyvinecopulib as pv
except ImportError:
    raise ImportError("pip install pyvinecopulib")


class VineScenarioTree:
    """
    Two-stage scenario tree via GARCH+vine copula (He & Zhang 2024).
    Drop-in replacement for ScenarioTree — produces the same CSV format.
    """

    def __init__(
        self,
        initial_prices: np.ndarray,
        historical_returns: np.ndarray,
        seed: Optional[int] = None,
    ):
        if initial_prices.ndim != 1:
            raise ValueError("initial_prices must be 1-D")
        Q = initial_prices.shape[0]
        if historical_returns.shape[1] != Q:
            raise ValueError(
                f"historical_returns has {historical_returns.shape[1]} columns "
                f"but initial_prices has {Q} assets"
            )

        self.initial_prices = initial_prices.copy()
        self.historical_returns = historical_returns.copy()
        self.Q = Q
        self.T = historical_returns.shape[0]
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Populated by fit()
        self._garch_params: List[dict] = []
        self._z_std: Optional[np.ndarray] = None   # (T, Q) standardized residuals
        self._vine: Optional[pv.Vinecop] = None
        self._fitted = False

        # Populated by build_tree()
        self.recourse_prices: Optional[np.ndarray] = None   # (J, Q)
        self.recourse_probs: Optional[np.ndarray] = None    # (J,)
        self.evaluate_prices: Optional[List[np.ndarray]] = None  # list of (E, Q)
        self.evaluate_probs: Optional[List[np.ndarray]] = None   # list of (E,)
        self._built = False

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Fit GARCH(1,1)-SkewStudent per asset, then R-vine on residuals."""
        Q, T = self.Q, self.T
        print(f"Fitting GARCH(1,1)-SkewStudent to {Q} assets ({T} obs)...")

        z = np.empty((T, Q))

        for i in range(Q):
            # percent scale → better numerical conditioning for GARCH
            ret_pct = self.historical_returns[:, i] * 100.0
            am = arch_model(ret_pct, mean='Constant', vol='GARCH',
                            p=1, q=1, dist='skewt')
            res = am.fit(disp='off', options={'ftol': 1e-9, 'maxiter': 500})

            idx = res.params.index.tolist()
            mu_i     = float(res.params[[k for k in idx if k in ('mu', 'Const', 'const')][0]])
            omega_i  = float(res.params['omega'])
            alpha_i  = float(res.params[[k for k in idx if k.startswith('alpha')][0]])
            beta_g_i = float(res.params[[k for k in idx if k.startswith('beta')][0]])

            cv = res.conditional_volatility
            last_h_i     = float((cv[-1] if not hasattr(cv, 'iloc') else cv.iloc[-1]) ** 2)
            rd = res.resid
            last_innov_i = float(rd[-1] if not hasattr(rd, 'iloc') else rd.iloc[-1])

            self._garch_params.append({
                'mu': mu_i, 'omega': omega_i,
                'alpha': alpha_i, 'beta': beta_g_i,
                'last_h': last_h_i, 'last_innov': last_innov_i,
            })
            sr = res.std_resid
            z[:, i] = sr if not hasattr(sr, 'to_numpy') else sr.to_numpy()

            if (i + 1) % 10 == 0 or i == Q - 1:
                print(f"  [{i+1}/{Q}] GARCH fits done")

        self._z_std = z

        # Empirical PIT (Hazen plotting position)
        from scipy.stats import rankdata as _rd
        u = np.column_stack([_rd(z[:, i]) / (T + 1) for i in range(Q)])

        print("Fitting R-vine copula...")
        controls = pv.FitControlsVinecop(
            family_set=pv.all,
            selection_criterion='bic',
            num_threads=min(4, os.cpu_count() or 1),
        )
        self._vine = pv.Vinecop(Q)
        self._vine.select(u, controls)
        self._fitted = True
        print("Vine copula fitted.")

    # ------------------------------------------------------------------
    def _stage1_sigmas(self) -> np.ndarray:
        """Forecast stage-1 conditional vols (percent scale)."""
        s = np.empty(self.Q)
        for i, p in enumerate(self._garch_params):
            h = p['omega'] + p['alpha'] * p['last_innov']**2 + p['beta'] * p['last_h']
            s[i] = np.sqrt(max(h, 1e-12))
        return s

    def _stage2_sigmas(self, ret1_pct: np.ndarray, sig1: np.ndarray) -> np.ndarray:
        """Update GARCH state given stage-1 realized return (pct) → stage-2 vols."""
        s = np.empty(self.Q)
        for i, p in enumerate(self._garch_params):
            h1 = sig1[i] ** 2
            innov1 = ret1_pct[i] - p['mu']
            h2 = p['omega'] + p['alpha'] * innov1**2 + p['beta'] * h1
            s[i] = np.sqrt(max(h2, 1e-12))
        return s

    def _sample_z(self, n: int, seed_offset: int = 0) -> np.ndarray:
        """
        Draw n joint residual vectors from vine copula and back-transform
        via empirical quantile of historical standardized residuals.
        Returns (n, Q).
        """
        T = self._z_std.shape[0]
        seed_arg = [] if self._seed is None else [self._seed + seed_offset]
        u_sim = self._vine.simulate(n=n, seeds=seed_arg)   # (n, Q) uniform

        z_sorted = np.sort(self._z_std, axis=0)             # (T, Q)
        grid = (np.arange(1, T + 1) - 0.5) / T             # Hazen positions

        z_sim = np.column_stack([
            np.interp(u_sim[:, i], grid, z_sorted[:, i])
            for i in range(self.Q)
        ])
        return z_sim

    def _simulate_stage1(self, n: int) -> tuple:
        """
        Simulate n stage-1 decimal returns.
        Returns (ret_dec (n,Q), sig1_pct (Q,)).
        """
        sig1 = self._stage1_sigmas()
        z = self._sample_z(n, seed_offset=0)
        ret_pct = np.column_stack([
            self._garch_params[i]['mu'] + sig1[i] * z[:, i]
            for i in range(self.Q)
        ])
        return ret_pct / 100.0, sig1

    def _simulate_stage2(
        self, n: int, ret1_pct: np.ndarray, sig1: np.ndarray, seed_offset: int
    ) -> np.ndarray:
        """
        Simulate n stage-2 decimal returns conditioned on stage-1 realization.
        """
        sig2 = self._stage2_sigmas(ret1_pct, sig1)
        z = self._sample_z(n, seed_offset=seed_offset)
        ret_pct = np.column_stack([
            self._garch_params[i]['mu'] + sig2[i] * z[:, i]
            for i in range(self.Q)
        ])
        return ret_pct / 100.0

    # ------------------------------------------------------------------
    def _moment_matching_lp(
        self,
        centroids: np.ndarray,
        large_sample: np.ndarray,
        beta: float = 0.45,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        LP moment matching (He & Zhang 2024, eq 13-21).

        Finds probability vector q over J cluster centroids that minimises
        the deviation from the distributional moments of large_sample.
        Mean is softly enforced with weight 1000× to guarantee feasibility
        while strongly prioritising mean preservation.

        Returns q (J,) summing to 1, all >= 0.
        """
        J, Q = centroids.shape
        N = large_sample.shape[0]

        # ── Target moments from large_sample ───────────────────────────
        M    = large_sample.mean(axis=0)                          # (Q,)
        C    = np.cov(large_sample, rowvar=False)                 # (Q,Q)
        sig  = np.sqrt(np.diag(C).clip(1e-16))                   # (Q,)
        cent = large_sample - M
        SK   = (cent**3).mean(axis=0) / sig**3                    # (Q,)
        KT   = (cent**4).mean(axis=0) / sig**4                    # (Q,)
        
        asset_vars = np.quantile(large_sample, beta, axis=0)      # (Q,)
        Rt   = asset_vars.mean()                                  # Scalar minimal tolerable return rate (Eq 12)
        TM   = (large_sample * (large_sample < Rt)).mean(axis=0) / (1.0 - beta)  # (Q,)

        # ── Centroid feature matrices ───────────────────────────────────
        cov_pairs = [(i, k) for i in range(Q) for k in range(i, Q)]
        n_cov  = len(cov_pairs)                  # Q*(Q+1)/2
        n_soft = n_cov + Q + Q + Q              # cov + skew + kurt + tail

        # raw second moment feature: r_ji * r_jk
        F_cov  = np.array([[centroids[j, i] * centroids[j, k]
                             for (i, k) in cov_pairs]
                            for j in range(J)])                    # (J, n_cov)
        target_cov = np.array([C[i, k] + M[i] * M[k]
                                for (i, k) in cov_pairs])         # (n_cov,)

        F_skew = ((centroids - M) ** 3) / sig**3                  # (J, Q)
        F_kurt = ((centroids - M) ** 4) / sig**4                  # (J, Q)
        tail_m = (centroids < Rt)
        F_tail = centroids * tail_m / (1.0 - beta)                # (J, Q)

        # ── Variable layout ─────────────────────────────────────────────
        # [q(J) | s+_mean(Q) s-_mean(Q) | s+_soft(n_soft) s-_soft(n_soft)]
        n_vars = J + 2 * Q + 2 * n_soft
        # W_MEAN controls how hard the LP fights to match the historical mean.
        # 1000 (old) caused the LP to zero out all downside centroids to nail
        # the positive bull-market mean, producing a tree with no downside.
        # 20 still makes mean the dominant objective but lets the LP retain
        # representative downside scenarios for proper CVaR computation.
        W_MEAN = 20.0

        c_obj = np.zeros(n_vars)
        c_obj[J        : J + Q]            = W_MEAN   # s+_mean
        c_obj[J + Q    : J + 2 * Q]        = W_MEAN   # s-_mean
        c_obj[J + 2*Q  : J + 2*Q + n_soft] = 1.0      # s+_soft
        c_obj[J + 2*Q + n_soft :]           = 1.0      # s-_soft

        # ── Equality constraints ─────────────────────────────────────────
        rows, rhs = [], []

        def _add(q_coefs, sp_idx, sm_idx, target):
            row = np.zeros(n_vars)
            row[:J]      = q_coefs
            row[sp_idx]  = -1.0
            row[sm_idx]  =  1.0
            rows.append(row)
            rhs.append(target)

        # 1. Sum = 1
        r = np.zeros(n_vars); r[:J] = 1.0
        rows.append(r); rhs.append(1.0)

        # 2. Mean (soft, high weight)
        for i in range(Q):
            _add(centroids[:, i], J + i, J + Q + i, M[i])

        # 3. Covariance
        base = J + 2 * Q
        for idx, t in enumerate(target_cov):
            _add(F_cov[:, idx], base + idx, base + n_soft + idx, t)

        # 4. Skewness
        for i in range(Q):
            idx = n_cov + i
            _add(F_skew[:, i], base + idx, base + n_soft + idx, SK[i])

        # 5. Kurtosis
        for i in range(Q):
            idx = n_cov + Q + i
            _add(F_kurt[:, i], base + idx, base + n_soft + idx, KT[i])

        # 6. Tail mean
        for i in range(Q):
            idx = n_cov + 2 * Q + i
            _add(F_tail[:, i], base + idx, base + n_soft + idx, TM[i])

        A_eq = np.array(rows)
        b_eq = np.array(rhs)
        bounds = [(0.0, None)] * n_vars

        if verbose:
            print(f"  LP: {J} centroids | {n_soft} soft constraints | "
                  f"{n_vars} vars | {len(b_eq)} eq rows")

        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs',
                      options={'time_limit': 120.0, 'disp': False})

        if res.status not in (0, 1):
            if verbose:
                print(f"  LP fallback (status {res.status}): using uniform weights")
            return np.ones(J) / J

        q = np.maximum(res.x[:J], 0.0)
        total = q.sum()
        if total < 1e-10:
            return np.ones(J) / J
        return q / total

    # ------------------------------------------------------------------
    def build_tree(
        self,
        num_recourse: int = 150,
        num_evaluate: int = 20,
        n_sim_stage1: int = 2000,
        n_sim_stage2: int = 100,
        beta: float = 0.45,
    ) -> None:
        """
        Build the two-stage scenario tree.

        num_recourse:  J   number of stage-1 (recourse) nodes
        num_evaluate:  E   number of stage-2 (evaluate) nodes per recourse
        n_sim_stage1:  large simulation pool for K-means (>= num_recourse)
        n_sim_stage2:  stage-2 paths simulated per cluster (>= num_evaluate)
        beta:          tail threshold for LP (He & Zhang optimal: 0.45)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before build_tree()")
        if n_sim_stage1 < num_recourse:
            raise ValueError("n_sim_stage1 must be >= num_recourse")
        if n_sim_stage2 < num_evaluate:
            n_sim_stage2 = num_evaluate

        # ── Stage 1: large simulation → K-means → LP ───────────────────
        print(f"\nSimulating {n_sim_stage1} stage-1 paths...")
        ret1, sig1 = self._simulate_stage1(n_sim_stage1)   # (N, Q) decimal

        print(f"K-means → {num_recourse} centroids...")
        km = KMeans(
            n_clusters=num_recourse, n_init=10, max_iter=300,
            random_state=self._seed if self._seed is not None else 42,
        )
        km.fit(ret1)
        centroids = km.cluster_centers_   # (J, Q) decimal returns

        # ── Diagnostic: pre-LP return distribution ──────────────────────
        tickers_sorted = None
        try:
            import pandas as pd
            from main import TICKERS as _T
            tickers_sorted = _T
        except Exception:
            pass
        _pcts = [5, 50, 95]
        print("\n  Pre-LP simulated returns (5th / 50th / 95th pct):")
        for qi in range(min(ret1.shape[1], 5)):
            vals = np.percentile(ret1[:, qi], _pcts) * 100
            lbl = tickers_sorted[qi] if tickers_sorted else f"asset{qi}"
            print(f"    {lbl:<6}: {vals[0]:+.2f}%  {vals[1]:+.2f}%  {vals[2]:+.2f}%")

        print("LP moment matching...")
        q = self._moment_matching_lp(centroids, ret1, beta=beta, verbose=True)

        # ── Diagnostic: post-LP effective distribution ───────────────────
        active = np.sum(q > 1e-8)
        print(f"\n  Post-LP: {active}/{len(q)} centroids have weight > 0")
        print("  Post-LP weighted means vs. raw means (should be close):")
        for qi in range(min(centroids.shape[1], 5)):
            raw_mean = ret1[:, qi].mean() * 100
            lp_mean  = (q * centroids[:, qi]).sum() * 100
            min_active = centroids[q > 1e-8, qi].min() * 100
            lbl = tickers_sorted[qi] if tickers_sorted else f"asset{qi}"
            print(f"    {lbl:<6}: raw_mean={raw_mean:+.3f}%  lp_mean={lp_mean:+.3f}%  "
                  f"min_active_centroid={min_active:+.2f}%")

        self.recourse_prices = self.initial_prices[np.newaxis, :] * (1.0 + centroids)
        self.recourse_probs  = q

        # ── Stage 2: conditional simulation per recourse node ──────────
        print(f"Building stage-2 nodes ({num_recourse} × {num_evaluate})...")
        self.evaluate_probs = []
        self.evaluate_prices = []

        for j in range(num_recourse):
            ret1_pct_j = centroids[j] * 100.0
            ret2 = self._simulate_stage2(
                n_sim_stage2, ret1_pct_j, sig1, seed_offset=j + 1
            )   # (n_sim_stage2, Q) decimal
            
            # Cluster the stage-2 samples
            km2 = KMeans(
                n_clusters=num_evaluate, n_init=3, max_iter=200,
                random_state=(self._seed + j) if self._seed is not None else None,
            )
            km2.fit(ret2)
            centroids2 = km2.cluster_centers_
            
            # LP moment matching for conditional probability (q^m)
            q2 = self._moment_matching_lp(centroids2, ret2, beta=beta, verbose=False)
            
            eval_p = self.recourse_prices[j] * (1.0 + centroids2)   # (E, Q)
            self.evaluate_prices.append(eval_p)
            self.evaluate_probs.append(q2)

            if (j + 1) % 10 == 0 or j == num_recourse - 1:
                print(f"  [{j+1}/{num_recourse}] stage-2 LPs solved")

        self._built = True
        total = num_recourse * num_evaluate
        print(
            f"\nTree built: {num_recourse} recourse × {num_evaluate} evaluate "
            f"= {total} rows | p_j [{q.min():.5f}, {q.max():.5f}] "
            f"sum={q.sum():.6f}"
        )

    # ------------------------------------------------------------------
    def export_tree_to_csv(self, filename: str) -> None:
        """
        Export tree in the format expected by the C++ solver.
        Same column layout as ScenarioTree.export_tree_to_csv:
            col 0:          recourse index j (0-indexed)
            col 1:          p_j
            col 2:          p_je  (conditional probability of evaluate node)
            cols 3..Q+2:    recourse prices P^j
            cols Q+3..2Q+2: evaluate prices P^{j,e}
        No header row.
        """
        if not self._built:
            raise RuntimeError("Call build_tree() before export_tree_to_csv()")

        J = len(self.recourse_prices)
        E = len(self.evaluate_prices[0])
        n_rows = J * E
        n_cols = 3 + 2 * self.Q

        out = np.empty((n_rows, n_cols), dtype=np.float64)
        r = 0
        for j in range(J):
            pj  = self.recourse_probs[j]
            rp  = self.recourse_prices[j]
            for e in range(E):
                out[r, 0] = j
                out[r, 1] = pj
                out[r, 2] = self.evaluate_probs[j][e]
                out[r, 3        : 3 + self.Q] = rp
                out[r, 3 + self.Q :          ] = self.evaluate_prices[j][e]
                r += 1

        dirpath = os.path.dirname(filename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        pd.DataFrame(out).to_csv(filename, index=False, header=False)

        print(
            f"Exported {n_rows} rows → '{filename}' "
            f"({J} recourse × {E} evaluate, {self.Q} assets)"
        )
