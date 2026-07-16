"""
The paper uses a shape-based copula scenario generator (Kaut & Wallace 2011)
for the recourse nodes and U(0.9, 1.1) multiplicative fluctuation for 
evaluate nodes.
It produces the CSV file rows like this:
    [recourse_idx | p_j | p_je | Q recourse prices | Q evaluate prices]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from src.scenario_generator import generate_scenarios


class ScenarioNode:
    def __init__(self, stage: int, data: np.ndarray, prob: float,
        parent: ScenarioNode | None = None,
    ):
        self.stage = stage          # 0 = root, 1 = recourse, 2 = evaluate
        self.data = data            # absolute prices for the Q assets
        self.prob = prob            # conditional probability p_n
        self.parent = parent        # link to a_{t-1}(n)
        self.children: List[ScenarioNode] = []

        # Path probability: product of conditional probs along the path
        self.path_prob = prob if parent is None else parent.path_prob * prob


class ScenarioTree:
    """
    Two-stage scenario tree built from historical returns using a shape-
    based copula. The tree has the structure required by Cui et al.'s
    model (Sec 3 of that paper):
        root (t=0)  ->  recourse nodes (t=1)  ->  evaluate nodes (t=2)

    Recourse nodes are sampled from the copula generator (preserving
    fat-tail and tail-dependence structure of historical returns).
    Evaluate nodes use the U(0.9, 1.1) multiplicative noise from
    Cui et al. Sec 5.3 — a deliberately simple model the paper itself
    notes is not optimal but adequate.

    Pass `evaluate_method="copula"` to instead generate evaluate nodes
    by re-running the copula on a tighter return distribution per
    recourse node — closer in spirit to a proper conditional tree.
    """

    def __init__(
        self,
        initial_prices: np.ndarray,
        historical_returns: np.ndarray,
        seed: int | None = None,
    ):
        """
        initial_prices:     (Q,) array of t=0 prices, one per asset.
        historical_returns: (T, Q) array of past returns, the empirical
                            distribution we sample from.
        seed:               RNG seed for reproducibility (controls both
                            copula sampling and U(0.9, 1.1) draws).
        """
        if initial_prices.ndim != 1:
            raise ValueError("initial_prices must be 1-D")
        if historical_returns.shape[1] != initial_prices.shape[0]:
            raise ValueError(
                f"historical_returns has {historical_returns.shape[1]} columns "
                f"but initial_prices has {initial_prices.shape[0]} assets"
            )

        self.initial_prices = initial_prices.copy()
        self.historical_returns = historical_returns
        self.num_assets = initial_prices.shape[0]
        self._rng = np.random.default_rng(seed)
        self._seed = seed  # passed into generate_scenarios for the copula step

        # Root node holds the t=0 prices
        self.root = ScenarioNode(
            stage=0, data=self.initial_prices.copy(), prob=1.0
        )

        # Populated by build_tree
        self.recourse_nodes: List[ScenarioNode] = []
        self.leaves: List[ScenarioNode] = []

    # ------------------------------------------------------------------
    def build_tree(
        self,
        num_recourse: int = 400,
        num_evaluate: int = 40,
        correct_moments_and_correlation: bool = True,
        evaluate_method: str = "uniform",
    ) -> None:
        """
        Build the two-stage tree.

        num_recourse:    number of t=1 nodes (paper uses 400)
        num_evaluate:    number of t=2 children per recourse node (paper uses 40)
        correct_moments_and_correlation:
                         passed to the copula generator; True applies the
                         Hoyland-Kaut-Wallace post-process to drive sample
                         moments and correlations to the historical targets.
                         Recommended for num_recourse >= 250 (paper Sec 3.3).
        evaluate_method: "uniform" reproduces Cui et al. Sec 5.3 — multiply
                         each recourse-node price by an independent
                         U(0.9, 1.1) draw. Cheap and what the paper does.
                         "copula" instead samples num_evaluate scenarios
                         from a copula fit to the historical returns,
                         scaled to a tighter range (~10%) — preserves
                         dependence structure conditional on the recourse
                         node, at the cost of more compute.
        """
        if evaluate_method not in {"uniform", "copula"}:
            raise ValueError("evaluate_method must be 'uniform' or 'copula'")

        # ---- Stage 1: copula-based recourse nodes -----------------------
        # generate_scenarios returns RETURNS, not prices.
        recourse_returns = generate_scenarios(
            historical=self.historical_returns,
            n_scenarios=num_recourse,
            correct_moments_and_correlation=correct_moments_and_correlation,
            seed=self._seed,
        )  # shape (num_recourse, Q)

        # Equal probability per node, matching paper Sec 6.1.
        p_j = 1.0 / num_recourse

        for j in range(num_recourse):
            # P^j_i = P^0_i * (1 + r^j_i)
            t1_prices = self.initial_prices * (1.0 + recourse_returns[j])

            recourse_node = ScenarioNode(
                stage=1,
                data=t1_prices,
                prob=p_j,
                parent=self.root,
            )
            self.root.children.append(recourse_node)
            self.recourse_nodes.append(recourse_node)

        # ---- Stage 2: evaluate nodes -----------------------------------
        p_je = 1.0 / num_evaluate

        if evaluate_method == "uniform":
            # Cui et al. Sec 5.3: independent U(0.9, 1.1) per asset per
            # evaluate node. Reproduces the original tree exactly when
            # the same seed is used.
            for recourse_node in self.recourse_nodes:
                for _ in range(num_evaluate):
                    fluctuation = self._rng.uniform(
                        low=0.9, high=1.1, size=self.num_assets
                    )
                    t2_prices = recourse_node.data * fluctuation
                    leaf = ScenarioNode(
                        stage=2,
                        data=t2_prices,
                        prob=p_je,
                        parent=recourse_node,
                    )
                    recourse_node.children.append(leaf)
                    self.leaves.append(leaf)

        else:  # "copula"
            # Re-sample the copula per recourse node, scaled to a tighter
            # range so evaluate nodes represent short-horizon noise around
            # the recourse-node prices. We pass a fresh seed per node so
            # children are not all identical.
            for j, recourse_node in enumerate(self.recourse_nodes):
                node_seed = None if self._seed is None else self._seed + j + 1
                # Scale historical returns by 0.1 to model short-horizon
                # fluctuation — magnitude similar to U(0.9, 1.1).
                eval_returns = generate_scenarios(
                    historical=self.historical_returns * 0.1,
                    n_scenarios=num_evaluate,
                    correct_moments_and_correlation=correct_moments_and_correlation,
                    seed=node_seed,
                )
                for e in range(num_evaluate):
                    t2_prices = recourse_node.data * (1.0 + eval_returns[e])
                    leaf = ScenarioNode(
                        stage=2,
                        data=t2_prices,
                        prob=p_je,
                        parent=recourse_node,
                    )
                    recourse_node.children.append(leaf)
                    self.leaves.append(leaf)

        print(
            f"Tree built: 1 root | {len(self.recourse_nodes)} recourse nodes "
            f"| {len(self.leaves)} evaluate leaves "
            f"(evaluate_method={evaluate_method!r})"
        )

    # ------------------------------------------------------------------
    def reduce(
        self,
        num_recourse: int,
        num_evaluate: int,
        seed: int | None = None,
    ) -> None:
        """
        Reduce the built tree to (num_recourse x num_evaluate) using two-stage
        k-means scenario reduction.

        Why reduce rather than building small directly
        ----------------------------------------------
        The HKW moment-matching post-process in generate_scenarios() is only
        reliable for n_scenarios >= 250 (paper Sec 3.3). Building a large tree
        first (e.g. 400 x 40) and then reducing preserves distributional quality
        while matching the TC-VAE tree's 150 x 20 structure so the two methods
        are directly comparable.

        Stage-1 (recourse) reduction
        -----------------------------
        k-means(num_recourse) on the J original stage-1 return vectors.
        Each centroid becomes the new representative recourse scenario.
        Probability: p_j = cluster_size / J  (probability-weighted, mirrors
        the TC-VAE exporter's approach so frontier comparisons are apples-to-apples).

        Stage-2 (evaluate) reduction per recourse cluster
        --------------------------------------------------
        For cluster j, pool the stage-2 returns (relative to each original
        parent's prices) from every original recourse node in the cluster.
        Run k-means(num_evaluate) on the pool, then compute evaluate prices
        as new_recourse_prices * (1 + centroid_return).
        Equal weight: p_je = 1 / num_evaluate.

        Pooling stage-2 RETURNS rather than absolute prices is correct because
        the evaluate fluctuations (U(0.9,1.1) or copula-based) are identically
        distributed across recourse nodes — only the scale (recourse price) differs.

        Overwrites self.recourse_nodes and self.leaves in-place so that
        export_tree_to_csv() works unchanged on the reduced tree.

        Parameters
        ----------
        num_recourse : target number of stage-1 nodes (must be < current J)
        num_evaluate : target children per recourse node (must be <= current E)
        seed         : RNG seed forwarded to k-means and any resampling
        """
        if not self.recourse_nodes:
            raise RuntimeError("Call build_tree() before reduce()")

        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances

        J_orig = len(self.recourse_nodes)
        E_orig = len(self.recourse_nodes[0].children)

        if num_recourse >= J_orig:
            raise ValueError(
                f"num_recourse={num_recourse} must be < current J={J_orig}. "
                f"Build a larger tree first."
            )
        if num_evaluate > E_orig:
            raise ValueError(
                f"num_evaluate={num_evaluate} must be <= current E={E_orig}."
            )

        km_seed = seed if seed is not None else 42
        rng = np.random.default_rng(seed)

        # ── Stage 1: k-medoids on recourse return vectors ─────────────────────
        # R1[j] = (P^j / P^0) - 1  in R^Q
        R1 = np.stack([
            node.data / self.initial_prices - 1.0
            for node in self.recourse_nodes
        ])  # (J_orig, Q)

        # Run k-means to find initial cluster centres, then snap each centre to
        # the nearest ACTUAL recourse node in return space.  This gives true
        # k-medoids: every representative is a real generated scenario, not a
        # synthetic average.  Preserves the copula's joint structure (marginals,
        # tail dependence) that HKW post-processing enforced.
        km1 = KMeans(n_clusters=num_recourse, n_init=10, max_iter=300,
                     random_state=km_seed)
        km1.fit(R1)
        centroids1 = km1.cluster_centers_  # (num_recourse, Q) — synthetic, will be snapped

        # Snap each centroid to its nearest real recourse node
        _CHUNK = 512
        medoid_idx = np.zeros(num_recourse, dtype=int)
        best_dist  = np.full(num_recourse, np.inf)
        for start in range(0, J_orig, _CHUNK):
            end = min(start + _CHUNK, J_orig)
            d = euclidean_distances(centroids1, R1[start:end])  # (num_recourse, chunk)
            local_best  = d.argmin(axis=1)
            local_dists = d[np.arange(num_recourse), local_best]
            improved = local_dists < best_dist
            best_dist[improved]  = local_dists[improved]
            medoid_idx[improved] = start + local_best[improved]

        # Reassign every original node to its nearest medoid (Voronoi partition)
        medoid_R1 = R1[medoid_idx]         # (num_recourse, Q) — real scenario returns
        labels1   = np.empty(J_orig, dtype=int)
        for start in range(0, J_orig, _CHUNK):
            end = min(start + _CHUNK, J_orig)
            d = euclidean_distances(medoid_R1, R1[start:end])
            labels1[start:end] = d.argmin(axis=0)

        # Voronoi-cell probabilities: p_j = cluster_size / J_orig.
        # This ensures the probability-weighted mean of medoid returns matches
        # the sample mean of all J_orig nodes (Wasserstein consistency property).
        counts1 = np.bincount(labels1, minlength=num_recourse).astype(float)
        p_j_arr = counts1 / counts1.sum()

        print(
            f"  Stage-1 k-medoids: p_j range "
            f"[{p_j_arr.min():.5f}, {p_j_arr.max():.5f}]  "
            f"cluster sizes [{int(counts1.min())}, {int(counts1.max())}]"
        )

        # ── Stage 2: pool evaluate returns per cluster, random-sample num_evaluate ─
        # Random sampling (not k-means) preserves the actual per-asset fluctuation
        # distribution and matches the TC-VAE exporter's approach exactly.
        # Pool stage-2 returns relative to each original parent's own prices so
        # the evaluate fluctuation is independent of the cluster's recourse level.
        p_je = 1.0 / num_evaluate
        new_recourse_nodes: List[ScenarioNode] = []
        new_leaves: List[ScenarioNode] = []

        for j_new in range(num_recourse):
            # Use the medoid's actual prices — no synthetic averaging
            new_rec_prices = self.recourse_nodes[medoid_idx[j_new]].data.copy()

            new_rec_node = ScenarioNode(
                stage=1,
                data=new_rec_prices,
                prob=float(p_j_arr[j_new]),
                parent=self.root,
            )

            member_indices = np.where(labels1 == j_new)[0]
            pool_rows = []
            for orig_idx in member_indices:
                orig_node = self.recourse_nodes[orig_idx]
                for leaf in orig_node.children:
                    pool_rows.append(leaf.data / orig_node.data - 1.0)

            pool = np.stack(pool_rows)  # (pool_size, Q) — real evaluate returns
            pool_size = len(pool)

            replace = pool_size < num_evaluate
            chosen  = rng.choice(pool_size, size=num_evaluate, replace=replace)

            for e in chosen:
                eval_prices = new_rec_prices * (1.0 + pool[e])
                leaf = ScenarioNode(
                    stage=2,
                    data=eval_prices,
                    prob=p_je,
                    parent=new_rec_node,
                )
                new_rec_node.children.append(leaf)
                new_leaves.append(leaf)

            new_recourse_nodes.append(new_rec_node)

        # Replace the tree in-place (export_tree_to_csv is unchanged)
        self.root.children = new_recourse_nodes
        self.recourse_nodes = new_recourse_nodes
        self.leaves = new_leaves

        print(
            f"Scenario reduction: {J_orig} x {E_orig} -> "
            f"{len(self.recourse_nodes)} x {num_evaluate}  "
            f"({len(self.leaves)} total rows)  "
            f"p_j range [{p_j_arr.min():.5f}, {p_j_arr.max():.5f}]  "
            f"(sum={p_j_arr.sum():.6f})"
        )

    # ------------------------------------------------------------------
    def export_tree_to_csv(self, filename: str) -> None:
        """
        Export the tree in the format the C++ solver expects.

        One row per evaluate node (so 16,000 rows for the default tree):
            col 0           : recourse node index j  (0-indexed)
            col 1           : p_j   (recourse node probability)
            col 2           : p_je  (evaluate node conditional probability)
            cols 3..Q+2     : P^j_i      — recourse node prices, Q values
            cols Q+3..2Q+2  : P^(j,e)_i  — evaluate node prices, Q values

        Header is omitted (C++ parser does not skip a header row).
        """
        if not self.recourse_nodes:
            raise RuntimeError("build_tree must be called before export_tree_to_csv")

        # Pre-allocate a single numpy array — much faster than building
        # rows in Python and converting via DataFrame for 16k rows.
        n_rows = sum(len(n.children) for n in self.recourse_nodes)
        n_cols = 3 + 2 * self.num_assets
        out = np.empty((n_rows, n_cols), dtype=np.float64)

        r = 0
        for j, recourse_node in enumerate(self.recourse_nodes):
            for leaf in recourse_node.children:
                out[r, 0] = j
                out[r, 1] = recourse_node.prob
                out[r, 2] = leaf.prob
                out[r, 3 : 3 + self.num_assets] = recourse_node.data
                out[r, 3 + self.num_assets : ] = leaf.data
                r += 1

        # Use pandas for the CSV write only — keeps formatting consistent
        # with the previous version. No header (matches old behavior).
        df = pd.DataFrame(out)
        df.to_csv(filename, index=False, header=False)

        print(
            f"Exported {n_rows} rows to '{filename}'  "
            f"({len(self.recourse_nodes)} recourse nodes × "
            f"{len(self.recourse_nodes[0].children)} evaluate nodes, "
            f"{self.num_assets} assets each)"
        )
        print(
            f"Row layout: [recourse_idx | p_j | p_je | "
            f"{self.num_assets}xP^j | {self.num_assets}xP^(j,e)]"
        )