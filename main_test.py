"""
Quick-test orchestrator — tiny tree, few frontier points, fast runtime.

Intended for visual sanity checks BEFORE committing to a full frontier run.
Not a substitute for the real run: CVaR estimates from 100 scenarios are noisy.

Key differences from main.py:
  - Regenerates a fresh copula tree with 20 recourse × 5 evaluate = 100 rows
  - 5 mu points instead of 15
  - Population 50 instead of 200
  - Cardinality K=5 instead of 10
  - Logs go to logs/test/
"""

import os
import subprocess
import numpy as np
import pandas as pd
import re
from datetime import datetime

from src.scenario_tree import ScenarioTree
from src.paths import optimizer_exe_path
from main import run_frontier


if __name__ == "__main__":
    tickers = sorted([
        "MMM", "AMZN", "AXP", "AMGN", "AAPL",
        "BA", "CAT", "CVX", "CSCO", "KO",
        "DIS", "GS", "HD", "HON", "IBM",
        "JNJ", "JPM", "MCD", "MRK", "MSFT",
        "NKE", "NVDA", "PG", "CRM", "SHW",
        "TRV", "UNH", "VZ", "V", "WMT"
    ])

    prices = pd.DataFrame({
        t: np.load(f"data/raw/{t}.npy") for t in tickers
    })
    returns = (prices / prices.shift(1) - 1.0).dropna()
    initial_prices = prices.iloc[-1].to_numpy()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_file = optimizer_exe_path(current_dir)
    log_dir = os.path.join(current_dir, "logs", "test")

    K = 5          # smaller cardinality — easier to spot concentration issues
    POPULATION = 50

    mean_returns = returns.mean()
    
    # Clamp the minimum target return to 0.0 so we don't sweep through guaranteed losses
    min_mu = max(0.0, float(mean_returns.min()))
    
    w_min = 0.01
    top_k_returns = mean_returns.nlargest(K)
    max_possible_return = top_k_returns.iloc[0] * (1.0 - (K - 1) * w_min) + top_k_returns.iloc[1:].sum() * w_min
    max_mu = float(max_possible_return) * 0.90
    mu_values = np.linspace(min_mu, max_mu, 5)  # 5 points: fast sweep

    # ---- Build a tiny copula tree on the fly --------------------------------
    test_csv = os.path.join(current_dir, "data", "scenarios", "tcvae_tree_test11.csv")
    # os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    # print("Building test scenario tree (20 recourse × 5 evaluate = 100 rows)...")
    # tree = ScenarioTree(
    #     initial_prices=initial_prices,
    #     historical_returns=returns.to_numpy(),
    #     seed=42,
    # )
    # tree.build_tree(
    #     num_recourse=20,
    #     num_evaluate=5,
    #     correct_moments_and_correlation=False,  # skip HKW — not useful at n=20
    #     evaluate_method="uniform",
    # )
    # tree.export_tree_to_csv(test_csv)

    # ---- Print a quick sanity check on the test tree ------------------------
    df_tree = pd.read_csv(test_csv, header=None)
    recourse_prices = df_tree.iloc[:, 3 : 3 + len(tickers)].to_numpy()
    # One row per evaluate node; take unique recourse rows
    recourse_returns_check = recourse_prices[::5] / initial_prices - 1.0
    print("\n--- Test-tree recourse return stats (should roughly match history) ---")
    print(f"  mean:   {recourse_returns_check.mean(axis=0).mean():.6f}  "
          f"(history: {returns.mean().mean():.6f})")
    print(f"  std:    {recourse_returns_check.std(axis=0).mean():.6f}  "
          f"(history: {returns.std().mean():.6f})")
    wmt_idx = tickers.index("WMT")
    print(f"  WMT recourse returns: {recourse_returns_check[:, wmt_idx]}")
    print()

    # ---- Run the tiny frontier ----------------------------------------------
    frontier = run_frontier(
        exe_file=exe_file,
        csv_file=test_csv,
        tickers=tickers,
        initial_prices=initial_prices,
        mu_values=mu_values,
        cardinality=K,
        population=POPULATION,
        method_label="test_tree",
        log_dir=log_dir,
    )

    # ---- Summary ------------------------------------------------------------
    print("\n=== Test Frontier Summary ===")
    for mu, cvar, asset_weights in frontier:
        print(f"mu={mu:.4f} | CVaR={cvar:.6f} | "
              + ", ".join(f"{t}: {w*100:.1f}%" for t, _, _, _, w in asset_weights))
