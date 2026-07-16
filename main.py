"""
Orchestrator: generate the copula scenario tree, then run efficient frontiers
against both copula_tree.csv and the pre-built tcvae_tree.csv. Saves
per-method logs with full portfolio weights for each frontier point.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import re
from datetime import datetime

from src.paths import optimizer_exe_path


def get_tree_expected_returns(csv_file, initial_prices, tickers):
    """Calculate the exact expected return of each asset as represented in the scenario tree."""
    df = pd.read_csv(csv_file, header=None)
    Q = len(tickers)
    p_j = df[1].to_numpy()
    p_je = df[2].to_numpy()
    eval_prices = df.iloc[:, 3+Q : 3+2*Q].to_numpy()
    
    probs = p_j * p_je
    expected_prices = np.sum(eval_prices * probs[:, np.newaxis], axis=0)
    
    return pd.Series(expected_prices / initial_prices - 1.0, index=tickers)


def get_mu_range(csv_file, initial_prices, tickers, K, n_mu, w_min=0.01):
    """Compute exact mu boundaries dynamically based on the tree being evaluated."""
    tree_mean_returns = get_tree_expected_returns(csv_file, initial_prices, tickers)
    
    # Clamp the minimum target return to 0.0 so we don't sweep through guaranteed losses
    min_mu = max(0.0, float(tree_mean_returns.min()))
    
    top_k_returns = tree_mean_returns.nlargest(K)
    max_possible_return = top_k_returns.iloc[0] * (1.0 - (K - 1) * w_min) + top_k_returns.iloc[1:].sum() * w_min
    max_mu = float(max_possible_return) * 0.90
    
    #return np.linspace(min_mu, max_mu, n_mu)
    return [0.0025]

def _parse_frontier_stdout(stdout: str, h: float = 100000.0):
    """
    Parse the C++ engine's stdout for a multi-mu frontier run.
    Each mu emits exactly three lines:
        FINAL_CVAR:<value>      (-1 = infeasible)
        PORTFOLIO_K:<indices>
        PORTFOLIO_W:<shares>
    Returns a list of (cvar_norm, k_indices, weights) or None per mu point.
    """
    results = []
    lines = stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        m = re.search(r"FINAL_CVAR:([0-9.eE+\-]+)", lines[i])
        if m:
            raw_cvar = float(m.group(1))
            k_indices, weights = [], []
            for j in range(i + 1, min(i + 3, len(lines))):
                km = re.search(r"PORTFOLIO_K:([0-9,]*)", lines[j])
                wm = re.search(r"PORTFOLIO_W:([0-9.eE+,\-]*)", lines[j])
                if km and km.group(1):
                    k_indices = [int(x) for x in km.group(1).split(",")]
                if wm and wm.group(1):
                    weights = [float(x) for x in wm.group(1).split(",")]
            if not k_indices:
                results.append(None)  # infeasible sentinel
            else:
                results.append((raw_cvar / h, k_indices, weights))
            i += 3
        else:
            i += 1
    return results


def run_frontier(exe_file, csv_file, tickers, initial_prices, mu_values,
                 cardinality, population, method_label, log_dir):
    Q = len(tickers)
    initial_prices_str = ",".join(map(str, initial_prices))

    # Pass ALL mu values in one call so the C++ engine warm-starts composition between levels.
    mu_str = ",".join(f"{mu:.10f}" for mu in mu_values)
    command = [exe_file, csv_file, str(Q), str(cardinality), str(population), mu_str, initial_prices_str]

    print(f"\n=== Generating Efficient Frontier [{method_label}] — {len(mu_values)} mu points ===")
    try:
        proc = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nC++ Engine exited with error code: {e.returncode}")
        if e.stdout:
            print("--- C++ Engine stdout ---\n" + e.stdout)
        return []

    raw_results = _parse_frontier_stdout(proc.stdout)
    while len(raw_results) < len(mu_values):
        raw_results.append(None)

    # ---- Phase 1: decode raw PBIL results ----------------------------------------
    # points[i] = (mu, cvar, asset_weights) or (mu, None, None) for infeasible
    points = []
    for mu, result in zip(mu_values, raw_results):
        if result is None:
            points.append((mu, None, None))
            continue
        cvar_norm, best_k, best_w = result
        asset_weights = []
        for idx, shares in zip(best_k, best_w):
            ticker = tickers[idx]
            price = initial_prices[idx]
            weight = (shares * price) / 100000.0
            asset_weights.append((ticker, idx, shares, price, weight))
        points.append((mu, cvar_norm, asset_weights))

    # ---- Phase 2: monotonicity enforcement ----------------------------------------
    # CVaR must be non-decreasing with mu. If PBIL found a lower CVaR at a higher
    # mu, that portfolio also satisfies any lower mu return constraint — propagate back.
    feasible_idx = [i for i, (_, cvar, _) in enumerate(points) if cvar is not None]
    corrections = 0
    if len(feasible_idx) > 1:
        running_min_cvar    = points[feasible_idx[-1]][1]
        running_min_weights = points[feasible_idx[-1]][2]
        for i in reversed(feasible_idx[:-1]):
            mu_i, cvar_i, weights_i = points[i]
            if cvar_i > running_min_cvar:
                points[i] = (mu_i, running_min_cvar, running_min_weights)
                corrections += 1
            else:
                running_min_cvar    = cvar_i
                running_min_weights = weights_i
    if corrections:
        print(f"[{method_label}] Monotonicity: corrected {corrections} point(s)")

    # ---- Phase 3: print, log, and build return value ------------------------------
    frontier = []
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"=== Efficient Frontier Log: {method_label} ===")
    log_lines.append(f"Generated: {timestamp}")
    log_lines.append(f"Tickers ({Q}): {', '.join(tickers)}")
    log_lines.append(f"Cardinality K={cardinality} | Population={population}")
    log_lines.append(f"Mu range: {mu_values[0]:.4f} – {mu_values[-1]:.4f} ({len(mu_values)} points)")
    log_lines.append(f"Scenario CSV: {csv_file}")
    if corrections:
        log_lines.append(f"Monotonicity corrections applied: {corrections} point(s)")
    log_lines.append("")

    for mu, cvar, asset_weights in points:
        if cvar is None:
            print(f"-> mu={mu:.4f}: INFEASIBLE")
            log_lines.append(f"mu={mu:.6f} | INFEASIBLE")
            log_lines.append("")
            continue

        frontier.append((mu, cvar, asset_weights))

        print(f"-> mu={mu:.4f}: Normalized CVaR = {cvar:.6f}")
        print("   Portfolio: " + ", ".join([f"{t}: {w*100:.2f}%" for t, _, _, _, w in asset_weights]))
        print()

        log_lines.append(f"--- mu={mu:.6f} ---")
        log_lines.append(f"  Normalized CVaR : {cvar:.8f}")
        log_lines.append(f"  {'Ticker':<6} {'AssetIdx':>8} {'Shares':>14} {'Price':>12} {'Weight':>10}")
        log_lines.append(f"  {'-'*6} {'-'*8} {'-'*14} {'-'*12} {'-'*10}")
        for ticker, idx, shares, price, weight in asset_weights:
            log_lines.append(
                f"  {ticker:<6} {idx:>8} {shares:>14.6f} {price:>12.4f} {weight*100:>9.4f}%"
            )
        log_lines.append("")

    log_lines.append("=== Summary Table ===")
    log_lines.append(f"{'Target Return':>14} | {'Normalized CVaR':>16}")
    log_lines.append(f"{'-'*14}-+-{'-'*16}")
    for mu, cvar, _ in frontier:
        log_lines.append(f"{mu:>14.6f} | {cvar:>16.8f}")
    log_lines.append("")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{method_label}_frontier.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[{method_label}] Log saved to: {log_path}")

    return frontier


def main():
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
    initial_prices = prices.iloc[-1].to_numpy()  # anchor to current prices so share counts are realistic
    print(initial_prices)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_file = optimizer_exe_path(current_dir)
    log_dir = os.path.join(current_dir, "logs")

    K = 5
    POPULATION = 200

    # recommended by paper Sec 3.3), then reduce via two-stage k-means.
    # Uncomment to regenerate; the CSV is checked in so you can skip this.
    copula_csv = os.path.join(current_dir, "data", "scenarios", "copula_tree.csv")
    # from src.scenario_tree import ScenarioTree
    # print("Building copula scenario tree...")
    # tree = ScenarioTree(
    #     initial_prices=initial_prices,
    #     historical_returns=returns.to_numpy(),
    #     seed=42,
    # )
    # tree.build_tree(
    #     num_recourse=400,
    #     num_evaluate=40,
    #     correct_moments_and_correlation=True,
    #     evaluate_method="uniform",
    # )
    # tree.reduce(num_recourse=150, num_evaluate=20, seed=42)
    # os.makedirs(os.path.dirname(copula_csv), exist_ok=True)
    # tree.export_tree_to_csv(copula_csv)

    vine_csv = os.path.join(current_dir, "data", "scenarios", "backtest", "week_000.csv")
    
    from src.vine_scenario_tree import VineScenarioTree
    print("Building vine copula scenario tree...")
    vtree = VineScenarioTree(initial_prices=initial_prices,
                            historical_returns=returns.to_numpy(), seed=42)
    vtree.fit()
    vtree.build_tree(num_recourse=150, num_evaluate=20,
                    n_sim_stage1=2000, n_sim_stage2=100, beta=0.45)
    vtree.export_tree_to_csv(vine_csv)

    scenario_files = {
        #"copula_tree": copula_csv,
        "tcvae_vine":  os.path.join(current_dir, "data", "scenarios", "backtest", "tcvae_week_000.csv"),
        "vine_copula": vine_csv,
    }

    frontiers = {}

    for method_label, csv_file in scenario_files.items():
        print(f"\n{'='*60}")    
        print(f" Method: {method_label}  |  CSV: {csv_file}")
        print(f"{'='*60}")

        mu_values = get_mu_range(csv_file, initial_prices, tickers, K, 15)
        print(f"Tree-specific mu values:\n{mu_values}\n")

        frontier = run_frontier(
            exe_file=exe_file,
            csv_file=csv_file,
            tickers=tickers,
            initial_prices=initial_prices,
            mu_values=mu_values,
            cardinality=K,
            population=POPULATION,
            method_label=method_label,
            log_dir=log_dir,
        )
        frontiers[method_label] = frontier

    # ---- Side-by-side comparison summary --------------------------------------
    print("\n=== Final Efficient Frontier Comparison ===")
    for method_label, frontier in frontiers.items():
        print(f"\n--- {method_label} ---")
        for mu, cvar, asset_weights in frontier:
            print(f"Target Return: {mu:.4f} | Normalized CVaR: {cvar:.6f}")
            for ticker, idx, shares, price, weight in asset_weights:
                if weight > 1e-4:
                    print(f"  {ticker}: {weight*100:.2f}%")
            print()


if __name__ == "__main__":
    main()
