"""
dynamic_backtest.py — rolling weekly backtest, runs both scenario methods in one pass.

Two methods are compared back-to-back against an equal-weight benchmark:

  vine   — GARCH + R-vine copula refit every week on a rolling window
  tcvae  — pre-built TC-VAE trees loaded from TCVAE_DIR, refreshed every
            TCVAE_RETRAIN_EVERY weeks

Each week both methods optimise for a single TARGET_MU (no frontier sweep).

Outputs
-------
  data/scenarios/backtest/week_NNN.csv    — vine scenario tree per week
  logs/backtest/vine_results.csv          — vine per-week records
  logs/backtest/tcvae_results.csv         — TC-VAE per-week records
  logs/backtest/summary.txt               — side-by-side comparison
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vine_scenario_tree import VineScenarioTree
from src.paths import optimizer_exe_path
from main import _parse_frontier_stdout

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL     = 100_000.0              # Starting portfolio value ($)
K                   = 5                      # Max assets in portfolio (cardinality)
POPULATION          = 100                    # PBIL population size
TARGET_MU           = 0.0025                # Target weekly return for the optimizer

# Vine copula scenario tree parameters
NUM_RECOURSE        = 120                    # Stage-1 nodes (K-means target)
NUM_EVALUATE        = 100                     # Stage-2 nodes per recourse node
N_SIM_STAGE1        = 5000                   # Simulation pool before K-means
N_SIM_STAGE2        = 200                    # Stage-2 paths per cluster
BETA                = 0.45                   # Tail threshold for LP moment matching

SEED                = 42                     # RNG seed for reproducibility

# TC-VAE: pre-built trees named tcvae_week_NNN.csv.  Absolute or relative to
# the project root.
TCVAE_DIR           = os.path.join("data", "scenarios", "backtest")

# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = sorted([
    "MMM", "AMZN", "AXP", "AMGN", "AAPL",
    "BA",  "CAT",  "CVX", "CSCO", "KO",
    "DIS", "GS",   "HD",  "HON",  "IBM",
    "JNJ", "JPM",  "MCD", "MRK",  "MSFT",
    "NKE", "NVDA", "PG",  "CRM",  "SHW",
    "TRV", "UNH",  "VZ",  "V",    "WMT",
])
Q            = len(TICKERS)
TICKER_INDEX = {t: i for i, t in enumerate(TICKERS)}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_prices(data_dir: str, tickers: list, suffix: str = "") -> np.ndarray:
    """Load price arrays for all tickers, return shape (T, Q)."""
    arrays = []
    for t in tickers:
        arr = np.load(os.path.join(data_dir, f"{t}{suffix}.npy"))
        arrays.append(arr)
    min_len = min(len(a) for a in arrays)
    return np.column_stack([a[:min_len] for a in arrays])


def portfolio_next_value(asset_weights: list, next_prices: np.ndarray) -> float:
    """
    Calculate next portfolio value precisely tracking actual shares held.
    """
    return sum(
        opt_shares * next_prices[TICKER_INDEX[ticker]]
        for ticker, _, opt_shares, _, _ in asset_weights
    )


def performance_metrics(values: list) -> dict:
    v  = np.array(values, dtype=float)
    wr = np.diff(v) / v[:-1]
    peak = np.maximum.accumulate(v)
    
    std_dev = wr.std(ddof=1) if len(wr) > 1 else 0.0
    
    # Downside deviation for Sortino Ratio
    downside_returns = wr[wr < 0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 0.0
    
    # Realized CVaR 95%
    sorted_losses = np.sort(-wr)[::-1]
    tail_len = max(1, int(np.ceil(len(wr) * 0.05)))
    realized_cvar = sorted_losses[:tail_len].mean() if len(wr) > 0 else 0.0
    
    cagr = ((v[-1] / v[0]) ** (52.0 / len(wr)) - 1.0) * 100.0 if len(wr) > 0 else 0.0

    return {
        "total_return_pct": (v[-1] / v[0] - 1.0) * 100.0,
        "cagr_pct":         cagr,
        "ann_vol_pct":      std_dev * np.sqrt(52) * 100.0 if std_dev > 0 else float("nan"),
        "sharpe":           (wr.mean() / std_dev * np.sqrt(52)
                             if std_dev > 0 else float("nan")),
        "sortino":          (wr.mean() / downside_std * np.sqrt(52)
                             if downside_std > 0 else float("nan")),
        "realized_cvar_pct": realized_cvar * 100.0,
        "win_rate_pct":     (np.sum(wr > 0) / len(wr)) * 100.0 if len(wr) > 0 else 0.0,
        "max_drawdown_pct": ((v - peak) / peak).min() * 100.0,
    }


def optimize_for_mu(exe_file, csv_file, tickers, initial_prices, mu, cardinality, population,
                    current_cash=100_000.0, current_holdings=None):
    """
    Call the C++ optimizer once for a single target return mu.
    Returns (cvar_norm, asset_weights) or None if infeasible.
    """
    if current_holdings is None:
        current_holdings = [0.0] * len(tickers)

    command = [
        exe_file, csv_file, str(len(tickers)), str(cardinality), str(population),
        f"{mu:.10f}",
        ",".join(map(str, initial_prices)),
        f"{current_cash:.6f}",
        ",".join(map(str, current_holdings)),
    ]
    try:
        proc = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"    C++ engine error (code {e.returncode})")
        return None

    results = _parse_frontier_stdout(proc.stdout)
    if not results or results[0] is None:
        return None

    total_wealth = current_cash + sum(h * p for h, p in zip(current_holdings, initial_prices))
    
    cvar_norm, k_indices, weights = results[0]
    asset_weights = [
        (tickers[idx], idx, shares, initial_prices[idx],
         (shares * initial_prices[idx]) / total_wealth if total_wealth > 0 else 0.0)
        for idx, shares in zip(k_indices, weights)
    ]
    return cvar_norm, asset_weights


def _record_cash(records: list, week: int, capital: float, reason: str) -> None:
    records.append({
        "week": week, "capital_start": round(capital, 2),
        "capital_end": round(capital, 2), "return_pct": 0.0,
        "selected_cvar": float("nan"), "portfolio": "CASH",
        "scenario_csv": "", "status": reason,
    })


# ── core backtest loop ─────────────────────────────────────────────────────────

def run_backtest(mode: str, exe_file: str, scenario_dir: str, tcvae_dir: str,
                 train_prices: np.ndarray, bt_prices: np.ndarray, seed: int = SEED, run_id: int = 0):
    """
    Run the weekly backtest for one scenario mode ("vine" or "tcvae").
    Returns (portfolio_values, records).
    """
    n_weeks = bt_prices.shape[0] - 1
    tag     = f"[{mode.upper():<5}]"

    capital = INITIAL_CAPITAL
    values  = [capital]
    records = []
    
    current_cash = INITIAL_CAPITAL
    current_holdings = [0.0] * Q

    print(f"\n{'═'*62}")
    print(f"  {mode.upper()} BACKTEST  —  {n_weeks} weeks  |  mu={TARGET_MU}")
    print(f"{'═'*62}")

    for t in range(n_weeks):
        week_start     = datetime.now()
        current_prices = bt_prices[t]
        next_prices    = bt_prices[t + 1]

        print(f"{tag} Week {t:>3}/{n_weeks}  capital=${capital:>12,.2f}  "
              f"{week_start.strftime('%H:%M:%S')}")

        # ── Obtain scenario CSV ────────────────────────────────────────────────
        if mode == "tcvae":
            scenario_csv  = os.path.join(tcvae_dir, f"tcvae_week_{t:03d}.csv")
            anchor_prices = current_prices
            if not os.path.exists(scenario_csv):
                print(f"    ERROR: missing {os.path.basename(scenario_csv)}")
                _record_cash(records, t, capital, "tcvae_missing")
                values.append(capital)
                continue
            print(f"    tree: {os.path.basename(scenario_csv)}")
        else:  # vine
            scenario_csv  = os.path.join(scenario_dir, f"vine_run{run_id}_week_{t:03d}.csv")
            anchor_prices = current_prices
            if os.path.exists(scenario_csv):
                print(f"    tree: {os.path.basename(scenario_csv)} (loaded from cache)")
            else:
                # Rolling window: drop t+1 oldest training weeks, add t+1 bt weeks.
                # Window always has T_train price rows → T_train-1 return rows.
                window_prices  = np.vstack([train_prices[t + 1:], bt_prices[:t + 1]])
                simple_returns = window_prices[1:] / window_prices[:-1] - 1.0

                vtree = VineScenarioTree(
                    initial_prices=current_prices,
                    historical_returns=simple_returns,
                    seed=seed,
                )
                vtree.fit()
                vtree.build_tree(
                    num_recourse=NUM_RECOURSE, num_evaluate=NUM_EVALUATE,
                    n_sim_stage1=N_SIM_STAGE1, n_sim_stage2=N_SIM_STAGE2,
                    beta=BETA,
                )
                vtree.export_tree_to_csv(scenario_csv)

        # ── Optimise for TARGET_MU ─────────────────────────────────────────────
        result = optimize_for_mu(exe_file, scenario_csv, TICKERS, anchor_prices,
                                 TARGET_MU, K, POPULATION, current_cash, current_holdings)

        if result is None:
            print(f"    infeasible at mu={TARGET_MU} — holding previous portfolio")
            next_capital = current_cash + sum(shares * next_prices[q] for q, shares in enumerate(current_holdings))
            weekly_ret = (next_capital / capital - 1.0) * 100.0
            records.append({
                "week":          t,
                "capital_start": round(capital, 2),
                "capital_end":   round(next_capital, 2),
                "return_pct":    round(weekly_ret, 4),
                "selected_cvar": float("nan"),
                "portfolio":     "HELD PREVIOUS",
                "scenario_csv":  scenario_csv,
                "status":        "infeasible",
            })
            capital = next_capital
            values.append(capital)
            continue

        sel_cvar, asset_weights = result

        # ── Mark to market ─────────────────────────────────────────────────────
        next_capital = portfolio_next_value(asset_weights, next_prices)
        
        # Extract new holdings configuration (assuming essentially fully invested)
        current_cash = 0.0
        current_holdings = [0.0] * Q
        for _, idx, shares, _, _ in asset_weights:
            current_holdings[idx] = shares
            
        weekly_ret   = (next_capital / capital - 1.0) * 100.0
        portfolio_str = " | ".join(
            f"{tk}:{w*100:.1f}%" for tk, _, _, _, w in asset_weights
        )
        elapsed = (datetime.now() - week_start).total_seconds()
        print(f"    CVaR={sel_cvar:.6f}  ret={weekly_ret:+.2f}%  "
              f"→ ${next_capital:,.2f}  [{elapsed:.1f}s]")
        print(f"    {portfolio_str}")

        records.append({
            "week":          t,
            "capital_start": round(capital, 2),
            "capital_end":   round(next_capital, 2),
            "return_pct":    round(weekly_ret, 4),
            "selected_cvar": round(sel_cvar, 8),
            "portfolio":     portfolio_str,
            "scenario_csv":  scenario_csv,
            "status":        "ok",
        })
        capital = next_capital
        values.append(capital)

    return values, records


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir     = os.path.join(current_dir, "data", "raw")
    exe_file     = optimizer_exe_path(current_dir)
    scenario_dir = os.path.join(current_dir, "data", "scenarios", "backtest")
    tcvae_dir    = (TCVAE_DIR if os.path.isabs(TCVAE_DIR)
                    else os.path.join(current_dir, TCVAE_DIR))
    log_dir      = os.path.join(current_dir, "logs", "backtest", "tc-vae")
    os.makedirs(scenario_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(exe_file):
        print(f"ERROR: optimizer not found at {exe_file}")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading price data...")
    train_prices = load_prices(data_dir, TICKERS, suffix="")
    bt_prices    = load_prices(data_dir, TICKERS, suffix="_bt")

    T_train = train_prices.shape[0]
    n_bt    = bt_prices.shape[0]
    n_weeks = n_bt - 1

    print(f"  Training : {T_train} wks  |  Backtest : {n_bt} wks  "
          f"({n_weeks} rebalancing periods)")
    print(f"  K={K}  Pop={POPULATION}  mu={TARGET_MU}  seed={SEED}")
    print(f"  Vine: J={NUM_RECOURSE} E={NUM_EVALUATE} "
          f"n_sim1={N_SIM_STAGE1} n_sim2={N_SIM_STAGE2} beta={BETA}")
    print(f"  TC-VAE dir: {tcvae_dir}")

    # ── Run both methods sequentially ─────────────────────────────────────────
    wall_start = datetime.now()

    tcvae_values_list = []
    tcvae_metrics_list = []
    
    for run_id in range(10):
        print(f"\n{'='*70}")
        print(f" RUNNING TC-VAE BACKTEST {run_id}/9")
        print(f"{'='*70}")

        current_tcvae_dir = os.path.join(tcvae_dir, str(run_id))

        tcvae_values, tcvae_records = run_backtest(
            "tcvae", exe_file, scenario_dir, current_tcvae_dir, train_prices, bt_prices, seed=SEED + run_id, run_id=run_id
        )
        
        df_tcvae = pd.DataFrame(tcvae_records)
        df_tcvae["portfolio_value"] = tcvae_values[1:]
        path_tcvae = os.path.join(log_dir, f"tcvae_results_{run_id}.csv")
        df_tcvae.to_csv(path_tcvae, index=False)
        print(f"TCVAE run {run_id} results → {path_tcvae}")

        tcvae_values_list.append(tcvae_values)
        tcvae_metrics_list.append(performance_metrics(tcvae_values))

    elapsed = (datetime.now() - wall_start).total_seconds()

    # ── Individual Summaries ───────────────────────────────────────────────────
    w = 13
    for run_id in range(10):
        t_vals = tcvae_values_list[run_id]
        t_m = tcvae_metrics_list[run_id]
        
        indiv_lines = [
            "=" * 70,
            f"  DYNAMIC BACKTEST SUMMARY - TC-VAE RUN {run_id}",
            f"  {n_weeks} weeks | mu={TARGET_MU} | K={K}",
            "=" * 70,
            f"  {'Metric':<26}  {'TC-VAE':>{w}}",
            f"  {'─'*26}  {'─'*w}",
            f"  {'Final capital ($)':<26}  {t_vals[-1]:>{w},.2f}",
            f"  {'Total return (%)':<26}  {t_m['total_return_pct']:>+{w}.2f}",
            f"  {'CAGR (%)':<26}  {t_m['cagr_pct']:>+{w}.2f}",
            f"  {'Ann. volatility (%)':<26}  {t_m['ann_vol_pct']:>{w}.2f}",
            f"  {'Sharpe ratio':<26}  {t_m['sharpe']:>{w}.3f}",
            f"  {'Sortino ratio':<26}  {t_m['sortino']:>{w}.3f}",
            f"  {'Max drawdown (%)':<26}  {t_m['max_drawdown_pct']:>{w}.2f}",
            f"  {'Realized CVaR 95% (wk %)':<26}  {t_m['realized_cvar_pct']:>{w}.2f}",
            f"  {'Win rate (%)':<26}  {t_m['win_rate_pct']:>{w}.1f}",
            "=" * 70,
        ]
        indiv_text = "\n".join(indiv_lines)
        indiv_path = os.path.join(log_dir, f"summary_{run_id}.txt")
        with open(indiv_path, "w", encoding="utf-8") as f:
            f.write(indiv_text + "\n")
    
    # ── Combined summary ───────────────────────────────────────────────────────
    avg_tcvae_m = {k: np.nanmean([m[k] for m in tcvae_metrics_list]) for k in tcvae_metrics_list[0]}
    avg_tcvae_final_capital = np.mean([v[-1] for v in tcvae_values_list])

    summary_lines = [
        "=" * 70,
        "  DYNAMIC BACKTEST SUMMARY (AVERAGE OF 10 RUNS - TC-VAE)",
        f"  {n_weeks} weeks | mu={TARGET_MU} | K={K} | elapsed={elapsed/60:.1f} min",
        "=" * 70,
        f"  {'Metric':<26}  {'TC-VAE (Avg)':>{w}}",
        f"  {'─'*26}  {'─'*w}",
        f"  {'Final capital ($)':<26}  {avg_tcvae_final_capital:>{w},.2f}",
        f"  {'Total return (%)':<26}  {avg_tcvae_m['total_return_pct']:>+{w}.2f}",
        f"  {'CAGR (%)':<26}  {avg_tcvae_m['cagr_pct']:>+{w}.2f}",
        f"  {'Ann. volatility (%)':<26}  {avg_tcvae_m['ann_vol_pct']:>{w}.2f}",
        f"  {'Sharpe ratio':<26}  {avg_tcvae_m['sharpe']:>{w}.3f}",
        f"  {'Sortino ratio':<26}  {avg_tcvae_m['sortino']:>{w}.3f}",
        f"  {'Max drawdown (%)':<26}  {avg_tcvae_m['max_drawdown_pct']:>{w}.2f}",
        f"  {'Realized CVaR 95% (wk %)':<26}  {avg_tcvae_m['realized_cvar_pct']:>{w}.2f}",
        f"  {'Win rate (%)':<26}  {avg_tcvae_m['win_rate_pct']:>{w}.1f}",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}\n")

    summary_path = os.path.join(log_dir, "summary_average.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Summary → {summary_path}")
    print(f"Individual summaries saved as summary_0.txt to summary_9.txt")


if __name__ == "__main__":
    main()
