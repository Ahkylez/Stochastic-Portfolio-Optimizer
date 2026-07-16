"""
Stability test suite — vine_copula vs TC-VAE scenario generators.
Kaut & Wallace (2007) in-sample and out-of-sample stability framework.

Produces:
  logs/stability/stability_report_<timestamp>.log
  logs/stability/frontiers/{method}_s{seed}_frontier.log  (one per tree)

Four sections in the report:
  1. In-sample stability  — 10 seeds x 2 methods, C++ optimizer, CVaR tables
  2. Cross-evaluation     — portfolio from tree A, CVaR on tree B (Python LP)
  3. Backtest             — 2025 weekly prices, tree-0 portfolios from each method
  4. Full frontier tables — every mu/CVaR point for every tree, all seeds

Usage:
    python stability_test.py [--fast] [--skip-vine-gen] [--method METHOD] [--mu-idx N]

Flags:
    --fast           pop=50, 5 mu points, small vine trees (quick smoke-test)
    --skip-vine-gen  assume vine_copula_tree{0..9}.csv already exist
    --method METHOD  run only one method: "tcvae" or "vine_copula" (default: both)
    --mu-idx N       frontier index for cross-eval / backtest (default: 7)

The script auto-imports any existing *_frontier.log files in logs/ into the
JSON cache before running, so previously completed frontier runs are never
repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from src.paths import optimizer_exe_path

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from main import run_frontier, get_tree_expected_returns, get_mu_range

TICKERS = sorted([
    "MMM", "AMZN", "AXP", "AMGN", "AAPL",
    "BA",  "CAT",  "CVX", "CSCO", "KO",
    "DIS", "GS",   "HD",  "HON",  "IBM",
    "JNJ", "JPM",  "MCD", "MRK",  "MSFT",
    "NKE", "NVDA", "PG",  "CRM",  "SHW",
    "TRV", "UNH",  "VZ",  "V",    "WMT",
])

ROOT       = os.path.dirname(os.path.abspath(__file__))
EXE        = optimizer_exe_path(ROOT)
SCENARIO_DIR = os.path.join(ROOT, "data", "scenarios")
RAW_DIR    = os.path.join(ROOT, "data", "raw")
LOG_DIR    = os.path.join(ROOT, "logs", "stability")
CACHE_FILE   = os.path.join(LOG_DIR, "_frontier_cache.json")
# Directories scanned when auto-importing existing frontier logs
IMPORT_DIRS  = [os.path.join(ROOT, "logs")]

H         = 100_000.0
BETA_CVAR = 0.95          # CVaR confidence level in the optimizer
N_SEEDS   = 10            # number of trees per method


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_prices() -> tuple[np.ndarray, np.ndarray]:
    prices = pd.DataFrame({t: np.load(os.path.join(RAW_DIR, f"{t}.npy")) for t in TICKERS})
    returns = (prices / prices.shift(1) - 1.0).dropna()
    return prices.iloc[-1].to_numpy(), returns.to_numpy()


def _vine_csv(seed: int) -> str:
    return os.path.join(SCENARIO_DIR, f"vine_copula_tree{seed}.csv")


def _tcvae_csv(seed: int) -> str:
    return os.path.join(SCENARIO_DIR, f"tcvae_tree{seed}.csv")

def _tcvae_vine_csv(seed: int) -> str:
    return os.path.join(SCENARIO_DIR, f"tcvae_vine_tree{seed}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Tree generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_vine_tree(seed: int, initial_prices: np.ndarray,
                       historical_returns: np.ndarray, fast: bool) -> str:
    """Generate and save vine_copula_tree{seed}.csv if it doesn't exist."""
    csv_path = _vine_csv(seed)
    if os.path.exists(csv_path):
        print(f"  [vine seed={seed}] exists, skipping generation")
        return csv_path

    from src.vine_scenario_tree import VineScenarioTree
    print(f"  [vine seed={seed}] generating tree…")
    t0 = time.time()

    tree = VineScenarioTree(initial_prices, historical_returns, seed=seed)
    tree.fit()
    if fast:
        tree.build_tree(num_recourse=30,  num_evaluate=10,
                        n_sim_stage1=200, n_sim_stage2=30, beta=0.45)
    else:
        tree.build_tree(num_recourse=400, num_evaluate=30,
                        n_sim_stage1=2000, n_sim_stage2=100, beta=0.45)

    os.makedirs(SCENARIO_DIR, exist_ok=True)
    tree.export_tree_to_csv(csv_path)
    print(f"  [vine seed={seed}] done in {time.time()-t0:.1f}s -> {csv_path}")
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# 3.  In-sample frontier runner (with JSON caching for resumability)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_load() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _cache_save(cache: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def import_logs_to_cache(cache: dict) -> None:
    """Scan log directories for .log files, parse them, and seed the cache."""
    import glob
    import re

    log_dirs = [LOG_DIR, os.path.join(LOG_DIR, "frontiers"), os.path.join(ROOT, "logs")]
    log_files = []
    for d in log_dirs:
        if os.path.exists(d):
            log_files.extend(glob.glob(os.path.join(d, "*.log")))

    # Remove duplicates
    log_files = list(set(os.path.abspath(f) for f in log_files))

    imported_count = 0
    for log_path in log_files:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except IOError:
            continue

        csv_path = None
        for line in lines:
            if line.startswith("Scenario CSV:"):
                csv_path = line.split(":", 1)[1].strip()
                break

        if not csv_path:
            continue

        method = None
        seed = None

        m_tcvae = re.search(r"tcvae_tree(\d+)\.csv", csv_path)
        m_vine = re.search(r"vine_copula_tree(\d+)\.csv", csv_path)
        m_tcvae_vine = re.search(r"tcvae_vine_tree(\d+)\.csv", csv_path)

        if m_tcvae_vine:
            method = "tcvae_vine"
            seed = int(m_tcvae_vine.group(1))
        elif m_tcvae:
            method = "tcvae"
            seed = int(m_tcvae.group(1))
        elif m_vine:
            method = "vine_copula"
            seed = int(m_vine.group(1))
        else:
            continue

        key = f"{method}_{seed}"
        frontier = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            m_inf = re.match(r"mu=([0-9.eE+\-]+)\s*\|\s*INFEASIBLE", line)
            if m_inf:
                frontier.append([float(m_inf.group(1)), None, None])
                i += 1
                continue

            m_mu = re.match(r"---\s*mu=([0-9.eE+\-]+)\s*---", line)
            if m_mu:
                mu_val = float(m_mu.group(1))
                cvar_val = None
                asset_weights = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("--- mu="):
                    inner_line = lines[i].strip()
                    if inner_line.startswith("Normalized CVaR"):
                        try:
                            cvar_val = float(inner_line.split(":")[1].strip())
                        except ValueError:
                            pass
                    elif inner_line.startswith("=== Summary Table ==="):
                        break
                    else:
                        parts = inner_line.split()
                        if len(parts) == 5 and parts[1].isdigit():
                            try:
                                asset_weights.append([
                                    parts[0], int(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4].rstrip("%")) / 100.0
                                ])
                            except ValueError:
                                pass
                    i += 1

                if cvar_val is not None and asset_weights:
                    frontier.append([mu_val, cvar_val, asset_weights])
                continue
            i += 1

        if frontier:
            if key not in cache or len(frontier) >= len(cache[key]):
                cache[key] = frontier
                imported_count += 1

    if imported_count > 0:
        _cache_save(cache)
        print(f"  Imported {imported_count} log file(s) into cache.")

def run_in_sample(
    method: str, seed: int, csv_path: str,
    initial_prices: np.ndarray, K: int, pop: int, n_mu: int,
    cache: dict,
) -> Optional[list]:
    """
    Run C++ optimizer on csv_path.  Returns frontier list or None on failure.
    Results are cached to avoid re-running if the script is interrupted.
    """
    key = f"{method}_{seed}"
    if key in cache:
        print(f"  [{method} seed={seed}] cached")
        raw = cache[key]
        # Deserialize: [mu, cvar, [[ticker,idx,shares,price,weight], ...]]
        return [(r[0], r[1], [tuple(aw) for aw in r[2]] if r[2] is not None else None) for r in raw]

    if not os.path.exists(csv_path):
        print(f"  [{method} seed={seed}] CSV not found: {csv_path}")
        return None

    mu_values = np.linspace(0.004, 0.023, n_mu)
    label     = f"{method}_s{seed}"
    frontier  = run_frontier(
        exe_file=EXE, csv_file=csv_path, tickers=TICKERS,
        initial_prices=initial_prices, mu_values=mu_values,
        cardinality=K, population=pop, method_label=label,
        log_dir=os.path.join(LOG_DIR, "frontiers"),
    )

    if frontier:
        # Serialize for JSON storage
        serializable = [
            [mu, cvar, [[aw[0], aw[1], aw[2], aw[3], aw[4]] for aw in aws] if aws is not None else None]
            for mu, cvar, aws in frontier
        ]
        cache[key] = serializable
        _cache_save(cache)

    return frontier


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Python CVaR evaluator (hold-and-evaluate, no stage-2 rebalancing)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cvar_python(
    csv_path: str,
    k_indices: list[int],
    w_shares: list[float],
    initial_prices: np.ndarray,
    beta: float = BETA_CVAR,
    h: float = H,
) -> Optional[float]:
    """
    Evaluate CVaR(beta) of a fixed portfolio on a scenario tree CSV.
    Mirrors the C++ objective (shortfall/alpha formulation) but without
    stage-2 rebalancing — the portfolio is held through to the evaluate nodes.

    Normalises by h so the result is directly comparable to C++ output.
    Returns None if the CSV is missing.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, header=None)
    Q_total   = (df.shape[1] - 3) // 2
    p_j       = df.iloc[:, 1].to_numpy()
    p_je      = df.iloc[:, 2].to_numpy()
    eval_p    = df.iloc[:, 3 + Q_total : 3 + 2 * Q_total].to_numpy()

    probs  = p_j * p_je                         # joint leaf probabilities
    k_arr  = np.array(k_indices, dtype=int)
    w_arr  = np.array(w_shares,  dtype=float)

    V0     = float(np.dot(w_arr, initial_prices[k_arr]))   # initial value
    V_je   = eval_p[:, k_arr] @ w_arr                      # (N,) leaf values

    # Normalized loss (consistent with C++ dividing by h)
    loss   = (V0 - V_je) / h

    # Rockafellar-Uryasev CVaR: sort losses descending, weight the tail
    idx    = np.argsort(loss)[::-1]
    L, P   = loss[idx], probs[idx]
    tail   = 1.0 - beta
    cum    = 0.0
    cvar   = 0.0
    for l, p in zip(L, P):
        if cum >= tail:
            break
        w = min(p, tail - cum)
        cvar += w * l
        cum  += w
    return float(cvar / tail)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Backtest
# ══════════════════════════════════════════════════════════════════════════════

def backtest_portfolio(
    label: str,
    k_indices: list[int],
    w_shares: list[float],
    bt_prices: np.ndarray,      # (T_bt, Q_total) weekly backtest prices
    initial_prices: np.ndarray, # (Q_total,) prices at training end
    h: float = H,
    beta: float = BETA_CVAR,
) -> dict:
    """
    Track a fixed portfolio through backtest weekly prices.
    Returns a dict of performance metrics.
    """
    k = np.array(k_indices, dtype=int)
    w = np.array(w_shares,  dtype=float)

    V0    = float(np.dot(w, initial_prices[k]))
    V_bt  = bt_prices[:, k] @ w    # (T_bt,) portfolio value each week

    weekly_ret = np.diff(V_bt) / V_bt[:-1]    # (T_bt-1,)
    total_ret  = (V_bt[-1] - V0) / V0
    T          = len(weekly_ret)
    ann_ret    = (1.0 + total_ret) ** (52.0 / T) - 1.0
    ann_std    = weekly_ret.std(ddof=1) * np.sqrt(52.0)
    sharpe     = ann_ret / ann_std if ann_std > 1e-12 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(V_bt)
    drawdowns   = (V_bt - running_max) / running_max
    max_dd      = float(drawdowns.min())

    # Realized CVaR of weekly losses
    losses        = -weekly_ret
    sorted_losses = np.sort(losses)[::-1]
    p_uniform     = 1.0 / T
    tail          = 1.0 - beta
    n_tail        = max(1, int(np.ceil(T * tail)))
    real_cvar     = float(sorted_losses[:n_tail].mean())

    return {
        "label":        label,
        "initial_value": V0,
        "final_value":  float(V_bt[-1]),
        "total_ret":    total_ret,
        "ann_ret":      ann_ret,
        "ann_std":      ann_std,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "realized_cvar_weekly": real_cvar,
        "n_weeks":      T,
        "weekly_values": V_bt.tolist(),
    }


def load_backtest_prices() -> Optional[np.ndarray]:
    """Load backtest price matrix (T_bt, Q). Returns None if files are missing."""
    arrays = []
    for t in TICKERS:
        path = os.path.join(RAW_DIR, f"{t}_bt.npy")
        if not os.path.exists(path):
            print(f"  Backtest data missing for {t}. Run: python data/fetch_backtest_data.py")
            return None
        arrays.append(np.load(path))

    min_len = min(len(a) for a in arrays)
    return np.column_stack([a[:min_len] for a in arrays])    # (T_bt, Q)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hr(char="═", width=78) -> str:
    return char * width


def _table_header(cols: list[str], widths: list[int]) -> str:
    line = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    sep  = "  ".join("-" * w for w in widths)
    return line + "\n  " + sep


def _fmt_cvar(v: Optional[float]) -> str:
    return f"{v:.6f}" if v is not None else "  N/A   "


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_report(
    in_sample:   dict,          # {method: {seed: frontier}}
    cross_eval:  dict,          # {(src_method, src_seed): {(eval_method, eval_seed): cvar}}
    backtest:    list[dict],    # list of backtest result dicts
    mu_idx:      int,
    args,
) -> list[str]:

    lines = []
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        _hr("═"), "",
        "  STABILITY TEST REPORT",
        f"  Generated   : {ts}",
        f"  Mode        : {'FAST' if args.fast else 'FULL'}",
        f"  Seeds       : 0 – {N_SEEDS-1}  (10 trees per method)",
        f"  Cross-eval μ: frontier index {mu_idx} (0-based)",
        "", _hr("═"),
    ]

    # ── Section 1: In-sample stability ──────────────────────────────────────
    lines += ["", _hr("─"), "  SECTION 1 — IN-SAMPLE CVaR STABILITY", _hr("─"), ""]
    lines.append(
        "  Each cell: normalized CVaR reported by the C++ optimizer for that\n"
        "  tree at the matching frontier point (μ rank 3, 6, 9 in a 15-point\n"
        "  sweep — low, mid, high return targets).\n"
    )

    for method in ("vine_copula", "tcvae", "tcvae_vine"):
        frontiers = in_sample.get(method, {})
        if not frontiers:
            lines.append(f"  [{method}] No results available.\n")
            continue

        # Pick 3 representative ranks at 25%, 50%, 75% of actual frontier length
        sample_len = next(
            (len(fr) for fr in frontiers.values() if fr),
            15,
        )
        r25 = max(0, sample_len // 4)
        r50 = max(0, sample_len // 2)
        r75 = max(0, min(3 * sample_len // 4, sample_len - 1))
        rank_idxs   = sorted(set([r25, r50, r75]))
        rank_labels = [f"rank_{r}" for r in rank_idxs]
        col_w = 10

        lines.append(f"  Method: {method}")
        header_cols = ["seed"] + rank_labels + ["mu_3", "mu_6", "mu_9"]
        header_ws   = [6] + [col_w]*3 + [10]*3
        lines.append("  " + _table_header(header_cols, header_ws))

        cvar_by_rank: dict[int, list[float]] = {r: [] for r in rank_idxs}

        for seed in range(N_SEEDS):
            fr = frontiers.get(str(seed)) or frontiers.get(seed)
            if not fr:
                lines.append(f"  {seed:>6}  {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}}"
                              + "  " + "  ".join(["       N/A"]*3))
                continue

            row_cvars = []
            row_mus   = []
            for ri in rank_idxs:
                if ri < len(fr):
                    row_cvars.append(fr[ri][1])
                    row_mus.append(fr[ri][0])
                    cvar_by_rank[ri].append(fr[ri][1])
                else:
                    row_cvars.append(None)
                    row_mus.append(None)

            cvar_cells = "  ".join(_fmt_cvar(c).rjust(col_w) for c in row_cvars)
            mu_cells   = "  ".join(
                (f"{m:.6f}" if m is not None else "  N/A   ").rjust(10)
                for m in row_mus
            )
            lines.append(f"  {seed:>6}  {cvar_cells}  {mu_cells}")

        # Summary rows
        lines.append("  " + "  ".join(["-"*6] + ["-"*col_w]*3 + ["-"*10]*3))
        mean_row = "  mean  "
        std_row  = "  std   "
        cv_row   = "  CV%   "
        for ri in rank_idxs:
            vals = cvar_by_rank[ri]
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                cv = 100 * s / m if m else 0.0
                mean_row += f"  {m:>{col_w}.6f}"
                std_row  += f"  {s:>{col_w}.6f}"
                cv_row   += f"  {cv:>{col_w}.2f}%"
            else:
                mean_row += f"  {'N/A':>{col_w}}"
                std_row  += f"  {'N/A':>{col_w}}"
                cv_row   += f"  {'N/A':>{col_w}}"
        lines += [mean_row, std_row, cv_row, ""]

    # ── Section 2: Cross-evaluation ──────────────────────────────────────────
    lines += ["", _hr("─"), "  SECTION 2 — CROSS-EVALUATION (OOS CVaR)", _hr("─"), ""]
    lines.append(
        "  Each row: a portfolio taken from the optimizer's in-sample result at\n"
        f"  frontier index {mu_idx}.  Each column: the tree on which that portfolio\n"
        "  is evaluated (hold-and-evaluate, no stage-2 rebalancing).\n"
        "  Diagonal = in-sample; off-diagonal = out-of-sample.\n"
    )

    if cross_eval:
        # Determine all source/eval keys present
        all_methods = ["vine_copula", "tcvae", "tcvae_vine"]
        col_labels  = []
        for em in all_methods:
            for es in range(N_SEEDS):
                prefix = "t_vi" if em == "tcvae_vine" else em[:4]
                col_labels.append(f"{prefix}_{es}")

        lines.append("  " + "  ".join(["src".ljust(12)] + [f"{c:>9}" for c in col_labels]))
        lines.append("  " + "  ".join(["-"*12] + ["-"*9]*len(col_labels)))

        for src_method in all_methods:
            for src_seed in range(N_SEEDS):
                src_key  = f"{src_method}_{src_seed}"
                row_vals = cross_eval.get(src_key, {})
                row_str  = f"  {src_key:<12}"
                for em in all_methods:
                    for es in range(N_SEEDS):
                        eval_key = f"{em}_{es}"
                        v = row_vals.get(eval_key)
                        row_str += f"  {_fmt_cvar(v):>9}"
                lines.append(row_str)

        # Summary: mean OOS CVaR per method
        lines += [""]
        lines.append("  Summary — mean CVaR when portfolio evaluated on OWN vs OTHER method's trees:")
        lines.append("  " + "-"*60)
        lines.append(f"  {'Portfolio source':<22} {'In-method (own)':>18}  {'Cross-method':>18}")
        lines.append("  " + "-"*60)
        for src_method in all_methods:
            for src_seed in range(N_SEEDS):
                src_key  = f"{src_method}_{src_seed}"
                row_vals = cross_eval.get(src_key, {})
                own   = [v for k, v in row_vals.items() if k.rsplit('_', 1)[0] == src_method and v is not None]
                cross = [v for k, v in row_vals.items() if k.rsplit('_', 1)[0] != src_method and v is not None]
                own_m   = f"{np.mean(own):.6f}"   if own   else "N/A"
                cross_m = f"{np.mean(cross):.6f}" if cross else "N/A"
                lines.append(f"  {src_key:<22} {own_m:>18}  {cross_m:>18}")
        lines.append("")
    else:
        lines.append("  [No cross-evaluation results]\n")

    # ── Section 3: Backtest ───────────────────────────────────────────────────
    lines += ["", _hr("─"), "  SECTION 3 — BACKTEST (2025 Weekly Prices)", _hr("─"), ""]
    lines.append(
        "  Portfolios from tree_0 of each method, held through 2025.\n"
        "  No rebalancing; weekly prices from data/raw/{ticker}_bt.npy.\n"
    )

    if backtest:
        col_w_bt = 16
        bt_cols  = ["Portfolio", "Initial $", "Final $", "Total Ret",
                    "Ann. Ret", "Ann. Std", "Sharpe", "Max DD", "CVaR95% (wk)"]
        bt_ws    = [24, 10, 10, 10, 10, 10, 8, 10, 14]
        lines.append("  " + _table_header(bt_cols, bt_ws))

        for r in backtest:
            lines.append(
                f"  {r['label']:<24}"
                f"  {r['initial_value']:>10,.0f}"
                f"  {r['final_value']:>10,.0f}"
                f"  {r['total_ret']*100:>9.2f}%"
                f"  {r['ann_ret']*100:>9.2f}%"
                f"  {r['ann_std']*100:>9.2f}%"
                f"  {r['sharpe']:>8.3f}"
                f"  {r['max_drawdown']*100:>9.2f}%"
                f"  {r['realized_cvar_weekly']*100:>13.4f}%"
            )

        # Weekly value table for tree_0 portfolios
        lines += ["", "  Weekly Portfolio Values:"]
        if backtest:
            n_weeks = backtest[0]["n_weeks"]
            lines.append("  " + "  ".join(["Week".rjust(5)]
                                           + [r["label"][:20].rjust(20) for r in backtest]))
            vals_all = [r["weekly_values"] for r in backtest]
            for t in range(min(n_weeks, 60)):    # cap at 60 rows in log
                row = f"  {t+1:>5}"
                for vals in vals_all:
                    row += f"  {vals[t]:>20,.2f}"
                lines.append(row)
            if n_weeks > 60:
                lines.append(f"  … ({n_weeks-60} more weeks not shown)")
    else:
        lines.append("  [No backtest results — run python data/fetch_backtest_data.py first]\n")

    lines += ["", _hr("═"), "  END OF REPORT", _hr("═"), ""]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Allow unicode box-draw chars in console output on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",          action="store_true",
                        help="Quick smoke-test: small trees, pop=50, 5 mu pts")
    parser.add_argument("--skip-vine-gen", action="store_true",
                        help="Skip vine tree generation (CSVs must already exist)")
    parser.add_argument("--import-logs", action="store_true",
                        help="Import existing frontier logs into the cache before running")
    parser.add_argument("--mu-idx", type=int, default=7,
                        help="Frontier index used for cross-eval and backtest (default: 7)")
    parser.add_argument("--method", default="both",
                        choices=["tcvae", "vine_copula", "tcvae_vine", "both", "all"],
                        help="Run frontiers for only one method; other is loaded from cache (default: all)")
    args = parser.parse_args()

    K      = 5
    POP    = 50  if args.fast else 200
    N_MU   = 5   if args.fast else 15

    run_methods = (["vine_copula", "tcvae", "tcvae_vine"] if args.method in ("both", "all") else [args.method])

    print(_hr("═"))
    print(f"  Stability Test   |  fast={args.fast}  |  K={K}  pop={POP}  μ_pts={N_MU}  method={args.method}")
    print(_hr("═"))

    # ── Load training data ───────────────────────────────────────────────────
    initial_prices, hist_returns = _load_prices()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    cache = _cache_load()

    # Always auto-import existing frontier logs so prior runs are available as cache hits.
    import_logs_to_cache(cache)
    if args.import_logs:
        pass  # already imported above; flag kept for backwards compatibility

    # ── Generate / verify tree CSVs ──────────────────────────────────────────
    vine_csvs  = {}
    tcvae_csvs = {}
    tcvae_vine_csvs = {}

    for seed in range(N_SEEDS):
        # Vine trees
        if not args.skip_vine_gen:
            vine_csvs[seed] = generate_vine_tree(
                seed, initial_prices, hist_returns, fast=args.fast)
        else:
            p = _vine_csv(seed)
            vine_csvs[seed] = p if os.path.exists(p) else None

        # TC-VAE trees (pre-existing; user provides these)
        p = _tcvae_csv(seed)
        tcvae_csvs[seed] = p if os.path.exists(p) else None
        if not os.path.exists(p):
            print(f"  [tcvae seed={seed}] CSV not found: {p}  -- will be skipped")

        p = _tcvae_vine_csv(seed)
        tcvae_vine_csvs[seed] = p if os.path.exists(p) else None
        if not os.path.exists(p):
            print(f"  [tcvae_vine seed={seed}] CSV not found: {p}  -- will be skipped")

    # ── In-sample optimization ───────────────────────────────────────────────
    print("\n" + _hr("─"))
    print("  Running in-sample frontiers…")
    print(_hr("─"))

    in_sample: dict[str, dict] = {"vine_copula": {}, "tcvae": {}, "tcvae_vine": {}}

    for seed in range(N_SEEDS):
        for method, csvs in [("vine_copula", vine_csvs), ("tcvae", tcvae_csvs), ("tcvae_vine", tcvae_vine_csvs)]:
            if method not in run_methods:
                # Load from cache only — no C++ run
                key = f"{method}_{seed}"
                raw = cache.get(key, [])
                if raw:
                    in_sample[method][seed] = [
                        (r[0], r[1], [tuple(aw) for aw in r[2]] if r[2] else None)
                        for r in raw
                    ]
                    print(f"  [{method} seed={seed}] loaded {len(raw)} points from cache")
                else:
                    in_sample[method][seed] = None
                    print(f"  [{method} seed={seed}] not in cache — skipping")
                continue
            csv_path = csvs.get(seed)
            fr = run_in_sample(method, seed, csv_path or "",
                               initial_prices, K, POP, N_MU, cache)
            in_sample[method][seed] = fr   # may be None

    # ── Cross-evaluation ─────────────────────────────────────────────────────
    print("\n" + _hr("─"))
    print("  Running cross-evaluation (Python CVaR)…")
    print(_hr("─"))

    # Build list of source portfolios: (method, seed, k_indices, w_shares)
    source_portfolios = []
    mu_idx = min(args.mu_idx, N_MU - 1)

    for method, csvs in [("vine_copula", vine_csvs), ("tcvae", tcvae_csvs), ("tcvae_vine", tcvae_vine_csvs)]:
        for seed in range(N_SEEDS):
            fr = in_sample[method].get(seed)
            if not fr or mu_idx >= len(fr):
                continue
            _, _, aws = fr[mu_idx]
            if aws is None:
                continue
            k_idx = [aw[1] for aw in aws]
            w_shr = [aw[2] for aw in aws]
            source_portfolios.append((method, seed, k_idx, w_shr))

    # Evaluate each source portfolio on every available tree
    cross_eval: dict[str, dict[str, Optional[float]]] = {}

    total_evals = len(source_portfolios) * N_SEEDS * 2
    done = 0
    for (src_m, src_s, k_idx, w_shr) in source_portfolios:
        src_key = f"{src_m}_{src_s}"
        cross_eval[src_key] = {}
        for eval_m, eval_csvs in [("vine_copula", vine_csvs), ("tcvae", tcvae_csvs), ("tcvae_vine", tcvae_vine_csvs)]:
            for eval_s in range(N_SEEDS):
                eval_csv = eval_csvs.get(eval_s)
                eval_key = f"{eval_m}_{eval_s}"
                v = evaluate_cvar_python(
                    eval_csv or "", k_idx, w_shr, initial_prices
                ) if eval_csv else None
                cross_eval[src_key][eval_key] = v
                done += 1
        print(f"  [{src_key}] cross-eval done ({done}/{total_evals})")

    # ── Backtest ─────────────────────────────────────────────────────────────
    print("\n" + _hr("─"))
    print("  Running backtest (2025 data)…")
    print(_hr("─"))

    bt_results = []
    bt_prices  = load_backtest_prices()

    if bt_prices is not None:
        for method, csvs in [("vine_copula", vine_csvs), ("tcvae", tcvae_csvs), ("tcvae_vine", tcvae_vine_csvs)]:
            fr = in_sample[method].get(0)
            if not fr or mu_idx >= len(fr):
                print(f"  [{method} seed=0] no frontier at idx {mu_idx}, skipping backtest")
                continue
            mu_val, _, aws = fr[mu_idx]
            if aws is None:
                print(f"  [{method} seed=0] infeasible at idx {mu_idx}, skipping backtest")
                continue
            k_idx = [aw[1] for aw in aws]
            w_shr = [aw[2] for aw in aws]
            label = f"{method}_tree0_mu{mu_val:.4f}"
            print(f"  Backtesting {label}…")
            bt_results.append(
                backtest_portfolio(label, k_idx, w_shr, bt_prices, initial_prices)
            )
    else:
        print("  Backtest skipped (data not available)")

    # ── Write report ─────────────────────────────────────────────────────────
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"stability_report_{ts_str}.log")

    report_lines = _build_report(in_sample, cross_eval, bt_results, mu_idx, args)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{_hr('═')}")
    print(f"  Report written -> {log_path}")
    print(_hr("═"))

    # Print summary to console
    print("\n  === In-sample CV% (std/mean) at mid-frontier ===")
    for method in ("vine_copula", "tcvae", "tcvae_vine"):
        cvars = []
        for seed in range(N_SEEDS):
            fr = in_sample[method].get(seed)
            if fr and mu_idx < len(fr):
                cvars.append(fr[mu_idx][1])
        if cvars:
            m, s = np.mean(cvars), np.std(cvars, ddof=1) if len(cvars) > 1 else 0.0
            print(f"  {method:<20}: mean={m:.6f}  std={s:.6f}  CV%={100*s/m:.2f}%")


if __name__ == "__main__":
    main()
