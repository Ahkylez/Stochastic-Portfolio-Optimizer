# Stochastic-Portfolio-Optimizer

A two-stage stochastic **portfolio optimization** engine that minimizes
**Conditional Value-at-Risk (CVaR)** under a cardinality constraint, driven by
scenario trees of uncertain future asset prices. The optimization core is a C++
hybrid metaheuristic built on [Google OR-Tools](https://developers.google.com/optimization),
wrapped in Python for scenario generation, backtesting, and efficient-frontier
analysis.

It is an implementation of:

> Tianxiang Cui, Ruibin Bai, Shusheng Ding, Andrew J. Parkes, Rong Qu, Fang He,
> and Jingpeng Li, **"A hybrid combinatorial approach to a two-stage stochastic
> portfolio optimization model with uncertain asset prices,"** *Soft Computing*,
> vol. 24, no. 4, pp. 2809–2831, Feb. 2020.

---

## A note on the state of this code

I'll be honest about what this repository is. This is **older code, finished
around May 2026**, and it started out *real messy* — the kind of project that
grows organically while you're chasing results and never quite gets tidied up.
It is **actively being cleaned up**: paths made portable, the build made
cross-platform, dead experiments pruned, and the structure straightened out.

I'm keeping this note here on purpose. This project was a lesson to me in **why
you keep a codebase clean as you go** instead of leaving it for "later" — later
is more expensive. If you're reading the history and wondering why some corners
look freshly swept and others don't, that's why: the cleanup is ongoing.

---

## The model in brief

The optimizer solves a **two-stage stochastic program**:

- **Stage 1 (here-and-now):** choose which assets to hold and how much, subject
  to a **cardinality constraint** (at most `K` assets) — a combinatorial choice.
- **Stage 2 (recourse):** under each future price scenario, evaluate the
  realized loss; the objective is the **CVaR** of the loss distribution across
  the scenario tree, for a target expected return `μ`.

Sweeping `μ` traces out an **efficient frontier** (return vs. tail risk).

### Why "hybrid combinatorial" (the Cui et al. approach)

The cardinality constraint makes asset *selection* a hard combinatorial problem,
while the *weight allocation* given a selection is a tractable linear/convex
program. Following the paper, the engine hybridizes the two:

- A **Population-Based Incremental Learning (PBIL)** metaheuristic searches over
  the combinatorial asset-selection space.
- For each candidate selection, an **LP/MILP solve (OR-Tools)** computes the
  optimal CVaR-minimizing weights.
- A **final exact MILP solve** is run on the global best to tighten the
  LP-relaxed result (`LP-relaxed CVaR → true MILP CVaR`).

See `src/pbil.cpp` and `src/hybrid_pbil.cpp`.

---

## Architecture

```
                Python (scenario generation + orchestration)
  ┌───────────────────────────────────────────────────────────────┐
  │  scenario generators  ──►  scenario tree CSV  ──►  run_frontier │
  └───────────────────────────────────────────────────────────────┘
                                    │  subprocess (CSV + args)
                                    ▼
                C++ engine  (build/optimizer_app, OR-Tools)
                PBIL search  +  LP/MILP CVaR solve
```

The Python side and the C++ engine are **decoupled through a CSV scenario-tree
contract** (see below). The optimizer never imports a scenario generator; it just
reads a CSV. That means *any* generator that emits the right CSV can drive it.

### Scenario generators (built in)

| Generator | File | Method |
|-----------|------|--------|
| Shape-based copula | `src/scenario_tree.py`, `src/scenario_generator.py` | Kaut & Wallace (2011) empirical copula + KS-optimal margins + Høyland–Kaut–Wallace (2003) moment matching |
| GARCH + vine copula | `src/vine_scenario_tree.py` | He & Zhang (2024): per-asset GARCH(1,1)-SkewStudent, R-vine copula (pyvinecopulib), K-means + LP moment matching |

### External scenario generator (TC-VAE)

A sibling project — a **Time-Causal VAE scenario generator** — produces scenario
trees in the same CSV format and drops them in as `tcvae_*.csv`. Because the
interface is just the CSV contract, TC-VAE trees are swapped in with no code
changes here. Those trees (e.g. the rolling-backtest `tcvae_week_NNN.csv` files)
are **regenerated from the TC-VAE project**, not stored in this repo.

---

## Repository layout

```
CMakeLists.txt            Cross-platform build (OR-Tools, OpenMP)
requirements.txt          Python deps for the generators / orchestration
src/
  pbil.cpp                PBIL metaheuristic (engine entry point)
  hybrid_pbil.cpp         Hybrid PBIL + LP/MILP CVaR solve
  optimizer_config.h      Engine parameters / types
  scenario_generator.py   Copula scenario primitives (Kaut & Wallace)
  scenario_tree.py        Copula two-stage tree builder + reduction
  vine_scenario_tree.py   GARCH + R-vine copula tree builder
  paths.py                Platform-aware locator for the built binary
main.py                   Efficient-frontier run (copula/vine vs TC-VAE)
dynamic_backtest.py       Rolling weekly backtest (primary entry point)
run_metrics.py            Thin driver around dynamic_backtest
stability_test.py         Kaut & Wallace stability test suite
analyze_tree.py           Scenario-tree diagnostics
```

`build/`, `data/`, and `logs/` are generated locally and are gitignored.

---

## Getting started

### 1. Build the C++ engine

Requires a C++17 compiler, CMake ≥ 3.18, and an **OR-Tools** C++ install.

**Linux / WSL (Ubuntu 24.04 example):**

```bash
sudo apt update && sudo apt install -y cmake ninja-build

# OR-Tools prebuilt C++ release (match your Ubuntu version):
mkdir -p ~/libs/or-tools && cd ~/libs
curl -L -o ortools.tar.gz \
  https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_ubuntu-24.04_cpp_v9.15.6755.tar.gz
tar -xzf ortools.tar.gz -C ~/libs/or-tools --strip-components=1 && rm ortools.tar.gz

# Build (defaults to OR_TOOLS_DIR=$HOME/libs/or-tools on Linux):
cd /path/to/Stochastic-Portfolio-Optimizer
cmake -S . -B build -G Ninja        # add -DOR_TOOLS_DIR=/your/path if elsewhere
cmake --build build                 # → build/optimizer_app
```

**Windows (MSVC):** set `OR_TOOLS_DIR` to your OR-Tools install and configure
with the Visual Studio generator; the build produces `build/Release/optimizer_app.exe`
and copies the OR-Tools DLLs beside it. The CMake handles both platforms.

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data

The scripts expect per-ticker price arrays in `data/raw/` (`<TICKER>.npy` for the
training window and `<TICKER>_bt.npy` for the backtest window) for the 30
large-cap US equities (Dow Jones Industrial Average constituents). TC-VAE
backtest trees go under `data/scenarios/backtest/<run_id>/`.

---

## Running

```bash
source .venv/bin/activate

python main.py            # generate efficient frontiers (vine copula + TC-VAE)
python dynamic_backtest.py   # rolling weekly backtest (needs TC-VAE trees)
python run_metrics.py     # driver around dynamic_backtest
python stability_test.py --fast   # quick stability smoke test
```

The Python layer builds/loads a scenario CSV, then calls the engine per frontier
point. You can also invoke the binary directly:

```
build/optimizer_app  <scenario_csv>  <Q>  <cardinality>  <population>  <mu1,mu2,...>  <price1,...,priceQ>
```

---

## Scenario-tree CSV format (the integration seam)

No header. Each row is one **evaluate node**, with `3 + 2Q` columns for `Q`
assets:

| Columns | Meaning |
|---------|---------|
| `0` | recourse-node index `j` (stage-1 node) |
| `1` | `p_j` — probability of recourse node `j` |
| `2` | `p_je` — conditional probability of evaluate node `e` given `j` (`Σ_e p_je = 1`) |
| `3 … 3+Q-1` | stage-1 (recourse) prices `Pʲ` for the `Q` assets |
| `3+Q … 3+2Q-1` | stage-2 (evaluate) prices `P^(j,e)` for the `Q` assets |

Prices are absolute (not returns). Any generator that emits this format — copula,
vine, or the external TC-VAE — can drive the optimizer unchanged.

---

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).
