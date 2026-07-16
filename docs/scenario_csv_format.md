# Scenario-tree CSV format

The optimizer consumes uncertainty as a **two-stage scenario tree** supplied as a
plain CSV file. This is the only interface between a scenario generator and the
engine: the built-in copula/vine generators, and external ones such as the
[TC-VAE generator](https://github.com/Ahkylez/TC-VAE-Scenario-Tree), all emit the
same format. Any generator that writes this format can drive the optimizer with
no code changes.

## Tree structure

A two-stage tree has:

- **Stage 1 — recourse nodes** `j = 0 … J-1`: the branch points where the
  here-and-now portfolio is evaluated. Each has a probability `p_j`.
- **Stage 2 — evaluate nodes** `e = 0 … E-1` under each recourse node: the
  outcomes used to measure loss / CVaR. Each has a conditional probability
  `p_je` given its parent, with `Σ_e p_je = 1` for every `j`.

Path probability of leaf `(j, e)` is `p_j · p_je`, and these sum to 1 over the
whole tree.

## Row and column layout

- **No header.** One **row per evaluate node**, so a tree with `J` recourse and
  `E` evaluate nodes has `J × E` rows.
- **`3 + 2Q` columns** for `Q` assets:

| Column(s) | Symbol | Meaning |
|-----------|--------|---------|
| `0` | `j` | recourse-node index (an integer, stored as a float e.g. `0.0`) |
| `1` | `p_j` | probability of recourse node `j` |
| `2` | `p_je` | conditional probability of evaluate node `e` given `j` |
| `3 … 3+Q-1` | `Pʲ` | **stage-1** (recourse) prices for the `Q` assets |
| `3+Q … 3+2Q-1` | `P^(j,e)` | **stage-2** (evaluate) prices for the `Q` assets |

Conventions:

- **Prices are absolute levels, not returns.** Initial prices `P0` are passed to
  the engine separately (see [Invocation](#invocation)); they are not in the CSV.
- Asset order is fixed by the caller and must match the tickers/`initial_prices`
  passed to the engine. The generators here use the tickers **sorted
  alphabetically**.
- All rows sharing a recourse index `j` repeat the same `p_j` and the same
  stage-1 price block `Pʲ`; they differ only in `p_je` and the stage-2 block.

## Example

For `Q = 30` assets, a tree with `J = 150` recourse × `E = 20` evaluate nodes has
`3 + 2·30 = 63` columns and `150 · 20 = 3000` rows. The first rows (all belonging
to recourse node `j = 0`, `p_j ≈ 0.00963`, `p_je = 1/20 = 0.05`):

```
0.0, 0.00962667, 0.05, 244.85, 250.04, …(28 more stage-1 prices)…, 244.85, 250.04, …(28 more stage-2 prices)…
0.0, 0.00962667, 0.05, 244.85, 250.04, …,                          , 251.10, 248.71, …
…                                                        (18 more evaluate nodes for j=0)
1.0, 0.00811432, 0.05, …                                 (evaluate nodes for j=1)
```

## Consistency checks

A well-formed tree satisfies:

- column count `== 3 + 2·Q`;
- `Σ_j p_j == 1` (sum over the **distinct** recourse nodes);
- `Σ_e p_je == 1` within each recourse node, hence `Σ_rows p_j·p_je == 1`;
- all prices `> 0`.

The training/verification tooling checks the column count and that
`Σ_rows p_j·p_je ≈ 1`.

## How the engine reads it

The Python orchestration parses the columns directly (see
[`main.py`](../main.py) — `run_frontier`, `get_mu_range`,
`get_tree_expected_returns`): `p_j = df[1]`, `p_je = df[2]`, stage-1 prices
`df[:, 3:3+Q]`, stage-2 prices `df[:, 3+Q:3+2Q]`. The tree's expected per-asset
return, used to set the frontier's target-return range, is

```
tree_return = Σ_rows (p_j · p_je) · P^(j,e) / P0 − 1
```

### Invocation

The compiled engine takes the CSV plus the dimensions and initial prices on the
command line:

```
optimizer_app  <scenario_csv>  <Q>  <cardinality>  <population>  <mu1,mu2,…>  <P0_1,…,P0_Q>
```

where `P0` is the vector of current (anchor) prices in the same asset order as
the tree columns.

## Generators that emit this format

- **Built-in copula** — `src/scenario_tree.py` (Kaut & Wallace 2011).
- **Built-in GARCH + vine copula** — `src/vine_scenario_tree.py` (He & Zhang 2024).
- **External TC-VAE** — the [TC-VAE Scenario Generation](https://github.com/Ahkylez/TC-VAE-Scenario-Tree)
  project writes `tcvae_week_NNN.csv` trees into `data/scenarios/backtest/<seed>/`.
