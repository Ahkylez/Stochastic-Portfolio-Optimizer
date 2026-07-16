#pragma once
#include <string>
#include <vector>

struct OptimizerConfig {
    std::string scenarioPath;
    int Q{};
    int K{};
    double mu{};
    int popsize{};
    
    // Deterministic input
    double h      = 100000.0;  // initial cash (V0)
    double initial_cash = 100000.0;
    std::vector<double> initial_holdings; // Size Q
    double eta_b  = 0.0;       // fixed buying cost per trade (modern markets are commission-free)
    double eta_s  = 0.0;       // fixed selling cost per trade
    double rho_b  = 0.0005;    // variable buying cost (0.05% or 5 bps for spread/slippage)
    double rho_s  = 0.0005;    // variable selling cost (0.05% or 5 bps for spread/slippage)
    
    // Diversification controls (interpreted as fractions of h)
    // Paper uses w_min=0.01, t_min=0.001 with normalized data; for raw $ prices
    // we scale these as fractions of initial wealth h to get meaningful constraints.
    double w_min  = 0.01;      // minimum holding per asset = 1% of h
    double t_min  = 0.01;      // minimum trade size = 1% of h
    
    double M_big  = 1e6;       // unused (now per-asset)
    double beta   = 0.95;      // CVaR confidence level
    std::vector<double> initial_prices;
    
    // Solver
    int solver_time_limit = 10000 ; // ms; J=150 LPs need more than 5s to prove optimality
    int na = 15;
    int generations = 60;         // was 50; more generations to escape local optima
    double lr = 0.1;
    double nelr = 0.075;
    double mr = 0.10;              // was 0.05; stronger mutations to escape basins
    double mp = 0.10;              // was 0.05; more frequent mutations
    bool use_lp_relaxation  = true;
    bool final_milp_solve   = true; // re-solve global best as full MILP at end of each run
};