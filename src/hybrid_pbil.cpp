#include "hybrid_pbil.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <omp.h>

#include "ortools/linear_solver/linear_solver.h"
#include "absl/log/globals.h"


HybridPBIL::HybridPBIL(const OptimizerConfig& cfg)
    : config{ cfg }
    , v(cfg.Q, 0.33)
    , gen(std::random_device{}())
    , uni(0.0, 1.0)
{
    parseTreeCSV(config.scenarioPath);
    precomputeExpectedPrices();
}

void HybridPBIL::parseTreeCSV(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open: " + path);
    }
    std::unordered_map<int, int> idxToNode;

    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream ss(line);
        std::vector<double> row;
        row.reserve(3 + 2 * config.Q);

        for (std::string val; std::getline(ss, val, ',');) {
            row.push_back(std::stod(val));
        }
        if (row.size() != static_cast<size_t>(3 + 2 * config.Q)) {
            throw std::runtime_error(
                "Tree CSV row has " + std::to_string(row.size()) +
                " columns, expected " + std::to_string(3 + 2 * config.Q)
            );
        }
        int    recourse_idx = static_cast<int>(row[0]);
        double p_j          = row[1];
        double p_je         = row[2];
        std::vector<double> recourse_prices(row.begin() + 3,             row.begin() + 3 + config.Q);
        std::vector<double> eval_prices_row(row.begin() + 3 + config.Q,  row.end());

        if (idxToNode.find(recourse_idx) == idxToNode.end()) {
            RecourseNode node;
            node.p_j    = p_j;
            node.prices = recourse_prices;
            recourseNodes.push_back(std::move(node));
            idxToNode[recourse_idx] = static_cast<int>(recourseNodes.size()) - 1;
        }

        int nodePos = idxToNode[recourse_idx];
        recourseNodes[nodePos].eval_prob.push_back(p_je);
        recourseNodes[nodePos].eval_prices.push_back(eval_prices_row);
    }

    std::cout << "Parsed " << recourseNodes.size() << " recourse nodes from: " << path << "\n";
    if (!recourseNodes.empty()) {
        std::cout << "  Each with " << recourseNodes[0].eval_prices.size() << " evaluate nodes\n";
    }
}

void HybridPBIL::precomputeExpectedPrices() {
    expectedEvalPrices.assign(recourseNodes.size(), std::vector<double>(config.Q, 0.0));
    for (size_t j = 0; j < recourseNodes.size(); j++) {
        const RecourseNode& node = recourseNodes[j];
        const int num_evals = node.eval_prices.size();
        for (int e = 0; e < num_evals; e++) {
            const double pe = node.eval_prob[e];
            const std::vector<double>& prices_e = node.eval_prices[e];
            for (int q = 0; q < config.Q; q++) {
                expectedEvalPrices[j][q] += pe * prices_e[q];
            }
        }
    }
    LOG(INFO) << "Pre-computed expected eval prices for "
              << recourseNodes.size() << " nodes x " << config.Q << " assets";
}

int HybridPBIL::getOverlap(const std::vector<int>& v1, const std::vector<int>& v2){
    int overlap = 0;
    size_t i = 0, j = 0;
    while (i < v1.size() && j < v2.size()) {
        if (v1[i] == v2[j]) {
            overlap++;
            i++;
            j++;
        } else if (v1[i] < v2[j]) {
            i++;
        } else {
            j++;
        }
    }
    return overlap;
}

bool HybridPBIL::isBadOrInfeasible(const std::vector<int>& k){
    for (const std::vector<int>& entry : infeasibleSolutions){
        if (getOverlap(k, entry) >= config.K - 1){
            return true;
        }
    }
    for (const std::vector<int>& entry: badSolutions){
        if (getOverlap(k, entry) >= config.K - 2){
            return true;
        }
    }
    return false;
}

void HybridPBIL::generateIndividual(Portfolio& portfolio){
    portfolio.c.clear();
    portfolio.k.clear();
    for (int i = 0; i < config.Q; i++){
        bool val = uni(gen) < v[i];
        portfolio.c.push_back(val);
    }

    int Kprime = std::count(portfolio.c.begin(), portfolio.c.end(), 1);

    std::vector<int> valid;
    for (size_t i = 0; i < portfolio.c.size(); i++){
        if (portfolio.c[i] == 1){
            valid.push_back((int)i);
        }
    }

    if (Kprime > config.K){
        std::shuffle(valid.begin(), valid.end(), gen);
        for (int i = 0; i < config.K; i++){
            portfolio.k.push_back(valid[i]);
        }
    } else if (Kprime < config.K){
        portfolio.k = valid;
        std::vector<int> notValid;
        for (size_t i = 0; i < portfolio.c.size(); i++){
            if (portfolio.c[i] == 0){
                notValid.push_back((int)i);
            }
        }
        int need = config.K - Kprime;
        std::shuffle(notValid.begin(), notValid.end(), gen);
        for (int i = 0; i < need; i++){
            portfolio.k.push_back(notValid[i]);
        }
    } else {
        portfolio.k = valid;
    }
    std::sort(portfolio.k.begin(), portfolio.k.end());
    std::fill(portfolio.c.begin(), portfolio.c.end(), 0);
    for (int asset : portfolio.k) portfolio.c[asset] = 1;
}

void HybridPBIL::hashSearch(std::vector<Portfolio> &population){
    for (Portfolio& individual : population){
        int attempts = 0;
        while (isBadOrInfeasible(individual.k) && attempts < 50){
            generateIndividual(individual);
            attempts++;
        }
    }
}

SolveResult HybridPBIL::solveOnePortfolio(const std::vector<int>& k, bool use_lp) const {
    namespace OR = operations_research;
    SolveResult result{std::numeric_limits<double>::infinity(), {}, false};

    const char* solver_name = use_lp ? "GLOP" : "SCIP";
    std::unique_ptr<OR::MPSolver> solver(OR::MPSolver::CreateSolver(solver_name));
    if (!solver) return result;

    if (use_lp) {
        solver->set_time_limit(500);
    } else {
        solver->set_time_limit(config.solver_time_limit > 0 ? config.solver_time_limit : 50010);
        solver->SetSolverSpecificParametersAsString("limits/memory=2048\nlimits/gap=0.25\n");
    }

    auto vname = [](const char*, const std::string&) -> std::string { return std::string(); };
    auto makeBin = [&](const std::string& name) -> OR::MPVariable* {
        return use_lp ? solver->MakeNumVar(0.0, 1.0, name) : solver->MakeBoolVar(name);
    };

    const double infinity = solver->infinity();
    OR::MPVariable* alpha = solver->MakeNumVar(-infinity, infinity, vname("alpha", ""));

    // Stage 1 variables
    std::vector<OR::MPVariable*> b1(config.K), s1(config.K), w1(config.K);
    std::vector<OR::MPVariable*> f1(config.K), g1(config.K);
    for (size_t i = 0; i < config.K; i++) {
        std::string idx = std::to_string(k[i]);
        b1[i] = solver->MakeNumVar(0.0, infinity, vname("b1_", idx));
        s1[i] = solver->MakeNumVar(0.0, infinity, vname("s1_", idx));
        w1[i] = solver->MakeNumVar(0.0, infinity, vname("w1_", idx));
        f1[i] = makeBin(vname("f1_", idx));
        g1[i] = makeBin(vname("g1_", idx));
    }

    // Stage 2 variables
    const int num_scenarios = recourseNodes.size();
    std::vector<OR::MPVariable*> z(num_scenarios);
    std::vector<std::vector<OR::MPVariable*>> b2(num_scenarios, std::vector<OR::MPVariable*>(config.K));
    std::vector<std::vector<OR::MPVariable*>> s2(num_scenarios, std::vector<OR::MPVariable*>(config.K));
    std::vector<std::vector<OR::MPVariable*>> w2(num_scenarios, std::vector<OR::MPVariable*>(config.K));
    std::vector<std::vector<OR::MPVariable*>> f2(num_scenarios, std::vector<OR::MPVariable*>(config.K));
    std::vector<std::vector<OR::MPVariable*>> g2(num_scenarios, std::vector<OR::MPVariable*>(config.K));

    for (int j = 0; j < num_scenarios; j++) {
        std::string s_idx = std::to_string(j);
        z[j] = solver->MakeNumVar(0.0, infinity, vname("z_", s_idx));
        for (size_t i = 0; i < config.K; i++) {
            std::string asset = s_idx + "_" + std::to_string(k[i]);
            b2[j][i] = solver->MakeNumVar(0.0, infinity, vname("b2_", asset));
            s2[j][i] = solver->MakeNumVar(0.0, infinity, vname("s2_", asset));
            w2[j][i] = solver->MakeNumVar(0.0, infinity, vname("w2_", asset));
            // Only generate these if we are doing the exact MILP solve
            if (!use_lp){
                f2[j][i] = makeBin(vname("f2_", asset));
                g2[j][i] = makeBin(vname("g2_", asset));
            }
        }
    }

    // Auto-liquidate assets held that are NOT in the newly selected `k`
    // and prep populate the `w0` state for assets that ARE in `k`
    double available_cash = config.initial_cash;
    std::vector<double> w0(config.K, 0.0);

    for (int q = 0; q < config.Q; q++) {
        if (config.initial_holdings[q] > 0) {
            auto it = std::find(k.begin(), k.end(), q);
            if (it != k.end()) {
                int idx = std::distance(k.begin(), it);
                w0[idx] = config.initial_holdings[q];
            } else {
                // Liquidate and incur selling costs immediately
                double P_0 = config.initial_prices[q];
                available_cash += config.initial_holdings[q] * P_0 * (1.0 - config.rho_s) - config.eta_s;
            }
        }
    }

    // -------- Stage 1 constraints --------
    OR::MPConstraint* c_cash1 = solver->MakeRowConstraint(available_cash, available_cash, vname("cash1", ""));

    for (size_t i = 0; i < config.K; i++) {
        std::string idx = std::to_string(k[i]);
        const double P_0 = config.initial_prices[k[i]];
        const double M_i = config.h / (P_0 * (1.0 + config.rho_b)) + 1.0;

        // Convert fractional w_min and t_min to per-asset unit floors
        // w_min as fraction of h: minimum dollar value per holding = w_min * h
        //                          -> minimum units = (w_min * h) / P_0
        const double w_min_units = (config.w_min * config.h) / P_0;
        const double t_min_units = (config.t_min * config.h) / P_0;

        // Eq 13: cash balance
        c_cash1->SetCoefficient(b1[i], P_0 * (1.0 + config.rho_b));
        c_cash1->SetCoefficient(f1[i], config.eta_b);
        c_cash1->SetCoefficient(s1[i], -P_0 * (1.0 - config.rho_s));
        c_cash1->SetCoefficient(g1[i], config.eta_s);

        // Eq 12: asset balance (carry forward existing units)
        OR::MPConstraint* c_asset1 = solver->MakeRowConstraint(w0[i], w0[i], vname("asset1_", idx));
        c_asset1->SetCoefficient(w1[i], 1.0);
        c_asset1->SetCoefficient(b1[i], -1.0);
        c_asset1->SetCoefficient(s1[i], 1.0);

        // Eq 15: minimum holding (with c_i = 1 implicit since asset is in k)
        OR::MPConstraint* c_min_hold1 = solver->MakeRowConstraint(w_min_units, infinity, vname("minh1_", idx));
        c_min_hold1->SetCoefficient(w1[i], 1.0);

        // Eq 16: t_min * f_i <= b_i
        OR::MPConstraint* c_min_buy1 = solver->MakeRowConstraint(0.0, infinity, vname("minb1_", idx));
        c_min_buy1->SetCoefficient(b1[i], 1.0);
        c_min_buy1->SetCoefficient(f1[i], -t_min_units);

        // Eq 17: t_min * g_i <= s_i
        OR::MPConstraint* c_min_sell1 = solver->MakeRowConstraint(0.0, infinity, vname("mins1_", idx));
        c_min_sell1->SetCoefficient(s1[i], 1.0);
        c_min_sell1->SetCoefficient(g1[i], -t_min_units);

        // Eq 18: f_i + g_i <= 1
        OR::MPConstraint* c_no_simul1 = solver->MakeRowConstraint(-infinity, 1.0, vname("nos1_", idx));
        c_no_simul1->SetCoefficient(f1[i], 1.0);
        c_no_simul1->SetCoefficient(g1[i], 1.0);

        // Eq 19: b_i <= f_i * M
        OR::MPConstraint* c_bigM_buy1 = solver->MakeRowConstraint(-infinity, 0.0, vname("bMb1_", idx));
        c_bigM_buy1->SetCoefficient(b1[i], 1.0);
        c_bigM_buy1->SetCoefficient(f1[i], -M_i);

        // Eq 20: s_i <= g_i * M
        OR::MPConstraint* c_bigM_sell1 = solver->MakeRowConstraint(-infinity, 0.0, vname("bMs1_", idx));
        c_bigM_sell1->SetCoefficient(s1[i], 1.0);
        c_bigM_sell1->SetCoefficient(g1[i], -M_i);
    }

    // -------- Stage 2 constraints --------
    OR::MPConstraint* c_target_return = solver->MakeRowConstraint(
        config.h * (1.0 + config.mu), infinity, vname("target_return", "")
    );

    for (int j = 0; j < num_scenarios; j++) {
        std::string s_idx = std::to_string(j);
        const RecourseNode& node = recourseNodes[j];
        const std::vector<double>& exp_prices = expectedEvalPrices[j];

        OR::MPConstraint* c_cash2 = solver->MakeRowConstraint(0.0, 0.0, vname("cash2_", s_idx));

        for (size_t i = 0; i < config.K; i++) {
            std::string idx = s_idx + "_" + std::to_string(k[i]);
            const double P_j = node.prices[k[i]];
            const double M_ij = config.h / (P_j * (1.0 + config.rho_b)) + 1.0;
            const double w_min_units_j = (config.w_min * config.h) / P_j;
            const double t_min_units_j = (config.t_min * config.h) / P_j;

            // Eq 26: cash balance for scenario j
            c_cash2->SetCoefficient(b2[j][i], P_j * (1.0 + config.rho_b));
            c_cash2->SetCoefficient(s2[j][i], -P_j * (1.0 - config.rho_s));

            if (!use_lp){
                c_cash2->SetCoefficient(f2[j][i], config.eta_b);
                c_cash2->SetCoefficient(g2[j][i], config.eta_s);
            }
            // Eq 25: asset balance scenario j
            OR::MPConstraint* c_asset2 = solver->MakeRowConstraint(0.0, 0.0, vname("asset2_", idx));
            c_asset2->SetCoefficient(w2[j][i], 1.0);
            c_asset2->SetCoefficient(w1[i], -1.0);
            c_asset2->SetCoefficient(b2[j][i], -1.0);
            c_asset2->SetCoefficient(s2[j][i], 1.0);

            // Eq 28: minimum holding scenario j
            OR::MPConstraint* c_min_hold2 = solver->MakeRowConstraint(w_min_units_j, infinity, vname("minh2_", idx));
            c_min_hold2->SetCoefficient(w2[j][i], 1.0);

            if (!use_lp){
                // Eq 29: minimum buy scenario j
                OR::MPConstraint* c_min_buy2 = solver->MakeRowConstraint(0.0, infinity, vname("minb2_", idx));
                c_min_buy2->SetCoefficient(b2[j][i], 1.0);
                c_min_buy2->SetCoefficient(f2[j][i], -t_min_units_j);

                // Eq 30: minimum sell scenario j
                OR::MPConstraint* c_min_sell2 = solver->MakeRowConstraint(0.0, infinity, vname("mins2_", idx));
                c_min_sell2->SetCoefficient(s2[j][i], 1.0);
                c_min_sell2->SetCoefficient(g2[j][i], -t_min_units_j);

                // Eq 31: no simultaneous buy/sell scenario j
                OR::MPConstraint* c_no_simul2 = solver->MakeRowConstraint(-infinity, 1.0, vname("nos2_", idx));
                c_no_simul2->SetCoefficient(f2[j][i], 1.0);
                c_no_simul2->SetCoefficient(g2[j][i], 1.0);

                // Eq 32: bigM buy scenario j
                OR::MPConstraint* c_bigM_buy2 = solver->MakeRowConstraint(-infinity, 0.0, vname("bMb2_", idx));
                c_bigM_buy2->SetCoefficient(b2[j][i], 1.0);
                c_bigM_buy2->SetCoefficient(f2[j][i], -M_ij);

                // Eq 33: bigM sell scenario j
                OR::MPConstraint* c_bigM_sell2 = solver->MakeRowConstraint(-infinity, 0.0, vname("bMs2_", idx));
                c_bigM_sell2->SetCoefficient(s2[j][i], 1.0);
                c_bigM_sell2->SetCoefficient(g2[j][i], -M_ij);
            }
        }

        // Eq 36-39: shortfall (uses precomputed expected eval prices)
        OR::MPConstraint* c_shortfall = solver->MakeRowConstraint(config.h, infinity, vname("short_", s_idx));
        c_shortfall->SetCoefficient(z[j], 1.0);
        c_shortfall->SetCoefficient(alpha, 1.0);

        for (size_t i = 0; i < config.K; i++) {
            const double E_P = exp_prices[k[i]];
            c_shortfall->SetCoefficient(w2[j][i], E_P);
            // Eq 40: target return accumulation
            c_target_return->SetCoefficient(w2[j][i], node.p_j * E_P);
        }
    }

    // -------- Objective: Eq 11 --------
    OR::MPObjective* objective = solver->MutableObjective();
    objective->SetMinimization();
    objective->SetCoefficient(alpha, 1.0);
    const double cvar_multiplier = 1.0 / (1.0 - config.beta);
    for (int j = 0; j < num_scenarios; j++) {
        objective->SetCoefficient(z[j], recourseNodes[j].p_j * cvar_multiplier);
    }

    // -------- Solve --------
    OR::MPSolver::ResultStatus status = solver->Solve();
    if (status == OR::MPSolver::OPTIMAL || status == OR::MPSolver::FEASIBLE) {
        result.fitness = objective->Value();
        result.w.reserve(config.K);
        for (size_t i = 0; i < config.K; i++) {
            result.w.push_back(w1[i]->solution_value());
        }
        result.feasible = true;
    }
    return result;
}

void HybridPBIL::evaluation(std::vector<Portfolio>& population, bool is_local_search = false) {
    // Phase 1: serial cache lookup
    std::vector<size_t> to_solve;
    to_solve.reserve(population.size());
    for (size_t p = 0; p < population.size(); p++) {
        auto it = fitnessCache.find(population[p].k);
        if (it != fitnessCache.end()) {
            population[p].fitness = it->second.first;
            population[p].w = it->second.second;
        } else {
            to_solve.push_back(p);
        }
    }

    if (to_solve.empty()) return;

    // Phase 2: parallel solve using LP relaxation for speed
    const int N = (int)to_solve.size();
    std::vector<SolveResult> results(N);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < N; idx++) {
        results[idx] = solveOnePortfolio(population[to_solve[idx]].k, config.use_lp_relaxation);
    }

    // Phase 3: serial merge into shared state
    for (int idx = 0; idx < N; idx++) {
        Portfolio& portfolio = population[to_solve[idx]];
        const SolveResult& r = results[idx];

        if (r.feasible) {
            portfolio.fitness = r.fitness;
            portfolio.w = r.w;
            fitnessCache[portfolio.k] = {r.fitness, r.w};
        } else {
            portfolio.fitness = std::numeric_limits<double>::infinity();
            portfolio.w.clear();
            fitnessCache[portfolio.k] = {std::numeric_limits<double>::infinity(), {}};

            if (!is_local_search){
                infeasibleSolutions.push_back(portfolio.k);
            }
            
        }
    }

    LOG(INFO) << "Evaluated " << N << " portfolios in parallel ("
              << (config.use_lp_relaxation ? "LP" : "MILP") << ")";
}

void HybridPBIL::localSearch(std::vector<Portfolio>& population){
    std::sort(population.begin(), population.end(), [](const Portfolio& a, const Portfolio& b){
        return a.fitness < b.fitness;
    });

    int top = (int)(population.size() * 0.20);
    LOG(INFO) << "--- Local Search on top " << top << " portfolios ---";

    for (int i = 0; i < top; i++){
        Portfolio& currentBest = population[i];

        // Build all candidates for this portfolio in one batch
        std::vector<Portfolio> candidates;
        candidates.reserve(currentBest.k.size() * config.na);

        for (size_t k_idx = 0; k_idx < currentBest.k.size(); k_idx++){
            int originalAsset = currentBest.k[k_idx];

            std::vector<std::pair<double, int>> neighbors;
            neighbors.reserve(config.Q);
            for (int asset = 0; asset < config.Q; asset++){
                if (asset != originalAsset){
                    double distance = std::abs(v[originalAsset] - v[asset]);
                    neighbors.push_back({distance, asset});
                }
            }
            std::sort(neighbors.begin(), neighbors.end());

            int neighborsChecked = 0;
            for (auto& pair : neighbors){
                if (neighborsChecked >= config.na) break;
                int neighborAsset = pair.second;

                if (std::find(currentBest.k.begin(), currentBest.k.end(), neighborAsset) != currentBest.k.end()) {
                    continue;
                }
                neighborsChecked++;

                Portfolio candidate = currentBest;
                candidate.k[k_idx] = neighborAsset;
                std::sort(candidate.k.begin(), candidate.k.end());

                if (isBadOrInfeasible(candidate.k)) continue;

                candidates.push_back(std::move(candidate));
            }
        }

        if (candidates.empty()) continue;

        // ONE batched parallel evaluation
        evaluation(candidates, true);

        // Best-improvement: pick the strongest improving neighbor
        auto best_it = std::min_element(candidates.begin(), candidates.end(),
            [](const Portfolio& a, const Portfolio& b){ return a.fitness < b.fitness; });

        if (best_it != candidates.end() && best_it->fitness < currentBest.fitness) {
            LOG(INFO) << "Local Search Improvement! "
                      << currentBest.fitness << " -> " << best_it->fitness;
            currentBest = *best_it;
            std::fill(currentBest.c.begin(), currentBest.c.end(), 0);
            for (int asset : currentBest.k) currentBest.c[asset] = 1;
        }
    }
}

void HybridPBIL::archive(const Portfolio& current_best, const Portfolio& current_worst) {
    if (!has_best_portfolio || current_best.fitness < global_best_portfolio.fitness) {
        global_best_portfolio = current_best;
        has_best_portfolio = true;
        LOG(INFO) << "*** NEW GLOBAL BEST! CVaR: " << global_best_portfolio.fitness << " ***";
    }

    if (current_worst.k.size() > 0) {
        badSolutions.push_back(current_worst.k);
    }
}

void HybridPBIL::update(const Portfolio& current_best, const Portfolio& current_worst){
    for (int i = 0; i < config.Q; i++){
        v[i] = v[i] * (1.0 - config.lr) + (current_best.c[i] * config.lr);
        if (current_best.c[i] != current_worst.c[i]){
            v[i] = v[i] * (1 - config.nelr) + current_best.c[i] * config.nelr;
        }
    }
}

void HybridPBIL::mutation(){
    for (int i = 0; i < config.Q; i++){
        if (uni(gen) < config.mp){
            if (uni(gen) < 0.5 || !has_best_portfolio){
                double r = uni(gen);
                v[i] = v[i] * (1 - config.mr) + r * config.mr;
            } else {
                v[i] = global_best_portfolio.c[i];
            }
        }
    }
}

Portfolio HybridPBIL::run(){
    for (int g = 0; g < config.generations; g++) {
        LOG(INFO) << "=== Generation " << g << " ===";
        std::vector<Portfolio> population{};

        if (has_best_portfolio) {
            population.push_back(global_best_portfolio);
        } else if (has_warm_seed) {
            // Warm-start: seed gen-0 with the previous mu's best composition.
            // It gets re-evaluated at the new mu so fitness is always current.
            population.push_back(warm_seed_portfolio);
        }

        while (population.size() < config.popsize) {
            Portfolio newPort{};
            generateIndividual(newPort);
            population.push_back(newPort);
        }

        hashSearch(population);
        evaluation(population);
        localSearch(population);

        std::sort(population.begin(), population.end(), [](const Portfolio& a, const Portfolio& b) {
            return a.fitness < b.fitness;
        });

        Portfolio current_best = population.front();
        Portfolio current_worst;
        bool has_worst = false;

        for (auto it = population.rbegin(); it != population.rend(); ++it) {
            if (it->fitness != std::numeric_limits<double>::infinity()) {
                current_worst = *it;
                has_worst = true;
                break;
            }
        }

        archive(current_best, current_worst);
        if (has_worst) {
            update(current_best, current_worst);
        }
        mutation();
    }

    // Final pass: re-solve global best as full MILP for the true CVaR value
    if (has_best_portfolio && config.use_lp_relaxation && config.final_milp_solve) {
        LOG(INFO) << "=== Final MILP solve on global best ===";
        SolveResult final_result = solveOnePortfolio(global_best_portfolio.k, false);
        if (final_result.feasible) {
            LOG(INFO) << "LP-relaxed CVaR: " << global_best_portfolio.fitness
                      << "  ->  True MILP CVaR: " << final_result.fitness;
            global_best_portfolio.fitness = final_result.fitness;
            global_best_portfolio.w = final_result.w;
        } else {
            LOG(WARNING) << "Final MILP solve failed; keeping LP-relaxed result";
        }
    }

    return global_best_portfolio;
}

void HybridPBIL::runFrontier(const std::vector<double>& mu_values) {
    for (double mu : mu_values) {
        // Carry forward the best composition as a warm seed for the next run.
        // v is already warm (it was updated by the previous run's PBIL loop).
        if (has_best_portfolio) {
            warm_seed_portfolio = global_best_portfolio;
            has_warm_seed = true;
        }

        // Reset per-mu state. The fitness cache is mu-dependent (CVaR changes
        // with the return constraint), so it must be cleared between runs.
        // Reset v to neutral prior so the new level starts with unbiased search;
        // the warm-seed portfolio is still injected into gen-0 for a good
        // starting composition without propagating v-bias from the previous level.
        config.mu = mu;
        has_best_portfolio = false;
        std::fill(v.begin(), v.end(), 0.33);
        fitnessCache.clear();
        infeasibleSolutions.clear();
        badSolutions.clear();

        LOG(INFO) << "=== Frontier: mu=" << std::fixed << std::setprecision(6) << mu << " ===";
        Portfolio result = run();

        if (result.fitness != std::numeric_limits<double>::infinity() && !result.k.empty()) {
            std::cout << "FINAL_CVAR:" << std::fixed << std::setprecision(6) << result.fitness << "\n";

            std::cout << "PORTFOLIO_K:";
            for (size_t i = 0; i < result.k.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << result.k[i];
            }
            std::cout << "\n";

            std::cout << "PORTFOLIO_W:";
            for (size_t i = 0; i < result.w.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << std::fixed << std::setprecision(6) << result.w[i];
            }
            std::cout << std::endl;
        } else {
            // Infeasible mu — emit sentinel so Python index alignment is preserved
            std::cout << "FINAL_CVAR:-1\nPORTFOLIO_K:\nPORTFOLIO_W:" << std::endl;
        }
    }
}