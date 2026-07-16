#pragma once

#include <vector>
#include <unordered_map>
#include <random>
#include <string>
#include <utility>

#include "optimizer_config.h"
#include "types.h"

class HybridPBIL {
private:
    OptimizerConfig config;

    Portfolio global_best_portfolio;
    bool has_best_portfolio = false;

    // Warm-start seed: best portfolio from the previous mu level.
    // v persists naturally; this injects the previous composition into gen-0.
    Portfolio warm_seed_portfolio;
    bool has_warm_seed = false;

    // Random
    std::mt19937 gen;
    std::uniform_real_distribution<double> uni;

    std::vector<double> v{};                       // Probability vector v = (v1,v2,...,vQ)
    std::vector<RecourseNode> recourseNodes;       // Flattened Tree Recourse Nodes, Nr
    std::vector<std::vector<double>> expectedEvalPrices;  // Pre-computed E[P_i^j] for all j, i

    std::unordered_map<std::vector<int>, std::pair<double, std::vector<double>>, VectorHash> fitnessCache;

    // Hash search with hashtables make no sense to me so we'll use vectors
    std::vector<std::vector<int>> infeasibleSolutions;
    std::vector<std::vector<int>> badSolutions;

    void parseTreeCSV(const std::string& path);
    void precomputeExpectedPrices();
    SolveResult solveOnePortfolio(const std::vector<int>& k, bool use_lp) const;

    int getOverlap(const std::vector<int>& v1, const std::vector<int>& v2);
    bool isBadOrInfeasible(const std::vector<int>& k);

public:
    HybridPBIL(const OptimizerConfig& cfg);

    void generateIndividual(Portfolio& portfolio);
    void hashSearch(std::vector<Portfolio>& population);
    void evaluation(std::vector<Portfolio>& population,  bool is_local_search);
    void localSearch(std::vector<Portfolio>& population);
    void archive(const Portfolio& current_best, const Portfolio& current_worst);
    void update(const Portfolio& current_best, const Portfolio& current_worst);
    void mutation();
    Portfolio run();

    // Run the full frontier: one PBIL run per mu value, warm-starting v and
    // the best portfolio composition between consecutive mu levels.
    // Emits one FINAL_CVAR / PORTFOLIO_K / PORTFOLIO_W block per mu to stdout.
    void runFrontier(const std::vector<double>& mu_values);
};