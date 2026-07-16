#pragma once

#include <vector>
#include <functional>
#include <cstddef>

struct Portfolio{
    /*  
        The papers reasoning for having k and c because I also thought it was not needed:
        "It might seem that using both vector c and vector k is redundant. 
        The idea is that vector c is used for learning purpose, 
        as we need to trace the whole efficient frontier, 
        and vector k is only used for one point (portfolio) each time."
    */
    std::vector<int> c{}; // 0 or 1, represents if that specific asset is chosen. Size Q 
    std::vector<int> k{}; // The K selected assets from c. 
    double fitness{}; // CVaR - Calculate fitness function F using LP solver. Maps K ints and target return mu to a real number R. F calculates CVaR.
    std::vector<double> w{}; // Portfolio weights/allocations (units held)
    // The goal is to find the portfolio with the least fitnitess (CVaR)
};

// https://stackoverflow.com/questions/29855908/c-unordered-set-of-vectors
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
}; 

struct RecourseNode {
    double p_j;                                    // probability of this recourse node
    std::vector<double> prices;                    // P^j_i, price of each asset. Size Q
    std::vector<double> eval_prob;                 // p_(j,e), for each evaluate node
    std::vector<std::vector<double>> eval_prices;  // P^(j,e)_i, size E*Q
};

struct SolveResult {
    double fitness;
    std::vector<double> w;
    bool feasible;
};
