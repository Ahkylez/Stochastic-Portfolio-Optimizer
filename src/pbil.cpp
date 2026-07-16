#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include "hybrid_pbil.h"
#include "optimizer_config.h"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"


int main(int argc, char* argv[]){
    absl::InitializeLog();
    // Force Abseil to print INFO logs to the console
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

    if (argc < 7){
        std::cerr << "Usage: pbil.exe <scenarioPath> <total_assets> <cardinality> <pop_size> <target_return> <initial_prices_csv> [initial_cash] [initial_holdings_csv]\n";
        return 1;
    }

    // argv[5] is a comma-separated list of mu values (one or more).
    // All mu levels are run in a single invocation so v warm-starts between them.
    std::string scenarioPath    = argv[1];
    int         totalAssets     = std::stoi(argv[2]);
    int         cardinality     = std::stoi(argv[3]);
    int         popSize         = std::stoi(argv[4]);
    std::string initialPricesStr = argv[6];

    std::vector<double> muValues;
    {
        std::stringstream mu_ss(argv[5]);
        std::string mu_item;
        while (std::getline(mu_ss, mu_item, ',')) {
            muValues.push_back(std::stod(mu_item));
        }
    }
    if (muValues.empty()) {
        std::cerr << "No mu values provided.\n";
        return 1;
    }

    std::vector<double> initialPrices;
    {
        std::stringstream ss(initialPricesStr);
        std::string item;
        while (std::getline(ss, item, ',')) {
            initialPrices.push_back(std::stod(item));
        }
    }

    std::cout << "--- C++ Engine Started ---\n";
    std::cout << "Scenarios: " << scenarioPath << "\n";
    std::cout << "Q=" << totalAssets << " K=" << cardinality
              << " Pop=" << popSize << " mu_count=" << muValues.size() << "\n";

    OptimizerConfig config;
    config.scenarioPath   = scenarioPath;
    config.Q              = totalAssets;
    config.K              = cardinality;
    config.popsize        = popSize;
    config.mu             = muValues[0];
    config.initial_prices = initialPrices;
    
    config.initial_cash = (argc >= 8) ? std::stod(argv[7]) : 100000.0;
    if (argc >= 9) {
        std::stringstream ss(argv[8]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            config.initial_holdings.push_back(std::stod(item));
        }
    } else {
        config.initial_holdings.assign(totalAssets, 0.0);
    }

    double V0 = config.initial_cash;
    for (int i = 0; i < totalAssets; i++) {
        V0 += config.initial_holdings[i] * initialPrices[i];
    }
    config.h = V0; // Update target wealth constraint reference

    HybridPBIL optimizer{config};
    optimizer.runFrontier(muValues);

    return 0;
}