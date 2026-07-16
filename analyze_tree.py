import os
import sys
import numpy as np
import pandas as pd

def main():
    tickers = sorted([
        "MMM", "AMZN", "AXP", "AMGN", "AAPL",
        "BA", "CAT", "CVX", "CSCO", "KO",
        "DIS", "GS", "HD", "HON", "IBM",
        "JNJ", "JPM", "MCD", "MRK", "MSFT",
        "NKE", "NVDA", "PG", "CRM", "SHW",
        "TRV", "UNH", "VZ", "V", "WMT"
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    prices_dict = {}
    for t in tickers:
        try:
            prices_dict[t] = np.load(os.path.join(current_dir, "data", "raw", f"{t}.npy"))
        except FileNotFoundError:
            print(f"Error: Could not find raw data for {t}.")
            return
            
    prices = pd.DataFrame(prices_dict)
    initial_prices = prices.iloc[-1].to_numpy()

    # Allow passing a csv file, default to tcvae_tree5.csv
    csv_file = os.path.join(current_dir, "data", "scenarios", "copula_tree.csv")
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    df = pd.read_csv(csv_file, header=None)

    Q = len(tickers)
    
    # Extract unique recourse prices to calculate ret1 stats
    df_recourse = df.drop_duplicates(subset=[0])
    recourse_prices = df_recourse.iloc[:, 3:3+Q].to_numpy()
    ret1 = recourse_prices / initial_prices - 1.0

    ret1_std = ret1.std(axis=0)
    ret1_min = ret1.min(axis=0)

    # All evaluate prices
    all_recourse_prices = df.iloc[:, 3:3+Q].to_numpy()
    evaluate_prices = df.iloc[:, 3+Q:3+2*Q].to_numpy()
    
    # ret2 calculated as return from recourse to evaluate node
    ret2 = evaluate_prices / all_recourse_prices - 1.0
    
    ret2_std = ret2.std(axis=0)
    ret2_min = ret2.min(axis=0)

    # Print the table header
    print(f"{'Ticker':<8} {'ret1 std':>8}   {'ret1 min':>8}  ||   {'ret2 std':>8}   {'ret2 min':>8}")
    
    for i, t in enumerate(tickers):
        print(f"{t:<8} {ret1_std[i]:>8.4f}   {ret1_min[i]:>8.4f}  ||   {ret2_std[i]:>8.4f}   {ret2_min[i]:>8.4f}")

if __name__ == "__main__":
    main()
