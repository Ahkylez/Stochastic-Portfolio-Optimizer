import os
import sys
import time
import dynamic_backtest
import main as frontier_main

def wait_for_tcvae_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Checking for TC-VAE scenario files...")
    
    while True:
        missing_files = []
        for run_id in range(10):
            target_dir = os.path.join(base_dir, "data", "scenarios", "backtest", str(run_id))
            for i in range(52):  # weeks 000 through 051
                filename = f"tcvae_week_{i:03d}.csv"
                filepath = os.path.join(target_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
                
        if not missing_files:
            print("All TC-VAE scenario files (week 000 to 051) for all runs are present.")
            break
            
        print(f"Missing {len(missing_files)} files (e.g., {os.path.basename(missing_files[0])} in run folder). Waiting 5 minutes before checking again...")
        time.sleep(300)

def main():
    # wait_for_tcvae_files()

    print("\n" + "=" * 70)
    print(" 2. RUNNING DYNAMIC BACKTEST (dynamic_backtest.py)")
    print("=" * 70)
    dynamic_backtest.main()


    # print("=" * 70)
    # print(" 1. GENERATING FRONTIERS (main.py)")
    # print("    Running on week 000 for Vine Copula (GARCH) and TC-VAE-VINE")
    # print("=" * 70)
    # frontier_main.main()
    # print("\nAll metrics runs completed successfully.")

if __name__ == "__main__":
    main()