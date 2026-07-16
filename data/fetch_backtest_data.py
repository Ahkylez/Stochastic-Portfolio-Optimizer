"""
Download 2025 weekly prices for all 30 Dow tickers.
Saved as data/raw/{ticker}_bt.npy (separate from training data).

Usage:
    python data/fetch_backtest_data.py
"""
import os
import numpy as np
import yfinance as yf

TICKERS = [
    "MMM", "AMZN", "AXP", "AMGN", "AAPL",
    "BA",  "CAT",  "CVX", "CSCO", "KO",
    "DIS", "GS",   "HD",  "HON",  "IBM",
    "JNJ", "JPM",  "MCD", "MRK",  "MSFT",
    "NKE", "NVDA", "PG",  "CRM",  "SHW",
    "TRV", "UNH",  "VZ",  "V",    "WMT",
]

START = "2025-01-01"
END   = "2025-12-31"


def download():
    print(f"Downloading weekly prices {START} to {END} for {len(TICKERS)} tickers...")
    df = yf.download(TICKERS, start=START, end=END, interval="1wk")["Close"]

    save_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(save_dir, exist_ok=True)

    saved, missing = [], []
    for ticker in TICKERS:
        if ticker not in df.columns:
            print(f"  WARNING: {ticker} not in download response")
            missing.append(ticker)
            continue
        arr = df[ticker].dropna().to_numpy()
        if len(arr) == 0:
            print(f"  WARNING: {ticker} has 0 backtest observations")
            missing.append(ticker)
            continue
        path = os.path.join(save_dir, f"{ticker}_bt.npy")
        np.save(path, arr)
        saved.append(ticker)

    print(f"Saved {len(saved)}/{len(TICKERS)} tickers to data/raw/{{ticker}}_bt.npy")
    if missing:
        print(f"Missing: {missing}")
    return df


if __name__ == "__main__":
    download()
