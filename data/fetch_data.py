import yfinance as yf
import numpy as np
import os


def download():
    START = '2015-01-01'
    END = '2025-01-01'
    tickers = [
        "MMM", "AMZN", "AXP", "AMGN", "AAPL", 
        "BA", "CAT", "CVX", "CSCO", "KO", 
        "DIS", "GS", "HD", "HON", "IBM", 
        "JNJ", "JPM", "MCD", "MRK", "MSFT", 
        "NKE", "NVDA", "PG", "CRM", "SHW", 
        "TRV", "UNH", "VZ", "V", "WMT"
    ]
    # The paper doesnt specifiy where they got the extended data so this is what I assume. Its close enough for my goals. 
    df = yf.download(tickers, start=START, end=END, interval="1wk")['Close']
    save_path = os.path.join(os.getcwd(), 'data', 'raw')
    for ticker in tickers:
        path = os.path.join(save_path, f'{ticker}.npy')
        data = df[ticker].to_numpy()
        np.save(path, data)


if __name__ == '__main__':
    download()