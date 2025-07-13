# scripts/download_data.py
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import os
import time
import warnings

warnings.filterwarnings('ignore')

def load_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(url)[0]
    return sp500['Symbol'].str.replace('.', '-', regex=False).tolist()

def load_nasdaq_symbols(path='data/nasdaq_screener.csv'):
    nasdaq = pd.read_csv(path)
    symbols = (
        nasdaq['Symbol']
        .dropna()
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .str.replace('.', '-', regex=False)
        .tolist()
    )
    return symbols

def filter_special_tickers(symbols):
    # Remove tickers with special chars like ^ or /
    filtered = [s for s in symbols if '^' not in s and '/' not in s]
    return filtered

def download_price_data(symbols, start_date, end_date, batch_size=20, sleep_secs=5):
    cache_file = f"data/prices_{start_date}_to_{end_date}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_pickle(cache_file)

    dfs = []
    filtered_symbols = filter_special_tickers(symbols)
    total_batches = (len(filtered_symbols) + batch_size - 1) // batch_size
    for i in range(0, len(filtered_symbols), batch_size):
        batch = filtered_symbols[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1} of {total_batches}")
        df_batch = yf.download(batch, start=start_date, end=end_date, progress=False)
        dfs.append(df_batch)
        time.sleep(sleep_secs)

    combined_df = pd.concat(dfs, axis=1)
    combined_df.to_pickle(cache_file)
    print(f"Saved downloaded data to {cache_file}")
    return combined_df

def load_kaggle_stocks(kaggle_folder='data/stocks', start_date=None, end_date=None):
    files = [f for f in os.listdir(kaggle_folder) if f.endswith('.txt')]
    dfs = []
    for file in files:
        filepath = os.path.join(kaggle_folder, file)
        if os.path.getsize(filepath) == 0:
            print(f"Skipping empty file: {file}")
            continue
        try:
            df = pd.read_csv(filepath, delimiter=',')
        except pd.errors.EmptyDataError:
            print(f"Skipping blank/malformed file: {file}")
            continue
        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"Skipping file with no data: {file}")
            continue
        df['Symbol'] = file.replace('.txt', '').upper()
        df['Date'] = pd.to_datetime(df['Date'])
        if start_date is not None and end_date is not None:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        dfs.append(df)
    if len(dfs) == 0:
        raise ValueError("No valid stock data loaded from Kaggle dataset.")
    return pd.concat(dfs, ignore_index=True)



def main():
    end_date = '2025-07-12'
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    sp500_symbols = load_sp500_symbols()
    nasdaq_symbols = load_nasdaq_symbols()

    print("Downloading S&P 500 data...")
    sp500_df = download_price_data(sp500_symbols, start_date, end_date)
    print(sp500_df.head())

    print("Downloading NASDAQ data...")
    nasdaq_df = download_price_data(nasdaq_symbols, start_date, end_date)
    print(nasdaq_df.head())

    print("Loading Kaggle dataset...")
    kaggle_df = load_kaggle_stocks(start_date=start_date, end_date=end_date)
    print(kaggle_df.head())

if __name__ == '__main__':
    main()
