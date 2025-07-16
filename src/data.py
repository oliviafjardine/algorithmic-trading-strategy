# scripts/download_data.py

from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import yfinance as yf
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
    return [s for s in symbols if '^' not in s and '/' not in s]

def download_price_data(symbols, start_date, end_date, name="all", batch_size=20, sleep_secs=5):
    cache_file = f"data/prices_{name}_{start_date}_to_{end_date}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_pickle(cache_file)
        print(df.head())
        return df

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

    # Flatten the MultiIndex: make each row (date, ticker, ...)
    flat_df = combined_df.stack(level=1).reset_index()
    flat_df = flat_df.rename(columns={'level_1': 'ticker', 'Date': 'date'})

    # Standardize column names (handle all variations of adj close)
    flat_df.columns = [
        col.replace('Adj Close', 'adj close').replace('adjclose', 'adj close').lower()
        for col in flat_df.columns
    ]

    print("Columns after flattening:", flat_df.columns.tolist())
    print("Sample rows:\n", flat_df.head())

    # Ensure 'close' exists
    if 'close' not in flat_df.columns:
        raise ValueError("No 'close' column found in downloaded data!")

    # Ensure 'adj close' exists and is filled using 'close' as fallback
    if 'adj close' not in flat_df.columns:
        flat_df['adj close'] = flat_df['close']
    else:
        flat_df['adj close'] = flat_df['adj close'].fillna(flat_df['close'])

    print("'adj close' null count:", flat_df['adj close'].isnull().sum())
    print("'close' null count:", flat_df['close'].isnull().sum())

    # Only keep and order the desired columns
    columns = ['date', 'ticker', 'adj close', 'close', 'high', 'low', 'open', 'volume']
    for col in columns:
        if col not in flat_df.columns:
            flat_df[col] = pd.NA  # Fill missing columns with NA

    flat_df = flat_df[columns]

    flat_df.to_pickle(cache_file)
    print(f"Saved downloaded data to {cache_file}")
    print(flat_df.head())
    return flat_df


def load_kaggle_stocks(kaggle_folder='data/stocks', start_date=None, end_date=None):
    files = [f for f in os.listdir(kaggle_folder) if f.endswith('.txt')]
    dfs = []
    for file in files:
        filepath = os.path.join(kaggle_folder, file)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath, delimiter=',')
        except pd.errors.EmptyDataError:
            continue
        if df.shape[0] == 0 or df.shape[1] == 0:
            continue
        df['ticker'] = file.replace('.txt', '').upper().split('.')[0]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'date'})
        if start_date is not None and end_date is not None:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        dfs.append(df)
    if len(dfs) == 0:
        raise ValueError("No valid stock data loaded from Kaggle dataset.")
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [col.lower() for col in combined.columns]
    if 'adj close' not in combined.columns and 'close' in combined.columns:
        combined['adj close'] = combined['close']
    columns = ['date', 'ticker', 'adj close', 'close', 'high', 'low', 'open', 'volume']
    for col in columns:
        if col not in combined.columns:
            combined[col] = pd.NA
    return combined[columns]

def main():
    print("Data module started.")

if __name__ == '__main__':
    main()
