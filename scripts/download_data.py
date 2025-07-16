from src.data import (
    load_sp500_symbols, load_nasdaq_symbols,
    download_price_data, load_kaggle_stocks
)
import pandas as pd

def main():
    end_date = '2025-07-12'
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    sp500_symbols = load_sp500_symbols()
    nasdaq_symbols = load_nasdaq_symbols()

    print("Downloading S&P 500 data...")
    sp500_df = download_price_data(sp500_symbols, start_date, end_date, name='sp500')
    print(sp500_df['ticker'].nunique())
    print(sp500_df['date'].min(), sp500_df['date'].max())
    print(f"S&P 500 loaded: {sp500_df.shape}")

    print("Downloading NASDAQ data...")
    nasdaq_df = download_price_data(nasdaq_symbols, start_date, end_date, name='nasdaq')
    print(f"NASDAQ loaded: {nasdaq_df.shape}")

    combined = pd.concat([sp500_df, nasdaq_df]).drop_duplicates(subset=['date', 'ticker'])
    print(f"Combined (merged) shape: {combined.shape}")
    print(combined.head())

    # Optionally, handle Kaggle data:
    # kaggle_df = load_kaggle_stocks(start_date=start_date, end_date=end_date)
    # print(f"Kaggle loaded: {kaggle_df.shape}")

if __name__ == '__main__':
    main()
