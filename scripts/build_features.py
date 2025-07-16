# src/scripts/build_features.py

import os
import time
import pandas as pd
from src.features.features import (
    garman_klass_vol, rsi, add_bollinger_bands, 
    add_macd, add_momentum_indicators
)
from src.features.preprocessing import preprocess

def build_features(raw_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_pickle(raw_path)

    df = garman_klass_vol(df)
    df = rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_momentum_indicators(df)

    df = preprocess(df)
    df.to_pickle(output_path)

    return df

def main():
    paths = {
        'SP500': 'data/raw/prices_sp500_2015-07-12_to_2025-07-12.pkl',
        'NASDAQ': 'data/raw/prices_nasdaq_2015-07-12_to_2025-07-12.pkl',
    }

    for name, raw_path in paths.items():
        print(f"\nProcessing {name} data...")
        output_path = f'data/processed/features_{name.lower()}.pkl'

        if os.path.exists(output_path):
            print(f"[SKIP] {output_path} already exists.")
            # continue
            df = pd.read_pickle(output_path)  # ← Load existing
        else:
            start = time.time()
            df = build_features(raw_path, output_path)
            duration = time.time() - start
            print(f"Features saved to {output_path} ({df.shape}) in {duration:.2f}s")

        # Always print this
        print("Columns:", sorted(df.columns.tolist()))
        print("Sample:")
        print(df.head())


if __name__ == '__main__':
    main()
