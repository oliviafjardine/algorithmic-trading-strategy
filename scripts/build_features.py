from src.features.features import (
    garman_klass_vol, rsi, add_bollinger_bands, 
    add_macd, add_momentum_indicators
)
from src.features.preprocessing import preprocess
import pandas as pd

def main():
    """Build and save feature-engineered datasets for both indices."""
    # Load raw price data
    sp500_path = 'data/prices_sp500_2015-07-12_to_2025-07-12.pkl'
    nasdaq_path = 'data/prices_nasdaq_2015-07-12_to_2025-07-12.pkl'
    
    for path, name in [(sp500_path, 'SP500'), (nasdaq_path, 'NASDAQ')]:
        print(f"\nProcessing {name} data...")
        df = pd.read_pickle(path)
        
        # Add technical indicators
        df = garman_klass_vol(df)
        df = rsi(df)
        df = add_bollinger_bands(df)
        df = add_macd(df)
        df = add_momentum_indicators(df)

        # Preprocess
        df = preprocess(df)
        
        # Save featured data
        output_path = f'data/features_{name.lower()}.pkl'
        df.to_pickle(output_path)
        print(f"Features saved to {output_path}")
        print(f"Shape: {df.shape}")
        print("\nFeature columns:", sorted(df.columns.tolist()))
        print("\nSample data:")
        print(df.head())
        print(f"{df.isna().any(axis=1).sum()} rows with NaN values out of {len(df)} total rows")



if __name__ == '__main__':
    main()