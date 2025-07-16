from src.universe import filter_top_liquid
import pandas as pd

def main():
    """Build and save filtered universes based on liquidity."""
    # Parameters
    n_stocks = 150  # Top N most liquid stocks
    window = 21     # Trading days for rolling average
    
    # Process each index
    for name in ['sp500', 'nasdaq']:
        print(f"\nProcessing {name.upper()} universe...")
        
        # Load featured data
        input_path = f'data/features_{name}.pkl'
        df = pd.read_pickle(input_path)
        
        # Filter for liquid stocks
        filtered_df = filter_top_liquid(
            data=df,
            n=n_stocks,
            window=window
        )
        
        # Save filtered universe
        output_path = f'data/universe_{name}.pkl'
        filtered_df.to_pickle(output_path)
        
        print(f"Universe saved to {output_path}")
        print(f"Original symbols: {df['ticker'].nunique()}")
        print(f"Filtered symbols: {filtered_df['ticker'].nunique()}")
        print("\nMost liquid stocks:")
        print(filtered_df.groupby('ticker')['dollar_volume'].mean()
              .sort_values(ascending=False).head())

if __name__ == '__main__':
    main()