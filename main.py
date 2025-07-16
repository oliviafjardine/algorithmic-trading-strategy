import os
import sys
import subprocess
import pandas as pd

def check_data_file(data_path):
    """Check if data file exists and is valid."""
    if not os.path.exists(data_path):
        return False, "Data file not found"
    try:
        df = pd.read_pickle(data_path)
        return True, f"Data file loaded successfully. Shape: {df.shape}"
    except Exception as e:
        return False, f"Error reading data file: {str(e)}"

def main():
    # Define data paths for both indices
    sp500_path = os.path.join('data', 'prices_sp500_2015-07-12_to_2025-07-12.pkl')
    nasdaq_path = os.path.join('data', 'prices_nasdaq_2015-07-12_to_2025-07-12.pkl')
    
    # Check both data files
    sp500_valid, sp500_message = check_data_file(sp500_path)
    nasdaq_valid, nasdaq_message = check_data_file(nasdaq_path)
    
    if not (sp500_valid and nasdaq_valid):
        print("Missing data files. Downloading data...")
        try:
            ret = subprocess.run(
                [sys.executable, "scripts/download_data.py"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Download output:")
            print(ret.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Data download failed with error:\n{e.stderr}")
            sys.exit(1)
        
        # Verify downloads were successful
        sp500_valid, sp500_message = check_data_file(sp500_path)
        nasdaq_valid, nasdaq_message = check_data_file(nasdaq_path)
        
        if not (sp500_valid and nasdaq_valid):
            print("Download completed but verification failed:")
            if not sp500_valid:
                print(f"S&P 500: {sp500_message}")
            if not nasdaq_valid:
                print(f"NASDAQ: {nasdaq_message}")
            sys.exit(1)
    
    print("Data verification:")
    print(f"S&P 500: {sp500_message}")
    print(f"NASDAQ: {nasdaq_message}")

    print("Data pipeline completed successfully.")

if __name__ == '__main__':
    main()