import os
import sys
import subprocess
import pandas as pd

def check_data_file(data_path):
    """Check if data file exists and can be loaded."""
    if not os.path.exists(data_path):
        return False, "Data file not found"
    try:
        df = pd.read_pickle(data_path)
        return True, f"Loaded successfully. Shape: {df.shape}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def run_script(script_path, name=""):
    """Run a Python script as a subprocess and handle output/errors."""
    print(f"\n=== Running: {script_path} ===")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"{name} failed with error:\n{result.stderr}")
        sys.exit(result.returncode)
    print(result.stdout)

def main():
    # --- Step 1: Data Download ---
    sp500_path = os.path.join('data', 'prices_sp500_2015-07-12_to_2025-07-12.pkl')
    nasdaq_path = os.path.join('data', 'prices_nasdaq_2015-07-12_to_2025-07-12.pkl')

    sp500_valid, sp500_msg = check_data_file(sp500_path)
    nasdaq_valid, nasdaq_msg = check_data_file(nasdaq_path)

    if not (sp500_valid and nasdaq_valid):
        print("Missing data files. Downloading data...")
        run_script("scripts/download_data.py", name="Data Download")

        # Re-check after download
        sp500_valid, sp500_msg = check_data_file(sp500_path)
        nasdaq_valid, nasdaq_msg = check_data_file(nasdaq_path)
        if not (sp500_valid and nasdaq_valid):
            print("Download completed but verification failed:")
            if not sp500_valid:
                print(f"S&P 500: {sp500_msg}")
            if not nasdaq_valid:
                print(f"NASDAQ: {nasdaq_msg}")
            sys.exit(1)

    print("Data verification:")
    print(f"  S&P 500: {sp500_msg}")
    print(f"  NASDAQ: {nasdaq_msg}")

    # --- Step 2: Feature Engineering ---
    features_path = os.path.join('data', 'features_sp500.pkl')
    if not os.path.exists(features_path):
        run_script("scripts/build_features.py", name="Feature Engineering")
        if not os.path.exists(features_path):
            print("Feature file not found after build. Exiting.")
            sys.exit(1)
        print(f"Features built and saved to {features_path}")
    else:
        print("Features already present.")

    # --- Step 3: Backtest ---
    run_script("scripts/run_backtest.py", name="Backtest")

    print("\nAll pipeline steps completed successfully.")

if __name__ == '__main__':
    main()
