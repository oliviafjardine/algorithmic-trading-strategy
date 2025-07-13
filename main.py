# main.py
from scripts import download_data, backtest, analysis

def main():
    print("Starting data download...")
    download_data.main()
    
    print("Starting backtest...")
    backtest.main()
    
    print("Starting analysis...")
    analysis.main()

if __name__ == '__main__':
    main()
