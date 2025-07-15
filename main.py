# main.py

import os


def main():
    data_path = os.path.join('data', 'prices_2015-07-12_to_2025-07-12.pkl')
    if not os.path.exists(data_path):
        print("Data file not found. Downloading data...")
        from src import data

        data.main()
    else:
        print("Data file found. Proceeding to backtest...")
    from scripts import backtest

    backtest.main()


if __name__ == '__main__':
    main()
