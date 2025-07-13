# Algorithmic Trading Strategy

This repository contains code and data for developing, analyzing, and backtesting algorithmic trading strategies using Python.

## Project Structure

- `main.py` — Main entry point for running trading strategies.
- `scripts/`
  - `analysis.py` — Tools for analyzing trading results and market data.
  - `backtest.py` — Backtesting framework for evaluating strategies.
  - `download_data.py` — Scripts for downloading and processing market data.
- `data/`
  - `nasdaq_screener.csv` — List of NASDAQ symbols.
  - `prices_*.pkl` — Cached price data.
  - `stocks/` — Individual stock data files.

## Features
- Download historical price data for S&P 500 and NASDAQ stocks
- Load and process Kaggle stock datasets
- Filter and clean ticker symbols
- Backtest trading strategies
- Analyze results with statistical and technical indicators

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/oliviafjardine/algorithmic-trading-strategy.git
   cd algorithmic-trading-strategy
   ```
2. Create and activate a Python environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate
   # or use conda
   conda create -n quant311 python=3.11
   conda activate quant311
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- To download market data:
  ```bash
  python scripts/download_data.py
  ```
- To run analysis or backtests, use the corresponding scripts in the `scripts/` folder.

## Data Sources
- Yahoo Finance (via yfinance)
- [Kaggle Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs#)
- [Wikipedia S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [NASDAQ symbol lists](https://www.nasdaq.com/market-activity/stocks/screener)

## Notes
- Data files are ignored by git (see `.gitignore`).
- Make sure you have an active internet connection for data downloads.

## License
MIT License