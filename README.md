# Algorithmic Trading Strategy

This repository contains code and data for developing, analyzing, and backtesting algorithmic trading strategies using Python.

## Data Sources

- **Yahoo Finance** (via `yfinance`): Historical price data for US stocks
- **Kaggle Huge Stock Market Dataset**: Individual stock price files (`data/stocks/*.txt`)
- **Wikipedia S&P 500**: Up-to-date S&P 500 symbol lists
- **NASDAQ Screener**: NASDAQ symbol lists (`data/nasdaq_screener.csv`)

## Project Structure

- `main.py` — Main entry point for running the full pipeline (data download, feature engineering, backtest)
- `scripts/`
  - `download_data.py` — Download and process market data
  - `build_features.py` — Build and save feature-engineered datasets
  - `run_backtest.py` — (Template) Run backtests on strategies
- `analysis/analysis.py` — (Template) Analyze backtest results
- `data/`
  - `nasdaq_screener.csv` — List of NASDAQ symbols
  - `prices_*.pkl` — Cached price data
  - `features_*.pkl` — Feature-engineered datasets
  - `stocks/` — Individual stock data files (from Kaggle)
- `src/`
  - `core/` — Data loading, universe construction
  - `features/` — Feature engineering and preprocessing
  - `models/` — (Reserved for future model code)
- `tests/` — Unit tests for features and universe logic

## Features & Technical Indicators

Implemented in `src/features/features.py`:

- **Garman-Klass Volatility**
- **RSI (Relative Strength Index)**
- **Bollinger Bands**
- **ATR (Average True Range)**
- **MACD (Moving Average Convergence Divergence)**
- **Daily returns, multi-horizon returns**
- **Rolling standard deviation**
- **SMA/EMA (Simple/Exponential Moving Average)**
- **OBV (On-Balance Volume)**
- **Dollar volume**
- **Momentum indicators**
- **Outlier clipping, preprocessing**

## Data & Universe Construction

- **Data loading**: From Yahoo Finance, Kaggle, and local cache (`src/core/data.py`)
- **Universe filtering**: Top N most liquid stocks by rolling average dollar volume (`src/core/universe.py`)
- **Preprocessing**: Sort, clean, and drop NaNs per ticker (`src/features/preprocessing.py`)

## Pipelines

### 1. Data Download Pipeline

- Downloads historical price data for S&P 500 and NASDAQ stocks using `yfinance`
- Loads Kaggle stock data for additional coverage
- Cleans and filters ticker symbols
- Caches price data in `.pkl` files for efficiency

### 2. Feature Engineering Pipeline

- Loads raw price data
- Computes all technical indicators and features
- Cleans and preprocesses data
- Saves feature datasets to `data/features_*.pkl`

### 3. Backtesting Pipeline

- (Template) Loads featured data and runs backtest logic (to be implemented)

### 4. Analysis Pipeline

- (Template) Analyzes backtest results (to be implemented)

## Makefile Commands

You can use the Makefile to run common tasks:

- `make pipeline` — Run the full pipeline (data download, feature engineering, backtest)
- `make download-data` — Download and cache all market data
- `make features` — Build and save feature-engineered datasets
- `make backtest` — Run the backtesting script (template)
- `make test` — Run all unit tests in `tests/`
- `make clean` — Remove cached data files and test cache
- `make venv` — Create a Conda environment and install dependencies

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/oliviafjardine/algorithmic-trading-strategy.git
   cd algorithmic-trading-strategy
   ```
2. Create and activate a Python environment (recommended: Conda Python 3.9 or 3.11):
   ```bash
   # With venv
   python -m venv venv
   source venv/bin/activate
   # Or with conda
   conda create -n quant311 python=3.11
   conda activate quant311
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Full Pipeline

- Using the Makefile:
  ```bash
  make pipeline
  ```
- Or directly:
  ```bash
  python main.py
  ```

### Individual Steps

- **Download data:**
  ```bash
  make download-data
  # or
  python scripts/download_data.py
  ```
- **Build features:**
  ```bash
  make features
  # or
  python scripts/build_features.py
  ```
- **Run backtest:**
  ```bash
  make backtest
  # or
  python scripts/run_backtest.py
  ```
- **Run tests:**
  ```bash
  make test
  # or
  python -m pytest tests/
  ```
- **Clean cache/data:**
  ```bash
  make clean
  ```

## Notes

- Data files are ignored by git (see `.gitignore`).
- Make sure you have an active internet connection for data downloads.
- Backtest and analysis modules are currently templates/placeholders.

## License

MIT License
