# Makefile for algorithmic-trading-strategy with Conda env

PYTHON=python
PROJECT_ROOT=$(shell pwd)
PYTHONPATH=$(PROJECT_ROOT)
ENV=PYTHONPATH=$(PYTHONPATH)

all: pipeline

pipeline:
	PYTHONUNBUFFERED=1 $(ENV) $(PYTHON) main.py

download-data:
	PYTHONUNBUFFERED=1 $(ENV) $(PYTHON) scripts/download_data.py

features:
	PYTHONUNBUFFERED=1 $(ENV) $(PYTHON) scripts/build_features.py

universe:
	PYTHONUNBUFFERED=1 $(ENV) $(PYTHON) scripts/build_universe.py

backtest:
	PYTHONUNBUFFERED=1 $(ENV) $(PYTHON) scripts/run_backtest.py

test:
	PYTHONUNBUFFERED=1 $(ENV) pytest tests/ --disable-warnings -v

# clean:
# 	rm -rf data/*.pkl data/*.parquet .pytest_cache

venv:
	conda create -y -n quant311 python=3.9
	conda activate quant311 && pip install --upgrade pip && pip install -r requirements.txt

.PHONY: all pipeline download-data features backtest test clean venv
