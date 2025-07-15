import pandas as pd
import numpy as np
import pytest

from src.indicators import build_all_features, validate_dataframe 

def make_sample_df():
    dates = pd.date_range('2024-01-01', periods=120)
    tickers = ['A', 'B']
    idx = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    df = pd.DataFrame({
        'open': np.random.uniform(90, 110, len(idx)),
        'high': np.random.uniform(100, 120, len(idx)),
        'low': np.random.uniform(80, 100, len(idx)),
        'close': np.random.uniform(90, 110, len(idx)),
        'adj close': np.random.uniform(90, 110, len(idx)),
        'volume': np.random.randint(1e6, 1e7, len(idx)),
    }, index=idx)
    return df

def test_feature_columns_exist():
    df = make_sample_df()
    validate_dataframe(df)
    features = build_all_features(df)
    expected = [
        'daily_return', 'rolling_std', 'sma20', 'ema20', 'rsi', 'macd',
        'macd_signal', 'bb_low', 'bb_mid', 'bb_high', 'obv', 'dollar_volume',
        'garman_klass_vol', 'atr', '1d_return', '5d_return', '10d_return', '21d_return', '60d_return'
    ]
    for col in expected:
        assert col in features.columns, f"Missing feature: {col}"

def test_outlier_clipping():
    df = make_sample_df()
    features = build_all_features(df)
    for col in ['daily_return', 'dollar_volume']:
        s = features[col].dropna()
        q01, q99 = s.quantile(0.01), s.quantile(0.99)
        assert (s >= q01 - 1e-8).all() and (s <= q99 + 1e-8).all()

def test_missing_column_raises():
    df = make_sample_df()
    df = df.drop('adj close', axis=1)
    with pytest.raises(ValueError):
        build_all_features(df)
