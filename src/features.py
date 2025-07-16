import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional
import warnings

def garman_klass_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Garman-Klass volatility estimate as 'garman_klass_vol' to DataFrame.
    
    The Garman-Klass estimator uses OHLC data to estimate volatility more efficiently
    than simple close-to-close returns.
    
    Args:
        df: DataFrame with 'high', 'low', 'open', 'adj close' columns
        
    Returns:
        DataFrame with added 'garman_klass_vol' column
    """
    required_cols = ['high', 'low', 'open', 'adj close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Clean data: replace zeros and negatives with NaN, then forward-fill
    df_clean = df[required_cols].copy()
    df_clean = df_clean.where(df_clean > 0).ffill()
    if df_clean.isna().any().any():
        warnings.warn("NaN values remain in garman_klass_vol inputs after cleaning")
    
    log_hl = np.log(df_clean['high']) - np.log(df_clean['low'])
    log_co = np.log(df_clean['adj close']) - np.log(df_clean['open'])
    vol = (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    df['garman_klass_vol'] = vol
    return df

def rsi(df: pd.DataFrame, period: int = 14, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds RSI indicator as 'rsi' column, grouped by ticker.

    Args:
        df: DataFrame with 'adj close' and 'ticker' columns
        period: RSI calculation period (default: 14)
        group_col: Column name for grouping (default: 'ticker')

    Returns:
        DataFrame with added 'rsi' column
    """
    if 'adj close' not in df.columns:
        raise ValueError("DataFrame must contain 'adj close' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    if isinstance(df.index, pd.MultiIndex):
        df['rsi'] = df.groupby(level=1)['adj close'].transform(
            lambda x: ta.rsi(close=x, length=period)
        )
    else:
        df['rsi'] = df.groupby(group_col)['adj close'].transform(
            lambda x: ta.rsi(close=x, length=period)
        )
    return df

def add_bollinger_bands(df: pd.DataFrame, col: str = 'adj close', period: int = 20, 
                       std_dev: float = 2.0, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds Bollinger Bands columns: 'bb_low', 'bb_mid', 'bb_high'.

    Args:
        df: DataFrame with price data and 'ticker' column
        col: Column to use for calculation (default: 'adj close')
        period: Period for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        group_col: Column name for grouping (default: 'ticker')

    Returns:
        DataFrame with added Bollinger Bands columns
    """
    if col not in df.columns:
        raise ValueError(f"DataFrame must contain '{col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    def bb_group(x):
        bb = ta.bbands(close=x, length=period, std=std_dev)
        if bb is None or bb.empty:
            return pd.DataFrame({
                'bb_low': np.nan,
                'bb_mid': np.nan,
                'bb_high': np.nan
            }, index=x.index)
        return pd.DataFrame({
            'bb_low': bb.iloc[:,0],
            'bb_mid': bb.iloc[:,1],
            'bb_high': bb.iloc[:,2]
        }, index=x.index)

    if isinstance(df.index, pd.MultiIndex):
        bb_df = df.groupby(level=1, group_keys=False)[col].apply(bb_group)
    else:
        bb_df = df.groupby(group_col, group_keys=False)[col].apply(bb_group)

    for band in ['bb_low', 'bb_mid', 'bb_high']:
        df[band] = bb_df[band].values
    return df

def add_atr(df: pd.DataFrame, period: int = 14, normalize: bool = True, 
            group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds ATR (Average True Range) as 'atr' column, grouped by ticker.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period (default: 14)
        normalize: Whether to normalize ATR (default: True)
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added 'atr' column
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    def atr_group(x):
        atr = ta.atr(high=x['high'], low=x['low'], close=x['close'], length=period)
        if normalize and atr.std() != 0:
            return (atr - atr.mean()) / atr.std()
        return atr

    if isinstance(df.index, pd.MultiIndex):
        df['atr'] = df.groupby(level=1, group_keys=False).apply(atr_group)
    else:
        df['atr'] = df.groupby(group_col, group_keys=False).apply(atr_group)
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
             col: str = 'adj close', group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds MACD and MACD Signal columns to df, grouped by ticker.

    Args:
        df: DataFrame with price data and 'ticker' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        col: Column to use for calculation (default: 'adj close')
        group_col: Column name for grouping (default: 'ticker')

    Returns:
        DataFrame with added 'macd' and 'macd_signal' columns
    """
    if col not in df.columns:
        raise ValueError(f"DataFrame must contain '{col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    def macd_group(x):
        cleaned_x = x.copy()
        cleaned_x = cleaned_x.replace(0, np.nan).where(cleaned_x > 0).ffill()
        if cleaned_x.isna().all() or len(cleaned_x.dropna()) < max(fast, slow, signal):
            return pd.DataFrame({
                'macd': np.nan,
                'macd_signal': np.nan
            }, index=x.index)
        
        try:
            macd_df = ta.macd(close=cleaned_x, fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                return pd.DataFrame({
                    'macd': np.nan,
                    'macd_signal': np.nan
                }, index=x.index)
            return macd_df[[f'MACD_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}']].rename(
                columns={f'MACD_{fast}_{slow}_{signal}': 'macd', f'MACDs_{fast}_{slow}_{signal}': 'macd_signal'}
            )
        except Exception:
            return pd.DataFrame({
                'macd': np.nan,
                'macd_signal': np.nan
            }, index=x.index)

    if isinstance(df.index, pd.MultiIndex):
        macd_results = df.groupby(level=1, group_keys=False)[col].apply(macd_group)
    else:
        macd_results = df.groupby(group_col, group_keys=False)[col].apply(macd_group)

    df['macd'] = macd_results['macd'].values
    df['macd_signal'] = macd_results['macd_signal'].values
    return df

def add_daily_return(df: pd.DataFrame, price_col: str = 'adj close', 
                    group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds 'daily_return' as percent change of price_col, grouped by ticker.
    
    Args:
        df: DataFrame with price data
        price_col: Column to calculate returns from (default: 'adj close')
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added 'daily_return' column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    if isinstance(df.index, pd.MultiIndex):
        df['daily_return'] = df.groupby(level=1)[price_col].pct_change()
    else:
        df['daily_return'] = df.groupby(group_col)[price_col].pct_change()
    return df

def add_rolling_std(df: pd.DataFrame, period: int = 20, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds 'rolling_std' column: rolling std dev of daily returns, grouped by ticker.
    
    Args:
        df: DataFrame with data
        period: Rolling window period (default: 20)
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added 'rolling_std' column
    """
    if 'daily_return' not in df.columns:
        df = add_daily_return(df, group_col=group_col)
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    if isinstance(df.index, pd.MultiIndex):
        df['rolling_std'] = df.groupby(level=1)['daily_return'].transform(
            lambda x: x.rolling(period, min_periods=1).std()
        )
    else:
        df['rolling_std'] = df.groupby(group_col)['daily_return'].transform(
            lambda x: x.rolling(period, min_periods=1).std()
        )
    return df

def add_sma(df: pd.DataFrame, period: int = 20, price_col: str = 'adj close', 
           out_col: Optional[str] = None, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds SMA (Simple Moving Average) of price_col to DataFrame.
    
    Args:
        df: DataFrame with price data
        period: SMA period (default: 20)
        price_col: Column to calculate SMA from (default: 'adj close')
        out_col: Output column name (default: f'sma{period}')
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added SMA column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    if out_col is None:
        out_col = f'sma{period}'

    if isinstance(df.index, pd.MultiIndex):
        df[out_col] = df.groupby(level=1)[price_col].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
    else:
        df[out_col] = df.groupby(group_col)[price_col].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
    return df

def add_ema(df: pd.DataFrame, period: int = 20, price_col: str = 'adj close', 
           out_col: Optional[str] = None, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds EMA (Exponential Moving Average) of price_col to DataFrame.
    
    Args:
        df: DataFrame with price data
        period: EMA period (default: 20)
        price_col: Column to calculate EMA from (default: 'adj close')
        out_col: Output column name (default: f'ema{period}')
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added EMA column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    if out_col is None:
        out_col = f'ema{period}'

    if isinstance(df.index, pd.MultiIndex):
        df[out_col] = df.groupby(level=1)[price_col].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
    else:
        df[out_col] = df.groupby(group_col)[price_col].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
    return df

def add_obv(df: pd.DataFrame, price_col: str = 'adj close', group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds On-Balance Volume (OBV) column to DataFrame, grouped by ticker.
    
    Args:
        df: DataFrame with price and volume data
        price_col: Price column to use for direction (default: 'adj close')
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added 'obv' column
    """
    required_cols = [price_col, 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    def obv_calc(x):
        price_change = x[price_col].diff()
        volume_direction = np.where(price_change > 0, x['volume'], 
                                  np.where(price_change < 0, -x['volume'], 0))
        return pd.Series(volume_direction, index=x.index).fillna(0).cumsum()

    if isinstance(df.index, pd.MultiIndex):
        df['obv'] = df.groupby(level=1, group_keys=False).apply(obv_calc)
    else:
        df['obv'] = df.groupby(group_col, group_keys=False).apply(obv_calc)
    return df

def add_dollar_vol(df: pd.DataFrame, price_col: str = 'adj close') -> pd.DataFrame:
    """
    Adds dollar volume (in millions) as 'dollar_volume'.
    
    Args:
        df: DataFrame with price and volume data
        price_col: Price column to use (default: 'adj close')
        
    Returns:
        DataFrame with added 'dollar_volume' column
    """
    required_cols = [price_col, 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    df['dollar_volume'] = (df[price_col] * df['volume']) / 1e6
    return df

def add_multi_horizon_returns(
    df: pd.DataFrame, 
    horizons: List[int] = [1, 5, 10, 21, 60], 
    price_col: str = 'adj close', 
    group_col: str = 'ticker'
) -> pd.DataFrame:
    """
    Adds return features for multiple time horizons to the DataFrame.
    
    Args:
        df: DataFrame with price data
        horizons: List of periods to calculate returns for (default: [1, 5, 10, 21, 60])
        price_col: Price column to use (default: 'adj close')
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added return columns
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    df = df.copy()

    for h in horizons:
        colname = f'{h}d_return'
        if isinstance(df.index, pd.MultiIndex):
            df[colname] = df.groupby(level=1)[price_col].pct_change(periods=h)
        else:
            df[colname] = df.groupby(group_col)[price_col].pct_change(periods=h)
    
    return df

def add_momentum_indicators(df: pd.DataFrame, period: int = 12, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Adds momentum indicators: ROC (Rate of Change) and Stochastic Oscillator.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ROC calculation (default: 12)
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with added momentum indicators
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    # Rate of Change
    if isinstance(df.index, pd.MultiIndex):
        df['roc'] = df.groupby(level=1)['close'].transform(
            lambda x: x.pct_change(periods=period) * 100
        )
    else:
        df['roc'] = df.groupby(group_col)['close'].transform(
            lambda x: x.pct_change(periods=period) * 100
        )

    # Stochastic Oscillator
    def stoch_calc(x):
        # Clean input data: replace zeros and negatives with NaN, then forward-fill
        cleaned_x = x[['high', 'low', 'close']].copy()
        cleaned_x = cleaned_x.where(cleaned_x > 0).ffill()
        if cleaned_x.isna().any().any() or len(cleaned_x.dropna()) < 14:  # Minimum period for stoch
            return pd.DataFrame({
                'stoch_k': np.nan,
                'stoch_d': np.nan
            }, index=x.index)
        
        try:
            stoch = ta.stoch(high=cleaned_x['high'], low=cleaned_x['low'], close=cleaned_x['close'])
            if stoch is None or stoch.empty:
                return pd.DataFrame({
                    'stoch_k': np.nan,
                    'stoch_d': np.nan
                }, index=x.index)
            return stoch[['STOCHk_14_3_3', 'STOCHd_14_3_3']].rename(
                columns={'STOCHk_14_3_3': 'stoch_k', 'STOCHd_14_3_3': 'stoch_d'}
            )
        except Exception:
            return pd.DataFrame({
                'stoch_k': np.nan,
                'stoch_d': np.nan
            }, index=x.index)

    if isinstance(df.index, pd.MultiIndex):
        for group_name, group_data in df.groupby(level=1):
            stoch_result = stoch_calc(group_data)
            df.loc[group_data.index, 'stoch_k'] = stoch_result['stoch_k']
            df.loc[group_data.index, 'stoch_d'] = stoch_result['stoch_d']
    else:
        for group_name, group_data in df.groupby(group_col):
            stoch_result = stoch_calc(group_data)
            df.loc[group_data.index, 'stoch_k'] = stoch_result['stoch_k']
            df.loc[group_data.index, 'stoch_d'] = stoch_result['stoch_d']
    
    return df

def build_all_features(df: pd.DataFrame, group_col: str = 'ticker', 
                      include_momentum: bool = True) -> pd.DataFrame:
    """
    Add all core features/indicators for quant trading/backtesting.
    
    Features added:
    - Returns: daily_return, multi-horizon returns (1d, 5d, 10d, 21d, 60d)
    - Volatility: rolling_std, garman_klass_vol, atr
    - Trend: sma20, ema20, macd, macd_signal
    - Mean Reversion: rsi, bb_low, bb_mid, bb_high
    - Volume: obv, dollar_volume
    - Momentum: roc, stoch_k, stoch_d (if include_momentum=True)
    
    Args:
        df: DataFrame with OHLC data
        group_col: Column name for grouping (default: 'ticker')
        include_momentum: Whether to include momentum indicators (default: True)
        
    Returns:
        DataFrame with all features added
    """
    try:
        # Core features
        df = add_daily_return(df, group_col=group_col)
        df = add_rolling_std(df, period=20, group_col=group_col)
        df = add_sma(df, period=20, group_col=group_col)
        df = add_ema(df, period=20, group_col=group_col)
        df = rsi(df, period=14, group_col=group_col)
        df = add_macd(df, group_col=group_col)
        df = add_bollinger_bands(df, group_col=group_col)
        df = add_obv(df, group_col=group_col)
        df = add_dollar_vol(df)
        df = garman_klass_vol(df)
        df = add_atr(df, group_col=group_col)
        df = add_multi_horizon_returns(df, group_col=group_col)
        
        # Optional momentum indicators
        if include_momentum:
            df = add_momentum_indicators(df, group_col=group_col)
        
        # Clip outliers in key columns - MUST be last step
        clip_columns = ['daily_return', 'dollar_volume'] + [f'{h}d_return' for h in [1,5,10,21,60]]
        df = clip_outliers(df, clip_columns, group_col=group_col)
        
        print(f"Successfully added {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adj close', 'volume']])} technical indicators")
        
    except Exception as e:
        print(f"Error building features: {str(e)}")
        raise
    
    return df

def clip_outliers(df: pd.DataFrame, columns: List[str], lower: float = 0.01, upper: float = 0.99, group_col: str = 'ticker') -> pd.DataFrame:
    """
    Clips outliers in specified columns to the given quantiles, grouped by ticker.
    
    Args:
        df: DataFrame to clip outliers from
        columns: List of column names to clip
        lower: Lower quantile threshold (default: 0.01)
        upper: Upper quantile threshold (default: 0.99)
        group_col: Column name for grouping (default: 'ticker')
        
    Returns:
        DataFrame with outliers clipped
    """
    df = df.copy()
    
    if group_col not in df.columns and not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"DataFrame must contain '{group_col}' column or have a MultiIndex")

    for col in columns:
        if col in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                tickers = df.index.get_level_values(1).unique()
                for ticker in tickers:
                    mask = df.index.get_level_values(1) == ticker
                    ticker_data = df.loc[mask, col]
                    if len(ticker_data.dropna()) > 0:
                        q_low = ticker_data.quantile(lower)
                        q_high = ticker_data.quantile(upper)
                        df.loc[mask, col] = ticker_data.clip(lower=q_low, upper=q_high)
            else:
                for ticker in df[group_col].unique():
                    mask = df[group_col] == ticker
                    ticker_data = df.loc[mask, col]
                    if len(ticker_data.dropna()) > 0:
                        q_low = ticker_data.quantile(lower)
                        q_high = ticker_data.quantile(upper)
                        df.loc[mask, col] = ticker_data.clip(lower=q_low, upper=q_high)
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame, skipping outlier clipping")
    
    return df

def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> bool:
    """
    Validates that DataFrame has required structure for technical analysis.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required columns (default: standard OHLCV)
        
    Returns:
        True if valid, raises ValueError if not
    """
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close', 'adj close', 'volume']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    if not isinstance(df.index, pd.MultiIndex):
        warnings.warn("DataFrame does not have MultiIndex structure. Some functions may not work as expected.")
    
    if len(df) < 100:
        warnings.warn("DataFrame has fewer than 100 rows. Some indicators may be unreliable.")
    
    return True

def main():
    print("Indicators module started.")

if __name__ == "__main__":
    main()