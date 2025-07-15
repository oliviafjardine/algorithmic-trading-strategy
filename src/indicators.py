import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional, Union
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
    
    log_hl = np.log(df['high']) - np.log(df['low'])
    log_co = np.log(df['adj close']) - np.log(df['open'])
    vol = (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    df['garman_klass_vol'] = vol
    return df

def rsi(df: pd.DataFrame, period: int = 14, group_level: int = 1) -> pd.DataFrame:
    """
    Adds RSI indicator as 'rsi' column, grouped by ticker.
    
    Args:
        df: DataFrame with 'adj close' column
        period: RSI calculation period (default: 14)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'rsi' column
    """
    if 'adj close' not in df.columns:
        raise ValueError("DataFrame must contain 'adj close' column")
    
    df['rsi'] = df.groupby(level=group_level)['adj close'].transform(
        lambda x: ta.rsi(close=x, length=period)
    )
    return df

def add_bollinger_bands(df: pd.DataFrame, col: str = 'adj close', period: int = 20, 
                       std_dev: float = 2.0, group_level: int = 1) -> pd.DataFrame:
    """
    Adds Bollinger Bands columns: 'bb_low', 'bb_mid', 'bb_high'.
    
    Args:
        df: DataFrame with price data
        col: Column to use for calculation (default: 'adj close')
        period: Period for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added Bollinger Bands columns
    """
    if col not in df.columns:
        raise ValueError(f"DataFrame must contain '{col}' column")
    
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
    
    bb_df = df.groupby(level=group_level)[col].apply(bb_group)
    for band in ['bb_low', 'bb_mid', 'bb_high']:
        df[band] = bb_df[band].values
    return df

def add_atr(df: pd.DataFrame, period: int = 14, normalize: bool = True, 
           group_level: int = 1) -> pd.DataFrame:
    """
    Adds ATR (Average True Range) as 'atr' column, grouped by ticker.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period (default: 14)
        normalize: Whether to normalize ATR (default: True)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'atr' column
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    def atr_group(x):
        atr = ta.atr(high=x['high'], low=x['low'], close=x['close'], length=period)
        if normalize and atr.std() != 0:
            return (atr - atr.mean()) / atr.std()
        return atr
    
    df['atr'] = df.groupby(level=group_level, group_keys=False).apply(atr_group)
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
            group_level: int = 1) -> pd.DataFrame:
    """
    Adds MACD and MACD Signal columns to df, grouped by ticker.
    
    Args:
        df: DataFrame with 'adj close' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'macd' and 'macd_signal' columns
    """
    if 'adj close' not in df.columns:
        raise ValueError("DataFrame must contain 'adj close' column")
    
    def macd_group(x):
        macd_df = ta.macd(close=x, fast=fast, slow=slow, signal=signal)
        if macd_df is None or macd_df.empty:
            return pd.DataFrame({
                'macd': np.nan,
                'macd_signal': np.nan
            }, index=x.index)
        return macd_df[['MACD_12_26_9', 'MACDs_12_26_9']].rename(
            columns={'MACD_12_26_9':'macd', 'MACDs_12_26_9':'macd_signal'}
        )
    
    macd_results = df.groupby(level=group_level)['adj close'].apply(macd_group)
    df['macd'] = macd_results['macd'].values
    df['macd_signal'] = macd_results['macd_signal'].values
    return df

def add_daily_return(df: pd.DataFrame, price_col: str = 'adj close', 
                    group_level: int = 1) -> pd.DataFrame:
    """
    Adds 'daily_return' as percent change of price_col, grouped by ticker.
    
    Args:
        df: DataFrame with price data
        price_col: Column to calculate returns from (default: 'adj close')
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'daily_return' column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    
    df['daily_return'] = df.groupby(level=group_level)[price_col].pct_change()
    return df

def add_rolling_std(df: pd.DataFrame, period: int = 20, group_level: int = 1) -> pd.DataFrame:
    """
    Adds 'rolling_std' column: rolling std dev of daily returns, grouped by ticker.
    
    Args:
        df: DataFrame with data
        period: Rolling window period (default: 20)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'rolling_std' column
    """
    if 'daily_return' not in df.columns:
        df = add_daily_return(df, group_level=group_level)
    
    df['rolling_std'] = df.groupby(level=group_level)['daily_return'].transform(
        lambda x: x.rolling(period, min_periods=1).std()
    )
    return df

def add_sma(df: pd.DataFrame, period: int = 20, price_col: str = 'adj close', 
           out_col: Optional[str] = None, group_level: int = 1) -> pd.DataFrame:
    """
    Adds SMA (Simple Moving Average) of price_col to DataFrame.
    
    Args:
        df: DataFrame with price data
        period: SMA period (default: 20)
        price_col: Column to calculate SMA from (default: 'adj close')
        out_col: Output column name (default: f'sma{period}')
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added SMA column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    
    if out_col is None:
        out_col = f'sma{period}'
    
    df[out_col] = df.groupby(level=group_level)[price_col].transform(
        lambda x: x.rolling(period, min_periods=1).mean()
    )
    return df

def add_ema(df: pd.DataFrame, period: int = 20, price_col: str = 'adj close', 
           out_col: Optional[str] = None, group_level: int = 1) -> pd.DataFrame:
    """
    Adds EMA (Exponential Moving Average) of price_col to DataFrame.
    
    Args:
        df: DataFrame with price data
        period: EMA period (default: 20)
        price_col: Column to calculate EMA from (default: 'adj close')
        out_col: Output column name (default: f'ema{period}')
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added EMA column
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    
    if out_col is None:
        out_col = f'ema{period}'
    
    df[out_col] = df.groupby(level=group_level)[price_col].transform(
        lambda x: x.ewm(span=period, adjust=False).mean()
    )
    return df

def add_obv(df: pd.DataFrame, price_col: str = 'adj close', group_level: int = 1) -> pd.DataFrame:
    """
    Adds On-Balance Volume (OBV) column to DataFrame, grouped by ticker.
    
    Args:
        df: DataFrame with price and volume data
        price_col: Price column to use for direction (default: 'adj close')
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added 'obv' column
    """
    required_cols = [price_col, 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    def obv_calc(x):
        price_change = x[price_col].diff()
        volume_direction = np.where(price_change > 0, x['volume'], 
                                  np.where(price_change < 0, -x['volume'], 0))
        return pd.Series(volume_direction, index=x.index).fillna(0).cumsum()
    
    df['obv'] = df.groupby(level=group_level, group_keys=False).apply(obv_calc)
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
    group_level: int = 1, 
    ticker_col: str = 'ticker'
) -> pd.DataFrame:
    """
    Adds return features for multiple time horizons to the DataFrame.
    
    Args:
        df: DataFrame with price data
        horizons: List of periods to calculate returns for (default: [1, 5, 10, 21, 60])
        price_col: Price column to use (default: 'adj close')
        group_level: MultiIndex level for grouping (default: 1)
        ticker_col: Ticker column name (default: 'ticker')
        
    Returns:
        DataFrame with added return columns
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    
    df = df.copy()
    
    # Detect ticker location
    is_multiindex = isinstance(df.index, pd.MultiIndex) and ticker_col in df.index.names
    has_ticker_col = ticker_col in df.columns

    if not is_multiindex and not has_ticker_col:
        raise ValueError(f"Could not find '{ticker_col}' as index level or column in the DataFrame.")

    for h in horizons:
        colname = f'{h}d_return'
        if is_multiindex:
            df[colname] = df.groupby(level=group_level)[price_col].pct_change(periods=h)
        else:
            df[colname] = df.groupby(ticker_col)[price_col].pct_change(periods=h)
    
    return df

def add_momentum_indicators(df: pd.DataFrame, group_level: int = 1) -> pd.DataFrame:
    """
    Adds momentum indicators: ROC (Rate of Change) and Stochastic Oscillator.
    
    Args:
        df: DataFrame with OHLC data
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with added momentum indicators
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Rate of Change (12-period)
    df['roc'] = df.groupby(level=group_level)['close'].transform(
        lambda x: x.pct_change(periods=12) * 100
    )
    
    # Stochastic Oscillator
    def stoch_calc(x):
        stoch = ta.stoch(high=x['high'], low=x['low'], close=x['close'])
        if stoch is None or stoch.empty:
            return pd.DataFrame({
                'stoch_k': np.nan,
                'stoch_d': np.nan
            }, index=x.index)
        return stoch[['STOCHk_14_3_3', 'STOCHd_14_3_3']].rename(
            columns={'STOCHk_14_3_3': 'stoch_k', 'STOCHd_14_3_3': 'stoch_d'}
        )
    
    # Initialize columns with NaN
    df['stoch_k'] = np.nan
    df['stoch_d'] = np.nan
    
    # Calculate stochastic oscillator for each group
    for group_name, group_data in df.groupby(level=group_level):
        stoch_result = stoch_calc(group_data)
        df.loc[group_data.index, 'stoch_k'] = stoch_result['stoch_k']
        df.loc[group_data.index, 'stoch_d'] = stoch_result['stoch_d']
    
    return df


def build_all_features(df: pd.DataFrame, group_level: int = 1, 
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
        group_level: MultiIndex level for grouping (default: 1)
        include_momentum: Whether to include momentum indicators (default: True)
        
    Returns:
        DataFrame with all features added
    """
    try:
        # Core features
        df = add_daily_return(df, group_level=group_level)
        df = add_rolling_std(df, period=20, group_level=group_level)
        df = add_sma(df, period=20, group_level=group_level)
        df = add_ema(df, period=20, group_level=group_level)
        df = rsi(df, period=14, group_level=group_level)
        df = add_macd(df, group_level=group_level)
        df = add_bollinger_bands(df, group_level=group_level)
        df = add_obv(df, group_level=group_level)
        df = add_dollar_vol(df)
        df = garman_klass_vol(df)
        df = add_atr(df, group_level=group_level)
        df = add_multi_horizon_returns(df, group_level=group_level)
        
        # Optional momentum indicators
        if include_momentum:
            df = add_momentum_indicators(df, group_level=group_level)
        
        # Clip outliers in key columns - MUST be last step
        clip_columns = ['daily_return', 'dollar_volume'] + [f'{h}d_return' for h in [1,5,10,21,60]]
        df = clip_outliers(df, clip_columns, group_level=group_level)
        
        print(f"Successfully added {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adj close', 'volume']])} technical indicators")
        
    except Exception as e:
        print(f"Error building features: {str(e)}")
        raise
    
    return df


def clip_outliers(df: pd.DataFrame, columns: List[str], lower: float = 0.01, upper: float = 0.99, group_level: int = 1) -> pd.DataFrame:
    """
    Clips outliers in specified columns to the given quantiles, grouped by ticker.
    
    Args:
        df: DataFrame to clip outliers from
        columns: List of column names to clip
        lower: Lower quantile threshold (default: 0.01)
        upper: Upper quantile threshold (default: 0.99)
        group_level: MultiIndex level for grouping (default: 1)
        
    Returns:
        DataFrame with outliers clipped
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Get all unique tickers
            tickers = df.index.get_level_values(group_level).unique()
            
            for ticker in tickers:
                # Get data for this ticker
                mask = df.index.get_level_values(group_level) == ticker
                ticker_data = df.loc[mask, col]
                
                if len(ticker_data.dropna()) > 0:
                    # Calculate quantiles for this ticker
                    q_low = ticker_data.quantile(lower)
                    q_high = ticker_data.quantile(upper)
                    
                    # Clip values for this ticker
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
    
    # Check for MultiIndex structure
    if not isinstance(df.index, pd.MultiIndex):
        warnings.warn("DataFrame does not have MultiIndex structure. Some functions may not work as expected.")
    
    # Check for sufficient data
    if len(df) < 100:
        warnings.warn("DataFrame has fewer than 100 rows. Some indicators may be unreliable.")
    
    return True

def main():
    """
    Example usage of the technical analysis library.
    """
    # This would typically be called with actual market data
    pass

if __name__ == "__main__":
    main()