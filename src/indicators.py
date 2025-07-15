import pandas as pd
import numpy as np
import pandas_ta as ta

def garman_klass_vol(df):
    """
    Adds Garman-Klass volatility estimate as 'garman_klass_vol' to DataFrame.
    """
    log_hl = np.log(df['high']) - np.log(df['low'])
    log_co = np.log(df['adj close']) - np.log(df['open'])
    vol = (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    df['garman_klass_vol'] = vol
    return df

def rsi(df, period=14, group_level=1):
    """
    Adds RSI indicator as 'rsi' column, grouped by ticker.
    """
    df['rsi'] = df.groupby(level=group_level)['adj close'].transform(lambda x: ta.rsi(close=x, length=period))
    return df

def add_bollinger_bands(df, col='adj close', period=20, group_level=1):
    """
    Adds Bollinger Bands columns: 'bb_low', 'bb_mid', 'bb_high'.
    """
    def bb_group(x):
        bb = ta.bbands(close=np.log1p(x), length=period)
        return pd.DataFrame({
            'bb_low': bb.iloc[:,0],
            'bb_mid': bb.iloc[:,1],
            'bb_high': bb.iloc[:,2]
        }, index=x.index)
    bb_df = df.groupby(level=group_level)[col].apply(bb_group)
    for band in ['bb_low', 'bb_mid', 'bb_high']:
        df[band] = bb_df[band].values
    return df

def add_atr(df, period=14, group_level=1):
    """
    Adds normalized ATR as 'atr' column, grouped by ticker.
    """
    def atr_group(x):
        atr = ta.atr(high=x['high'], low=x['low'], close=x['close'], length=period)
        return (atr - atr.mean()) / atr.std()
    df['atr'] = df.groupby(level=group_level, group_keys=False).apply(atr_group)
    return df

def add_macd(df, fast=12, slow=26, signal=9, group_level=1):
    """
    Adds MACD and MACD Signal columns to df, grouped by ticker.
    """
    def macd_group(x):
        macd_df = ta.macd(close=x, fast=fast, slow=slow, signal=signal)
        return macd_df[['MACD_12_26_9', 'MACDs_12_26_9']].rename(
            columns={'MACD_12_26_9':'macd', 'MACDs_12_26_9':'macd_signal'}
        )
    macd_results = df.groupby(level=group_level)['adj close'].apply(macd_group)
    df['macd'] = macd_results['macd'].values
    df['macd_signal'] = macd_results['macd_signal'].values
    return df

def add_daily_return(df, price_col='adj close', group_level=1):
    """
    Adds 'daily_return' as percent change of price_col, grouped by ticker.
    """
    df['daily_return'] = df.groupby(level=group_level)[price_col].pct_change()
    return df

def add_rolling_std(df, period=20, group_level=1):
    """
    Adds 'rolling_std' column: rolling std dev of daily returns, grouped by ticker.
    """
    if 'daily_return' not in df.columns:
        df = add_daily_return(df, group_level=group_level)
    df['rolling_std'] = df.groupby(level=group_level)['daily_return'].transform(lambda x: x.rolling(period).std())
    return df

def add_sma(df, period=20, price_col='adj close', out_col=None, group_level=1):
    """
    Adds SMA (Simple Moving Average) of price_col to DataFrame.
    """
    if out_col is None:
        out_col = f'sma{period}'
    df[out_col] = df.groupby(level=group_level)[price_col].transform(lambda x: x.rolling(period).mean())
    return df

def add_ema(df, period=20, price_col='adj close', out_col=None, group_level=1):
    """
    Adds EMA (Exponential Moving Average) of price_col to DataFrame.
    """
    if out_col is None:
        out_col = f'ema{period}'
    df[out_col] = df.groupby(level=group_level)[price_col].transform(lambda x: x.ewm(span=period, adjust=False).mean())
    return df

def add_obv(df, price_col='adj close', group_level=1):
    """
    Adds On-Balance Volume (OBV) column to DataFrame, grouped by ticker.
    """
    def obv_calc(x):
        return (np.sign(x[price_col].diff()) * x['volume']).fillna(0).cumsum()
    df['obv'] = df.groupby(level=group_level, group_keys=False).apply(obv_calc)
    return df

def add_dollar_vol(df, price_col='adj close'):
    """
    Adds dollar volume (in millions) as 'dollar_volume'.
    """
    df['dollar_volume'] = (df[price_col] * df['volume']) / 1e6
    return df

def build_all_features(df, group_level=1):
    """
    Add all core features/indicators for quant trading/backtesting.
    Columns added: daily_return, rolling_std, sma20, ema20, rsi, macd, macd_signal,
    bb_low, bb_mid, bb_high, obv, dollar_volume, garman_klass_vol, atr.
    """
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
    return df


def main():
    pass

if __name__ == "__main__":
    main()
