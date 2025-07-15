import pandas as pd
import numpy as np
import pandas_ta as ta

def garman_klass_vol(df):
    log_hl = np.log(df['high']) - np.log(df['low'])
    log_co = np.log(df['adj close']) - np.log(df['open'])
    vol = (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    df['garman_klass_vol'] = vol
    return df

def rsi(df, period=14, group_level=1):
    """Add RSI to DataFrame, grouped by ticker."""
    df['rsi'] = df.groupby(level=group_level)['adj close'].transform(lambda x: ta.rsi(close=x, length=period))
    return df

def add_bollinger_bands(df, col='adj close', period=20, group_level=1):
    """
    Adds bb_low, bb_mid, bb_high columns (Bollinger Bands) to df.
    Assumes multi-index DataFrame, grouped by group_level (e.g., ticker).
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
    Adds ATR (normalized) to DataFrame, grouped by ticker.
    """
    def atr_group(x):
        atr = ta.atr(high=x['high'], low=x['low'], close=x['close'], length=period)
        return (atr - atr.mean()) / atr.std()
    df['atr'] = df.groupby(level=group_level, group_keys=False).apply(atr_group)
    return df

import pandas_ta as ta

def add_macd(df, fast=12, slow=26, signal=9, group_level=1):
    """
    Adds macd and macd_signal columns to df, grouped by group_level.
    """
    def macd_group(x):
        macd_df = ta.macd(close=x, fast=fast, slow=slow, signal=signal)
        # Return just the columns we want (same index as x)
        return macd_df[['MACD_12_26_9', 'MACDs_12_26_9']].rename(
            columns={'MACD_12_26_9':'macd', 'MACDs_12_26_9':'macd_signal'}
        )

    macd_results = df.groupby(level=group_level)['adj close'].apply(macd_group)
    # Flatten multiindex if needed
    df['macd'] = macd_results['macd'].values
    df['macd_signal'] = macd_results['macd_signal'].values
    return df

def main():
    pass

if __name__ == "__main__":
    main()
