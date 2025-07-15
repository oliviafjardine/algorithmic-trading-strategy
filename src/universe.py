def filter_top_liquid(data, n=150, window=21, price_col='adj close', group_level='ticker'):
    """
    Filters to the top n most liquid stocks each day/month by rolling average dollar volume.
    Keeps all columns, including dollar volume metrics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with MultiIndex (date, ticker) or columns 'date', 'ticker'.
    n : int
        Number of most liquid stocks to keep per period.
    window : int
        Window (in trading days) for rolling average dollar volume.
    price_col : str
        Which price column to use for dollar volume calculation.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with all original and liquidity columns.
    """
    df = data.copy()

    # Calculate dollar volume if not present
    if 'dollar_volume' not in df.columns:
        df['dollar_volume'] = df[price_col] * df['volume']

    # Calculate rolling mean dollar volume per ticker
    if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
        dv_roll = (
            df['dollar_volume']
            .unstack(group_level)
            .rolling(window)
            .mean()
            .stack()
        )
        df['rolling_dollar_vol'] = dv_roll
    else:
        df['rolling_dollar_vol'] = (
            df.groupby('ticker')['dollar_volume'].transform(lambda x: x.rolling(window).mean())
        )

    # Rank within each date
    df['dollar_vol_rank'] = df.groupby('date')['rolling_dollar_vol'].rank(ascending=False, method='first')

    # Keep only top n
    filtered = df[df['dollar_vol_rank'] <= n]

    return filtered


def main():
    pass

if __name__ == "__main__":
    main()
