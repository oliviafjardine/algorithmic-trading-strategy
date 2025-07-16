import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature-engineered financial data by:
    - Sorting by ticker and date
    - Removing rows with any NaNs per ticker (avoids early-indicator artifacts)
    - Retaining the 'ticker' column for downstream modeling
    """
    # Defensive check: ensure required columns exist
    required_cols = {"ticker", "date"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Sort to preserve time-series integrity
    df = df.sort_values(["ticker", "date"])

    # Drop NaNs per ticker (safely retain ticker column)
    df_cleaned = (
        df.groupby("ticker", group_keys=False)
        .apply(lambda g: g.dropna(), include_groups=False)
        .reset_index(drop=True)
    )

    # Explicitly reattach 'ticker' if dropped (safeguard)
    if "ticker" not in df_cleaned.columns:
        df_cleaned["ticker"] = df["ticker"].values[df_cleaned.index]

    return df_cleaned
