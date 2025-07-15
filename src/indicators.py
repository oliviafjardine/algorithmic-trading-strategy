import pandas as pd
import numpy as np

def garman_klass_vol(df):
    log_hl = np.log(df['high']) - np.log(df['low'])
    log_co = np.log(df['adj close']) - np.log(df['open'])
    vol = (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    df['garman_klass_vol'] = vol
    return df
def main():
    pass

if __name__ == "__main__":
    main()