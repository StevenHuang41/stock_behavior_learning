import numpy as np
import pandas as pd
from typing import Optional, Literal


def stock_split_event(
    df: pd.DataFrame,
    date: Literal["YYYY-MM-DD"],
    ratio: float
) -> pd.DataFrame:
    df.loc[df.index > date, ['Open', 'Close']] *= ratio
    return df

def prerpocess(
    df: pd.DataFrame,
    n: int | list[int]=[5, 20],
    *,
    hasStockSplited: bool=False,
    split_date: Literal["YYYY-MM-DD"],
    split_ratio: float,
) -> pd.DataFrame:
    df = df[['Close', 'Open', 'Volume']].copy()

    ## replace 0 value in col 'Open' with previous col 'Close'
    df['Open'] = df['Open'].where(df['Open'] != 0, df['Close'].shift())
    assert (df['Open'].to_numpy() != 0).all()
    assert (df['Close'].to_numpy() != 0).all()

    # droplevel
    df = df.droplevel('Ticker', axis=1)

    # stock split: 4 for 1 after 2025-6-6
    if hasStockSplited:
        df = stock_split_event(df, split_date, split_ratio)

    # fill 0 volume recored with avg_volume
    volume_mean = int(df['Volume'][df['Volume'] != 0].mean())
    df.loc[df['Volume'] == 0, 'Volume'] = volume_mean

    # add avg of n days closing price
    list_n = n
    if type(n) != list:
        list_n = [n]

    for d in list_n:
        df[f'avg_of_{d}_days'] = \
            df['Close'].rolling(window=d).mean()

    df = df.dropna()

    ## add stock state
    # add trend state
    stable_factor = 0.001
    col_trend = [f'Trend_{i}' for i in range(len(list_n))]
    for i, d in enumerate(list_n):
        conditions = [
            (1 + stable_factor) * df[f'avg_of_{d}_days'] < df['Close'],
            (1 - stable_factor) * df[f'avg_of_{d}_days'] > df['Close'],
        ]
        choices = ['up', 'down']

        df[f'Trend_{i}'] = np.select(conditions, choices, default='stable')
        # col_trend.append(f'Trend_{i}')

    # add volume state
    conditions = [
        (df['Volume'] > df['Volume'].quantile(0.66)),
        (df['Volume'] < df['Volume'].quantile(0.33)),
    ]
    choices = ['high', 'low']
    df.loc[:, 'volume_state'] = \
        np.select(conditions, choices, default='normal')

    # stock split: 4 for 1 after 2025-06-06
    # df.loc[df.index > '2025-06-06', 'Close'] /= 4

    return df.loc[:, ['Open', 'Close', *col_trend, 'volume_state']]
    

if __name__ == "__main__":
    import yfinance as yf

    stock_data = yf.download("0050.TW",
                             period="max",
                             auto_adjust=True)

    stock_data = prerpocess(stock_data,
                            hasStockSplited=True,
                            split_date='2025-06-06',
                            split_ratio=4)


    print(stock_data)