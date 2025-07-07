import numpy as np
import pandas as pd

def prerpocess(df: pd.DataFrame, n: int | list[int]) -> pd.DataFrame:
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
    for i in range(len(list_n) - 1):
        df[f'Trend_{i + 1}'] = (
            df[f'avg_of_{list_n[i]}_days'] -
            df[f'avg_of_{list_n[i + 1]}_days']
        ) / df[f'avg_of_20_days']

        conditions = [
            (df[f'Trend_{i + 1}'] > 0),
            (df[f'Trend_{i + 1}'] < 0),
        ]
        choices = ['up', 'down']
        df[f'Trend_{i + 1}'] = \
            np.select(conditions, choices, default='stable')

    # add volume state
    conditions = [
        (df['Volume'] > df['Volume'].quantile(0.66)),
        (df['Volume'] < df['Volume'].quantile(0.33)),
    ]
    choices = ['high', 'low']
    df['volume_state'] = np.select(conditions, choices, default='normal')


    return df
    

if __name__ == "__main__":

    import yfinance as yf

    stock_data = yf.download("0050.TW", period="max", auto_adjust=True)
    stock_data = stock_data.droplevel('Ticker', axis=1)

    stock_data = prerpocess(stock_data, [5, 20, 60])

    print(stock_data)

