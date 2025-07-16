import numpy as np
import pandas as pd
from typing import Optional, Literal, List
import yfinance as yf
from sklearn.preprocessing import OneHotEncoder

# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(transform_output='pandas')

def deep_agent_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    exclude_cols = ['Open', 'Close']
    encode_cols = [col for col in df.columns if col not in exclude_cols]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[('onehot_pipe', encoder, encode_cols)],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    df_ = preprocessor.fit_transform(df)
    cols_astype = [col for col in df_.columns if col not in exclude_cols]
    df_[cols_astype] = df_[cols_astype].astype(int)
    df_ = df_[exclude_cols + cols_astype]
    
    return df_


def prerpocess(
    stock_no: str,
    df: pd.DataFrame,
    avg_days: List[int],
) -> pd.DataFrame:
    df = df[['Close', 'Open', 'Volume']].copy()

    ## replace 0 value in col 'Open' with previous col 'Close'
    df['Open'] = df['Open'].where(df['Open'] != 0, df['Close'].shift())
    assert (df['Open'].to_numpy() != 0).all()
    assert (df['Close'].to_numpy() != 0).all()

    ## droplevel
    df = df.droplevel('Ticker', axis=1)

    ## stock split
    ticker = yf.Ticker(stock_no)
    splits = ticker.splits

    if len(splits):
        split_dates = [str(d).split(' ')[0] for d in splits.index]
        split_ratio = splits.values

        for d, r in zip(split_dates, split_ratio):
            df.loc[df.index > d, ['Open', 'Close']] *= r

    ## yf does not updates the information of 0050.TW stock split
    ## we manuelly adjust it, but hopefully it would update in the future
    ## should be remove in the deplopyment
    # df.loc[df.index > '2025-06-06', ['Open', 'Close']] *= 4
        
    ## fill 0 volume recored with avg_volume
    volume_mean = int(df['Volume'][df['Volume'] != 0].mean())
    df.loc[df['Volume'] == 0, 'Volume'] = volume_mean

    # add avg of n days closing price
    for d in avg_days:
        df[f'avg_of_{d}_days'] = \
            df['Close'].rolling(window=d).mean()

    df = df.dropna()

    ## add stock state
    # add trend state
    stable_factor = 0.001
    col_trend = [f'Trend_{i}' for i in range(len(avg_days))]
    for i, d in enumerate(avg_days):
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

    return df.loc[:, ['Open', 'Close', *col_trend, 'volume_state']]
    

if __name__ == "__main__":
    import yfinance as yf

    stock_data = yf.download("0050.TW",
                             period="max",
                             auto_adjust=True)

    stock_data = prerpocess("0050.TW", stock_data, avg_days=[5, 20])
    print(stock_data)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    stock_data = deep_agent_preprocess(stock_data)

    print(stock_data)