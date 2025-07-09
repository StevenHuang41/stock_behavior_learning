import sys
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import itertools

from packages.preprocess import prerpocess
# note: 2025/6/6-9 stock split

# states
TRENDS = ['up', 'down', 'stable']
VOLUME_STATUS = ['high', 'low', 'normal']
PORTFOLIO_STATUS = ['empty', 'holding']
STATES = list(itertools.product(*[TRENDS] * 2, VOLUME_STATUS, PORTFOLIO_STATUS))

ACTIONS = ['buy', 'sell', 'hold']

def main():

    # get stock number
    if len(sys.argv) == 2:
        stock_no = sys.argv[1]

    elif len(sys.argv) == 4:
        splited = True
        stock_no = (sys.argv[1]).strip()
        split_date = (sys.argv[2]).strip()
        split_ratio = float((sys.argv[3]).strip())
    else :
        stock_no = input("Enter a Stock number:\n")
        splited = input(f"Has {stock_no} splited?\n")
        splited = splited.strip().lower()
        splited = True if 'y' in splited else False
        if splited:
            split_date = input("When did it split? ['YYYY-MM-DD']\n").strip()
            split_ratio = float(input("What is the split ratio? (n-for-1)\n").strip())

    # download stock data from yf
    stock_data = yf.download(stock_no, period="max", auto_adjust=True)

    # do preprocessing
    stock_data = prerpocess(stock_data,
                            hasStockSplited=splited,
                            split_date=split_date,
                            split_ratio=split_ratio)

    # pd.set_option('display.max_rows', None)

    print(stock_data)


if __name__ == "__main__":
    main()
