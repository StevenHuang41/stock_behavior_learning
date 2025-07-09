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
    if len(sys.argv) > 1:
        stock_no = sys.argv[1]
    else :
        stock_no = input("Enter a Stock number:\n")

    # remove redundant column
    stock_data = yf.download(stock_no, period="max", auto_adjust=True)
    stock_data = stock_data.droplevel('Ticker', axis=1)

    # fill 0 volume with avg_volume
    # add avg of n days closing price
    # add trend state
    # add volume state
    stock_data = prerpocess(stock_data)

    print(stock_data)
    print(*STATES, sep='\n')












if __name__ == "__main__":
    main()
