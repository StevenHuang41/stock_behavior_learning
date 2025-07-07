import sys
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf

from packages.preprocess import prerpocess
# note: 2025/6/6-9 stock split



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
    stock_data = prerpocess(stock_data)

    # add additional column of n days avg_Close price













if __name__ == "__main__":
    main()
